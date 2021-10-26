from __future__ import annotations

import dataclasses
import os
import pathlib
import signal
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

if TYPE_CHECKING:
    from .trainer import TrainStateProtocol

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

T = TypeVar("T")
TensorOrFloat = Union[np.ndarray, torch.Tensor, float]


@dataclasses.dataclass
class TensorboardLogData:
    scalars: Dict[str, TensorOrFloat] = dataclasses.field(default_factory=dict)

    @staticmethod
    def merge(a: TensorboardLogData, b: TensorboardLogData) -> TensorboardLogData:
        return TensorboardLogData(scalars=dict(**a.scalars, **b.scalars))

    def extend(self, scalars: Dict[str, TensorOrFloat] = {}) -> TensorboardLogData:
        return TensorboardLogData.merge(self, TensorboardLogData(scalars=scalars))


@dataclasses.dataclass(frozen=True)
class Experiment:
    identifier: str
    """The name of the experiment."""

    data_dir: pathlib.Path = dataclasses.field(init=False)
    log_dir: pathlib.Path = dataclasses.field(init=False)
    checkpoint_dir: pathlib.Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        root = pathlib.Path("./experiments")

        super().__setattr__("data_dir", root / self.identifier)
        super().__setattr__("log_dir", self.data_dir / "tb")
        super().__setattr__("checkpoint_dir", self.data_dir / "checkpoints")

    def assert_new(self) -> Experiment:
        """Makes sure that there are no existing checkpoints, logs, or metadata."""
        assert not self.data_dir.exists() or tuple(self.data_dir.iterdir()) == ()
        return self

    def assert_exists(self) -> Experiment:
        """Makes sure that there are existing checkpoints, logs, or metadata."""
        assert self.data_dir.exists() and tuple(self.data_dir.iterdir()) != ()
        return self

    # =================================================================== #
    # Properties.
    # =================================================================== #

    # Note: This hack is necessary because the SummaryWriter object instantly creates
    # a directory upon construction and thus would break the `assert_*` functionality.
    @property
    def summary_writer(self) -> SummaryWriter:
        if not hasattr(self, "__summary_writer__"):
            object.__setattr__(
                self,
                "__summary_writer__",
                SummaryWriter(self.log_dir),
            )
        return object.__getattribute__(self, "__summary_writer__")

    # =================================================================== #
    # Checkpointing.
    # =================================================================== #

    def save_checkpoint(
        self,
        target: TrainStateProtocol,
        step: int,
        prefix: str = "ckpt_",
        keep: int = 10,
    ) -> None:
        """Save a snapshot of the train state to disk."""
        self._ensure_directory_exists(self.checkpoint_dir)

        # Create a snapshot of the state.
        snapshot_state = {}
        for k, v in target.__dict__.items():
            if hasattr(v, "state_dict"):
                snapshot_state[k] = v.state_dict()
        snapshot_state["steps"] = target.steps

        # Save to disk.
        checkpoint_path = self.checkpoint_dir / f"{prefix}{step}.ckpt"
        self._atomic_save(checkpoint_path, snapshot_state)

        # Trim extraneous checkpoints.
        self._trim_checkpoints(keep)

    def restore_checkpoint(
        self,
        target: TrainStateProtocol,
        step: Optional[int] = None,
        prefix: str = "ckpt_",
    ):
        """Restore a snapshot of the train state from disk."""
        # Get latest checkpoint if no step has been provided.
        if step is None:
            step = self._get_latest_checkpoint_step()

        checkpoint_path = self.checkpoint_dir / f"{prefix}{step}.ckpt"
        snapshot_state = torch.load(checkpoint_path, map_location="cpu")

        delattr(target, "steps")
        setattr(target, "steps", snapshot_state["steps"])

        for k, v in target.__dict__.items():
            if hasattr(v, "state_dict"):
                getattr(target, k).load_state_dict(snapshot_state[k])

    # =================================================================== #
    # Logging.
    # =================================================================== #

    def log(self, log_data: TensorboardLogData, step: int) -> None:
        """Logging helper for TensorBoard."""
        for k, v in log_data.scalars.items():
            self.summary_writer.add_scalar(tag=k, scalar_value=v, global_step=step)
        self.summary_writer.flush()

    def write_metadata(self, name: str, object: Any) -> None:
        """Serialize an object to disk as a yaml file."""
        self._ensure_directory_exists(self.data_dir)
        assert not name.endswith(".yaml")

        path = self.data_dir / (name + ".yaml")
        print(f"Writing metadata to {path}")
        with open(path, "w") as fp:
            yaml.dump(object, fp)

    def read_metadata(self, name: str, expected_type: Type[T]) -> T:
        """Load an object from the experiment's metadata directory."""
        path = self.data_dir / (name + ".yaml")

        with open(path, "r") as fp:
            output = yaml.load(fp, Loader=yaml.Loader)

        assert isinstance(output, expected_type)
        return output

    # =================================================================== #
    # Helper functions.
    # =================================================================== #

    def _ensure_directory_exists(self, path: pathlib.Path) -> None:
        """Helper for ensuring that a directory exists."""
        if not path.exists():
            path.mkdir(parents=True)

    def _trim_checkpoints(self, keep: int) -> None:
        """Helper for deleting older checkpoints."""
        # Get a list of checkpoints.
        ckpts = list(self.checkpoint_dir.glob(pattern="*.ckpt"))

        # Sort in reverse `step` order.
        ckpts.sort(key=lambda f: -int(f.stem.split("_")[-1]))

        # Remove until `keep` remain.
        while len(ckpts) - keep > 0:
            ckpts.pop().unlink()

    def _get_latest_checkpoint_step(self) -> int:
        """Helper for returning the step of the latest checkpoint."""
        ckpts = list(self.checkpoint_dir.glob(pattern="*.ckpt"))
        ckpts.sort(key=lambda f: -int(f.stem.split("_")[-1]))
        return int(ckpts[0].stem.split("_")[-1])

    def _atomic_save(self, save_path: pathlib.Path, snapshot: Dict[str, Any]) -> None:
        """Helper for safely saving to disk."""
        # Ignore Ctrl-C while saving.
        try:
            orig_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
        except ValueError:
            # Signal throws a ValueError if we're not in the main thread.
            orig_handler = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir) / "tmp.ckpt"
            torch.save(snapshot, tmp_path)
            # `rename` is POSIX-compliant and thus, is an atomic operation.
            # Ref: https://docs.python.org/3/library/os.html#os.rename
            os.rename(tmp_path, save_path)

        # Restore SIGINT handler.
        if orig_handler is not None:
            signal.signal(signal.SIGINT, orig_handler)
