from ibc import dataset
import torch
import os

from network import mlp_ebm

device = torch.device('cuda')

class TrainConfig:
    experiment_name: str
    seed: int = 0
    device_type: str = "cuda"
    train_dataset_size: int = 10
    test_dataset_size: int = 500
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_batch_size: int = 8
    test_batch_size: int = 64
    coord_conv: bool = False
    dropout_prob:  None
    num_workers: int = 1
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    stochastic_optimizer_train_samples: int = 64


def make_dataloaders(train_config):
    """Initialize train/test dataloaders based on config values."""
    # Train split.
    train_dataset_config = dataset.DatasetConfig(
        dataset_size=train_config.train_dataset_size,
        seed=train_config.seed,
    )
    train_dataset = dataset.CoordinateRegression(train_dataset_config)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Test split.
    test_dataset_config = dataset.DatasetConfig(
        dataset_size=train_config.test_dataset_size,
        seed=train_config.seed,
    )
    test_dataset = dataset.CoordinateRegression(test_dataset_config)
    test_dataset.exclude(train_dataset.coordinates)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.test_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": train_dataloader,
        "test": test_dataloader,
    }
    
def main():
    # Initialize train and test dataloaders.
    train_config = TrainConfig
    dataloaders = make_dataloaders(train_config)
    train_dataloaders = dataloaders['train']
    print('bounds', train_dataloaders.dataset.get_target_bounds())
    exp_name = 'coordinate_regression'
    path = './agent_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    batch_size = 256
    num_counter_sample = 256
    for batch in dataloaders["train"]:
        input, target = batch
        print(input.shape)
        print(target.shape)
        break
    # network = mlp_ebm.MLPEBM((act_shape[0]+1), 1, normalizer='Batch', dense_layer_type='spectral_norm').to(device)

if __name__ == "__main__":
    main()