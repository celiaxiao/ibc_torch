"""Implements a tf_agents compatible mlp-ebm."""
import gin
from network import resnet, spectral_norm
import torch.nn as nn
import torch


@gin.configurable
class MLPEBM(nn.Module):
    """MLP-EBM compatible with tfagents."""

    def __init__(self,
                 input_dim,
                 out_dim,
                 width=512,
                 depth=2,
                 rate=0.1,
                 activation='relu',
                 dense_layer_type='regular'):
        super().__init__()

        # Define MLP.
        hidden_sizes = [width for _ in range(depth)]
        # print("obs_dim", obs_dim, "hidden_sizes", hidden_sizes)
        self._mlp = resnet.ResNetLayer(
            hidden_sizes, rate, input_dim=input_dim,
            dense_layer_type=dense_layer_type, activation=activation)\

        # Define projection to energy.
        self._project_energy = nn.Linear(hidden_sizes[-1], out_dim)
        if dense_layer_type == 'regular':
            pass
        elif dense_layer_type == 'spectral_norm':
            self._project_energy = spectral_norm.SpectralNorm(
                self._project_energy)

    def forward(self, inputs):
        # obs: dict of named obs_spec.
        # act:   [B x act_spec]
        obs, act = inputs

        # TODO: must make sure we calculate correct input shape when we create the network
        # Combine dict of observations to concatenated tensor. [B x T x obs_spec] 
        if isinstance(obs, dict):
            obs = torch.concat([torch.flatten(obs[key]) for key in obs.keys()], axis=-1)
        # Flatten obs across time: [B x T * obs_spec]
        batch_size = act.shape[0]
        obs = torch.reshape(obs, [batch_size, -1])
        # print(obs.shape)
        # Concat [obs, act].
        x = torch.concat([obs, act], -1)
        # print("concat shape", x.shape)
        x = torch.flatten(x).float()
        # Forward mlp.
        x = self._mlp(x)

        # Project to energy.
        x = self._project_energy(x)

        # Squeeze extra dim.
        x = torch.squeeze(x, axis=-1)

        return x
