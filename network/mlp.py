"""Implements a tf_agents compatible mlp-ebm."""
from network.layers import resnet, spectral_norm
import torch.nn as nn
import torch


class MLP(nn.Module):
    """MLP only model."""
    # TODO: figure out size
    def __init__(self,
                 input_dim:int,
                 out_dim:int,
                 width=512,
                 depth=2,
                 rate=0.1,
                 activation='relu',
                 dense_layer_type='regular',
                 normalizer=None):
        super().__init__()

        # Define MLP.
        hidden_sizes = [width for _ in range(depth)]
        # print("obs_dim", obs_dim, "hidden_sizes", hidden_sizes)
        self._mlp = resnet.ResNetLayer(
            hidden_sizes, rate, input_dim=input_dim,
            dense_layer_type=dense_layer_type, activation=activation, normalizer=normalizer)

        # Define projection to action prediction
        self._project_prediction = nn.Linear(hidden_sizes[-1], out_dim)

    def forward(self, x):

        x = self._mlp(x)

        # Project to energy.
        x = self._project_prediction(x)

        # Squeeze extra dim.
        x = torch.squeeze(x, axis=-1)

        return x
