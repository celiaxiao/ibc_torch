import torch
import torch.nn as nn

from network.spectral_norm import SpectralNorm


class ResNetLayer(nn.Module):
    def __init__(
            self, hidden_sizes, rate, input_dim,
            activation='relu', normalizer=None, dense_layer_type='regular'):
        """normalizer should be ['Batch', 'Layer', None]."""
        super().__init__()
        self.normalizer = normalizer
        self._weight_layers = []
        self._norm_layers = []
        self._activation_layers = []
        self._dropouts = []
        # layers = []
        # ResNet wants layers to be even numbers,
        # but remember there will be an additional
        # layer just to project to the first hidden size.
        assert len(hidden_sizes) % 2 == 0
        # print("input dim", input_dim, "hidden_sizes", hidden_sizes)
        self._projection_layer = nn.Linear(input_dim, hidden_sizes[0])
        if dense_layer_type == 'regular':
            pass
        elif dense_layer_type == 'spectral_norm':
            self._projection_layer = SpectralNorm(self._projection_layer)

        self.identity = nn.Identity()
        # Note: source use 2 lists with range step size 2
        for l in range(len(hidden_sizes)):
            self._weight_layers.append(
                nn.Linear(hidden_sizes[l], hidden_sizes[l]))
            # layers.append(
            #     nn.Linear(hidden_sizes[l], hidden_sizes[l]))
            if self.normalizer == 'Batch':
                self._norm_layers.append(nn.BatchNorm1d(hidden_sizes[l]))
                # layers.append(nn.BatchNorm2d(hidden_sizes[l]))
            elif self.normalizer == 'Layer':
                self._norm_layers.append(
                    nn.LayerNorm(eps=1e-6))
                # layers.append(nn.LayerNorm(eps=1e-6))
            elif self.normalizer is None:
                pass
            else:
                raise ValueError('Expected a different normalizer.')
            if activation == 'relu':
                self._activation_layers.append(nn.ReLU())
                # layers.append(nn.ReLU())
            elif activation == 'swish':
                self._activation_layers.append(nn.SiLU())
                # layers.append(nn.SiLU())
            else:
                raise ValueError('Expected a different layer activation.')
            self._dropouts.append(nn.Dropout(p=rate))
            # layers.append(nn.Dropout(p=rate))
        # self._layers = nn.Sequential(*layers)
        self._weight_layers = nn.ModuleList(self._weight_layers)
        self._norm_layers = nn.ModuleList(self._norm_layers)
        self._activation_layers = nn.ModuleList(self._activation_layers)
        self._dropouts = nn.ModuleList(self._dropouts)

        # print("check resnet", self._weight_layers, self._activation_layers,self._norm_layers, self._dropouts)

    def forward(self, x):
        # print("getting _projection_layer input in resnet", x)
        x = self._projection_layer(x)
        # print("getting _projection_layer out in resnet", x)
        # Do forward pass through resnet layers.
        for l in range(0,len(self._weight_layers),2):
            assert not torch.isnan(x).any()
            x_start_block = self.identity(x)
            if self.normalizer is not None:
                x = self._norm_layers[l](x)
                assert not torch.isnan(x).any()
            x = self._activation_layers[l](x)
            x = self._dropouts[l](x)
            x = self._weight_layers[l](x)
            assert not torch.isnan(x).any()
            # repeat and add the initial input
            if self.normalizer is not None:
                x = self._norm_layers[l+1](x)
                assert not torch.isnan(x).any()
            x = self._activation_layers[l+1](x)
            x = self._dropouts[l+1](x)
            x = self._weight_layers[l+1](x)
            assert not torch.isnan(x).any()
            x = x_start_block + x
            assert not torch.isnan(x).any()
        return x
