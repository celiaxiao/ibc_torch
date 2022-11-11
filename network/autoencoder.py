"""Implements a tf_agents compatible mlp-ebm."""
from network.layers import pointnet
import torch.nn as nn
import torch


class AutoEncoder(nn.Module):
    """autoencoder with pointnet as encoder and FC as decoder """
    def __init__(self,
                 visual_num_points:int,
                 visual_num_channels:int,
                 latent_dim:int,
                 ):
        super().__init__()

        self.encoder = pointnet(in_dim=[visual_num_points, visual_num_channels], out_dim=latent_dim, normalize=True)
        # Define MLP.
        self._mlp = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, visual_num_points * visual_num_channels)
    )

    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self._mlp(x)

    def forward(self, inputs):
        # inputs [B, 1024, visual_num_channels]

        # x: [B, latent_dim]
        latent = self.encode(inputs)

        # x: [B, 1024, visual_num_channels]
        reconstruct = self.decode(latent)

        return reconstruct
