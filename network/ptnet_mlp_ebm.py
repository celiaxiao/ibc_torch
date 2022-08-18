"""Implements a tf_agents compatible mlp-ebm."""
from network.layers import resnet, spectral_norm, pointnet
import torch.nn as nn
import torch

class PTNETMLPEBM(nn.Module):
    """MLP-EBM that process obs dict separately before concat with act"""
    # TODO: figure out size
    def __init__(self,
                 xyz_input_dim:int,
                 agent_input_dim:int,
                 act_input_dim:int,
                 out_dim:int,
                 width=512,
                 depth=2,
                 rate=0.1,
                 activation='relu',
                 dense_layer_type='regular',
                 normalizer=None):
        super().__init__()

        # Define pointcloud processing pointnet layer
        self._ptnet = pointnet.pointNetLayer(out_dim=512)

        # Define MLP.
        hidden_sizes = [width for _ in range(depth)]
        # print("obs_dim", obs_dim, "hidden_sizes", hidden_sizes)
        self._mlp = resnet.ResNetLayer(
            hidden_sizes, rate, input_dim=512+agent_input_dim+act_input_dim,
            dense_layer_type=dense_layer_type, activation=activation, normalizer=normalizer)

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
        xyz, agent, act = inputs
        
        # TODO: must make sure we calculate correct input shape when we create the network
        # Combine dict of observations to concatenated tensor. [B x T x obs_spec] 
        if isinstance(xyz, dict):
            batch_size = obs[list(obs.keys())[0]].shape[0]
            obs = torch.concat([torch.flatten(obs[key]) for key in obs.keys()], axis=-1)
        else:
            batch_size = xyz.shape[0]

        # Flatten obs across time: [B x T * obs_spec]
        xyz = torch.reshape(xyz, [batch_size, -1])
        agent = torch.reshape(agent, [batch_size, -1])
        # print("xyz, agent", xyz.shape, agent.shape)

        xyz = self._ptnet(xyz)
        print("shape after pointnet", xyz.shape)

        # Concat [obs, act].
        x = torch.concat([xyz, agent, act], -1)
        print("concat shape", x.shape)
        # Forward mlp.
        x = self._mlp(x)

        # Project to energy.
        x = self._project_energy(x)

        # Squeeze extra dim.
        x = torch.squeeze(x, axis=-1)

        return x