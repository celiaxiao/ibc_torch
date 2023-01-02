from network import mlp_ebm, mlp
from network.layers import pointnet, resnet

import torch
device = torch.device('cuda')

def create_network(config):
    network_visual=None
    resume_step = config['resume_from_step'] if config['resume_from_step'] else 0

    if config['visual_type'] == 'pointnet':
        network_visual = pointnet.pointNetLayer(in_dim=[config['visual_num_channels'], config['visual_num_points']], out_dim=config['visual_output_dim'], normalize=config['visual_normalize'])

        visual_input_dim = config['visual_num_points'] * config['visual_num_channels']

        if config['agent_type'] == 'ibc':
            network = mlp_ebm.MLPEBM(
            (config['visual_output_dim'] + config['obs_dim'] - visual_input_dim + config['act_dim']), 1, 
            width=config['width'], depth=config['depth'],
            normalizer=config['mlp_normalizer'], rate=config['rate'],
            dense_layer_type=config['dense_layer_type']).to(device)

        elif config['agent_type'] == 'mse':
            # Define MLP.
            network = mlp.MLP(input_dim=(config['visual_output_dim'] + config['obs_dim'] - visual_input_dim), out_dim=config['act_dim'], width=config['width'], depth=config['depth'],
            normalizer=config['mlp_normalizer'], rate=config['rate'])

        if resume_step > 0:
            network_visual.load_state_dict(torch.load(
            f"{config['checkpoint_path']}step_{resume_step}_pointnet.pt"))
            network.load_state_dict(torch.load(
            f"{config['checkpoint_path']}step_{resume_step}_mlp.pt"))
    
    else:
        if config['agent_type'] == 'ibc':
            network = mlp_ebm.MLPEBM(
            (config['obs_dim'] + config['act_dim']), 1, 
            width=config['width'], depth=config['depth'],
            normalizer=config['mlp_normalizer'], rate=config['rate'],
            dense_layer_type=config['dense_layer_type']).to(device)
        elif config['agent_type'] == 'mse':
            # Define MLP.
            network = mlp.MLP(input_dim=config['obs_dim'], out_dim=config['act_dim'], width=config['width'], depth=config['depth'],
            normalizer=config['mlp_normalizer'], rate=config['rate'])
        if resume_step > 0:
            network.load_state_dict(torch.load(
            f"{config['checkpoint_path']}step_{resume_step}_mlp.pt"))

    return network, network_visual