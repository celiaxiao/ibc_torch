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

def create_and_load_pretrained_network(config):
    # prepare pretrained extra feature (target position) model
    pretrained_config = config.copy()
    pretrained_config['act_dim'] = 2
    pretrained_config['obs_dim'] = 3072
    pretrained_config['checkpoint_path'] = f"work_dirs/formal_exp/{config['env_name']}/predict_target/checkpoints/"
    pretrained_config['resume_from_step'] = 20000
    print("[network | info] loading pretrained extra feature model at step", pretrained_config['resume_from_step'])
    pretrained_network, pretrained_network_visual = create_network(pretrained_config)
    return pretrained_network, pretrained_network_visual