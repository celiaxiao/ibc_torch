from network import mlp_ebm, mlp
from network.layers import pointnet, resnet
from environments.maniskill.maniskill_env import FillEnvPointcloud
from data.dataset_maniskill import *

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
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

def load_dataset(config):
    if 'h5' in config['dataset_dir']:
        from diffuser.datasets.d4rl import get_dataset_from_h5
        env = FillEnvPointcloud(control_mode=config['control_mode'], obs_mode=config['obs_mode'])
        dataset = get_dataset_from_h5(env, h5path=config['dataset_dir'])
        # only keep xyz
        print("[dataset|info] raw observation dim", dataset['observations'].shape)
        channel = dataset['observations'].shape[-1]
        dataset['observations'] = dataset['observations'][:, :, :channel//2]
        # flatten observation
        batch_size = dataset['observations'].shape[0]
        dataset['observations'] = dataset['observations'].reshape(batch_size, -1)
        if config['use_extra']:
            # TODO: should have fix this when collecting the h5 file. Only use the last frame of extra info
            # if config['num_frames'] is not None and config['num_frames'] > 1:
            # dataset['extra'] = dataset['extra'][:, -23:]
            # if we're not using all 4 of the extra info (default)
            if len(config['extra_info']) < 4:
                # use the infomation listed in the config['extra_info] array
                print("[dataset|info] using extra info as observation. with info name", config['extra_info'])
                extra = {'qpos': dataset['extra'][:, :7], 'qvel': dataset['extra'][:, 7:14], 
                        'tcp_pose': dataset['extra'][:, 14:21], 'target': dataset['extra'][:, 21:]}
                dataset['extra'] = np.concatenate([extra[info_name] for info_name in config['extra_info']], axis = -1)
            print("[dataset|info] using extra info as observation. extra dim", dataset['extra'].shape)
            dataset['observations'] = np.concatenate([dataset['observations'], dataset['extra']], axis = -1)
        # np.save('/home/yihe/ibc_torch/work_dirs/demos/hang_obs.npy', np.array(observations, dtype=object))
        dataset = maniskill_dataset(dataset['observations'], dataset['actions'], 'cuda')
    else:
        dataset = torch.load(config['dataset_dir'])
    if config['data_amount']:
        assert config['data_amount'] <= len(dataset), f"Not enough data for {config['data_amount']} pairs!"
        dataset = torch.utils.data.Subset(dataset, range(config['data_amount']))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
        generator=torch.Generator(device='cuda'), 
        shuffle=True)
    # print("config['obs_dim'], dataset[0][0].size()[0]", config['obs_dim'], dataset[0][0].size()[0])
    assert config['obs_dim']==dataset[0][0].size()[0], "obs_dim in dataset mismatch config"
    assert config['act_dim']==dataset[0][1].size()[0], "act_dim in dataset mismatch config"

    return dataloader