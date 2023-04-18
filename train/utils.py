from environments.maniskill.composition_env import CompositionPoints
from environments.block_pushing.multimodal_push_wrapper import BlockPushMultimodalWrapper
from network import mlp_ebm, mlp
from network.layers import pointnet, resnet
from environments.maniskill.maniskill_env import *
from data.dataset_maniskill import *

import gym
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from moviepy.editor import ImageSequenceClip
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

def get_env(config):
    fn = None
    if config["obs_mode"] == 'particles':
        if config["env_name"] == 'Hang-v0':
            fn = HangEnvParticle
        elif config["env_name"] == 'Fill-v0':
            fn = FillEnvParticle
        elif config["env_name"] == 'Excavate-v0':
            fn = ExcavateEnvParticle
    elif config["obs_mode"] == 'pointcloud':
        if config["env_name"] == 'Hang-v0':
            fn = HangEnvPointcloud
        elif config["env_name"] == 'Fill-v0':
            fn = FillEnvPointcloud
        elif config["env_name"] == 'Excavate-v0':
            fn = ExcavateEnvPointcloud
    elif config['env_name'] == 'OpenCabinetDoor-v1':
            fn = OpenCabinetDoorState
    elif config['env_name'] == 'CompositionPoints-v0':
        fn = CompositionPoints
    elif config['env_name'] == 'BlockPushMultimodal-v0':
        fn = BlockPushMultimodalWrapper

    if fn is not None:
        target_env = fn(control_mode=config["control_mode"], obs_mode=config["obs_mode"])
    else:
        target_env = gym.make(config["env_name"], control_mode=config["control_mode"], obs_mode=config["obs_mode"], model_ids='1000')
    if "num_frames" in config and config["num_frames"] is not None:
        print("Using FrameStackWrapper with num_frames", config["num_frames"])
        env = FrameStackWrapper(env, num_frames=config["num_frames"])
    return target_env

def load_validation_dataset(config):
    if config['val_dataset_dir'] is not None:
        config['dataset_dir'] = config['val_dataset_dir']
        return load_dataset(config)
    return None

class Normalizer:
    def __init__(self, data):
        self.means = dict()
        for key in data.keys():
            try:
                self.means[key] = data[key].mean()
            except:  # noqa: E722
                pass
        self.stds = dict()
        for key in data.keys():
            try:
                self.stds[key] = data[key].std()
            except:  # noqa: E722
                pass
        print("means", self.means)

    def normalize(self, x_in, key):
        return (x_in - self.means[key]) / self.stds[key]

    def denormalize(self, x_in, key):
        return x_in * self.stds[key] + self.means[key]

def load_h5_dataset(config):
    from diffuser.datasets.d4rl import get_dataset_from_h5
    env = get_env(config)
    print(f"loading dataset from {config['dataset_dir']=}")
    dataset = get_dataset_from_h5(env, h5path=config['dataset_dir'])
    return dataset
        
def load_dataset(config):
    normalizer = None
    if 'h5' in config['dataset_dir']:
        
        dataset = load_h5_dataset(config)
        print("[dataset|info] raw observation dim", dataset['observations'].shape)
         # only keep xyz for pointcloud dataset
        if config["obs_mode"] == "pointcloud":
            channel = dataset['observations'].shape[-1]
            dataset['observations'] = dataset['observations'][:, :, :channel//2]
        # flatten observation
        batch_size = dataset['observations'].shape[0]
        dataset['observations'] = dataset['observations'].reshape(batch_size, -1)
        if config['use_extra']:
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
        if config['normalize']:
            normalizer = Normalizer(dataset)
        if config['noise']:
            dataset = noised_dataset(dataset['observations'], dataset['actions'], 'cuda', normalizer=normalizer)
        else:
            dataset = maniskill_dataset(dataset['observations'], dataset['actions'], 'cuda', normalizer=normalizer)
    else:
        dataset = torch.load(config['dataset_dir'])    
    print("config['obs_dim'], dataset[0][0].size()[0]", config['obs_dim'], dataset[0][0].size()[0])
    assert config['obs_dim']==dataset[0][0].size()[0], "obs_dim in dataset mismatch config"
    assert config['act_dim']==dataset[0][1].size()[0], "act_dim in dataset mismatch config"

    return dataset

def load_dataloader(config):
    if config['data_amount']:
        assert config['data_amount'] <= len(dataset), f"Not enough data for {config['data_amount']} pairs!"
        dataset = torch.utils.data.Subset(dataset, range(config['data_amount']))
    dataset = load_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
        generator=torch.Generator(device='cuda'), 
        shuffle=True)
    return dataloader


def animate(imgs, fps=20, path="animate.mp4"):
        imgs = ImageSequenceClip(imgs, fps=fps)
        imgs.write_videofile(path, fps=fps)