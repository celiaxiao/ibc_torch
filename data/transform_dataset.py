import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

class Ibc_dataset(Dataset):
    def __init__(self, experiences, device=None):
        if device is None:
            device = torch.device('cuda')
        for idx in range(len(experiences)):
            obs_dict = experiences[idx]['observation']
            # print(obs_dict)
            for key in obs_dict:
                obs_dict[key] = torch.tensor(obs_dict[key]).float().to(device)
            experiences[idx]['observation'] = obs_dict
            experiences[idx]['action'] = torch.tensor(experiences[idx]['action']).float().to(device)
            # print('cast tensor', experiences[idx], '\n')
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        obs_dict = self.experiences[idx]['observation']
        obs = torch.concat([torch.flatten(obs_dict[key]) for key in obs_dict.keys()], axis=-1)
        act = self.experiences[idx]['action'].float()
        return obs, act.squeeze()

class d4rl_dataset(Dataset):
    def __init__(self, observations, actions, device=None):
        experiences = []
        if device is None:
            device = torch.device('cuda')
        for idx in range(len(observations)):
            exp = {}
            # print(obs_dict)
            exp['observation'] = torch.tensor(observations[idx]).to(device)
            exp['action'] = torch.tensor(actions[idx]).to(device)
            experiences.append(exp)
            # print('cast tensor', experiences[idx], '\n')
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        obs = self.experiences[idx]['observation']
        act = self.experiences[idx]['action']
        return obs, act.squeeze() 

if __name__ == '__main__':
    '''
    {'observation': {
        'effector_target_translation': <tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[ 0.36545557, -0.03870075]]], dtype=float32)>, 
        'target_orientation': <tf.Tensor: shape=(1, 1, 1), dtype=float32, numpy=array([[[-2.9088342]]], dtype=float32)>, 
        'block_translation': <tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[ 0.36407387, -0.01264255]]], dtype=float32)>, 
        'effector_translation': <tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[ 0.365777  , -0.03960275]]], dtype=float32)>, 
        'target_translation': <tf.Tensor: shape=(1, 1, 2), dtype=float32, numpy=array([[[0.42538625, 0.11610205]]], dtype=float32)>, 
        'block_orientation': <tf.Tensor: shape=(1, 1, 1), dtype=float32, numpy=array([[[2.946957]]], dtype=float32)>}, 
    'action': <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[-0.02287993, -0.01908409]], dtype=float32)>}
    '''
    dataset_path = 'data/block_push_states_location/oracle_push*.tfrecord'
    preload_numpy_dir = 'data/block_push_states_location/block.npy'
    torch_dataset_dir = 'data/block_push_states_location/block.pt'
    # preload_dataset(dataset_path, preload_numpy_dir)
    # load_dataset(preload_numpy_dir, torch_dataset_dir)

