import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from train import get_data as data_module

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
        
class Particle_dataset(Dataset):
    def __init__(self, experiences, device=None):
        if device is None:
            device = torch.device('cuda')
        for idx in range(len(experiences)):
            obs_dict = experiences[idx]['observation']
            for key in obs_dict.keys():
                obs_dict[key] = torch.tensor(obs_dict[key].numpy()).float().to(device)
            experiences[idx]['observation'] = obs_dict
            experiences[idx]['action'] = torch.tensor(experiences[idx]['action'].numpy()).float().to(device)
            # print('cast tensor', experiences[idx], '\n')
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        obs_dict = self.experiences[idx]['observation']
        obs = torch.concat([torch.flatten(obs_dict[key]) for key in obs_dict.keys()], axis=-1)
        act = self.experiences[idx]['action'].float()
        return obs, act.squeeze()

def preload_dataset(torch_dataset_dir, preload_numpy_dir):
    batch_size = 1
    # env_name = get_env_name(task='PARTICLE', 
    #     shared_memory_eval=False, use_image_obs=False)
    create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
        dataset_path=torch_dataset_dir,
    sequence_length=1, replay_capacity=10000, batch_size=batch_size, for_rnn=False,
    dataset_eval_fraction=0.0, flatten_action=True)
    train_data, _ = create_train_and_eval_fns_unnormalized()
    experiences = []

    train_data = train_data.prefetch(buffer_size=10)
    for exp, _ in train_data.take(int(1e5)):
      obs, act = exp
      experience = {'observation':obs, 'action':act}
      experiences.append(experience)

    experiences = np.array(experiences)
    dataset = Particle_dataset(experiences, torch.device('cpu'))
    with open(preload_numpy_dir, 'wb') as f:
        np.save(f, dataset.experiences)
    print(dataset.__len__(), dataset.__getitem__(0))

def load_dataset(preload_numpy_dir, torch_dataset_dir):
    print("loading from npy")
    with open(preload_numpy_dir, 'rb') as f:
        experiences = np.load(f, allow_pickle=True)
    dataset = Particle_dataset(experiences)
    print(dataset.__len__(), dataset.__getitem__(0))
    torch.save(dataset, torch_dataset_dir)

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
    load_dataset(preload_numpy_dir, torch_dataset_dir)

