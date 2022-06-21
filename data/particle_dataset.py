import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from train import get_data as data_module

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

def preload_dataset():
    batch_size = 1
    # env_name = get_env_name(task='PARTICLE', 
    #     shared_memory_eval=False, use_image_obs=False)
    create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
        dataset_path='data/particle/2d_oracle_particle*.tfrecord',
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
    with open('data/particle/particle_large.npy', 'wb') as f:
        np.save(f, dataset.experiences)
    print(dataset.__len__(), dataset.__getitem__(0))

def load_dataset():
    print("loading from npy")
    with open('data/particle/particle_large.npy', 'rb') as f:
        experiences = np.load(f, allow_pickle=True)
    dataset = Particle_dataset(experiences)
    print(dataset.__len__(), dataset.__getitem__(0))
    torch.save(dataset, 'data/particle/particle_large.pt')

if __name__ == '__main__':
    # preload_dataset()
    load_dataset()

