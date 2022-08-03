import numpy as np
import torch
from torch.utils.data import Dataset
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

def save_dataset(dataset, env_name):
    print(dataset['observations'].shape, dataset['actions'].shape) # An N x dim_observation Numpy array of observations

    ibc_dataset = d4rl_dataset(dataset['observations'], dataset['actions'])
    print(ibc_dataset.__len__())
    
    print(ibc_dataset.__getitem__(1))   
    torch.save(ibc_dataset, './d4rl/' + env_name + '.pt')

def get_action_stat(dataset, env_name):
    action = dataset['actions']
    print(action.min(0), action.max(0))
    action_stat = {}
    action_stat['min'] = action.min(0)
    action_stat['max'] = action.max(0)
    with open('./d4rl/' + env_name + '_action_stat.pt', 'wb') as f:
        np.save(f, action_stat)
    
    
if __name__ == '__main__':
    import gym
    import d4rl # Import required to register environments
    
    # door-human: 6729 experience pair, obs shape (6729, 39), act shape (6729, 28)
    # hammer-human: 11310 experience pair, obs shape(11310, 46) act shape(11310, 26)
    # relocate human: 9942 experience pair, obs shape(9942, 39) act shape(9942, 30)
    # pen human: 5000 experience pair, obs shape(5000, 45) act shape(5000, 24)

    # Create the environment
    env_name = 'pen-human'
    env = gym.make(env_name + '-v0')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    print(dataset['observations'].shape, dataset['actions'].shape) # An N x dim_observation Numpy array of observations
    save_dataset(dataset, env_name)
    get_action_stat(dataset, env_name)
    