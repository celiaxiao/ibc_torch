import numpy as np
import torch
from torch.utils.data import Dataset

class particle_dataset(Dataset):
    '''
    Dataset for maniskill2 softbody envs, particles obs mode that needs manual concatenation for pointcloud and agent
    Input:
    -- observations: list of (xyz, agent) pairs. xyz should already be downsampled, agent should be pad to 1d array
    -- actions: list of actions. index should match observations
    '''
    def __init__(self, observations, actions, device=None):
        experiences = []
        if device is None:
            device = torch.device('cuda')
        for idx in range(len(observations)):
            exp = {}
            # exp['xyz'] = torch.tensor(observations[idx][0]).to(device)
            # exp['agent'] = torch.tensor(observations[idx][1]).to(device)
            exp['observation'] = torch.tensor(np.concatenate((observations[idx][0].reshape(-1, 1).squeeze(), observations[idx][1]))).to(device)
            exp['action'] = torch.tensor(actions[idx]).to(device)
            experiences.append(exp)
            # print(exp['observation'].shape)
            # exit(0)
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        # xyz = self.experiences[idx]['xyz']
        # agent = self.experiences[idx]['agent']
        obs = self.experiences[idx]['observation']
        act = self.experiences[idx]['action']
        return obs, act

class state_dataset(Dataset):
    '''
    Dataset for maniskill2 softbody envs, state-based 
    Input:
    -- observations: only qpos of controller, and target be end pose of tcp in one episode.
    -- actions: list of actions. index should match observations
    '''
    def __init__(self, observations, actions, device=None):
        experiences = []
        if device is None:
            device = torch.device('cuda')
        for idx in range(len(observations)):
            exp = {}
            # exp['xyz'] = torch.tensor(observations[idx][0]).to(device)
            # exp['agent'] = torch.tensor(observations[idx][1]).to(device)
            exp['observation'] = torch.tensor(observations[idx]).to(device)
            exp['action'] = torch.tensor(actions[idx]).to(device)
            experiences.append(exp)
            # print(exp['observation'].shape)
            # exit(0)
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        # xyz = self.experiences[idx]['xyz']
        # agent = self.experiences[idx]['agent']
        obs = self.experiences[idx]['observation']
        act = self.experiences[idx]['action']
        return obs, act

class maniskill_dataset(Dataset):
    '''
    Dataset for maniskill2 softbody envs, no post processing needed
    Input:
    -- observations: list of observations. Need to be pre-processed to 1d. Visual components first.
    -- actions: list of actions. index should match observations
    -- normalizer: instance of the Normalizer class
    -- device: device to use for torch tensors (default is 'cuda')
    '''
    def __init__(self, observations, actions, device=None, normalizer=None):
        if device is None:
            device = torch.device('cuda')
        self.normalizer = normalizer
        if self.normalizer is not None:
            # Normalize the data
            observations = self.normalizer.normalize(observations, 'observations')
            actions = self.normalizer.normalize(actions, 'actions')
        self.observations = torch.tensor(observations).float().to(device)
        self.actions = torch.tensor(actions).float().to(device)
        

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        act = self.actions[idx]
        return (obs, act)

class noised_dataset(maniskill_dataset):
    '''
    Dataset for maniskill2 softbody envs, no post processing needed
    Input:
    -- observations: list of observations. Need to be pre-processed to 1d. Visual components first.
    -- actions: list of actions. index should match observations
    '''
    def __init__(self, observations, actions, *arg, **kwargs):
        super().__init__(observations, actions, *arg, **kwargs)
        self.std = self.observations.std()
        print('add noise with std: ', self.std)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        act = self.actions[idx]
        
        # add 0.1*randome gaussian noise to the observation. 0.1 is determined empirically
        obs += torch.randn_like(obs) * 0.05 * self.std
        return (obs, act)

def save_dataset(dataset, env_name):
    print(dataset['observations'].shape, dataset['actions'].shape) # An N x dim_observation Numpy array of observations

    ibc_dataset = particle_dataset(dataset['observations'], dataset['actions'])
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
    # Hang-v0: ?? experience pair, xyz shape (??, 1024, 3), agent shape (??, 25), act shape (??, 8)

    observations = np.load("/home/yihe/ibc_torch/work_dirs/formal_demos/Hang-v0/pcd_observations.npy", allow_pickle=True)
    actions = np.load("/home/yihe/ibc_torch/work_dirs/formal_demos/Hang-v0/actions.npy")

    dataset = pointcloud_dataset(observations, actions)
    print(len(dataset))
    torch.save(dataset, '/home/yihe/ibc_torch/work_dirs/formal_demos/Hang-v0/pd_joint_delta_pcd.pt')
    

    