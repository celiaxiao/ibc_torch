import os
from mani_skill2.envs.mpm.hang_env import HangEnv
from mani_skill2.envs.mpm.fill_env import FillEnv
from mani_skill2.envs.mpm.excavate_env import ExcavateEnv
import numpy as np
from numpy.random import default_rng
import urllib.request
import h5py
from tqdm import tqdm

DATASET_PATH = 'data/softbody'
def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath

def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath

class HangEnvParticle(HangEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Particles obs-mode.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        self._max_episode_steps = 350
        super().__init__(*args, **kwargs)
        
    def get_obs(self):
        obs = super().get_obs()

        xyz = obs['particles']['x'][np.random.choice(range(len(obs['particles']['x'])), size=1024, replace=False)]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], self.rod.get_pose().p,self.rod.get_pose().q))

        return np.concatenate((xyz.reshape(-1,1).squeeze(), agent))

    def get_dataset(self, h5path=None):
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        return data_dict

class HangEnvState(HangEnv):
    '''
    State obs-mode. 
    Reset will randomly select a target pose and its corresponding seed from input file.
    Target file will be a list of (seed, pose) tuples.
    '''
    def __init__(self, target_file) -> None:
        self.all_targets = np.load(target_file, allow_pickle=True)
        self.target = None
        super().__init__()

    def reset(self, seed=0, reconfigure=True):
        rng = default_rng(seed=seed)
        reset_seed, reset_target = self.all_targets[rng.choice(len(self.all_targets), 1)[0]]
        print(reset_seed, reset_target)
        self.target = reset_target
        super().reset(seed=reset_seed, reconfigure=reconfigure)

        return self.get_obs()

    def get_obs(self):
        obs = super().get_obs()
        return np.hstack((obs['agent']['qpos'], self.target))

class FillEnvParticle(FillEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Particles obs-mode.
    '''
    def get_obs(self):
        obs = super().get_obs()

        xyz = obs['particles']['x']
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], self.beaker_x, self.beaker_y))

        return np.concatenate((xyz.reshape(-1,1).squeeze(), agent))

class ExcavateEnvParticle(ExcavateEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Particles obs-mode.
    '''
    def get_obs(self):
        obs = super().get_obs()

        xyz = obs['particles']['x'][np.random.choice(range(len(obs['particles']['x'])), size=1024, replace=False)]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], np.array([(self.target_num - 250)/900.])))

        return np.concatenate((xyz.reshape(-1,1).squeeze(), agent))