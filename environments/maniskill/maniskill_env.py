from mani_skill2.envs.mpm.hang_env import HangEnv
from mani_skill2.envs.mpm.fill_env import FillEnv
from mani_skill2.envs.mpm.excavate_env import ExcavateEnv
import numpy as np
from numpy.random import default_rng

class HangEnvParticle(HangEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Particles obs-mode.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        self._max_episode_steps = 350
        super().__init__(*args, **kwargs)
        
    def get_obs(self):
        obs = super().get_obs()

        xyzw = obs['particles']['x'][np.random.choice(range(len(obs['particles']['x'])), size=1024, replace=False)]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], self.rod.get_pose().p,self.rod.get_pose().q))

        return np.concatenate((xyzw.reshape(-1,1).squeeze(), agent))

class HangEnvPointcloud(HangEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Pointcloud obs-mode.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(obs_mode='pointcloud', *args, **kwargs)

    def get_obs(self):
        obs = super().get_obs()

        # all valid indices of points
        valid = np.array(range(len(obs['pointcloud']['xyzw'])))[obs['pointcloud']['xyzw'][:,:3][:, 2] > 0]
        sel = np.random.choice(valid, size=1024, replace=False)
        xyzw = obs['pointcloud']['xyzw'][sel]
        rgb = obs['pointcloud']['rgb'][sel]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], obs['extra']['tcp_pose'], obs['extra']['target']))

        return {'pointcloud':{'xyzw':xyzw, 'rgb':rgb}, 'extra':agent}

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
    def __init__(self, *args, **kwargs) -> None:
        self._max_episode_steps = 250
        super().__init__(*args, **kwargs)
    def get_obs(self):
        obs = super().get_obs()

        xyzw = obs['particles']['x']
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], self.beaker_x, self.beaker_y))

        return np.concatenate((xyzw.reshape(-1,1).squeeze(), agent))

class FillEnvPointcloud(FillEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Particles obs-mode.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(obs_mode='pointcloud', *args, **kwargs)

    def get_obs(self):
        obs = super().get_obs()

        valid = np.array(range(len(obs['pointcloud']['xyzw'])))[obs['pointcloud']['xyzw'][:,:3][:, 2] > 0]
        sel = np.random.choice(valid, size=1024, replace=False)
        xyzw = obs['pointcloud']['xyzw'][sel]
        rgb = obs['pointcloud']['rgb'][sel]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], obs['extra']['tcp_pose'], obs['extra']['target']))

        return {'pointcloud':{'xyzw':xyzw, 'rgb':rgb}, 'extra':agent}

class ExcavateEnvParticle(ExcavateEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Particles obs-mode.
    '''
    def __init__(self, *args, **kwargs) -> None:
        self._max_episode_steps = 350
        super().__init__(*args, **kwargs)
    def get_obs(self):
        obs = super().get_obs()

        xyzw = obs['particles']['x'][np.random.choice(range(len(obs['particles']['x'])), size=1024, replace=False)]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], np.array([(self.target_num - 250)/900.])))

        return np.concatenate((xyzw.reshape(-1,1).squeeze(), agent))

class ExcavateEnvPointcloud(ExcavateEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper. Particles obs-mode.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(obs_mode='pointcloud', *args, **kwargs)

    def get_obs(self):
        obs = super().get_obs()
        xyz = obs['pointcloud']['xyzw'][:,:3]
        valid = np.array(range(len(obs['pointcloud']['xyzw'])))[obs['pointcloud']['xyzw'][:,:3][:, 2] > 0]
        sel = np.random.choice(valid, size=1024, replace=False)
        xyzw = obs['pointcloud']['xyzw'][sel]
        rgb = obs['pointcloud']['rgb'][sel]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], obs['extra']['tcp_pose'], obs['extra']['target']))

        return {'pointcloud':{'xyzw':xyzw, 'rgb':rgb}, 'extra':agent}