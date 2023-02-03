from mani_skill2.envs.mpm.hang_env import HangEnv
from mani_skill2.envs.mpm.fill_env import FillEnv
from mani_skill2.envs.mpm.excavate_env import ExcavateEnv
from mani_skill2.envs.ms1.open_cabinet_door_drawer import OpenCabinetDoorEnv
import numpy as np
from numpy.random import default_rng
from gym.core import Wrapper
import json

from pyrl.utils.data import GDict
from pyrl.utils.meta import Registry

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
        kwargs['obs_mode'] = 'pointcloud'
        self._max_episode_steps = 350
        super().__init__(*args, **kwargs)

    def get_obs(self):
        obs = super().get_obs()
        xyz = obs['pointcloud']['xyzw'][:,:3]
        valid = np.array(range(len(obs['pointcloud']['xyzw'])))[obs['pointcloud']['xyzw'][:,:3][:, 2] > 0.01]
        sel = np.random.choice(valid, size=1024, replace=False)
        xyz = xyz[sel]
        rgb = obs['pointcloud']['rgb'][sel]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], obs['extra']['tcp_pose'], obs['extra']['target']))

        return {'pointcloud':{'xyz':xyz, 'rgb':rgb}, 'extra':agent}

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
        kwargs['obs_mode'] = 'pointcloud'
        self._max_episode_steps = 250
        super().__init__(*args, **kwargs)

    def get_obs(self):
        obs = super().get_obs()
        xyz = obs['pointcloud']['xyzw'][:,:3]
        valid = np.array(range(len(obs['pointcloud']['xyzw'])))[obs['pointcloud']['xyzw'][:,:3][:, 2] > 0.01]
        sel = np.random.choice(valid, size=1024, replace=False)
        xyz = xyz[sel]
        rgb = obs['pointcloud']['rgb'][sel]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], obs['extra']['tcp_pose'], obs['extra']['target']))

        return {'pointcloud':{'xyz':xyz, 'rgb':rgb}, 'extra':agent}

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
        kwargs['obs_mode'] = 'pointcloud'
        self._max_episode_steps = 350
        super().__init__(*args, **kwargs)

    def get_obs(self):
        obs = super().get_obs()
        xyz = obs['pointcloud']['xyzw'][:,:3]
        valid = np.array(range(len(obs['pointcloud']['xyzw'])))[obs['pointcloud']['xyzw'][:,:3][:, 2] > 0.01]
        sel = np.random.choice(valid, size=1024, replace=False)
        xyz = xyz[sel]
        rgb = obs['pointcloud']['rgb'][sel]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], obs['extra']['tcp_pose'], obs['extra']['target']))

        return {'pointcloud':{'xyz':xyz, 'rgb':rgb}, 'extra':agent}

WRAPPERS = Registry("wrappers of gym environments")
class ExtendedWrapper(Wrapper):
    def __getattr__(self, name):
        # gym standard do not support name with '_'
        return getattr(self.env, name)
@WRAPPERS.register_module()
class FrameStackWrapper(ExtendedWrapper):
    def __init__(self, env, num_frames: int, **kwargs) -> None:
        super().__init__(env)
        self.num_frames = num_frames
        self.obs_mode = getattr(self.env, "obs_mode", "state")
        self.frames = []
        self.pos_encoding = np.eye(num_frames, dtype=np.uint8)

    def observation(self):
        if self.obs_mode == "pointcloud":
            num_points = self.frames[0]['pointcloud']["xyz"].shape[0]
            pos_encoding = np.repeat(self.pos_encoding, num_points, axis=0)
            obs = GDict.concat(self.frames, axis=0, wrapper=False)
            # use the last frame extra information
            last_frame = self.frames[-1]
            last_extra_info = last_frame['extra']
            # for frame in self.frames:
            #     print("self.frames", frame["pointcloud"]["xyz"].shape)
            # print(f"{pos_encoding=}")
            obs["pos_encoding"] = pos_encoding
            obs["pointcloud"]["xyz"] = np.concatenate([obs["pointcloud"]["xyz"], obs["pos_encoding"]], axis=-1)
            obs["pointcloud"]["rgb"] = np.concatenate([obs["pointcloud"]["rgb"], obs["pos_encoding"]], axis=-1)
            obs['extra'] = last_extra_info
            return obs
        else:
            return GDict.concat(self.frames, axis=-3, wrapper=False)

    def step(self, actions, idx=None):
        next_obs, rewards, dones, infos = self.env.step(actions)
        self.frames = self.frames[1:] + [next_obs]
        return self.observation(), rewards, dones, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames = [obs] * self.num_frames
        return self.observation()
    
    def get_obs(self):
        return self.observation()

class OpenCabinetDoorState(OpenCabinetDoorEnv):
    def __init__(self, *args, **kwargs):
        
        self._max_episode_steps = 200
        self.model_dict = json.load(open('/home/caiwei/data/rigid_body_envs/OpenCabinetDoor-v1/models_encode_dict.txt'))
        super().__init__(*args, **kwargs)
        # self.model_encoding = model_encoding
        
    def get_obs(self):
        obs = super().get_obs()
        # print(f"{self.model_id=}, {self.target_link_idx=}")
        model_count = self.model_dict[self.model_id][str(self.target_link_idx)]
        model_encoding = np.zeros(66)
        model_encoding[model_count] = 1
        return np.hstack([obs, model_encoding])
        
        