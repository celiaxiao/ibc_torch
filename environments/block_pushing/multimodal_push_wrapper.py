import numpy as np
from environments.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from environments.maniskill.maniskill_env import ExtendedWrapper

class BlockPushMultimodalWrapper(BlockPushMultimodal):
    def __init__(self, obs_mode, control_mode, *args, **kwargs):
        '''
        obs_mode: state or rgb
        '''
        super().__init__(*args, **kwargs)
        self._max_episode_steps = 200
    
    def get_obs(self):
        obs_dict = super()._compute_state()
        observation = np.concatenate([obs_dict['block_translation'], obs_dict['block_orientation'], obs_dict['target_translation'], obs_dict['target_orientation'],
                                      obs_dict['block2_translation'], obs_dict['block2_orientation'], obs_dict['target2_translation'], obs_dict['target2_orientation'],
                                      obs_dict['effector_translation'], obs_dict['effector_target_translation']], axis=-1).flatten()
        
        return observation
    
    def flatten_obs(self, obs_dict):
        return np.concatenate([obs_dict['block_translation'], obs_dict['block_orientation'], obs_dict['target_translation'], obs_dict['target_orientation'],
                                      obs_dict['block2_translation'], obs_dict['block2_orientation'], obs_dict['target2_translation'], obs_dict['target2_orientation'],
                                      obs_dict['effector_translation'], obs_dict['effector_target_translation']], axis=-1).flatten()
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self.flatten_obs(observation), reward, done, info
    
    def reset(self, seed=None, reset_poses = True):
        if seed is not None:
            np.random.seed(seed)
        observation = super().reset(reset_poses)
        return self.flatten_obs(observation)

class HistorykWrapper(ExtendedWrapper):
    def __init__(self, env, history_len: int, **kwargs) -> None:
        super().__init__(env)
        self.history_len = history_len
        self.obs_mode = getattr(self.env, "obs_mode", "state")
        self.history_observations = []
        self.pos_encoding = np.eye(history_len, dtype=np.uint8)

    def observation(self):
        return np.concat(self.history_observations, axis=0, wrapper=False)
    
    def action(self):
        return np.concat(self.history_actions, axis=0, wrapper=False)
    
    def step(self, actions, idx=None):
        next_obs, rewards, dones, infos = self.env.step(actions)
        self.history_observations = self.history_observations[1:] + [dict(next_obs)]
        return self.observation(), rewards, dones, infos

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        print(obs_dict)
        obs = np.concatenate([obs_dict['block_translation'], obs_dict['block_orientation'], obs_dict['target_translation'], obs_dict['target_orientation'],
                                      obs_dict['block2_translation'], obs_dict['block2_orientation'], obs_dict['target2_translation'], obs_dict['target2_orientation'],
                                      obs_dict['effector_translation'], obs_dict['effector_target_translation']], axis=-1).flatten()
        self.frames = [obs] * self.history_len
        return self.frames
    
    def get_obs(self):
        return self.observation()
