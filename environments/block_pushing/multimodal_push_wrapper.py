import numpy as np
from environments.block_pushing.block_pushing_multimodal import BlockPushMultimodal

class BlockPushMultimodalWrapper(BlockPushMultimodal):
    def __init__(self, obs_mode, control_mode, *args, **kwargs):
        '''
        obs_mode: state or rgb
        '''
        super().__init__(*args, **kwargs)
        self._max_episode_steps = self.max_episode_steps
    
    def get_obs(self):
        obs_dict = super()._compute_state()
        observation = np.concatenate([obs_dict['block_translation'], obs_dict['block_orientation'], obs_dict['target_translation'], obs_dict['target_orientation'],
                                      obs_dict['block2_translation'], obs_dict['block2_orientation'], obs_dict['target2_translation'], obs_dict['target2_orientation'],
                                      obs_dict['effector_translation'], obs_dict['effector_target_translation']], axis=-1).flatten()
        
        return observation
    
    def reset(self, seed=None, reset_poses = True):
        if seed is not None:
            np.random.seed(seed)
        observation = super().reset(reset_poses)
        return observation