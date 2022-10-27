# from diffuser.datasets.sequence import SequenceDataset
# import mani_skill2.envs

# dataset = SequenceDataset(env='maze2d-large-v1')
from environments.maniskill.maniskill_env import *
env = FillEnvParticle(control_mode='pd_joint_delta_pos', obs_mode='particles')
env.unwrapped
    
print(env._max_episode_steps)

print("observation space", env.observation_space.shape)