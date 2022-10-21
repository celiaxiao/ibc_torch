# from diffuser.datasets.sequence import SequenceDataset
# import mani_skill2.envs

# dataset = SequenceDataset(env='maze2d-large-v1')
from environments.maniskill.maniskill_env import *
env = HangEnvParticle(control_mode='pd_joint_delta_pos', obs_mode='particles')
env.unwrapped
dataset = (env.get_dataset('./data/softbody/Hang-v0/Hang-v0_particles_pd_joint_delta_pos_processed.h5'))
print(dataset['terminals'][-100:])
for i in range(len(dataset['terminals'])):
    if dataset['terminals'][i] == True:
        # print(i)
        pass
    
print(env._max_episode_steps)

print("observation space", env.observation_space.shape)