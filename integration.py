# from diffuser.datasets.sequence import SequenceDataset
# import mani_skill2.envs

# dataset = SequenceDataset(env='maze2d-large-v1')
from environments.maniskill.maniskill_env import *
# env = FillEnvParticle(control_mode='pd_joint_delta_pos', obs_mode='particles')
# env.unwrapped
    
# print(env._max_episode_steps)

# print("observation space", env.observation_space.shape)
import open3d as o3d
import transforms3d
import gym
import mani_skill2.envs
def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

key_to_callback = {}
key_to_callback[ord("K")] = change_background_to_black

env = FillEnvPointcloud(control_mode='pd_joint_delta_pos')
# env = gym.make('Hang-v0', obs_mode='pointcloud', control_mode='pd_joint_delta_pos')
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
objects = o3d.geometry.PointCloud()
# objects.points = o3d.utility.Vector3dVector(obs['pointcloud'])
objects.points = o3d.utility.Vector3dVector(obs['pointcloud']['xyz'])
objects.colors = o3d.utility.Vector3dVector(obs['pointcloud']['rgb'] / 255.0)
img_res = (128, 128)
ext = None
int = None
o3d.visualization.draw_geometries_with_key_callbacks([objects], key_to_callback)

