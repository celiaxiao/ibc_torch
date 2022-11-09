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

# env = FillEnvPointcloud(control_mode='pd_joint_delta_pos')
env = gym.make('Hang-v0', obs_mode='pointcloud', control_mode='pd_joint_delta_pos')
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
objects = o3d.geometry.PointCloud()
# objects.points = o3d.utility.Vector3dVector(obs['pointcloud'])
objects.points = o3d.utility.Vector3dVector(obs['pointcloud']['xyzw'][:,:3])
objects.colors = o3d.utility.Vector3dVector(obs['pointcloud']['rgb'])
img_res = (128, 128)
ext = None
int = None
o3d.visualization.draw_geometries_with_key_callbacks([objects], key_to_callback)

def get_o3d_camera(int, image_res):
    fx, fy = int[0, 0], int[1, 1]
    cx, cy = int[0, 2], int[1, 2]
    w, h = image_res[1], image_res[0]
    cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    return cam

def get_int(image_res):
    fov = 0.23
    image_res = image_res
    int = np.array([
        np.array([2 * fov / image_res[1], 0, -fov - 1e-5,]),
        np.array([0, 2 * fov / image_res[1], -fov - 1e-5,]),
        [0, 0, 1]
    ])
    return np.linalg.inv(int)

vis = o3d.visualization.Visualizer()
vis.create_window(width=512, height=512, visible=False)
if isinstance(objects, tuple) or isinstance(objects, list):
    for geom in objects:
        vis.add_geometry(geom)
else:
    vis.add_geometry(objects)
ctr = vis.get_view_control()

if int is None:
    int = get_int(img_res)

cam_param = get_o3d_camera(int, img_res)
o3d_cam = o3d.camera.PinholeCameraParameters()
o3d_cam.intrinsic = cam_param
if ext is None:
    def lookat(center, theta, phi, radius):
        R = transforms3d.euler.euler2mat(theta, phi, 0., 'sxyz')
        b = np.array([0, 0, radius], dtype=float)
        back = R[0:3, 0:3].dot(b)
        return R, center - back
    R, t = lookat([0.5, 0.5, 0.5], 0., 0., 3.)
    ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
    ext = np.linalg.pinv(ext)
o3d_cam.extrinsic = ext #self.get_ext()
#change bg color
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])

ctr.convert_from_pinhole_camera_parameters(o3d_cam, allow_arbitrary=True)
vis.update_renderer()
image = vis.capture_screen_float_buffer(do_render=True)
vis.destroy_window()
image = np.uint8(np.asarray(image) * 255)
from PIL import Image
im = Image.fromarray(image)
im.save("test_pointcloud.png")