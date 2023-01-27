'''
Full pipeline to process maniskill demos in .h5 and .json to customized dataset.
Argument specification: TODO
exmaple to transform h5:
CUDA_VISIBLE_DEVICES=0 python data/maniskill_full_pipeline.py \
 --h5_path=data/softbody/Hang-v0/trajectory.none.pd_joint_delta_pos.h5 --json_path=data/softbody/Hang-v0/trajectory.none.pd_joint_delta_pos.json \
 --env_name=Hang-v0 --new_h5_path=data/softbody/Hang-v0/
'''
from absl import flags
import sys
import os
import warnings
# from pyrl.utils.data import GDict
import h5py
import json
import torch
import numpy as np
from data.dataset_maniskill import *
from environments.maniskill.maniskill_env import *
import gym

from train.utils import animate

flags.DEFINE_string('h5_path', None, 'h5 demo path')
flags.DEFINE_string('json_path', None, 'json demo path')
flags.DEFINE_string('stat_path', None, 'demo stats path across different models')

flags.DEFINE_string('env_name', 'Hang-v0', 'env name')
flags.DEFINE_enum('obs_mode', default='particles', enum_values=['particles', 'state', 'pointcloud'], help='')
flags.DEFINE_enum('control_mode', default='pd_joint_delta_pos', enum_values=['pd_joint_delta_pos', 'base_pd_joint_vel_arm_pd_joint_vel'], help='env control_mode')
flags.DEFINE_string('model_ids', None, 'model ids for the current env, used in OpenCabinet')
flags.DEFINE_integer('num_frames', None, 'number of frames to stack. If > 1, will use a frame_stack wrapper')
flags.DEFINE_string('raw_data_path', None, 'Path to save raw obs/act')
flags.DEFINE_string('dataset_path', None, 'Path to save torch dataset')
flags.DEFINE_string('new_h5_path', None, 'Path to save processed h5 file')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def extract_episodes(json_data, num_train_episodes, num_val_episodes=None):
    train_episodes = json_data[:num_train_episodes]
    val_episodes = None
    if num_val_episodes is not None:
        val_episodes = json_data[num_train_episodes: num_val_episodes + num_train_episodes]
    return train_episodes, val_episodes
    
def load_and_merge_open_cabinet_dataset(stat_path, h5_prefix, raw_data_path, dataset_path, new_h5_path, prefix):
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    stat_file = open(stat_path, "r")
    for line in stat_file:
        model_ids = line[:4]
        link = line[5]
        print(f"{model_ids=}, {link=}")
        h5_path = os.path.join(h5_prefix, model_ids, f"link_{link}", "trajectory.h5")
        json_path = os.path.join(h5_prefix, model_ids, f"link_{link}", "trajectory.json")
        target_env = gym.make(FLAGS.env_name, obs_mode=FLAGS.obs_mode, model_ids=model_ids)
        # TODO: hardcode extraing 30 episodes per (model, link) 
        obs, actions, rewards, dones, _ = replay_episodes(h5_path, json_path, target_env, num_train_episodes=30)
        all_obs = all_obs + obs
        all_actions = all_actions + actions
        all_rewards = all_rewards + rewards
        all_dones = all_dones + dones
        print(f'finished {model_ids=}, {link=}', len(all_obs), len(all_actions), len(all_rewards), len(all_dones))
    if new_h5_path:
        with h5py.File(f"{new_h5_path}/{prefix}_processed.h5", 'w') as f:
            f['observations'] = all_obs
            f['actions'] = all_actions
            f['rewards'] = all_rewards
            f['terminals'] = all_dones
        
        
def replay_episodes(h5_path, json_path, target_env, num_train_episodes=None, num_val_episodes=None):
    h5_demo = h5py.File(h5_path, 'r')
    # h5_demo = GDict.from_hdf5(h5_path)
    json_data = json.load(open(json_path))
    print("converting.....")
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_extras = []
    # TODO: not support for generating validation set yet
    if num_train_episodes is not None or num_val_episodes is not None:
        train_episodes, val_episodes = extract_episodes(json_data['episodes'], num_train_episodes, num_val_episodes)
        json_data['episodes'] = train_episodes
        
    for episode in json_data['episodes']:
        for i in range(1):
            episode_id = 'traj_' + str(episode['episode_id'])
            target_env.reset(seed=episode["episode_seed"], reconfigure=True)
            print("starting episode", episode_id, "episode_seed", episode["episode_seed"])
            action_count = 0
            for action in h5_demo[episode_id]['actions']:
                obs = target_env.get_obs()
                # stack the xyz and rgb into single vector
                if FLAGS.obs_mode == 'pointcloud':
                    all_obs.append(np.hstack((obs['pointcloud']['xyz'], obs['pointcloud']['rgb'])))
                    all_extras.append(obs['extra'])
                    if len(all_obs) == 1:
                        # Need to manually put these into config files
                        print('obs shape', all_obs[0].shape)
                        print('action shape', action.shape)
                else:
                    all_obs.append(obs)
                all_actions.append(action)
                
                if not isinstance(obs, dict)  and len(all_obs) == 1:
                    # Need to manually put these into config files
                    print('obs shape', obs.shape)
                    print('action shape', action.shape)

                if i > 0 and action_count > 20: # start to add noise after 50 steps
                    rand_noise = np.zeros_like(action)
                    if len(action) == 8:
                        rand_noise = np.concatenate((np.random.normal(scale=(0.1, 0.2, 0.5, 1, 1, 1.5, 2)), np.zeros(1)))
                    elif len(action) == 7:
                        add_noise = np.random.rand()
                        if add_noise > 0.7:
                            rand_noise = np.random.normal(scale=(0.1, 0.2, 0.5, 1, 1, 1, 1))
                    action = np.add(action, rand_noise)

                _, rew, done, _ = target_env.step(action)
                action_count += 1
                all_rewards.append(rew)
                all_dones.append(done)
                if done:
                    break
            # manually mark the end step to done=True
            if not all_dones[-1]:
                all_dones[-1] = True
                warnings.warn(f"manually mark {episode_id} to success")
            print(f'finished {episode_id}, success status {target_env.evaluate()}')
            print(len(all_obs), len(all_actions), len(all_rewards), len(all_dones))
    return all_obs, all_actions, all_rewards, all_dones, all_extras

def convert_dataset(h5_path, json_path, target_env, raw_data_path, dataset_path, new_h5_path, prefix):
    
    all_obs, all_actions, all_rewards, all_dones, all_extras = replay_episodes(h5_path, json_path, target_env)
    
    if raw_data_path:
        np.save(f"{raw_data_path}/{prefix}_observations.npy", all_obs)
        np.save(f"{raw_data_path}/{prefix}_actions.npy", all_actions)

    if dataset_path:
        dataset = maniskill_dataset(all_obs, all_actions)
        torch.save(dataset, f"{dataset_path}/{prefix}_dataset.pt")

    if new_h5_path:
        with h5py.File(f"{new_h5_path}/{prefix}_processed.h5", 'w') as f:
            f['observations'] = all_obs
            f['actions'] = all_actions
            f['rewards'] = all_rewards
            f['terminals'] = all_dones
            if FLAGS.obs_mode == 'pointcloud':
                f['extra'] = all_extras

if __name__ == '__main__':

    if FLAGS.raw_data_path and not os.path.exists(FLAGS.raw_data_path):
        os.makedirs(FLAGS.raw_data_path)
    if FLAGS.dataset_path and not os.path.exists(FLAGS.dataset_path):
        os.makedirs(FLAGS.dataset_path)
    if FLAGS.new_h5_path and not os.path.exists(FLAGS.new_h5_path):
        os.makedirs(FLAGS.new_h5_path)

    # Currently only supporting particles and pointcloud obs mode
    fn = None
    if FLAGS.obs_mode == 'particles':
        if FLAGS.env_name == 'Hang-v0':
            fn = HangEnvParticle
        elif FLAGS.env_name == 'Fill-v0':
            fn = FillEnvParticle
        elif FLAGS.env_name == 'Excavate-v0':
            fn = ExcavateEnvParticle
    elif FLAGS.obs_mode == 'pointcloud':
        if FLAGS.env_name == 'Hang-v0':
            fn = HangEnvPointcloud
        elif FLAGS.env_name == 'Fill-v0':
            fn = FillEnvPointcloud
        elif FLAGS.env_name == 'Excavate-v0':
            fn = ExcavateEnvPointcloud

    if fn is not None:
        target_env = fn(control_mode=FLAGS.control_mode, obs_mode=FLAGS.obs_mode)
    else:
        target_env = gym.make(FLAGS.env_name, control_mode=FLAGS.control_mode, obs_mode=FLAGS.obs_mode)
    
    if FLAGS.num_frames is not None and FLAGS.num_frames > 0:
        print("Using FrameStackWrapper with num_frames", FLAGS.num_frames)
        target_env = FrameStackWrapper(target_env, num_frames=FLAGS.num_frames)

    prefix = f"{FLAGS.env_name}_{FLAGS.obs_mode}_{FLAGS.control_mode}"
    if FLAGS.num_frames is not None and FLAGS.num_frames > 0:
        prefix = prefix + f"_frames_{FLAGS.num_frames}"
    print(prefix)
    if "OpenCabinet" in FLAGS.env_name:
        load_and_merge_open_cabinet_dataset(stat_path=FLAGS.stat_path, h5_prefix=FLAGS.h5_path, raw_data_path=FLAGS.raw_data_path, dataset_path=FLAGS.dataset_path, new_h5_path=FLAGS.new_h5_path, prefix=prefix)
        exit(0)
    convert_dataset(h5_path=FLAGS.h5_path, json_path=FLAGS.json_path, target_env=target_env, raw_data_path=FLAGS.raw_data_path, dataset_path=FLAGS.dataset_path, new_h5_path=FLAGS.new_h5_path, prefix=prefix)
