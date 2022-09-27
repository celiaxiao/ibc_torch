'''
Full pipeline to process maniskill demos in .h5 and .json to customized dataset.
Argument specification: TODO
'''
from absl import flags
import sys
import os

# from pyrl.utils.data import GDict
import h5py
import json
import torch
import numpy as np
from data.dataset_maniskill import *
from environments.maniskill.maniskill_env import *

flags.DEFINE_string('h5_path', None, 'h5 demo path')
flags.DEFINE_string('json_path', None, 'json demo path')
flags.DEFINE_string('env_name', None, 'env name')
flags.DEFINE_enum('obs_mode', default='particles', enum_values=['particles'], help='')
flags.DEFINE_enum('control_mode', default='pd_joint_delta_pos', enum_values=['pd_joint_delta_pos'], help='env control_mode')
flags.DEFINE_string('raw_data_path', None, '')
flags.DEFINE_string('dataset_path', None, '')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def convert_dataset(h5_path, json_path, target_env, raw_data_path, dataset_path, prefix):
    h5_demo = h5py.File(h5_path, 'r')
    # h5_demo = GDict.from_hdf5(h5_path)
    json_data = json.load(open(json_path))

    all_obs = []
    all_actions = []

    for episode in json_data['episodes']:
        episode_id = 'traj_' + str(episode['episode_id'])
        target_env.reset(seed=episode["episode_seed"])

        for action in h5_demo[episode_id]['actions']:
            obs = target_env.get_obs()

            all_obs.append(obs)
            all_actions.append(action)
            
            if len(all_obs) == 1:
                # Need to manually put these into config files
                print('obs shape', obs.shape)
                print('action shape', action.shape)

            target_env.step(action)

        print(f'finished {episode_id}, success status {target_env.evaluate()}')

    np.save(f'{raw_data_path}/{prefix}_observations.npy', all_obs)
    np.save(f'{raw_data_path}/{prefix}_actions.npy', all_actions)

    dataset = maniskill_dataset(all_obs, all_actions)
    torch.save(dataset, f"{dataset_path}/{prefix}_dataset.pt")

if __name__ == '__main__':

    if not os.path.exists(FLAGS.raw_data_path):
        os.makedirs(FLAGS.raw_data_path)
    if not os.path.exists(FLAGS.dataset_path):
        os.makedirs(FLAGS.dataset_path)

    # Currently only supporting particles obs mode
    if FLAGS.env_name == 'Hang-v0':
        fn = HangEnvParticle
    elif FLAGS.env_name == 'Fill-v0':
        fn = FillEnvParticle
    elif FLAGS.env_name == 'Excavate-v0':
        fn = ExcavateEnvParticle

    target_env = fn(control_mode=FLAGS.control_mode, obs_mode=FLAGS.obs_mode)
    prefix = f"{FLAGS.env_name}_{FLAGS.obs_mode}_{FLAGS.control_mode}"
    print(prefix)

    convert_dataset(h5_path=FLAGS.h5_path, json_path=FLAGS.json_path, target_env=target_env, raw_data_path=FLAGS.raw_data_path, dataset_path=FLAGS.dataset_path, prefix=prefix)