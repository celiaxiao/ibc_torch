# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Binary to perform oracle data collection.

Note that in the above command TFAgent tfrecords and the episode json contain
redundant data. In most applications you would store one or the other.
"""

import os
from typing import Sequence
import torch
from absl import app
from absl import flags
from absl import logging
from data.transform_dataset import Ibc_dataset
from environments.block_pushing import block_pushing
from environments.collect.utils import get_env as get_env_module
from environments.collect.utils import get_oracle as get_oracle_module
from environments.collect.utils import serialize as serialize_module
import numpy as np  # pylint: disable=unused-import,g-bad-import-order
import tensorflow as tf
from tf_agents.environments import suite_gym  # pylint: disable=unused-import,g-bad-import-order
from tf_agents.trajectories import policy_step
from torch.utils.data import Dataset
flags.DEFINE_enum(
    'task',
    None,
    block_pushing.BlockTaskVariant._member_names_,  # pylint: disable=protected-access
    'Which task to run')
flags.DEFINE_bool('use_image_obs', False,
                  'Whether to include image observations.')
flags.DEFINE_bool('fixed_start_poses', False, 'Whether to use fixed start '
                  'poses.')
flags.DEFINE_bool('noisy_ee_pose', False, 'Whether to use noisy pose '
                  'for end effector so it does not start in exact position.')
flags.DEFINE_bool('no_episode_step_limit', False,
                  'If True, remove max_episode_steps step limit.')
flags.DEFINE_string(
    'dataset_path', './data',
    'If set a dataset of the oracle output will be saved '
    'to the given path.')
flags.DEFINE_integer('dataset_nshards', 1, 'Number of dataset shards to save.')
flags.DEFINE_string(
    'pybullet_state_path', None,
    'If set a json record of full pybullet, action and state '
    'will be saved to the given path.')
flags.DEFINE_bool('shared_memory', False, 'Shared memory for pybullet.')
flags.DEFINE_bool('save_successes_only', True,
                  'Whether to save only successful episodes.')
flags.DEFINE_integer('num_episodes', 1,
                     'The number of episodes to collect.')
flags.DEFINE_integer('worker_id', 0, 'Worker id of the replica.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas')
flags.DEFINE_bool('video', False,
                  'If true record a video of the evaluations.')
flags.DEFINE_string('video_path', None, 'Path to save videos at.')

FLAGS = flags.FLAGS
flags.mark_flags_as_required(['task'])

MAX_EPISODE_RECORD_STEPS = 10_000

def main(argv):
  del argv

  # Load an environment for this task with no step limit.
  env = get_env_module.get_env(
      FLAGS.task,
      use_image_obs=FLAGS.use_image_obs,
      fixed_start_poses=FLAGS.fixed_start_poses,
      noisy_ee_pose=FLAGS.noisy_ee_pose,
      max_episode_steps=np.inf if FLAGS.no_episode_step_limit else None)

  # Get an oracle.
  oracle_policy = get_oracle_module.get_oracle(env, FLAGS.task)

  # env resets are done via directly restoring pybullet state. Update
  # internal state now that we've added additional visual objects.
  if hasattr(env, 'save_state'):
    env.save_state()

  # If dataset_path specified, create tfrecord observers for writing out
  # training data.
  observers = serialize_module.get_tfrecord_observers(
      env,
      tfagents_path=FLAGS.dataset_path,
      worker_id=FLAGS.worker_id,
      dataset_nshards=FLAGS.dataset_nshards)

  cur_observer = 0
  num_episodes = 0
  num_failures = 0
  total_num_steps = 0
  experiences = []
  min_act = []
  max_act = []
  while True:
    logging.info('Starting episode %d.', num_episodes)
    episode_data = serialize_module.EpisodeData(
        time_step=[], action=[], pybullet_state=[])

    time_step = env.reset()
    episode_data.time_step.append(time_step)
    episode_data.pybullet_state.append(env.get_pybullet_state())

    if hasattr(env, 'instruction') and env.instruction is not None:
      logging.info('Current instruction: %s',
                   env.decode_instruction(env.instruction))

    done = time_step.is_last()
    reward = 0.0

    if 'instruction' in time_step.observation:
      instruction = time_step.observation['instruction']
      nonzero_ints = instruction[instruction != 0]
      nonzero_bytes = bytes(nonzero_ints)
      clean_text = nonzero_bytes.decode('utf-8')
      logging.info(clean_text)

    while not done:
      action = oracle_policy.action(time_step,
                                    oracle_policy.get_initial_state(1)).action

      time_step = env.step(action)

      if len(episode_data.action) < MAX_EPISODE_RECORD_STEPS:
        episode_data.action.append(
            policy_step.PolicyStep(action=action, state=(), info=()))
        episode_data.time_step.append(time_step)
        episode_data.pybullet_state.append(env.get_pybullet_state())

      done = time_step.is_last()
      reward = time_step.reward

    if done:  # episode terminated normally (not on manual reset w.o. saving).
      # Skip saving if it didn't end in success.
      if FLAGS.save_successes_only and reward <= 0:
        print('Skipping episode that did not end in success.')
        num_failures += 1
        continue

      num_episodes += 1

      if observers:
        total_num_steps += len(episode_data.action)
        print('Recording', len(episode_data.action), 'length episode to shard',cur_observer)
        # serialize_module.write_tfagents_data(
        #     episode_data, observers[cur_observer])
        # cur_observer = (cur_observer + 1) % len(observers)
        obs = [data.observation for data in episode_data.time_step]
        act = [data.action for data in episode_data.action]
        
        for i in range(len(act)):
          experience = {'observation':obs[i], 'action':act[i]}
          experiences.append(experience)
          actions = np.array(act)
          min_act.append(actions.min())
          max_act.append(actions.max())
        # print(obs)
        # print(act)
      if num_episodes >= FLAGS.num_episodes:
        experiences = np.array(experiences)
        dataset = Ibc_dataset(experiences)
        torch.save(dataset, FLAGS.dataset_path+'/block_push_states_location/multiple_push.pt')
        print("act min max", np.array(min_act).min(), np.array(max_act).max())
        print(
            'Num episodes:', num_episodes, 'Num failures:', num_failures )
        print('Avg steps:', total_num_steps / num_episodes)

        return


if __name__ == '__main__':
  app.run(main)
