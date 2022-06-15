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

"""Gets observation and action normalizers from data."""
import collections
import gin
from train import stats, constants
from train import get_data as data_module
import tensorflow as tf
import tensorflow_datasets as tfds

from eval.eval_env import get_env_name

def train_dataloader(dataset):
    return tfds.as_numpy(dataset)

def flatten_observation(sample, info):
  obs, action = sample
  flat_obs = tf.nest.flatten(obs)
  flat_obs = tf.concat(flat_obs, axis=-1)
  return (flat_obs, action), info


def drop_info_and_float_cast(sample, _):
  obs, action = sample

  for img_key in constants.IMG_KEYS:
    if isinstance(obs, dict) and img_key in obs:
      obs[img_key] = tf.image.convert_image_dtype(
          obs[img_key], dtype=tf.float32)

  return tf.nest.map_structure(lambda t: tf.cast(t, tf.float32), (obs, action))


NormalizationInfo = collections.namedtuple(
    'NormalizationInfo',
    ['obs_norm_layer', 'act_norm_layer', 'act_denorm_layer',
     'min_actions', 'max_actions'])


@gin.configurable
def get_normalizers(train_data,
                    batch_size,
                    env_name,
                    nested_obs=False,
                    nested_actions=False,
                    num_batches=100,
                    num_samples=None):
  """Computes stats and creates normalizer layers from stats."""
  statistics_dataset = train_data

  if env_name not in constants.D4RL_TASKS and not nested_obs:
    statistics_dataset = statistics_dataset.map(
        flatten_observation,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(100)

  statistics_dataset = statistics_dataset.map(
      drop_info_and_float_cast,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(100)

  # You can either ask for num_batches (by default, used originally),
  # or num_samples (which doesn't penalize you for using bigger batches).
  if num_samples is None:
    num_samples = num_batches * batch_size

  # Create observation and action normalization layers.
  (obs_norm_layer, act_norm_layer, act_denorm_layer,
   min_actions, max_actions) = (
       stats.compute_dataset_statistics(
           statistics_dataset,
           num_samples=num_samples,
           nested_obs=nested_obs,
           nested_actions=nested_actions))

  # Define a function used to normalize training data inside a tf.data .map().
  def norm_train_data_fn(experience):
    obs_and_act, _ = experience
    obs = obs_and_act[0]
    for img_key in constants.IMG_KEYS:
      if isinstance(obs, dict) and img_key in obs:
        obs[img_key] = tf.image.convert_image_dtype(
            obs[img_key], dtype=tf.float32)
    act = obs_and_act[1]
    normalized_obs = obs_norm_layer(obs)
    normalized_act = act_norm_layer(act)
    return (normalized_obs, normalized_act)

  norm_info = NormalizationInfo(obs_norm_layer,
                                act_norm_layer,
                                act_denorm_layer,
                                min_actions,
                                max_actions)
  return norm_info, norm_train_data_fn

if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)
    batch_size = 8
    env_name = get_env_name(task='PARTICLE', 
        shared_memory_eval=False, use_image_obs=False)
    create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
        dataset_path='data/particle/2d_oracle_particle_1.tfrecord',
    sequence_length=2, replay_capacity=10000, batch_size=batch_size, for_rnn=False,
    dataset_eval_fraction=0.0, flatten_action=True)
    train_data, _ = create_train_and_eval_fns_unnormalized()
    print('--------------------------dataset-------------------------------')
    print(train_data)
    # numpy_data = train_dataloader(train_data)
    # print(numpy_data)
    # for temp in train_data.as_numpy_iterator():
    #   experience,  _ = temp
    #   obs, act = experience
    #   print([obs[key].shape for key in obs], act.shape)
    # #   break

    # (norm_info, norm_train_data_fn) = get_normalizers(
    #   train_data, batch_size, env_name, nested_obs=True)
    # print(norm_info)
    # create_train_and_eval_fns = data_module.get_data_fns(
    #     dataset_path='data/particle/2d_oracle_particle*.tfrecord',
    # sequence_length=2, replay_capacity=10000, batch_size=batch_size, for_rnn=False,
    # dataset_eval_fraction=0.0, flatten_action=True, norm_function=norm_train_data_fn)
    # train_data, _ = create_train_and_eval_fns()
    # print(train_data)
    # count = 0
    for batch in train_dataloader(train_data):
      print(batch)
      

