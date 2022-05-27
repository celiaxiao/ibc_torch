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

"""Utilities to precompute statistics on a dataset."""

from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import tqdm

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if x.ndim == 1:
          x = x[None,]
        if x.shape[1:] != self.mean.shape:
          x = x.reshape((-1,)+self.mean.shape)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
    @property
    def std(self):
      return np.sqrt(self.var)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def _action_update(action, min_action, max_action):
  """Updates the action statistics."""
  action = action.numpy()

  # Adding a batch dimension so that numpy can do per-dimension min
  action = action[None, ]
  # print("_action_update", action.shape)
  if None in min_action:
    min_action = action.min(axis=0)
    max_action = action.max(axis=0)
  else:
    min_action = np.minimum(min_action, action.min(axis=0))
    max_action = np.maximum(max_action, action.max(axis=0))
  # print("_action_update", max_action.shape)
  return min_action, max_action

def compute_dataset_statistics(dataset, num_samples, nested_obs=True,
                               nested_actions=False,
                               min_max_actions=False,
                               use_sqrt_std=False):
  """Uses Chan's algorithm to compute mean, std in a single pass.

  Args:
    dataset: Dataset to compute statistics on. Should return batches of (obs,
      action) tuples. If `nested` is not True obs, and actions should be
      flattened first.
    num_samples: Number of samples to take from the dataset.
    nested_obs: If True generates a nest of norm layers matching the obs
      structures.
    nested_actions: If True generates a nest of norm layers matching the action
      structures.
    min_max_actions: If True, use [-1,1] instead of 0-mean, unit-var.
    use_sqrt_std: If True, divide by sqrt(std) instead of std for normalization.

  Returns:
    obs_norm_layer, per-dimension normalizer to 0-mean, unit-variance
    act_norm_layer, per-dimension normalizer to 0-mean, unit-variance
    min_action, shape [dim_A], per-dimension max actions in dataset
    max_action, shape [dim_A], per-dimension min actions in dataset
  """
  obs_statistics = None
  act_statistics = None

  def identity(x):
    return x

  def sqrt(x):
    return np.sqrt(x)

  if use_sqrt_std:
    get = sqrt
  else:
    get = identity

  with tqdm.tqdm(
      desc="Computing Dataset Statistics", total=num_samples) as progress_bar:

    observation = None
    action = None

    for observation, action in dataset.unbatch().take(num_samples):
      flat_obs = [observation]
      if isinstance(observation, dict):
        flat_obs = [observation[key] for key in observation]
      flat_actions = [action]
      if isinstance(action, dict):
        flat_actions = [action[key] for key in action]
      if obs_statistics is None:
        # Initialize all params
        num_obs = len(flat_obs)
        num_act = len(flat_actions)

        if not nested_obs and num_obs > 1:
          raise ValueError("Found too many observations, make sure you set "
                           "`nested=True` or you flatten them.")

        if not nested_actions and num_act > 1:
          raise ValueError("Found too many actions, make sure you set "
                           "`nested=True` or you flatten them.")

        # [0] on the observation to take single value out of time dim.
        obs_statistics = [RunningMeanStd(shape=o.shape[-1:]) for o in flat_obs]
        act_statistics = [
            RunningMeanStd(shape=a.shape) for a in flat_actions
        ]
        print("flatten observtaion shape", [o.shape for o in flat_obs])
        print("flatten action shape", [a.shape for a in flat_actions])

        min_actions = [None for _ in range(num_act)]
        max_actions = [None for _ in range(num_act)]

      for obs, obs_stat in zip(flat_obs, obs_statistics):
        # Iterate over time dim.
        for o in obs:
          obs_stat.update(o.numpy())

      for act, act_stat in zip(flat_actions, act_statistics):
        act_stat.update(act.numpy())

      min_actions, max_actions = _action_update(action=flat_actions[0],
        min_action=min_actions, max_action=max_actions)

      progress_bar.update(1)
  assert obs_statistics[0].count > 2

  obs_norm_layers = []
  act_norm_layers = []
  act_denorm_layers = []
  for obs_stat in obs_statistics:
    obs_norm_layers.append(
        StdNormalizationLayer(mean=obs_stat.mean, std=get(obs_stat.std)))

  for act_stat in act_statistics:
    if not min_max_actions:
      act_norm_layers.append(
          StdNormalizationLayer(mean=act_stat.mean, std=get(act_stat.std)))
      act_denorm_layers.append(
          StdDenormalizationLayer(mean=act_stat.mean, std=get(act_stat.std)))
    else:
      act_norm_layers.append(
          MinMaxNormalizationLayer(vmin=min_actions, vmax=max_actions))
      act_denorm_layers.append(
          MinMaxDenormalizationLayer(vmin=min_actions, vmax=max_actions))

  if nested_obs:
    pass
  else:
    obs_norm_layers = obs_norm_layers[0]

  # actions will not be nested
  act_norm_layers = act_norm_layers[0]
  act_denorm_layers = act_denorm_layers[0]
  min_actions = torch.tensor(min_actions)
  max_actions = torch.tensor(max_actions)

  # Initialize act_denorm_layers:
  act_denorm_layers(min_actions)
  return (obs_norm_layers, act_norm_layers, act_denorm_layers, min_actions,
          max_actions)


EPS = torch.tensor(np.finfo(np.float32).eps)

class nestNormLayer(nn.Module):
  def __init__(self, obs_norm_layers):
    self.obs_norm_layers = obs_norm_layers
  
  def forward(self, observations):
    for index, key in enumerate(observations):
      obs = observations[key]
      observations[key] = self.obs_norm_layers[index](obs)


class IdentityLayer(nn.Module):

  def __init__(self, cast_dtype:torch.dtype):
    super(IdentityLayer, self).__init__()
    self.cast_dtype = cast_dtype

  def forward(self, x, ):
    return x.type(self.cast_dtype)


class StdNormalizationLayer(nn.Module):
  """Maps an un-normalized vector to zmuv."""

  def __init__(self, mean, std):
    super(StdNormalizationLayer, self).__init__()
    self._mean = torch.tensor(mean).float()
    self._std = torch.tensor(std).float()

  def forward(self, vector, ):
    vector = vector.float()
    return (vector - self._mean) / torch.maximum(self._std, EPS)


class StdDenormalizationLayer(nn.Module):
  """Maps a zmuv-normalized vector back to its original mean and std."""

  def __init__(self, mean, std):
    super(StdDenormalizationLayer, self).__init__()
    self._mean = torch.tensor(mean).float()
    self._std = torch.tensor(std).float()

  def forward(self, vector, mean_offset=True, ):
    vector = vector.float()
    result = (vector * torch.maximum(self._std, EPS))
    if mean_offset:
      result += self._mean
    return result


class MinMaxLayer(nn.Module):

  def __init__(self, vmin, vmax):
    super(MinMaxLayer, self).__init__()
    self._min = torch.tensor(vmin).float()
    self._max = torch.tensor(vmax).float()
    self._mean_range = (self._min + self._max) / 2.0
    self._half_range = (0.5*(self._max - self._min))
    # Half_range shouldn't already be negative.
    self._half_range = torch.maximum(self._half_range, EPS)


class MinMaxNormalizationLayer(MinMaxLayer):
  """Maps an un-normalized vector to -1, 1."""

  def forward(self, vector, ):
    vector = vector.float()
    return (vector - self._mean_range) / self._half_range


class MinMaxDenormalizationLayer(MinMaxLayer):
  """Maps -1, 1 vector back to un-normalized."""

  def forward(self, vector, ):
    vector = vector.float()
    return (vector * self._half_range) + self._mean_range
