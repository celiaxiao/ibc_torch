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

"""MCMC algorithms to optimize samples from EBMs."""

import collections
import agents.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import numpy as np

# This global makes it easier to switch on/off tf.range in this file.
# Which I am often doing in order to debug anything in the binaries
# that use this.
my_range = torch.range
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def categorical_bincount(count, log_ps, n):
    """Given (un-norm.) log_probs, sample and count how many times for each index.

    Args:
        count: int, the amount of times to draw from the probabilities.
        log_ps: tensor shape [B x n], for each batch, n different numbers which can
        be unnormalized log probabilities that will get drawn from.
        n: should be the second entry in shape of log_ps.
    Returns:
        tensor shape [B x n], the counts of how many times each index was
        sampled.
    """
    # TODO: use default tensor for now (source:Force CPU since it's about 3x faster on CPU than GPU for current models.
    # samples shape: [B x n]
    # samples = torch.random.categorical(log_ps, num_samples=count, dtype=torch.int32)
    samples = torch.multinomial(torch.exp(log_ps), num_samples=count, replacement=True) # multinomial need probs
    # bincounts = tf.math.bincount(
    #     samples, minlength=n, maxlength=n, axis=-1)
    batch_size, count = samples.shape
    # intervals is used to separate tensor within each batch
    intervals = torch.arange(batch_size) * count # [batch_size,]
    intervals = intervals[:,None].expand(samples.shape) # [B,n]
    bincounts = torch.bincount((samples+intervals).reshape(-1), minlength=count*batch_size).reshape([batch_size, count]) # [B, n]
    return bincounts


def iterative_dfo(network,
                  batch_size,  # B
                  observations,  # B*n x obs_spec or B x obs_spec if late_fusion
                  action_samples,  # B*n x act_spec
                  policy_state,
                  num_action_samples,  # n
                  min_actions,
                  max_actions,
                  temperature=1.0,
                  num_iterations=3,
                  iteration_std=0.33,
                  training=False,
                  late_fusion=False,
                  tfa_step_type=()):
  """Update samples through ~Metropolis Hastings / CEM.

  Args:
    network: Any model that computes E(obs, act) in R^1
    batch_size: note that the first tensor dimension of observations and action_
      samples is actually batch_size * num_action_samples.  So we need to know
      the batch size so we can do reshaping for the softmax.
    observations: tensor shape [batch_size * num_action_samples x obs_spec] or
      [batch_size x obs_spec] if late_fusion
    action_samples: tensor shape [batch_size * num_action_samples x act_spec]
    policy_state: if the model is stateful, this is its state.
    num_action_samples: mixed in with batches in first dimension (see note
      above), this is the number of samples per batch.
    min_actions: shape (act_spec), clip to these min values during optimization.
    max_actions: shape (act_spec), clip to these max values during optimization.
    temperature: Scale the distribution by a temperature parameter.
    num_iterations: Number of DFO iterations to perform.
    iteration_std: Scale of the perturbation on the actions on every
      iteration.
    training: whether or not model is training.
    late_fusion: whether or not we are doing late fusion pixel-ebm.
    tfa_step_type: TF Agents step type.
  Returns:
    optimized probabilities, action_samples, new_policy_state
  """
  if late_fusion:
    # Embed observations once.
    obs_encodings = network.encode(observations, training=training)
    # Tile embeddings to match actions.
    obs_encodings = utils.tile_batch(obs_encodings, num_action_samples)

  def update_selected_actions(samples, policy_state):
    if late_fusion:
      # Repeatedly hand in the precomputed obs encodings.
      net_logits, new_policy_state = network(
          (observations, samples),
          step_type=tfa_step_type,
          training=training,
          network_state=policy_state,
          observation_encoding=obs_encodings)
    else:
      net_logits, new_policy_state = network(
          (observations, samples),
          step_type=tfa_step_type,
          network_state=policy_state,
          training=training)

    # Shape is just (B * n), for example (4096,) for B=2, n=2048
    net_logits = torch.reshape(net_logits, (batch_size, num_action_samples))
    # Shape is now (B, n), for example (2, 2048) for B=2, n=2048
    # Note: bincount takes log probabilities, and doesn't expect normalized,
    # so can skip softmax.
    log_probs = net_logits / temperature
    # Shape is still (B, n), for example (2, 2048) for B=2, n=2048
    actions_selected = categorical_bincount(num_action_samples, log_probs,
                                            num_action_samples)
    # Shape is still (B, n), for example (2, 2048) for B=2, n=2048
    # actions_selected = tf.ensure_shape(actions_selected, log_probs.shape)
    assert actions_selected.shape == log_probs.shape
    # actions_selected = tf.cast(actions_selected, dtype=tf.int32)
    actions_selected = actions_selected.type(torch.IntTensor)

    # Flatten back to (B * n), for example (4096,) for B=2, n=2048
    actions_selected = torch.reshape(actions_selected, (-1,))

    repeat_indices = torch.repeat_interleave(
        my_range(batch_size * num_action_samples), actions_selected)
    # repeat_indices = tf.ensure_shape(repeat_indices, actions_selected.shape)
    assert repeat_indices.shape == actions_selected.shape
    return log_probs, torch.gather(
        samples, dim =0, index=repeat_indices), new_policy_state

  log_probs, action_samples, new_policy_state = update_selected_actions(
      action_samples, policy_state)

  for _ in my_range(num_iterations - 1):
    # tf normal distribution default with mean=0.0, stddev=1.0,
    action_samples += torch.normal(mean=0,std=1,size=torch.shape(action_samples)) * iteration_std
    action_samples = torch.clamp(action_samples,
                                      min_actions,
                                      max_actions)
    log_probs, action_samples, new_policy_state = update_selected_actions(
        action_samples, new_policy_state)
    iteration_std *= 0.5  # Shrink sampling by half each iter.

  probs = nn.Softmax(log_probs, axis=1)
  probs = torch.reshape(probs, (-1,))
  # Shapes are: (B*n), (B*n x act_spec), and whatever for new_policy_state.
  return probs, action_samples, new_policy_state


def gradient_wrt_act(energy_network,
                     observations,
                     actions,
                     training,
                     network_state,
                     tfa_step_type,
                     apply_exp,
                     obs_encoding=None):
  """Compute dE(obs,act)/dact, also return energy."""
  # with tf.GradientTape() as g:
    # g.watch(actions)
  action_grad = actions.clone().detach().requires_grad_(True)
  if obs_encoding is not None:
    energies, _ = energy_network((observations, action_grad),
                                  training=training,
                                  network_state=network_state,
                                  step_type=tfa_step_type,
                                  observation_encoding=obs_encoding)
  else:
    energies, _ = energy_network((observations, action_grad),
                                  training=training,
                                  network_state=network_state,
                                  step_type=tfa_step_type)
  # If using a loss function that involves the exp(energies),
  # should we apply exp() to the energy when taking the gradient?
  if apply_exp:
    energies = torch.exp(energies)
  # My energy sign is flipped relative to Igor's code,
  # so -1.0 here.
  energies.backward()
  denergies_dactions = action_grad.grad.data * -1.0
  return denergies_dactions, energies


def compute_grad_norm(grad_norm_type, de_dact):
  """Given de_dact and the type, compute the norm."""
  if grad_norm_type is not None:
    grad_norm_type_to_ord = {'1': 1,
                             '2': 2,
                             'inf': np.inf}
    grad_type = grad_norm_type_to_ord[grad_norm_type]
    grad_norms = torch.linalg.norm(de_dact, axis=1, ord=grad_type)
  else:
    # It will be easier to manage downstream if we just fill this with zeros.
    # Rather than have this be potentially a None type.
    grad_norms = torch.zeros_like(de_dact[:, 0])
  return grad_norms


def langevin_step(energy_network,
                  observations,
                  actions,
                  training,
                  policy_state,
                  tfa_step_type,
                  noise_scale,
                  grad_clip,
                  delta_action_clip,
                  stepsize,
                  apply_exp,
                  min_actions,
                  max_actions,
                  grad_norm_type,
                  obs_encoding):
  """Single step of Langevin update."""
  l_lambda = 1.0
  # Langevin dynamics step
  de_dact, energies = gradient_wrt_act(energy_network,
                                       observations,
                                       actions,
                                       training,
                                       policy_state,
                                       tfa_step_type,
                                       apply_exp,
                                       obs_encoding)

  # This effectively scales the gradient as if the actions were
  # in a min-max range of -1 to 1.
  delta_action_clip = delta_action_clip * 0.5*(max_actions - min_actions)

  # TODO(peteflorence): can I get rid of this copy, for performance?
  # Times 1.0 since I don't trust tf.identity to make a deep copy.
  unclipped_de_dact = de_dact * 1.0
  grad_norms = compute_grad_norm(grad_norm_type, unclipped_de_dact)

  if grad_clip is not None:
    de_dact = torch.clamp(de_dact, -grad_clip, grad_clip)
  gradient_scale = 0.5  # this is in the Langevin dynamics equation.
  de_dact = (gradient_scale * l_lambda * de_dact +
             torch.normal(mean=0, std=1, size=torch.shape(actions)) * l_lambda * noise_scale)
  delta_actions = stepsize * de_dact

  # Clip to box.
  delta_actions = torch.clamp(delta_actions, -delta_action_clip,
                                   delta_action_clip)
  # TODO(peteflorence): investigate more clipping to sphere:
  # delta_actions = tf.clip_by_norm(
  #  delta_actions, delta_action_clip, axes=[1])

  actions = actions - delta_actions
  actions = torch.clamp(actions,
                             min_actions,
                             max_actions)

  return actions, energies, grad_norms


class ExponentialSchedule:
  """Exponential learning rate schedule for Langevin sampler."""

  def __init__(self, init, decay):
    self._decay = decay
    self._latest_lr = init

  def get_rate(self, index):
    """Get learning rate. Assumes calling sequentially."""
    del index
    self._latest_lr *= self._decay
    return self._latest_lr


class PolynomialSchedule:
  """Polynomial learning rate schedule for Langevin sampler."""

  def __init__(self, init, final, power, num_steps):
    self._init = init
    self._final = final
    self._power = power
    self._num_steps = num_steps

  def get_rate(self, index):
    """Get learning rate for index."""
    return ((self._init - self._final) *
            ((1 - (float(index) / float(self._num_steps-1))) ** (self._power))
            ) + self._final


def update_chain_data(num_iterations,
                      step_index,
                      actions,
                      energies,
                      grad_norms,
                      full_chain_actions,
                      full_chain_energies,
                      full_chain_grad_norms):
  """Helper function to keep track of data during the mcmc."""
  # I really wish tensorflow made assignment-by-index easy.
  # Then this function could just be:
  # full_chain_actions[step_index] = actions
  # full_chain_energies[step_index] = energies
  # full_chain_grad_norms[step_index] = grad_norms

  iter_onehot = F.one_hot(step_index, num_iterations)[Ellipsis, None]
  iter_onehot = torch.broadcast_to(iter_onehot, torch.shape(full_chain_energies))

  new_energies = energies * iter_onehot
  full_chain_energies += new_energies

  new_grad_norms = grad_norms * iter_onehot
  full_chain_grad_norms += new_grad_norms

  iter_onehot = iter_onehot[Ellipsis, None]
  iter_onehot = torch.broadcast_to(iter_onehot, torch.shape(full_chain_actions))
  actions_expanded = actions[None, Ellipsis]
  actions_expanded = torch.broadcast_to(actions_expanded, torch.shape(iter_onehot))
  new_actions_expanded = actions_expanded * iter_onehot
  full_chain_actions += new_actions_expanded
  return full_chain_actions, full_chain_energies, full_chain_grad_norms


@gin.configurable
def langevin_actions_given_obs(
    energy_network,
    observations,  # B*n x obs_spec or B x obs_spec if late_fusion
    action_samples,  # B*n x act_spec
    policy_state,
    min_actions,
    max_actions,
    num_action_samples,
    num_iterations=25,
    training=False,
    tfa_step_type=(),
    sampler_stepsize_init=1e-1,
    sampler_stepsize_decay=0.8,  # if using exponential langevin rate.
    noise_scale=1.0,
    grad_clip=None,
    delta_action_clip=0.1,
    stop_chain_grad=True,
    apply_exp=False,
    use_polynomial_rate=True,  # default is exponential
    sampler_stepsize_final=1e-5,  # if using polynomial langevin rate.
    sampler_stepsize_power=2.0,  # if using polynomial langevin rate.
    return_chain=False,
    grad_norm_type = 'inf',
    late_fusion=False):
  """Given obs and actions, use dE(obs,act)/dact to perform Langevin MCMC."""
  stepsize = sampler_stepsize_init
  # actions = tf.identity(action_samples)
  identity =  nn.Identity()
  actions = identity(action_samples)

  if use_polynomial_rate:
    schedule = PolynomialSchedule(sampler_stepsize_init, sampler_stepsize_final,
                                  sampler_stepsize_power, num_iterations)
  else:  # default to exponential rate
    schedule = ExponentialSchedule(sampler_stepsize_init,
                                   sampler_stepsize_decay)

  b_times_n = torch.shape(action_samples)[0]
  act_dim = torch.shape(action_samples)[-1]

  # Note 2: to work inside the tf.range, we have to initialize all these
  # outside the loop.

  # Note 1: for 1 step, there are [0, 1] points in the chain
  # grad norms will be for [0, ... N-1]

  # full_chain_actions is actually currently [1, ..., N]
  full_chain_actions = torch.zeros((num_iterations, b_times_n, act_dim))
  # full_chain_energies will also be for [0, ..., N-1]
  full_chain_energies = torch.zeros((num_iterations, b_times_n))
  # full_chain_grad_norms will be for [0, ..., N-1]
  full_chain_grad_norms = torch.zeros((num_iterations, b_times_n))

  # you can go compute Nth energy and grad_norm if you'd like later.
  if late_fusion:
    obs_encoding = energy_network.encode(observations, training=training)
    obs_encoding = utils.tile_batch(obs_encoding, num_action_samples)
  else:
    obs_encoding = None

  for step_index in my_range(num_iterations):
    actions, energies, grad_norms = langevin_step(energy_network,
                                                  observations,
                                                  actions,
                                                  training,
                                                  policy_state,
                                                  tfa_step_type,
                                                  noise_scale,
                                                  grad_clip,
                                                  delta_action_clip,
                                                  stepsize,
                                                  apply_exp,
                                                  min_actions,
                                                  max_actions,
                                                  grad_norm_type,
                                                  obs_encoding)
    # TODO: i don't see the graph of gradients of 'actions' is populated
    if stop_chain_grad:
      # actions = tf.stop_gradient(actions)
      actions = actions.detach()
    stepsize = schedule.get_rate(step_index + 1)  # Get it for the next round.

    if return_chain:
      (full_chain_actions, full_chain_energies,
       full_chain_grad_norms) = update_chain_data(num_iterations, step_index,
                                                  actions, energies, grad_norms,
                                                  full_chain_actions,
                                                  full_chain_energies,
                                                  full_chain_grad_norms)

  if return_chain:
    data_fields = ['actions', 'energies', 'grad_norms']
    ChainData = collections.namedtuple('ChainData', data_fields)
    chain_data = ChainData(full_chain_actions, full_chain_energies,
                           full_chain_grad_norms)
    return actions, chain_data
  else:
    return actions


def get_probabilities(energy_network,
                      batch_size,
                      num_action_samples,
                      observations,
                      actions,
                      training,
                      temperature=1.0):
  """Get probabilities to post-process Langevin results."""
  net_logits, _ = energy_network(
      (observations, actions), training=training)
  net_logits = torch.reshape(net_logits, (batch_size, num_action_samples))
  probs = nn.Softmax(net_logits / temperature, axis=1)
  probs = torch.reshape(probs, (-1,))
  return probs