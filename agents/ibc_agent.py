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

"""Implicit BC agent."""

import copy
import functools



from agents import mcmc
from losses import emb_losses
from losses import gradient_losses
from agents.utils import dict_flatten, tile_batch
import torch
import torch.nn as nn

class ImplicitBCAgent():
  """implementing implicit behavioral cloning."""

  def __init__(self,
               action_spec:int,
               cloning_network,
               optimizer,
               min_action=None,
               max_action=None,
               obs_norm_layer=None,
               act_norm_layer=None,
               act_denorm_layer=None,
               num_counter_examples=256,
               debug_summaries = False,
               summarize_grads_and_vars = False,
               train_step_counter = 0,
               fraction_dfo_samples=0.,
               fraction_langevin_samples=1.0,
               ebm_loss_type='info_nce',
               late_fusion=False,
               compute_mse=False,
               run_full_chain_under_gradient=False,
               return_full_chain=True,
               add_grad_penalty=True,
               grad_norm_type='inf',
               softmax_temperature=1.0):
    super().__init__()
    self.action_spec = action_spec
    self.min_action = min_action
    self.max_action = max_action
    self._action_spec = action_spec
    self._obs_norm_layer = obs_norm_layer
    self._act_norm_layer = act_norm_layer
    self._act_denorm_layer = act_denorm_layer
    self.cloning_network = cloning_network

    self._optimizer = optimizer
    self._num_counter_examples = num_counter_examples
    self._fraction_dfo_samples = fraction_dfo_samples
    self._fraction_langevin_samples = fraction_langevin_samples
    assert self._fraction_dfo_samples + self._fraction_langevin_samples <= 1.0
    assert self._fraction_dfo_samples >= 0.
    assert self._fraction_langevin_samples >= 0.
    self.ebm_loss_type = ebm_loss_type

    self._run_full_chain_under_gradient = run_full_chain_under_gradient

    self._return_full_chain = return_full_chain
    self._add_grad_penalty = add_grad_penalty
    self._grad_norm_type = grad_norm_type

    self._softmax_temperature = softmax_temperature

    self._late_fusion = late_fusion
    self._compute_mse = compute_mse

    self.train_step_counter = train_step_counter

    # Collect policy would normally be used for data collection. In a BCAgent
    # we don't expect to use it, unless we want to upgrade this to a DAGGER like
    # setup.
    # collect_policy = ibc_policy.IbcPolicy(
    #     time_step_spec=time_step_spec,
    #     action_spec=action_spec,
    #     action_spec=action_spec,
    #     actor_network=cloning_network,
    #     late_fusion=late_fusion,
    #     obs_norm_layer=self._obs_norm_layer,
    #     act_denorm_layer=self._act_denorm_layer,
    # )
    # policy = greedy_policy.GreedyPolicy(collect_policy)
    if self.ebm_loss_type == 'info_nce':
      self._kl = nn.KLDivLoss(reduction='none')

  def train(self, experience):
    loss_info = self._loss(experience)
    assert torch.isfinite(loss_info['loss']).all()
    self._optimizer.zero_grad()
    loss_info['loss'].mean().backward()
    self._optimizer.step()
    self.train_step_counter += 1
    return loss_info

  def get_eval_loss(self, experience):
    loss_dict = self._loss(experience,)
    return loss_dict
# 
  def _loss(self,
            experience):
    # ** Note **: Obs spec includes time dim. but hilighted here since we have
    # to deal with it.
    # Observation: [B , T , obs_spec]
    # Action:      [B , act_spec]
    observations, actions = experience

    # Use first observation to figure out batch/time sizes as they should be the
    # same across all observations.
    # single_obs = tf.nest.flatten(observations)[0]
    if isinstance(observations, dict):
        single_obs = observations[list(observations.keys())[0]]
        observations = dict_flatten(observations)
    else:
        single_obs = observations
    # print('single_obs', single_obs.shape)
    batch_size = single_obs.shape[0]
    observations = observations.reshape([batch_size, -1])
    # Now tile and setup observations to be: [B * n+1 x obs_spec]
    # TODO(peteflorence): could potentially save memory by not calling
    # tile_batch both here and in _make_counter_example_actions...
    # but for a later optimization.
    # They only differ by having one more tiled obs or not.
    if self._late_fusion:
      maybe_tiled_obs = observations
    else:
      maybe_tiled_obs = tile_batch(observations,
                                              self._num_counter_examples + 1)
    assert maybe_tiled_obs.shape[0] == batch_size * (self._num_counter_examples + 1)
    # print('maybe_tiled_obs', maybe_tiled_obs.shape)
    # [B x 1 x act_spec]
    expanded_actions = actions[:,None]
    # generate counter examples
    if not self._run_full_chain_under_gradient:
      counter_example_actions, combined_true_counter_actions, chain_data = (
          self._make_counter_example_actions(observations,
                                             expanded_actions.detach(), batch_size))
    else:
      counter_example_actions, combined_true_counter_actions, chain_data = (
          self._make_counter_example_actions(observations,
                                             expanded_actions, batch_size))
    # print('combined_true_counter_actions',combined_true_counter_actions.shape)
    # print('counter_example_actions',counter_example_actions.shape)
    
    assert combined_true_counter_actions.shape == \
      torch.Size([batch_size * (self._num_counter_examples + 1), self.action_spec])
    
    if self._late_fusion:
        # Do one cheap forward pass.
        obs_embeddings = self.cloning_network.encode(maybe_tiled_obs)
        # Tile embeddings to match actions.
        obs_embeddings = tile_batch(
            obs_embeddings, self._num_counter_examples + 1)
        # Feed in the embeddings as "prior embeddings"
        # (with no subsequent step). Network does nothing with unused_obs.
        unused_obs = maybe_tiled_obs
        network_inputs = (unused_obs,
                        combined_true_counter_actions.detach())
        # [B * n+1]
        predictions = self.cloning_network(
            network_inputs, observation_encoding=obs_embeddings)

    else:
        network_inputs = (maybe_tiled_obs,
                        combined_true_counter_actions.detach())
        # [B * n+1]
        predictions = self.cloning_network(network_inputs)
        # print(predictions.size(), network_inputs[0].size(), network_inputs[1].size())
    # [B, n+1]
    predictions = torch.reshape(predictions,
                                [batch_size, self._num_counter_examples + 1])

    per_example_loss, debug_dict = self._compute_ebm_loss(
        batch_size, predictions)

    if self._add_grad_penalty:
        grad_loss = gradient_losses.grad_penalty(
            self.cloning_network,
            self._grad_norm_type,
            batch_size,
            chain_data,
            maybe_tiled_obs,
            combined_true_counter_actions,
        )
        per_example_loss += grad_loss
    else:
        grad_loss = None

    # TODO(peteflorence): add energy regularization?

    # Aggregate losses uses some TF magic to make sure aggregation across
    # TPU replicas does the right thing. It does mean we have to calculate
    # per_example_losses though.
    # agg_loss = common.aggregate_losses(
    #     per_example_loss=per_example_loss,
    #     sample_weight=weights,
    #     regularization_loss=self.cloning_network.losses)
    # total_loss = agg_loss.total_loss

    losses_dict = {
        'loss': per_example_loss
    }

    losses_dict.update(debug_dict)
    if grad_loss is not None:
        losses_dict['grad_loss'] = torch.mean(grad_loss)


    opt_dict = dict()
    if chain_data is not None and chain_data.energies is not None:
        energies = chain_data.energies
        opt_dict['overall_energies_avg'] = torch.mean(energies)
        first_energies = energies[0]
        opt_dict['first_energies_avg'] = torch.mean(first_energies)
        final_energies = energies[-1]
        opt_dict['final_energies_avg'] = torch.mean(final_energies)

    if chain_data is not None and chain_data.grad_norms is not None:
        grad_norms = chain_data.grad_norms
        opt_dict['overall_grad_norms_avg'] = torch.mean(grad_norms)
        first_grad_norms = grad_norms[0]
        opt_dict['first_grad_norms_avg'] = torch.mean(first_grad_norms)
        final_grad_norms = grad_norms[-1]
        opt_dict['final_grad_norms_avg'] = torch.mean(final_grad_norms)

    losses_dict.update(opt_dict)
    return losses_dict

  def _compute_ebm_loss(
      self,
      batch_size,  # B
      predictions,  # [B x n+1] with true in column [:, -1]
      ):
    
    per_example_loss, debug_dict = emb_losses.info_nce(
        predictions, batch_size, self._num_counter_examples,
        self._softmax_temperature, self._kl)
    return per_example_loss, debug_dict

  def _make_counter_example_actions(
      self,
      observations,  # B x obs_spec
      expanded_actions,  # B x 1 x act_spec
      batch_size):
    """Given observations and true actions, create counter example actions."""
    # Note that T (time dimension) would be included in obs_spec.
    # Counter example actions [B , num_counter_examples , act_spec]
    if len(self.min_action) > 1:
      random_uniform_example_actions = \
        torch.distributions.uniform.Uniform(self.min_action,self.max_action).sample(\
            [batch_size, self._num_counter_examples])
    else:
      random_uniform_example_actions = \
        torch.distributions.uniform.Uniform(self.min_action,self.max_action).sample(\
            [batch_size, self._num_counter_examples, self._action_spec])
    random_uniform_example_actions = random_uniform_example_actions.reshape((batch_size * self._num_counter_examples, self._action_spec))
    # print("check random uniform sample shape", random_uniform_example_actions.shape)
    # If not optimizing, just return.
    if (self._fraction_dfo_samples == 0.0 and
        self._fraction_langevin_samples == 0.0):
      counter_example_actions = random_uniform_example_actions
      chain_data = None
    else:
      # Reshape to put B and num counter examples on same tensor dimenison
      # [B*num_counter_examples x act_spec]
      random_uniform_example_actions = torch.reshape(
          random_uniform_example_actions,
          (batch_size * self._num_counter_examples, -1))

      if self._late_fusion:
        maybe_tiled_obs_n = observations
      else:
        maybe_tiled_obs_n = tile_batch(observations,
                                                  self._num_counter_examples)

      dfo_opt_counter_example_actions = None
      if self._fraction_dfo_samples > 0.:
        if self._return_full_chain:
          raise NotImplementedError('Not implemented to return dfo chain.')

        # Use all uniform actions to seed the optimization,
        # even though we will only pick a subset later.
        _, dfo_opt_counter_example_actions, _ = mcmc.iterative_dfo(
            self.cloning_network,
            batch_size,
            maybe_tiled_obs_n,
            random_uniform_example_actions,
            policy_state=(),
            num_action_samples=self._num_counter_examples,
            min_actions=self.min_action,
            max_actions=self.max_action,
            late_fusion=self._late_fusion,)
        chain_data = None
      lang_opt_counter_example_actions = None
      if self._fraction_langevin_samples > 0.:
        # TODO(peteflorence): in the case of using a fraction <1.0,
        # we could reduce the amount in langevin that are optimized.
        langevin_return = mcmc.langevin_actions_given_obs(
            self.cloning_network,
            maybe_tiled_obs_n,
            random_uniform_example_actions,
            min_actions=self.min_action,
            max_actions=self.max_action,
            return_chain=self._return_full_chain,
            grad_norm_type=self._grad_norm_type,
            num_action_samples=self._num_counter_examples)
        if self._return_full_chain:
          lang_opt_counter_example_actions, chain_data = langevin_return
        else:
          lang_opt_counter_example_actions = langevin_return
          chain_data = None

      list_of_counter_examples = []

      fraction_init_samples = (1. - self._fraction_dfo_samples -
                               self._fraction_langevin_samples)

      # Compute indices based on fractions.
      init_num_indices = int(fraction_init_samples * self._num_counter_examples)
      dfo_num_indices = (
          int(self._fraction_dfo_samples * self._num_counter_examples))
      langevin_num_indices = (
          int(self._fraction_langevin_samples * self._num_counter_examples))
      residual = (
          self._num_counter_examples - init_num_indices - dfo_num_indices -
          langevin_num_indices)
      assert residual >= 0
      # If there was a rounding that caused a residual, ascribe those to init.
      init_num_indices += residual

      used_index = 0
      if init_num_indices > 0:
        some_init_counter_example_actions = torch.reshape(
            random_uniform_example_actions,
            (batch_size, self._num_counter_examples,
             -1))[:, :init_num_indices, :]
        used_index += init_num_indices
        list_of_counter_examples.append(some_init_counter_example_actions)

      if dfo_num_indices > 0.:
        next_index = used_index + dfo_num_indices
        some_dfo_counter_example_actions = torch.reshape(
            dfo_opt_counter_example_actions,
            (batch_size, self._num_counter_examples,
             -1))[:, used_index:next_index, :]
        used_index = next_index
        list_of_counter_examples.append(some_dfo_counter_example_actions)

      if langevin_num_indices > 0.:
        next_index = used_index + langevin_num_indices
        some_lang_counter_example_actions = torch.reshape(
            lang_opt_counter_example_actions,
            (batch_size, self._num_counter_examples,
             -1))[:, used_index:next_index, :]
        used_index = next_index
        list_of_counter_examples.append(some_lang_counter_example_actions)

      assert used_index == self._num_counter_examples

      counter_example_actions = torch.concat(list_of_counter_examples, axis=1)
    counter_example_actions = torch.reshape(
        counter_example_actions, (batch_size, self._num_counter_examples, -1))

    def concat_and_squash_actions(counter_example, action):
      return torch.reshape(
          torch.concat([counter_example, action], axis=1),
          [-1, self.action_spec])

    # Batch consists of num_counter_example rows followed by 1 true action.
    # [B * (n + 1) x act_spec]
    assert list(counter_example_actions.shape) == [batch_size, self._num_counter_examples, self.action_spec]
    # print("check counter example shape",counter_example_actions.shape, expanded_actions.shape)
    combined_true_counter_actions = concat_and_squash_actions(counter_example_actions, expanded_actions)

    return counter_example_actions, combined_true_counter_actions, chain_data
