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

"""EBM loss functions."""

import tensorflow as tf


def info_nce_tf(predictions,
             batch_size,
             num_counter_examples,
             softmax_temperature,
             kl):
  """EBM loss: can you classify the correct example?

  Args:
    predictions: [B x n+1] with true in column [:, -1]
    batch_size: B
    num_counter_examples: n
    softmax_temperature: the temperature of the softmax.
    kl: a KL Divergence loss object

  Returns:
    (loss per each element in the batch, and an optional
     dictionary with any loss objects to log)
  """
  softmaxed_predictions = tf.nn.softmax(
      predictions / softmax_temperature, axis=-1)

  # [B x n+1] with 1 in column [:, -1]
  indices = tf.ones(
      (batch_size,), dtype=tf.int32) * num_counter_examples
  labels = tf.one_hot(indices, depth=num_counter_examples + 1)

  per_example_loss = kl(labels, softmaxed_predictions)
  return per_example_loss, dict()

import gin
from agents import mcmc_tf
import tensorflow as tf


@gin.configurable
def grad_penalty(energy_network,
                 grad_norm_type,
                 batch_size,
                 chain_data,
                 observations,
                 combined_true_counter_actions,
                 training,
                 only_apply_final_grad_penalty=True,
                 grad_margin=1.0,
                 square_grad_penalty=True,
                 grad_loss_weight=1.0):
  """Calculate losses based on some norm of dE/dactions from mcmc samples."""
  # Case 1: only add a gradient penalty on the final step.
  if only_apply_final_grad_penalty:
    de_dact, _ = mcmc.gradient_wrt_act(
        energy_network,
        observations,
        tf.stop_gradient(combined_true_counter_actions),
        training,
        network_state=(),
        tfa_step_type=(),
        apply_exp=False)  # TODO(peteflorence): config this.

    # grad norms should now be shape (b*(n+1))
    grad_norms = mcmc.compute_grad_norm(grad_norm_type, de_dact)
    grad_norms = tf.reshape(grad_norms, (batch_size, -1))

  else:
    # Case 2: the full chain was under the gradient tape, or langevin_step
    # stop_chain_grad was set to True. Either way just go add penalties to all
    # the norms.
    assert chain_data.grad_norms is not None
    # grad_norms starts out as: (num_iterations, B*n)
    grad_norms = chain_data.grad_norms
    # now grad_norms is shape: (B*n, num_iterations)
    grad_norms = tf.transpose(grad_norms, perm=[1, 0])
    # now grad_norms is shape: (B, n*num_iterations)
    grad_norms = tf.reshape(grad_norms, (batch_size, -1))

  if grad_margin is not None:
    grad_norms -= grad_margin
    # assume 1e10 is big enough
    grad_norms = tf.clip_by_value(grad_norms, 0., 1e10)

  if square_grad_penalty:
    grad_norms = grad_norms**2

  grad_loss = tf.reduce_mean(grad_norms, axis=1)
  return grad_loss * grad_loss_weight


