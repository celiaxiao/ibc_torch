from turtle import shape
from agents.mcmc import langevin_actions_given_obs
from agents.mcmc_tf import langevin_actions_given_obs_tf
from agents.test_mcmc import _get_mock_energy_network, _get_mock_energy_network_tf
from losses.gradient_losses import grad_penalty
import torch
from losses.emb_losses import info_nce
from losses.losses_tf import grad_penalty_tf, info_nce_tf
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import numpy as np
def test_emb_loss():
    batch_size = 8
    num_counter_examples = 2
    softmax_temperature=1.0
    preditions_np = np.random.randn(batch_size, num_counter_examples+1)
    kl_tf = tf.keras.losses.KLDivergence(
          reduction=tf.keras.losses.Reduction.NONE)
    kl_torch = nn.KLDivLoss(reduction='none')

    pred_torch = torch.tensor(preditions_np)
    loss_torch, _ = info_nce(pred_torch, batch_size, num_counter_examples, softmax_temperature, kl_torch)

    pred_tf = tf.constant(preditions_np)
    loss_tf,_ = info_nce_tf(pred_tf, batch_size, num_counter_examples, softmax_temperature, kl_tf)
    # print("tf loss", loss_tf.shape)
    # print(loss_torch, loss_tf)
    # print("loss difference", loss_torch.numpy() - loss_tf.numpy(), np.max(loss_torch.numpy() - loss_tf.numpy()))
    # np.allclose(loss_torch.numpy() - loss_tf.numpy())
    assert np.max(loss_torch.numpy() - loss_tf.numpy()) < 1e-5
    assert (loss_torch.shape== loss_tf.shape)

def test_gradient_loss():
    energy_network_torch = _get_mock_energy_network()
    # Forces network to create variables.
    energy_network_torch(((), torch.randn(1, 2)))

    energy_network_tf = _get_mock_energy_network_tf()
    energy_network_tf(((), np.random.randn(1, 2).astype(np.float32)))

    batch_size = 2
    num_action_samples = 128
    obs = ()
    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                         2).astype(np.float32)
    # test 2 norm
    grad_tf = grad_penalty_tf(energy_network_tf, '2',batch_size, None, obs, init_action_samples, True)
    grad_torch = grad_penalty(energy_network_torch, '2', batch_size, None, obs, torch.tensor(init_action_samples), True)
    assert grad_tf.shape == grad_torch.shape
    # 2 norm can be large sometimes
    assert np.max(grad_torch.numpy() - grad_tf.numpy()) < 100

    # test 1 norm
    grad_tf_1 = grad_penalty_tf(energy_network_tf, '1',batch_size, None, obs, init_action_samples, True)
    grad_torch_1 = grad_penalty(energy_network_torch, '1', batch_size, None, obs, torch.tensor(init_action_samples), True)
    assert grad_tf.shape == grad_torch.shape
    # print(np.max(grad_torch_1.numpy() - grad_tf_1.numpy()))
    assert np.max(grad_torch_1.numpy() - grad_tf_1.numpy()) < 100

    # test inf norm
    grad_tf_inf = grad_penalty_tf(energy_network_tf, 'inf',batch_size, None, obs, init_action_samples, True)
    grad_torch_inf = grad_penalty(energy_network_torch, 'inf', batch_size, None, obs, torch.tensor(init_action_samples), True)
    assert grad_tf.shape == grad_torch.shape
    # print("inf norm", grad_torch_inf, grad_tf_inf)
    # print("inf norm difference", np.max(grad_torch_inf.numpy() - grad_tf_inf.numpy()))
    assert np.max(grad_torch_inf.numpy() - grad_tf_inf.numpy()) < 100


def test_gradient_loss_chain():
    batch_size = 2
    num_action_samples = 128
    energy_network_torch = _get_mock_energy_network()
    # Forces network to create variables.
    energy_network_torch(((), torch.randn(1, 2)))

    energy_network_tf = _get_mock_energy_network_tf()
    energy_network_tf(((), np.random.randn(1, 2).astype(np.float32)))
    obs = ()
    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                            2).astype(np.float32)

    _, chain_data_torch = langevin_actions_given_obs(
        energy_network_torch,
        obs,
        torch.tensor(init_action_samples),
        policy_state=(),
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        training=False,
        num_iterations=25,
        num_action_samples=num_action_samples,
        return_chain=True,
        tfa_step_type=())
    
    _, chain_data_tf = langevin_actions_given_obs_tf(
        energy_network_tf,
        obs,
        init_action_samples,
        policy_state=(),
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        training=False,
        num_iterations=25,
        num_action_samples=num_action_samples,
        return_chain=True,
        tfa_step_type=())
    
    grad_tf = grad_penalty_tf(energy_network_tf, '2',batch_size, chain_data_tf, obs, init_action_samples, True)
    grad_torch = grad_penalty(energy_network_torch, '2', batch_size, chain_data_torch, obs, torch.tensor(init_action_samples), True)

    assert grad_tf.shape == grad_torch.shape
    # print("grad", grad_torch, grad_tf)
    # print("grad difference", np.max(grad_torch.numpy() - grad_tf.numpy()))
    # Note that mock network * 1e2 for the output, so 100 is small enough
    assert np.max(grad_torch.numpy() - grad_tf.numpy()) < 100

if __name__ == "__main__":
    test_emb_loss()
    test_gradient_loss()
    test_gradient_loss_chain()
