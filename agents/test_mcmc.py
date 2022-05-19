from environments.particle.particle import ParticleEnv
import gym
from network import mlp_ebm
import numpy as np
import torch
import torch.nn as nn
from agents.mcmc import *
from agents.mcmc_tf import compute_grad_norm_tf, gradient_wrt_act_tf
import tensorflow as tf
import os

def test_compute_grad_norm():
    print('------------test_compute_grad_norm-----------------')

    batch_size = 8
    n = 3
    de_dact = np.random.randn(batch_size,n+1)
    d_torch = torch.tensor(de_dact)
    d_tf = tf.constant(de_dact)
    grad_norm_torch = compute_grad_norm(None, d_torch)
    grad_norm_tf = compute_grad_norm_tf(None, d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (grad_norm_tf.numpy() == grad_norm_torch.numpy()).all()

    grad_norm_torch = compute_grad_norm('inf', d_torch)
    grad_norm_tf = compute_grad_norm_tf('inf', d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (np.round(grad_norm_tf.numpy(),4) == np.round(grad_norm_torch.numpy(),4)).all()

    grad_norm_torch = compute_grad_norm('1', d_torch)
    grad_norm_tf = compute_grad_norm_tf('1', d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (np.round(grad_norm_tf.numpy(),4) == np.round(grad_norm_torch.numpy(),4)).all()

    grad_norm_torch = compute_grad_norm('2', d_torch)
    grad_norm_tf = compute_grad_norm_tf('2', d_tf)
    # print(grad_norm_tf, grad_norm_torch)
    assert (np.round(grad_norm_tf.numpy(),4) == np.round(grad_norm_torch.numpy(),4)).all()
    print('pass')
    

def _test_batched_categorical_bincount_shapes( batch_size):
    num_samples = 256
    probs = np.random.rand(batch_size, num_samples)
    count = 128
    for i in range(batch_size):
      probs[i] = probs[i] / probs[i].sum() 
    indices_counts = categorical_bincount(count, torch.tensor(probs+ 1e-6).log(),
                                               num_samples)
    assert indices_counts.shape[0] == batch_size
    assert indices_counts.shape[1] == num_samples

def test_batched_categorical_bincount_shapes():
    print('------------test_batched_categorical_bincount_shapes-----------------')

    for batch_size in [1, 2]:
        _test_batched_categorical_bincount_shapes(batch_size)
    print('pass')

def test_batched_categorical_bincount_correct():
    print('------------test_batched_categorical_bincount_correct-----------------')

    eps = 1e-6
    probs = np.array([[eps, eps, 1.-eps],
                        [eps, 1.-eps, eps]])
    count = 5
    indices_counts = categorical_bincount(count, torch.tensor(probs).log(),
                                                probs.shape[1]).numpy()
    # Assert sampled "count" times:
    for i in range(2):
        assert indices_counts[i].sum() == count
    # Assert the most-counted probs are correct.
    assert np.argmax(indices_counts[0]) == 2
    assert np.argmax(indices_counts[1]) == 1
    print('pass')

def _get_network_and_time_step():
    env = gym.make("InvertedPendulum-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # print("obs space", env.observation_space, "action space",env.action_space)
    energy_network = mlp_ebm.MLPEBM((obs_dim + act_dim) \
         , 1)
    return energy_network, env.reset()

def _get_network_and_env_with_dict():
    env = ParticleEnv()
    # env = gym.make("CartPole-v1")
    # env = suite_gym.wrap_env(env)
    # env.observation_space is a dict object
    obs_dim = env.observation_space[list(env.observation_space.spaces)[0]].shape[0]
    act_dim = env.action_space.shape[0]
    num_dict = 1
    if isinstance(env.observation_space.spaces, dict):
        num_dict = len(env.observation_space.spaces.keys())
    print("obs space", obs_dim, "action space",act_dim)
    print("predict input shape", (obs_dim*num_dict+act_dim))
    energy_network = mlp_ebm.MLPEBM((obs_dim*num_dict + act_dim) \
         , 1)
    return energy_network, env.reset()

def _get_mock_energy_network():
    class EnergyNet(nn.Module):

        def __init__(self, energy_scale=1e2):
            super(EnergyNet, self).__init__()
            self.mean = torch.tensor([0.3, 0.4])
            self.energy_scale = energy_scale

        def forward(self, x):
            """Mock network."""
            _, actions = x
            # print("mock network", actions, self.mean)
            return -(torch.linalg.norm(actions - self.mean, dim=1)
                    * self.energy_scale)**2

    return EnergyNet()

def _get_mock_energy_network_tf():
    class EnergyNet(tf.keras.Model):

      def __init__(self, energy_scale=1e2):
        super(EnergyNet, self).__init__()
        self.mean = np.array([0.3, 0.4])
        self.energy_scale = energy_scale

      def call(self, x, step_type=(), network_state=(), training=()):
        """Mock network."""
        _, actions = x
        return -(tf.linalg.norm(actions - self.mean, axis=1)
                 * self.energy_scale)**2, ()

    return EnergyNet()

def test_gradient_wrt_act():
    print('------------test_gradient_wrt_act-----------------')

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

    deda_torch, energy_torch = gradient_wrt_act(
        energy_network_torch,
        obs,
        torch.tensor(init_action_samples),
        network_state=(),
        training=False,
        tfa_step_type=(),
        apply_exp = False)
    deda_tf, energy_tf = gradient_wrt_act_tf(
        energy_network_tf,
        obs,
        init_action_samples,
        network_state=(),
        training=False,
        tfa_step_type=(),
        apply_exp = False)
    assert deda_tf.shape == deda_torch.shape
    assert energy_torch.shape == energy_tf.shape
    # print("comparing deda", np.max(deda_tf.numpy() - deda_torch.numpy()))
    # print("comparing output eneger", np.max(energy_tf.numpy() - energy_torch.detach().numpy()))
    assert np.max(deda_tf.numpy() - deda_torch.numpy()) < 0.01
    assert np.max(energy_tf.numpy() - energy_torch.detach().numpy()) < 0.01
    print('pass')

def test_shapes_iterative_dfo():
    print('------------test_shapes_iterative_dfo-----------------')

    batch_size = 2
    num_action_samples = 2048
    energy_network, time_step = _get_network_and_env_with_dict()

    obs = time_step
    # "Batch" the observations by replicating
    for key in obs.keys():
        batch_obs = torch.tensor(obs[key])[None, Ellipsis]
        obs[key] = torch.concat([batch_obs] * (batch_size * num_action_samples),axis=0)

    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                            2).astype(np.float32)
    init_action_samples = torch.tensor(init_action_samples)
    # Forces network to create variables.
    with torch.no_grad():
        energy_network((obs, init_action_samples))

    probs, action_samples, _ = iterative_dfo(
        energy_network,
        batch_size,
        obs,
        init_action_samples,
        policy_state=(),
        temperature=1.0,
        num_action_samples=num_action_samples,
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        num_iterations=3,
        iteration_std=1e-1,
        training=False,
        tfa_step_type=())
    assert action_samples.shape == init_action_samples.shape
    assert probs.shape[0] == batch_size * num_action_samples
    print('pass')

def test_correct_iterative_dfo():
    print('------------test_correct_iterative_dfo-----------------')
    
    energy_network = _get_mock_energy_network()
    # Forces network to create variables.
    energy_network(((), torch.randn(1, 2)))

    batch_size = 2
    num_action_samples = 2048
    obs = ()
    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                            2).astype(np.float32)
    init_action_samples = torch.tensor(init_action_samples)
    probs, action_samples, _ = iterative_dfo(
        energy_network,
        batch_size,
        obs,
        init_action_samples,
        policy_state=(),
        temperature=1.0,
        num_action_samples=num_action_samples,
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        num_iterations=10,
        iteration_std=1e-1,
        training=False,
        tfa_step_type=())
    assert tf.linalg.norm(action_samples[np.argmax(probs)] - \
                            energy_network.mean) < 0.01
    print('pass')


def test_correct_langevin():
    print('------------test_correct_langevin-----------------')
    batch_size = 2
    num_action_samples = 128
    energy_network = _get_mock_energy_network()
    # Forces network to create variables.
    energy_network(((),torch.randn(1, 2)))
    obs = ()
    init_action_samples = torch.rand(batch_size * num_action_samples,
                                            2)

    action_samples = langevin_actions_given_obs(
        energy_network,
        obs,
        init_action_samples,
        policy_state=(),
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        training=False,
        num_iterations=25,
        num_action_samples=num_action_samples,
        tfa_step_type=())
    assert torch.linalg.norm(action_samples[0] - \
                            energy_network.mean) < 0.1
    print('pass')
    
# just check no runtime error
def test_langevin_chain():
    print('------------test_langevin_chain-----------------')
    batch_size = 2
    num_action_samples = 128
    energy_network = _get_mock_energy_network()
    # Forces network to create variables.
    energy_network(((),torch.randn(1, 2)))
    obs = ()
    init_action_samples = torch.rand(batch_size * num_action_samples,
                                            2)

    action_samples,chain_data = langevin_actions_given_obs(
        energy_network,
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
    print('pass')
    
    # print("chain", chain_data.grad_norms.shape, chain_data.energies.shape,chain_data.actions.shape)

def test_mlp():
    print('------------test_mlp-----------------')
    batch_size = 2
    num_action_samples = 2048
    energy_network, time_step = _get_network_and_time_step()
    print("timestep", time_step)
    obs = torch.tensor(time_step)[None, Ellipsis]
    obs = torch.concat([obs] * (batch_size * num_action_samples),
                            axis=0).float()
    init_action_samples = np.random.rand(batch_size * num_action_samples,1).astype(dtype=np.float32)
    init_action_samples = torch.tensor(init_action_samples)
    # Forces network to create variables.
    out = energy_network((obs, init_action_samples))
    print(energy_network, energy_network._mlp._weight_layers)
    print("out", out.shape)
    print('pass')


def test_mlp_with_dict_env():
    print('------------test_mlp_with_dict_env-----------------')
    batch_size = 2
    num_action_samples = 2048
    energy_network, obs = _get_network_and_env_with_dict()
    # "Batch" the observations by replicating
    for key in obs.keys():
        batch_obs = torch.tensor(obs[key])[None, Ellipsis]
        obs[key] = torch.concat([batch_obs] * (batch_size * num_action_samples),axis=0)
    # act dim = 2 
    init_action_samples = np.random.rand(batch_size * num_action_samples,2,).astype(dtype=np.float32)
    init_action_samples = torch.tensor(init_action_samples)
    # Forces network to create variables.
    out = energy_network((obs, init_action_samples))
    print(energy_network, energy_network._mlp._weight_layers)
    print("out", out.shape)
    print('pass')

if __name__ == "__main__":
    test_compute_grad_norm()
    test_batched_categorical_bincount_correct()
    test_batched_categorical_bincount_shapes()
    test_mlp()
    test_mlp_with_dict_env()
    test_shapes_iterative_dfo()
    test_correct_iterative_dfo()
    test_correct_langevin()
    test_langevin_chain()
    
