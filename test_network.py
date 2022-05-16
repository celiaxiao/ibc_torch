from environments.block_pushing import block_pushing
import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.environments import suite_gym
import torch
from network import pixel_ebm
NUM_SAMPLES = 500
TIME_DIM = 2

def generate_dataset( images):
    image_size = None
    if images:
      image_size = (64, 64)

    env = block_pushing.BlockPush(
        task=block_pushing.BlockTaskVariant.PUSH, image_size=image_size)
    env = suite_gym.wrap_env(env)
    _observation_spec = env.observation_spec()
    _action_spec = env.action_spec()

    rng = np.random.RandomState(42)
    sample_obs = array_spec.sample_spec_nest(
        _observation_spec, rng, outer_dims=(NUM_SAMPLES, TIME_DIM))
    sample_act = array_spec.sample_spec_nest(
        _action_spec, rng, outer_dims=(NUM_SAMPLES,))
    data = (sample_obs, sample_act)
    dataset = tf.data.Dataset.from_tensors(
        tf.nest.map_structure(tf.convert_to_tensor, data))
    return dataset

def generate_env(images):
    image_size = None
    if images:
      image_size = (64, 64)

    env = block_pushing.BlockPush(
        task=block_pushing.BlockTaskVariant.PUSH, image_size=image_size)
    # env = suite_gym.wrap_env(env)
    observation_spec = env.observation_space.spaces
    action_spec = env.action_space
    print("obs", observation_spec.keys(), "Action", action_spec)
    return env, observation_spec, action_spec

def flatten_observation(obs, action):
#   print(obs)
  flat_obs = tf.nest.flatten(obs)
  flat_obs = tf.concat(flat_obs, axis=-1)
  return flat_obs, action

def get_data():
    dataset = generate_dataset(images=True)
    # dataset = dataset.map(flatten_observation)

    data = list(iter(dataset))[0]
    obs, actions = data
    # print("obs", obs.shape)
    # rgb = obs['rgb']
    # obs['rgb'] = None
    # if isinstance(obs, dict):
    #     observations = torch.concat([torch.flatten(torch.tensor(obs[key])) for key in obs.keys()], axis=-1)
    # print(observations.shape, rgb.shape)
    # for key in obs.keys():
    #     print("key",key, obs[key].shape,  obs[key].dtype)
    # print("action", actions.shape)
    return data

if __name__ == "__main__":
    # dataset = generate_dataset(images=True)
    # print(dataset)
    env, observation_spec, action_spec = generate_env(images=True)
    print(observation_spec['rgb'].shape)
    data = get_data()        
    obs, actions = data
    if isinstance(obs, dict):
        for key in obs.keys():
            obs[key] = torch.tensor(obs[key].numpy())
    # old rgb [N, hist, W, H, C]
    obs['rgb'] = obs['rgb'].permute(0,4, 2, 3, 1)
    actions = torch.tensor(actions.numpy())
    model = pixel_ebm.PixelEBM(obs, action_spec.shape[0], "MaxConv", 'DenseResnetValue', N=NUM_SAMPLES)
    print(model)
    out = model((obs, actions))
    print(out)

