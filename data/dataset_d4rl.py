import sys
import numpy as np
import torch
from absl import flags
from data.transform_dataset import d4rl_dataset
from torch.utils.data import Dataset
flags.DEFINE_string(
    'task',
    None,
    'Which task to run')
flags.DEFINE_string(
    'dataset_path', './data/d4rl/',
    'If set a dataset of the oracle output will be saved '
    'to the given path.')
FLAGS = flags.FLAGS
flags.mark_flags_as_required(['task'])
FLAGS(sys.argv)


def save_dataset(dataset, env_name):
    print(dataset['observations'].shape, dataset['actions'].shape) # An N x dim_observation Numpy array of observations

    ibc_dataset = d4rl_dataset(dataset['observations'], dataset['actions'])
    print(ibc_dataset.__len__())
    
    print(ibc_dataset.__getitem__(1))   
    torch.save(ibc_dataset, FLAGS.dataset_path + env_name + '.pt')

def get_action_stat(dataset, env_name):
    action = dataset['actions']
    print(action.min(0), action.max(0))
    action_stat = {}
    action_stat['min'] = action.min(0)
    action_stat['max'] = action.max(0)
    with open(FLAGS.dataset_path + env_name + '_action_stat.pt', 'wb') as f:
        np.save(f, action_stat)
    
    
if __name__ == '__main__':
    import gym
    import d4rl # Import required to register environments
    
    # door-human: 6729 experience pair, obs shape (6729, 39), act shape (6729, 28)
    # hammer-human: 11310 experience pair, obs shape(11310, 46) act shape(11310, 26)
    # relocate human: 9942 experience pair, obs shape(9942, 39) act shape(9942, 30)
    # pen human: 5000 experience pair, obs shape(5000, 45) act shape(5000, 24)

    # Create the environment
    env_name = FLAGS.task
    env = gym.make(env_name)

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    print(dataset['observations'].shape, dataset['actions'].shape) # An N x dim_observation Numpy array of observations
    save_dataset(dataset, env_name)
    get_action_stat(dataset, env_name)
    