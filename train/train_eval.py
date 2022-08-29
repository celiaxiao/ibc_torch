from audioop import maxpp
import os
import sys
from absl import flags
from absl import logging
from agents import ibc_agent, eval_policy
from agents.ibc_policy import IbcPolicy
from agents.utils import save_config, tile_batch, get_sampling_spec
# from eval import eval_env as eval_env_module
from train import make_video as video_module
from train import get_eval_actor as eval_actor_module
from network import mlp_ebm, ptnet_mlp_ebm
from network.layers import pointnet
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch.utils.data import DataLoader, Dataset
from data import policy_eval
from data.transform_dataset import Ibc_dataset
# from data.dataset_d4rl import d4rl_dataset
from data.dataset_maniskill import particle_dataset, state_dataset
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda')

flags.DEFINE_string(
    'train_task',
    None,
    'Which task to run')
# flags.DEFINE_string(
#     'dataset_path', './data',
#     'If set a dataset of the oracle output will be saved '
#     'to the given path.')
flags.DEFINE_integer('obs_dim', 10,
                     'The (total) dimension of the observation')
flags.DEFINE_integer('act_dim', 2,
                     'The dimension of the action.')
flags.DEFINE_integer('eval_episodes', 5,
                     'The dimension of the action.')
flags.DEFINE_string(
    'exp_name', 'experiment',
    'the experiment name')
flags.DEFINE_boolean('eval', False, 'eval only')
flags.DEFINE_integer('eval_epoch', 0, 'which checkpoint to evaluate')
FLAGS = flags.FLAGS
FLAGS(sys.argv) 
writer = SummaryWriter(log_dir='./runs/'+FLAGS.exp_name)
def load_dataset(dataset_dir):
    # os.chdir('..')
    dataset = torch.load(dataset_dir)
    # os.chdir('./tests')
    return dataset

def evaluation_step(eval_episodes, eval_env, eval_actor, name_scope_suffix=''):
    """Evaluates the agent in the environment."""
    logging.info('Evaluating policy.')
  
    # This will eval on seeds:
    # [0, 1, ..., eval_episodes-1]
    for eval_seed in range(eval_episodes):
        eval_env.seed(eval_seed)
        eval_actor.reset()  # With the new seed, the env actually needs reset.
        eval_actor.run()
    # eval_actor.log_metrics()
    # eval_actor.write_metric_summaries()
    return eval_actor.metrics

def eval(exp_name, epoch, image_obs, task, goal_tolerance, obs_dim, act_dim, min_action, max_action):
    """main ibc eval loop
    The checkpoint will be found in tests/policy_exp/${exp_name}

    Args:
        exp_name (string): the name for this experiment
        epoch (int): which checkpoint to eval from
        image_obs (bool): whether using image as part of the observation
        task (string): gym env name
        goal_tolerance (float): tolerance for current position vs the goal
        obs_dim (int): observation space dimension
        act_dim (int): action space dimension
        min_action (float[]): minimal value for action in each dimension
        max_action (float[]): maximum value for action in each dimension
    """
    checkpoint_path = 'tests/policy_exp/'+exp_name+'/'
    path = checkpoint_path + 'eval/'
    if not os.path.exists(path):
        os.makedirs(path)
    num_policy_sample = 512
    lr = 5e-4
    act_shape = [act_dim]
    uniform_boundary_buffer = 0.05
    normalizer=None
    dense_layer_type='spectral_norm'
    rate = 0.
    width = 512
    depth = 8
    use_dfo = False
    use_langevin = True
    optimize_again = True
    inference_langevin_noise_scale = 0.5
    again_stepsize_init = 1e-5
    eval_episodes = FLAGS.eval_episodes
    save_config(locals(), path)   

    max_action = torch.tensor(max_action).float()
    min_action = torch.tensor(min_action).float()
    # action sampling based on min/max action +- buffer.
    min_action, max_action = get_sampling_spec({'minimum':-1*torch.ones(act_dim), 'maximum':torch.ones(act_dim)}, 
        min_action, max_action, uniform_boundary_buffer)
    print('updating boundary', min_action, max_action)
    network = mlp_ebm.MLPEBM((act_shape[0]+obs_dim), 1, width=width, depth=depth,
        normalizer=normalizer, rate=rate,
        dense_layer_type=dense_layer_type).to(device)
    network.load_state_dict(torch.load(checkpoint_path+str(epoch)+".pt"))
    print("loading checkpoint from epoch", epoch)
    ibc_policy = IbcPolicy( actor_network = network,
        action_spec= int(act_shape[0]), #hardcode
        min_action = min_action, 
        max_action = max_action,
        num_action_samples=num_policy_sample,
        use_dfo=use_dfo,
        use_langevin=use_langevin,
        optimize_again=optimize_again,
        inference_langevin_noise_scale=inference_langevin_noise_scale,
        again_stepsize_init=again_stepsize_init
    )

    # To get policy output action, call action = ibc_policy.act({'observation':obs}).squeeze()
    # obs need to be in dim 1 * obs_dim (batch_size=1)

    env_name = eval_env_module.get_env_name(task, False,
                                            image_obs)
    print(('Got env name:', env_name))
    eval_env = eval_env_module.get_eval_env(
        env_name, 1, goal_tolerance, 1)
    env_name_clean = env_name.replace('/', '_')
    
    policy = eval_policy.Oracle(eval_env, policy=ibc_policy, mse=False)
    logging.info('Evaluating', epoch)
    eval_actor, success_metric = eval_actor_module.get_eval_actor(
                            policy,
                            env_name,
                            eval_env,
                            epoch,
                            eval_episodes,
                            path,
                            viz_img=False,
                            summary_dir_suffix=env_name_clean)
    
    metrics = evaluation_step(
        eval_episodes,
        eval_env,
        eval_actor,
        name_scope_suffix=f'_{env_name}')
    # for m in metrics:
    #     writer.add_scalar(m.name, m.result(), epoch)
    logging.info('Done evaluation')
    log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
    logging.info('\n\t\t '.join(log))
    with open(path+str(epoch)+'_eval.txt', 'w') as f:
        f.write(' '.join(map(str, log)))
    print("evaluation at epoch", epoch, "\n", log)
    
def train(exp_name, dataset_dir, image_obs, task, goal_tolerance, obs_dim, act_dim, min_action, max_action):
    """main ibc train loop
    The checkpoint will be saved in tests/policy_exp/${exp_name}

    Args:
        exp_name (string): the name for this experiment
        dataset_dir (string): path to the dataset directory
        image_obs (bool): whether using image as part of the observation
        task (string): gym env name
        goal_tolerance (float): tolerance for current position vs the goal
        obs_dim (int): observation space dimension
        act_dim (int): action space dimension
        min_action (float[]): minimal value for action in each dimension
        max_action (float[]): maximum value for action in each dimension
    """
    path = 'work_dirs/policy_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    batch_size = 512
    num_counter_sample = 7
    num_policy_sample = 512
    lr = 5e-4
    act_shape = [act_dim]
    uniform_boundary_buffer = 0.05
    normalizer=None
    dense_layer_type='spectral_norm'
    rate = 0.
    width = 512
    depth = 8
    use_visual = True

    fraction_langevin_samples = 1.
    fraction_dfo_samples = 0.
    add_grad_penalty = True
    use_dfo = False
    use_langevin = True
    optimize_again = True
    inference_langevin_noise_scale = 0.5
    again_stepsize_init = 1e-5
    num_epoch = int(1e4)
    eval_episodes = FLAGS.eval_episodes
    checkpoint_interval = 50
    eval_interval = 50
    mcmc_iteration = 3
    run_full_chain_under_gradient = False
    save_config(locals(), path)   

    max_action = torch.tensor(max_action).float()
    min_action = torch.tensor(min_action).float()

    # action sampling based on min/max action +- buffer.
    min_action, max_action = get_sampling_spec({'minimum':-1*torch.ones(act_dim), 'maximum':torch.ones(act_dim)}, 
        min_action, max_action, uniform_boundary_buffer)
    print('updating boundary', min_action, max_action)

    # prepare training network
    network_visual = pointnet.pointNetLayer(out_dim=512)
    network = mlp_ebm.MLPEBM((act_shape[0]+obs_dim), 1, width=width, depth=depth,
        normalizer=normalizer, rate=rate,
        dense_layer_type=dense_layer_type).to(device)
    # network = ptnet_mlp_ebm.PTNETMLPEBM(xyz_input_dim=1024, agent_input_dim=25, act_input_dim=8, out_dim=1).to(device)
    # load state dict
    # network.load_state_dict(torch.load('/home/yihe/ibc_torch/work_dirs/policy_exp/hang_10kPairs/50.pt'))
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=lr)

    # get ibc agent
    agent = ibc_agent.ImplicitBCAgent(action_spec=int(act_shape[0]), cloning_network=network,
        optimizer=optim, num_counter_examples=num_counter_sample,
        min_action=min_action, max_action=max_action, add_grad_penalty=add_grad_penalty,
        fraction_dfo_samples=fraction_dfo_samples, fraction_langevin_samples=fraction_langevin_samples, 
        return_full_chain=False, run_full_chain_under_gradient=run_full_chain_under_gradient)

    # load dataset
    dataset = load_dataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
        generator=torch.Generator(device='cuda'), 
        shuffle=True)

    for epoch in tqdm.trange(num_epoch):
        for experience in iter(dataloader):
            # print("from dataloader", experience[0].size(), experience[1].size())
            visual_embed = network_visual(experience[0][:,:1024*3].reshape((-1, 1024, 3)))
            # print(visual_embed.shape)
            experience = (torch.concat([visual_embed, experience[0][:,1024*3:]], -1), experience[1])
            # print("after visual embedding", experience[0].size(), experience[1].size())
            # TODO: process pointcloud here
            loss_dict = agent.train(experience)
            grad_norm, grad_max, weight_norm, weight_max = network_info(network)
            
            # if grad_norm > 100:
                # torch.save(network.state_dict(), path+str(epoch)+'grad_exp'+'.pt')
                # from IPython import embed
                # embed()
            #     exit(0)
        
            writer.add_scalar('loss/step',loss_dict['loss'].mean().item(), agent.train_step_counter)
            writer.add_scalar('info/grad_norm',grad_norm, agent.train_step_counter)
            writer.add_scalar('info/grad_max',grad_max, agent.train_step_counter)
            writer.add_scalar('info/weight_norm',weight_norm, agent.train_step_counter)
            writer.add_scalar('info/weight_max',weight_max, agent.train_step_counter)
        print(agent.train_step_counter)

        if epoch % eval_interval == 0 :
            print("loss at epoch",epoch, loss_dict['loss'].mean().item())
            
        # if epoch % checkpoint_interval == 0 and epoch != 0:
        if epoch % checkpoint_interval == 0:
            torch.save(network.state_dict(), path+'mlp_'+str(epoch)+'.pt')
            torch.save(network_visual.state_dict(), path+'pointnet_'+str(epoch)+'.pt')
    writer.close()

def train_mse(exp_name, dataset_dir, image_obs, task, goal_tolerance, obs_dim):
    path = 'tests/mse_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    batch_size = 8192
    lr = 1e-4
    act_shape = [2]
    num_epoch = int(5e3)
    eval_episodes = 5
   
    save_config(locals(), path)   
    network = nn.Sequential(
        nn.Linear(obs_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, act_shape[0])
    )
    # network = mlp_ebm.MLPEBM(input_dim=4, out_dim=act_shape[0], 
    #     depth=0, width=64)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=lr)
    env_name = eval_env_module.get_env_name(task, False,
                                            image_obs)
    print(('Got env name:', env_name))
    eval_env = eval_env_module.get_eval_env(
        env_name, 1, goal_tolerance, 1)
    dataset = load_dataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for epoch in tqdm.trange(num_epoch):
        for experience in iter(dataloader):
            obs, act_truth = experience
            if isinstance(obs, dict):
                exp_batch_size = obs[list(obs.keys())[0]].shape[0]
                obs = torch.concat([torch.flatten(obs[key]) for key in obs.keys()], axis=-1)
            else:
                exp_batch_size = obs.shape[0]
            # Flatten obs across time: [B x T * obs_spec]
            obs = torch.reshape(obs, [exp_batch_size, -1])
            # print('----------------------', obs.mean(), obs.std(), obs[:3], act_truth.mean(), act_truth.std(), act_truth[:3])
            act_predict = network(obs)
            # print('act predict', act_predict, 'truth', act_truth)
            optim.zero_grad()
            loss = nn.MSELoss()
            output = loss(act_predict, act_truth)
            output.backward()
            optim.step()
        if epoch % 50 == 0:
            print("loss at epoch",epoch, output.item())
            # plot single traj
            policy = eval_policy.Oracle(eval_env, policy=network, mse=True)
            video_module.make_video(
                policy,
                eval_env,
                path,
                step=np.array(epoch)) # agent.train_step)
        if epoch % 100 == 0:
            policy_eval.evaluate(eval_episodes, task, False, False, False, 
                static_policy=network, video=False, output_path=path+str(epoch), mse=True)
        if epoch % 500 == 0 and epoch != 0:
            torch.save(network.state_dict(), path+'checkpoints/'+str(epoch)+'.pt')
    


@torch.no_grad()
def network_info(network, ord=2):
    '''
    Helper function to get norm and max of gradient of network.
    Copied from pyrl.
    '''
    grads = [torch.norm(_.grad.detach(), ord) for _ in network.parameters() if _.requires_grad and _.grad is not None]
    grad_norm = torch.norm(torch.stack(grads), ord).item() if len(grads) > 0 else 0.0
    # if grad_norm > 2.5:
    #     from IPython import embed
    #     embed()
    grad_max = torch.max(torch.stack(grads)).item() if len(grads) > 0 else 0.0

    weights = [torch.norm(_.detach(), ord) for _ in network.parameters()]
    weight_norm = torch.norm(torch.stack(weights), ord).item() if len(weights) > 0 else 0.0
    weight_max = torch.max(torch.stack(weights)).item() if len(weights) > 0 else 0.0
    
    return grad_norm, grad_max, weight_norm, weight_max


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # exp_name='multimodal'
    # dataset_dir = 'data/block_push_states_location/multimodal.pt'
    task = FLAGS.train_task
    if task in ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED', 'PUSH_DISCONTINUOUS', 'PUSH_MULTIMODAL']:
        max_action = [0.03, 0.03]
        min_action = [-0.03, -0.03]
        dataset_dir = 'data/block_push_states_location/push_250.pt'
    elif task == 'PARTICLE':
        max_action = [1, 1]
        min_action = [0, 0]
    elif task == 'door-human-v0':
        obs_dim = 39
        act_dim = 28
        dataset_dir = 'data/d4rl/door-human.pt'
        with open('./data/d4rl/door-human_action_stat.pt', 'rb') as f:
            action_stat = np.load(f, allow_pickle=True).item()
            max_action = action_stat['max']
            min_action = action_stat['min']
    elif task == 'hammer-human-v0':
        obs_dim = 46
        act_dim = 26
        dataset_dir = 'data/d4rl/hammer-human.pt'
        with open('./data/d4rl/hammer-human_action_stat.pt', 'rb') as f:
            action_stat = np.load(f, allow_pickle=True).item()
            max_action = action_stat['max']
            min_action = action_stat['min']
    elif task == 'relocate-human-v0':
        obs_dim = 39
        act_dim = 30
        dataset_dir = 'data/d4rl/relocate-human.pt'
        with open('./data/d4rl/relocate-human_action_stat.pt', 'rb') as f:
            action_stat = np.load(f, allow_pickle=True).item()
            max_action = action_stat['max']
            min_action = action_stat['min']
    elif task == 'pen-human-v0':
        obs_dim = 45
        act_dim = 24
        dataset_dir = 'data/d4rl/pen-human.pt'
        with open('./data/d4rl/pen-human_action_stat.pt', 'rb') as f:
            action_stat = np.load(f, allow_pickle=True).item()
            max_action = action_stat['max']
            min_action = action_stat['min']
    elif task == 'Hang-v0':
        obs_dim = 512+25
        act_dim = 8
        dataset_dir = '/home/yihe/ibc_torch/work_dirs/demos/hang_10kPairs.pt'
        max_action = [1.0] * 8
        min_action = [-1.0] * 8
    else:
        raise ValueError("I don't recognize this task to train.")
    image_obs = False
    goal_tolerance = 0.02
    exp_name=FLAGS.exp_name
    # if FLAGS.eval:
    if False:
        eval(exp_name=exp_name, epoch=FLAGS.eval_epoch, image_obs=image_obs,
          task=task, goal_tolerance=goal_tolerance, obs_dim=obs_dim, act_dim=act_dim, 
          min_action=min_action, max_action=max_action)
    else:
        train(exp_name=exp_name, dataset_dir=dataset_dir, image_obs=image_obs,
          task=task, goal_tolerance=goal_tolerance, obs_dim=obs_dim, act_dim=act_dim, 
          min_action=min_action, max_action=max_action)
    # train_mse(exp_name, dataset_dir, image_obs, task, goal_tolerance, obs_dim)