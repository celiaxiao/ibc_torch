import os
from agents import ibc_agent, eval_policy
from agents.ibc_policy import IbcPolicy
from agents.utils import get_sampling_spec, save_config, tile_batch
from eval import eval_env as eval_env_module
from train import make_video as video_module
from network import mlp_ebm
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
device = torch.device('cuda')

def load_dataset(dataset_dir):
    # os.chdir('..')
    dataset = torch.load(dataset_dir)
    # os.chdir('./tests')
    return dataset

def train(exp_name, dataset_dir, image_obs, task, goal_tolerance, obs_dim):
    path = 'tests/policy_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    batch_size = 512
    num_counter_sample = 512
    num_policy_sample = 16384
    lr = 1e-3
    act_shape = [2]
    max_action = [0.03, 0.03]
    min_action = [-0.03, -0.03]
    uniform_boundary_buffer = 0.05
    normalizer=None
    dense_layer_type='regular'
    rate = 0.
    width = 256
    depth = 8
    fraction_langevin_samples = 0.
    fraction_dfo_samples = 0.
    add_grad_penalty=False
    use_dfo=True
    use_langevin=False
    optimize_again=False
    num_epoch = int(1e4)
    eval_episodes = 20
    mcmc_iteration = 3
    save_config(locals(), path)   

    max_action = torch.tensor(max_action).float()
    min_action = torch.tensor(min_action).float()
    # action sampling based on min/max action +- buffer.
    min_action, max_action = get_sampling_spec({'minimum':torch.tensor([-0.1,-0.1]), 'maximum':torch.tensor([0.1,0.1])}, 
        min_action, max_action, uniform_boundary_buffer)
    network = mlp_ebm.MLPEBM((act_shape[0]+obs_dim), 1, width=width, depth=depth,
        normalizer=normalizer, rate=rate,
        dense_layer_type=dense_layer_type).to(device)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=lr)
    agent = ibc_agent.ImplicitBCAgent(action_spec=int(act_shape[0]), cloning_network=network,
        optimizer=optim, num_counter_examples=num_counter_sample,
        min_action=min_action, max_action=max_action, add_grad_penalty=add_grad_penalty,
        fraction_dfo_samples=fraction_dfo_samples, fraction_langevin_samples=fraction_langevin_samples, return_full_chain=False)
    ibc_policy = IbcPolicy( actor_network = network,
        action_spec= int(act_shape[0]), #hardcode
        min_action = min_action, 
        max_action = max_action,
        num_action_samples=num_policy_sample,
        use_dfo=use_dfo,
        use_langevin=use_langevin,
        optimize_again=optimize_again
    )
    env_name = eval_env_module.get_env_name(task, False,
                                            image_obs)
    print(('Got env name:', env_name))
    eval_env = eval_env_module.get_eval_env(
        env_name, 1, goal_tolerance, 1)
    # policy_eval.evaluate(1, 'PUSH', False, False, False, 
    #             static_policy=policy, video=True, output_path=path+str(0))
    dataset = load_dataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
         generator=torch.Generator(device='cuda'),)
    for epoch in tqdm.trange(num_epoch):
        for experience in iter(dataloader):
            loss_dict = agent.train(experience)
        
        if epoch % 50 == 0 and epoch > 0:
            print("loss at epoch",epoch, loss_dict['loss'].sum().item())
            # evaluate
            policy = eval_policy.Oracle(eval_env, policy=ibc_policy, mse=False)
            video_module.make_video(
                policy,
                eval_env,
                path,
                step=np.array(epoch)) # agent.train_step)
        if epoch % 50 == 0 and epoch > 0:
            policy_eval.evaluate(eval_episodes, task, False, False, False, 
                static_policy=ibc_policy, video=False, output_path=path+str(epoch))
        if epoch % 500 == 0 and epoch > 0:
            torch.save(network.state_dict(), path+str(epoch)+'.pt')

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
        num_workers=8, generator=torch.Generator(device='cuda'),)
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
            torch.save(network.state_dict(), path+str(epoch)+'.pt')
    

def eval_mse(exp_name):
    path = './mse_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    act_shape = [2]
   
    eval_iteration = 5000
    network = nn.Sequential(
        nn.Linear(4, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, act_shape[0])
    )

    network.load_state_dict(torch.load(path+'10000.pt'))
    obs = torch.tensor([[10., 10., 0., 0.]])
    traj = [obs[0, 2:].cpu().numpy()]
    for i in range(eval_iteration):
        act = network(obs)
        if torch.norm(act) < 1e-6:
            print(act)
            break
        obs[:, 2:] += act
        traj.append(obs[0, 2:].detach().cpu().numpy())
        # print(traj)
    traj = np.array(traj)
    print(traj.shape)
    plt.plot(traj[:,0], traj[:,1])
    plt.scatter([10],[10], color='red')
    plt.savefig(path+'eval_10000'+'.png')
    plt.close()

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    exp_name='multimodal_speed2'
    dataset_dir = 'data/block_push_states_location/multimodal_150.pt'
    image_obs = False
    task = "PUSH_MULTIMODAL"
    obs_dim = 16
    # task = "PUSH"
    goal_tolerance = 0.02
    # exp_name='multi_push'
    # dataset_dir = 'data/block_push_states_location/multiple_push.pt'
    train(exp_name, dataset_dir, image_obs, task, goal_tolerance, obs_dim)
    # train_mse(exp_name, dataset_dir, image_obs, task, goal_tolerance, obs_dim)
    # eval_mse(exp_name)