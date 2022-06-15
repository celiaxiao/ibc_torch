import os
from agents import ibc_agent
from agents.ibc_policy import IbcPolicy
from agents.utils import save_config, tile_batch
from network import mlp_ebm
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch.utils.data import DataLoader, Dataset
from data import trajectory_dataset, policy_eval

device = torch.device('cuda')

def load_dataset(dataset_dir):
    os.chdir('..')
    dataset = torch.load(dataset_dir)
    os.chdir('./tests')
    return dataset

def train(exp_name, dataset_dir):
    path = './policy_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    batch_size = 256
    num_counter_sample = 8
    num_policy_sample = 512
    lr = 1e-3
    act_shape = [2]
    max_action = [0.5, 0.5]
    min_action = [-0.5,-0.5]
    normalizer=None
    dense_layer_type='regular'
    rate = 0.
    fraction_langevin_samples = 0.
    fraction_dfo_samples = 1.
    use_dfo=True
    use_langevin=False
    optimize_again=False
    eval_iteration = 350
    num_epoch = int(3e4)
    save_config(locals(), path)   

    max_action = torch.tensor(max_action).float()
    min_action = torch.tensor(min_action).float()
    network = mlp_ebm.MLPEBM((act_shape[0]+8), 1, 
        normalizer=normalizer, rate=rate,
        dense_layer_type=dense_layer_type).to(device)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=lr)
    agent = ibc_agent.ImplicitBCAgent(action_spec=int(act_shape[0]), cloning_network=network,
        optimizer=optim, num_counter_examples=num_counter_sample,
        min_action=min_action, max_action=max_action,
        fraction_dfo_samples=fraction_dfo_samples, fraction_langevin_samples=fraction_langevin_samples, return_full_chain=False)
    policy = IbcPolicy( actor_network = network,
        action_spec= int(act_shape[0]), #hardcode
        min_action = min_action, 
        max_action = max_action,
        num_action_samples=num_policy_sample,
        use_dfo=use_dfo,
        use_langevin=use_langevin,
        optimize_again=optimize_again
    )

    dataset = load_dataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for epoch in tqdm.trange(num_epoch):
        for experience in iter(dataloader):
            loss_dict = agent.train(experience)
        
        if epoch % 500 == 0:
            print("loss at epoch",epoch, loss_dict['loss'].sum().item())
            # plot single traj
            obs = {'target': torch.tensor([[10., 10.]]), 'observation': torch.tensor([[0., 0.]])} # start state
            traj = [obs['observation'].squeeze().cpu().numpy()]
            for i in range(eval_iteration):
                time_step = {'observations': obs}
                act = policy.act(time_step)
                next_point = (obs['observation'] + act)
                obs['observation'] = next_point
                traj.append(next_point.squeeze().cpu().numpy())
            traj = np.array(traj)
            # print(traj)
            plt.plot(traj[:,0], traj[:,1])
            plt.scatter(10,10, color='red')
            plt.savefig(path+'epoch'+str(epoch)+'.png')
            plt.close()
        if epoch % 5000 == 0:
            torch.save(network.state_dict(), path+str(epoch)+'.pt')

def train_mse(exp_name, dataset_dir):
    path = './mse_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    batch_size = 256
    lr = 1e-4
    act_shape = [2]
    num_epoch = int(1e4)
   
    eval_iteration = 350
    save_config(locals(), path)   
    network = nn.Sequential(
        nn.Linear(4, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, act_shape[0])
    )
    # network = mlp_ebm.MLPEBM(input_dim=4, out_dim=act_shape[0], 
    #     depth=0, width=64)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=lr)
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
        if epoch % 1000 == 0:
            print("loss at epoch",epoch, output.item())
            # plot single traj
            obs = torch.tensor([[10., 10., 0., 0.]])
            traj = [obs[0, 2:].cpu().numpy()]
            for i in range(eval_iteration):
                act = network(obs)
                obs[:, 2:] += act
                traj.append(obs[0, 2:].detach().cpu().numpy())
                # print(traj)
            traj = np.array(traj)
            # print(traj)
            plt.plot(traj[:,0], traj[:,1])
            plt.scatter([10],[10], color='red')
            plt.savefig(path+'epoch'+str(epoch)+'.png')
            plt.close()
        
        if epoch+1 % 10000 == 0:
            torch.save(network.state_dict(), path+str(epoch)+'.pt')
    

def demo(dataset_dir):
    dataset = load_dataset(dataset_dir)

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
    exp_name='particle'
    dataset_dir = 'data/trajectory/5line_noise.pt'
    train(exp_name, dataset_dir)
    # train_mse(exp_name, dataset_dir)
    # eval_mse(exp_name)