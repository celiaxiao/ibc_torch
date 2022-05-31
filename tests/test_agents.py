import os
from agents import ibc_agent
from agents.utils import tile_batch
from network import mlp_ebm
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold, datasets
import os

import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# torch.cuda.set_device(2)
device = torch.device('cuda')
def get_sample_distribution():
    mix = D.Categorical(torch.ones(2,))
    comp = D.Normal(torch.tensor([0.0,4.]), torch.tensor([1.0,1.]))
    gmm = D.MixtureSameFamily(mix, comp)
    return gmm
    # TODO: debug
    # return D.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

def get_distribution_sample(num_sample:int):
    gmm = get_sample_distribution()
    # act_space is 1
    return gmm.sample([num_sample, 1]).float() # [num_sample, 1]
    

def get_dataset_sample(num_sample:int):
    X, y = datasets.make_circles(
    n_samples=num_sample, factor=0.5, noise=0.05, random_state=0)
    return X

def train(useGaussian:bool):
    if(useGaussian):
        gmm = get_sample_distribution()
        # get sampling space max and min
        large_sample = gmm.sample([10000])
        act_shape = [1]
        exp_name = 'use_gaussian_langevin2'
        # plt.title('Demo Gaussian Mixture Model')
        
    else:
        large_sample = get_dataset_sample(10000)
        act_shape = [2]
        exp_name = 'use_sklearn'
    path = './agent_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    if useGaussian:
        plt.hist(large_sample.cpu().numpy(), bins=100)
        plt.savefig('./agent_exp/'+exp_name+'/demo.png')
        plt.close()
    sample_max = 100
    sample_min = -100
    print("distribution max min is", sample_max, sample_min)
    batch_size = 256
    num_counter_sample = 256
    
    # default obs
    obs = torch.rand([batch_size,1], dtype=torch.float32)
    network = mlp_ebm.MLPEBM((act_shape[0]+1), 1, normalizer='Batch', dense_layer_type='spectral_norm').to(device)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=1e-5)

    agent = ibc_agent.ImplicitBCAgent(time_step_spec=1, action_spec=1, cloning_network=network,
        optimizer=optim, num_counter_examples=num_counter_sample,
        min_action=large_sample.min(), max_action= large_sample.max(),
        fraction_dfo_samples=0., fraction_langevin_samples=1., return_full_chain=False)
    for i in tqdm.trange(int(2e4)):
        # get data from target distribution
        if (useGaussian):
            data = get_distribution_sample(batch_size)
            # print("check input shape", data.shape)
        else:
            target_data = get_dataset_sample(batch_size)
            data = torch.tensor(target_data)[:,None].float()
        
        loss_dict = agent.train((obs,data))
        if i%1000 == 0:
            print("loss at step",i, loss_dict['loss'].sum().item())
            if useGaussian:
                # check mcmc kl distance
                counter_example_actions, _, _ = agent._make_counter_example_actions(obs, data[:, None], batch_size)
                sampling = counter_example_actions.reshape([-1]).cpu().numpy()
                plt.hist(sampling, bins=100)
                plt.savefig(path+str(i)+'.png')
                plt.close()
                # plt.show()
            else:
                pass


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    train(useGaussian=True)