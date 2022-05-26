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
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# torch.cuda.set_device(2)
device = torch.device('cuda')
def get_sample_distribution():
    # Construct Gaussian Mixture Model in 1D consisting of 5 equally
    # weighted normal distributions
    mix = D.Categorical(torch.ones(5,))
    comp = D.Normal(torch.randn(5,), torch.rand(5,))
    gmm = D.MixtureSameFamily(mix, comp)
    return gmm
    # TODO: debug
    # return D.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

def get_distribution_sample(num_sample:int):
    gmm = get_sample_distribution()
    return gmm.sample([num_sample, 1])

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
        exp_name = 'use_gaussian'
        # plt.title('Demo Gaussian Mixture Model')
        plt.hist(large_sample.cpu().numpy(), bins=100)
        plt.savefig('./mcmc_exp/'+exp_name+'/demo.png')
        plt.close()
        
    else:
        large_sample = get_dataset_sample(10000)
        act_shape = [2]
        exp_name = 'use_sklearn'
    path = './mcmc_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    sample_max = large_sample.max().item()
    sample_min = large_sample.min().item()
    print("distribution max min is", sample_max, sample_min)
    batch_size = 64
    num_counter_sample = 256
    
    # default obs
    obs = torch.rand([batch_size,1], dtype=torch.float32)
    network = mlp_ebm.MLPEBM((act_shape[0]+1) , 1, dense_layer_type='spectral_norm').to(device)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=1e-3)

    agent = ibc_agent.ImplicitBCAgent(time_step_spec=1, action_spec=1, cloning_network=network,
        optimizer=optim, num_counter_examples=num_counter_sample,
        min_action=large_sample.min(), max_action= large_sample.max())
    data = get_distribution_sample(batch_size)
    loss_dict = agent.train((obs,data))
    print(loss_dict)
    print('current step', agent.train_step_counter)

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    train(useGaussian=True)