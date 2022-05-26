from agents import mcmc, mcmc_tf
from agents.utils import tile_batch
from losses import emb_losses, gradient_losses
from network import mlp_ebm
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import os

import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# torch.cuda.set_device(2)
device = torch.device('cuda')
def get_sample_distribution():
    # Construct Gaussian Mixture Model in 1D consisting of 5 equally
    # weighted normal distributions
    # mix = D.Categorical(torch.ones(5,))
    # comp = D.Normal(torch.tensor([0.0,1.,2.,3.,4.]), torch.tensor([1.0,0.5,0.3,0.5,0.2,]))
    mix = D.Categorical(torch.ones(2,))
    comp = D.Normal(torch.tensor([0.0,4.]), torch.tensor([1.0,1.]))
    gmm = D.MixtureSameFamily(mix, comp)
    return gmm
    # TODO: debug
    # return D.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

def get_distribution_sample(num_sample:int):
    gmm = get_sample_distribution()
    # act_space is 1
    return gmm.sample([num_sample, 1])[:,None].float() # [num_sample, 1, 1]

def get_dataset_sample(num_sample:int):
    X, y = datasets.make_circles(
    n_samples=num_sample, factor=0.5, noise=0.05, random_state=0)
    return X

def make_counter_example_actions(
      observations,  # B x obs_spec
      expanded_actions,  # B x 1 x act_spec
      batch_size,
      cloning_network,
      num_counter_examples, act_shape, 
      act_min=-100, act_max=100,
      use_tf:bool = False):
    """Given observations and true actions, create counter example actions."""
    # Note that T (time dimension) would be included in obs_spec.
    # TODO: obtain action max and min
    high, low = act_max, act_min
    maybe_tiled_obs_n = tile_batch(observations,
                                                  num_counter_examples)
    # Counter example actions [B x num_counter_examples x act_spec]
    # TODO: use gaussian instead of uniform distribution
    # random_uniform_example_actions = \
    #     torch.distributions.uniform.Uniform(low,high).sample(\
    #         [batch_size, num_counter_examples]+act_shape)
    random_uniform_example_actions = expanded_actions + torch.normal(0, torch.ones([batch_size, num_counter_examples]+act_shape))
    random_uniform_example_actions = torch.reshape(
          random_uniform_example_actions,
          (batch_size * num_counter_examples, -1))
    # Use all uniform actions to seed the optimization
    if use_tf:
        _, counter_example_actions, _ = mcmc_tf.iterative_dfo(
            cloning_network,
            batch_size,
            maybe_tiled_obs_n,
            random_uniform_example_actions,
            policy_state=(),
            num_action_samples=num_counter_examples,
            min_actions=low * np.ones(act_shape),
            max_actions=high * np.ones(act_shape)
        )
        counter_example_actions = torch.tensor(counter_example_actions.numpy())
    else:
        _, counter_example_actions, _ = mcmc.iterative_dfo(
            cloning_network,
            batch_size,
            maybe_tiled_obs_n,
            random_uniform_example_actions,
            policy_state=(),
            num_action_samples=num_counter_examples,
            min_actions=low * torch.ones(act_shape),
            max_actions=high * torch.ones(act_shape))
    
    def concat_and_squash_actions(counter_example, action):
      return torch.reshape(
          torch.concat([counter_example, action], axis=1),
          [-1] + (act_shape))
    counter_example_actions = counter_example_actions.reshape([batch_size, num_counter_examples, -1])
    # print("check mcmc output shape",counter_example_actions.shape, expanded_actions.shape)
    # Batch consists of num_counter_example rows followed by 1 true action.
    # [B * (n + 1) x act_spec]
    combined_true_counter_actions = \
        concat_and_squash_actions(counter_example_actions, expanded_actions)
    return counter_example_actions, combined_true_counter_actions


def train(useGaussian:bool, use_tf:bool = False):
    if(useGaussian):
        gmm = get_sample_distribution()
        # get sampling space max and min
        large_sample = gmm.sample([10000])
        act_shape = [1]
        exp_name = 'use_gaussian_spectralnorm'
        # plt.title('Demo Gaussian Mixture Model')
        
    else:
        large_sample = get_dataset_sample(10000)
        act_shape = [2]
        exp_name = 'use_sklearn'
    path = './mcmc_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    if useGaussian:
        plt.hist(large_sample.cpu().numpy(), bins=100)
        plt.savefig('./mcmc_exp/'+exp_name+'/demo.png')
        plt.close()
    sample_max = 100
    sample_min = -100
    print("distribution max min is", sample_max, sample_min)
    batch_size = 256
    num_counter_sample = 256
    
    # default obs
    obs = torch.rand([batch_size,1], dtype=torch.float32)
    network = mlp_ebm.MLPEBM((act_shape[0]+1) , 1, normalizer='Batch', dense_layer_type="spectral_norm").to(device)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=1e-5)
    maybe_tiled_obs = tile_batch(obs, num_counter_sample + 1)

    for i in tqdm.trange(int(2e4)):
        # get data from target distribution
        if (useGaussian):
            data = get_distribution_sample(batch_size)
        else:
            target_data = get_dataset_sample(batch_size)
            data = torch.tensor(target_data)[:,None].float()

        # [B * n , act_spec], [B * (n + 1) , act_spec]
        
        counter_example_actions, combined_true_counter_actions = \
                make_counter_example_actions(obs, data,batch_size,network,\
                    num_counter_sample, act_shape, act_min=sample_min, 
                    act_max=sample_max, use_tf=use_tf)
        network_inputs = (maybe_tiled_obs,
                        combined_true_counter_actions.detach())
        # [B * n+1]
        predictions = network(network_inputs)
        # print("check counter example shape",counter_example_actions.shape, combined_true_counter_actions.shape)
        predictions = torch.reshape(predictions,
                                    [batch_size, num_counter_sample + 1])
        # print('---------------------------------------------------------')
        # print("checking output and input", predictions, combined_true_counter_actions)
        optim.zero_grad()
        loss, _ = emb_losses.info_nce(predictions,batch_size,num_counter_sample)
        grad_loss = gradient_losses.grad_penalty(
            network,
            'inf',
            batch_size,
            None,
            maybe_tiled_obs,
            combined_true_counter_actions,
            True,
        )
        # loss += grad_loss
        loss.sum().backward()
        optim.step()
        # print("loss at step",i, loss.sum().item())
        if i%1000 == 0:
            print("loss at step",i, loss.sum().item())
            if useGaussian:
                # check mcmc kl distance
                sampling = counter_example_actions.reshape([-1]).cpu().numpy()
                plt.hist(sampling, bins=100)
                plt.savefig(path+str(i)+'.png')
                plt.close()
                # plt.show()
            else:
                (fig, subplots) = plt.subplots(1,1, figsize=(8,8))
                ax1 = subplots
                # ax2 = subplots[1]
                sampling = counter_example_actions.cpu().numpy()
                ax1.scatter(sampling[:, 0], sampling[:, 1], c="r")
                # ax2.scatter(target_data[:,0], target_data[:,1])
                plt.axis("tight")
                plt.savefig(path+str(i)+'.png')
                plt.close()
                # plt.show()
    

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    train(useGaussian=True, use_tf=False)
