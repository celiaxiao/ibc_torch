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
from sklearn import manifold, datasets
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

    agent = ibc_agent.ImplicitBCAgent( action_spec=1, cloning_network=network,
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

def train_discontinuity(exp_name):
    path = './agent_exp/'+exp_name+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    batch_size = 256
    num_counter_sample = 16
    num_policy_sample = 512
    lr = 1e-3
    act_shape = [1]
    max_action = [1]
    min_action = [0]
    normalizer=None
    dense_layer_type='regular'
    rate = 0.
    fraction_langevin_samples = 0.
    fraction_dfo_samples = 1.
    use_dfo=False
    use_langevin=True
    optimize_again=True
    num_epoch = int(1e4)
    save_config(locals(), path)   

    max_action = torch.tensor(max_action).float()
    min_action = torch.tensor(min_action).float()
    network = mlp_ebm.MLPEBM((act_shape[0]+1), 1, 
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

    obs = torch.rand([500, 1])*2
    act = (obs > 1)
    plt.scatter(obs.squeeze().cpu(), act.squeeze().cpu())
    plt.savefig('./agent_exp/'+exp_name+'/demo.png')
    plt.close()

    for epoch in tqdm.trange(int(1e4)):
        experience = (obs, act)
        loss_dict = agent.train(experience)
        if epoch % 1000 == 0:
            print("loss at epoch",epoch, loss_dict['loss'].sum().item())
            x = torch.rand([500, 1])*2
            y=[]
            for single_x in x:
                single_y = policy.act({'observations':single_x}).detach().squeeze().cpu().numpy()
                y.append(single_y)
            y = np.array(y)
            # print(x.shape, y.shape)
            plt.scatter(x.squeeze().cpu().numpy(), y)
            plt.savefig(path+'epoch'+str(epoch)+'.png')
            plt.close()
        
if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # train(useGaussian=True)
    train_discontinuity(exp_name='discontinuity_langevin_wo_argmax')