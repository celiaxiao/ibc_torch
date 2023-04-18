import os
import sys
from absl import flags

from agents import ibc_agent, mse_agent
from agents.utils import save_config, tile_batch, get_sampling_spec

from network import mlp_ebm, mlp
from network.layers import pointnet, resnet
# import more visual layers here if needed

import torch

from train import utils

import wandb

device = torch.device('cuda')

# General exp info
flags.DEFINE_string('env_name', None, 'Train env name')
flags.DEFINE_string('exp_name', 'experiment', 'the experiment name')
flags.DEFINE_string('control_mode', None, 'Control mode for maniskill envs')
flags.DEFINE_string('obs_mode', None, 'Observation mode for maniskill envs')
flags.DEFINE_string('model_ids', None, 'model ids for the current env, used in OpenCabinet')
flags.DEFINE_boolean('use_extra', False, 'whether using extra information as observations')
flags.DEFINE_string('dataset_dir', None, 'Demo data path')
flags.DEFINE_integer('data_amount', None, 'Number of (obs, act) pair use in training data')
flags.DEFINE_list('extra_info', ['qpos', 'qvel', 'tcp_pose', 'target'], "list of extra information to include")
flags.DEFINE_boolean('normalize', False,
                     'normalize the observation and action')
flags.DEFINE_boolean('noise', False,
                     'add noise to the observation')
# General training config
flags.DEFINE_integer('batch_size', 512, 'Training batch size')
flags.DEFINE_float('lr', 5e-4, 'Initial optimizer learning rate')
flags.DEFINE_integer('total_steps', int(2e6), 'Total training steps')
flags.DEFINE_integer('epoch_checkpoint', 1000, 'Save checkpoint every x epoch')
flags.DEFINE_integer('step_checkpoint', 5000, 
                     'Save checkpoint every x gradient steps')
flags.DEFINE_integer('resume_from_step', None, 
                     'Resume from previous checkpoint')

flags.DEFINE_integer('log_freq', 1000, 
                     'frequency of logging loss info in number of steps')
# Network input dimensions
flags.DEFINE_integer('obs_dim', 10,
                     'The (total) dimension of the observation')
flags.DEFINE_integer('act_dim', 2,
                     'The dimension of the action.')
flags.DEFINE_integer('visual_num_points', None,
                     'Number of points as visual input')
flags.DEFINE_integer('visual_num_channels', 3,
                     '6 if with rgb or 3 only xyz')
flags.DEFINE_integer('visual_output_dim', None,
                     'Dimension for visual network output')

# Action sampling config
flags.DEFINE_string('action_spec_file', None,
                    'use action spec file if act lim not same across dims')
flags.DEFINE_float('min_action', -1.0,
                   'action lower lim across all dims')
flags.DEFINE_float('max_action', 1.0,
                   'action upper lim across all dims')
flags.DEFINE_float('uniform_boundary_buffer', 0.05, '')

# Visual network configs
flags.DEFINE_string('visual_type', None, 'Visual network type')
flags.DEFINE_boolean('visual_normalize', False,
                     'Apply layer normalization for visual network')

# Mlp network configs
flags.DEFINE_string('mlp_normalizer', None,
                     'Normalizater for mlp network')
flags.DEFINE_string('dense_layer_type', 'spectral_norm',
                     'Dense layer type for resnet layer in mlp network')
flags.DEFINE_float('rate', 0., 'Dropout rate for resnet layer in mlp network')
flags.DEFINE_integer('width', 512, 'Mlp network width')
flags.DEFINE_integer('depth', 8, 'Mlp network resnet depth')

# Ibc Agent configs
flags.DEFINE_enum('agent_type', default='ibc', enum_values=['ibc', 'mse'], 
                  help='Type of agent to use')
flags.DEFINE_integer('num_counter_sample', 8, 
                     'Number of counter sample in mcmc')
flags.DEFINE_float('fraction_dfo_samples', 0., 
                   'Fraction of dfo generated in negative sampels')
flags.DEFINE_integer('train_dfo_iterations', 3, 
                     'Number of dfo iterations for training')
flags.DEFINE_float('fraction_langevin_samples', 1., 
                   'Fraction of langevin generated in negative sampels')
flags.DEFINE_integer('train_langevin_iterations', 100, 
                     'Number of dfo iterations for evaluation')
flags.DEFINE_boolean('add_grad_penalty', False, 
                     'Add gradient penalty for training')
flags.DEFINE_boolean('run_full_chain_under_gradient', False,
                     'run_full_chain_under_gradient')

FLAGS = flags.FLAGS
FLAGS(sys.argv)
torch.autograd.set_detect_anomaly(True)
    
def train(config):
    """main ibc train loop
    The checkpoint will be saved in tests/formal_exp/${env_name}/${exp_name}

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
    # Create experiment and checkpoint directory
    path = f"work_dirs/formal_exp/{config['env_name']}/{config['exp_name']}/"
    checkpoint_path = path + 'checkpoints/'
    config['checkpoint_path'] = checkpoint_path
    logging_path = path + 'wandb/'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    print(path, checkpoint_path)

    max_action = torch.tensor(config['max_action']).float()
    min_action = torch.tensor(config['min_action']).float()

    # action sampling based on min/max action +- buffer.
    min_action, max_action = get_sampling_spec({'minimum':-1*torch.ones(config['act_dim']), 'maximum':torch.ones(config['act_dim'])}, 
    min_action, max_action, config['uniform_boundary_buffer'])
    config['min_action'] = min_action
    config['max_action'] = max_action
    print('updating boundary', min_action, max_action)

    # prepare training network
    network, network_visual = utils.create_network(config)
    print (network,[param.shape for param in list(network.parameters())] )
    optim = torch.optim.Adam(network.parameters(), lr=config['lr'])

    # prepare agent
    agent = create_agent(config, network, optim)

    # load dataset
    dataloader = utils.load_dataloader(config)

    # prepare visualization
    wandb.init(project='ibc', name=config['exp_name'], group=config['env_name'], dir=logging_path, config=config)

    # main training loop
    epoch = 0
    resume_step = config['resume_from_step'] if config['resume_from_step'] else 0
    while agent.train_step_counter < config['total_steps']:
        for experience in iter(dataloader):
            experience = (experience[0].to(device=device, dtype=torch.float), experience[1])
            # print(experience[0], experience[1], type(experience[0]), type(experience[1]))

            if config['visual_type'] is not None:
                visual_input_dim = config['visual_num_points'] * config['visual_num_channels']
                visual_embed = network_visual(experience[0][:,:visual_input_dim].reshape((-1, config['visual_num_points'], config['visual_num_channels'])))
                # if config['use_extra']:
                visual_embed = torch.concat([visual_embed, experience[0][:,visual_input_dim:]], -1)
                experience = (visual_embed, experience[1])
            loss_dict = agent.train(experience)
            grad_norm, grad_max, weight_norm, weight_max = network_info(network)
            
            if agent.train_step_counter % config['log_freq'] == 0:
                print(agent.train_step_counter)
                print("loss at steps", agent.train_step_counter, loss_dict['loss'].mean().item())
            wandb.log({
                'loss':loss_dict['loss'].mean().item(),
                'grad_norm':grad_norm
            })

            if agent.train_step_counter % config['step_checkpoint'] == 0:
                torch.save(network.state_dict(), checkpoint_path+'step_'+str(agent.train_step_counter+resume_step)+'_mlp.pt')
                if config['visual_type'] == 'pointnet':
                    torch.save(network_visual.state_dict(), checkpoint_path+'step_'+str(agent.train_step_counter+resume_step)+'_pointnet.pt')
        
        epoch += 1
        
        # if epoch % config['epoch_checkpoint'] == 0:
        #     torch.save(network.state_dict(), checkpoint_path+'epoch_'+str(epoch)+'_mlp.pt')
        #     torch.save(network_visual.state_dict(), checkpoint_path+'epoch_'+str(epoch)+'_pointnet.pt')

def create_agent(config, network, optim):
    if config['agent_type'] == 'ibc':
        agent = ibc_agent.ImplicitBCAgent(
        action_spec=config['act_dim'], 
        cloning_network=network,
        optimizer=optim, 
        num_counter_examples=config['num_counter_sample'],
        min_action=torch.tensor(config['min_action']), 
        max_action=torch.tensor(config['max_action']), 
        add_grad_penalty=config['add_grad_penalty'],
        fraction_dfo_samples=config['fraction_dfo_samples'], fraction_langevin_samples=config['fraction_langevin_samples'], 
        return_full_chain=False, 
        run_full_chain_under_gradient=config['run_full_chain_under_gradient']
        )
    elif config['agent_type'] == 'mse':
        agent = mse_agent.MSEAgent(network=network, optim=optim)
    else:
        print(f"Agent type {config['agent_type']} not supported. Exiting.")
        exit(0)

    return agent

@torch.no_grad()
def network_info(network, ord=2):
    '''
    Helper function to get norm and max of gradient of network.
    Mainly for debugging purpose.
    Copied from pyrl.
    '''
    grads = [torch.norm(_.grad.detach(), ord) for _ in network.parameters() if _.requires_grad and _.grad is not None]
    grad_norm = torch.norm(torch.stack(grads), ord).item() if len(grads) > 0 else 0.0
    grad_max = torch.max(torch.stack(grads)).item() if len(grads) > 0 else 0.0

    weights = [torch.norm(_.detach(), ord) for _ in network.parameters()]
    weight_norm = torch.norm(torch.stack(weights), ord).item() if len(weights) > 0 else 0.0
    weight_max = torch.max(torch.stack(weights)).item() if len(weights) > 0 else 0.0
    
    return grad_norm, grad_max, weight_norm, weight_max


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    config = FLAGS.flag_values_dict()
    print(config)

    # if config['action_spec_file'] is not None:
    #     with open(config['action_spec_file'], 'rb') as f:
    #         action_stat = np.load(f, allow_pickle=True).item()
    #         config['max_action'] = action_stat['max']
    #         config['min_action'] = action_stat['min']

    train(config)