from multiprocessing import reduction
import os
import sys
from absl import flags

from agents import ibc_agent, eval_policy
from agents.ibc_policy import IbcPolicy
from network import mlp_ebm, ptnet_mlp_ebm
from network.layers import pointnet
from environments.maniskill.maniskill_env import *

from torch.utils.data import DataLoader, Dataset
from data.dataset_maniskill import *
import torch
import numpy as np
import tqdm
import json
from moviepy.editor import ImageSequenceClip

device = torch.device('cuda')

# General exp info
flags.DEFINE_string('env_name', None, 'Train env name')
flags.DEFINE_string('train_exp_name', None, 'the training experiment name')
flags.DEFINE_string('eval_exp_name', None, 'the evaluation experiment name')
flags.DEFINE_string('control_mode', None, 'Control mode for maniskill envs')
flags.DEFINE_string('obs_mode', None, 'Observation mode for maniskill envs')
flags.DEFINE_string('reward_mode', 'dense', 'If using dense reward')
flags.DEFINE_integer('max_episode_steps', 350, 'Max step allowed in env')
flags.DEFINE_string('dataset_dir', None, 'Demo data path')
flags.DEFINE_integer('data_amount', None, 'Number of (obs, act) pair use in training data')

# General eval info
flags.DEFINE_integer('num_episodes', 1, 'number of new seed episodes')
flags.DEFINE_integer('eval_step', None, 'Checkpoint step to eval')
flags.DEFINE_integer('eval_epoch', None, 'Checkpoint epoch to eval')
flags.DEFINE_boolean('compute_mse', True, 'Compute mse')

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
flags.DEFINE_string('visual_normalizer', None,
                     'Normalizater for visual network')

# Mlp network configs
flags.DEFINE_string('mlp_normalizer', None,
                     'Normalizater for mlp network')
flags.DEFINE_string('dense_layer_type', 'spectral_norm',
                     'Dense layer type for resnet layer in mlp network')
flags.DEFINE_float('rate', 0., 'Dropout rate for resnet layer in mlp network')
flags.DEFINE_integer('width', 512, 'Mlp network width')
flags.DEFINE_integer('depth', 8, 'Mlp network resnet depth')

# Ibc Policy configs
flags.DEFINE_integer('num_policy_sample', 512, 
                     'Number of policy samples')
flags.DEFINE_boolean('use_dfo', False, 
                   'Use dfo for negative sampels')
flags.DEFINE_integer('eval_dfo_iterations', 3, 
                     'Number of dfo iterations for training')
flags.DEFINE_boolean('use_langevin', True, 
                   'Use langevin for negative sampels')
flags.DEFINE_integer('eval_langevin_iterations', 100, 
                     'Number of langevin iterations for evaluation')
flags.DEFINE_boolean('optimize_again', True, "optimize again")
flags.DEFINE_float('inference_langevin_noise_scale', 0.5, '')
flags.DEFINE_float('again_stepsize_init', float(1e-05), '')



FLAGS = flags.FLAGS
FLAGS(sys.argv)


class Evaluation:
    def __init__(self, config):
        self.config = config

        # Create evaluation env
        self.env = self.create_env()

        # Create network and load from checkpoint
        self.network_visual, self.network = self.create_and_load_network()

        # Create ibc policy to evaluate
        self.ibc_policy = self.create_policy()

        # Create evaluation config -- TODO: read from eval_cfg
        self.save_video = True
        self.episode_id = 0
        self.eval_info = {}
        self.eval_info_path = f"work_dirs/formal_exp/{self.config['env_name']}/{self.config['train_exp_name']}/eval/{self.config['eval_step']}_{self.config['eval_exp_name']}/"
        if not os.path.exists(self.eval_info_path):
            os.makedirs(self.eval_info_path)

    def create_env(self):
        # manually select env to create
        if self.config['env_name'] == 'Hang-v0':
            if self.config['obs_mode'] == 'particles':
                fn = HangEnvParticle
            elif self.config['obs_mode'] == 'pointcloud':
                pass 
        
        elif self.config['env_name'] == 'Fill-v0':
            if self.config['obs_mode'] == 'particles':
                fn = FillEnvParticle
            elif self.config['obs_mode'] == 'pointcloud':
                pass 

        elif self.config['env_name'] == 'Excavate-v0':
            if self.config['obs_mode'] == 'particles':
                fn = ExcavateEnvParticle
            elif self.config['obs_mode'] == 'pointcloud':
                pass 
        
        else:
            print(f"Env {self.config['env_name']} obs mode {self.config['obs_mode']} not supported! Exiting")
            exit(0)
        
        return fn(obs_mode=self.config['obs_mode'], control_mode=self.config['control_mode'],reward_mode=self.config['reward_mode'])

    def create_and_load_network(self):
        config = self.config

        # create network
        network_visual=None
        if config['visual_type'] == 'pointnet':
            network_visual = pointnet.pointNetLayer(in_channel=config['visual_num_channels'], out_dim=config['visual_output_dim'])

            visual_input_dim = config['visual_num_points'] * config['visual_num_channels']

            network = mlp_ebm.MLPEBM(
            (config['visual_output_dim'] + config['obs_dim'] - visual_input_dim + config['act_dim']), 1, 
            width=config['width'], depth=config['depth'],
            normalizer=config['mlp_normalizer'], rate=config['rate'],
            dense_layer_type=config['dense_layer_type']).to(device)
    
        else:
            network = mlp_ebm.MLPEBM(
            (config['obs_dim'] + config['act_dim']), 1, 
            width=config['width'], depth=config['depth'],
            normalizer=config['mlp_normalizer'], rate=config['rate'],
            dense_layer_type=config['dense_layer_type']).to(device)

        checkpoint_path = f"work_dirs/formal_exp/{config['env_name']}/{config['train_exp_name']}/checkpoints/"

        if config['eval_step'] is not None:
            if network_visual is not None:
                # TODO: remove hardcoded visual type
                network_visual.load_state_dict(torch.load(
                    f"{checkpoint_path}step_{config['eval_step']}_pointnet.pt"
                ))
            network.load_state_dict(torch.load(
                f"{checkpoint_path}step_{config['eval_step']}_mlp.pt"
            ))
        elif config['eval_epoch'] is not None:
            if network_visual is not None:
                # TODO: remove hardcoded visual type
                network_visual.load_state_dict(torch.load(
                    f"{checkpoint_path}epoch_{config['eval_epoch']}_pointnet.pt"
                ))
            network.load_state_dict(torch.load(
                f"{checkpoint_path}epoch_{config['eval_epoch']}_mlp.pt"
            ))
        else:
            print("checkpoint to evaluate is not specified. Exiting.")
            exit(0)
        
        return network_visual, network

    def create_policy(self):
        ibc_policy = IbcPolicy( 
            actor_network = self.network,
            action_spec= self.config['act_dim'], #hardcode
            min_action=torch.tensor([self.config['min_action']] * self.config['act_dim']).float(),
            max_action=torch.tensor([self.config['max_action']] * self.config['act_dim']).float(),
            num_action_samples=self.config['num_policy_sample'],
            use_dfo=self.config['use_dfo'],
            dfo_iterations=self.config['eval_dfo_iterations'],
            use_langevin=self.config['use_langevin'],
            langevin_iterations=self.config['eval_langevin_iterations'],
            optimize_again=self.config['optimize_again'],
            inference_langevin_noise_scale=self.config['inference_langevin_noise_scale'],
            again_stepsize_init=self.config['again_stepsize_init']
        )

        return ibc_policy

    def run_eval(self):
        '''
        Main eval loop. 
        '''
        if not os.path.exists(self.eval_info_path+'videos/'):
            os.makedirs(self.eval_info_path+'videos/')

        for idx in range(self.config['num_episodes']):
            self.episode_id = idx
            seed = np.random.randint(low=5000, high=6000)
            total_reward, success, num_steps = self.run_single_episode(video_path=f"{self.eval_info_path}videos/{idx}",seed=seed)
            print(f'eval_traj_{idx}:', total_reward, success, num_steps)
            self.eval_info[f'eval_traj_{idx}'] = {
                'seed':seed, 'total_reward':total_reward,
                'success':success, 'num_steps':num_steps
            }

        with open(f"{self.eval_info_path}traj_info.json", 'w') as f:
            json.dump(self.eval_info, f, indent=4)

        # np.save(eval_info_path+'eval_info.npy', self.eval_info)

    
    def run_single_episode(self, video_path, seed=2):
        '''
        Run evaluation for a single episode on a given seed.
        '''
        self.env.reset(seed=seed)
        total_reward = 0
        done = False
        success = False
        imgs = [self.env.render("rgb_array")]
        
        # for num_steps in range(len(self.env._max_episode_steps)):
        for num_steps in range(self.config['max_episode_steps']):
            if done:
                success = True
                break

            # print("at step ", num_steps)

            # get current observation -- preprocessing handled by env wrapper
            obs = self.env.get_obs()
            obs = torch.tensor(obs).to(device, dtype=torch.float).expand(1, -1)
            
            if self.network_visual is not None:
                visual_input_dim = self.config['visual_num_points'] * self.config['visual_num_channels']
                visual_embed = self.network_visual(obs[:,:visual_input_dim].reshape((-1, self.config['visual_num_points'], config['visual_num_channels'])))
                obs = torch.concat([visual_embed, obs[:,visual_input_dim:]], -1)

            # get predicted action from policy
            act = self.ibc_policy.act({'observations':obs}).squeeze()
            # print('act shape', act.shape)
            # print("prediceted action", act)

            # step and get rew, done
            _, rew, done, _ = self.env.step(act.detach().cpu().numpy())
            # print("Current step reward:", rew)
            # print("Traj is done:", bool(done))

            # save info and update steps
            total_reward += rew
            imgs.append(self.env.render("rgb_array"))
        
        self.animate(imgs, path=f"{video_path}_seed={seed}_success={success}.mp4")
        
        return total_reward, success, num_steps

    def compute_mse(self):
        '''
        compute the average mse error between demo and policy prediction.
        '''

        # Load dataset and split into training and validation
        dataset = torch.load(self.config['dataset_dir'])
        if self.config['data_amount']:
            assert self.config['data_amount'] <= len(dataset), f"Not enough data for {self.config['data_amount']} pairs!"
            train_dataset = torch.utils.data.Subset(dataset, range(self.config['data_amount']))
            validate_dataset = torch.utils.data.Subset(dataset, range(self.config['data_amount'], len(dataset)))
        else:
            train_dataset = dataset
            validate_dataset = None

        train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(range(len(train_dataset)), size=200))
        if validate_dataset and len(validate_dataset) > 200:
            validate_dataset = torch.utils.data.Subset(validate_dataset, np.random.choice(range(len(validate_dataset)), size=200))
        # print(len(dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=10, 
            generator=torch.Generator(device='cuda'), 
            shuffle=True)
        if validate_dataset:
            validate_dataloader = DataLoader(validate_dataset, batch_size=10, 
            generator=torch.Generator(device='cuda'), 
            shuffle=True)  


        # Compute mse loss on training dataset
        loss_fn = torch.nn.MSELoss(reduction='mean')
        train_mse_loss = []

        for obs, act_gt in iter(train_dataloader):
            obs = obs.to(device=device, dtype=torch.float)
            if self.network_visual is not None:
                visual_input_dim = self.config['visual_num_points'] * self.config['visual_num_channels']
                visual_embed = self.network_visual(obs[:,:visual_input_dim].reshape((-1, self.config['visual_num_points'], config['visual_num_channels'])))
                obs = torch.concat([visual_embed, obs[:,visual_input_dim:]], -1)

            act_pred = self.ibc_policy.act({'observations':obs})

            train_mse_loss.append(loss_fn(act_gt, act_pred).item())


        # Compute mse loss on validation dataset
        if validate_dataset:
            validate_mse_loss = []

            for obs, act_gt in iter(validate_dataloader):
                obs = obs.to(device=device, dtype=torch.float)
                if self.network_visual is not None:
                    visual_input_dim = self.config['visual_num_points'] * self.config['visual_num_channels']
                    visual_embed = self.network_visual(obs[:,:visual_input_dim].reshape((-1, self.config['visual_num_points'], config['visual_num_channels'])))
                    obs = torch.concat([visual_embed, obs[:,visual_input_dim:]], -1)

                act_pred = self.ibc_policy.act({'observations':obs})

                # print(loss_fn(act_gt, act_pred).item())
                validate_mse_loss.append(loss_fn(act_gt, act_pred).item())
        
        with open(f"{self.eval_info_path}mse_info.json", 'w') as f:
            if validate_dataset:
                json.dump({'training_data_mse': np.mean(train_mse_loss), 'validation_data_mse': np.mean(validate_mse_loss)}, f, indent=4)
            else:
                json.dump({'training_data_mse': np.mean(train_mse_loss), 'validation_data_mse': None}, f, indent=4)


    def animate(self, imgs, fps=20, path="animate.mp4"):
        imgs = ImageSequenceClip(imgs, fps=fps)
        imgs.write_videofile(path, fps=fps)


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    config = FLAGS.flag_values_dict()
    print(config)

    eval = Evaluation(config)

    if FLAGS.num_episodes > 0:
        eval.run_eval()
    if FLAGS.compute_mse:
        eval.compute_mse()