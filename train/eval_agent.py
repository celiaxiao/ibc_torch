import os
from multiprocessing import reduction

import sys
from absl import flags

from agents import ibc_agent, eval_policy
from agents.ibc_policy import IbcPolicy
from agents.mse_policy import MsePolicy
from agents.utils import save_json
from network import mlp_ebm, ptnet_mlp_ebm
from network.layers import pointnet
from environments.maniskill.maniskill_env import *
from train import make_video as video_module
from train import get_eval_actor as eval_actor_module
from train import utils
from eval import eval_env as eval_env_module
from torch.utils.data import DataLoader, Dataset
from data.dataset_maniskill import *
import torch
import numpy as np
import tqdm
import json

device = torch.device('cuda')

# General exp info
flags.DEFINE_string('env_name', None, 'Train env name')
flags.DEFINE_string('train_exp_name', None, 'the training experiment name')
flags.DEFINE_string('eval_exp_name', None, 'the evaluation experiment name')
flags.DEFINE_string('control_mode', None, 'Control mode for maniskill envs')
flags.DEFINE_string('obs_mode', None, 'Observation mode for maniskill envs')
flags.DEFINE_string('reward_mode', 'dense', 'If using dense reward')
flags.DEFINE_integer('max_episode_steps', 350, 'Max step allowed in env')
flags.DEFINE_boolean('use_extra', False, 'whether using extra information as observations')
flags.DEFINE_integer('num_frames', None, 'number of frames to stack. If > 1, will use a frame_stack wrapper')
flags.DEFINE_string('dataset_dir', None, 'Demo data path')
flags.DEFINE_integer('data_amount', None, 'Number of (obs, act) pair use in training data')
flags.DEFINE_float('single_step_max_reward', 0, 'Max reward possible in each env.step()')
flags.DEFINE_string('eval_seeds_file', None, 'Json file that contains evaluation seeds')
flags.DEFINE_integer('main_test_seed', 0, 'seed for episode seed generator if eval seeds file not provided')
flags.DEFINE_boolean('image_obs', False, 'using image observation')
flags.DEFINE_float('goal_tolerance', 0.02, 'tolerance for current position vs the goal')
flags.DEFINE_boolean('viz_img', False, 'visualize image in eval')
flags.DEFINE_boolean('save_video', True, 'visualize videos in eval')
flags.DEFINE_enum('agent_type', default='ibc', enum_values=['ibc', 'mse'], 
                  help='Type of agent to use')
flags.DEFINE_list('extra_info', ['qpos', 'qvel', 'tcp_pose', 'target'], "list of extra information to include")
flags.DEFINE_string('predict_target', None, 'whether we are setting the mse output to be target position')


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
flags.DEFINE_boolean('visual_normalize', False,
                     'Apply layer normalization for visual network')

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
        if self.config['env_name'] in ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED', 'PUSH_DISCONTINUOUS', 'PUSH_MULTIMODAL',
                          'PARTICLE', 'door-human-v0', 'hammer-human-v0', 'relocate-human-v0', 'pen-human-v0']:
            self.env_name = eval_env_module.get_env_name(config['env_name'], False,
                                            config['image_obs'])
            print(('Got env name:', self.env_name))
            self.env = eval_env_module.get_eval_env(
                self.env_name, 1, config['goal_tolerance'], 1)
            self.env_name_clean = self.env_name.replace('/', '_')
        else:
            self.env = self.create_env()

        # Create network and load from checkpoint
        self.network_visual, self.network = self.create_and_load_network()
        # create and load pretrained network
        self.pretrained_network, self.pretrained_network_visual = utils.create_and_load_pretrained_network(self.config)

        # Create ibc policy to evaluate
        self.ibc_policy = self.create_policy()

        # Create evaluation config -- TODO: read from eval_cfg
        self.save_video = config['save_video']
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
                fn = HangEnvPointcloud
        
        elif self.config['env_name'] == 'Fill-v0':
            if self.config['obs_mode'] == 'particles':
                fn = FillEnvParticle
            elif self.config['obs_mode'] == 'pointcloud':
                fn = FillEnvPointcloud 

        elif self.config['env_name'] == 'Excavate-v0':
            if self.config['obs_mode'] == 'particles':
                fn = ExcavateEnvParticle
            elif self.config['obs_mode'] == 'pointcloud':
                fn = ExcavateEnvPointcloud 
        
        else:
            print(f"Env {self.config['env_name']} obs mode {self.config['obs_mode']} not supported! Exiting")
            exit(0)
        
        env = fn(obs_mode=self.config['obs_mode'], control_mode=self.config['control_mode'],reward_mode=self.config['reward_mode'])
        if FLAGS.num_frames is not None and FLAGS.num_frames > 0:
            print("Using FrameStackWrapper with num_frames", FLAGS.num_frames)
            env = FrameStackWrapper(env, num_frames=FLAGS.num_frames)
        return env

    def create_and_load_network(self):
        config = self.config

        checkpoint_path = f"work_dirs/formal_exp/{config['env_name']}/{config['train_exp_name']}/checkpoints/"
        config['checkpoint_path'] = checkpoint_path
        config['resume_from_step'] = config['eval_step']

        network, network_visual = utils.create_network(config)
        
        return network_visual, network

    def create_policy(self):
        if self.config['agent_type'] == 'ibc':
            eval_policy = IbcPolicy( 
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
        elif self.config['agent_type'] == 'mse':
            eval_policy = MsePolicy(actor_network=self.network)

        return eval_policy

    def run_eval(self):
        '''
        Main eval loop. 
        '''
        if not os.path.exists(self.eval_info_path+'videos/'):
            os.makedirs(self.eval_info_path+'videos/')
        known_seed = [3,7,8,9,10]
        rewards_info = np.zeros(self.config['num_episodes'])
        shifted_rewards_info = np.zeros(self.config['num_episodes'])
        success_info = np.zeros(self.config['num_episodes'])
        target_distances = np.zeros((self.config['num_episodes'], 2))
        if self.config['eval_seeds_file']:
            seeds_file = json.load(open(self.config['eval_seeds_file']))
            seeds = [seeds_file['episodes'][i]['episode_seed'] for i in range(self.config['num_episodes'])]
        else:
            rng = np.random.default_rng(self.config['main_test_seed'])
            seeds = rng.choice(range(5000, 6000), size=self.config['num_episodes'], replace=False)

        for idx in range(self.config['num_episodes']):
            self.episode_id = idx
            seed = seeds[idx]
            # seed = known_seed[idx]
            total_reward, success, num_steps, shifted_reward, target_distance = self.run_single_episode(video_path=f"{self.eval_info_path}videos/{idx}",seed=seed)
            self.eval_info[f'eval_traj_{idx}'] = {
                'seed':seed, 'total_reward':total_reward,
                'success':success, 'num_steps':num_steps,
                'target_distance': target_distance,
            }
            rewards_info[idx] = total_reward
            shifted_rewards_info[idx] = shifted_reward
            success_info[idx] = success
            target_distances[idx] = target_distance
        self.eval_info[f'summary'] = {
                'success_rate':success_info.mean(), 
                'avg_rewards':rewards_info.mean(),  'max_rewards': rewards_info.max(),
                'min_rewards': rewards_info.min(), 'max_shifted_rewards': shifted_rewards_info.max(),
                'min_shifted_rewards': shifted_rewards_info.min(),
                'mean_target_distance': target_distances.mean(axis=0),
                'std_target_distance': target_distances.std(axis=0),
            }

        # with open(f"{self.eval_info_path}traj_info.json", 'w') as f:
        #     json.dump(self.eval_info, f, indent=4)
        save_json(self.eval_info, self.eval_info_path, "traj_info.json")
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
        shifted_reward = 0
        target_distance = torch.zeros(2)
        rm_predicted_target = utils.RunningMeanStd(shape=(2))
        # for num_steps in range(len(self.env._max_episode_steps)):
        for num_steps in range(1,self.config['max_episode_steps']+1):
            if done:
                success = True
                break
            # get current observation -- preprocessing handled by env wrapper
            obs = self.env.get_obs()
            if self.config['use_extra']:
                extra = obs['extra']
            if self.config['obs_mode'] == 'pointcloud':
                # only keep xyz
                obs = obs['pointcloud']['xyz']
                # flatten obs
                obs = obs.reshape(-1)
            with torch.no_grad():
                # if num_steps < 120:
                batch_observation = torch.Tensor(obs[None, :]).cuda()
                pretrain_visual_input_dim = config['visual_num_points'] * config['visual_num_channels']
                pretrain_visual_embed = self.pretrained_network_visual(batch_observation[:,:pretrain_visual_input_dim].reshape((-1, config['visual_num_points'], config['visual_num_channels'])))
                
                predict_target = self.pretrained_network(pretrain_visual_embed)
                target_distance += torch.nn.functional.mse_loss(predict_target.squeeze(), torch.Tensor(extra[21:])).item()
                
                rm_predicted_target.update(predict_target.cpu().numpy())
                # print(f"[eval | info] {rm_predicted_target.mean=}, gt target {extra[21:]=}")
                extra[21:] = rm_predicted_target.mean
            if self.config['use_extra']:
                # print('[eval|info] using extra info as obs. extra info shape', extra.shape)
                # if len(config['extra_info']) < 4:
                #     extra_dict = {'qpos': extra[:7], 'qvel': extra[7:14], 
                #         'tcp_pose': extra[14:21], 'target': extra[21:]}
                #     extra = np.concatenate([extra_dict[info_name] for info_name in config['extra_info']], axis = -1)
                obs = np.concatenate([obs, extra], axis=-1)
            
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
            shifted_reward += rew - self.config['single_step_max_reward']
            if self.save_video:
                imgs.append(self.env.render("rgb_array"))
        if self.save_video:
            utils.animate(imgs, path=f"{video_path}_seed={seed}_success={success}.mp4")
        else:
            print(f"seed={seed}_success={success}")
        target_distance = target_distance / num_steps
        
        return total_reward, success, num_steps, shifted_reward, target_distance.cpu().numpy()
            
        
        return total_reward, success, num_steps, shifted_reward

    def compute_mse(self):
        '''
        compute the average mse error between demo and policy prediction.
        '''

        # Load dataset and split into training and validation
        dataset = utils.load_customized_dataset(self.config)
        if self.config['data_amount']:
            # print("self.config['data_amount'], len(dataset)", self.config['data_amount'], len(dataset))
            assert self.config['data_amount'] <= len(dataset), f"Not enough data for {self.config['data_amount']} pairs!"
            train_dataset = torch.utils.data.Subset(dataset, range(self.config['data_amount']))
            validate_dataset = torch.utils.data.Subset(dataset, range(self.config['data_amount'], len(dataset)))
            print("loading validation dataset...... length", len(validate_dataset))
            assert validate_dataset is not None
        else:
            train_dataset = dataset
            validate_dataset = None

        # train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(range(len(train_dataset)), size=200))
        # if validate_dataset and len(validate_dataset) > 200:
        #     validate_dataset = torch.utils.data.Subset(validate_dataset, np.random.choice(range(len(validate_dataset)), size=200))
        
        train_dataloader = DataLoader(train_dataset, batch_size=64, 
            generator=torch.Generator(device='cuda'), 
            shuffle=True)
        if validate_dataset:
            validate_dataloader = DataLoader(validate_dataset, batch_size=64, 
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

            # print(obs.shape, act_gt.shape)

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

                # print("validate", loss_fn(act_gt, act_pred).item())
                validate_mse_loss.append(loss_fn(act_gt, act_pred).item())
        
        with open(f"{self.eval_info_path}mse_info.json", 'w') as f:
            if validate_dataset:
                json.dump({'training_data_mse': np.mean(train_mse_loss), 
                           'training_data_mse_max': np.max(train_mse_loss), 
                           'training_data_mse_min': np.min(train_mse_loss), 
                           'validation_data_mse': np.mean(validate_mse_loss),
                           'validation_data_mse_max': np.max(validate_mse_loss), 
                           'validation_data_mse_min': np.min(validate_mse_loss), 
                           }, f, indent=4)
            else:
                json.dump({'training_data_mse': np.mean(train_mse_loss), 
                           'training_data_mse_max': np.max(train_mse_loss), 
                           'training_data_mse_min': np.min(train_mse_loss),
                           'validation_data_mse': None}, f, indent=4)

    def eval_ibc_task(self):
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
        
        policy = eval_policy.Oracle(self.env, policy=self.ibc_policy, mse=False)
        # logging.info('Evaluating', epoch)
        for idx in range(max(1, int(self.config['num_episodes']/10))):
            video_module.make_video(
                    policy,
                    self.env,
                    f"{self.eval_info_path}/videos",
                    f"{self.config['eval_step']}_{idx}")
        eval_actor, success_metric = eval_actor_module.get_eval_actor(
                                policy,
                                self.env_name,
                                self.env,
                                self.config['eval_step'],
                                self.config['num_episodes'],
                                self.eval_info_path,
                                viz_img=self.config['viz_img'],
                                summary_dir_suffix=self.env_name_clean)
        
        metrics = self.evaluation_step(
            self.config['num_episodes'],
            self.env,
            eval_actor,
            name_scope_suffix=f'_{self.env_name}')
        # for m in metrics:
        #     writer.add_scalar(m.name, m.result(), epoch)
        print('Done evaluation')
        log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
        print('\n\t\t '.join(log))
        with open(self.eval_info_path+'metrics.txt', 'w') as f:
            f.write(' '.join(map(str, log)))
        print("evaluation at step", self.config['eval_step'], "\n", log)
    
    
    def evaluation_step(self, eval_episodes, eval_env, eval_actor, name_scope_suffix=''):
        """Evaluates the agent in the environment."""
        print('Evaluating policy.')
    
        # This will eval on seeds:
        # [0, 1, ..., eval_episodes-1]
        for eval_seed in range(eval_episodes):
            eval_env.seed(eval_seed)
            eval_actor.reset()  # With the new seed, the env actually needs reset.
            eval_actor.run()
        return eval_actor.metrics
    
if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    config = FLAGS.flag_values_dict()
    if config['action_spec_file'] is not None:
        with open(config['action_spec_file'], 'rb') as f:
            action_stat = np.load(f, allow_pickle=True).item()
            config['max_action'] = action_stat['max']
            config['min_action'] = action_stat['min']
    print(config)

    eval = Evaluation(config)
    
    if FLAGS.env_name in ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED', 'PUSH_DISCONTINUOUS', 'PUSH_MULTIMODAL',
                          'PARTICLE', 'door-human-v0', 'hammer-human-v0', 'relocate-human-v0', 'pen-human-v0']:
        eval.eval_ibc_task()

    elif FLAGS.num_episodes > 0:
        eval.run_eval()
    if FLAGS.compute_mse:
        eval.compute_mse()