from agents import ibc_agent, eval_policy
from agents.ibc_policy import IbcPolicy
from network import mlp_ebm, ptnet_mlp_ebm
from environments.maniskill.maniskill_env import HangEnvIbc
import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import json
device = torch.device('cpu')

class Evaluation:
    def __init__(
        self, 
        env,
        network_backbone,
        network_ckpt,
        agent_cfg,
        eval_cfg
    ):
        # Create evaluation env
        self.env = env

        # Create policy network and load from checkpoint
        self.network = ptnet_mlp_ebm.PTNETMLPEBM(xyz_input_dim=1024, agent_input_dim=25, act_input_dim=8, out_dim=1).to(device)
        self.network.load_state_dict(torch.load(network_ckpt))
        print("Loaded checkpoint from path", network_ckpt)

        # Create policy to evaluate
        self.ibc_policy = IbcPolicy( 
            actor_network = self.network,
            action_spec= int(agent_cfg['act_shape'][0]), #hardcode
            # min_action = agent_cfg['min_action'], 
            # max_action = agent_cfg['max_action'],
            min_action=torch.tensor([float(i) for i in agent_cfg['min_action']]).float(),
            max_action=torch.tensor([float(i) for i in agent_cfg['max_action']]).float(),
            num_action_samples=int(agent_cfg['num_policy_sample']),
            use_dfo=agent_cfg['use_dfo'],
            use_langevin=agent_cfg['use_langevin'],
            optimize_again=agent_cfg['optimize_again'],
            inference_langevin_noise_scale=float(agent_cfg['inference_langevin_noise_scale']),
            again_stepsize_init=float(agent_cfg['again_stepsize_init'])
        )

        # Create evaluation config -- TODO: read from eval_cfg
        self.num_episodes = 1
        self.save_video = True
        self.episode_id = 0
        self.eval_info = []
        self.eval_info_path = '/home/yihe/ibc_torch/work_dirs/policy_exp/hang_10kPairs/eval/' + 'eval_info.npy'

    def run_eval(self):
        '''
        Main eval loop. 
        '''
        for idx in range(self.num_episodes):
            self.episode_id = idx
            total_reward, reach_TimeLimit, num_steps = self.run_single_episode()
            self.eval_info[idx] = [total_reward, reach_TimeLimit, num_steps]

        np.save(self.eval_info_path, self.eval_info)

    
    def run_single_episode(self, seed=0):
        '''
        Run evaluation for a single episode on a given seed.
        '''
        self.env.reset(seed=seed)
        total_reward = 0
        done = False
        
        # for num_steps in range(len(self.env._max_episode_steps)):
        for num_steps in range(350):
            if done:
                break

            print("at step ", num_steps)

            # get current observation -- preprocessing handled by env wrapper
            obs = self.env.get_obs()
            obs = torch.tensor(obs).to(device).expand(1, -1)
            # print('obs shape', obs.size())

            # get predicted action from policy
            act = self.ibc_policy.act({'observations':obs}).squeeze()
            # print('act shape', act.shape)
            print("prediceted action", act)

            # step and get rew, done
            _, rew, done, _ = self.env.step(act)

            # save info and update steps
            total_reward += rew
        
        return total_reward, not done, num_steps


if __name__ == "__main__":
    env = HangEnvIbc()
    print(env.get_obs().shape)

    agent_cfg = json.load(open('/home/yihe/ibc_torch/work_dirs/policy_exp/hang_10kPairs/config.json'))

    print(agent_cfg['use_dfo'])

    eval = Evaluation(
        env=env, 
        network_backbone=None,
        network_ckpt='/home/yihe/ibc_torch/work_dirs/policy_exp/hang_10kPairs/600.pt',
        agent_cfg=agent_cfg,
        eval_cfg=None
        )

    eval.run_eval()


