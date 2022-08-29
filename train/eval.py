from agents import ibc_agent, eval_policy
from agents.ibc_policy import IbcPolicy
from network import mlp_ebm, ptnet_mlp_ebm
from network.layers import pointnet
from environments.maniskill.maniskill_env import HangEnvParticle, HangEnvState
import torch
import numpy as np
import tqdm
import json
from moviepy.editor import ImageSequenceClip
device = torch.device('cuda')

class Evaluation:
    def __init__(
        self, 
        env,
        network_backbone,
        visual_backbone,
        network_ckpt,
        visual_ckpt,
        agent_cfg,
        eval_cfg
    ):
        # Create evaluation env
        self.env = env

        # Create policy network and load from checkpoint
        # self.network = ptnet_mlp_ebm.PTNETMLPEBM(xyz_input_dim=1024, agent_input_dim=25, act_input_dim=8, out_dim=1).to(device)
        self.network = network_backbone
        self.network_visual = visual_backbone

        self.network.load_state_dict(torch.load(network_ckpt))
        self.network_visual.load_state_dict(torch.load(visual_ckpt))
        print("Loaded checkpoint from path", network_ckpt, visual_ckpt)

        # Create policy to evaluate
        self.ibc_policy = IbcPolicy( 
            actor_network = self.network,
            action_spec= int(agent_cfg['act_shape'][0]), #hardcode
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
        self.eval_video_path = "/home/yihe/ibc_torch/work_dirs/policy_exp/speedup_largeBatch/eval/50pt.mp4"

    def run_eval(self):
        '''
        Main eval loop. 
        '''
        for idx in range(self.num_episodes):
            self.episode_id = idx
            total_reward, reach_TimeLimit, num_steps = self.run_single_episode()
            self.eval_info[idx] = [total_reward, reach_TimeLimit, num_steps]

        np.save(self.eval_info_path, self.eval_info)

    
    def run_single_episode(self, seed=3):
        '''
        Run evaluation for a single episode on a given seed.
        '''
        self.env.reset(seed=seed)
        total_reward = 0
        done = False
        imgs = [self.env.render("rgb_array")]
        
        # for num_steps in range(len(self.env._max_episode_steps)):
        for num_steps in range(350):
            if done:
                break

            print("at step ", num_steps)

            # get current observation -- preprocessing handled by env wrapper
            obs = self.env.get_obs()
            obs = torch.tensor(obs).to(device).expand(1, -1)
            # print('obs shape', obs.size())
            visual_embed = self.network_visual(obs[:,:1024*3].reshape(-1, 1024, 3))
            obs = torch.concat([visual_embed, obs[:,1024*3:]], -1)
            print('visual embed shape:', visual_embed.size(), ' new obs shape:', obs.size())
            # exit(0)

            # get predicted action from policy
            act = self.ibc_policy.act({'observations':obs}).squeeze()
            # print('act shape', act.shape)
            print("prediceted action", act)

            # step and get rew, done
            _, rew, done, _ = self.env.step(act.detach().cpu().numpy())
            print(rew, done)

            # save info and update steps
            total_reward += rew
            imgs.append(self.env.render("rgb_array"))
        
        self.animate(imgs, path=self.eval_video_path)
        
        return total_reward, not done, num_steps

    def animate(self, imgs, fps=20, path="animate.mp4"):
        imgs = ImageSequenceClip(imgs, fps=fps)
        imgs.write_videofile(path, fps=fps)


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    env = HangEnvParticle()
    # env = HangEnvState(target_file='/home/yihe/ibc_torch/work_dirs/demos/hang_state_test_target.npy')
    print(env.get_obs().shape)

    # agent_cfg = json.load(open('/home/yihe/ibc_torch/work_dirs/policy_exp/state-100lag/config.json'))
    agent_cfg = json.load(open('/home/yihe/ibc_torch/work_dirs/policy_exp/speedup_largeBatch/config.json'))

    # network_backbone = ptnet_mlp_ebm.PTNETMLPEBM(xyz_input_dim=1024, agent_input_dim=25, act_input_dim=8, out_dim=1)
    network_backbone = mlp_ebm.MLPEBM(537+8, 1, width=512, depth=8, normalizer=None, rate=0., dense_layer_type='spectral_norm')
    visual_backbone = pointnet.pointNetLayer(out_dim=512)

    eval = Evaluation(
        env=env, 
        network_backbone=network_backbone,
        visual_backbone=visual_backbone,
        network_ckpt='/home/yihe/ibc_torch/work_dirs/policy_exp/speedup_largeBatch/mlp_50.pt',
        visual_ckpt='/home/yihe/ibc_torch/work_dirs/policy_exp/speedup_largeBatch/pointnet_50.pt',
        agent_cfg=agent_cfg,
        eval_cfg=None
        )

    eval.run_eval()


