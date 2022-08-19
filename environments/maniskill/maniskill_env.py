from mani_skill2.envs.mpm.hang_env import HangEnv
import numpy as np

class HangEnvIbc(HangEnv):
    '''
    Maniskill HangEnv with ibc-formatted observation wrapper.
    '''
    def get_obs(self):
        obs = super().get_obs()

        xyz = obs['particles']['x'][np.random.choice(range(len(obs['particles']['x'])), size=1024, replace=False)]
        agent = np.hstack((obs['agent']['qpos'], obs['agent']['qvel'], self.rod.get_pose().p,self.rod.get_pose().q))

        return np.concatenate((xyz.reshape(-1,1).squeeze(), agent))
       