import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
class CompositionPoints(gym.Env):
    
    def __init__(self, obs_mode=4, control_mode=1e-2, *args, **kwargs) -> None:
        super().__init__()
        self.num_points = int(obs_mode)
        self.atol = float(control_mode)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_points*4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_points*2,))
        self.curr = np.random.rand(self.num_points, 2) 
        self.end = np.random.rand(self.num_points, 2) 
        self._max_episode_steps = 100
    
    def evaluate(self):
        return -np.linalg.norm(self.curr - self.end)
    
    def get_obs(self):
        observation = np.concatenate([self.curr, self.end], axis=-1).flatten()
        return observation
    
    def step(self, action):
        self.curr += action.reshape(self.curr.shape)
        self.curr = np.clip(self.curr, 0, 1)
        reward = self.evaluate()
        done = np.allclose(self.curr, self.end, atol=self.atol)
        observation = self.get_obs()
        return observation, reward, done, {}

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.curr = np.random.rand(self.num_points, 2) 
        self.end = np.random.rand(self.num_points, 2) 
        observation = self.get_obs()
        return observation
    
    def render(self, *args):
        plt.clf()
        plt.scatter(self.end[:, 0], self.end[:, 1], color='red')
        plt.scatter(self.curr[:, 0], self.curr[:, 1])
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Convert the plot to a numpy array
        fig = plt.gcf()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        return img

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)

