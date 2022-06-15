import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset


def get_target_action(current, next):
    return next - current

def get_action_decay(current, next):
    return next - current

def simple_traj(start, end, x):
    slope = (start - end)[1]/(start - end)[0]
    intercept = start[1] - slope*start[0]
    return slope * x + intercept

def simple_traj_noise(start,end,x):
    y = simple_traj(start,end,x)
    return y + np.random.normal()

def quadratic_traj1(start, end, x):
    x_start = start[0]
    x_end = end[0]
    return (x-x_end)*(x-x_start) + simple_traj(start, end, x)

def quadratic_traj2(start, end, x):
    x_start = start[0]
    x_end = end[0]
    return -1*(x-x_end)*(x-x_start) + simple_traj(start, end, x)

def experience_line_decay(start, end, traj):
    experiences = []
    curr = start[0]
    while(np.abs(end[0]-curr) > 1e-5):
        obs = {'target': end, 'state':np.array([curr,  traj(start, end, curr)])}
        next_point = curr + (10. - curr)/20.
        act = get_target_action(next=np.array([next_point, traj(start, end, next_point)]),
             current=np.array([curr, traj(start, end, curr)]))
        curr = next_point
        experience = {'observation': obs, 'action': act}
        # print(experience)
        experiences.append(experience)
    obs_stop = {'target': end, 'state':end}
    act_stop = np.array([0, 0], dtype=np.float32)
    experience_stop = {'observation': obs_stop, 'action': act_stop}
    experiences.append(experience_stop)
    # print(traj, experiences)
    return experiences

def experience_line_dense(start, end, traj):
    experiences = []
    x = np.linspace(start[0], end[0], 300)
    for curr in x:
        obs = {'target': end, 'state':np.array([curr,  traj(start, end, curr)])}
        next_point = curr + (10. - curr)/10.
        act = get_target_action(next=np.array([next_point, traj(start, end, next_point)]),
                current=np.array([curr, traj(start, end, curr)]))
        experience = {'observation': obs, 'action': act}
        # print(experience)
        experiences.append(experience)
    obs_stop = {'target': end, 'state':end}
    act_stop = np.array([0, 0], dtype=np.float32)
    experience_stop = {'observation': obs_stop, 'action': act_stop}
    experiences.append(experience_stop)
    # print(traj, experiences)
    return experiences

def get_experience_noise(start, end, numLine=5):
    experiences = []
    for i in range (numLine):
        experiences_line = experience_line_dense(start, end, simple_traj_noise)
        experiences_line_obs = [experience['observation'] for experience in experiences_line]
        line = np.array([obs['state'] for obs in experiences_line_obs], dtype=np.float32)
        # print('line number', i, line)
        print(line.shape) # current state only
        end_noise = line[-2]
        for i in range(len(experiences_line)):
            experiences_line[i]['observation']['target'] = end_noise
        experiences_line[-1]['observation']['state'] = end_noise

        # for exp in experiences_line:
        #     print(exp)
        experiences += experiences_line

    return experiences



class Simple_trajectory_dataset(Dataset):
    def __init__(self, experiences):
        device = torch.device('cuda')
        for idx in range(len(experiences)):
            obs_dict = experiences[idx]['observation']
            for key in obs_dict.keys():
                obs_dict[key] = torch.tensor(obs_dict[key]).float().to(device)
            experiences[idx]['observation'] = obs_dict
            experiences[idx]['action'] = torch.tensor(experiences[idx]['action']).float().to(device)
            # print('cast tensor', experiences[idx], '\n')
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        obs_dict = self.experiences[idx]['observation']
        obs = torch.concat([torch.flatten(obs_dict[key]) for key in obs_dict.keys()], axis=-1)
        act = self.experiences[idx]['action'].float()
        return obs, act
   
if __name__ == '__main__':
    start = np.array([0, 0], dtype=np.float32)
    end = np.array([10,10], dtype=np.float32)
    experiences = get_experience_noise(start, end)
    # print(experiences[:3])
    # line
    # experiences_line = experience_line_dense(start, end, simple_traj_noise)
    # experiences_line_obs = [experience['observation'] for experience in experiences_line]
    # line = np.array([obs['state'] for obs in experiences_line_obs], dtype=np.float32)
    # print(line.shape) # current state only
    # plt.scatter(line[:,0], line[:,1])
    # plt.savefig('trajectory/test_line_noise.png')
    # plt.close()
    # experiences = experiences_line

    # quadratic 1
    # experiences_qua = experience_line_dense(start, end, quadratic_traj1)
    # experiences_qua_obs = [experience['observation'] for experience in experiences_qua]
    # line = np.array([obs['state'] for obs in experiences_qua_obs], dtype=np.float32)
    # print(line.shape) # current state only
    # plt.scatter(line[:,0], line[:,1])
    # plt.savefig('trajectory/test_quadratic1_dense.png')
    # plt.close()
    # experiences = experiences_qua

    # # quadratic 2
    # experiences_qua2 = experience_line_dense(start, end, quadratic_traj2)
    # experiences_qua2_obs = [experience['observation'] for experience in experiences_qua2]
    # line = np.array([obs['state'] for obs in experiences_qua2_obs], dtype=np.float32)
    # print(line.shape) # current state only
    # plt.scatter(line[:,0], line[:,1])
    # plt.savefig('trajectory/test_quadratic2_dense.png')
    # plt.close()
    line_all = np.array([exp['action'] for exp in experiences], dtype=np.float32)
    print(line_all.shape)
    # # experiences = experiences_line + experiences_qua + experiences_qua2
    experiences = np.array(experiences)
    dataset = Simple_trajectory_dataset(experiences)
    torch.save(dataset, 'trajectory/5line_noise.pt')
    
