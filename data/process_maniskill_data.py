'''
Converts maniskill trajectories in h5 file to observations and action lists.
'''
from pyrl.utils.data import GDict
import numpy as np

if __name__ == '__main__':
    original_h5 = GDict.from_hdf5('/home/yihe/ibc_torch/work_dirs/formal_demos/Hang-v0/trajectory.none.pd_joint_delta_pos_pcd.h5')
    observations = None
    actions = None

    for traj in original_h5.keys():
        
        visual = np.concatenate((original_h5[traj]['obs']['rgb'], original_h5[traj]['obs']['xyz']), axis=2)
        visual = visual.reshape(visual.shape[0], -1)
        obs = np.concatenate((visual, original_h5[traj]['obs']['state']), axis=1)

        if observations is not None:
            observations = np.concatenate((observations, obs), axis=0)
            actions = np.concatenate((actions, original_h5[traj]['actions']), axis=0)
        else:
            observations = obs
            actions = original_h5[traj]['actions']
    
    print(observations.shape, actions.shape)
    
    np.save('/home/yihe/ibc_torch/work_dirs/formal_demos/Hang-v0/pcd_observations.npy', observations)
    np.save('/home/yihe/ibc_torch/work_dirs/formal_demos/Hang-v0/action.npy', actions)