## Workflow

Maniskill processing:
- pointcloud:
CUDA_VISIBLE_DEVICES=0 python data/maniskill_full_pipeline.py --h5_path=/home/caiwei/data/soft_body_envs/Fill-v0/trajectory.none.pd_joint_delta_pos.h5 --json_path=/home/caiwei/data/soft_body_envs/Fill-v0/trajectory.none.pd_joint_delta_pos.json --env_name=Fill-v0 --new_h5_path=/home/caiwei/data/soft_body_envs/Fill-v0/ --obs_mode=pointcloud
traj 174 finishs at step 31074
last traj starts at 34258
total 193 trajs

CUDA_VISIBLE_DEVICES=1 python data/maniskill_full_pipeline.py --h5_path=/home/caiwei/data/soft_body_envs/Hang-v0/trajectory.none.pd_joint_delta_pos.h5 --json_path=/home/caiwei/data/soft_body_envs/Hang-v0/trajectory.none.pd_joint_delta_pos.json --env_name=Hang-v0 --new_h5_path=/home/caiwei/data/soft_body_envs/Hang-v0/ --obs_mode=pointcloud
traj 170 finishs at step 45420
total 191 trajs

CUDA_VISIBLE_DEVICES=0 python data/maniskill_full_pipeline.py --h5_path=/home/caiwei/data/soft_body_envs/Excavate-v0/trajectory.none.pd_joint_delta_pos.h5 --json_path=/home/caiwei/data/soft_body_envs/Excavate-v0/trajectory.none.pd_joint_delta_pos.json --env_name=Excavate-v0 --new_h5_path=/home/caiwei/data/soft_body_envs/Excavate-v0/ --obs_mode=pointcloud
traj 179 finished at step 45420
last traj starts at 45345
total 199 trajs

CUDA_VISIBLE_DEVICES=0 python data/maniskill_full_pipeline.py  --h5_path=/home/caiwei/data/soft_body_envs/Fill-v0/trajectory.none.pd_joint_delta_pos.h5 --json_path=/home/caiwei/data/soft_body_envs/Fill-v0/trajectory.none.pd_joint_delta_pos.json  --env_name=Fill-v0 --new_h5_path=/home/caiwei/data/soft_body_envs/Fill-v0/
total 193 * 2 trajs
total 74149 steps
last 10 traj starts at 70310

Fill-v0 extra information
obs['agent']['qpos'].shape (7,)
obs['agent']['qvel'].shape (7,)
obs['extra']['tcp_pose'].shape (7,)
obs['extra']['target'].shape (2,)

For each task we will **(1) acquire data** either by:

  - (a) Generating training data from scratch with scripted oracles (via `policy_eval.py`), **OR**
  - (b) Downloading training data from the web.

And then **(2) run a train+eval** by:

  - Running both training and evaluation in one script (via `train_eval.py`)

Note that each train+eval will spend a minute or two
computing normalization statistics, then start training with example printouts:

```bash
I1013 22:26:42.807687 139814213846848 triggers.py:223] Step: 100, 11.514 steps/sec
I1013 22:26:48.352215 139814213846848 triggers.py:223] Step: 200, 18.036 steps/sec
```

And at certain intervals (set in the configs), run evaluations:

```bash
I1013 22:19:30.002617 140341789730624 train_eval.py:343] Evaluating policy.
...
I1013 22:21:11.054836 140341789730624 actor.py:196]
		 AverageReturn = 21.162763595581055
		 AverageEpisodeLength = 48.79999923706055
		 AverageFinalGoalDistance = 0.016136236488819122
		 AverageSuccessMetric = 1.0

```

There is **Tensorboard** support which can be obtained (for default configs) by running the following (and then going to `localhost:6006` in a browser.  (Might be slightly different for you to set up -- let us know if there are any issues.)

```bash
tensorboard --logdir /tmp/ibc_logs
```

And several chunks of useful information can be found in the train+eval log dirs for each experiment, which will end up for example at `/tmp/ibc_logs/mlp_ebm` after running the first suggested training.  For example `operative-gin-config.txt` will save out all the hyperparameters used for that training.
