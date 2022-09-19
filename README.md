## Implicit Behavioral Cloning - PyTorch

Pytorch implementation of <a href="https://arxiv.org/abs/2109.00137">Implicit Behavioral Cloning</a>.

## Install

```bash
pip install -e .
```

## Maniskill Environments Full Pipeline
### Dataset processing
First, you need to download raw demos from [maniskill Google Drive](https://drive.google.com/drive/folders/1QCYgcmRs9SDhXj6fVWPzuv7ZSBL94q2R), make sure to download both the h5 and json file!  

The demo is in `pd_joint_pos` control mode, but we need `pd_joint_delta_pos` for learning. So please use the control mode conversion tool provided by maniskill to process the demo files using script below. You should see two new files `trajectory.none.pd_joint_delta_pos.h5/.json`, these are the path that you need to pass in for next step.  
```bash
# Replay demonstrations with control_mode=pd_joint_delta_pos
cd YOUR_MANISKILL_DIR
CUDA_VISIBLE_DEVICES=0 python tools/replay_trajectory.py --traj-path YOUR_DEMO_PATH/trajectory.h5 \
  --save-traj --target-control-mode pd_joint_delta_pos --obs-mode none --num-procs 5
```
Note that for `Excavate-v0`, might need to add either `--allow-failure` or `--max-retry=5` to the end.  

After this, please run following script to generate training dataset. 
```bash
cd ibc_torch
CUDA_VISIBLE_DEVICES=0 python data/maniskill_full_pipeline.py \
--h5_path=CONVERTED_H5_FILE_PATH --json_path=CONVERTED_JSON_PATH \
--env_name=YOUR_ENV_NAME(e.g. Hang-v0) --raw_data_path=YOUR_NPY_DATA_PATH --dataset_path=YOUR_DATASET_PATH
```
When running this script, it should print you at first the `obs_dim` and `act_dim` for the env and demo. Record this and we will need them later in training and evaluation config file.  
Also, it should print to console replay success status for each trajectory in demo. Please let me know if there are a lot of False.

### Training
Both training and evaluation are configured using absl Flags. You can find sample training config file under `train/configs`. 
There are a few changes that you might want to make:
- env_name: change it to the env name you are training
- dataset_dir: your dataset generated in last step
- obs_dim/act_dim: put the number previous script printed
- **visual_num_points**: 1024 for Hang-v0 and Excavate-v0, 704 for Fill-v0
- data_amount: (Optional) use first x pairs of data in dataset

Before running training, you might want to log in to wandb in your teminal first.
Then, run the following script to train
```bash
python train/train.py --flagfile=EVAL_CFG_FILE_PATH --exp_name=YOUR_EXP_NAME (and any other special configs)
```

### Evaluation
Similar to training, you might want to make some modifications to the evaluation sample config file. 
Some evaluation arguments:
- max_episode_steps: 350 for Hang-v0, 250 for Fill-v0 and Excavate-v0
- dataset_dir: same dataset directory as training
- num_episodes: how many (random) seeds you want to evaluate your model on
- eval_step: the checkpoint you want to evaluate
- compute_mse: (Optional) add this argument if you want to compute mse to demo

Then, run the following script to evaluate
```bash
CUDA_VISIBLE_DEVICES=0 python train/eval.py --flagfile=EVAL_CFG_FILE_PATH --exp_name=YOUR_EXP_NAME
```

### Trouble Shoot
All `CUDA_VISIBLE_DEVICES=x` are required, except for training. Maniskill softbody envs requires this.

