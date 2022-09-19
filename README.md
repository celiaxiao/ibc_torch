## Implicit Behavioral Cloning - PyTorch

Pytorch implementation of <a href="https://arxiv.org/abs/2109.00137">Implicit Behavioral Cloning</a>.

## Install

```bash
pip install -e .
```

## Maniskill Environments Full Pipeline
### Dataset processing
First, you need to download raw demos from [maniskill Google Drive](https://drive.google.com/drive/folders/1QCYgcmRs9SDhXj6fVWPzuv7ZSBL94q2R), make sure to download both the h5 and json file!  
The demo is in `pd_joint_pos` control mode, but we need `pd_joint_delta_pos` for learning. So please use the control mode conversion tool provided by maniskill to process the demo files using script below: 
```bash
# Replay demonstrations with control_mode=pd_joint_delta_pos
cd YOUR_MANISKILL_DIR
CUDA_VISIBLE_DEVICES=0 python tools/replay_trajectory.py --traj-path YOUR_DEMO_PATH/trajectory.h5 \
  --save-traj --target-control-mode pd_joint_delta_pos --obs-mode none --num-procs 5
```