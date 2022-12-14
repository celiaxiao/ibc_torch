#!/bin/bash
if [ -z "$1" ]
  then
    GPUID=$1
else
    GPUID=2
fi
CUDA_VISIBLE_DEVICES=$GPUID python train/train_eval.py --train_task=door-human-v0 --exp_name=door_human \
--eval_episodes=100

# CUDA_VISIBLE_DEVICES=2 python train/train_eval.py --train_task=door-human-v0 --exp_name=door_human_ckpt --eval --eval_epoch=9500 --eval_episodes=1