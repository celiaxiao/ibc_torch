#!/bin/bash
if [ -z "$1" ]
  then
    GPUID=$1
else
    GPUID=2
fi
CUDA_VISIBLE_DEVICES=$GPUID python train/train_eval.py --train_task=door-human-v0 --exp_name=door_human
