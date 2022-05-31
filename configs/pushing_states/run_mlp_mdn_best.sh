#!/bin/bash

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/pushing_states/mlp_mdn_best.gin \
  --task=PUSH \
  --tag=mdn \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/block_push_states_location/oracle_push*.tfrecord'" \
  --video
