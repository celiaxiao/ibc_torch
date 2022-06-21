import policy_eval
from environments.block_pushing import block_pushing
from environments.collect.utils import get_oracle as get_oracle_module
from train import make_video as video_module
import numpy as np
from tf_agents.environments import suite_gym
# print("evaluating.....")
policy_eval.evaluate(5, 'PUSH', False, False, False, 
    static_policy='push', video=True, output_path='data/block')
print('done!')