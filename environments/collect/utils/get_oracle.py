# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gets oracles."""
import environments.block_pushing.oracles.oriented_push_oracle as oriented_push_oracle_module
import environments.block_pushing.oracles.reach_oracle as reach_oracle_module
import environments.block_pushing.oracles.discontinuous_push_oracle as discontinuous_push_oracle
import environments.block_pushing.oracles.multimodal_push_oracle as multimodal_push_oracle
import environments.particle.particle_oracles as particle_oracles
def get_oracle(env, task):
  """Gets an oracle for a given task."""
  if task == 'REACH':
    oracle_policy = reach_oracle_module.ReachOracle(env)
  elif task == 'PUSH':
    oracle_policy = oriented_push_oracle_module.OrientedPushOracle(env)
  elif task == 'PUSH_DISCONTINUOUS':
    oracle_policy = discontinuous_push_oracle.DiscontinuousOrientedPushOracle(env)
  elif task == 'PUSH_MULTIMODAL':
    oracle_policy = multimodal_push_oracle.MultimodalOrientedPushOracle(env)
  elif task == 'PARTICLE':
    oracle_policy = particle_oracles.ParticleOracle(env)
  else:
    raise ValueError('oracle not registered.')
  return oracle_policy
