# coding=utf-8
# Copyright 2024 Jaxpruner Authors.
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

"""This file contains apis to use the jax prunder libraries."""
import gin
from jaxpruner import api
import ml_collections


@gin.configurable
def create_updater_from_config(
    pruner_type='magnitude',
    dist_type='erk',
    update_end_step=1,
    update_freq=1000,
    update_start_step=1,
    sparsity=None,
):
  """Gin based wrapper around jaxpruner create function."""
  sparsity_config = ml_collections.ConfigDict()
  sparsity_config.algorithm = pruner_type
  sparsity_config.dist_type = dist_type
  sparsity_config.update_end_step = update_end_step
  sparsity_config.update_freq = update_freq
  sparsity_config.update_start_step = update_start_step
  sparsity_config.sparsity = sparsity
  return api.create_updater_from_config(sparsity_config)
