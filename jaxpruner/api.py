# coding=utf-8
# Copyright 2023 Jaxpruner Authors.
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

"""This file contains apis to use the JaxPruner libraries."""
import copy
import functools
import logging

from jaxpruner import algorithms
from jaxpruner import base_updater
from jaxpruner import sparsity_distributions
from jaxpruner import sparsity_schedules
from jaxpruner import sparsity_types
import ml_collections
import optax


ALGORITHM_REGISTRY = {
    'no_prune': base_updater.NoPruning,
    'magnitude': algorithms.MagnitudePruning,
    'random': algorithms.RandomPruning,
    'saliency': algorithms.SaliencyPruning,
    'magnitude_ste': algorithms.SteMagnitudePruning,
    'random_ste': algorithms.SteRandomPruning,
    'global_magnitude': algorithms.GlobalMagnitudePruning,
    'global_saliency': algorithms.GlobalSaliencyPruning,
    'static_sparse': algorithms.StaticRandomSparse,
    'rigl': algorithms.RigL,
    'set': algorithms.SET,
}
ALGORITHMS = tuple(ALGORITHM_REGISTRY.keys())


def create_updater_from_config(
    sparsity_config,
):
  """Gets a sparsity updater based on the given sparsity config.

  A sample usage of this api is in below.
  ```
    sparsity_config = ml_collections.ConfigDict()
    # Required
    sparsity_config.algorithm = 'magnitude'
    sparsity_config.dist_type = 'erk'
    sparsity_config.sparsity = 0.8

    # Optional
    sparsity_config.update_freq = 10
    sparsity_config.update_end_step = 1000
    sparsity_config.update_start_step = 200
    sparsity_config.sparsity_type = 'nm_2,4'

    updater = create_updater_from_config(sparsity_config)
  ```

  - `algorithm`: str, one of jaxpruner.ALGORITHMS
  - `dist_type`: str, 'erk' or 'uniform'.
  - `update_freq`: int, passed to PeriodicSchedule.
  - `update_end_step`: int, passed to PeriodicSchedule.
  - `update_start_step`: int, if None or doesn't exist NoUpdateSchedule is
    used. If equal to `update_end_step`, OneShotSchedule is used. Otherwise
    PolynomialSchedule is used.
  - `sparsity`: str, float or jaxpruner.SparsityType, if float, then
    SparsityType.Unstructured is used. If str in '{N}:{M}' format,
    then SparsityType.NbyM is used. If str in '{N}x{M}' format,
    then SparsityType.Block is used. User can also pass the desired
    SparsityType directly. Everything else under `sparsity_config`
    is passed to the algorithm directly.

  Args:
    sparsity_config: configuration for the algorithm. See options above.

  Returns:
    one of the jaxpruner.ALGORITHMS initiated with the given configuration.
  """
  logging.info('Creating  updater for %s', sparsity_config.algorithm)
  if sparsity_config.algorithm == 'no_prune':
    return base_updater.NoPruning()

  config = copy.deepcopy(sparsity_config).unlock()

  if config.dist_type == 'uniform':
    config.sparsity_distribution_fn = sparsity_distributions.uniform
  elif config.dist_type == 'erk':
    config.sparsity_distribution_fn = sparsity_distributions.erk
  else:
    raise ValueError(
        f'dist_type: {config.dist_type} is not supported. '
        'Use `erk` or `uniform`'
    )
  del config.dist_type

  if config.get('filter_fn', None):
    if not config.algorithm.startswith('global_'):
      new_fn = functools.partial(
          config.sparsity_distribution_fn, filter_fn=config.filter_fn
      )
      config.sparsity_distribution_fn = new_fn
      del config.filter_fn

  if config.get('custom_sparsity_map', None):
    if not config.algorithm.startswith('global_'):
      new_fn = functools.partial(
          config.sparsity_distribution_fn,
          custom_sparsity_map=config.custom_sparsity_map,
      )
      config.sparsity_distribution_fn = new_fn
      del config.custom_sparsity_map

  if config.algorithm.startswith('global_'):
    # Distribution function is not used.
    del config.sparsity_distribution_fn
  else:
    kwargs = {'sparsity': config.sparsity}
    del config.sparsity
    if config.get('filter_fn', None):
      kwargs['filter_fn'] = config.filter_fn
      del config.filter_fn
    config.sparsity_distribution_fn = functools.partial(
        config.sparsity_distribution_fn, **kwargs
    )

  if config.get('sparsity_type', None):
    s_type = config.sparsity_type
    if isinstance(s_type, str) and s_type.startswith('nm'):
      # example: nm_2,4
      n, m = s_type.split('_')[1].strip().split(',')
      del config.sparsity_type
      config.sparsity_type = sparsity_types.NByM(int(n), int(m))
    elif isinstance(s_type, str) and (s_type.startswith('block')):
      # example: block_4,4
      n, m = s_type.split('_')[1].strip().split(',')
      del config.sparsity_type
      config.sparsity_type = sparsity_types.Block(block_shape=(int(n), int(m)))
    elif not isinstance(s_type, sparsity_types.SparsityType):
      raise ValueError(f'Sparsity type {s_type} is not supported.')

  if config.algorithm in ALGORITHM_REGISTRY:
    updater_type = ALGORITHM_REGISTRY[config.algorithm]
    if config.algorithm in ('rigl', 'set'):
      config.drop_fraction_fn = optax.cosine_decay_schedule(
          config.get('drop_fraction', 0.1), config.update_end_step
      )
    del config.algorithm
  else:
    raise ValueError(
        f'Sparsity algorithm {config.algorithm} is not supported.'
        ' Please use jaxpruner.ALGORITHMS or ensure that your'
        ' custom algoritms are defined there.'
    )

  if config.get('update_start_step', None) is None:
    config.scheduler = sparsity_schedules.NoUpdateSchedule()
  elif config.update_end_step == config.update_start_step:
    config.scheduler = sparsity_schedules.OneShotSchedule(
        target_step=config.update_end_step
    )
  else:
    config.scheduler = sparsity_schedules.PolynomialSchedule(
        update_freq=config.update_freq,
        update_start_step=config.update_start_step,
        update_end_step=config.update_end_step,
    )
  for field_name in (
      'update_freq',
      'update_start_step',
      'update_end_step',
      'drop_fraction',
  ):
    if hasattr(config, field_name):
      delattr(config, field_name)

  updater = updater_type(**config)

  return updater
