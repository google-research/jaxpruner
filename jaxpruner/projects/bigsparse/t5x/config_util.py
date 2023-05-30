"""This file contains apis to use the jax prunder libraries."""
import gin
from jaxpruner import api
from jaxpruner.projects.bigsparse.t5x import ste_efficient
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
  api.ALGORITHM_REGISTRY['magnitude_ste_eff'] = (
      ste_efficient.EffSteMagnitudePruning
  )
  api.ALGORITHM_REGISTRY['magnitude_ste_eff_v2'] = (
      ste_efficient.EffSteMagnitudePruningV2
  )
  api.ALGORITHM_REGISTRY['random_ste_eff'] = ste_efficient.EffSteRandomPruning
  sparsity_config = ml_collections.ConfigDict()
  sparsity_config.algorithm = pruner_type
  sparsity_config.dist_type = dist_type
  sparsity_config.update_end_step = update_end_step
  sparsity_config.update_freq = update_freq
  sparsity_config.update_start_step = update_start_step
  sparsity_config.sparsity = sparsity
  return api.create_updater_from_config(sparsity_config)
