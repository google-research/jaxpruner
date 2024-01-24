"""This file contains apis to use the jax prunder libraries."""

import gin
import jax
import jax.numpy as jnp
from jaxpruner import api
from jaxpruner.projects.extensions import acdc
from jaxpruner.projects.extensions import iterative_nm
import ml_collections
import numpy as np
import optax


@gin.configurable
def create_updater_from_config(
    pruner_type='magnitude',
    dist_type='erk',
    update_end_step=1,
    update_freq=1000,
    update_start_step=1,
    sparsity=None,
    embed_sparsity=None,
    custom_sparsity_map=None,
    init_dense_steps_end=0,
    final_sparse_steps_start=0,
    cycle_sparse_steps=0,
    cycle_dense_steps=0,
    sparsity_type=None,
    ste_oneshot=False,
):
  """Gin based wrapper around jaxpruner create function."""
  iterative_nm.add_to_jaxpruner()
  acdc.add_to_jaxpruner()

  sparsity_config = ml_collections.ConfigDict()
  sparsity_config.algorithm = pruner_type
  sparsity_config.dist_type = dist_type
  sparsity_config.update_end_step = update_end_step
  sparsity_config.update_freq = update_freq
  sparsity_config.update_start_step = update_start_step
  sparsity_config.sparsity = sparsity

  if pruner_type == 'magnitude_ste_eff_v2':
    sparsity_config.oneshot = ste_oneshot
  if pruner_type == 'acdc':
    sparsity_config.init_dense_steps_end = init_dense_steps_end
    sparsity_config.final_sparse_steps_start = final_sparse_steps_start
    sparsity_config.cycle_sparse_steps = cycle_sparse_steps
    sparsity_config.cycle_dense_steps = cycle_dense_steps
  if sparsity_type is not None:
    sparsity_config.sparsity_type = sparsity_type

  def custom_filter_fn(key, param):
    return (param.ndim > 1) and ('rel_embedding' not in key)
    # Version which skips all embeddings
    # key = '/'.join(key)
    # res = param.ndim > 1 and ('kernel' in key)
    # res = res and ('embed' not in key) and ('logit' not in key)
    # return res

  if embed_sparsity == 'auto':
    embed_sparsity = 1 - 1 / (1 / (1 - sparsity)) ** 0.5

  sparsity_config.filter_fn = custom_filter_fn
  if custom_sparsity_map:
    sparsity_config.custom_sparsity_map = custom_sparsity_map
  elif embed_sparsity:
    custom_map = {
        ('decoder', 'logits_dense', 'kernel'): embed_sparsity,
        ('decoder', 'token_embedder', 'embedding'): embed_sparsity,
        ('token_embedder', 'embedding'): embed_sparsity,
    }

    sparsity_config.custom_sparsity_map = custom_map

  return api.create_updater_from_config(sparsity_config)


@gin.configurable
def b2_schedule(limit=1.0):
  """AdaFactor beta2 schedule with an upper limit."""

  def tmp(i, exponent):
    t = jnp.array(i + 1, jnp.float32)
    return jnp.minimum(1.0 - t ** (-exponent), jnp.array(limit, jnp.float32))

  return tmp


@gin.configurable
def clip_by_block_rms_sparse(threshold: float) -> optax.GradientTransformation:
  """AdaFactor RMS clipping relative only to nonzero params."""

  def init_fn(params):
    del params
    return optax.EmptyState()

  def update_fn(updates, state, params):
    def _clip_fn_sparse(u, p):
      nonzero = jax.numpy.count_nonzero(p)
      clip_denom = jnp.maximum(
          1.0, jnp.sqrt(jnp.sum((u * (p != 0)) ** 2) / nonzero) / threshold
      )
      return u / clip_denom

    updates = jax.tree_util.tree_map(_clip_fn_sparse, updates, params)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


@gin.configurable
def scale_by_param_block_rms_sparse(
    min_scale: float = 1e-3,
) -> optax.GradientTransformation:
  """AdaFactor parameter RMS scaling relative to only nonzero params."""

  def init_fn(params):
    del params
    return optax.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError()

    def _rms_sparse(p):
      rms = optax.safe_root_mean_squares(p, min_scale)
      rms = rms**2
      rms = rms * np.prod(p.shape) / jnp.count_nonzero(p)
      return rms**0.5

    updates = jax.tree_util.tree_map(
        lambda u, p: u * _rms_sparse(p), updates, params
    )
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)
