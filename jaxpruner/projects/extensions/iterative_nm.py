"""Straight Through Estimator based sparse training methods.

Reference: https://arxiv.org/abs/1308.3432
"""
import dataclasses
import functools
import jax
import jax.numpy as jnp
import jaxpruner
from jaxpruner import base_updater
from jaxpruner import mask_calculator
from jaxpruner import sparsity_types

BaseUpdater = base_updater.BaseUpdater


def iterative_nm_topk_mask_calculator(
    scores: jnp.ndarray, sparsity: float, sparsity_type: sparsity_types.NByM
) -> jnp.ndarray:
  """Given a tensor of scores creates a binary mask.

  Args:
    scores: top-scores are kept
    sparsity: of the generated mask.
    sparsity_type: defines the final target.

  Returns:
    array, same shape and type as scores.
  """

  def custom_topk(scores, sparsity):
    nm_mask = mask_calculator.topk_n_by_m_mask_calculator(scores, sparsity_type)
    highest_score = jnp.max(scores) + 1e-5
    new_scores = jnp.where(nm_mask == 1, highest_score, scores)
    return mask_calculator._topk_mask_calculator_internal(new_scores, sparsity)  # pylint: disable=protected-access

  return jax.lax.cond(
      sparsity == 0,
      lambda scores, _: jnp.ones_like(scores, dtype=mask_calculator.MASK_DTYPE),
      custom_topk,
      scores,
      sparsity,
  )


def custom_get_topk_fn(sparsity_type):
  """Adds support for new type."""
  if isinstance(sparsity_type, sparsity_types.NByM):
    return functools.partial(
        iterative_nm_topk_mask_calculator, sparsity_type=sparsity_type
    )
  else:
    return mask_calculator.get_topk_fn(sparsity_type)


@dataclasses.dataclass
class IterativeNMPruning(BaseUpdater):
  """Implements magnitude based pruning."""

  def __post_init__(self):
    """Define variables."""
    # TODO Try enabling different sparsity types for different layers.
    self.topk_fn = custom_get_topk_fn(self.sparsity_type)
    # We set default value None, so that we dont't need to initialize jax
    # to read the file. Without this we get error when running the tests.
    if self.rng_seed is None:
      self.rng_seed = jax.random.PRNGKey(8)

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state, grads
    param_magnitudes = jax.tree_map(jnp.abs, params)
    return param_magnitudes


def add_to_jaxpruner():
  """Add the 'iterative_nm' algorithm to the jaxpruner."""
  jaxpruner.register_algorithm('iterative_nm', IterativeNMPruning)
