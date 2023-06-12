"""Straight Through Estimator based sparse training methods.

Reference: https://arxiv.org/abs/1308.3432
"""
import dataclasses
import chex
import jax
import jax.numpy as jnp
from jaxpruner import base_updater
from jaxpruner.algorithms import pruners
import optax


@dataclasses.dataclass
class EffSteMixinV2:
  """Implements Mixin for straight through estimator.

  This version uses the scaled gradients when calculating scores. This way, we
  can get a more accurate masks for the next step.

  Attributes:
    mask_decay_coef: float, defines weight decay on pruned weights.
  """
  mask_decay_coef = 0.0

  def pre_forward_update(
      self, params: chex.ArrayTree, sparse_state: base_updater.SparseState
  ) -> chex.ArrayTree:
    return self.apply_masks(
        params, sparse_state.masks, is_packed=self.use_packed_masks
    )

  def post_gradient_update(
      self, params: chex.ArrayTree, opt_state: base_updater.SparseState
  ) -> chex.ArrayTree:
    # Keep dense weights
    del opt_state
    return params

  def wrap_optax(
      self, inner: optax.GradientTransformation
  ) -> optax.GradientTransformation:
    """Wraps an existing optax transformation and adds sparsity related updates.

    The gradient transformation provided (`inner`) is called as it is at
    initialization and during the update steps. In addition to this, a sparse
    state is created using the `self.init_state` function, which includes
    variables like masks needed by the algorithms. The sparse state is updated
    according to the given schedule using `update_state` function.

    Args:
      inner: An optax gradient transformation.

    Returns:
      An updated optax gradient transformation.
    """

    def init_fn(params):
      sparse_state = self.init_state(params)
      sparse_state = sparse_state._replace(inner_state=inner.init(params))
      return sparse_state

    def update_fn(updates, state, params):
      if self.mask_decay_coef != 0.0:
        # SR-STE decay function.
        def _decay_fn(param, grad, mask):
          return grad + param * (1 - mask) * self.mask_decay_coef

        updates = jax.tree_map(_decay_fn, params, updates, state.masks)

      new_updates, new_inner_state = inner.update(
          updates, state.inner_state, params
      )

      is_update_step = self.scheduler.is_mask_update_iter(state.count)
      # Pass scaled gradients new_updates.
      no_update_op = lambda state, *_: state
      new_state = jax.lax.cond(
          is_update_step,
          self.update_state,
          no_update_op,
          state,
          params,
          new_updates,
      )

      if self.is_sparse_gradients:
        new_updates = self.apply_masks(new_updates, new_state.masks)

      new_state = new_state._replace(
          count=optax.safe_int32_increment(new_state.count),
          inner_state=new_inner_state,
      )
      return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


@dataclasses.dataclass
class EffSteMagnitudePruningV2(EffSteMixinV2, base_updater.BaseUpdater):
  """Magnitude pruner, which updates weight with straight through estimator."""

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state
    param_magnitudes = jax.tree_map(lambda p, g: jnp.abs(p + g), params, grads)
    return param_magnitudes


@dataclasses.dataclass
class EffSteMixin:
  """Implements Mixin for straight through estimator."""

  def pre_forward_update(
      self, params: chex.ArrayTree, sparse_state: base_updater.SparseState
  ) -> chex.ArrayTree:
    return self.apply_masks(
        params, sparse_state.masks, is_packed=self.use_packed_masks
    )

  def post_gradient_update(
      self, params: chex.ArrayTree, opt_state: base_updater.SparseState
  ) -> chex.ArrayTree:
    # Keep dense weights
    del opt_state
    return params


@dataclasses.dataclass
class EffSteMagnitudePruning(EffSteMixin, pruners.MagnitudePruning):
  """Magnitude pruner, which updates weight with straight through estimator."""

  pass


@dataclasses.dataclass
class EffSteRandomPruning(EffSteMixin, pruners.RandomPruning):
  """Random pruner, which updates weight with straight through estimator."""

  pass
