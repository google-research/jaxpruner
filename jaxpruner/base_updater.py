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

"""This file implements common sparse / dense2sparse training algorithms."""
import dataclasses
import functools
import logging
from typing import Any, Callable, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxpruner import mask_calculator
from jaxpruner import sparsity_distributions
from jaxpruner import sparsity_schedules
from jaxpruner import sparsity_types
import optax

FilterFnType = sparsity_distributions.FilterFnType
CustomSparsityMapType = sparsity_distributions.CustomSparsityMapType

SparsityDistributionFnType = Callable[
    [
        chex.ArrayTree,
        sparsity_types.SparsityType,
        Optional[FilterFnType],
        Optional[CustomSparsityMapType],
    ],
    chex.ArrayTree,
]


class SparseState(NamedTuple):
  """Holds sparsity related state and the original optimization state."""

  masks: chex.ArrayTree
  inner_state: Any = None
  target_sparsities: Optional[chex.ArrayTree] = None
  count: Optional[chex.Array] = None


@functools.partial(jax.jit, static_argnums=2)
def apply_mask(param, mask, is_packed=False):
  if mask is None:
    return param
  elif not is_packed:
    return param * mask
  else:
    unpacked_mask = jnp.unpackbits(mask)[: param.size]
    unpacked_mask = unpacked_mask.reshape(param.shape)
    return param * unpacked_mask


@dataclasses.dataclass
class BaseUpdater(object):
  """Implements base class and the common update schedule."""

  scheduler: Any = sparsity_schedules.NoUpdateSchedule()
  # Whether to skip gradient updates when updating masks.
  skip_gradients: bool = False
  # TODO Create documentation for BaseUpdater.
  # Whether to zero out the gradients of pruned weights.
  is_sparse_gradients: bool = False
  # TODO Make sparsity default to be 0.
  sparsity_type: sparsity_types.SparsityType = sparsity_types.Unstructured()
  sparsity_distribution_fn: Optional[SparsityDistributionFnType] = (
      sparsity_distributions.uniform
  )
  rng_seed: Optional[jax.random.PRNGKey] = None
  use_packed_masks: bool = False

  def __post_init__(self):
    """Define variables."""
    # TODO Try enabling different sparsity types for different layers.
    self.topk_fn = mask_calculator.get_topk_fn(self.sparsity_type)
    # We set default value None, so that we dont't need to initialize jax
    # to read the file. Without this we get error when running the tests.
    if self.rng_seed is None:
      self.rng_seed = jax.random.PRNGKey(8)

  def apply_masks(
      self,
      params,
      masks,
      is_packed = False,
  ):
    """Sparsifies parameters according to the masks.

    Args:
      params: A tree of parameters.
      masks: A tree of masks, the structure of the nested tree is identical to
        that of params.
      is_packed: A boolean, if True, masks are assumed to be packed into bits
        and thus unpacked before sparsification.

    Returns:
      A tree of masked parameters.
    """
    if is_packed is None:
      is_packed = self.use_packed_masks
    return jax.tree_map(
        functools.partial(apply_mask, is_packed=is_packed), params, masks
    )

  def calculate_scores(
      self,
      params,
      sparse_state = None,
      grads = None,
  ):
    """Calculates sparsity scores of a given tree of parameters.

    Scores returned by this function is used to determining sparsity (e.g.,
    binary mask creation w/ top-k function). During training, this function is
    invoked by the `update_state` method which updates the sparse state during
    update iterations. This function is also invoked by the `instant_sparsify`
    method.

    Args:
      params: A tree of parameters.
      sparse_state: An optional sparse state.
      grads: An optional tree of gradients.

    Returns:
      A tree of the sparsity scores of parameters.
    Raises:
      NotImplemeterError.
    """
    raise NotImplementedError

  def get_initial_masks(
      self, params, target_sparsities
  ):
    """Generate initial mask. This is only used when .wrap_optax() is called."""

    def mask_fn(p, target):
      if target is None:
        return None
      return jnp.ones(p.shape, dtype=mask_calculator.MASK_DTYPE)

    masks = jax.tree_map(mask_fn, params, target_sparsities)
    return masks

  def create_masks(self, scores, sparsities):
    def topk_ifnot_none(score, sparsity):
      return None if sparsity is None else self.topk_fn(score, sparsity)

    return jax.tree_map(topk_ifnot_none, scores, sparsities)

  def init_state(self, params):
    """Creates the sparsity state."""
    if self.sparsity_distribution_fn is None:
      target_sparsities = None
    else:
      target_sparsities = self.sparsity_distribution_fn(params)
    logging.info('target_sparsities: %s', target_sparsities)
    masks = self.get_initial_masks(params, target_sparsities)
    if self.use_packed_masks:
      masks = jax.tree_map(jnp.packbits, masks)
    return SparseState(
        masks=masks,
        target_sparsities=target_sparsities,
        count=jnp.zeros([], jnp.int32),
    )

  def update_state(
      self,
      sparse_state,
      params,
      grads,
  ):
    """Update masks in sparse_state according to the current parameters and gradients.

    Args:
      sparse_state: A wrapped optimization state. Original optimization state is
        stored under `.inner`.
      params: A tree of parameters.
      grads: A tree of gradients.

    Returns:
      A sparse state with the updated masks tree.
    """
    sparsities = self.scheduler.get_sparsity_at_step(
        sparse_state.target_sparsities, sparse_state.count
    )
    scores = self.calculate_scores(
        params, sparse_state=sparse_state, grads=grads
    )
    new_masks = self.create_masks(scores, sparsities)
    if self.use_packed_masks:
      new_masks = jax.tree_map(jnp.packbits, new_masks)
    return sparse_state._replace(masks=new_masks)

  def wrap_optax(
      self, inner
  ):
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
      is_update_step = self.scheduler.is_mask_update_iter(state.count)
      no_update_op = lambda state, *_: state
      new_state = jax.lax.cond(
          is_update_step,
          self.update_state,
          no_update_op,
          state,
          params,
          updates,
      )

      if self.is_sparse_gradients:
        new_updates = self.apply_masks(updates, new_state.masks)
      else:
        new_updates = updates

      should_skip_inner = jnp.logical_and(is_update_step, self.skip_gradients)

      def no_inner_update(updates, inner_state, params):
        # Set gradients to zero and don't update the step.
        del params
        zero_updates = jax.tree_map(jnp.zeros_like, updates)
        return zero_updates, inner_state

      new_updates, new_inner_state = jax.lax.cond(
          should_skip_inner,
          no_inner_update,
          inner.update,
          new_updates,
          new_state.inner_state,
          params,
      )
      new_state = new_state._replace(
          count=optax.safe_int32_increment(new_state.count),
          inner_state=new_inner_state,
      )
      return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)

  def instant_sparsify(
      self,
      params,
      grads = None,
  ):
    """Instantly sparsifies parameters with the sparsity distribution provided at initialization.

    Args:
      params: A tree of parameters.
      grads: An optional tree of gradients.

    Returns:
      Either pruned parameters or a tuple of pruned parameters and masks.
    """
    if self.sparsity_distribution_fn is None:
      raise ValueError(
          'sparsity_distribution_fn needs to be provided for sparsification.'
      )
    sparsities = self.sparsity_distribution_fn(params)
    scores = self.calculate_scores(params, grads=grads)
    masks = self.create_masks(scores, sparsities)
    masked_params = self.apply_masks(params, masks, is_packed=False)
    if self.use_packed_masks:
      masks = jax.tree_map(jnp.packbits, masks)
    return masked_params, masks

  def pre_forward_update(
      self, params, opt_state
  ):
    """Used to transform paramaters before forward pass."""
    del opt_state
    return params

  def post_gradient_update(
      self, params, sparse_state
  ):
    """Used to transform parameters after gradient updates.

    The default behavior of this function is to sparsifiy parameters with masks
    after gradient updates.

    Args:
      params: A tree of parameters.
      sparse_state: A sparse state containing masks.

    Returns:
      A tree of pruned parameters.
    """
    return self.apply_masks(
        params, sparse_state.masks, is_packed=self.use_packed_masks
    )


@dataclasses.dataclass
class NoPruning(BaseUpdater):
  """Implements no-operation updater, which keeps the parameters dense."""
  sparsity_distribution_fn: Optional[SparsityDistributionFnType] = None

  def get_initial_masks(
      self, params, target_sparsities
  ):
    del target_sparsities
    return jax.tree_map(lambda p: None, params)

  def update_state(self, sparse_state, params, grads):
    """Identity operation, returns the state unmodified."""
    return sparse_state

  def instant_sparsify(self, params, grads=None):
    """Returns parameters as it is, together w/ None masks."""
    return params, self.get_initial_masks(params, None)
