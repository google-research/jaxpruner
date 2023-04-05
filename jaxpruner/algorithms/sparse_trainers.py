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

"""Implementation of common (dynamic) sparse training algorithms."""
import dataclasses
import functools
from typing import Callable, Union

import chex
import jax
import jax.numpy as jnp
from jaxpruner import base_updater
from jaxpruner.algorithms import pruners
import optax


@dataclasses.dataclass
class StaticRandomSparse(base_updater.BaseUpdater):
  """Initializes sparsity randomly and optimizes using that sparsity."""

  is_sparse_gradients: bool = True

  def update_state(self, sparse_state, params, grads):
    """Returns sparse_state unmodified."""
    del params, grads
    return sparse_state

  def get_initial_masks(
      self, params, target_sparsities
  ):
    """Generate initial mask. This is only used when .wrap_optax() is called."""
    scores = pruners.generate_random_scores(params, self.rng_seed)
    init_masks = self.create_masks(scores, target_sparsities)
    return init_masks

  def instant_sparsify(self, params, grads=None):
    raise RuntimeError(
        'instant_sparsify function is not supported in sparse training methods.'
    )


def _restart_using_mask(target_tree, masks):
  # Set values to zero where mask is 1.
  return jax.tree_map(
      lambda a, m: a if m is None else a * (1 - m), target_tree, masks
  )


def restart_inner_state(state, masks_activated):
  """Restarts Adam and SGD state for the connections activated to zero."""
  if isinstance(state, optax.TraceState):
    restarted_trace = _restart_using_mask(state.trace, masks_activated)
    state = state._replace(trace=restarted_trace)
  elif isinstance(state, optax.ScaleByAdamState):
    new_mu = _restart_using_mask(state.mu, masks_activated)
    new_nu = _restart_using_mask(state.nu, masks_activated)
    state = state._replace(mu=new_mu, nu=new_nu)
  elif isinstance(state, (optax.EmptyState, optax.ScaleByScheduleState)):
    return state
  elif isinstance(state, optax.MaskedState):
    state = state._replace(
        inner_state=restart_inner_state(state.inner_state, masks_activated)
    )
  elif isinstance(state, tuple):
    # Note that optax state's are named tuples, and they would pass this test.
    # TODO find a way to check optax states.
    state = tuple(restart_inner_state(s, masks_activated) for s in state)
  else:
    raise ValueError('Unrecognized optimizer state of type: %s' % type(state))
  return state


@dataclasses.dataclass
class SET(StaticRandomSparse):
  """Implements dynamic sparse training with random growth.

  Reference: https://arxiv.org/abs/1707.04780

  Attributes:
    eps: used when calculating a smaller/larger value than existing min/max.
    drop_fraction_fn: Given nonnegative step number calculates drop fraction.
    is_debug: If true runs additional assertions during mask update.
  """

  eps: float = 1e-5
  drop_fraction_fn: Callable[Union[chex.Array, int], float] = lambda _: 0.3
  is_debug: bool = False
  skip_gradients: bool = True

  def _update_masks(self, old_mask, drop_score, grow_score, drop_fraction):
    """Updates the mask by replacing connections with potentially new ones."""
    density = jnp.sum(old_mask) / old_mask.size
    sparsity = 1 - density

    intermedieate_density = density * (1 - drop_fraction)
    intermedieate_sparsity = 1 - intermedieate_density

    # Reduce the scores for existing zeros so that they are pruned first.
    lowest_score = jnp.min(drop_score) - self.eps
    new_drop_score = jnp.where(old_mask == 0, lowest_score, drop_score)
    dropped_mask = self.topk_fn(new_drop_score, intermedieate_sparsity)
    if self.is_debug:
      # All ones in the dropped mask should exist in the original mask.
      chex.assert_trees_all_close(
          jnp.sum(dropped_mask * (1 - old_mask)), jnp.array(0)
      )

    # Raise the scores for existing connections so that they are selected first.
    highest_score = jnp.max(grow_score) + self.eps
    new_grow_scores = jnp.where(dropped_mask == 1, highest_score, grow_score)
    updated_mask = self.topk_fn(new_grow_scores, sparsity)
    if self.is_debug:
      # All ones in the dropped mask should exist in the original mask.
      chex.assert_trees_all_close(
          jnp.sum(dropped_mask * (1 - updated_mask)), jnp.array(0)
      )
    return updated_mask

  def _get_drop_scores(self, sparse_state, params, grads):
    del sparse_state, grads
    return jax.tree_map(jnp.abs, params)

  def _get_grow_scores(self, sparse_state, params, grads):
    new_rng = jax.random.fold_in(self.rng_seed, sparse_state.count)
    random_scores = pruners.generate_random_scores(params, new_rng)
    return random_scores

  def update_state(self, sparse_state, params, grads):
    """Updates the mask tree and returns a dict of metrics."""
    drop_scores = self._get_drop_scores(sparse_state, params, grads)
    grow_scores = self._get_grow_scores(sparse_state, params, grads)
    current_drop_fraction = self.drop_fraction_fn(sparse_state.count)
    update_masks_fn = functools.partial(
        self._update_masks, drop_fraction=current_drop_fraction
    )
    new_masks = jax.tree_map(
        update_masks_fn, sparse_state.masks, drop_scores, grow_scores
    )
    masks_activated = jax.tree_map(
        lambda old_mask, new_mask: (old_mask == 0) & (new_mask == 1),
        sparse_state.masks,
        new_masks,
    )
    new_inner_state = restart_inner_state(
        sparse_state.inner_state, masks_activated
    )
    return sparse_state._replace(masks=new_masks, inner_state=new_inner_state)


@dataclasses.dataclass
class RigL(SET):
  """Implements dynamic sparse training with gradient based growth.

  Reference: https://arxiv.org/abs/1911.11134
  """

  def _get_grow_scores(self, sparse_state, params, grads):
    del sparse_state, params
    return jax.tree_map(jnp.abs, grads)
