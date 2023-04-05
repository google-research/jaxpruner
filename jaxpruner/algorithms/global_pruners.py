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

"""This file implements common global pruning algorithms."""
import dataclasses
from typing import Optional, Callable

import chex
import flax
import jax
import jax.numpy as jnp
from jaxpruner import base_updater
from jaxpruner import sparsity_distributions
from jaxpruner import sparsity_types
from jaxpruner.algorithms import pruners


@dataclasses.dataclass
class GlobalPruningMixin:
  """Implements Mixin for global pruning."""

  custom_sparsity_map: Optional[dict[str, float]] = None
  filter_fn: Callable[[tuple[str], chex.Array], bool] = (
      sparsity_distributions.NOT_DIM_ONE_FILTER_FN
  )
  normalization_eps: float = 1e-8
  use_normalization: bool = True
  sparsity: float = 0.0

  def init_state(self, params):
    """Regular init. but target_sparsities are set to orignal sparsity."""
    if not isinstance(self.sparsity_type, sparsity_types.Unstructured):
      raise AttributeError(
          f'Sparsity type {self.sparsity_type.__class__} is not supported for'
          ' GlobalPruningMixin.'
      )
    masks = self.create_masks(params, 0.0)
    if self.use_packed_masks:
      masks = jax.tree_map(jnp.packbits, masks)
    # Global sparsity only needs global target sparsity.
    return base_updater.SparseState(
        masks=masks,
        target_sparsities=self.sparsity,
        count=jnp.zeros([], jnp.int32),
    )

  def instant_sparsify(
      self, params, grads = None
  ):
    # Global sparsity doesn't require sparsity distribution as it uses
    # global ordering of scores.
    scores = self.calculate_scores(params, grads=grads)
    masks = self.create_masks(scores, self.sparsity)
    if self.use_packed_masks:
      masks = jax.tree_map(jnp.packbits, masks)
    return self.apply_masks(params, masks, is_packed=False), masks

  def create_masks(self, scores, target_sparsity):
    """Creates masks using global ordering."""
    custom_sparsity_map = self.custom_sparsity_map or {}

    def _unified_filter_fn(k, score):
      return self.filter_fn(k, score) and k not in custom_sparsity_map

    def _maybe_normalize(score):
      if self.use_normalization:
        return score / (jnp.linalg.norm(score) + self.normalization_eps)
      else:
        return score

    flat_scores = flax.traverse_util.flatten_dict(scores)
    filtered_scores = {
        k: _maybe_normalize(score)
        for k, score in flat_scores.items()
        if _unified_filter_fn(k, score)
    }
    ordered_keys = sorted(filtered_scores.keys())

    filtered_scores_concat = jnp.concatenate(
        [filtered_scores[k] for k in ordered_keys], axis=None
    )

    flat_mask_concat = self.topk_fn(filtered_scores_concat, target_sparsity)
    res_dict = {}
    cur_index = 0
    for k in ordered_keys:
      param = filtered_scores[k]
      next_index = cur_index + param.size
      flat_mask = flat_mask_concat[cur_index:next_index]
      res_dict[k] = jnp.reshape(flat_mask, param.shape)
      cur_index = next_index

    for k, score in flat_scores.items():
      if k in ordered_keys:
        pass
      elif k in custom_sparsity_map:
        res_dict[k] = self.topk_fn(score, custom_sparsity_map[k])
      else:
        res_dict[k] = None
    return_val = flax.traverse_util.unflatten_dict(res_dict)
    if isinstance(scores, flax.core.frozen_dict.FrozenDict):
      return_val = flax.core.freeze(return_val)
    return return_val


@dataclasses.dataclass
class GlobalMagnitudePruning(GlobalPruningMixin, pruners.MagnitudePruning):
  """Magnitude pruner with global ordering."""

  pass


@dataclasses.dataclass
class GlobalSaliencyPruning(GlobalPruningMixin, pruners.SaliencyPruning):
  """Saliency pruner with global ordering."""

  pass
