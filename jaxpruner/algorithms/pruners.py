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

"""This file implements common dense2sparse pruning algorithms."""
import dataclasses
import chex
import jax
import jax.numpy as jnp
from jaxpruner import base_updater

BaseUpdater = base_updater.BaseUpdater


@dataclasses.dataclass
class MagnitudePruning(BaseUpdater):
  """Implements magnitude based pruning."""

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state, grads
    param_magnitudes = jax.tree_map(jnp.abs, params)
    return param_magnitudes


@dataclasses.dataclass
class SaliencyPruning(BaseUpdater):
  """Implements saliency (magnitude*gradient) based pruning."""

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state
    saliencies = jax.tree_map(lambda p, g: jnp.abs(p * g), params, grads)
    return saliencies


def generate_random_scores(
    params, rng_seed
):
  """Generates random values matching the shape in params tree."""
  num_vars = len(jax.tree_util.tree_leaves(params))
  treedef = jax.tree_util.tree_structure(params)
  all_keys = jax.random.split(rng_seed, num=num_vars)

  return jax.tree_util.tree_map(
      lambda p, k: jax.random.uniform(k, shape=p.shape, dtype=p.dtype),
      params,
      jax.tree_util.tree_unflatten(treedef, all_keys),
  )


@dataclasses.dataclass
class RandomPruning(BaseUpdater):
  """Implements random pruning."""

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del grads
    if sparse_state is None:
      new_rng = self.rng_seed
    else:
      new_rng = jax.random.fold_in(self.rng_seed, sparse_state.count)

    random_scores = generate_random_scores(params, new_rng)

    # Apply mask so that we ensure pruned connections have lowest score.
    if sparse_state is not None:
      random_scores = self.apply_masks(random_scores, sparse_state.masks)
    return random_scores
