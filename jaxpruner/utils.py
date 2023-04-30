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

"""Common utils for sparse training."""
from typing import Dict

import chex
import flax
import jax
import jax.numpy as jnp


def summarize_sparsity(
    param_tree, only_total_sparsity = False
):
  """Calculates the sparsity of a given parameter tree.

  Args:
    param_tree: A parameters tree
    only_total_sparsity: A boolean, if True returns only the total sparsity of
      the given parameters tree; the key is `_total_sparsity`. Otherwise
      summarizes the sparsity of every parameters in the tree separately.

  Returns:
    A dict summarizing the sparsity of each parameter.
  """
  non_zeros = jax.tree_map(jnp.count_nonzero, param_tree)
  sizes = jax.tree_map(jnp.size, param_tree)

  # Cast to float to avoid overflow errors on very large models
  sparsity_fn = lambda nnz, size: (1 - jnp.float32(nnz) / jnp.float32(size))

  if only_total_sparsity:
    summary_dict = {}
  else:
    sparsities = jax.tree_map(sparsity_fn, non_zeros, sizes)
    if isinstance(sparsities, list):
      summary_dict = dict([(str(k), v) for (k, v) in enumerate(sparsities)])
    else:
      summary_dict = flax.traverse_util.flatten_dict(sparsities, sep='/')
  total_sparsity = sparsity_fn(
      sum(jax.tree_util.tree_leaves(non_zeros)),
      sum(jax.tree_util.tree_leaves(sizes)),
  )
  summary_dict['_total_sparsity'] = total_sparsity

  return summary_dict


def summarize_intersection(
    mask_tree1,
    mask_tree2,
    only_total_intersection = False,
):
  """Calculates the intersection between 1's of between masks.

  Args:
    mask_tree1: A tree of mask parameters
    mask_tree2: A tree of mask parameters
    only_total_intersection: A boolean, if True returns only the total relative
      intersection of the given mask trees; with the key `_total_intersection`.
      Otherwise summarizes the relative intersection of every parameters in the
      tree separately.

  Returns:
    A dict summarizing the sparsity of each parameter.
  """

  def intersection_fn(a, b):
    if a is None:
      return 0
    return jnp.sum((a == 1) & (b == 1))

  intersections = jax.tree_map(intersection_fn, mask_tree1, mask_tree2)
  ones = jax.tree_map(lambda a: 0 if a is None else jnp.sum(a), mask_tree1)

  def _safe_division(a, b):
    b = jnp.maximum(b, 1)
    return a / b

  if only_total_intersection:
    summary_dict = {}
  else:
    relative_intersections = jax.tree_map(_safe_division, intersections, ones)
    if isinstance(mask_tree1, list):
      summary_dict = dict(
          [(str(k), v) for (k, v) in enumerate(relative_intersections)]
      )
    else:
      summary_dict = flax.traverse_util.flatten_dict(
          relative_intersections, sep='/'
      )

  total_intersection = _safe_division(
      sum(jax.tree_util.tree_leaves(intersections)),
      sum(jax.tree_util.tree_leaves(ones)),
  )

  summary_dict['_total_intersection'] = total_intersection

  return summary_dict
