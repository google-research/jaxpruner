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

"""Calculate the mask with inputs such as score values."""

import functools

import jax
import jax.numpy as jnp
from jaxpruner import sparsity_types
import numpy as np


MASK_DTYPE = 'uint8'


def get_topk_fn(sparsity_type):
  """Returns the correct top_k function.

  Args:
    sparsity_type: Returns the relevant top_k function. For NByM sparsity
      returned function is wrapped with given N/M values. Similarly for Block
      sparsity block_shape and use_avg_pooling is fixed. This means returned
      function uses these fixed values.
  """
  if (
      isinstance(sparsity_type, sparsity_types.Unstructured)
      or sparsity_type == sparsity_types.Unstructured
  ):
    return topk_mask_calculator
  elif isinstance(sparsity_type, sparsity_types.Block):
    return functools.partial(
        topk_block_mask_calculator,
        block_height=sparsity_type.block_shape[0],
        block_width=sparsity_type.block_shape[1],
        use_avg_pooling=sparsity_type.use_avg_pooling,
    )
  elif isinstance(sparsity_type, sparsity_types.NByM):
    return lambda scores, _: topk_n_by_m_mask_calculator(scores, sparsity_type)
  else:
    raise ValueError(f'Not a supported sparsity type: {sparsity_type}')


@jax.jit
def _topk_mask_calculator_internal(scores, sparsity):
  """Creates a binary mask of same shape and type of scores, given sparsity."""
  flat_scores = jnp.reshape(scores, -1)
  num_ones = jnp.round(flat_scores.size * (1 - sparsity)).astype(int)
  num_ones = jnp.maximum(1, num_ones)

  topk_index = jnp.argsort(-flat_scores)[num_ones - 1]
  topk_threshold = flat_scores[topk_index]

  mask_by_value = scores >= topk_threshold

  # If there are multiple scores with the same value as `topk_threshold`,
  # thresholding cannot prune those duplicate scores. Therefore,`mask_by_value`
  # may have less sparsity than `sparsity` requested. For example, consider a
  # case where the scores have all the same values.
  #
  # To set priority among those duplicate socres, the index of the flattened
  # array is used, the scores with lower indices have higher precedence. The
  # scores with higher indicies than `topk_index` will be masked out. This is
  # applicable since the `argsort` indicies are sorted if they have the same
  # values. This will ensure that the generated mask has the desired `sparsity`.
  mask_by_index = (jnp.arange(flat_scores.size) <= topk_index) | (
      flat_scores != topk_threshold
  )
  mask_by_index = jnp.reshape(mask_by_index, scores.shape)
  return (mask_by_value * mask_by_index).astype(MASK_DTYPE)


def _pool_2d(
    scores,
    block_height,
    block_width,
    use_avg_pooling,
):
  """Performs 2D pooling on scores with block_shape + scores_shape[2:] window."""
  height_remain = scores.shape[0] % block_height
  width_remain = scores.shape[1] % block_width
  # Reduce all dims >= 2.
  pool_window = tuple([block_height, block_width] + list(scores.shape[2:]))
  if use_avg_pooling:
    padding = [(0, height_remain), (0, width_remain)] + [
        (0, 0) for _ in range(scores.ndim - 2)
    ]
    pooled = jax.lax.reduce_window(
        scores, 0.0, jax.lax.add, pool_window, pool_window, padding
    )

    if height_remain > 0 or width_remain > 0:
      ones = jax.lax.reduce_window(
          jnp.ones(scores.shape),
          0,
          jax.lax.add,
          pool_window,
          pool_window,
          padding,
      )
      pooled = pooled / ones.astype(pooled.dtype)
  else:
    if height_remain > 0 or width_remain > 0:
      padding_config = [(0, height_remain, 0), (0, width_remain, 0)] + [
          (0, 0, 0) for _ in range(scores.ndim - 2)
      ]
      scores_padded = jax.lax.pad(scores, -jnp.inf, padding_config)
    else:
      scores_padded = scores

    pooled = jax.lax.reduce_window(
        scores_padded,
        -jnp.inf,
        jax.lax.max,
        pool_window,
        pool_window,
        'VALID',
    )

  # Since all dims >= 2 are reduced, pooled.shape is always (A, B, 1, ..., 1).
  pooled_2d = pooled.reshape(*pooled.shape[:2])
  # Return as (A, B) shaped.
  return pooled_2d


@functools.partial(jax.jit, static_argnames=['sparsity_type'])
def topk_n_by_m_mask_calculator(
    scores, sparsity_type
):
  """Given a score of matrix creates a binary n by m mask.

  Applies sparsity to the last dimension of the score array.

  Args:
    scores: top-scores are kept
    sparsity_type: N:M structured sparsity.

  Returns:
    array, same shape and type as scores.

  Raises ValueError:
    if N > M in N:M sparsity.
  """
  n = sparsity_type.n
  m = sparsity_type.m
  if n > m:
    raise ValueError(
        f'N({n}) must be <= M({m}) in N({n}):M({m}) structured sparsity.'
    )
  length = jnp.size(scores)
  if length % m != 0:
    raise ValueError(
        f'size(scores) ({length}) is not divisible by M ({m})'
    )
  group = int(length / m)
  scores_temp = scores.reshape(group, m, order='C')
  # Calling argsort twice calculates ranks for each element at each row.
  ranks = scores_temp.argsort().argsort()
  mask = (ranks >= m - n).astype(jnp.uint8)
  return mask.reshape(scores.shape, order='C')


# TODO: Enable different layers having different block shapes.
@functools.partial(
    jax.jit, static_argnames=['block_height', 'block_width', 'use_avg_pooling']
)
def topk_block_mask_calculator(
    scores,
    sparsity,
    block_height,
    block_width,
    use_avg_pooling,
):
  """Given a score of matrix creates a binary block mask."""
  scores = jnp.array(scores)
  if scores.ndim < 2:
    pooled_scores = _pool_2d(
        scores.reshape(1, -1), block_height, block_width, use_avg_pooling
    )
    pooled_scores = jnp.squeeze(pooled_scores, axis=0)

    pooled_mask = _topk_mask_calculator_internal(pooled_scores, sparsity)
    mask = jnp.repeat(pooled_mask, block_width, axis=0)[: scores.shape[0]]
  else:
    pooled_scores = _pool_2d(scores, block_height, block_width, use_avg_pooling)

    pooled_mask = _topk_mask_calculator_internal(pooled_scores, sparsity)
    mask = jnp.repeat(pooled_mask, block_height, axis=0)[: scores.shape[0]]
    mask = jnp.repeat(mask, block_width, axis=1)[:, : scores.shape[1]]
    if scores.ndim > 2:
      mask = jnp.expand_dims(mask, 2)
      mask = jnp.repeat(mask, np.prod(scores.shape[2:]), axis=2).reshape(
          *scores.shape
      )
  return mask


def topk_mask_calculator(scores, sparsity):
  """Given a tensor of scores creates a binary mask.

  Args:
    scores: top-scores are kept
    sparsity: of the generated mask.

  Returns:
    array, same shape and type as scores.
  """
  return jax.lax.cond(
      sparsity == 0,
      lambda scores, _: jnp.ones_like(scores, dtype=MASK_DTYPE),
      _topk_mask_calculator_internal,
      scores,
      sparsity,
  )
