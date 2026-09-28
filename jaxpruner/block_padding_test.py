# coding=utf-8
# Copyright 2026 Jaxpruner Authors.
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

"""Regression tests for partial blocks in structured pruning masks."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jaxpruner import mask_calculator
from jaxpruner import sparsity_types
import numpy as np


def _reference_mask(scores, block_shape, use_avg_pooling, sparsity):
  """Rank only real, possibly partial blocks without padding the input."""
  matrix = scores.reshape(1, -1) if scores.ndim == 1 else scores
  height, width = block_shape
  blocks = []
  for row in range(0, matrix.shape[0], height):
    for col in range(0, matrix.shape[1], width):
      region = (slice(row, row + height), slice(col, col + width))
      values = matrix[region]
      value = values.mean() if use_avg_pooling else values.max()
      blocks.append((value, region))
  order = sorted(range(len(blocks)), key=lambda i: (-blocks[i][0], i))
  keep = max(1, round(len(blocks) * (1 - sparsity)))
  expected = np.zeros(matrix.shape, dtype=np.uint8)
  for index in order[:keep]:
    expected[blocks[index][1]] = 1
  return expected.reshape(scores.shape)


class BlockPaddingTest(parameterized.TestCase):

  @parameterized.product(
      case=[
          ((5, 5), (4, 4)),
          ((7, 7), (5, 5)),
          ((4, 7), (2, 3)),
          ((7, 4), (3, 2)),
          ((1, 2), (4, 5)),
          ((3, 2), (4, 4)),
          ((8, 6), (4, 3)),
          ((5, 5, 2, 3), (4, 4)),
          ((5,), (1, 4)),
      ],
      use_avg_pooling=[False, True],
      compiled=[False, True],
  )
  def test_partial_blocks_match_reference(
      self, case, use_avg_pooling, compiled
  ):
    shape, block_shape = case
    scores = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    # Negative inputs also check that max pooling ignores padded zeroes.
    scores -= scores.size
    expected = _reference_mask(scores, block_shape, use_avg_pooling, 0.5)
    topk = mask_calculator.get_topk_fn(
        sparsity_types.Block(block_shape, use_avg_pooling))
    if compiled:
      topk = jax.jit(topk)
    actual = topk(jnp.asarray(scores), 0.5)
    self.assertEqual(actual.shape, scores.shape)
    self.assertEqual(actual.dtype, np.dtype('uint8'))
    np.testing.assert_array_equal(actual, expected)

  @parameterized.product(use_avg_pooling=[False, True], sparsity=[0.0, 0.75])
  def test_partial_blocks_at_other_sparsities(self, use_avg_pooling, sparsity):
    scores = np.arange(35, dtype=np.float32).reshape(5, 7)
    topk = mask_calculator.get_topk_fn(
        sparsity_types.Block((4, 3), use_avg_pooling))
    expected = _reference_mask(scores, (4, 3), use_avg_pooling, sparsity)
    np.testing.assert_array_equal(topk(scores, sparsity), expected)

  @parameterized.product(use_avg_pooling=[False, True], compiled=[False, True])
  def test_tied_partial_blocks_use_existing_index_priority(
      self, use_avg_pooling, compiled
  ):
    scores = np.full((5, 5), -2.0, dtype=np.float32)
    topk = mask_calculator.get_topk_fn(
        sparsity_types.Block((4, 4), use_avg_pooling))
    if compiled:
      topk = jax.jit(topk)
    expected = _reference_mask(scores, (4, 4), use_avg_pooling, 0.5)
    np.testing.assert_array_equal(topk(scores, 0.5), expected)


if __name__ == '__main__':
  absltest.main()
