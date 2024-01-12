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

"""Tests for utility functions in mask_calculator.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jaxpruner import mask_calculator
from jaxpruner import sparsity_types


SCORE_1 = [
    [
        [0.69, 0.37, 0.96, 0.81],
        [0.79, 0.55, 0.88, 0.53],
        [0.66, 0.88, 0.4, 0.83],
        [0.13, 0.8, 0.91, 0.96],
    ],
    [
        [0.34, 0.17, 0.07, 0.74],
        [0.13, 0.13, 0.83, 0.76],
        [0.13, 0.92, 0.25, 0.75],
        [0.37, 0.42, 0.65, 0.46],
    ],
]
MASK_UNS_1 = [
    [[0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
]
MASK_N2_M4_1 = [
    [[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
    [[1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
]
MASK_N2_M4_AXIS1_1 = [
    [[1, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1]],
    [[1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]],
]
MASK_CHANNEL_AXIS1_1 = [
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
]
MASK_CHANNEL_AXIS2_1 = [
    [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ],
    [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ],
]

SCORE_2 = [1, 1, 1, 1, 1, 1, 1, 1]
MASK_UNS_2 = [1, 1, 1, 1, 0, 0, 0, 0]
MASK_N2_M4_2 = [0, 0, 1, 1, 0, 0, 1, 1]

SCORE_3 = [1, 2, 3, 4, 5, 6, 7, 8]

SCORE_4 = [1, 0.5, 0.3, 0.2, 1, 1.5, 1.8, 2.0, 1, 1]
MASK_UNS_4 = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0]

SCORE_5 = [
    [1, 0.5, 0.3, 0.2, 1, 1.5, 1.8, 2.0, 1, 1],
    [1, 0.5, 0.3, 0.2, 1, 1.5, 1.8, 2.0, 1, 1],
]
MASK_UNS_5 = [[1, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]]
MASK_CHANNEL_AXIS1_5 = [
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
]


class MaskCalculatorTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(
      (SCORE_1, 0.8, MASK_UNS_1),
      (SCORE_2, 0.5, MASK_UNS_2),
      (SCORE_4, 0.5, MASK_UNS_4),
      (SCORE_5, 0.5, MASK_UNS_5),
  )
  def testScoreBasedTopKWithUnstructuredInput(
      self, score, sparsity, expected_mask
  ):
    topk_fn = mask_calculator.get_topk_fn(sparsity_types.Unstructured)
    topk_fn_in_jit = jax.jit(topk_fn)
    self.assertTrue(
        jnp.array_equal(
            topk_fn(jnp.array(score), sparsity), jnp.array(expected_mask)
        )
    )
    self.assertTrue(
        jnp.array_equal(
            topk_fn_in_jit(jnp.array(score), sparsity), jnp.array(expected_mask)
        )
    )

  @parameterized.product(jit_compile=[False, True], pool_type=['AVG', 'MAX'])
  def testScoreBasedTopKWithBlockInput2dScores(self, jit_compile, pool_type):
    sparsity_type = sparsity_types.Block(
        block_shape=(2, 3),
        use_avg_pooling=(pool_type == 'AVG'),
    )
    topk_fn = mask_calculator.get_topk_fn(sparsity_type)
    if jit_compile:
      topk_fn = jax.jit(topk_fn)

    scores = [
        [0.69, 0.37, 0.96, 0.81, 0.79, 0.55, 0.88, 0.53],
        [0.66, 0.88, 0.4, 0.83, 0.13, 0.8, 0.91, 0.96],
        [0.34, 0.17, 0.07, 0.74, 0.13, 0.13, 0.83, 0.76],
        [0.13, 0.92, 0.25, 0.75, 0.37, 0.42, 0.65, 0.46],
    ]
    mask = topk_fn(scores, sparsity=0.2)

    expected_mask_avg = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    expected_mask_max = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]

    if pool_type == 'AVG':
      self.assertTrue(jnp.array_equal(mask, jnp.array(expected_mask_avg)))
    else:
      self.assertTrue(jnp.array_equal(mask, jnp.array(expected_mask_max)))

  @parameterized.product(jit_compile=[False, True], pool_type=['AVG', 'MAX'])
  def testScoreBasedTopKWithBlockInput1dScores(self, jit_compile, pool_type):
    sparsity_type = sparsity_types.Block(
        block_shape=(1, 3),
        use_avg_pooling=(pool_type == 'AVG'),
    )
    topk_fn = mask_calculator.get_topk_fn(sparsity_type)
    if jit_compile:
      topk_fn = jax.jit(topk_fn)

    scores = [0.69, 0.37, 0.96, 0.81, 0.79, 0.55, 0.88, 0.53]
    mask = topk_fn(scores, sparsity=0.2)

    expected_mask_avg = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    expected_mask_max = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    if pool_type == 'AVG':
      self.assertTrue(jnp.array_equal(mask, jnp.array(expected_mask_avg)))
    else:
      self.assertTrue(jnp.array_equal(mask, jnp.array(expected_mask_max)))

  @parameterized.product(jit_compile=[False, True], pool_type=['AVG', 'MAX'])
  def testScoreBasedTopKWithBlockInput3dScores(self, jit_compile, pool_type):
    sparsity_type = sparsity_types.Block(
        block_shape=(2, 2),
        use_avg_pooling=(pool_type == 'AVG'),
    )
    topk_fn = mask_calculator.get_topk_fn(sparsity_type)
    if jit_compile:
      topk_fn = jax.jit(topk_fn)

    scores = [
        [[0.69, 0.37], [0.96, 0.81], [0.79, 0.55], [0.88, 0.53]],
        [[0.66, 0.88], [0.4, 0.83], [0.13, 0.8], [0.91, 0.96]],
        [[0.34, 0.17], [0.07, 0.74], [0.13, 0.13], [0.83, 0.76]],
        [[0.13, 0.92], [0.25, 0.75], [0.37, 0.42], [0.65, 0.46]],
    ]
    mask = topk_fn(scores, sparsity=0.2)

    expected_mask_avg = [
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
    ]
    expected_mask_max = [
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
    ]

    if pool_type == 'AVG':
      self.assertTrue(jnp.array_equal(mask, jnp.array(expected_mask_avg)))
    else:
      self.assertTrue(jnp.array_equal(mask, jnp.array(expected_mask_max)))

  @parameterized.parameters(
      (SCORE_1, sparsity_types.NByM(n=2, m=4), MASK_N2_M4_1),
      (SCORE_2, sparsity_types.NByM(n=2, m=4), MASK_N2_M4_2),
      (SCORE_3, sparsity_types.NByM(n=4, m=2), None),
      (SCORE_1, sparsity_types.NByM(n=2, m=4, axis=1), MASK_N2_M4_AXIS1_1),
      (SCORE_3, sparsity_types.NByM(n=4, m=2, axis=0), None),
  )
  def testScoreBasedTopKWithNbyMInput(
      self, score, sparsity_type, expected_mask
  ):
    topk_fn = mask_calculator.get_topk_fn(sparsity_type)
    topk_fn_in_jit = jax.jit(topk_fn)

    if sparsity_type.n > sparsity_type.m:
      with self.assertRaises(ValueError):
        topk_fn(jnp.array(score), 0.5)
        topk_fn_in_jit(jnp.array(score), 0.5)
      return

    self.assertTrue(
        jnp.array_equal(
            topk_fn(jnp.array(score), 0.5), jnp.array(expected_mask)
        )
    )
    self.assertTrue(
        jnp.array_equal(
            topk_fn_in_jit(jnp.array(score), 0.5),
            jnp.array(expected_mask),
        )
    )

  @parameterized.parameters(
      dict(shape=[18, 4000, 4000], axis=-1),
      dict(shape=[18, 4000, 4000], axis=-2),
  )
  def testTopKWithNbyMLimitMemoryUsage(self, shape, axis):
    # `score`` takes about 2.304 GB (4 * 36 * 4000 * 4000 / 10**6) of memory.
    # Since the size of HBM is 8 GB for each core of Jellyfish. This test only
    # pass if the peak memory does not exceed 8GB of memory.
    score = jax.random.normal(jax.random.PRNGKey(0), shape)
    topk_fn = mask_calculator.get_topk_fn(
        sparsity_types.NByM(n=2, m=4, axis=axis)
    )
    topk_fn_in_jit = jax.jit(topk_fn)
    topk_fn_in_jit(score, 0.5)

  @parameterized.parameters(
      dict(shape=[2, 4, 4], axis=0),
      dict(shape=[2, 5, 3], axis=1),
  )
  def testTopKWithNbyMTargetAxisNotDivisibleByM(self, shape, axis):
    score = jax.random.normal(jax.random.PRNGKey(0), shape)
    topk_fn = mask_calculator.get_topk_fn(
        sparsity_types.NByM(n=2, m=4, axis=axis)
    )
    with self.assertRaises(ValueError):
      topk_fn(score, 0.5)

  @parameterized.parameters(
      dict(shape=[2, 4, 4], axis=1),
      dict(shape=[2, 4, 8], axis=2),
  )
  def testTopKWithNbyMGenerateNonPrunedMaskForSparsityZero(self, shape, axis):
    score = jax.random.normal(jax.random.PRNGKey(0), shape)
    topk_fn = mask_calculator.get_topk_fn(
        sparsity_types.NByM(n=2, m=4, axis=axis)
    )
    non_pruned_mask = topk_fn(score, 0)
    pruned_mask = topk_fn(score, 0.5)

    self.assertEqual((non_pruned_mask == 0).mean(), 0)
    self.assertEqual((pruned_mask == 0).mean(), 0.5)

  @parameterized.parameters(
      (SCORE_1, 0.5, sparsity_types.Channel(axis=1), MASK_CHANNEL_AXIS1_1),
      (SCORE_1, 0.5, sparsity_types.Channel(axis=-2), MASK_CHANNEL_AXIS1_1),
      (SCORE_1, 0.5, sparsity_types.Channel(), MASK_CHANNEL_AXIS2_1),
      (SCORE_1, 0.5, sparsity_types.Channel(axis=2), MASK_CHANNEL_AXIS2_1),
      (SCORE_1, 0.5, sparsity_types.Channel(axis=-1), MASK_CHANNEL_AXIS2_1),
      (SCORE_5, 0.5, sparsity_types.Channel(axis=1), MASK_CHANNEL_AXIS1_5),
  )
  def testScoreBasedTopKWithChannelInput(
      self, score, sparsity, sparsity_type, expected_mask
  ):
    topk_fn = mask_calculator.get_topk_fn(sparsity_type)
    topk_fn_in_jit = jax.jit(topk_fn)

    self.assertTrue(
        jnp.array_equal(
            topk_fn(jnp.array(score), sparsity), jnp.array(expected_mask)
        )
    )
    self.assertTrue(
        jnp.array_equal(
            topk_fn_in_jit(jnp.array(score), sparsity),
            jnp.array(expected_mask),
        )
    )


if __name__ == '__main__':
  absltest.main()
