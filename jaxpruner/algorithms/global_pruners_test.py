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

"""Tests for global pruning algorithms."""
from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
from jaxpruner.algorithms import global_pruners


class GlobalPrunerTest(parameterized.TestCase, absltest.TestCase):

  def testSETDefaultArguments(self):
    pruner = global_pruners.GlobalMagnitudePruning()
    self.assertIsNone(pruner.custom_sparsity_map)
    self.assertTrue(pruner.use_normalization)
    self.assertEqual(pruner.sparsity, 0)

  @parameterized.parameters([
      global_pruners.GlobalMagnitudePruning,
      global_pruners.GlobalSaliencyPruning,
  ])
  def testGlobalPruning(self, updater):
    # Sparsity is passed here for the topk_function to work.
    updater = updater(filter_fn=lambda *_: True, sparsity=0.5)
    score_tree = {
        'w': jnp.array([0.1, 0.9, -0.2, 0.5]),
        'w2': jnp.array([0.7, 0.8]),
    }
    expected_tree = {'w': jnp.array([0, 1, 0, 0]), 'w2': jnp.array([1, 1])}

    chex.assert_trees_all_close(
        updater.create_masks(score_tree, target_sparsity=0.5), expected_tree
    )


if __name__ == '__main__':
  absltest.main()
