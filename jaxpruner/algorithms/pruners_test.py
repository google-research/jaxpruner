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

"""Tests for common pruners."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
from jaxpruner import sparsity_distributions
from jaxpruner import utils
from jaxpruner.algorithms import pruners


class PrunerTest(parameterized.TestCase, absltest.TestCase):

  def testMagnitudePruning(self):
    updater = pruners.MagnitudePruning()
    param_tree = {'w': jnp.array([0.1, 0.9, -0.2, 0.5])}
    expected = {'w': jnp.array([0.1, 0.9, 0.2, 0.5])}
    chex.assert_trees_all_close(updater.calculate_scores(param_tree), expected)

  def testSaliencyPruning(self):
    updater = pruners.SaliencyPruning()
    param_tree = {'w': jnp.array([0.1, 0.9, 0.2, 0.5])}
    grad_tree = {'w': jnp.array([5, 1, 3, 0.5])}
    expected = {'w': jnp.array([0.5, 0.9, 0.6, 0.25])}
    chex.assert_trees_all_close(
        updater.calculate_scores(params=param_tree, grads=grad_tree), expected
    )

  def testRandomPruning(self):
    target_sparsity = 0.5
    sparsity_fn = functools.partial(
        sparsity_distributions.uniform, sparsity=target_sparsity
    )
    updater = pruners.RandomPruning(sparsity_distribution_fn=sparsity_fn)
    param_tree = {'w': jax.random.normal(jax.random.PRNGKey(0), (10, 10))}
    updater.rng_seed = jax.random.PRNGKey(0)
    score1 = updater.calculate_scores(param_tree)
    updater.rng_seed = jax.random.PRNGKey(5)
    score2 = updater.calculate_scores(param_tree)
    self.assertFalse(
        jnp.array_equal(
            score1['w'],
            score2['w'],
        )
    )

    mask = updater.create_masks(score1, {'w': target_sparsity})
    self.assertEqual(utils.summarize_sparsity(mask)['_total_sparsity'], 0.5)


if __name__ == '__main__':
  absltest.main()
