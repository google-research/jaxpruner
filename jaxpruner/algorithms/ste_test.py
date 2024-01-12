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

"""Tests for Straight Through Estimator based sparse training methods."""
import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
from jaxpruner import sparsity_distributions
from jaxpruner import utils
from jaxpruner.algorithms import ste
import optax


class STETest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters([ste.SteRandomPruning, ste.SteMagnitudePruning])
  def testStePruningGeneratesMaskEachIterationAndKeepDenseWeight(self, updater):
    dist_fn = functools.partial(sparsity_distributions.uniform, sparsity=0.5)
    updater = updater(sparsity_distribution_fn=dist_fn)
    # Ensure weight diverge from zero
    grad_fn = jax.grad(lambda p: -1 * jax.lax.abs(p['w']).sum())
    params = {'w': jax.random.normal(jax.random.PRNGKey(0), (10, 10))}
    optimizer = updater.wrap_optax(optax.adam(0.0001))
    opt_state = optimizer.init(params)

    updater.calculate_scores = mock.MagicMock(wraps=updater.calculate_scores)
    updater.scheduler = mock.MagicMock(wraps=updater.scheduler)
    for _ in range(10):
      pruned_params = updater.pre_forward_update(params, opt_state)

      # Check functions to scores are calculated.
      updater.calculate_scores.assert_called()
      updater.scheduler.get_sparsity_at_step.assert_called()

      grad = grad_fn(pruned_params)
      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      updated_params = updater.post_gradient_update(params, opt_state)

      # Validate pruned weight has sparsity of 0.5
      pruned_weight_sparsity = utils.summarize_sparsity(
          pruned_params, only_total_sparsity=True
      )['_total_sparsity']
      self.assertEqual(pruned_weight_sparsity, 0.5)

      # post_gradient_update does not change parameters
      self.assertEqual(params, updated_params)
      params = updated_params

      # original weight must keep in dense
      weight_sparsity = utils.summarize_sparsity(
          params, only_total_sparsity=True
      )['_total_sparsity']
      self.assertEqual(weight_sparsity, 0)

      updater.calculate_scores.reset_mock()
      updater.scheduler.get_sparsity_at_step.reset_mock()

  def testSteMagnitudePruningInstantSparsify(self):
    dist_fn = functools.partial(sparsity_distributions.uniform, sparsity=0.5)
    updater = ste.SteMagnitudePruning(sparsity_distribution_fn=dist_fn)
    weight = jnp.array([
        [0.11, -1.08, -1.6, 0.37],
        [-0.3, 1.03, 0.02, -0.78],
        [-0.95, 0.47, -1.3, 0.87],
        [1.96, 3.02, 0.3, -0.92],
    ])
    pruned_weight = jnp.array([
        [0.0, -1.08, -1.6, 0.0],
        [0.0, 1.03, 0.0, 0.0],
        [-0.95, 0.0, -1.3, 0.0],
        [1.96, 3.02, 0.0, -0.92],
    ])
    chex.assert_trees_all_close(
        pruned_weight, updater.instant_sparsify(weight)[0]
    )

  def testSteRandomPruningInstantSparsify(self):
    dist_fn = functools.partial(sparsity_distributions.uniform, sparsity=0.5)
    updater = ste.SteRandomPruning(sparsity_distribution_fn=dist_fn)
    weight = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    pruned_weight, _ = updater.instant_sparsify(weight)
    pruned_weight_sparsity = utils.summarize_sparsity(
        pruned_weight, only_total_sparsity=True
    )['_total_sparsity']
    self.assertEqual(pruned_weight_sparsity, 0.5)


if __name__ == '__main__':
  absltest.main()
