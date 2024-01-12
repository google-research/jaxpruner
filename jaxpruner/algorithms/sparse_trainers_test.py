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

"""Tests for common sparse training algorithms."""
import functools
from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
from jaxpruner import sparsity_distributions
from jaxpruner.algorithms import sparse_trainers
import optax


class SparseTrainerTest(parameterized.TestCase, absltest.TestCase):

  def testStaticRandomSparseDefaultArguments(self):
    self.assertTrue(sparse_trainers.StaticRandomSparse().is_sparse_gradients)

  def testSETDefaultArguments(self):
    updater = sparse_trainers.SET()
    self.assertTrue(updater.is_sparse_gradients)
    self.assertTrue(updater.skip_gradients)
    self.assertFalse(updater.is_debug)

  def testStaticRandomSparse(self):
    sparsity = 0.75
    sparsity_fn = functools.partial(
        sparsity_distributions.uniform, sparsity=sparsity
    )
    updater = sparse_trainers.StaticRandomSparse(
        sparsity_distribution_fn=sparsity_fn
    )
    param_tree = {'w': jnp.array([0.1, 0.9, 0.2, 0.5])}
    sparsity_tree = {'w': sparsity}
    initial_masks = updater.get_initial_masks(param_tree, sparsity_tree)
    self.assertEqual(jnp.sum(initial_masks['w']), 1)

  @parameterized.parameters(
      # Masked drop [X, 1, X ,-1, X, 8, -2, X]
      # grow [9, 1, 4 ,1, 5, 8, 2, 3]
      [
          (0.0, [0, 1, 0, 1, 0, 1, 1, 0]),
          (0.25, [1, 1, 0, 1, 0, 1, 0, 0]),
          (0.5, [1, 1, 0, 0, 1, 1, 0, 0]),
          (0.75, [1, 0, 1, 0, 1, 1, 0, 0]),
          (1.0, [1, 0, 1, 0, 1, 1, 0, 0]),
      ]
  )
  def testRigLUpdateMask(self, drop_fraction, expected_mask):
    expected_mask = jnp.array(expected_mask, dtype=jnp.uint8)
    sparsity_fn = functools.partial(
        sparsity_distributions.uniform, sparsity=0.5
    )
    updater = sparse_trainers.RigL(
        sparsity_distribution_fn=sparsity_fn, is_debug=True
    )
    old_mask = jnp.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=jnp.uint8)
    param = jnp.array([-9, 1, 4, -1, 5, 8, -2, 3], dtype=jnp.float32)

    drop_score = param
    grow_score = jnp.abs(param)
    updated_mask = updater._update_masks(
        old_mask, drop_score, grow_score, drop_fraction=drop_fraction
    )
    self.assertSequenceAlmostEqual(expected_mask, updated_mask)

  @parameterized.product(
      updater=[sparse_trainers.RigL, sparse_trainers.SET],
      sparsity=[0.0, 0.5, 0.95],
      shape=[(1,), (5,), (4, 5), (2, 9, 1), (2, 2, 4, 4)],
  )
  def testDynamicSparseUpdateStateConstantSparsity(
      self, updater, sparsity, shape
  ):
    dist_fn = functools.partial(
        sparsity_distributions.uniform,
        filter_fn=lambda *_: True,
        sparsity=sparsity,
    )
    updater = updater(sparsity_distribution_fn=dist_fn, skip_gradients=True)
    # Ensure weight diverge from zero
    params = {'w': jax.random.normal(jax.random.PRNGKey(0), shape)}
    optimizer = updater.wrap_optax(optax.sgd(0.1))
    opt_state = optimizer.init(params)
    grad = {'w': jax.random.normal(jax.random.PRNGKey(8), shape)}
    old_masks = opt_state.masks
    opt_state = updater.update_state(opt_state, params, grad)
    # Check masks have same sparsity.
    self.assertEqual(jnp.sum(old_masks['w']), jnp.sum(opt_state.masks['w']))

  @parameterized.product(
      shape=[(5,), (4, 5), (2, 9, 1), (2, 2, 4, 4)], is_adam=[False, True]
  )
  def testDynamicSparseUpdateStateInnerReset(self, shape, is_adam):
    dist_fn = functools.partial(
        sparsity_distributions.uniform, filter_fn=lambda *_: True, sparsity=0.5
    )
    updater = sparse_trainers.RigL(sparsity_distribution_fn=dist_fn)
    # Ensure weight diverge from zero
    params = {'w': jax.random.normal(jax.random.PRNGKey(0), shape)}
    if is_adam:
      optimizer = updater.wrap_optax(optax.adam(0.001))
      acc_names = ['mu', 'nu']
    else:
      optimizer = updater.wrap_optax(optax.sgd(0.001, momentum=0.9))
      acc_names = ['trace']
    opt_state = optimizer.init(params)
    # Set trace to ones.
    acc_state = opt_state.inner_state[0]
    new_values = {
        k: jax.tree_map(jnp.ones_like, getattr(acc_state, k)) for k in acc_names
    }
    inner_state = (acc_state._replace(**new_values), *opt_state.inner_state[1:])
    opt_state = opt_state._replace(inner_state=inner_state)

    grad = {'w': jax.random.normal(jax.random.PRNGKey(8), shape)}
    old_masks = opt_state.masks
    opt_state = updater.update_state(opt_state, params, grad)
    # Check new connections have 0 trace
    new_connections = (old_masks['w'] == 0) & (opt_state.masks['w'] == 1)
    print(new_connections)
    for acc_key in acc_names:
      new_acc = getattr(opt_state.inner_state[0], acc_key)['w']
      self.assertEqual(jnp.sum(new_acc[new_connections]), 0.0)
      self.assertNotEqual(jnp.sum(new_acc[~new_connections]), 0.0)

  def testRestartUsingMask(self):
    target_tree = {'a': [-9, 1, 4, -1, 5, 8, -2, 3]}
    masks = {'a': [0, 1, 0, 1, 0, 1, 1, 0]}
    expected = {'a': [-9, 0, 4, 0, 5, 0, 0, 3]}
    result = sparse_trainers._restart_using_mask(target_tree, masks)
    chex.assert_trees_all_close(result, expected)


if __name__ == '__main__':
  absltest.main()
