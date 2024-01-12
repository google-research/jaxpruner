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

"""Tests for utility functions in sparsity_distributions.py."""
from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax.numpy as jnp
from jaxpruner import sparsity_distributions
import numpy as np


class SparsityDistributionTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(
      ({'a': np.zeros((2, 4))}, {'a': 0.5}),
      ([np.zeros((2, 4)), np.ones((4,))], [0.5, None]),
      (
          {
              'a': np.zeros((2, 4)),
              'b': {'c': np.zeros((3, 6, 7)), 'd': np.ones((7, 4))},
          },
          {
              'a': 0.5,
              'b': {'c': 0.5, 'd': 0.5},
          },
      ),
      (
          {
              'a': np.zeros((2, 4)),
              'b': {'c': np.zeros((3, 6, 7)), 'd': np.ones((4,))},
          },
          {
              'a': 0.5,
              'b': {'c': 0.5, 'd': None},
          },
      ),
  )
  def testUniformSparsityMapGenerator(self, params, expected_result):
    self.assertEqual(
        sparsity_distributions.uniform(params, 0.5),
        expected_result,
    )

  def testUniformSparsityMapGeneratorWithCustomMap(self):
    param_tree = {
        'a': np.zeros((2, 4)),
        'b': {'c': np.zeros((3, 6, 7)), 'd': np.ones((7, 4))},
    }
    custom_map = {('b', 'd'): 0.1}
    result = sparsity_distributions.uniform(
        param_tree, 0.5, custom_sparsity_map=custom_map
    )

    self.assertEqual(result['b']['d'], 0.1)
    self.assertEqual(result['b']['c'], 0.5)
    self.assertEqual(result['a'], 0.5)

  @parameterized.parameters(
      ({'a': np.zeros((2, 4))}, {'a': 0.5}),
      (
          {
              'a': np.zeros((2, 4)),
              'b': {'c': np.zeros((3, 6, 7)), 'd': np.ones((7, 4))},
          },
          {
              'a': 0.0,
              'b': {'c': 0.64, 'd': 0.0},
          },
      ),
      (
          {
              'a': np.zeros((2, 4)),
              'b': {'c': np.zeros((3, 6, 7)), 'd': np.ones((4,))},
          },
          {
              'a': 0.0,
              'b': {
                  'c': 0.53,
                  'd': None,  # ndim <= 1 variable is not sparsified in default.
              },
          },
      ),
  )
  def testERKSparsityMapGenerator(self, params, expected_result):
    result = sparsity_distributions.erk(params, 0.5)
    flat_result = flax.traverse_util.flatten_dict(result)
    flat_expected_result = flax.traverse_util.flatten_dict(expected_result)
    self.assertSameElements(flat_result, flat_expected_result)
    for key, value in flat_result.items():
      if value is None:
        self.assertIsNone(flat_expected_result[key])
      else:
        self.assertAlmostEqual(value, flat_expected_result[key], places=2)

  def testERKSparsityMapGeneratorWithCustomMap(self):
    param_tree = {
        'a': np.zeros((2, 4)),
        'b': {'c': np.zeros((3, 6, 7)), 'd': np.ones((7, 4))},
    }
    custom_map = {('b', 'd'): 0.1}
    result = sparsity_distributions.erk(
        param_tree, 0.5, custom_sparsity_map=custom_map
    )

    self.assertEqual(result['b']['d'], 0.1)
    self.assertAlmostEqual(result['b']['c'], 0.53, places=2)

  def testERKSparsityMapGeneratorJaxArray(self):
    # Can't use parameterized test here since jax arrays can be created after
    # main function is called.
    with self.assertRaisesRegex(
        ValueError,
        'Single parameter is provided. Please provide a paramater tree.',
    ):
      sparsity_distributions.erk(jnp.ones((2, 2)), 0.5)

  def testERKSparsityMapGeneratorNumpyArray(self):
    with self.assertRaisesRegex(
        ValueError,
        'Single parameter is provided. Please provide a paramater tree.',
    ):
      sparsity_distributions.erk(np.ones((2, 2)), 0.5)

  @parameterized.parameters(
      (({'a': (2, 4)}, 0.5, True, 1.0), {'a': 0.5}),
      (
          ({'a': (2, 4), 'b': (3, 6), 'c': (2, 4, 6)}, 0.5, True, 1.0),
          {'a': 0.0, 'b': 0.31, 'c': 0.655},
      ),
      (
          ({'a': (2, 4), 'b': (3, 6), 'c': (2, 4, 6)}, 0.5, False, 1.0),
          {'a': 0.21, 'b': 0.47, 'c': 0.56},
      ),
      (
          ({'a': (2, 4), 'b': (3, 6), 'c': (2, 4, 6)}, 0.8, True, 1.0),
          {'a': 0.556, 'b': 0.70, 'c': 0.85},
      ),
      (
          ({'a': (2, 4), 'b': (3, 6), 'c': (2, 4, 6)}, 0.5, True, 2.0),
          {'a': 0.0, 'b': 0.033, 'c': 0.758},
      ),
  )
  def testGetSparsityERKSucceed(self, inputs, expected_result):
    var_shape_dict, default_sparsity, include_kernel, erk_power_scale = inputs
    result = sparsity_distributions.get_sparsities_erdos_renyi(
        var_shape_dict, default_sparsity, None, include_kernel, erk_power_scale
    )
    self.assertSameElements(result, expected_result)
    for key, value in result.items():
      self.assertAlmostEqual(value, expected_result[key], places=2)

  def testGetSparsityERKOutOfRangeDefaultSparsity(self):
    with self.assertRaisesRegex(ValueError, 'Default sparsity'):
      sparsity_distributions.get_sparsities_erdos_renyi({'a': (1, 1)}, -1)

  def testGetSparsityERKEmptyVarShapeDict(self):
    with self.assertRaisesRegex(ValueError, 'Variable shape dictionary'):
      sparsity_distributions.get_sparsities_erdos_renyi({}, 0.5)

  def testGetSparsityERKFollowsCustomSparsityMap(self):
    var_shape_dict = {'a': (2, 4), 'b': (3, 7, 6)}
    result = sparsity_distributions.get_sparsities_erdos_renyi(
        var_shape_dict, 0.1, {'a': 0.7, 'c': 0.1}
    )
    self.assertEqual(result['a'], 0.7)
    # Custom map didn't add a new variable not in var_shape_dict.
    self.assertNotIn('c', result)


if __name__ == '__main__':
  absltest.main()
