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

"""Tests for utility functions in utils.py."""
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jaxpruner import utils


class UtilsTest(parameterized.TestCase, absltest.TestCase):

  def testSparsitySummary(self):
    param_tree = {
        'a': jnp.ones((2, 2)),
        'b': {
            'c': None,
            'd': jnp.array([1, 0, 1, 0.001]),
            'e': jnp.zeros([2, 1]),
        },
    }
    expected_summary = {
        '_total_sparsity': 0.3,
        '_nparams': 10.0,
        '_nparams_nnz': 7.0,
    }
    self.assertEqual(
        utils.summarize_sparsity(param_tree, only_total_sparsity=True),
        expected_summary,
    )

    expected_summary = {
        '_total_sparsity': 0.3,
        '_nparams': 10.0,
        '_nparams_nnz': 7.0,
        'a': jnp.array([0]),
        'b/c': None,
        'b/d': jnp.array([0.25]),
        'b/e': jnp.array([1.0]),
    }
    self.assertEqual(utils.summarize_sparsity(param_tree), expected_summary)

  def testInstersectionSummary(self):
    mask_tree = {
        'a': jnp.array([0, 0, 1, 1]),
        'b': {
            'c': None,
            'd': jnp.array([1, 0]),
            'e': jnp.array([1, 0]),
        },
    }
    second_mask_tree = {
        'a': jnp.array([0, 1, 0, 1]),
        'b': {
            'c': None,
            'd': jnp.array([1, 0]),
            'e': jnp.array([0, 1]),
        },
    }
    expected_summary = {'_total_intersection': 0.5}
    self.assertEqual(
        utils.summarize_intersection(
            mask_tree, second_mask_tree, only_total_intersection=True
        ),
        expected_summary,
    )

    expected_summary = {
        '_total_intersection': 0.5,
        'a': jnp.array([0.5]),
        'b/c': None,
        'b/d': jnp.array([1.0]),
        'b/e': jnp.array([0.0]),
    }
    self.assertEqual(
        utils.summarize_intersection(
            mask_tree,
            second_mask_tree,
        ),
        expected_summary,
    )


if __name__ == '__main__':
  absltest.main()
