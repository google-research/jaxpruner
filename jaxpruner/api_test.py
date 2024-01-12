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

"""Tests for api functions in api.py."""
import inspect
from absl.testing import absltest

import jaxpruner
from jaxpruner import api
from jaxpruner import sparsity_schedules
from jaxpruner import sparsity_types
import ml_collections


class ScenicConfigTest(absltest.TestCase):

  def testCreateUpdater(self):
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = 'magnitude'
    sparsity_config.update_freq = 10
    sparsity_config.update_start_step = 200
    sparsity_config.update_end_step = 1000
    sparsity_config.sparsity = 0.8
    sparsity_config.dist_type = 'erk'
    updater = api.create_updater_from_config(sparsity_config)

    self.assertIsInstance(updater, jaxpruner.MagnitudePruning)

    self.assertIsInstance(updater.sparsity_type, sparsity_types.Unstructured)

    self.assertIsInstance(
        updater.scheduler, sparsity_schedules.PolynomialSchedule
    )
    self.assertEqual(updater.scheduler.update_freq, sparsity_config.update_freq)
    self.assertEqual(
        updater.scheduler.update_start_step, sparsity_config.update_start_step
    )
    self.assertEqual(
        updater.scheduler.update_end_step, sparsity_config.update_end_step
    )

    func_args = inspect.signature(updater.sparsity_distribution_fn).parameters
    self.assertLen(func_args, 4)
    self.assertIn('param_tree', func_args)
    self.assertEqual(func_args['sparsity'].default, sparsity_config.sparsity)

  def testCreateUpdaterNoPruning(self):
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = 'no_prune'
    updater = api.create_updater_from_config(sparsity_config)

    self.assertIsInstance(updater, jaxpruner.NoPruning)

  def testCreateUpdaterErrorWithWrongDistFn(self):
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = 'magnitude'
    sparsity_config.dist_type = 'not_a_distribution_function'

    with self.assertRaisesRegex(ValueError, 'dist_type:'):
      api.create_updater_from_config(sparsity_config)

  def testCreateUpdaterErrorWithWrongType(self):
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = 'not_a_correct_algorithm'
    sparsity_config.dist_type = 'uniform'
    sparsity_config.sparsity = 0.9

    with self.assertRaisesRegex(ValueError, 'jaxpruner.all_algorithm_names()'):
      api.create_updater_from_config(sparsity_config)

  def testCreateUpdaterErrorWithWrongSparsityType(self):
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = 'magnitude'
    sparsity_config.dist_type = 'uniform'
    sparsity_config.sparsity = 0.4
    sparsity_config.sparsity_type = 'wrong_sparsity_type'

    with self.assertRaisesRegex(ValueError, 'Sparsity type'):
      api.create_updater_from_config(sparsity_config)

  def testCreateUpdaterNoUpdateScheduleWithNoStartStep(self):
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = 'magnitude'
    sparsity_config.dist_type = 'uniform'
    sparsity_config.sparsity = 0.9
    updater = api.create_updater_from_config(sparsity_config)

    self.assertIsInstance(
        updater.scheduler, sparsity_schedules.NoUpdateSchedule
    )


if __name__ == '__main__':
  absltest.main()
