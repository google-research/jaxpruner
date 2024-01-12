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

"""Tests for utility functions in sparsity_schedules.py."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jaxpruner import sparsity_schedules as schedules


class SparsityUtilsTest(parameterized.TestCase, absltest.TestCase):

  def testNoUpdateSchedule(self):
    target = {
        'a': 0.5,
        'b': {
            'c': None,
            'd': 0.5,
        },
    }
    scheduler = schedules.NoUpdateSchedule()
    self.assertEqual(
        scheduler.get_sparsity_at_step(
            target_sparsities=target,
            step=10,
        ),
        target,
    )
    self.assertFalse(scheduler.is_mask_update_iter(step=19))

  def testOneShotSchedule(self):
    target = {
        'a': 0.5,
        'b': {
            'c': None,
            'd': 0.5,
        },
    }
    scheduler = schedules.OneShotSchedule(target_step=20)
    self.assertEqual(
        scheduler.get_sparsity_at_step(
            target_sparsities=target,
            step=10,
        ),
        target,
    )
    self.assertTrue(scheduler.is_mask_update_iter(step=20))
    self.assertFalse(scheduler.is_mask_update_iter(step=19))

  def testPeriodicSchedule(self):
    scheduler = schedules.PeriodicSchedule(
        update_freq=5, update_start_step=0, update_end_step=20
    )

    self.assertTrue(scheduler.is_mask_update_iter(step=10))
    self.assertFalse(scheduler.is_mask_update_iter(step=9))
    self.assertFalse(scheduler.is_mask_update_iter(step=None))

  def testPeriodicScheduleIdentitySparsity(self):
    scheduler = schedules.PeriodicSchedule(
        update_freq=5, update_start_step=0, update_end_step=20
    )
    target = {'a': 0.5}
    c_sparsity = scheduler.get_sparsity_at_step(target, 9)

    self.assertEqual(c_sparsity, target)

  def testPolynomialDecaySchedule(self):
    target = {
        'a': 0.5,
        'b': {
            'c': None,
            'd': 0.5,
        },
    }
    expected = {
        'a': 0.4375,
        'b': {
            'c': None,
            'd': 0.4375,
        },
    }

    scheduler = schedules.PolynomialSchedule(
        update_freq=5, update_start_step=0, update_end_step=20
    )

    self.assertEqual(
        scheduler.get_sparsity_at_step(
            target_sparsities=target,
            step=9,
        ),
        expected,
    )

    jit_fn = jax.jit(scheduler.get_sparsity_at_step)
    self.assertEqual(jit_fn(target, 9), expected)

  def testPolynomialDecaySparsityTreeWithWrongEndStepValue(self):
    with self.assertRaisesRegex(ValueError, 'end_step'):
      _ = schedules.PolynomialSchedule(
          1, update_start_step=1, update_end_step=0
      ).get_sparsity_at_step({}, 0)


if __name__ == '__main__':
  absltest.main()
