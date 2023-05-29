# coding=utf-8
# Copyright 2023 Jaxpruner Authors.
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

"""Defines sparsity schedule classes for sparse training."""
import dataclasses
import logging

import chex
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class NoUpdateSchedule(object):
  """Implements no update scheduler."""

  def get_sparsity_at_step(
      self,
      target_sparsities,
      step,
  ):
    del step
    return target_sparsities

  def is_mask_update_iter(self, step):
    del step
    return False


@dataclasses.dataclass(frozen=True)
class OneShotSchedule(NoUpdateSchedule):
  """Implements one shot update schedule."""

  target_step: int

  def is_mask_update_iter(self, step):
    return step == self.target_step


@dataclasses.dataclass(frozen=True)
class PeriodicSchedule(NoUpdateSchedule):
  """Implements periodic update schedule."""

  update_freq: int
  update_start_step: int
  update_end_step: int

  def is_mask_update_iter(self, step):
    if step is None:
      return False
    is_update_step = jnp.logical_and(
        step % self.update_freq == 0, step >= self.update_start_step
    )
    if self.update_end_step:
      is_update_step = jnp.logical_and(
          is_update_step, step <= self.update_end_step
      )
    return is_update_step


@dataclasses.dataclass(frozen=True)
class PolynomialSchedule(PeriodicSchedule):
  """Extends periodic schedule with polynomial sparsity schedule."""

  power: int = 3

  def get_sparsity_at_step(
      self,
      target_sparsities,
      step,
  ):
    """Returns a tree of sparsities at a given step in polynomial decaying.

    Sparsity starts from 0 and increased gradually using a polynomial function
    between start and end steps such that target sparsities are matched.

    Args:
      target_sparsities: target sparsities for pruning.
      step: int, step of training.

    Returns:
      a tree of same shape as the self.mask where None leaves stay None.
    """
    decay_fn = _polynomial_decay_sparsity
    start = self.update_start_step
    end = self.update_end_step
    if end < start:
      raise ValueError(
          'The end_step cannot be smaller than the start_step. '
          f'Got end: {end}, start: {start}'
      )
    elif end == start:
      progress = 1.0
    else:
      length = jnp.array(end - start, dtype=float)
      progress = jnp.clip((step + 1 - start), 0, length).astype(float)
      progress = progress / length
    logging.info('Calculating new sparsity, %s, %s', step, progress)
    return jax.tree_map(
        lambda t: decay_fn(progress, t, self.power), target_sparsities
    )


def _polynomial_decay_sparsity(progress, target, power):
  """Calculate the desired sparsity with polynomial decaying function.

  Args:
    progress: A float value of training step progress between pruning start and
      end step. This value should be between 0 and 1.
    target: A float target sparsity level desired to reach at the end step.
    power: An int polynomial order decides convergence rate to the target
      sparsity.

  Returns:
    Calculated sparsity value at the current training step progress.
  """
  if target is None:
    # Unpruned nodes stay unpruned.
    return None
  else:
    sparsity = target * (1.0 - (1.0 - progress) ** power)
    target = sparsity
    return target
