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

"""Straight Through Estimator based sparse training methods.

Reference: https://arxiv.org/abs/1308.3432
"""
import dataclasses

import chex
from jaxpruner import base_updater
from jaxpruner.algorithms import pruners


@dataclasses.dataclass
class SteMixin:
  """Implements Mixin for straight through estimator."""

  def pre_forward_update(
      self, params, opt_state
  ):
    # Update mask for every iterations
    sparsities = self.scheduler.get_sparsity_at_step(
        opt_state.target_sparsities, opt_state.count
    )
    scores = self.calculate_scores(params, sparse_state=opt_state)
    new_masks = self.create_masks(scores, sparsities)
    return self.apply_masks(params, new_masks, is_packed=False)

  def post_gradient_update(
      self, params, opt_state
  ):
    # Keep dense weights
    del opt_state
    return params


@dataclasses.dataclass
class SteMagnitudePruning(SteMixin, pruners.MagnitudePruning):
  """Magnitude pruner, which updates weight with straight through estimator."""

  pass


@dataclasses.dataclass
class SteRandomPruning(SteMixin, pruners.RandomPruning):
  """Random pruner, which updates weight with straight through estimator."""

  pass
