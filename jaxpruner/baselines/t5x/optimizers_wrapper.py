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

"""T5X Optimizer Support.

Tools for wrapping Optax optimizers and handling SPMD annotations for use with
pjit.

Additional support for the legacy Adafactor implementation.
"""

from typing import Optional, Sequence

import jax

from jaxpruner import base_updater
import optax

from t5x import optimizers as t5x_optimizers


OptaxWrapper = t5x_optimizers.OptaxWrapper
OptimizerState = t5x_optimizers.OptimizerState
OptimizerDef = t5x_optimizers.OptimizerDef
OptaxStatePartitionRules = t5x_optimizers.OptaxStatePartitionRules
SparseState = base_updater.SparseState
BaseUpdater = base_updater.BaseUpdater


# TODO Push this to t5x
def jp_partition_fn(state, params_axes):
  return SparseState(
      inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
          state.inner_state, params_axes
      ),
      count=None,
      target_sparsities=jax.tree_map(lambda x: None, state.target_sparsities),
      masks=params_axes,
  )


def fac_partition_fn(state, params_axes):
  return optax.FactoredState(  # pytype: disable=wrong-arg-types  # numpy-scalars
      count=None,
      v_row=OptaxStatePartitionRules.derive_optax_logical_axes(
          state.v_row, params_axes
      ),
      v_col=OptaxStatePartitionRules.derive_optax_logical_axes(
          state.v_col, params_axes
      ),
      v=OptaxStatePartitionRules.derive_optax_logical_axes(
          state.v, params_axes
      ),
  )



OptaxStatePartitionRules._RULES[SparseState] = jp_partition_fn
OptaxStatePartitionRules._RULES[optax.FactoredState] = fac_partition_fn



class SparseOptaxWrapper(OptaxWrapper):
  """Wrapper to make optax optimizer compatible with T5X."""

  def __init__(
      self,
      optax_optimizer,
      sparsity_updater = None,
  ):
    """Initializer.

    Args:
      optax_optimizer: An optax optimizer.
      sparsity_updater: A jaxpruner sparsity/pruning updater
    """
    super().__init__(optax_optimizer=optax_optimizer)
    self.sparsity_updater = sparsity_updater
    if self.sparsity_updater:
      self.optax_optimizer = sparsity_updater.wrap_optax(self.optax_optimizer)

  def apply_gradient(self, hyper_params, params, state, grads):
    """Applies gradient.

    Args:
      hyper_params: Unused hyper parameters.
      params: PyTree of the parameters.
      state: A named tuple containing the state of the optimizer.
      grads: PyTree of the gradients for the parameters.

    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    del hyper_params

    updates, new_optax_state = self.optax_optimizer.update(
        grads, state.param_states, params
    )
    new_params = optax.apply_updates(params, updates)
    if self.sparsity_updater:
      new_params = self.sparsity_updater.post_gradient_update(  # pytype: disable=wrong-arg-types
          new_params, new_optax_state
      )
    return new_params, OptimizerState(
        step=state.step + 1, param_states=new_optax_state
    )


def chain(
    transformations,
):
  return optax.chain(*transformations)
