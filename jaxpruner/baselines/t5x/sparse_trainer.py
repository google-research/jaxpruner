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

"""SparseTrainer that inherits from Trainer.

To create a custom trainer, subclass `BaseTrainer` and implement
`_partitioned_train_step` and `_partitioned_eval_step` methods,
possibly by re-using the utility functions provided in this module.
"""

from typing import Optional, TYPE_CHECKING
import cached_property
import clu.metrics

import jax.numpy as jnp

import jaxpruner

from t5x import models
from t5x import train_state as train_state_lib
from t5x import trainer

BatchType = trainer.BatchType
FlaxMutables = trainer.FlaxMutables
Rng = trainer.Rng
MutableMetricMapType = trainer.MutableMetricMapType
PartitionSpec = trainer.PartitionSpec

if TYPE_CHECKING:  # See b/163639353
  cached_property = property
else:
  cached_property = cached_property.cached_property


def train_with_lr(
    train_state,
    batch,
    learning_rate,
    dropout_rng,
    model,
    num_microbatches,
    weight_metrics_computer = None,
    data_partition_spec = PartitionSpec("data")):
  """Main training function with LR schedule."""
  # pytype: disable=attribute-error
  sparsity_updater = model.optimizer_def.sparsity_updater
  # pytype: enable=attribute-error
  forward_params = sparsity_updater.pre_forward_update(
      train_state.params, train_state.param_states
  )
  forward_train_state = train_state.replace_params(forward_params)

  grad_accum, metrics, flax_mutables = trainer.accumulate_grads_microbatched(
      model,
      forward_train_state,
      batch,
      dropout_rng,
      num_microbatches,
      data_partition_spec,
  )
  new_train_state, metrics = trainer.apply_grads(
      train_state,
      grad_accum,
      metrics,
      learning_rate,
      weight_metrics_computer,
      other_state_variables={"flax_mutables": flax_mutables}
      if flax_mutables else None)
  sparsity_metrics = {
      k: clu.metrics.Average.from_model_output(v)  # pytype: disable=wrong-arg-types  # jnp-array
      for k, v in jaxpruner.summarize_sparsity(
          forward_params, only_total_sparsity=True
      ).items()
  }
  metrics.update(sparsity_metrics)
  return new_train_state, metrics


def eval_step(
    model,
    train_state,
    batch,
):
  """Default evaluation step."""
  # pytype: disable=attribute-error
  sparsity_updater = model.optimizer_def.sparsity_updater
  # pytype: enable=attribute-error
  forward_params = sparsity_updater.pre_forward_update(
      train_state.params, train_state.param_states
  )
  if not train_state.flax_mutables:
    _, metrics = model.eval_fn(forward_params, batch)  # pytype: disable=wrong-arg-types  # jax-ndarray
  else:
    # If the training state contains mutable variables, then we expect the
    # model to accept this extra arguments in the eval function.
    # pytype: disable=wrong-arg-count
    # pytype: disable=wrong-arg-types
    _, metrics = model.eval_fn(forward_params, batch, train_state.flax_mutables)
    # pytype: enable=wrong-arg-count
    # pytype: enable=wrong-arg-types
  return metrics


class SparseTrainer(trainer.Trainer):
  """Training loop with sparsity (pruning) and with optional microbatches."""

  @cached_property
  def _partitioned_train_step(self):
    def train_step(train_state, batch):
      return train_with_lr(
          train_state,
          batch,
          learning_rate=self._learning_rate_fn(train_state.step),
          dropout_rng=self._get_step_rng(train_state.step),  # pytype: disable=wrong-arg-types  # jax-ndarray
          model=self._model,
          num_microbatches=self._num_microbatches,
          weight_metrics_computer=self._weight_metrics_computer,
          data_partition_spec=self._partitioner.data_partition_spec,
      )

    return self._partitioner.partition(
        train_step,
        in_axis_resources=(self._train_state_axes,
                           self._partitioner.data_partition_spec),
        out_axis_resources=(self._train_state_axes, None),
        donate_argnums=(0,))

  @cached_property
  def _partitioned_eval_step(self):
    return self._partitioner.partition(
        lambda *args, **kwargs: eval_step(self._model, *args, **kwargs),
        in_axis_resources=(
            self._train_state_axes,
            self._partitioner.data_partition_spec,
        ),
        out_axis_resources=None,
    )
