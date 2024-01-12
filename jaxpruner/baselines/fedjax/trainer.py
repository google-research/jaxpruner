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

r"""Trainer for Federated EMNIST with pruning.

This file demonstrates how to easily experiment with various pruning methods on
federated training of EMNIST using FedJAX with the JaxPruner library.

Pruning is conducted on the server, and is easily added to federated training by
  * wrapping the existing server optax optimizer with a JaxPruner updater
  * applying a post gradient update step on the server model

This code is forked from FedJAX's fed_avg example:
https://github.com/google/fedjax/blob/main/examples/fed_avg.py

And the implementation is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

The two necessary changes to integrating JaxPruner with the standard FedJAX
fed_avg implementation are called out in the comments with `CHANGE`.

"""

import csv
import os
from typing import Any, Callable, Mapping, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import fedjax
import jax
import jax.numpy as jnp
import jaxpruner
import ml_collections
import optax
from tensorflow.io import gfile


ClientId = bytes
Grads = fedjax.Params

_ALGORITHM = flags.DEFINE_enum(
    'algorithm',
    'no_prune',
    jaxpruner.api.ALGORITHMS,
    'Which pruning algorithm to apply.',
)

_DIST_TYPE = flags.DEFINE_enum(
    'dist_type',
    'uniform',
    ['uniform', 'erk'],
    'Which sparisty distribution to use for pruning.',
)

_SPARSITY = flags.DEFINE_float('sparsity', 0.8, 'Target sparsity for pruning.')

_UPDATE_START_STEP = flags.DEFINE_integer(
    'update_start_step', 250, 'When to start pruning.'
)

_RANDOM_SEED = flags.DEFINE_integer('random_seed', 0, 'Random seed.')

_WORKDIR = flags.DEFINE_string(
    'workdir',
    '/tmp/jaxpruner/fedjax',
    'Root directory for writing experiment output.',
)

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name', None, 'Name of specific job/trainer config.'
)


def get_sparsity_metrics(params):
  sparsity_summary_dict = jaxpruner.utils.summarize_sparsity(
      param_tree=params, only_total_sparsity=False
  )
  return sparsity_summary_dict


@fedjax.dataclass
class ServerState:
  """State of server passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
  """

  params: fedjax.Params
  opt_state: fedjax.OptState


def federated_averaging(
    grad_fn,
    client_optimizer,
    server_optimizer,
    server_updater,  # CHANGE: include JaxPruner updater
    client_batch_hparams,
):
  """Builds the basic implementation of federated averaging."""

  def init(params):
    opt_state = server_optimizer.init(params)
    return ServerState(params, opt_state)

  def apply(
      server_state,
      clients,
  ):
    client_diagnostics = {}
    # We use a list here for clarity, but we strongly recommend avoiding loading
    # all client outputs into memory since the outputs can be quite large
    # depending on the size of the model.
    client_delta_params_weights = []
    for client_id, client_dataset, client_rng in clients:
      delta_params = client_update(
          server_state.params, client_dataset, client_rng
      )
      client_delta_params_weights.append((delta_params, len(client_dataset)))
      # We record the l2 norm of client updates as an example, but it is not
      # required for the algorithm.
      client_diagnostics[client_id] = {
          'delta_l2_norm': fedjax.tree_util.tree_l2_norm(delta_params)
      }
    mean_delta_params = fedjax.tree_util.tree_mean(client_delta_params_weights)
    server_state = server_update(server_state, mean_delta_params)
    return server_state, client_diagnostics

  def client_update(server_params, client_dataset, client_rng):
    params = server_params
    opt_state = client_optimizer.init(params)
    for batch in client_dataset.shuffle_repeat_batch(client_batch_hparams):
      client_rng, use_rng = jax.random.split(client_rng)
      grads = grad_fn(params, batch, use_rng)  # pytype: disable=wrong-arg-types  # jax-ndarray
      opt_state, params = client_optimizer.apply(grads, opt_state, params)
    delta_params = jax.tree_util.tree_map(
        lambda a, b: a - b, server_params, params
    )
    return delta_params

  def server_update(server_state, mean_delta_params):
    opt_state, params = server_optimizer.apply(
        mean_delta_params, server_state.opt_state, server_state.params
    )
    # CHANGE: apply the JaxPruner updater's post gradient update procedure
    # to the server model.
    updated_params = server_updater.post_gradient_update(params, opt_state)
    return ServerState(updated_params, opt_state)

  return fedjax.FederatedAlgorithm(init, apply)


class _MetricsLogger:
  """Logger for TensorBoard."""

  def __init__(self, root_dir):
    self.logger = fedjax.training.Logger(root_dir=root_dir)

  def log(self, eval_metrics, round_num=0):
    """Logs eval metrics to TensorBoard."""
    for eval_name, metrics in eval_metrics.items():
      for metric_name, metric_value in metrics.items():
        self.logger.log(f'{eval_name}', metric_name, metric_value, round_num)


def _aggregate_client_diagnostics(client_diagnostics):
  """Aggregates diagnostics across clients (uniform average)."""
  aggregate_diagnostics = {}
  for diagnostics in client_diagnostics.values():
    for k, v in diagnostics.items():
      if k not in aggregate_diagnostics:
        aggregate_diagnostics[k] = []
      aggregate_diagnostics[k].append(v)
  for k, v in aggregate_diagnostics.items():
    aggregate_diagnostics[k] = jnp.mean(jnp.array(v))
  return aggregate_diagnostics


def train(
    config,
    model,
    train_fd,
    test_fd,
    metrics_logger,
    workdir,
):
  """Training loop for federated EMNIST classification."""
  server_updater = jaxpruner.create_updater_from_config(config.sparsity_config)
  server_gradient_transform = optax.adam(
      learning_rate=config.server_optimizer.learning_rate,
      b1=config.server_optimizer.b1,
      b2=config.server_optimizer.b2,
      eps=config.server_optimizer.eps,
  )
  server_gradient_transform = server_updater.wrap_optax(
      server_gradient_transform
  )

  # Scalar loss function with model parameters, batch of examples, and seed
  # PRNGKey as input.
  def loss(params, batch, rng):
    # `rng` used with `apply_for_train` to apply dropout during training.
    preds = model.apply_for_train(params, batch, rng)
    # Per example loss of shape [batch_size].
    example_loss = model.train_loss(batch, preds)
    return jnp.mean(example_loss)

  # Gradient function of `loss` w.r.t. to model `params` (jitted for speed).
  grad_fn = jax.jit(jax.grad(loss))

  # Create federated averaging algorithm, with server-side pruning.
  client_optimizer = fedjax.optimizers.sgd(
      learning_rate=config.client_optimizer.learning_rate
  )
  server_optimizer = fedjax.optimizers.create_optimizer_from_optax(
      server_gradient_transform  # note this includes the pruning algorithm
  )

  # Hyperparameters for client local traing dataset preparation.
  client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
      batch_size=config.train_batch_size
  )
  algorithm = federated_averaging(
      grad_fn,
      client_optimizer,
      server_optimizer,
      server_updater,
      client_batch_hparams,
  )

  # Initialize model parameters and algorithm server state.
  init_params = model.init(jax.random.PRNGKey(config.random_seed))
  server_state = algorithm.init(init_params)

  # Train and eval loop.
  train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
      fd=train_fd, num_clients=config.num_clients, seed=config.random_seed
  )
  for round_num in range(1, config.num_rounds + 1):
    # Sample clients without replacement for training.
    clients = train_client_sampler.sample()
    # Run one round of training on sampled clients.
    server_state, client_diagnostics = algorithm.apply(server_state, clients)
    diagnostics = _aggregate_client_diagnostics(client_diagnostics)
    logging.info(
        '[round %d] client_diagnostics = %s', round_num, client_diagnostics
    )
    logging.info('[round %d] diagnostics = %s', round_num, diagnostics)

    if round_num % 10 == 0:
      # Periodically evaluate the trained server model parameters.
      # Read and combine clients' train and test datasets for evaluation.
      client_ids = [cid for cid, _, _ in clients]
      train_eval_datasets = [cds for _, cds in train_fd.get_clients(client_ids)]
      test_eval_datasets = [cds for _, cds in test_fd.get_clients(client_ids)]
      train_eval_batches = fedjax.padded_batch_client_datasets(
          train_eval_datasets, batch_size=config.eval_batch_size
      )
      test_eval_batches = fedjax.padded_batch_client_datasets(
          test_eval_datasets, batch_size=config.eval_batch_size
      )

      # Run evaluation metrics defined in `model.eval_metrics`.
      train_metrics = fedjax.evaluate_model(
          model,
          server_state.params,  # pytype: disable=wrong-arg-types  # jax-ndarray
          train_eval_batches,
      )
      test_metrics = fedjax.evaluate_model(
          model,
          server_state.params,  # pytype: disable=wrong-arg-types  # jax-ndarray
          test_eval_batches,
      )
      sparsity_metrics = get_sparsity_metrics(server_state.params)
      metrics = {
          'train': train_metrics,
          'test': test_metrics,
          'sparsity': sparsity_metrics,
      }
      metrics_logger.log(metrics, round_num)

      if diagnostics is not None:
        for dn, dv in diagnostics.items():
          metrics_logger.logger.log('.', dn, dv, round_num)

  # Save final trained model parameters to file.
  fedjax.serialization.save_state(
      server_state.params, os.path.join(workdir, 'params')
  )

  full_eval_dataset = fedjax.padded_batch_federated_data(
      test_fd, batch_size=config.eval_batch_size
  )
  final_metrics = {
      'final': fedjax.evaluate_model(  # pytype: disable=wrong-arg-types  # jax-ndarray
          model, server_state.params, full_eval_dataset
      )
  }
  metrics_logger.log(final_metrics, round_num=round_num + 1)
  with gfile.Open(workdir + '/metrics.csv', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    metrics_to_save = list(final_metrics['final'].items()) + list(
        metrics['sparsity'].items()
    )
    csv_writer.writerows(
        [[k for k, _ in metrics_to_save], [v for _, v in metrics_to_save]]
    )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError(
        'Expected no command-line arguments, got: {}'.format(argv)
    )

  config = ml_collections.ConfigDict()
  config.sparsity_config = ml_collections.ConfigDict()
  config.sparsity_config.algorithm = _ALGORITHM.value
  config.sparsity_config.dist_type = _DIST_TYPE.value
  config.sparsity_config.update_freq = 10
  config.sparsity_config.update_start_step = _UPDATE_START_STEP.value
  config.sparsity_config.update_end_step = 750
  config.sparsity_config.sparsity = _SPARSITY.value

  config.num_clients = 50
  config.num_rounds = 1000
  config.train_batch_size = 32
  config.eval_batch_size = 64
  config.random_seed = _RANDOM_SEED.value

  config.server_optimizer = ml_collections.ConfigDict()
  config.server_optimizer.learning_rate = 10 ** (-2.5)
  config.server_optimizer.b1 = 0.9
  config.server_optimizer.b2 = 0.999
  config.server_optimizer.eps = 10 ** (-4)

  config.client_optimizer = ml_collections.ConfigDict()
  config.client_optimizer.learning_rate = 10 ** (-1.5)

  # Uses cpu only for data loading to save GPU/TPU memory
  fedjax.training.set_tf_cpu_only()
  # Load train and test federated data for EMNIST.
  train_fd, test_fd = fedjax.datasets.emnist.load_data(
      only_digits=False, mode='tff_sstable'
  )

  # Create CNN model with dropout.
  model = fedjax.models.emnist.create_conv_model(only_digits=False)

  # Metrics
  workdir = _WORKDIR.value
  if _EXPERIMENT_NAME.value is not None:
    workdir = os.path.join(workdir, _EXPERIMENT_NAME.value)
  gfile.MakeDirs(workdir)
  metrics_logger = _MetricsLogger(workdir)

  train(config, model, train_fd, test_fd, metrics_logger, workdir)


if __name__ == '__main__':
  app.run(main)
