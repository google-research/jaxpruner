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

"""Implementation of a DQN+Prunner agent in JAX."""
import functools
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
from jaxpruner import utils
from jaxpruner.baselines.dopamine import sparse_util
import numpy as onp
import optax


@functools.partial(jax.jit, static_argnums=(0, 4, 11, 12))
def train(
    network_def,
    online_params,
    forward_params,
    target_params,
    optimizer,
    optimizer_state,
    states,
    actions,
    next_states,
    rewards,
    terminals,
    cumulative_gamma,
    loss_type,
):
  """Run the training step."""

  def loss_fn(params, target):
    def q_online(state):
      return network_def.apply(params, state)

    q_values = jax.vmap(q_online)(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    if loss_type == 'huber':
      return jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
    return jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

  def q_target(state):
    return network_def.apply(target_params, state)

  target = dqn_agent.target_q(
      q_target, next_states, rewards, terminals, cumulative_gamma
  )
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(forward_params, target)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params
  )
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss


@gin.configurable
class DqnAgentPruner(dqn_agent.JaxDQNAgent):
  """A JAX implementation of the DQN+Pruner agent."""

  def _build_networks_and_optimizer(self):
    self._rng, init_rng, updater_rng = jax.random.split(self._rng, num=3)
    self.online_params = self.network_def.init(init_rng, x=self.state)
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.updater = sparse_util.create_updater_from_config(rng_seed=updater_rng)
    self.post_gradient_update = jax.jit(self.updater.post_gradient_update)
    self.pre_forward_update = jax.jit(self.updater.pre_forward_update)
    self.optimizer = self.updater.wrap_optax(self.optimizer)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_params to target_network_params if training
    steps is a multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        states = self.preprocess_fn(self.replay_elements['state'])
        next_states = self.preprocess_fn(self.replay_elements['next_state'])
        forward_params = self.pre_forward_update(
            self.online_params, self.optimizer_state
        )
        self.optimizer_state, self.online_params, loss = train(
            self.network_def,
            self.online_params,
            forward_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            self.replay_elements['action'],
            next_states,
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self.cumulative_gamma,
            self._loss_type,
        )
        self.online_params = self.post_gradient_update(
            self.online_params, self.optimizer_state
        )
        total_sparsity = utils.summarize_sparsity(
            forward_params, only_total_sparsity=True
        )
        if (
            self.training_steps > 0
            and self.training_steps % self.summary_writing_frequency == 0
        ):
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                [
                    statistics_instance.StatisticsInstance(
                        'Loss',
                        onp.asarray(loss),
                        step=(self.training_steps // self.update_period),
                    ),
                    statistics_instance.StatisticsInstance(
                        'TotalSparsity',
                        onp.asarray(total_sparsity['_total_sparsity']),
                        step=(self.training_steps // self.update_period),
                    ),
                ],
                collector_allowlist=self._collector_allowlist,
            )
            self.collector_dispatcher.flush()
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
