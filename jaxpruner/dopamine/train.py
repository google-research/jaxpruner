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

# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The entry point for running a Dopamine agent.

"""

from absl import app
from absl import flags
from absl import logging

from dopamine.discrete_domains import run_experiment
import gin
from jaxpruner.dopamine import dqn_agent_pruner

import tensorflow.compat.v2 as tf


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')


FLAGS = flags.FLAGS




@gin.configurable
def create_agent_jaxpruner(
    sess, environment, summary_writer=None, debug_mode=False
):
  """Creates an agent.

  Args:
    sess: A `tf.compat.v1.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.
  """
  del sess
  if not debug_mode:
    summary_writer = None
  return dqn_agent_pruner.DqnAgentPruner(
      num_actions=environment.action_space.n, summary_writer=summary_writer
  )


@gin.configurable
def create_runner_jaxpruner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return run_experiment.Runner(base_dir, create_agent_jaxpruner)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return run_experiment.TrainRunner(base_dir, create_agent_jaxpruner)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_v2_behavior()

  base_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = create_runner_jaxpruner(base_dir)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
