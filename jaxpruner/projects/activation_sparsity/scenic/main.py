"""Main file for scenic training."""

from absl import flags
from absl import logging
from clu import metric_writers
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from jaxpruner.projects.activation_sparsity.scenic import activation_sparsity
from jaxpruner.projects.activation_sparsity.scenic import classification_trainer
import ml_collections
from scenic import app
from scenic.model_lib import models
from scenic.train_lib import train_utils


FLAGS = flags.FLAGS


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Main function for Scenic."""
  data_rng, rng = jax.random.split(rng)

  if config.checkpoint:
    # When restoring from a checkpoint, change the dataset seed to ensure that
    # the example order is new. With deterministic data, this ensures enough
    # randomization and in the future with deterministic data + random access,
    # we can feed the global step to the dataset loader to always continue
    # reading the rest of the data if we resume a job that was interrupted.
    checkpoint_path = checkpoints.latest_checkpoint(workdir)
    logging.info('CHECKPOINT PATH: %s', checkpoint_path)
    if checkpoint_path is not None:
      global_step = train_utils.checkpoint_path_step(checkpoint_path) or 0
      logging.info('Folding global_step %s into dataset seed.', global_step)
      data_rng = jax.random.fold_in(data_rng, global_step)

  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address
  )
  c_manager = activation_sparsity.get_activation_context(
      config.sparsity_config.activation_sparsity
  )
  with c_manager:
    model_cls = models.get_model_cls(config.model_name)
    classification_trainer.train(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer,
    )


if __name__ == '__main__':
  app.run(main=main)
