"""Tests for jaxpruner extensions."""


from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jaxpruner
from jaxpruner.projects.bigsparse.jaxpruner_extensions import acdc
from jaxpruner.projects.bigsparse.jaxpruner_extensions import adam_fisher
import ml_collections
import optax


class ExtensionsTest(parameterized.TestCase, absltest.TestCase):
  """Tests for jaxpruner extensions."""

  def testAdamBasedFisherPruning(self):
    """Test the 'adam-fisher' algorithm."""

    adam_fisher.add_to_jaxpruner()
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.dist_type = 'uniform'
    sparsity_config.sparsity = 0.5
    sparsity_config.algorithm = 'adam-fisher'
    sparsity_config.reset_adam = True
    sparsity_config.skip_gradients = True
    sparsity_config.update_start_step = 0
    sparsity_config.update_end_step = 0
    updater = jaxpruner.create_updater_from_config(sparsity_config)

    tx = optax.adamw(1e-3)
    tx = updater.wrap_optax(tx)
    params = {'w': jnp.array([0.1, 0.9, 0.2, 0.5])}
    state = tx.init(params)
    grads = {'w': jnp.array([5, 1, 3, 0.5])}
    adamstate = adam_fisher.find_adamstate(state.inner_state)
    adamstate = adamstate._replace(mu=grads)
    adamstate = adamstate._replace(nu=jax.tree_map(lambda g: g**2, grads))
    state = state._replace(
        inner_state=adam_fisher.replace_adamstate(state.inner_state, adamstate)
    )

    # Test score calculation
    expected = {'w': params['w'] ** 2 * grads['w'] ** 2}
    chex.assert_trees_all_close(
        updater.calculate_scores(
            params=params, sparse_state=state, grads=grads
        ),
        expected,
    )

    # Test state resetting
    _, state = tx.update(grads, state, params)
    adamstate = adam_fisher.find_adamstate(state.inner_state)
    state = state._replace(
        inner_state=adam_fisher.replace_adamstate(state.inner_state, adamstate)
    )
    chex.assert_trees_all_close(
        adamstate.mu, jax.tree_map(jnp.zeros_like, params)
    )
    chex.assert_trees_all_close(
        adamstate.nu, jax.tree_map(jnp.zeros_like, params)
    )

  def testACDC(self):
    """Test the 'acdc' aalgorithm."""

    acdc.add_to_jaxpruner()
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.dist_type = 'uniform'
    sparsity_config.sparsity = 0.5
    sparsity_config.algorithm = 'acdc'
    sparsity_config.init_dense_steps_end = 1
    sparsity_config.final_sparse_steps_start = 5
    sparsity_config.cycle_sparse_steps = 1
    sparsity_config.cycle_dense_steps = 1
    updater = jaxpruner.create_updater_from_config(sparsity_config)

    tx = optax.adamw(1e-3)
    tx = updater.wrap_optax(tx)
    params = {'w': jnp.array([[0.1, 0.9], [0.2, 0.5]])}
    state = tx.init(params)
    grads = {'w': jnp.array([[5.0, 1.0], [3.0, 0.5]])}

    _, state = tx.update(grads, state, params)
    chex.assert_trees_all_close(jnp.mean(state.masks['w']), 1.0)
    for _ in range(2):
      _, state = tx.update(grads, state, params)
      chex.assert_trees_all_close(jnp.mean(state.masks['w']), 0.5)
      _, state = tx.update(grads, state, params)
      chex.assert_trees_all_close(jnp.mean(state.masks['w']), 1.0)
    _, state = tx.update(grads, state, params)
    chex.assert_trees_all_close(jnp.mean(state.masks['w']), 0.5)


if __name__ == '__main__':
  absltest.main()
