"""Tests for activation sparsity context."""
from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxpruner import sparsity_types
from jaxpruner.projects.activation_sparsity.scenic import activation_sparsity


class ActivationSparsityTest(parameterized.TestCase, absltest.TestCase):

  def testActivationWrapper(self):
    preactivation = jax.random.uniform(jax.random.PRNGKey(8), (4, 4)) + 0.5
    self.assertEqual(jnp.sum(nn.gelu(preactivation) == 0), 0)
    self.assertEqual(jnp.sum(nn.relu(preactivation) == 0), 0)
    self.assertEqual(jnp.sum(nn.leaky_relu(preactivation) == 0), 0)
    with activation_sparsity.add_activation_sparsity(
        sparsity_types.NByM(1, 4), allowlist=('relu', 'gelu')
    ):
      self.assertEqual(jnp.sum(nn.gelu(preactivation) == 0), 12)
      self.assertEqual(jnp.sum(nn.relu(preactivation) == 0), 12)
      self.assertEqual(jnp.sum(nn.leaky_relu(preactivation) == 0), 0)

  def testLayerWrapperSingle(self):
    rng = jax.random.PRNGKey(8)
    preactivation = jax.random.uniform(rng, (4, 4)) + 0.5
    layer = nn.Dense(5)
    params = layer.init(rng, preactivation)
    self.assertEqual(jnp.sum(layer.apply(params, preactivation) == 0), 0)
    with activation_sparsity.add_activation_sparsity_layers(
        lambda *_: (sparsity_types.NByM(1, 5), None)
    ):
      layer = nn.Dense(5)
      self.assertEqual(jnp.sum(layer.apply(params, preactivation) == 0), 16)

  def testLayerWrapperNetwork(self):
    # All kernel paramaters in a laer have 25 weights.
    class LeNet(nn.Module):
      """A simple CNN model."""

      @nn.compact
      def __call__(self, x):
        x1 = nn.Conv(name='conv1', features=5, kernel_size=(2, 2))(x)
        x = nn.avg_pool(x1, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1)) + 1  # to make it >0.
        x2 = nn.Dense(name='dense1', features=5)(x)
        x = x2 + 1  # to make it >0.
        x3 = nn.Dense(name='dense2', features=5)(x)
        return (x1, x2, x3)

    model = LeNet()
    rng = jax.random.PRNGKey(8)
    preactivation = jax.random.uniform(rng, (1, 2, 2, 5)) + 0.5

    params = model.init(rng, preactivation)
    # Make params all positive so that activations stay positive.
    params = jax.tree_map(jnp.abs, params)
    outputs = model.apply(params, preactivation)
    for out in outputs:
      self.assertEqual(jnp.sum(out == 0), 0)

    # Activation sparsity after dense layers.
    with activation_sparsity.add_activation_sparsity_layers(
        lambda *_: (sparsity_types.NByM(1, 5), None), allowlist=('Dense',)
    ):
      model = LeNet()
      outputs = model.apply(params, preactivation)
      for out in outputs[:1]:
        self.assertEqual(jnp.sum(out == 0), 0)
      for out in outputs[1:]:
        self.assertEqual(jnp.sum(out == 0), 4)

    # Activation sparsity after convolutional layers.
    with activation_sparsity.add_activation_sparsity_layers(
        lambda *_: (sparsity_types.NByM(1, 5), None), allowlist=('Conv',)
    ):
      outputs = model.apply(params, preactivation)
      for out in outputs[:1]:
        self.assertEqual(jnp.sum(out == 0), 16)
      for out in outputs[1:]:
        self.assertEqual(jnp.sum(out == 0), 0)


if __name__ == '__main__':
  absltest.main()
