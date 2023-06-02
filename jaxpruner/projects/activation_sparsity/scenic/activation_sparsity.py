"""Defines context manager for activation sparsity."""
import contextlib
import logging
from flax import linen as nn
from jaxpruner import mask_calculator
from jaxpruner import sparsity_types


def get_activation_context(sparsity_string, allowlist=('relu',)):
  """Returns activation context with sparsity type and level."""
  if not isinstance(sparsity_string, str):
    sparsity = float(sparsity_string)
    sparsity_type = sparsity_types.Unstructured
  elif sparsity_string.startswith('nm'):
    # example: nm_2,4
    n, m = sparsity_string.split('_')[1].strip().split(',')
    sparsity_type = sparsity_types.NByM(int(n), int(m))
    sparsity = None
  elif sparsity_string.startswith('block'):
    # example: block_4,4_0.8
    block_shape, sparsity = sparsity_string.split('_')[1:3]
    n, m = block_shape.strip().split(',')
    sparsity_type = sparsity_types.Block(block_shape=(int(n), int(m)))
    sparsity = float(sparsity)
  else:
    raise ValueError(
        f'Sparsity string for activation {sparsity_string} is not supported.'
    )
  logging.info(
      'Activation sparsity context: %s, sparsity: %f, allowlist: %s',
      sparsity_type,
      sparsity,
      allowlist,
  )
  return add_activation_sparsity(sparsity_type, sparsity, allowlist=allowlist)


@contextlib.contextmanager
def add_activation_sparsity(sparsity_type, sparsity=None, allowlist=('relu',)):
  """Context manager for replacing flax activations."""
  original_functions = {name: getattr(nn, name) for name in allowlist}
  topk_fn = mask_calculator.get_topk_fn(sparsity_type)
  try:
    for name, original_fn in original_functions.items():

      def wrapped_fn(x, fn=original_fn):
        y = fn(x)
        return topk_fn(y, sparsity) * y

      setattr(nn, name, wrapped_fn)
    yield
  finally:
    for name, original_fn in original_functions.items():
      setattr(nn, name, original_fn)


@contextlib.contextmanager
def add_activation_sparsity_layers(
    sparsity_fn,
    allowlist=('Dense',),
    score_fn=lambda x: x,
    getter_fn=lambda name: getattr(nn, name),
    setter_fn=lambda name, obj: setattr(nn, name, obj),
):
  """Context manager for replacing flax layers."""

  def get_class(name):
    base_layer = getter_fn(name)

    class CustomClass(base_layer):
      """Identity layer, convenient for giving a name to an array."""

      def __call__(self, inputs):
        output = super().__call__(inputs)
        path = self.scope.path
        sparsity_type, sparsity = sparsity_fn(path, output.shape)
        if sparsity_type is not None:
          topk_fn = mask_calculator.get_topk_fn(sparsity_type)
          topk_scores = score_fn(output)
          out = topk_fn(topk_scores, sparsity) * output
          return out
        else:
          return output

    return CustomClass

  original_layers = {name: getter_fn(name) for name in allowlist}
  try:
    for name, _ in original_layers.items():
      setter_fn(name, get_class(name))
    yield
  finally:
    for name, original_fn in original_layers.items():
      setter_fn(name, original_fn)
