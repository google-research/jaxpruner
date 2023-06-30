"""Jaxpruner implementation for Adam-based Fisher-pruning."""


import dataclasses
import jax
import jax.numpy as jnp
import jaxpruner
import optax


def find_adamstate(state):
  """Recursively searches for an Adam state."""
  if isinstance(state, optax.TraceState):
    return None
  elif isinstance(state, optax.ScaleByAdamState):
    return state
  elif isinstance(state, (optax.EmptyState, optax.ScaleByScheduleState)):
    return None
  elif isinstance(state, optax.MaskedState):
    return find_adamstate(state.inner_state)
  elif isinstance(state, tuple):
    for s in state:
      s1 = find_adamstate(s)
      if s1 is not None:
        return s1
    return None
  else:
    raise ValueError('Unrecognized optimizer state of type: %s' % type(state))


def replace_adamstate(state, replacement):
  """Recursively searches for an Adam state."""
  if isinstance(state, optax.TraceState):
    return state
  elif isinstance(state, optax.ScaleByAdamState):
    return replacement
  elif isinstance(state, (optax.EmptyState, optax.ScaleByScheduleState)):
    return state
  elif isinstance(state, optax.MaskedState):
    return state._replace(
        inner_state=replace_adamstate(state.inner_state, replacement)
    )
  elif isinstance(state, tuple):
    return tuple(replace_adamstate(s, replacement) for s in state)
  else:
    raise ValueError('Unrecognized optimizer state of type: %s' % type(state))


@dataclasses.dataclass
class AdamBasedFisherPruning(jaxpruner.BaseUpdater):
  """Implements diagonal Fisher pruning reusing the Adam second moment.

  A diagonal version of https://arxiv.org/abs/2004.14340 implemented via Adam.
  """

  # Whether to reset the Adam state after pruning
  reset_adam: bool = False

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del grads
    adamstate = find_adamstate(sparse_state.inner_state)
    if adamstate is None:
      raise ValueError('No Adam state found.')
    scores = jax.tree_map(lambda p, nu: p**2 * nu, params, adamstate.nu)
    del sparse_state
    return scores

  def update_state(self, sparse_state, params, grads):
    sparse_state = super().update_state(sparse_state, params, grads)
    if self.reset_adam:
      adamstate = find_adamstate(sparse_state.inner_state)
      adamstate = adamstate._replace(
          mu=jax.tree_map(jnp.zeros_like, adamstate.mu),
          nu=jax.tree_map(jnp.zeros_like, adamstate.nu),
      )
      sparse_state = sparse_state._replace(
          inner_state=replace_adamstate(sparse_state.inner_state, adamstate)
      )
    return sparse_state


def add_to_jaxpruner():
  """Add the 'adam-fisher' algorithm to the jaxpruner."""
  jaxpruner.ALGORITHM_REGISTRY['adam-fisher'] = AdamBasedFisherPruning
  jaxpruner.ALGORITHMS = tuple(jaxpruner.ALGORITHM_REGISTRY.keys())
