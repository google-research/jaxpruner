"""Jaxpruner implementation for AC/DC sparse training."""


import dataclasses
import jax
import jax.numpy as jnp
import jaxpruner
import optax


def reset_optimizer(state):
  """Resets Adam and SGD state."""
  if isinstance(state, optax.TraceState):
    new_trace = jax.tree_map(jnp.zeros_like, state.trace)
    state = state._replace(new_trace)
  elif isinstance(state, optax.ScaleByAdamState):
    new_mu = jax.tree_map(jnp.zeros_like, state.mu)
    new_nu = jax.tree_map(jnp.zeros_like, state.nu)
    state = state._replace(mu=new_mu, nu=new_nu)
  elif isinstance(state, optax.FactoredState):  # support AdaFactor as well
    new_v_col = jax.tree_map(jnp.zeros_like, state.v_col)
    new_v_row = jax.tree_map(jnp.zeros_like, state.v_row)
    new_v = jax.tree_map(jnp.zeros_like, state.v)
    state = state._replace(v_col=new_v_col, v_row=new_v_row, v=new_v)
  elif isinstance(state, (optax.EmptyState, optax.ScaleByScheduleState)):
    return state
  elif isinstance(state, optax.MaskedState):
    state = state._replace(inner_state=reset_optimizer(state.inner_state))
  elif isinstance(state, tuple):
    # Note that optax state's are named tuples, and they would pass this test.
    # TODO find a way to check optax states.
    state = tuple(reset_optimizer(s) for s in state)
  else:
    raise ValueError('Unrecognized optimizer state of type: %s' % type(state))
  return state


@dataclasses.dataclass
class ACDC(jaxpruner.algorithms.MagnitudePruning):
  """Implements AC/DC pruning - https://arxiv.org/abs/2106.12379."""

  # Number of initial dense steps
  init_dense_steps_end: int = 0
  # Final sparse finetuning phase starts after this many steps
  final_sparse_steps_start: int = 0
  # Number of sparse steps in cycle
  cycle_sparse_steps: int = 0
  # Number of dense steps in cycle
  cycle_dense_steps: int = 0
  # Whether to reset the optimizer state on every phase flip
  reset_optimizer: bool = False

  def __post_init__(self):
    super().__post_init__()
    # we handle the entire scheduling in the ACDC `update_state()`
    self.scheduler = jaxpruner.sparsity_schedules.PeriodicSchedule(
        update_freq=1,
        update_start_step=0,
        update_end_step=2**31 - 1,  # max 32bit integer
    )

  def update_state(self, sparse_state, params, grads):
    step = sparse_state.count

    def cycle():
      cycle_steps = self.cycle_sparse_steps + self.cycle_dense_steps
      step_in_cycle = (step - self.init_dense_steps_end) % cycle_steps

      updated_sparse_state = sparse_state
      start_final_sparse = step == self.final_sparse_steps_start

      def sparse_start():
        scores = self.calculate_scores(
            params, sparse_state=updated_sparse_state, grads=grads
        )
        new_masks = self.create_masks(
            scores, updated_sparse_state.target_sparsities
        )
        if self.use_packed_masks:
          new_masks = jax.tree_map(jnp.packbits, new_masks)
        ret = updated_sparse_state._replace(masks=new_masks)
        return reset_optimizer(ret) if self.reset_optimizer else ret

      updated_sparse_state = jax.lax.cond(
          (step_in_cycle == 0) | start_final_sparse,
          sparse_start,
          lambda: updated_sparse_state,
      )

      def dense_start():
        new_masks = self.get_initial_masks(
            params, updated_sparse_state.target_sparsities
        )
        if self.use_packed_masks:
          new_masks = jax.tree_map(jnp.unpackbits, new_masks)
        ret = updated_sparse_state._replace(masks=new_masks)
        return reset_optimizer(ret) if self.reset_optimizer else ret

      updated_sparse_state = jax.lax.cond(
          (step_in_cycle == self.cycle_sparse_steps) & ~start_final_sparse,
          dense_start,
          lambda: updated_sparse_state,
      )

      return updated_sparse_state

    jax.debug.print('{x}, {y}', x=step, y=self.init_dense_steps_end)
    cycling = self.init_dense_steps_end <= step
    cycling &= step <= self.final_sparse_steps_start
    return jax.lax.cond(cycling, cycle, lambda: sparse_state)


def add_to_jaxpruner():
  """Add the 'acdc' algorithm to the jaxpruner."""
  jaxpruner.register_algorithm('acdc', ACDC)
