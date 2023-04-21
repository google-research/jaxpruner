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

"""Sparsity distribution module with functions generates sparsity tree.

For Jax Pruner, users can allocate different desired sparsity for each different
parameter (weight), so the functions in this file help users to allocate the
sparsity to meet the overall target sparsity.

Note that all functions in this file must have the same input/output signatures.
"""
import logging
from typing import Callable, Mapping, Optional, Union, Tuple

import chex
import flax
import jax
import numpy as np

FilterFnType = Callable[[Tuple[str], chex.Array], bool]
CustomSparsityMapType = Mapping[Tuple[str], float]

KERNEL_FILTER_FN = lambda key, param: key[-1] == 'kernel'
NOT_DIM_ONE_FILTER_FN = lambda key, param: param.ndim > 1


def uniform(
    params,
    sparsity,
    filter_fn = NOT_DIM_ONE_FILTER_FN,
    custom_sparsity_map = None,
):
  """Uniformly distribute the target sparsity to variables.

  Args:
    params: paramater or tree of parameters.
    sparsity: float, in [0, 1). fraction of weights to be removed from pruned
      parameters (i.e. filter_fn(param) == True).
    filter_fn: used to decide whether to include the parameter in sparsity
      calculation. Default is a function that returns true whenever the key[-1]
      == 'kernel'.
    custom_sparsity_map: A dictionary with keys corresponding to the flattened
      keys of parameters (leaves) in a tree. This is to assign specific sparsity
      to given variables. Keys should match corresponding values at
      `flax.traverse_util.flatten_dict(param_tree)`.

  Returns:
    sparsity_tree: same shape as param_tree with float or None leaves.
      For leaves with filter_fn(param)==False, None values are used.
  """

  def _get_sparsity(key, param):
    if custom_sparsity_map and key in custom_sparsity_map:
      return custom_sparsity_map[key]
    elif filter_fn(key, param):
      return sparsity
    else:
      return None

  if isinstance(params, chex.Array):
    return _get_sparsity(('kernel',), params)

  elif isinstance(params, list):
    return jax.tree_map(lambda p: _get_sparsity('', p), params)

  flat_dict = flax.traverse_util.flatten_dict(params)
  res_dict = {}
  for k, p in flat_dict.items():
    res_dict[k] = _get_sparsity(k, p)

  return_val = flax.traverse_util.unflatten_dict(res_dict)
  if isinstance(params, flax.core.frozen_dict.FrozenDict):
    return_val = flax.core.freeze(return_val)
  return return_val


def erk(
    param_tree,
    sparsity,
    filter_fn = NOT_DIM_ONE_FILTER_FN,
    custom_sparsity_map = None,
):
  """Allocate the sparsity to variables according to erdos-renyl method.

  Args:
    param_tree: PyTree, of parameters.
    sparsity: float, in [0, 1). fraction of weights to be removed.
    filter_fn: used to decide whether to include the parameter in sparsity
      calculation. Default is a function that returns true whenever the key[-1]
      == 'kernel'.
    custom_sparsity_map: A dictionary with variable name in tree map as a key
      and sparsity value as a value. This is to assign specific sparsity to
      specific variables.

  Returns:
    sparsity_tree: same shape as param_tree with float or None leaves.
      For leaves with filter_fn(key, param)==False, None values are used.
  """
  # TODO: Add support for including excluded params in the calculations.
  if isinstance(param_tree, (chex.Array, chex.ArrayNumpy)):
    raise ValueError(
        'Single parameter is provided. Please provide a paramater tree.'
    )

  flat_dict = flax.traverse_util.flatten_dict(param_tree)
  filtered_shape_dict = {
      k: p.shape for k, p in flat_dict.items() if filter_fn(k, p)
  }

  sparsities = get_sparsities_erdos_renyi(
      filtered_shape_dict, sparsity, custom_sparsity_map
  )
  res_dict = {
      k: sparsities[k] if k in sparsities else None
      for k, p in flat_dict.items()
  }
  return_val = flax.traverse_util.unflatten_dict(res_dict)
  if isinstance(param_tree, flax.core.frozen_dict.FrozenDict):
    return_val = flax.core.freeze(return_val)
  return return_val


def get_n_zeros(size, sparsity):
  return int(np.floor(sparsity * size))


def get_sparsities_erdos_renyi(
    var_shape_dict,
    default_sparsity,
    custom_sparsity_map=None,
    include_kernel=True,
    erk_power_scale=1.0,
):
  """Given the method, returns the sparsity of individual layers as a dict.

  It ensures that the non-custom layers have a total parameter count as the one
  with uniform sparsities. In other words for the layers which are not in the
  custom_sparsity_map the following equation should be satisfied.

  N_i refers to the parameter count at layer i.
  (p_i * eps) gives the sparisty of layer i.

  # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
  # where p_i is np.sum(var_i.shape) / np.prod(var_i.shape)
  # for each i, eps*p_i needs to be in [0, 1].
  Args:
    var_shape_dict: dict, of shape of all Variables to prune.
    default_sparsity: float, between 0 and 1.
    custom_sparsity_map: dict or None, <str, float> key/value pairs where the
      mask correspond whose name is '{key}/mask:0' is set to the corresponding
      sparsity value.
    include_kernel: bool, if True kernel dimension are included in the scaling.
    erk_power_scale: float, if given used to take power of the ratio. Use
      scale<1 to make the erdos_renyi softer.

  Returns:
    sparsities, dict of where keys() are equal to all_masks and individiual
      masks are mapped to the their sparsities.
  """
  logging.info(
      (
          'ERK distribution is called with'
          ' default_sparsity=%s, include_kernel=%s, erk_power_scale=%f,'
      ),
      default_sparsity,
      include_kernel,
      erk_power_scale,
  )
  if not var_shape_dict:
    raise ValueError('Variable shape dictionary should not be empty')
  if default_sparsity is None or default_sparsity < 0 or default_sparsity > 1:
    raise ValueError('Default sparsity should be a value between 0 and 1.')

  # We have to enforce custom sparsities and then find the correct scaling
  # factor.
  if custom_sparsity_map is None:
    custom_sparsity_map = {}
  is_eps_valid = False

  # # The following loop will terminate worst case when all masks are in the
  # custom_sparsity_map. This should probably never happen though, since once
  # we have a single variable or more with the same constant, we have a valid
  # epsilon. Note that for each iteration we add at least one variable to the
  # custom_sparsity_map and therefore this while loop should terminate.
  dense_layers = set()
  while not is_eps_valid:
    # We will start with all layers and try to find right epsilon. However if
    # any probablity exceeds 1, we will make that layer dense and repeat the
    # process (finding epsilon) with the non-dense layers.
    # We want the total number of connections to be the same. Let say we have
    # four layers with N_1, ..., N_4 parameters each. Let say after some
    # iterations probability of some dense layers (3, 4) exceeded 1 and
    # therefore we added them to the dense_layers set. Those layers will not
    # scale with erdos_renyi, however we need to count them so that target
    # paratemeter count is achieved. See below.
    # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
    #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
    # eps * (p_1 * N_1 + p_2 * N_2) =
    #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
    # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

    divisor = 0
    rhs = 0
    raw_probabilities = {}
    for var_name, var_shape in var_shape_dict.items():
      n_param = np.prod(var_shape)
      n_zeros = get_n_zeros(n_param, default_sparsity)
      if var_name in dense_layers:
        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
        rhs -= n_zeros
      elif var_name in custom_sparsity_map:
        # We ignore custom_sparsities in erdos-renyi calculations.
        pass
      else:
        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
        # equation above.
        n_ones = n_param - n_zeros
        rhs += n_ones
        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
        if include_kernel:
          raw_probabilities[var_name] = (
              np.sum(var_shape) / np.prod(var_shape)
          ) ** erk_power_scale
        else:
          n_in, n_out = var_shape[-2:]
          raw_probabilities[var_name] = (n_in + n_out) / (n_in * n_out)

        # Note that raw_probabilities[mask] * n_param gives the individual
        # elements of the divisor.
        divisor += raw_probabilities[var_name] * n_param

    # By multipliying individual probabilites with epsilon, we should get the
    # number of parameters per layer correctly.
    eps = rhs / divisor
    # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
    # mask to 0., so they become part of dense_layers sets.
    max_prob = np.max(list(raw_probabilities.values()))
    max_prob_one = max_prob * eps
    if max_prob_one > 1:
      is_eps_valid = False
      for var_name, raw_prob in raw_probabilities.items():
        if raw_prob == max_prob:
          logging.info('Sparsity of var: %s had to be set to 0.', var_name)
          dense_layers.add(var_name)
    else:
      is_eps_valid = True

  sparsities = {}
  # With the valid epsilon, we can set sparsities of the remaning layers.
  for var_name, var_shape in var_shape_dict.items():
    n_param = np.prod(var_shape)
    if var_name in custom_sparsity_map:
      sparsities[var_name] = custom_sparsity_map[var_name]
      logging.info(
          'layer: %s has custom sparsity: %f', var_name, sparsities[var_name]
      )
    elif var_name in dense_layers:
      sparsities[var_name] = 0.0
    else:
      probability_one = eps * raw_probabilities[var_name]
      sparsities[var_name] = 1.0 - probability_one
    logging.info(
        'layer: %s, shape: %s, sparsity: %f',
        var_name,
        var_shape,
        sparsities[var_name],
    )
  return sparsities
