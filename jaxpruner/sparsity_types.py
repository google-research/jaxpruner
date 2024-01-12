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

"""Sparsity distribution module with functions generates sparsity tree.

For Jax Pruner, users can allocate different desired sparsity for each different
parameter (weight), so the functions in this file help users to allocate the
sparsity to meet the overall target sparsity.

Note that all functions in this file must have the same input/output signatures.
"""
from typing import Any, NamedTuple, Tuple, Union


class Unstructured(NamedTuple):
  pass


class Block(NamedTuple):
  # This block is allocated for the last two dims of the corresponding variable.
  block_shape: Tuple[int, int]
  # If true, pool type is AVG. Otherwise, pool type is MAX.
  use_avg_pooling: bool = True


class NByM(NamedTuple):
  # This n-by-m operation works for the target axis of the corresponding
  # variable.
  n: int
  m: int
  axis: int = -1


class Channel(NamedTuple):
  # This channel-wise operations works for the target axis of the corresponding
  # variable.
  axis: int = -1


SparsityType = Union[Unstructured, Block, NByM, Channel]


# TODO following is needed since isintance(obj, SparsityType) fails.
# TODO check whether there is a better way.
def is_sparsity_type(instance_to_check):
  """Checks whether given object is a sparsity type."""
  return type(instance_to_check) in [Unstructured, Block, NByM, Channel]
