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

"""Algorithms implemented in jaxpruner."""


from jaxpruner.algorithms.global_pruners import GlobalMagnitudePruning
from jaxpruner.algorithms.global_pruners import GlobalSaliencyPruning
from jaxpruner.algorithms.pruners import MagnitudePruning
from jaxpruner.algorithms.pruners import RandomPruning
from jaxpruner.algorithms.pruners import SaliencyPruning
from jaxpruner.algorithms.sparse_trainers import RigL
from jaxpruner.algorithms.sparse_trainers import SET
from jaxpruner.algorithms.sparse_trainers import StaticRandomSparse
from jaxpruner.algorithms.ste import SteMagnitudePruning
from jaxpruner.algorithms.ste import SteRandomPruning
