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

"""APIs for jaxpruner."""

from jaxpruner.algorithms import *


from jaxpruner.api import ALGORITHM_REGISTRY
from jaxpruner.api import all_algorithm_names
from jaxpruner.api import create_updater_from_config
from jaxpruner.api import register_algorithm
from jaxpruner.base_updater import apply_mask
from jaxpruner.base_updater import BaseUpdater
from jaxpruner.base_updater import NoPruning
from jaxpruner.base_updater import SparseState
from jaxpruner.sparsity_schedules import NoUpdateSchedule
from jaxpruner.sparsity_schedules import OneShotSchedule
from jaxpruner.sparsity_schedules import PeriodicSchedule
from jaxpruner.sparsity_schedules import PolynomialSchedule
from jaxpruner.sparsity_types import SparsityType
from jaxpruner.utils import summarize_intersection
from jaxpruner.utils import summarize_sparsity
