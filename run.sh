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

#!/bin/bash

set -e
set -x

virtualenv -p python3 env
source env/bin/activate

pip install -r jaxpruner/requirements.txt
python -m jaxpruner.base_updater_test
python -m jaxpruner.mask_calculator_test
python -m jaxpruner.sparsity_distributions_test
python -m jaxpruner.sparsity_schedules_test
python -m jaxpruner.base_updater_test
python -m jaxpruner.utils_test
python -m jaxpruner.algorithms.global_pruners_test
python -m jaxpruner.algorithms.pruners_test
python -m jaxpruner.algorithms.sparse_trainers_test
python -m jaxpruner.algorithms.ste_test
