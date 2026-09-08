# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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
# ==============================================================================

"""300-epoch run changing only direct-parameter transform bounds."""

from dataclasses import replace

from examples.experimental.optim_parameter_fitting.configs.basic_1cv_bounded_direct_adam_e300 import (
    CONFIG as BASE_CONFIG,
)
from examples.experimental.optim_parameter_fitting.models import hh_1cv_classic_bounded_direct

CONFIG = replace(
    BASE_CONFIG,
    model=hh_1cv_classic_bounded_direct(bound_multipliers=(0.1, 2.0)),
    initialization=replace(BASE_CONFIG.initialization, target_relative_range=(0.5, 1.5)),
)
