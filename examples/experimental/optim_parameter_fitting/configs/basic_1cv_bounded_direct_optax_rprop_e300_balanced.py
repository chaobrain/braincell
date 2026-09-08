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

"""Single-scale Rprop with target-amplitude-balanced protocol MSE."""

from dataclasses import replace

from examples.experimental.optim_parameter_fitting.configs.basic_1cv_bounded_direct_optax_rprop_e300_wide import (
    CONFIG as BASE_CONFIG,
)
from examples.experimental.optim_parameter_fitting.losses import protocol_balanced_voltage_mse

CONFIG = replace(
    BASE_CONFIG,
    loss=protocol_balanced_voltage_mse(std_floor_mv=5.0),
)
