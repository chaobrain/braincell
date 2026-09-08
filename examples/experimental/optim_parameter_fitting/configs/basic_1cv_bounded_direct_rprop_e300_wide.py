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

"""Wide-bound 300-epoch baseline replacing Adam with Rprop only."""

from dataclasses import replace

from examples.experimental.optim_parameter_fitting.configs.basic_1cv_bounded_direct_adam_e300_wide import (
    CONFIG as BASE_CONFIG,
)
from examples.experimental.optim_parameter_fitting.optimizers import rprop

CONFIG = replace(
    BASE_CONFIG,
    stages=(
        rprop(
            epochs=300,
            learning_rate=0.01,
            etas=(0.5, 1.2),
            step_sizes=(1e-6, 50.0),
            gradient_method="rtrl",
            checkpoint_every=10,
        ),
    ),
)
