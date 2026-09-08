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

"""Plain-Adam zero baseline: one CV, three bounded direct HH conductances."""

from examples.experimental.optim_parameter_fitting.config import (
    ExperimentConfig,
    InitializationConfig,
    ReportingConfig,
)
from examples.experimental.optim_parameter_fitting.datasets import feature_step_dataset
from examples.experimental.optim_parameter_fitting.losses import raw_voltage_mse
from examples.experimental.optim_parameter_fitting.models import hh_1cv_classic_bounded_direct
from examples.experimental.optim_parameter_fitting.optimizers import adam

CONFIG = ExperimentConfig(
    name="basic_1cv_bounded_direct_adam",
    model=hh_1cv_classic_bounded_direct(),
    dataset=feature_step_dataset(target_source="regenerate"),
    loss=raw_voltage_mse(),
    initialization=InitializationConfig(seed=0, num_candidates=64),
    stages=(
        adam(
            epochs=180,
            learning_rate=0.01,
            gradient_method="rtrl",
            checkpoint_every=10,
        ),
    ),
    reporting=ReportingConfig(
        validation_rmse_threshold_mv=5.0,
        parameter_relative_rms_threshold=0.1,
    ),
)
