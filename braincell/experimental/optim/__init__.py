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

"""Experimental optimization BPTT/RTRL gradient interfaces."""

from braincell.experimental.optim.gradients import (
    FullRTRLDiagnostic,
    RolloutGradientEngine,
    RolloutGradientResult,
    TrajectoryGradientEngine,
    TrajectoryGradientResult,
    build_rollout_value_and_grad,
    build_trajectory_value_and_grad,
)

__all__ = [
    "FullRTRLDiagnostic",
    "RolloutGradientEngine",
    "RolloutGradientResult",
    "TrajectoryGradientEngine",
    "TrajectoryGradientResult",
    "build_rollout_value_and_grad",
    "build_trajectory_value_and_grad",
]
