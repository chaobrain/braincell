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

"""Interchangeable lightweight execution models for detailed Cells."""

from braincell.reduction.core import (
    ReductionContext,
    ReductionInputGroup,
    ReductionInputGroupSchema,
    ReductionInputs,
    ReductionModel,
    ReductionOutput,
    ReductionRecording,
    ReductionSynapse,
    ReductionView,
    ReductionViewCollection,
)
from braincell.reduction.toy import (
    EventAccumulatorReduction,
    PayloadAccumulatorReduction,
    SynapticKernelAccumulatorReduction,
)
from braincell.reduction.runtime import ReductionInputLayout, ReductionInputRuntime, build_reduction_input_runtime
from braincell.reduction.dif import DIFReduction

_BUILTIN_MODELS = {"dif": DIFReduction}


__all__ = [
    "DIFReduction",
    "EventAccumulatorReduction",
    "PayloadAccumulatorReduction",
    "ReductionContext",
    "ReductionInputGroup",
    "ReductionInputGroupSchema",
    "ReductionInputLayout",
    "ReductionInputRuntime",
    "ReductionInputs",
    "ReductionModel",
    "ReductionOutput",
    "ReductionRecording",
    "ReductionSynapse",
    "ReductionView",
    "ReductionViewCollection",
    "SynapticKernelAccumulatorReduction",
    "build_reduction_input_runtime",
]
