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

import importlib

from .branch import (
    ApicalDendrite,
    Axon,
    BasalDendrite,
    Branch,
    CustomBranch,
    Dendrite,
    Soma,
    branch_class_for_type,
)
from .metrics import MorphMetrics
from .morpho import MorphoEdge, Morpho, MorphoBranch

__all__ = [
    "ApicalDendrite",
    "Axon",
    "BasalDendrite",
    "Branch",
    "CustomBranch",
    "Dendrite",
    "MorphoEdge",
    "Morpho",
    "MorphoBranch",
    "MorphMetrics",
    "Soma",
    "branch_class_for_type",
    "vis",
]


def __getattr__(name: str):
    if name == "vis":
        return importlib.import_module(".vis", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
