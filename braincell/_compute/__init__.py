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

"""Private compute layer: scheduling and runtime state built from CVs.

This package turns the immutable CV/node-tree declaration layer into the
runtime structures the solver and mechanism-lowering code execute against:

- :mod:`braincell._compute.scheduling` — node-tree scheduling helpers
- :mod:`braincell._compute.runtime` — mechanism lowering and runtime state
- :mod:`braincell._compute.table` — mechanism-object bookkeeping

This package is private. External code should access the same symbols
through :mod:`braincell` re-exports where appropriate.
"""

from .scheduling import NodeScheduling
from braincell._discretization.base import NodeTree

__all__ = [
    "NodeScheduling",
    "NodeTree",
]
