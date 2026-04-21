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

"""Compute-graph layer: execution view and runtime state built from CVs.

This package turns the immutable CV layer into the structures the voltage
solver and mechanism-lowering code actually execute against:

- :mod:`braincell.compute._point_tree` — merged point/edge execution graph
- :mod:`braincell.compute._runtime` — mechanism lowering and runtime state
- :mod:`braincell.compute._assignment_table` — mechanism-object bookkeeping

The public surface is intentionally small; most callers interact with these
pieces indirectly through :class:`braincell.Cell`.
"""

from ._point_tree import PointScheduling, PointTree

__all__ = [
    "PointScheduling",
    "PointTree",
]
