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

"""Lifetime checks shared by live selections of discretized model objects."""


class DiscreteView:
    """Reject public operations on selections from another grid generation."""

    __slots__ = ("_view_cell", "_view_generation")

    def _bind_view(self, cell):
        _ = cell._discretization
        object.__setattr__(self, "_view_cell", cell)
        object.__setattr__(self, "_view_generation", cell._view_generation)

    def _check_view(self):
        try:
            cell = object.__getattribute__(self, "_view_cell")
        except AttributeError:
            return
        cell._check_view_generation(object.__getattribute__(self, "_view_generation"))

    def __getattribute__(self, name):
        if not name.startswith("_"):
            object.__getattribute__(self, "_check_view")()
        return object.__getattribute__(self, name)
