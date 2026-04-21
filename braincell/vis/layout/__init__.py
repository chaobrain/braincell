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

"""2D layout engine for :mod:`braincell.vis`.

This package turns a :class:`~braincell.morph.Morphology` into a tuple
of :class:`LayoutBranch2D` objects that the scene builders consume.
The public entry point is :func:`build_layout_branches_2d`; everything
else is either an internal helper or a re-export for backwards
compatibility with the pre-split ``braincell.vis.layout2d`` module.

Sub-modules
-----------
- ``_common``     — shared dataclasses, constants, tree-analysis helpers.
- ``_geometry``   — pure-numeric construction and sampling primitives.
- ``_collision``  — collision scoring for stem candidate ranking.
- ``_fan``        — recursive fan layout family (leaf-weighted sectors).
- ``_stem``       — stem / trunk_first layout family.
- ``_balloon``    — balloon layout family (leaf-count weighted cones).
- ``_radial``     — radial-360 layout family (full circle spread).
- ``_legacy``     — original legacy layout (kept for back-compat).
- ``_dispatch``   — ``build_layout_branches_2d`` dispatcher.
"""


from ._cache import LayoutCache, get_default_layout_cache
from ._common import LayoutBranch2D
from ._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig
from ._dispatch import build_layout_branches_2d
from ._geometry import (
    point_on_layout_branch,
    sample_layout_branch,
    tangent_on_layout_branch,
)

__all__ = [
    "DEFAULT_LAYOUT_CONFIG",
    "LayoutBranch2D",
    "LayoutCache",
    "LayoutConfig",
    "build_layout_branches_2d",
    "get_default_layout_cache",
    "point_on_layout_branch",
    "sample_layout_branch",
    "tangent_on_layout_branch",
]
