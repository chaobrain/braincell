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

"""Radial-360 layout family.

Spreads the root stems around a full 2π circle, then recursively
narrows the allowed interval for each child. Useful for dense
morphologies where the stem layout would produce heavy occlusion in
one half-plane.
"""

import math

import numpy as np

from braincell.morph import MorphoBranch
from braincell.morph.morphology import Morphology

from ._common import (
    LayoutBranch2D,
    _LayoutSpec2D,
    _leaf_counts_by_branch,
    _normalize_min_branch_angle_rad,
    _weighted_child_intervals,
    walk_layout_top_down,
)
from ._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig
from ._geometry import (
    _make_layout_branch,
    _vector_angle_rad,
    sample_layout_branch,
)

_RADIAL_MIN_CHILD_SPAN_RAD = math.radians(40.0)


def _build_layout_branches_radial_360(
    morpho: Morphology,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
    layout_config: LayoutConfig | None = None,
) -> tuple[LayoutBranch2D, ...]:
    config = layout_config or DEFAULT_LAYOUT_CONFIG
    min_branch_angle_rad = _normalize_min_branch_angle_rad(min_branch_angle_deg)
    leaf_counts = _leaf_counts_by_branch(morpho.root)
    layouts: dict[int, LayoutBranch2D] = {}

    root = morpho.root
    root_spec = layout_specs[root.index]
    layouts[root.index] = _make_layout_branch(
        root,
        spec=root_spec,
        attach_um=np.zeros(2, dtype=float),
        attach_angle_rad=0.0,
        target_angle_rad=0.0,
        child_x=0.0,
        bend_fraction=config.radial_bend_fraction,
    )
    root_span = config.radial_root_span_rad
    _layout_children_radial_360(
        root,
        layout_specs=layout_specs,
        leaf_counts=leaf_counts,
        layouts=layouts,
        interval=(-root_span / 2.0, root_span / 2.0),
        min_branch_angle_rad=min_branch_angle_rad,
        layout_config=config,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _layout_children_radial_360(
    parent: MorphoBranch,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    leaf_counts: dict[int, int],
    layouts: dict[int, LayoutBranch2D],
    interval: tuple[float, float],
    min_branch_angle_rad: float,
    layout_config: LayoutConfig,
) -> None:
    """Lay out every descendant of ``parent``.

    A child's wedge derives from its parent's finished interval alone,
    so a sibling group can be placed before any of it is descended into.
    :func:`~braincell.vis.layout._common.walk_layout_top_down` owns the
    iterative walk (and the recursion-avoidance invariant behind it);
    this function only says what a child's placement is.
    """
    # frame: (branch, angular interval allocated to it)
    Frame = tuple[MorphoBranch, tuple[float, float]]

    def _place(frame: Frame, children: tuple[MorphoBranch, ...]) -> list[Frame]:
        node, node_interval = frame
        child_intervals = _weighted_child_intervals(
            children, interval=node_interval, weights=leaf_counts, min_gap_rad=min_branch_angle_rad
        )
        node_layout = layouts[node.index]
        frames: list[Frame] = []
        for child, child_interval in child_intervals:
            attach_um, attach_tangent_um = sample_layout_branch(node_layout, child.parent_x)
            child_angle_rad = 0.5 * (child_interval[0] + child_interval[1])
            layouts[child.index] = _make_layout_branch(
                child,
                spec=layout_specs[child.index],
                attach_um=attach_um,
                attach_angle_rad=_vector_angle_rad(attach_tangent_um),
                target_angle_rad=child_angle_rad,
                child_x=float(child.child_x),
                bend_fraction=layout_config.radial_bend_fraction,
            )
            child_span_rad = min(
                layout_config.radial_child_span_rad,
                max(child_interval[1] - child_interval[0], _RADIAL_MIN_CHILD_SPAN_RAD),
            )
            frames.append((child, (child_angle_rad - child_span_rad / 2.0, child_angle_rad + child_span_rad / 2.0)))
        return frames

    walk_layout_top_down((parent, interval), _place)
