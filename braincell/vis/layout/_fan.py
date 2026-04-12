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

"""Fan layout family.

The fan family is a simple straight-branch layout.

* The root soma is centered in layout space, so root ``parent_x`` means
  left / center / right attachment.
* Root children are partitioned into left, middle-upper/lower, and
  right sectors.
* Every branch is a straight centerline; angular variation comes only
  from the sector assignment and sibling fan-out, not from internal
  curvature.
"""

import math

import numpy as np

from braincell.morph import MorphoBranch, Morphology

from ._common import (
    LayoutBranch2D,
    _LayoutSpec2D,
    _leaf_counts_by_branch,
    _normalize_min_branch_angle_rad,
)
from ._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig
from ._geometry import (
    _make_centered_horizontal_layout_branch,
    _make_straight_layout_branch,
    sample_layout_branch,
)

_LEFT_ROOT_CENTER_RAD = math.pi
_UPPER_ROOT_CENTER_RAD = math.pi / 2.0
_LOWER_ROOT_CENTER_RAD = -math.pi / 2.0
_RIGHT_ROOT_CENTER_RAD = 0.0


def _build_layout_branches_fan(
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
    layouts[root.index] = _make_centered_horizontal_layout_branch(
        root,
        spec=layout_specs[root.index],
        center_um=np.zeros(2, dtype=float),
    )
    _layout_children_fan(
        root,
        layout_specs=layout_specs,
        leaf_counts=leaf_counts,
        layouts=layouts,
        interval=None,
        min_branch_angle_rad=min_branch_angle_rad,
        is_root=True,
        layout_config=config,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _layout_children_fan(
    parent: MorphoBranch,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    leaf_counts: dict[int, int],
    layouts: dict[int, LayoutBranch2D],
    interval: tuple[float, float] | None,
    min_branch_angle_rad: float,
    is_root: bool,
    layout_config: LayoutConfig,
) -> None:
    children = parent.children
    if not children:
        return

    if is_root:
        allocations = _allocate_root_fan_children(
            children,
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
            layout_config=layout_config,
        )
    else:
        assert interval is not None
        allocations = _allocate_fan_sector_children(
            children=children,
            interval=interval,
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
        )

    parent_layout = layouts[parent.index]
    for child, child_interval, child_angle in allocations:
        attach_um, attach_tangent_um = sample_layout_branch(parent_layout, child.parent_x)
        _ = attach_tangent_um
        layouts[child.index] = _make_straight_layout_branch(
            child,
            spec=layout_specs[child.index],
            attach_um=attach_um,
            target_angle_rad=child_angle,
            child_x=float(child.child_x),
        )
        _layout_children_fan(
            child,
            layout_specs=layout_specs,
            leaf_counts=leaf_counts,
            layouts=layouts,
            interval=child_interval,
            min_branch_angle_rad=min_branch_angle_rad,
            is_root=False,
            layout_config=layout_config,
        )


def _allocate_root_fan_children(
    children: tuple[MorphoBranch, ...],
    *,
    leaf_counts: dict[int, int],
    min_branch_angle_rad: float,
    layout_config: LayoutConfig,
) -> list[tuple[MorphoBranch, tuple[float, float], float]]:
    left_children: list[MorphoBranch] = []
    middle_children: list[MorphoBranch] = []
    right_children: list[MorphoBranch] = []
    for child in children:
        parent_x = float(child.parent_x)
        if parent_x <= 0.0:
            left_children.append(child)
        elif abs(parent_x - 0.5) <= 1e-12:
            middle_children.append(child)
        else:
            right_children.append(child)

    allocations: list[tuple[MorphoBranch, tuple[float, float], float]] = []
    allocations.extend(
        _allocate_root_sector(
            tuple(_sorted_fan_children(left_children, leaf_counts=leaf_counts)),
            interval=_sector_interval(_LEFT_ROOT_CENTER_RAD, layout_config.fan_root_left_span_rad),
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
        )
    )
    allocations.extend(
        _allocate_root_middle_sectors(
            tuple(middle_children),
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
            layout_config=layout_config,
        )
    )
    allocations.extend(
        _allocate_root_sector(
            tuple(_sorted_fan_children(right_children, leaf_counts=leaf_counts)),
            interval=_sector_interval(_RIGHT_ROOT_CENTER_RAD, layout_config.fan_root_right_span_rad),
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
        )
    )
    return allocations


def _allocate_root_middle_sectors(
    children: tuple[MorphoBranch, ...],
    *,
    leaf_counts: dict[int, int],
    min_branch_angle_rad: float,
    layout_config: LayoutConfig,
) -> list[tuple[MorphoBranch, tuple[float, float], float]]:
    if not children:
        return []
    ordered = _sorted_fan_children(children, leaf_counts=leaf_counts)
    upper_count = (len(ordered) + 1) // 2
    upper = tuple(ordered[:upper_count])
    lower = tuple(ordered[upper_count:])
    allocations: list[tuple[MorphoBranch, tuple[float, float], float]] = []
    allocations.extend(
        _allocate_root_sector(
            upper,
            interval=_sector_interval(
                _UPPER_ROOT_CENTER_RAD,
                layout_config.fan_root_middle_upper_span_rad,
            ),
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
        )
    )
    allocations.extend(
        _allocate_root_sector(
            lower,
            interval=_sector_interval(
                _LOWER_ROOT_CENTER_RAD,
                layout_config.fan_root_middle_lower_span_rad,
            ),
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
        )
    )
    return allocations


def _allocate_root_sector(
    children: tuple[MorphoBranch, ...],
    *,
    interval: tuple[float, float],
    leaf_counts: dict[int, int],
    min_branch_angle_rad: float,
) -> list[tuple[MorphoBranch, tuple[float, float], float]]:
    if not children:
        return []
    return _allocate_fan_sector_children(
        children=children,
        interval=interval,
        leaf_counts=leaf_counts,
        min_branch_angle_rad=min_branch_angle_rad,
    )


def _sector_interval(center_rad: float, span_rad: float) -> tuple[float, float]:
    return (center_rad - span_rad / 2.0, center_rad + span_rad / 2.0)


def _allocate_fan_sector_children(
    *,
    children: tuple[MorphoBranch, ...],
    interval: tuple[float, float],
    leaf_counts: dict[int, int],
    min_branch_angle_rad: float,
) -> list[tuple[MorphoBranch, tuple[float, float], float]]:
    ordered = _center_out_to_left_right(_sorted_fan_children(children, leaf_counts=leaf_counts))
    if not ordered:
        return []
    if len(ordered) == 1:
        child = ordered[0]
        return [(child, interval, 0.5 * (interval[0] + interval[1]))]

    span = interval[1] - interval[0]
    required_gap = min_branch_angle_rad * (len(ordered) - 1)
    if span <= required_gap:
        centers = np.linspace(interval[0], interval[1], num=len(ordered))
        child_width = span / len(ordered) if len(ordered) else 0.0
        allocations: list[tuple[MorphoBranch, tuple[float, float], float]] = []
        for idx, child in enumerate(ordered):
            center = float(centers[idx])
            child_interval = (center - 0.5 * child_width, center + 0.5 * child_width)
            allocations.append((child, child_interval, center))
        return allocations

    available_span = span - required_gap
    total_weight = sum(max(float(leaf_counts.get(child.index, 1.0)), 1e-6) for child in ordered)
    cursor = interval[0]
    allocations = []
    for index, child in enumerate(ordered):
        weight = max(float(leaf_counts.get(child.index, 1.0)), 1e-6)
        width = available_span * (weight / total_weight) if total_weight > 0.0 else available_span / len(ordered)
        child_interval = (cursor, cursor + width)
        allocations.append((child, child_interval, 0.5 * (child_interval[0] + child_interval[1])))
        cursor = child_interval[1]
        if index != len(ordered) - 1:
            cursor += min_branch_angle_rad
    return allocations


def _sorted_fan_children(
    children: tuple[MorphoBranch, ...] | list[MorphoBranch],
    *,
    leaf_counts: dict[int, int],
) -> list[MorphoBranch]:
    return sorted(
        children,
        key=lambda child: (-leaf_counts.get(child.index, 1), child.index),
    )


def _center_out_to_left_right(children: list[MorphoBranch]) -> list[MorphoBranch]:
    if not children:
        return []
    left: list[MorphoBranch] = []
    right: list[MorphoBranch] = []
    center: list[MorphoBranch] = []
    if len(children) % 2 == 1:
        center.append(children[0])
        start = 1
    else:
        start = 0
    for index, child in enumerate(children[start:], start=start):
        side_slot = index - start
        if side_slot % 2 == 0:
            left.append(child)
        else:
            right.append(child)
    return list(reversed(left)) + center + right
