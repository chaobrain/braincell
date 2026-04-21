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

"""Shared dataclasses, constants, and tree-analysis helpers for the 2D
layout engine.

This module holds the pieces that every layout family consumes:

* ``LayoutBranch2D`` — the frozen per-branch layout result.
* ``_LayoutSpec2D`` — the length/radius spec derived once from a
  :class:`~braincell.morph.Morphology`.
* Angle / span constants shared across stem, balloon, and radial
  dispatchers (``_ROOT_GROUP_EPSILON_RAD``, ``_AXON_ROOT_BASE_ANGLE_RAD``,
  …).
* Helpers to index a morphology tree (``_leaf_counts_by_branch``,
  ``_path_lengths_um_by_branch``), pick the "trunk" child from a fork,
  and allocate an angular interval to weighted children.

Nothing in this module depends on any specific layout family, so every
other module in ``braincell.vis.layout`` can import from it without risk
of cycles.
"""

import math
from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell.morph import MorphoBranch
from braincell.morph._morphology import Morphology

# ---------------------------------------------------------------------------
# Shared angle / bend constants
# ---------------------------------------------------------------------------

_DEFAULT_SIDE_BRANCH_ANGLE_RAD = math.radians(35.0)
_DEFAULT_SIDE_BRANCH_STEP_RAD = math.radians(20.0)
_AXON_ROOT_BASE_ANGLE_RAD = -math.pi / 2.0
_DENDRITE_ROOT_BASE_ANGLE_RAD = math.pi / 2.0
_ROOT_GROUP_EPSILON_RAD = math.radians(1.0)
_DEFAULT_BEND_FRACTION = 0.4


# ---------------------------------------------------------------------------
# Dataclasses shared across layout families
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LayoutSpec2D:
    segment_lengths_um: np.ndarray
    radii_proximal_um: np.ndarray
    radii_distal_um: np.ndarray

    @property
    def total_length_um(self) -> float:
        return float(np.sum(self.segment_lengths_um))


@dataclass(frozen=True)
class LayoutBranch2D:
    branch_index: int
    branch_name: str
    branch_type: str
    segment_points_um: np.ndarray
    radii_proximal_um: np.ndarray
    radii_distal_um: np.ndarray
    total_length_um: float
    segment_directions_um: np.ndarray
    segment_normals_um: np.ndarray
    cumulative_lengths_um: np.ndarray

    @property
    def direction_um(self) -> np.ndarray:
        return self.segment_directions_um[-1]

    @property
    def normal_um(self) -> np.ndarray:
        return self.segment_normals_um[-1]

    @property
    def start_direction_um(self) -> np.ndarray:
        return self.segment_directions_um[0]

    @property
    def end_direction_um(self) -> np.ndarray:
        return self.segment_directions_um[-1]


# ---------------------------------------------------------------------------
# Morphology → layout spec helpers
# ---------------------------------------------------------------------------

def _build_layout_specs(morpho: Morphology) -> dict[int, _LayoutSpec2D]:
    return {
        branch.index: _LayoutSpec2D(
            segment_lengths_um=np.asarray(branch.lengths.to_decimal(u.um), dtype=float),
            radii_proximal_um=np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float),
            radii_distal_um=np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float),
        )
        for branch in morpho.branches
    }


def _normalize_min_branch_angle_rad(value: float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise TypeError(f"min_branch_angle_deg must be a non-negative float or None, got {value!r}.")
    min_branch_angle_deg = float(value)
    if min_branch_angle_deg < 0.0:
        raise ValueError(f"min_branch_angle_deg must be >= 0, got {min_branch_angle_deg!r}.")
    return math.radians(min_branch_angle_deg)


def _leaf_counts_by_branch(root: MorphoBranch) -> dict[int, int]:
    counts: dict[int, int] = {}

    def visit(node: MorphoBranch) -> int:
        if node.n_children == 0:
            counts[node.index] = 1
            return 1
        total = sum(visit(child) for child in node.children)
        counts[node.index] = total
        return total

    visit(root)
    return counts


def _path_lengths_um_by_branch(root: MorphoBranch) -> dict[int, float]:
    path_lengths_um: dict[int, float] = {}

    def visit(node: MorphoBranch) -> float:
        branch_length_um = float(node.length.to_decimal(u.um))
        if node.n_children == 0:
            path_lengths_um[node.index] = branch_length_um
            return branch_length_um
        max_child_length_um = max(visit(child) for child in node.children)
        path_lengths_um[node.index] = branch_length_um + max_child_length_um
        return path_lengths_um[node.index]

    visit(root)
    return path_lengths_um


# ---------------------------------------------------------------------------
# Trunk / side child selection
# ---------------------------------------------------------------------------

def _pick_trunk_child(
    children: tuple[MorphoBranch, ...],
    *,
    subtree_path_lengths_um: dict[int, float],
    branch_order: dict[int, int],
) -> MorphoBranch:
    return max(
        children,
        key=lambda child: (
            subtree_path_lengths_um[child.index],
            float(child.length.to_decimal(u.um)),
            -branch_order[child.index],
        ),
    )


def _resolve_trunk_child_angle(
    child: MorphoBranch,
    *,
    parent_angle_rad: float,
    min_branch_angle_rad: float,
) -> float:
    if child.parent_x == child.child_x:
        return parent_angle_rad + _side_branch_offsets_rad(min_branch_angle_rad, n_offsets=1)[0]
    return parent_angle_rad


def _side_branch_offsets_rad(min_branch_angle_rad: float, *, n_offsets: int) -> list[float]:
    if n_offsets <= 0:
        return []
    base_offset_rad = max(min_branch_angle_rad, _DEFAULT_SIDE_BRANCH_ANGLE_RAD)
    step_rad = max(min_branch_angle_rad, _DEFAULT_SIDE_BRANCH_STEP_RAD)
    offsets: list[float] = []
    for offset_index in range(n_offsets):
        slot_index = offset_index // 2
        offsets.append(base_offset_rad + slot_index * step_rad)
    return offsets


def _clamp_angle_to_root_group(angle_rad: float, *, group_name: str) -> float:
    if group_name == "axon":
        return min(max(angle_rad, -math.pi + _ROOT_GROUP_EPSILON_RAD), -_ROOT_GROUP_EPSILON_RAD)
    return min(max(angle_rad, _ROOT_GROUP_EPSILON_RAD), math.pi - _ROOT_GROUP_EPSILON_RAD)


# ---------------------------------------------------------------------------
# Weighted angular allocation
# ---------------------------------------------------------------------------

def _allocate_weighted_angles(
    children: tuple[MorphoBranch, ...],
    *,
    interval: tuple[float, float],
    weights: dict[int, float],
) -> dict[int, float]:
    child_intervals = _weighted_child_intervals(children, interval=interval, weights=weights, min_gap_rad=0.0)
    return {child.index: 0.5 * (child_interval[0] + child_interval[1]) for child, child_interval in child_intervals}


def _weighted_child_intervals(
    children: tuple[MorphoBranch, ...] | list[MorphoBranch],
    *,
    interval: tuple[float, float],
    weights: dict[int, float],
    min_gap_rad: float,
) -> list[tuple[MorphoBranch, tuple[float, float]]]:
    children = tuple(children)
    if not children:
        return []
    if len(children) == 1:
        return [(children[0], interval)]

    span_rad = interval[1] - interval[0]
    required_gap_rad = min_gap_rad * (len(children) - 1)
    available_span_rad = max(span_rad - required_gap_rad, 0.0)
    total_weight = sum(max(float(weights.get(child.index, 1.0)), 1e-6) for child in children)
    cursor_rad = interval[0]
    child_intervals: list[tuple[MorphoBranch, tuple[float, float]]] = []
    for child_index, child in enumerate(children):
        weight = max(float(weights.get(child.index, 1.0)), 1e-6)
        width_rad = available_span_rad * weight / total_weight if total_weight > 0.0 else available_span_rad / len(
            children)
        child_interval = (cursor_rad, cursor_rad + width_rad)
        child_intervals.append((child, child_interval))
        cursor_rad = child_interval[1]
        if child_index != len(children) - 1:
            cursor_rad += min_gap_rad
    return child_intervals


# ---------------------------------------------------------------------------
# Leaf-count-driven child region allocation
# ---------------------------------------------------------------------------
#
# ``_allocate_child_regions_leaf_weighted`` was historically named
# ``_allocate_child_regions_legacy`` because it first appeared in the
# legacy layout, but the balloon family also relies on it. The helper
# therefore lives in the shared module under its original name; the
# behaviour is unchanged.

def _allocate_child_regions_legacy(
    *,
    children: tuple[MorphoBranch, ...],
    interval: tuple[float, float],
    leaf_counts: dict[int, int],
    min_branch_angle_rad: float,
) -> list[tuple[MorphoBranch, tuple[float, float], float]]:
    if len(children) == 1:
        return [(children[0], interval, 0.5 * (interval[0] + interval[1]))]

    span = interval[1] - interval[0]
    required_gap = min_branch_angle_rad * (len(children) - 1)
    if span < required_gap:
        return _allocate_child_regions_legacy_fallback(children=children, interval=interval)
    return _allocate_child_regions_legacy_with_gap(
        children=children,
        interval=interval,
        leaf_counts=leaf_counts,
        min_branch_angle_rad=min_branch_angle_rad,
    )


def _allocate_child_regions_legacy_with_gap(
    *,
    children: tuple[MorphoBranch, ...],
    interval: tuple[float, float],
    leaf_counts: dict[int, int],
    min_branch_angle_rad: float,
) -> list[tuple[MorphoBranch, tuple[float, float], float]]:
    total_weight = sum(leaf_counts[child.index] for child in children)
    available_span = (interval[1] - interval[0]) - min_branch_angle_rad * (len(children) - 1)
    cursor = interval[0]
    allocations: list[tuple[MorphoBranch, tuple[float, float], float]] = []

    for child_index, child in enumerate(children):
        weight = leaf_counts[child.index]
        width = 0.0 if total_weight == 0 else available_span * (weight / total_weight)
        child_interval = (cursor, cursor + width)
        allocations.append((child, child_interval, 0.5 * (child_interval[0] + child_interval[1])))
        cursor = child_interval[1]
        if child_index != len(children) - 1:
            cursor += min_branch_angle_rad

    return allocations


def _allocate_child_regions_legacy_fallback(
    *,
    children: tuple[MorphoBranch, ...],
    interval: tuple[float, float],
) -> list[tuple[MorphoBranch, tuple[float, float], float]]:
    width = (interval[1] - interval[0]) / len(children)
    cursor = interval[0]
    allocations: list[tuple[MorphoBranch, tuple[float, float], float]] = []

    for child in children:
        child_interval = (cursor, cursor + width)
        allocations.append((child, child_interval, 0.5 * (child_interval[0] + child_interval[1])))
        cursor = child_interval[1]

    return allocations
