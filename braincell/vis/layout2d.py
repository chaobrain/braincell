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

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from braincell._units import u
from braincell.morpho import Morpho, MorphoBranch


_ROOT_CHILD_SPAN_RAD = math.radians(120.0)


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
    direction_um: np.ndarray
    normal_um: np.ndarray


def build_layout_branches_2d(morpho: Morpho, *, mode: str) -> tuple[LayoutBranch2D, ...]:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_layout_branches_2d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if mode not in {"tree", "frustum"}:
        raise ValueError(f"Unsupported layout mode {mode!r}.")

    specs = _build_layout_specs(morpho, mode=mode)
    leaf_counts = _leaf_counts_by_branch(morpho.root)
    layouts: dict[int, LayoutBranch2D] = {}

    root = morpho.root
    root_index = root.index
    root_spec = specs[root_index]
    layouts[root_index] = _make_layout_branch(
        root,
        spec=root_spec,
        attach_um=np.zeros(2, dtype=float),
        angle_rad=0.0,
        child_x=0.0,
    )
    _layout_children(
        root,
        specs=specs,
        leaf_counts=leaf_counts,
        layouts=layouts,
        interval=(-_ROOT_CHILD_SPAN_RAD / 2.0, _ROOT_CHILD_SPAN_RAD / 2.0),
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def point_on_layout_branch(layout: LayoutBranch2D, x: float) -> np.ndarray:
    return layout.segment_points_um[0] + layout.direction_um * (float(x) * layout.total_length_um)


def _build_layout_specs(morpho: Morpho, *, mode: str) -> dict[int, _LayoutSpec2D]:
    if mode == "tree":
        branch_lengths_um = np.array(
            [float(branch.length.to_decimal(u.um)) for branch in morpho.branches],
            dtype=float,
        )
        unit_length_um = max(float(np.median(branch_lengths_um)), 1.0)
        return {
            branch.index: _LayoutSpec2D(
                segment_lengths_um=np.array([unit_length_um], dtype=float),
                radii_proximal_um=np.array([float(branch.mean_radius.to_decimal(u.um))], dtype=float),
                radii_distal_um=np.array([float(branch.mean_radius.to_decimal(u.um))], dtype=float),
            )
            for branch in morpho.branches
        }

    return {
        branch.index: _LayoutSpec2D(
            segment_lengths_um=np.asarray(branch.lengths.to_decimal(u.um), dtype=float),
            radii_proximal_um=np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float),
            radii_distal_um=np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float),
        )
        for branch in morpho.branches
    }


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


def _layout_children(
    parent: MorphoBranch,
    *,
    specs: dict[int, _LayoutSpec2D],
    leaf_counts: dict[int, int],
    layouts: dict[int, LayoutBranch2D],
    interval: tuple[float, float],
) -> None:
    children = parent.children
    if not children:
        return

    total_weight = sum(leaf_counts[child.index] for child in children)
    interval_start, interval_end = interval
    cursor = interval_start
    parent_layout = layouts[parent.index]

    for child in children:
        weight = leaf_counts[child.index]
        width = (interval_end - interval_start) * (weight / total_weight)
        child_interval = (cursor, cursor + width)
        child_angle = 0.5 * (child_interval[0] + child_interval[1])
        attach_um = point_on_layout_branch(parent_layout, child.parent_x)
        layouts[child.index] = _make_layout_branch(
            child,
            spec=specs[child.index],
            attach_um=attach_um,
            angle_rad=child_angle,
            child_x=float(child.child_x),
        )
        _layout_children(
            child,
            specs=specs,
            leaf_counts=leaf_counts,
            layouts=layouts,
            interval=child_interval,
        )
        cursor += width


def _make_layout_branch(
    branch: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    angle_rad: float,
    child_x: float,
) -> LayoutBranch2D:
    direction_um = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
    normal_um = np.array([-direction_um[1], direction_um[0]], dtype=float)
    start_um = np.asarray(attach_um, dtype=float) - direction_um * (child_x * spec.total_length_um)
    cumulative_lengths_um = np.concatenate(([0.0], np.cumsum(spec.segment_lengths_um)))
    segment_points_um = start_um + np.outer(cumulative_lengths_um, direction_um)
    return LayoutBranch2D(
        branch_index=branch.index,
        branch_name=branch.name,
        branch_type=branch.type,
        segment_points_um=segment_points_um,
        radii_proximal_um=spec.radii_proximal_um,
        radii_distal_um=spec.radii_distal_um,
        total_length_um=spec.total_length_um,
        direction_um=direction_um,
        normal_um=normal_um,
    )
