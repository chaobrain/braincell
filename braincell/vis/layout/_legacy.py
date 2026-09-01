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

"""Legacy ``root_layout='legacy'`` layout family.

This is the original layout from the first 2D renderer, kept only for
callers that pin ``root_layout='legacy'`` to reproduce older figures.
The stem family subsumes it for everything new and is the default.

The module also holds a handful of helpers from earlier iterations
that are not currently referenced anywhere in the live layout code
(``_assign_group_trunk_first_angles`` / ``_assign_child_trunk_first_angles``,
``_make_layout_branch_to_y``, ``_horizontal_segment_points_um``). They
are preserved here — rather than silently deleted during the mechanical
split — so the git history of each symbol stays intact and a future
layout can revive them without digging through git log. They are clearly
quarantined in the "Unused legacy helpers" section below.
"""

import numpy as np

from braincell.morph import MorphoBranch
from braincell.morph.morphology import Morphology

from ._common import (
    LayoutBranch2D,
    _LayoutSpec2D,
    _allocate_child_regions_legacy,
    _clamp_angle_to_root_group,
    _leaf_counts_by_branch,
    _normalize_min_branch_angle_rad,
    _pick_trunk_child,
    _resolve_trunk_child_angle,
    _side_branch_offsets_rad,
    walk_layout_top_down,
)
from ._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig
from ._geometry import (
    _layout_branch_from_points,
    _make_layout_branch,
    _smoothstep,
    _vector_angle_rad,
    point_on_layout_branch,
    sample_layout_branch,
)


# ---------------------------------------------------------------------------
# Live legacy layout (still reachable via root_layout='legacy')
# ---------------------------------------------------------------------------


def _build_layout_branches_legacy(
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
        bend_fraction=config.default_bend_fraction,
    )
    root_span = config.legacy_root_child_span_rad
    _layout_children_legacy(
        root,
        layout_specs=layout_specs,
        leaf_counts=leaf_counts,
        layouts=layouts,
        interval=(-root_span / 2.0, root_span / 2.0),
        min_branch_angle_rad=min_branch_angle_rad,
        layout_config=config,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _layout_children_legacy(
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

    A child's region derives from its parent's finished interval alone,
    so a sibling group can be placed before any of it is descended into.
    :func:`~braincell.vis.layout._common.walk_layout_top_down` owns the
    iterative walk (and the recursion-avoidance invariant behind it);
    this function only says what a child's placement is.
    """
    # frame: (branch, angular region allocated to it)
    Frame = tuple[MorphoBranch, tuple[float, float]]

    def _place(frame: Frame, children: tuple[MorphoBranch, ...]) -> list[Frame]:
        node, node_interval = frame
        allocations = _allocate_child_regions_legacy(
            children=children,
            interval=node_interval,
            leaf_counts=leaf_counts,
            min_branch_angle_rad=min_branch_angle_rad,
        )
        node_layout = layouts[node.index]
        frames: list[Frame] = []
        for child, child_interval, child_angle in allocations:
            attach_um, attach_tangent_um = sample_layout_branch(node_layout, child.parent_x)
            layouts[child.index] = _make_layout_branch(
                child,
                spec=layout_specs[child.index],
                attach_um=attach_um,
                attach_angle_rad=_vector_angle_rad(attach_tangent_um),
                target_angle_rad=child_angle,
                child_x=float(child.child_x),
                bend_fraction=layout_config.default_bend_fraction,
            )
            frames.append((child, child_interval))
        return frames

    walk_layout_top_down((parent, interval), _place)


# ---------------------------------------------------------------------------
# Unused legacy helpers (preserved for history)
# ---------------------------------------------------------------------------
#
# The helpers below are not referenced from any live code path. They
# survive from earlier layout iterations (a trunk_first angle
# assignment pass and a dendrogram fallback layout) and are kept here
# so the git blame stays meaningful if someone needs to revive the
# corresponding feature. If you're adding code, prefer the stem /
# balloon / radial families instead of calling into this section.


def _assign_group_trunk_first_angles(
    children: tuple[MorphoBranch, ...],
    *,
    base_angle_rad: float,
    group_name: str,
    subtree_path_lengths_um: dict[int, float],
    branch_order: dict[int, int],
    min_branch_angle_rad: float,
) -> dict[int, float]:
    assignments = _assign_child_trunk_first_angles(
        children,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
        min_branch_angle_rad=min_branch_angle_rad,
        base_angle_rad_by_child={child.index: base_angle_rad for child in children},
    )
    return {
        child_index: _clamp_angle_to_root_group(angle_rad, group_name=group_name)
        for child_index, angle_rad in assignments.items()
    }


def _assign_child_trunk_first_angles(
    children: tuple[MorphoBranch, ...],
    *,
    subtree_path_lengths_um: dict[int, float],
    branch_order: dict[int, int],
    min_branch_angle_rad: float,
    base_angle_rad_by_child: dict[int, float],
) -> dict[int, float]:
    trunk_child = _pick_trunk_child(
        children,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
    )
    side_children = tuple(child for child in children if child is not trunk_child)
    side_children = tuple(
        sorted(
            side_children,
            key=lambda child: (-subtree_path_lengths_um[child.index], branch_order[child.index]),
        )
    )
    side_offset_sequence = _side_branch_offsets_rad(min_branch_angle_rad, n_offsets=len(side_children))
    assignments: dict[int, float] = {
        trunk_child.index: _resolve_trunk_child_angle(
            trunk_child,
            parent_angle_rad=base_angle_rad_by_child[trunk_child.index],
            min_branch_angle_rad=min_branch_angle_rad,
        )
    }

    for child_index, child in enumerate(side_children):
        sign = 1.0 if child_index % 2 == 0 else -1.0
        assignments[child.index] = base_angle_rad_by_child[child.index] + sign * side_offset_sequence[child_index]

    return assignments


def _make_layout_branch_to_y(
    branch: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    target_y_um: float,
    child_x: float,
) -> LayoutBranch2D:
    total_length_um = spec.total_length_um
    if total_length_um <= 0.0:
        return _layout_branch_from_points(
            branch,
            spec,
            np.repeat(np.asarray(attach_um, dtype=float)[None, :], len(spec.segment_lengths_um) + 1, axis=0),
        )

    cumulative_lengths_um = np.concatenate(([0.0], np.cumsum(spec.segment_lengths_um)))
    start_fractions = cumulative_lengths_um[:-1] / total_length_um
    end_fractions = cumulative_lengths_um[1:] / total_length_um
    dy_total_um = float(target_y_um)
    smooth_start = _smoothstep(start_fractions)
    smooth_end = _smoothstep(end_fractions)
    dy_segments_um = dy_total_um * (smooth_end - smooth_start)
    dx_segments_um = np.sqrt(np.maximum(spec.segment_lengths_um**2 - dy_segments_um**2, 0.0))

    raw_segment_points_um = np.zeros((len(spec.segment_lengths_um) + 1, 2), dtype=float)
    for segment_index in range(len(spec.segment_lengths_um)):
        raw_segment_points_um[segment_index + 1] = raw_segment_points_um[segment_index] + np.array(
            [dx_segments_um[segment_index], dy_segments_um[segment_index]], dtype=float
        )

    raw_layout = _layout_branch_from_points(branch, spec, raw_segment_points_um)
    attach_point_um = point_on_layout_branch(raw_layout, child_x)
    translation_um = np.asarray(attach_um, dtype=float) - attach_point_um
    return _layout_branch_from_points(branch, spec, raw_segment_points_um + translation_um)


def _horizontal_segment_points_um(segment_lengths_um: np.ndarray, *, start_um: np.ndarray) -> np.ndarray:
    start_um = np.asarray(start_um, dtype=float)
    cumulative_x_um = np.concatenate(([0.0], np.cumsum(segment_lengths_um)))
    return np.column_stack(
        (
            start_um[0] + cumulative_x_um,
            np.full(len(cumulative_x_um), start_um[1], dtype=float),
        )
    )
