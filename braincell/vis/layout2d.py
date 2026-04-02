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


import math
from dataclasses import dataclass

import numpy as np

import brainunit as u
from braincell.morpho import Morpho, MorphoBranch

_ROOT_CHILD_SPAN_RAD = math.radians(120.0)
_DEFAULT_SIDE_BRANCH_ANGLE_RAD = math.radians(35.0)
_DEFAULT_SIDE_BRANCH_STEP_RAD = math.radians(20.0)
_AXON_ROOT_BASE_ANGLE_RAD = -math.pi / 2.0
_DENDRITE_ROOT_BASE_ANGLE_RAD = math.pi / 2.0
_ROOT_GROUP_EPSILON_RAD = math.radians(1.0)
_VALID_ROOT_LAYOUTS = {"type_split", "legacy"}
_VALID_LAYOUT_FAMILIES = {"stem", "trunk_first", "balloon", "radial_360"}
_LAYOUT_FAMILY_ALIASES = {"trunk_first": "stem"}
_BALLOON_ROOT_SPAN_RAD = math.radians(180.0)
_BALLOON_CHILD_SPAN_RAD = math.radians(120.0)
_BALLOON_TYPE_SPLIT_SPAN_RAD = math.radians(110.0)
_STEM_ROOT_GROUP_SPAN_RAD = math.radians(120.0)
_STEM_ROOT_FULL_SPAN_RAD = math.radians(150.0)
_RADIAL_ROOT_SPAN_RAD = 2.0 * math.pi
_RADIAL_CHILD_SPAN_RAD = math.radians(150.0)
_DEFAULT_BEND_FRACTION = 0.4
_BALLOON_BEND_FRACTION = 0.22
_RADIAL_BEND_FRACTION = 0.25
_COLLISION_MARGIN_UM = 2.0
_COLLISION_RETRY_LIMIT = 8
_STEM_COLLISION_WINDOW = 24


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


@dataclass(frozen=True)
class _StemAngleProfile:
    launch_angle_rad: float
    settle_angle_rad: float
    tail_angle_rad: float


def build_layout_branches_2d(
    morpho: Morpho,
    *,
    mode: str,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout_family: str = "stem",
) -> tuple[LayoutBranch2D, ...]:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_layout_branches_2d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if mode not in {"tree", "frustum"}:
        raise ValueError(f"Unsupported layout mode {mode!r}.")
    if root_layout not in _VALID_ROOT_LAYOUTS:
        raise ValueError(f"Unsupported root layout {root_layout!r}.")
    if layout_family not in _VALID_LAYOUT_FAMILIES:
        raise ValueError(f"Unsupported 2D layout family {layout_family!r}.")
    layout_family = _LAYOUT_FAMILY_ALIASES.get(layout_family, layout_family)

    layout_specs = _build_layout_specs(morpho)
    if layout_family == "balloon":
        return _build_layout_branches_balloon(
            morpho,
            layout_specs=layout_specs,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
        )
    if layout_family == "radial_360":
        return _build_layout_branches_radial_360(
            morpho,
            layout_specs=layout_specs,
            min_branch_angle_deg=min_branch_angle_deg,
        )
    if root_layout == "legacy":
        return _build_layout_branches_legacy(
            morpho,
            layout_specs=layout_specs,
            min_branch_angle_deg=min_branch_angle_deg,
        )
    if mode == "frustum":
        return _build_layout_branches_stem_linear(
            morpho,
            layout_specs=layout_specs,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
        )
    return _build_layout_branches_stem(
        morpho,
        layout_specs=layout_specs,
        min_branch_angle_deg=min_branch_angle_deg,
        root_layout=root_layout,
    )


def point_on_layout_branch(layout: LayoutBranch2D, x: float) -> np.ndarray:
    point_um, _ = sample_layout_branch(layout, x)
    return point_um


def tangent_on_layout_branch(layout: LayoutBranch2D, x: float) -> np.ndarray:
    _, tangent_um = sample_layout_branch(layout, x)
    return tangent_um


def sample_layout_branch(layout: LayoutBranch2D, x: float) -> tuple[np.ndarray, np.ndarray]:
    if layout.total_length_um <= 0.0:
        return np.asarray(layout.segment_points_um[0], dtype=float), np.asarray(layout.direction_um, dtype=float)

    arc_length_um = float(np.clip(x, 0.0, 1.0)) * layout.total_length_um
    segment_index = int(np.searchsorted(layout.cumulative_lengths_um[1:], arc_length_um, side="right"))
    segment_index = min(max(segment_index, 0), len(layout.segment_directions_um) - 1)
    start_length_um = float(layout.cumulative_lengths_um[segment_index])
    segment_length_um = float(layout.cumulative_lengths_um[segment_index + 1] - start_length_um)
    direction_um = np.asarray(layout.segment_directions_um[segment_index], dtype=float)
    start_um = np.asarray(layout.segment_points_um[segment_index], dtype=float)
    if segment_length_um <= 0.0:
        return start_um, direction_um
    offset_um = arc_length_um - start_length_um
    return start_um + direction_um * offset_um, direction_um


def _build_layout_specs(morpho: Morpho) -> dict[int, _LayoutSpec2D]:
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


def _build_layout_branches_stem_linear(
    morpho: Morpho,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
    root_layout: str,
) -> tuple[LayoutBranch2D, ...]:
    min_branch_angle_rad = _normalize_min_branch_angle_rad(min_branch_angle_deg)
    subtree_path_lengths_um = _path_lengths_um_by_branch(morpho.root)
    branch_order = {branch.index: index for index, branch in enumerate(morpho.branches)}
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
        bend_fraction=_DEFAULT_BEND_FRACTION,
    )
    _layout_children_stem_linear(
        root,
        layout_specs=layout_specs,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
        layouts=layouts,
        min_branch_angle_rad=min_branch_angle_rad,
        root_layout=root_layout,
        stem_depth=0,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _build_layout_branches_stem(
    morpho: Morpho,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
    root_layout: str,
) -> tuple[LayoutBranch2D, ...]:
    min_branch_angle_rad = _normalize_min_branch_angle_rad(min_branch_angle_deg)
    subtree_path_lengths_um = _path_lengths_um_by_branch(morpho.root)
    branch_order = {branch.index: index for index, branch in enumerate(morpho.branches)}
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
        bend_fraction=_DEFAULT_BEND_FRACTION,
    )
    _layout_children_stem(
        root,
        layout_specs=layout_specs,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
        layouts=layouts,
        min_branch_angle_rad=min_branch_angle_rad,
        root_layout=root_layout,
        stem_depth=0,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _build_layout_branches_balloon(
    morpho: Morpho,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
    root_layout: str,
) -> tuple[LayoutBranch2D, ...]:
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
        bend_fraction=_BALLOON_BEND_FRACTION,
    )
    _layout_children_balloon(
        root,
        layout_specs=layout_specs,
        leaf_counts=leaf_counts,
        layouts=layouts,
        parent_angle_rad=0.0,
        min_branch_angle_rad=min_branch_angle_rad,
        root_layout=root_layout,
        is_root=True,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _build_layout_branches_radial_360(
    morpho: Morpho,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
) -> tuple[LayoutBranch2D, ...]:
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
        bend_fraction=_RADIAL_BEND_FRACTION,
    )
    _layout_children_radial_360(
        root,
        layout_specs=layout_specs,
        leaf_counts=leaf_counts,
        layouts=layouts,
        interval=(-math.pi, math.pi),
        min_branch_angle_rad=min_branch_angle_rad,
        is_root=True,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _layout_children_stem_linear(
    parent: MorphoBranch,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    subtree_path_lengths_um: dict[int, float],
    branch_order: dict[int, int],
    layouts: dict[int, LayoutBranch2D],
    min_branch_angle_rad: float,
    root_layout: str,
    stem_depth: int,
) -> None:
    children = parent.children
    if not children:
        return

    parent_layout = layouts[parent.index]
    if parent.parent is None:
        angle_assignments = _assign_root_stem_angles(
            children,
            subtree_path_lengths_um=subtree_path_lengths_um,
            root_layout=root_layout,
        )
    else:
        attach_angle_rad_by_child = {
            child.index: _vector_angle_rad(tangent_on_layout_branch(parent_layout, child.parent_x))
            for child in children
        }
        angle_assignments = _assign_child_stem_angles(
            children,
            subtree_path_lengths_um=subtree_path_lengths_um,
            branch_order=branch_order,
            parent_angle_rad_by_child=attach_angle_rad_by_child,
            min_branch_angle_rad=min_branch_angle_rad,
            stem_depth=stem_depth,
        )

    trunk_child = _pick_trunk_child(
        children,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
    )

    attach_um, attach_tangent_um = sample_layout_branch(parent_layout, trunk_child.parent_x)
    layouts[trunk_child.index] = _make_layout_branch(
        trunk_child,
        spec=layout_specs[trunk_child.index],
        attach_um=attach_um,
        attach_angle_rad=_vector_angle_rad(attach_tangent_um),
        target_angle_rad=angle_assignments[trunk_child.index],
        child_x=float(trunk_child.child_x),
        bend_fraction=_DEFAULT_BEND_FRACTION,
    )
    _layout_children_stem_linear(
        trunk_child,
        layout_specs=layout_specs,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
        layouts=layouts,
        min_branch_angle_rad=min_branch_angle_rad,
        root_layout=root_layout,
        stem_depth=stem_depth,
    )

    side_children = [
        child for child in children if child is not trunk_child
    ]
    side_children.sort(key=lambda child: (float(child.parent_x), -subtree_path_lengths_um[child.index], branch_order[child.index]))
    for side_index, child in enumerate(side_children):
        attach_um, attach_tangent_um = sample_layout_branch(parent_layout, child.parent_x)
        attach_angle_rad = _vector_angle_rad(attach_tangent_um)
        candidate_layout = _resolve_side_stem_layout(
            child,
            spec=layout_specs[child.index],
            attach_um=attach_um,
            attach_angle_rad=attach_angle_rad,
            base_target_angle_rad=angle_assignments[child.index],
            side_index=side_index,
            stem_depth=stem_depth + 1,
            min_branch_angle_rad=min_branch_angle_rad,
            existing_layouts=tuple(layouts.values())[-48:],
        )
        layouts[child.index] = candidate_layout
        _layout_children_stem_linear(
            child,
            layout_specs=layout_specs,
            subtree_path_lengths_um=subtree_path_lengths_um,
            branch_order=branch_order,
            layouts=layouts,
            min_branch_angle_rad=min_branch_angle_rad,
            root_layout=root_layout,
            stem_depth=stem_depth + 1,
        )


def _layout_children_stem(
    parent: MorphoBranch,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    subtree_path_lengths_um: dict[int, float],
    branch_order: dict[int, int],
    layouts: dict[int, LayoutBranch2D],
    min_branch_angle_rad: float,
    root_layout: str,
    stem_depth: int,
) -> None:
    children = parent.children
    if not children:
        return

    parent_layout = layouts[parent.index]
    if parent.parent is None:
        angle_assignments = _assign_root_stem_angles(
            children,
            subtree_path_lengths_um=subtree_path_lengths_um,
            root_layout=root_layout,
        )
    else:
        attach_angle_rad_by_child = {
            child.index: _vector_angle_rad(tangent_on_layout_branch(parent_layout, child.parent_x))
            for child in children
        }
        angle_assignments = _assign_child_stem_angles(
            children,
            subtree_path_lengths_um=subtree_path_lengths_um,
            branch_order=branch_order,
            parent_angle_rad_by_child=attach_angle_rad_by_child,
            min_branch_angle_rad=min_branch_angle_rad,
            stem_depth=stem_depth,
        )

    trunk_child = _pick_trunk_child(
        children,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
    )
    recent_layouts = tuple(layouts.values())[-_STEM_COLLISION_WINDOW:]
    trunk_attach_um, trunk_attach_tangent_um = sample_layout_branch(parent_layout, trunk_child.parent_x)
    trunk_attach_angle_rad = _vector_angle_rad(trunk_attach_tangent_um)
    trunk_preferred_sign = _preferred_turn_sign(
        attach_angle_rad=trunk_attach_angle_rad,
        target_angle_rad=angle_assignments[trunk_child.index],
        fallback_sign=1.0 if stem_depth % 2 == 0 else -1.0,
    )
    layouts[trunk_child.index] = _resolve_trunk_stem_tree_layout(
        trunk_child,
        spec=layout_specs[trunk_child.index],
        attach_um=trunk_attach_um,
        attach_angle_rad=trunk_attach_angle_rad,
        desired_tail_angle_rad=angle_assignments[trunk_child.index],
        preferred_sign=trunk_preferred_sign,
        min_branch_angle_rad=min_branch_angle_rad,
        existing_layouts=recent_layouts,
    )
    _layout_children_stem(
        trunk_child,
        layout_specs=layout_specs,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
        layouts=layouts,
        min_branch_angle_rad=min_branch_angle_rad,
        root_layout=root_layout,
        stem_depth=stem_depth,
    )

    side_children = [child for child in children if child is not trunk_child]
    side_children.sort(key=lambda child: (float(child.parent_x), -subtree_path_lengths_um[child.index], branch_order[child.index]))
    base_side_sign = 1.0 if stem_depth % 2 == 0 else -1.0
    for side_index, child in enumerate(side_children):
        attach_um, attach_tangent_um = sample_layout_branch(parent_layout, child.parent_x)
        attach_angle_rad = _vector_angle_rad(attach_tangent_um)
        fallback_sign = base_side_sign if side_index % 2 == 0 else -base_side_sign
        preferred_sign = _preferred_turn_sign(
            attach_angle_rad=attach_angle_rad,
            target_angle_rad=angle_assignments[child.index],
            fallback_sign=fallback_sign,
        )
        candidate_layout = _resolve_side_stem_tree_layout(
            child,
            spec=layout_specs[child.index],
            attach_um=attach_um,
            attach_angle_rad=attach_angle_rad,
            desired_tail_angle_rad=angle_assignments[child.index],
            preferred_sign=preferred_sign,
            min_branch_angle_rad=min_branch_angle_rad,
            existing_layouts=tuple(layouts.values())[-_STEM_COLLISION_WINDOW:],
        )
        layouts[child.index] = candidate_layout
        _layout_children_stem(
            child,
            layout_specs=layout_specs,
            subtree_path_lengths_um=subtree_path_lengths_um,
            branch_order=branch_order,
            layouts=layouts,
            min_branch_angle_rad=min_branch_angle_rad,
            root_layout=root_layout,
            stem_depth=stem_depth + 1,
        )


def _layout_children_balloon(
    parent: MorphoBranch,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    leaf_counts: dict[int, int],
    layouts: dict[int, LayoutBranch2D],
    parent_angle_rad: float,
    min_branch_angle_rad: float,
    root_layout: str,
    is_root: bool,
) -> None:
    children = parent.children
    if not children:
        return

    angle_assignments = _assign_balloon_child_angles(
        children,
        leaf_counts=leaf_counts,
        parent_angle_rad=parent_angle_rad,
        min_branch_angle_rad=min_branch_angle_rad,
        root_layout=root_layout,
        is_root=is_root,
    )
    parent_layout = layouts[parent.index]
    for child in children:
        attach_um, attach_tangent_um = sample_layout_branch(parent_layout, child.parent_x)
        child_angle_rad = angle_assignments[child.index]
        layouts[child.index] = _make_layout_branch(
            child,
            spec=layout_specs[child.index],
            attach_um=attach_um,
            attach_angle_rad=_vector_angle_rad(attach_tangent_um),
            target_angle_rad=child_angle_rad,
            child_x=float(child.child_x),
            bend_fraction=_BALLOON_BEND_FRACTION,
        )
        _layout_children_balloon(
            child,
            layout_specs=layout_specs,
            leaf_counts=leaf_counts,
            layouts=layouts,
            parent_angle_rad=child_angle_rad,
            min_branch_angle_rad=min_branch_angle_rad,
            root_layout=root_layout,
            is_root=False,
        )


def _layout_children_radial_360(
    parent: MorphoBranch,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    leaf_counts: dict[int, int],
    layouts: dict[int, LayoutBranch2D],
    interval: tuple[float, float],
    min_branch_angle_rad: float,
    is_root: bool,
) -> None:
    children = parent.children
    if not children:
        return

    if is_root:
        child_intervals = _weighted_child_intervals(children, interval=interval, weights=leaf_counts, min_gap_rad=min_branch_angle_rad)
    else:
        child_intervals = _weighted_child_intervals(children, interval=interval, weights=leaf_counts, min_gap_rad=min_branch_angle_rad)

    parent_layout = layouts[parent.index]
    for child, child_interval in child_intervals:
        attach_um, attach_tangent_um = sample_layout_branch(parent_layout, child.parent_x)
        child_angle_rad = 0.5 * (child_interval[0] + child_interval[1])
        layouts[child.index] = _make_layout_branch(
            child,
            spec=layout_specs[child.index],
            attach_um=attach_um,
            attach_angle_rad=_vector_angle_rad(attach_tangent_um),
            target_angle_rad=child_angle_rad,
            child_x=float(child.child_x),
            bend_fraction=_RADIAL_BEND_FRACTION,
        )
        child_span_rad = min(_RADIAL_CHILD_SPAN_RAD, max(child_interval[1] - child_interval[0], math.radians(40.0)))
        _layout_children_radial_360(
            child,
            layout_specs=layout_specs,
            leaf_counts=leaf_counts,
            layouts=layouts,
            interval=(child_angle_rad - child_span_rad / 2.0, child_angle_rad + child_span_rad / 2.0),
            min_branch_angle_rad=min_branch_angle_rad,
            is_root=False,
        )


def _assign_root_stem_angles(
    children: tuple[MorphoBranch, ...],
    *,
    subtree_path_lengths_um: dict[int, float],
    root_layout: str,
) -> dict[int, float]:
    if root_layout == "type_split":
        axon_children = tuple(child for child in children if child.type == "axon")
        dend_children = tuple(child for child in children if child.type != "axon")
        if axon_children and dend_children:
            assignments: dict[int, float] = {}
            assignments.update(
                _allocate_weighted_angles(
                    axon_children,
                    interval=(
                        _AXON_ROOT_BASE_ANGLE_RAD - _STEM_ROOT_GROUP_SPAN_RAD / 2.0,
                        _AXON_ROOT_BASE_ANGLE_RAD + _STEM_ROOT_GROUP_SPAN_RAD / 2.0,
                    ),
                    weights=subtree_path_lengths_um,
                )
            )
            assignments.update(
                _allocate_weighted_angles(
                    dend_children,
                    interval=(
                        _DENDRITE_ROOT_BASE_ANGLE_RAD - _STEM_ROOT_GROUP_SPAN_RAD / 2.0,
                        _DENDRITE_ROOT_BASE_ANGLE_RAD + _STEM_ROOT_GROUP_SPAN_RAD / 2.0,
                    ),
                    weights=subtree_path_lengths_um,
                )
            )
            return assignments
    return _allocate_weighted_angles(
        children,
        interval=(-_STEM_ROOT_FULL_SPAN_RAD / 2.0, _STEM_ROOT_FULL_SPAN_RAD / 2.0),
        weights=subtree_path_lengths_um,
    )


def _assign_child_stem_angles(
    children: tuple[MorphoBranch, ...],
    *,
    subtree_path_lengths_um: dict[int, float],
    branch_order: dict[int, int],
    parent_angle_rad_by_child: dict[int, float],
    min_branch_angle_rad: float,
    stem_depth: int,
) -> dict[int, float]:
    trunk_child = _pick_trunk_child(
        children,
        subtree_path_lengths_um=subtree_path_lengths_um,
        branch_order=branch_order,
    )
    assignments: dict[int, float] = {}
    assignments[trunk_child.index] = _resolve_trunk_child_angle(
        trunk_child,
        parent_angle_rad=parent_angle_rad_by_child[trunk_child.index],
        min_branch_angle_rad=min_branch_angle_rad,
    )
    side_children = [child for child in children if child is not trunk_child]
    side_children.sort(key=lambda child: (float(child.parent_x), -subtree_path_lengths_um[child.index], branch_order[child.index]))
    base_offset_rad = max(min_branch_angle_rad, math.radians(48.0))
    step_rad = max(min_branch_angle_rad * 0.75, math.radians(14.0))
    stem_side = 1.0 if stem_depth % 2 == 0 else -1.0
    for side_index, child in enumerate(side_children):
        slot = side_index // 2
        sign = stem_side if side_index % 2 == 0 else -stem_side
        assignments[child.index] = parent_angle_rad_by_child[child.index] + sign * (base_offset_rad + slot * step_rad)
    return assignments


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
        width_rad = available_span_rad * weight / total_weight if total_weight > 0.0 else available_span_rad / len(children)
        child_interval = (cursor_rad, cursor_rad + width_rad)
        child_intervals.append((child, child_interval))
        cursor_rad = child_interval[1]
        if child_index != len(children) - 1:
            cursor_rad += min_gap_rad
    return child_intervals


def _resolve_side_stem_layout(
    child: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    attach_angle_rad: float,
    base_target_angle_rad: float,
    side_index: int,
    stem_depth: int,
    min_branch_angle_rad: float,
    existing_layouts: tuple[LayoutBranch2D, ...],
) -> LayoutBranch2D:
    preferred_sign = 1.0 if side_index % 2 == 0 else -1.0
    if stem_depth % 2 == 1:
        preferred_sign *= -1.0
    base_offset_rad = max(abs(_shortest_angle_delta_rad(base_target_angle_rad - attach_angle_rad)), max(min_branch_angle_rad, math.radians(42.0)))
    candidate_signs = (preferred_sign, -preferred_sign)
    candidate_offsets_rad = (
        base_offset_rad,
        base_offset_rad + math.radians(12.0),
        max(base_offset_rad - math.radians(10.0), math.radians(20.0)),
        base_offset_rad + math.radians(24.0),
    )
    candidate_bend_fractions = (0.32, 0.22, 0.45, 0.28)
    best_layout: LayoutBranch2D | None = None
    best_score = float("inf")
    attempts = 0
    for sign in candidate_signs:
        for offset_rad in candidate_offsets_rad:
            for bend_fraction in candidate_bend_fractions:
                target_angle_rad = attach_angle_rad + sign * offset_rad
                candidate_layout = _make_layout_branch(
                    child,
                    spec=spec,
                    attach_um=attach_um,
                    attach_angle_rad=attach_angle_rad,
                    target_angle_rad=target_angle_rad,
                    child_x=float(child.child_x),
                    bend_fraction=bend_fraction,
                )
                score = _layout_collision_score(candidate_layout, existing_layouts)
                if score < best_score:
                    best_score = score
                    best_layout = candidate_layout
                if score <= 0.0:
                    return candidate_layout
                attempts += 1
                if attempts >= _COLLISION_RETRY_LIMIT:
                    return best_layout if best_layout is not None else candidate_layout
    if best_layout is None:
        raise RuntimeError("Failed to resolve side stem layout candidate.")
    return best_layout


def _resolve_trunk_stem_tree_layout(
    child: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    attach_angle_rad: float,
    desired_tail_angle_rad: float,
    preferred_sign: float,
    min_branch_angle_rad: float,
    existing_layouts: tuple[LayoutBranch2D, ...],
) -> LayoutBranch2D:
    return _resolve_stem_tree_layout(
        child,
        spec=spec,
        attach_um=attach_um,
        attach_angle_rad=attach_angle_rad,
        desired_tail_angle_rad=desired_tail_angle_rad,
        preferred_sign=preferred_sign,
        min_branch_angle_rad=min_branch_angle_rad,
        existing_layouts=existing_layouts,
        branch_role="trunk",
    )


def _resolve_side_stem_tree_layout(
    child: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    attach_angle_rad: float,
    desired_tail_angle_rad: float,
    preferred_sign: float,
    min_branch_angle_rad: float,
    existing_layouts: tuple[LayoutBranch2D, ...],
) -> LayoutBranch2D:
    return _resolve_stem_tree_layout(
        child,
        spec=spec,
        attach_um=attach_um,
        attach_angle_rad=attach_angle_rad,
        desired_tail_angle_rad=desired_tail_angle_rad,
        preferred_sign=preferred_sign,
        min_branch_angle_rad=min_branch_angle_rad,
        existing_layouts=existing_layouts,
        branch_role="side",
    )


def _resolve_stem_tree_layout(
    child: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    attach_angle_rad: float,
    desired_tail_angle_rad: float,
    preferred_sign: float,
    min_branch_angle_rad: float,
    existing_layouts: tuple[LayoutBranch2D, ...],
    branch_role: str,
) -> LayoutBranch2D:
    if branch_role not in {"trunk", "side"}:
        raise ValueError(f"Unsupported stem branch_role {branch_role!r}.")

    best_layout: LayoutBranch2D | None = None
    best_score = float("inf")
    desired_tail_delta_rad = _shortest_angle_delta_rad(desired_tail_angle_rad - attach_angle_rad)
    for profile in _stem_profile_candidates(
        attach_angle_rad=attach_angle_rad,
        desired_tail_angle_rad=desired_tail_angle_rad,
        preferred_sign=preferred_sign,
        min_branch_angle_rad=min_branch_angle_rad,
        branch_role=branch_role,
        n_segments=len(spec.segment_lengths_um),
    ):
        candidate_layout = _make_stem_tree_branch(
            child,
            spec=spec,
            attach_um=attach_um,
            child_x=float(child.child_x),
            profile=profile,
        )
        collision_score = _layout_collision_score(candidate_layout, existing_layouts)
        turn_span_rad = abs(_shortest_angle_delta_rad(profile.launch_angle_rad - profile.tail_angle_rad))
        tail_delta_rad = abs(_shortest_angle_delta_rad(profile.tail_angle_rad - desired_tail_angle_rad))
        settle_delta_rad = abs(_shortest_angle_delta_rad(profile.settle_angle_rad - profile.tail_angle_rad))
        launch_delta_rad = abs(_shortest_angle_delta_rad(profile.launch_angle_rad - attach_angle_rad))
        total_score = collision_score * 100.0
        total_score += 3.0 * tail_delta_rad
        total_score += 0.8 * settle_delta_rad
        if turn_span_rad > (math.pi / 2.0):
            total_score += 6.0 * (turn_span_rad - math.pi / 2.0)
        if branch_role == "trunk":
            total_score += 0.75 * abs(_shortest_angle_delta_rad(profile.tail_angle_rad - attach_angle_rad))
        else:
            target_opening_rad = max(abs(desired_tail_delta_rad), math.radians(55.0))
            if launch_delta_rad < target_opening_rad:
                total_score += 2.0 * (target_opening_rad - launch_delta_rad)
        if total_score < best_score:
            best_score = total_score
            best_layout = candidate_layout
            if collision_score <= 0.0 and tail_delta_rad <= math.radians(35.0):
                break

    if best_layout is None:
        raise RuntimeError("Failed to resolve stem tree layout candidate.")
    return best_layout


def _stem_profile_candidates(
    *,
    attach_angle_rad: float,
    desired_tail_angle_rad: float,
    preferred_sign: float,
    min_branch_angle_rad: float,
    branch_role: str,
    n_segments: int,
) -> tuple[_StemAngleProfile, ...]:
    desired_tail_offset_rad = _shortest_angle_delta_rad(desired_tail_angle_rad - attach_angle_rad)
    desired_sign = 1.0 if desired_tail_offset_rad >= 0.0 else -1.0
    if abs(desired_tail_offset_rad) < math.radians(6.0):
        desired_sign = preferred_sign

    if branch_role == "trunk":
        sign_sequence = (desired_sign,)
        tail_offsets_rad = (
            desired_tail_offset_rad,
            desired_sign * max(min_branch_angle_rad * 0.5, math.radians(12.0)),
        )
        launch_magnitudes_rad = (
            max(abs(desired_tail_offset_rad), math.radians(18.0)),
            max(abs(desired_tail_offset_rad) + math.radians(40.0), math.radians(58.0)),
        )
        settle_ratios = (0.55, 0.35)
    else:
        sign_sequence = (preferred_sign, -preferred_sign)
        base_tail_magnitude_rad = max(abs(desired_tail_offset_rad), max(min_branch_angle_rad, math.radians(32.0)))
        tail_offsets_rad = (
            preferred_sign * base_tail_magnitude_rad,
            desired_tail_offset_rad,
        )
        launch_magnitudes_rad = (math.radians(75.0), math.radians(105.0), math.radians(60.0))
        settle_ratios = (0.55, 0.35, 0.18)

    sign_sequence = tuple(dict.fromkeys(1.0 if sign >= 0.0 else -1.0 for sign in sign_sequence))
    if n_segments <= 1:
        launch_magnitudes_rad = (0.0,)
        settle_ratios = (0.0,)
    elif n_segments == 2:
        settle_ratios = (0.35,)
        launch_magnitudes_rad = launch_magnitudes_rad[:2]
    else:
        launch_magnitudes_rad = launch_magnitudes_rad[:2]
        settle_ratios = settle_ratios[:2]

    profiles: list[_StemAngleProfile] = []
    seen_keys: set[tuple[int, int, int]] = set()
    for sign in sign_sequence:
        if sign == 0.0:
            continue
        for tail_offset_rad in tail_offsets_rad:
            if branch_role == "side":
                tail_angle_rad = attach_angle_rad + sign * abs(tail_offset_rad)
            else:
                tail_angle_rad = attach_angle_rad + tail_offset_rad
            for launch_magnitude_rad in launch_magnitudes_rad:
                launch_angle_rad = attach_angle_rad + sign * launch_magnitude_rad
                for settle_ratio in settle_ratios:
                    settle_angle_rad = _blend_angle_rad(tail_angle_rad, launch_angle_rad, settle_ratio)
                    key = (
                        round(_shortest_angle_delta_rad(launch_angle_rad) * 1000.0),
                        round(_shortest_angle_delta_rad(settle_angle_rad) * 1000.0),
                        round(_shortest_angle_delta_rad(tail_angle_rad) * 1000.0),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    profiles.append(
                        _StemAngleProfile(
                            launch_angle_rad=launch_angle_rad,
                            settle_angle_rad=settle_angle_rad,
                            tail_angle_rad=tail_angle_rad,
                        )
                    )
    return tuple(profiles)


def _preferred_turn_sign(
    *,
    attach_angle_rad: float,
    target_angle_rad: float,
    fallback_sign: float,
) -> float:
    delta_rad = _shortest_angle_delta_rad(target_angle_rad - attach_angle_rad)
    if abs(delta_rad) >= math.radians(5.0):
        return 1.0 if delta_rad > 0.0 else -1.0
    return 1.0 if fallback_sign >= 0.0 else -1.0


def _layout_collision_score(candidate: LayoutBranch2D, existing_layouts: tuple[LayoutBranch2D, ...]) -> float:
    score = 0.0
    candidate_min_um = np.min(candidate.segment_points_um, axis=0) - _COLLISION_MARGIN_UM
    candidate_max_um = np.max(candidate.segment_points_um, axis=0) + _COLLISION_MARGIN_UM
    for existing in existing_layouts:
        existing_min_um = np.min(existing.segment_points_um, axis=0) - _COLLISION_MARGIN_UM
        existing_max_um = np.max(existing.segment_points_um, axis=0) + _COLLISION_MARGIN_UM
        if np.any(candidate_max_um < existing_min_um) or np.any(existing_max_um < candidate_min_um):
            continue
        score += _polyline_collision_score(
            candidate.segment_points_um,
            existing.segment_points_um,
            margin_um=_COLLISION_MARGIN_UM,
        )
    return score


def _polyline_collision_score(points_a_um: np.ndarray, points_b_um: np.ndarray, *, margin_um: float) -> float:
    score = 0.0
    for segment_a_index in range(len(points_a_um) - 1):
        a0 = points_a_um[segment_a_index]
        a1 = points_a_um[segment_a_index + 1]
        for segment_b_index in range(len(points_b_um) - 1):
            b0 = points_b_um[segment_b_index]
            b1 = points_b_um[segment_b_index + 1]
            if _segments_share_endpoint(a0, a1, b0, b1):
                continue
            if _segments_intersect(a0, a1, b0, b1):
                score += 1000.0
                continue
            distance_um = _segment_distance_um(a0, a1, b0, b1)
            if distance_um < margin_um:
                score += margin_um - distance_um
    return score


def _segments_share_endpoint(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
    endpoint_pairs = ((a0, b0), (a0, b1), (a1, b0), (a1, b1))
    return any(np.linalg.norm(point_a - point_b) <= 1e-6 for point_a, point_b in endpoint_pairs)


def _segments_intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orientation(a0, a1, b0)
    o2 = orientation(a0, a1, b1)
    o3 = orientation(b0, b1, a0)
    o4 = orientation(b0, b1, a1)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _segment_distance_um(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    return min(
        _point_to_segment_distance_um(a0, b0, b1),
        _point_to_segment_distance_um(a1, b0, b1),
        _point_to_segment_distance_um(b0, a0, a1),
        _point_to_segment_distance_um(b1, a0, a1),
    )


def _point_to_segment_distance_um(point_um: np.ndarray, seg0_um: np.ndarray, seg1_um: np.ndarray) -> float:
    seg_vec_um = seg1_um - seg0_um
    seg_len_sq_um = float(np.dot(seg_vec_um, seg_vec_um))
    if seg_len_sq_um <= 0.0:
        return float(np.linalg.norm(point_um - seg0_um))
    projection = float(np.dot(point_um - seg0_um, seg_vec_um) / seg_len_sq_um)
    projection = min(max(projection, 0.0), 1.0)
    closest_um = seg0_um + projection * seg_vec_um
    return float(np.linalg.norm(point_um - closest_um))


def _assign_balloon_child_angles(
    children: tuple[MorphoBranch, ...],
    *,
    leaf_counts: dict[int, int],
    parent_angle_rad: float,
    min_branch_angle_rad: float,
    root_layout: str,
    is_root: bool,
) -> dict[int, float]:
    if is_root and root_layout == "type_split":
        axon_children = tuple(child for child in children if child.type == "axon")
        dend_children = tuple(child for child in children if child.type != "axon")
        if axon_children and dend_children:
            assignments: dict[int, float] = {}
            assignments.update(
                _allocate_balloon_group_angles(
                    axon_children,
                    leaf_counts=leaf_counts,
                    interval=(
                        _AXON_ROOT_BASE_ANGLE_RAD - _BALLOON_TYPE_SPLIT_SPAN_RAD / 2.0,
                        _AXON_ROOT_BASE_ANGLE_RAD + _BALLOON_TYPE_SPLIT_SPAN_RAD / 2.0,
                    ),
                    min_branch_angle_rad=min_branch_angle_rad,
                )
            )
            assignments.update(
                _allocate_balloon_group_angles(
                    dend_children,
                    leaf_counts=leaf_counts,
                    interval=(
                        _DENDRITE_ROOT_BASE_ANGLE_RAD - _BALLOON_TYPE_SPLIT_SPAN_RAD / 2.0,
                        _DENDRITE_ROOT_BASE_ANGLE_RAD + _BALLOON_TYPE_SPLIT_SPAN_RAD / 2.0,
                    ),
                    min_branch_angle_rad=min_branch_angle_rad,
                )
            )
            return assignments

    span_rad = _BALLOON_ROOT_SPAN_RAD if is_root else _BALLOON_CHILD_SPAN_RAD
    return _allocate_balloon_group_angles(
        children,
        leaf_counts=leaf_counts,
        interval=(parent_angle_rad - span_rad / 2.0, parent_angle_rad + span_rad / 2.0),
        min_branch_angle_rad=min_branch_angle_rad,
    )


def _allocate_balloon_group_angles(
    children: tuple[MorphoBranch, ...],
    *,
    leaf_counts: dict[int, int],
    interval: tuple[float, float],
    min_branch_angle_rad: float,
) -> dict[int, float]:
    allocations = _allocate_child_regions_legacy(
        children=children,
        interval=interval,
        leaf_counts=leaf_counts,
        min_branch_angle_rad=min_branch_angle_rad,
    )
    return {child.index: child_angle for child, _, child_angle in allocations}


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


def _build_layout_branches_legacy(
    morpho: Morpho,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
) -> tuple[LayoutBranch2D, ...]:
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
        bend_fraction=_DEFAULT_BEND_FRACTION,
    )
    _layout_children_legacy(
        root,
        layout_specs=layout_specs,
        leaf_counts=leaf_counts,
        layouts=layouts,
        interval=(-_ROOT_CHILD_SPAN_RAD / 2.0, _ROOT_CHILD_SPAN_RAD / 2.0),
        min_branch_angle_rad=min_branch_angle_rad,
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
) -> None:
    children = parent.children
    if not children:
        return

    allocations = _allocate_child_regions_legacy(
        children=children,
        interval=interval,
        leaf_counts=leaf_counts,
        min_branch_angle_rad=min_branch_angle_rad,
    )
    parent_layout = layouts[parent.index]
    for child, child_interval, child_angle in allocations:
        attach_um, attach_tangent_um = sample_layout_branch(parent_layout, child.parent_x)
        layouts[child.index] = _make_layout_branch(
            child,
            spec=layout_specs[child.index],
            attach_um=attach_um,
            attach_angle_rad=_vector_angle_rad(attach_tangent_um),
            target_angle_rad=child_angle,
            child_x=float(child.child_x),
            bend_fraction=_DEFAULT_BEND_FRACTION,
        )
        _layout_children_legacy(
            child,
            layout_specs=layout_specs,
            leaf_counts=leaf_counts,
            layouts=layouts,
            interval=child_interval,
            min_branch_angle_rad=min_branch_angle_rad,
        )


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


def _make_layout_branch(
    branch: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    attach_angle_rad: float,
    target_angle_rad: float,
    child_x: float,
    bend_fraction: float,
) -> LayoutBranch2D:
    segment_angles_rad = _segment_angles_rad(
        spec.segment_lengths_um,
        attach_angle_rad=attach_angle_rad,
        target_angle_rad=target_angle_rad,
        bend_fraction=bend_fraction,
    )
    return _layout_branch_from_angles(
        branch,
        spec=spec,
        attach_um=attach_um,
        child_x=child_x,
        segment_angles_rad=segment_angles_rad,
    )


def _make_stem_tree_branch(
    branch: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    child_x: float,
    profile: _StemAngleProfile,
) -> LayoutBranch2D:
    segment_angles_rad = _stem_segment_angles_rad(
        spec.segment_lengths_um,
        launch_angle_rad=profile.launch_angle_rad,
        settle_angle_rad=profile.settle_angle_rad,
        tail_angle_rad=profile.tail_angle_rad,
    )
    return _layout_branch_from_angles(
        branch,
        spec=spec,
        attach_um=attach_um,
        child_x=child_x,
        segment_angles_rad=segment_angles_rad,
    )


def _layout_branch_from_angles(
    branch: MorphoBranch,
    *,
    spec: _LayoutSpec2D,
    attach_um: np.ndarray,
    child_x: float,
    segment_angles_rad: np.ndarray,
) -> LayoutBranch2D:
    segment_directions_um = np.column_stack((np.cos(segment_angles_rad), np.sin(segment_angles_rad)))
    segment_normals_um = np.column_stack((-segment_directions_um[:, 1], segment_directions_um[:, 0]))
    cumulative_lengths_um = np.concatenate(([0.0], np.cumsum(spec.segment_lengths_um)))
    raw_segment_points_um = np.zeros((len(spec.segment_lengths_um) + 1, 2), dtype=float)
    for segment_index, segment_length_um in enumerate(spec.segment_lengths_um):
        raw_segment_points_um[segment_index + 1] = (
            raw_segment_points_um[segment_index]
            + segment_directions_um[segment_index] * float(segment_length_um)
        )

    raw_layout = LayoutBranch2D(
        branch_index=branch.index,
        branch_name=branch.name,
        branch_type=branch.type,
        segment_points_um=raw_segment_points_um,
        radii_proximal_um=spec.radii_proximal_um,
        radii_distal_um=spec.radii_distal_um,
        total_length_um=spec.total_length_um,
        segment_directions_um=segment_directions_um,
        segment_normals_um=segment_normals_um,
        cumulative_lengths_um=cumulative_lengths_um,
    )
    attach_point_um = point_on_layout_branch(raw_layout, child_x)
    translation_um = np.asarray(attach_um, dtype=float) - attach_point_um
    return _layout_branch_from_points(branch, spec, raw_segment_points_um + translation_um)


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
        raw_segment_points_um[segment_index + 1] = (
            raw_segment_points_um[segment_index]
            + np.array([dx_segments_um[segment_index], dy_segments_um[segment_index]], dtype=float)
        )

    raw_layout = _layout_branch_from_points(branch, spec, raw_segment_points_um)
    attach_point_um = point_on_layout_branch(raw_layout, child_x)
    translation_um = np.asarray(attach_um, dtype=float) - attach_point_um
    return _layout_branch_from_points(branch, spec, raw_segment_points_um + translation_um)


def _layout_branch_from_points(
    branch: MorphoBranch,
    spec: _LayoutSpec2D,
    segment_points_um: np.ndarray,
) -> LayoutBranch2D:
    segment_points_um = np.asarray(segment_points_um, dtype=float)
    segment_vectors_um = np.diff(segment_points_um, axis=0)
    segment_lengths_um = np.linalg.norm(segment_vectors_um, axis=1)
    safe_lengths_um = np.where(segment_lengths_um > 0.0, segment_lengths_um, 1.0)
    segment_directions_um = segment_vectors_um / safe_lengths_um[:, None]
    segment_directions_um[segment_lengths_um == 0.0] = np.array([1.0, 0.0], dtype=float)
    segment_normals_um = np.column_stack((-segment_directions_um[:, 1], segment_directions_um[:, 0]))
    cumulative_lengths_um = np.concatenate(([0.0], np.cumsum(segment_lengths_um)))
    return LayoutBranch2D(
        branch_index=branch.index,
        branch_name=branch.name,
        branch_type=branch.type,
        segment_points_um=segment_points_um,
        radii_proximal_um=spec.radii_proximal_um,
        radii_distal_um=spec.radii_distal_um,
        total_length_um=float(cumulative_lengths_um[-1]),
        segment_directions_um=segment_directions_um,
        segment_normals_um=segment_normals_um,
        cumulative_lengths_um=cumulative_lengths_um,
    )


def _horizontal_segment_points_um(segment_lengths_um: np.ndarray, *, start_um: np.ndarray) -> np.ndarray:
    start_um = np.asarray(start_um, dtype=float)
    cumulative_x_um = np.concatenate(([0.0], np.cumsum(segment_lengths_um)))
    return np.column_stack(
        (
            start_um[0] + cumulative_x_um,
            np.full(len(cumulative_x_um), start_um[1], dtype=float),
        )
    )


def _dendrogram_y_units_by_branch(
    root: MorphoBranch,
    *,
    root_layout: str,
) -> tuple[dict[int, float], dict[int, float]]:
    ordered_leaves = _ordered_leaf_branches(root, root_layout=root_layout)
    leaf_positions = {
        leaf.index: float(position - (len(ordered_leaves) - 1) / 2.0)
        for position, leaf in enumerate(ordered_leaves)
    }
    branch_positions: dict[int, float] = {}

    def visit(node: MorphoBranch) -> float:
        if node.n_children == 0:
            branch_positions[node.index] = leaf_positions[node.index]
            return branch_positions[node.index]
        child_positions = [visit(child) for child in node.children]
        branch_positions[node.index] = float(np.mean(child_positions))
        return branch_positions[node.index]

    visit(root)
    return branch_positions, leaf_positions


def _ordered_leaf_branches(root: MorphoBranch, *, root_layout: str) -> list[MorphoBranch]:
    if root_layout == "type_split":
        axon_children = tuple(child for child in root.children if child.type == "axon")
        dend_children = tuple(child for child in root.children if child.type != "axon")
        if axon_children and dend_children:
            ordered_children = tuple(reversed(axon_children)) + dend_children
            return [leaf for child in ordered_children for leaf in _leaf_branches_dfs(child)]
    return _leaf_branches_dfs(root)


def _leaf_branches_dfs(node: MorphoBranch) -> list[MorphoBranch]:
    if node.n_children == 0:
        return [node]
    leaves: list[MorphoBranch] = []
    for child in node.children:
        leaves.extend(_leaf_branches_dfs(child))
    return leaves


def _dendrogram_unit_scale_um(morpho: Morpho, y_units_by_branch: dict[int, float]) -> float:
    max_ratio = 0.0
    for branch in morpho.branches:
        parent = branch.parent
        if parent is None:
            continue
        branch_length_um = float(branch.length.to_decimal(u.um))
        if branch_length_um <= 0.0:
            continue
        unit_dy = abs(y_units_by_branch[branch.index] - y_units_by_branch[parent.index])
        max_ratio = max(max_ratio, unit_dy / branch_length_um)
    if max_ratio <= 0.0:
        return 1.0
    return 0.6 / max_ratio


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * values * (3.0 - 2.0 * values)


def _blend_angle_rad(angle_a_rad: float, angle_b_rad: float, weight_b: float) -> float:
    weight_b = float(np.clip(weight_b, 0.0, 1.0))
    return angle_a_rad + weight_b * _shortest_angle_delta_rad(angle_b_rad - angle_a_rad)


def _segment_angles_rad(
    segment_lengths_um: np.ndarray,
    *,
    attach_angle_rad: float,
    target_angle_rad: float,
    bend_fraction: float,
) -> np.ndarray:
    total_length_um = float(np.sum(segment_lengths_um))
    if total_length_um <= 0.0:
        return np.full(len(segment_lengths_um), target_angle_rad, dtype=float)
    if len(segment_lengths_um) == 1:
        return np.array([target_angle_rad], dtype=float)

    angle_delta_rad = _shortest_angle_delta_rad(target_angle_rad - attach_angle_rad)
    bend_fraction = float(np.clip(bend_fraction, 1e-3, 1.0))
    cumulative_mid_fraction = (np.cumsum(segment_lengths_um) - 0.5 * segment_lengths_um) / total_length_um
    segment_angles_rad: list[float] = []
    for segment_index, mid_fraction in enumerate(cumulative_mid_fraction):
        if segment_index == 0:
            segment_angles_rad.append(attach_angle_rad)
            continue
        if mid_fraction >= bend_fraction:
            segment_angles_rad.append(target_angle_rad)
        else:
            segment_angles_rad.append(attach_angle_rad + angle_delta_rad * _smoothstep(mid_fraction / bend_fraction))
    return np.asarray(segment_angles_rad, dtype=float)


def _stem_segment_angles_rad(
    segment_lengths_um: np.ndarray,
    *,
    launch_angle_rad: float,
    settle_angle_rad: float,
    tail_angle_rad: float,
) -> np.ndarray:
    n_segments = len(segment_lengths_um)
    if n_segments <= 0:
        return np.empty(0, dtype=float)
    if n_segments == 1:
        return np.array([tail_angle_rad], dtype=float)
    if n_segments == 2:
        return np.array([launch_angle_rad, tail_angle_rad], dtype=float)
    if n_segments == 3:
        return np.array([launch_angle_rad, settle_angle_rad, tail_angle_rad], dtype=float)

    total_length_um = float(np.sum(segment_lengths_um))
    if total_length_um <= 0.0:
        return np.full(n_segments, tail_angle_rad, dtype=float)

    cumulative_mid_fraction = (np.cumsum(segment_lengths_um) - 0.5 * segment_lengths_um) / total_length_um
    segment_angles_rad: list[float] = []
    for segment_index, mid_fraction in enumerate(cumulative_mid_fraction):
        if segment_index == 0:
            segment_angles_rad.append(launch_angle_rad)
            continue
        if segment_index == n_segments - 1 or mid_fraction >= 0.72:
            segment_angles_rad.append(tail_angle_rad)
            continue
        if mid_fraction <= 0.28:
            segment_angles_rad.append(_blend_angle_rad(launch_angle_rad, settle_angle_rad, _smoothstep(mid_fraction / 0.28)))
            continue
        if mid_fraction <= 0.72:
            weight = _smoothstep((mid_fraction - 0.28) / 0.44)
            segment_angles_rad.append(_blend_angle_rad(settle_angle_rad, tail_angle_rad, weight))
            continue
        segment_angles_rad.append(tail_angle_rad)
    if len(segment_angles_rad) >= 2:
        segment_angles_rad[-2] = tail_angle_rad
    segment_angles_rad[-1] = tail_angle_rad
    return np.asarray(segment_angles_rad, dtype=float)


def _shortest_angle_delta_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _vector_angle_rad(vector_um: np.ndarray) -> float:
    vector_um = np.asarray(vector_um, dtype=float)
    return math.atan2(float(vector_um[1]), float(vector_um[0]))
