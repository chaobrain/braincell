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

"""Stem layout family.

The stem family is the default 2D layout. It picks a "trunk" child at
every fork (the one that continues the longest path) and places
siblings as side branches with alternating signs and collision-aware
angle profiles.

Two flavours exist:

* ``_build_layout_branches_stem`` — the *tree* variant, used for line
  drawings. It scores several three-segment angle profiles (launch,
  settle, tail) per candidate via :mod:`_collision` and picks the
  best.
* ``_build_layout_branches_stem_linear`` — the *frustum* variant. It
  keeps each child on a single-bend angle because the scene builder
  emits one quadrilateral per segment, so extra turns produce visible
  gaps between frustums.

Both share the same root-angle assignment, trunk-child selection, and
sibling ordering helpers defined below.
"""


import math
from dataclasses import dataclass

import numpy as np

from braincell.morph import MorphoBranch, Morphology

from ._collision import _build_collision_index, _layout_collision_score
from ._common import (
    LayoutBranch2D,
    _allocate_weighted_angles,
    _AXON_ROOT_BASE_ANGLE_RAD,
    _DENDRITE_ROOT_BASE_ANGLE_RAD,
    _LayoutSpec2D,
    _normalize_min_branch_angle_rad,
    _path_lengths_um_by_branch,
    _pick_trunk_child,
    _resolve_trunk_child_angle,
)
from ._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig
from ._geometry import (
    _layout_branch_from_angles,
    _make_layout_branch,
    _shortest_angle_delta_rad,
    _blend_angle_rad,
    _smoothstep,
    _vector_angle_rad,
    sample_layout_branch,
    tangent_on_layout_branch,
)


@dataclass(frozen=True)
class _StemAngleProfile:
    launch_angle_rad: float
    settle_angle_rad: float
    tail_angle_rad: float


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _build_layout_branches_stem_linear(
    morpho: Morphology,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
    root_layout: str,
    layout_config: LayoutConfig | None = None,
) -> tuple[LayoutBranch2D, ...]:
    config = layout_config or DEFAULT_LAYOUT_CONFIG
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
        bend_fraction=config.default_bend_fraction,
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
        layout_config=config,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


def _build_layout_branches_stem(
    morpho: Morphology,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
    root_layout: str,
    layout_config: LayoutConfig | None = None,
) -> tuple[LayoutBranch2D, ...]:
    config = layout_config or DEFAULT_LAYOUT_CONFIG
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
        bend_fraction=config.default_bend_fraction,
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
        layout_config=config,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


# ---------------------------------------------------------------------------
# Recursive child placement
# ---------------------------------------------------------------------------

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
    layout_config: LayoutConfig,
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
            layout_config=layout_config,
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
        bend_fraction=layout_config.default_bend_fraction,
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
        layout_config=layout_config,
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
            layout_config=layout_config,
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
            layout_config=layout_config,
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
    layout_config: LayoutConfig,
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
            layout_config=layout_config,
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
    collision_window = layout_config.stem_collision_window
    recent_layouts = tuple(layouts.values())[-collision_window:]
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
        layout_config=layout_config,
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
        layout_config=layout_config,
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
            existing_layouts=tuple(layouts.values())[-collision_window:],
            layout_config=layout_config,
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
            layout_config=layout_config,
        )


# ---------------------------------------------------------------------------
# Angle assignment
# ---------------------------------------------------------------------------

def _assign_root_stem_angles(
    children: tuple[MorphoBranch, ...],
    *,
    subtree_path_lengths_um: dict[int, float],
    root_layout: str,
    layout_config: LayoutConfig | None = None,
) -> dict[int, float]:
    config = layout_config or DEFAULT_LAYOUT_CONFIG
    if root_layout == "type_split":
        axon_children = tuple(child for child in children if child.type == "axon")
        dend_children = tuple(child for child in children if child.type != "axon")
        if axon_children and dend_children:
            assignments: dict[int, float] = {}
            assignments.update(
                _allocate_weighted_angles(
                    axon_children,
                    interval=(
                        _AXON_ROOT_BASE_ANGLE_RAD - config.stem_root_group_span_rad / 2.0,
                        _AXON_ROOT_BASE_ANGLE_RAD + config.stem_root_group_span_rad / 2.0,
                    ),
                    weights=subtree_path_lengths_um,
                )
            )
            assignments.update(
                _allocate_weighted_angles(
                    dend_children,
                    interval=(
                        _DENDRITE_ROOT_BASE_ANGLE_RAD - config.stem_root_group_span_rad / 2.0,
                        _DENDRITE_ROOT_BASE_ANGLE_RAD + config.stem_root_group_span_rad / 2.0,
                    ),
                    weights=subtree_path_lengths_um,
                )
            )
            return assignments
    return _allocate_weighted_angles(
        children,
        interval=(-config.stem_root_full_span_rad / 2.0, config.stem_root_full_span_rad / 2.0),
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


# ---------------------------------------------------------------------------
# Candidate resolution (stem linear / frustum path)
# ---------------------------------------------------------------------------

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
    layout_config: LayoutConfig,
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
    retry_limit = layout_config.collision_retry_limit
    collision_index = _build_collision_index(existing_layouts, layout_config=layout_config)
    margin_um = layout_config.collision_margin_um
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
                score = collision_index.scored_candidate(candidate_layout, margin_um=margin_um)
                if score < best_score:
                    best_score = score
                    best_layout = candidate_layout
                if score <= 0.0:
                    return candidate_layout
                attempts += 1
                if attempts >= retry_limit:
                    return best_layout if best_layout is not None else candidate_layout
    if best_layout is None:
        raise RuntimeError("Failed to resolve side stem layout candidate.")
    return best_layout


# ---------------------------------------------------------------------------
# Candidate resolution (stem tree / line path)
# ---------------------------------------------------------------------------

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
    layout_config: LayoutConfig,
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
        layout_config=layout_config,
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
    layout_config: LayoutConfig,
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
        layout_config=layout_config,
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
    layout_config: LayoutConfig,
) -> LayoutBranch2D:
    if branch_role not in {"trunk", "side"}:
        raise ValueError(f"Unsupported stem branch_role {branch_role!r}.")

    best_layout: LayoutBranch2D | None = None
    best_score = float("inf")
    desired_tail_delta_rad = _shortest_angle_delta_rad(desired_tail_angle_rad - attach_angle_rad)
    collision_weight = layout_config.stem_collision_weight
    tail_delta_weight = layout_config.stem_tail_delta_weight
    settle_delta_weight = layout_config.stem_settle_delta_weight
    overturn_weight = layout_config.stem_overturn_weight
    trunk_tail_delta_weight = layout_config.stem_trunk_tail_delta_weight
    side_opening_weight = layout_config.stem_side_opening_weight
    # Build the spatial hash once per fork so each profile candidate
    # pays O(neighbours) instead of O(existing_segments).
    collision_index = _build_collision_index(existing_layouts, layout_config=layout_config)
    margin_um = layout_config.collision_margin_um
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
        collision_score = collision_index.scored_candidate(candidate_layout, margin_um=margin_um)
        turn_span_rad = abs(_shortest_angle_delta_rad(profile.launch_angle_rad - profile.tail_angle_rad))
        tail_delta_rad = abs(_shortest_angle_delta_rad(profile.tail_angle_rad - desired_tail_angle_rad))
        settle_delta_rad = abs(_shortest_angle_delta_rad(profile.settle_angle_rad - profile.tail_angle_rad))
        launch_delta_rad = abs(_shortest_angle_delta_rad(profile.launch_angle_rad - attach_angle_rad))
        total_score = collision_score * collision_weight
        total_score += tail_delta_weight * tail_delta_rad
        total_score += settle_delta_weight * settle_delta_rad
        if turn_span_rad > (math.pi / 2.0):
            total_score += overturn_weight * (turn_span_rad - math.pi / 2.0)
        if branch_role == "trunk":
            total_score += trunk_tail_delta_weight * abs(_shortest_angle_delta_rad(profile.tail_angle_rad - attach_angle_rad))
        else:
            target_opening_rad = max(abs(desired_tail_delta_rad), math.radians(55.0))
            if launch_delta_rad < target_opening_rad:
                total_score += side_opening_weight * (target_opening_rad - launch_delta_rad)
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


# ---------------------------------------------------------------------------
# Stem-specific segment angle interpolation and branch construction
# ---------------------------------------------------------------------------

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
