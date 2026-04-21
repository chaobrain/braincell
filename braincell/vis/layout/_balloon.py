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

"""Balloon layout family.

Arranges each fork's children inside a constrained angular cone
centered on the parent direction. Children are weighted by leaf count,
so a subtree with more leaves occupies a wider angular slice. The name
comes from the characteristic "puffing outwards" shape.
"""


import numpy as np

from braincell.morph import MorphoBranch
from braincell.morph._morphology import Morphology

from ._common import (
    LayoutBranch2D,
    _AXON_ROOT_BASE_ANGLE_RAD,
    _DENDRITE_ROOT_BASE_ANGLE_RAD,
    _LayoutSpec2D,
    _allocate_child_regions_legacy,
    _leaf_counts_by_branch,
    _normalize_min_branch_angle_rad,
)
from ._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig
from ._geometry import (
    _make_layout_branch,
    _vector_angle_rad,
    sample_layout_branch,
)


def _build_layout_branches_balloon(
    morpho: Morphology,
    *,
    layout_specs: dict[int, _LayoutSpec2D],
    min_branch_angle_deg: float | None,
    root_layout: str,
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
        bend_fraction=config.balloon_bend_fraction,
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
        layout_config=config,
    )
    return tuple(layouts[branch.index] for branch in morpho.branches)


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
    layout_config: LayoutConfig,
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
        layout_config=layout_config,
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
            bend_fraction=layout_config.balloon_bend_fraction,
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
            layout_config=layout_config,
        )


def _assign_balloon_child_angles(
    children: tuple[MorphoBranch, ...],
    *,
    leaf_counts: dict[int, int],
    parent_angle_rad: float,
    min_branch_angle_rad: float,
    root_layout: str,
    is_root: bool,
    layout_config: LayoutConfig | None = None,
) -> dict[int, float]:
    config = layout_config or DEFAULT_LAYOUT_CONFIG
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
                        _AXON_ROOT_BASE_ANGLE_RAD - config.balloon_type_split_span_rad / 2.0,
                        _AXON_ROOT_BASE_ANGLE_RAD + config.balloon_type_split_span_rad / 2.0,
                    ),
                    min_branch_angle_rad=min_branch_angle_rad,
                )
            )
            assignments.update(
                _allocate_balloon_group_angles(
                    dend_children,
                    leaf_counts=leaf_counts,
                    interval=(
                        _DENDRITE_ROOT_BASE_ANGLE_RAD - config.balloon_type_split_span_rad / 2.0,
                        _DENDRITE_ROOT_BASE_ANGLE_RAD + config.balloon_type_split_span_rad / 2.0,
                    ),
                    min_branch_angle_rad=min_branch_angle_rad,
                )
            )
            return assignments

    span_rad = config.balloon_root_span_rad if is_root else config.balloon_child_span_rad
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
