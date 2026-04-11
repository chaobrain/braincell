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

"""Geometric primitives for the 2D layout engine.

This module collects the pure-numeric helpers that every layout family
uses to turn a sequence of segment angles + lengths into a
:class:`~braincell.vis.layout._common.LayoutBranch2D` and to sample
points / tangents along an existing branch layout:

* ``point_on_layout_branch`` / ``tangent_on_layout_branch`` /
  ``sample_layout_branch`` — the public sampling API used by scene
  builders and overlays.
* ``_make_layout_branch`` / ``_layout_branch_from_angles`` /
  ``_layout_branch_from_points`` — construction helpers for layout
  families.
* Angle-space utilities (``_segment_angles_rad``, ``_smoothstep``,
  ``_blend_angle_rad``, ``_shortest_angle_delta_rad``,
  ``_vector_angle_rad``).

Nothing in this module mutates state; every function is a pure
transformation over numpy arrays and float scalars.
"""


import math

import numpy as np

from braincell.morph import MorphoBranch

from ._common import LayoutBranch2D, _LayoutSpec2D


# ---------------------------------------------------------------------------
# Sampling along an existing layout branch
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Layout branch construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Angle interpolation helpers
# ---------------------------------------------------------------------------

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


def _shortest_angle_delta_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _vector_angle_rad(vector_um: np.ndarray) -> float:
    vector_um = np.asarray(vector_um, dtype=float)
    return math.atan2(float(vector_um[1]), float(vector_um[0]))
