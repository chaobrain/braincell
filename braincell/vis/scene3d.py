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


from collections import OrderedDict

import brainunit as u
import numpy as np

from braincell.morph import Morphology
from .config import (
    highlight_alpha as _highlight_alpha,
    highlight_color as _highlight_color,
    marker_color as _marker_color,
    marker_radius_3d_um as _marker_radius_3d_um,
)
from .scene import (
    BranchPolyline3D,
    BranchTypeBatch3D,
    HighlightStroke3D,
    Marker3D,
    OverlaySpec,
    RenderScene3D,
    alpha_for_3d_tube,
    color_for_branch_type,
)


def build_render_scene_3d(
    morpho: Morphology,
    *,
    mode: str = "geometry",
    overlay: OverlaySpec | None = None,
) -> RenderScene3D:
    if not isinstance(morpho, Morphology):
        raise TypeError(f"build_render_scene_3d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if mode not in {"geometry", "skeleton"}:
        raise ValueError(f"Unsupported 3D mode {mode!r}. Expected 'geometry' or 'skeleton'.")
    if not morpho.has_full_point_geometry:
        raise ValueError(
            "3D mode='geometry' requires complete point geometry on every branch. "
            "Use vis2d(layout='stem', shape='line') or vis2d(layout='stem', shape='frustum') "
            "for length/radius-only morphologies."
        )
    _reject_values_overlay(overlay)

    branches: list[BranchPolyline3D] = []
    for branch_index in range(len(morpho)):
        branch_view = morpho.branch(index=branch_index)
        branch = branch_view.branch
        if branch.points_proximal is None or branch.points_distal is None:
            raise ValueError(
                f"Branch {branch_view.name!r} lacks complete 3D point geometry and cannot be rendered with "
                "mode='geometry'. Use layout='stem' with shape='line' or shape='frustum' in 2D instead."
            )
        points_um = np.vstack(
            [
                np.asarray(branch.points_proximal[0].to_decimal(u.um), dtype=float),
                np.asarray(branch.points_distal.to_decimal(u.um), dtype=float),
            ]
        )
        radii_um = np.concatenate(
            [
                np.atleast_1d(np.asarray(branch.radii_proximal[0].to_decimal(u.um), dtype=float)),
                np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float),
            ]
        )
        if points_um.shape[0] != radii_um.shape[0]:
            raise ValueError(
                f"Branch {branch_view.name!r} produced mismatched point/radius arrays for 3D rendering."
            )
        branches.append(
            BranchPolyline3D(
                branch_index=branch_index,
                branch_name=branch_view.name,
                branch_type=branch_view.type,
                points_um=points_um,
                radii_um=radii_um,
            )
        )

    grouped: OrderedDict[str, list[BranchPolyline3D]] = OrderedDict()
    for branch in branches:
        grouped.setdefault(branch.branch_type, []).append(branch)

    batches: list[BranchTypeBatch3D] = []
    for branch_type, type_branches in grouped.items():
        points_all: list[np.ndarray] = []
        radii_all: list[np.ndarray] = []
        lines_all: list[np.ndarray] = []
        branch_indices: list[int] = []
        branch_names: list[str] = []
        offset = 0
        for branch in type_branches:
            n_points = len(branch.points_um)
            cell = np.arange(offset, offset + n_points, dtype=np.int64)
            cell = np.insert(cell, 0, n_points)
            points_all.append(branch.points_um)
            radii_all.append(branch.radii_um)
            lines_all.append(cell)
            branch_indices.append(branch.branch_index)
            branch_names.append(branch.branch_name)
            offset += n_points
        batches.append(
            BranchTypeBatch3D(
                branch_type=branch_type,
                color_rgb=color_for_branch_type(branch_type),
                opacity=alpha_for_3d_tube(),
                branch_indices=tuple(branch_indices),
                branch_names=tuple(branch_names),
                points_um=np.vstack(points_all),
                radii_um=np.concatenate(radii_all),
                lines=np.concatenate(lines_all).astype(np.int64),
            )
        )

    branch_lookup = {branch.branch_index: branch for branch in branches}
    highlight_strokes, markers = _build_overlay_primitives_3d(overlay, branch_lookup)

    return RenderScene3D(
        branches=tuple(branches),
        batches=tuple(batches),
        highlight_strokes=highlight_strokes,
        markers=markers,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Overlay resolution helpers
# ---------------------------------------------------------------------------

def _branch_cumulative_um(branch: BranchPolyline3D) -> np.ndarray:
    diffs = np.diff(branch.points_um, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _point_at_3d(branch: BranchPolyline3D, cumulative_um: np.ndarray, x: float) -> np.ndarray:
    total = float(cumulative_um[-1]) if cumulative_um.size else 0.0
    if total <= 0.0 or branch.points_um.shape[0] == 0:
        return branch.points_um[0].copy() if branch.points_um.size else np.zeros(3, dtype=float)
    clamped = float(np.clip(x, 0.0, 1.0))
    target = clamped * total
    idx = int(np.searchsorted(cumulative_um[1:], target, side="right"))
    idx = min(max(idx, 0), len(branch.points_um) - 2)
    seg_start = float(cumulative_um[idx])
    seg_end = float(cumulative_um[idx + 1])
    seg_len = seg_end - seg_start
    if seg_len <= 0.0:
        return branch.points_um[idx].copy()
    alpha = (target - seg_start) / seg_len
    return (1.0 - alpha) * branch.points_um[idx] + alpha * branch.points_um[idx + 1]


def _radius_at_3d(branch: BranchPolyline3D, cumulative_um: np.ndarray, x: float) -> float:
    total = float(cumulative_um[-1]) if cumulative_um.size else 0.0
    if total <= 0.0 or branch.radii_um.size == 0:
        return float(branch.radii_um[0]) if branch.radii_um.size else 1.0
    clamped = float(np.clip(x, 0.0, 1.0))
    target = clamped * total
    idx = int(np.searchsorted(cumulative_um[1:], target, side="right"))
    idx = min(max(idx, 0), len(branch.radii_um) - 2)
    seg_start = float(cumulative_um[idx])
    seg_end = float(cumulative_um[idx + 1])
    seg_len = seg_end - seg_start
    if seg_len <= 0.0:
        return float(branch.radii_um[idx])
    alpha = (target - seg_start) / seg_len
    return float((1.0 - alpha) * branch.radii_um[idx] + alpha * branch.radii_um[idx + 1])


def _subpolyline_3d(
    branch: BranchPolyline3D,
    cumulative_um: np.ndarray,
    prox: float,
    dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = (float(prox), float(dist)) if prox <= dist else (float(dist), float(prox))
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    total = float(cumulative_um[-1]) if cumulative_um.size else 0.0
    if total <= 0.0 or branch.points_um.shape[0] == 0:
        return (
            branch.points_um[:1].copy() if branch.points_um.size else np.zeros((0, 3), dtype=float),
            branch.radii_um[:1].copy() if branch.radii_um.size else np.zeros(0, dtype=float),
        )
    start = _point_at_3d(branch, cumulative_um, lo)
    end = _point_at_3d(branch, cumulative_um, hi)
    start_r = _radius_at_3d(branch, cumulative_um, lo)
    end_r = _radius_at_3d(branch, cumulative_um, hi)
    start_arc = lo * total
    end_arc = hi * total
    interior_mask = (cumulative_um > start_arc + 1e-12) & (cumulative_um < end_arc - 1e-12)
    interior_points = branch.points_um[interior_mask]
    interior_radii = branch.radii_um[interior_mask]
    points = np.vstack([start[None, :], interior_points, end[None, :]])
    radii = np.concatenate([[start_r], interior_radii, [end_r]])
    return points, radii


def _build_overlay_primitives_3d(
    overlay: OverlaySpec | None,
    branch_lookup: dict[int, BranchPolyline3D],
) -> tuple[tuple[HighlightStroke3D, ...], tuple[Marker3D, ...]]:
    if overlay is None:
        return (), ()

    strokes: list[HighlightStroke3D] = []
    markers: list[Marker3D] = []

    cumulative_cache: dict[int, np.ndarray] = {}

    def _cumulative(branch: BranchPolyline3D) -> np.ndarray:
        cache = cumulative_cache.get(branch.branch_index)
        if cache is None:
            cache = _branch_cumulative_um(branch)
            cumulative_cache[branch.branch_index] = cache
        return cache

    region = overlay.region
    if region is not None:
        color = _highlight_color()
        alpha = _highlight_alpha()
        for branch_index, prox, dist in region.intervals:
            branch = branch_lookup.get(int(branch_index))
            if branch is None:
                continue
            cumulative = _cumulative(branch)
            points, radii = _subpolyline_3d(branch, cumulative, float(prox), float(dist))
            if points.shape[0] < 2:
                continue
            strokes.append(
                HighlightStroke3D(
                    branch_index=branch.branch_index,
                    branch_name=branch.branch_name,
                    branch_type=branch.branch_type,
                    points_um=points,
                    radii_um=radii,
                    color_rgb=color,
                    opacity=alpha,
                )
            )

    locset = overlay.locset
    if locset is not None:
        color = _marker_color()
        radius = _marker_radius_3d_um()
        for point in locset.points:
            branch_index, x = int(point[0]), float(point[1])
            branch = branch_lookup.get(branch_index)
            if branch is None:
                continue
            cumulative = _cumulative(branch)
            position = _point_at_3d(branch, cumulative, x)
            markers.append(
                Marker3D(
                    branch_index=branch_index,
                    x=x,
                    position_um=position,
                    color_rgb=color,
                    radius_um=radius,
                )
            )

    return tuple(strokes), tuple(markers)


def _reject_values_overlay(overlay: OverlaySpec | None) -> None:
    if overlay is not None and overlay.values is not None:
        raise NotImplementedError(
            "Color-by-values overlays are scheduled for M6 Phase 3. "
            "Only region= and locset= overlays are wired through in the "
            "current release."
        )
