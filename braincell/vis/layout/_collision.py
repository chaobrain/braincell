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

"""Collision scoring for 2D layout candidates.

The stem layout family proposes several candidate placements for each
child branch and ranks them by a score that combines:

1. Physical overlap with already-placed branches (computed here).
2. How close the candidate's tail direction is to the desired target.
3. How "opening" the launch angle is relative to the attach tangent.

This module owns piece (1). Two scoring backends are provided:

``_layout_collision_score``
    Scores a single candidate against a tuple of existing layouts.
    Builds a fresh :class:`_SegmentSpatialHash` internally and
    queries it once; useful when the caller does not already have a
    prepared index (e.g. the stem-linear family, which scores against
    a rolling window of the last 48 branches).

``_SegmentSpatialHash``
    A 2D uniform-grid spatial index over segment AABBs, consumed by
    the stem-tree family via ``build_collision_index`` +
    ``scored_candidate``. Letting the caller build the index once
    per fork and reuse it across all profile candidates turns the
    inner loop from O(|existing_segments|) to
    O(|segments within cell_size_um + margin|).

Both backends return the same numeric score so the stem scoring
function is invariant to the backend choice. The score is a sum of:

* a large penalty for proper segment intersection (``1000.0`` per
  offending segment pair),
* a soft "too close" penalty equal to ``margin_um - distance_um`` for
  pairs whose closest approach is below ``collision_margin_um``.

Shared endpoints (parent → child attach) are excluded so that a legal
fork does not accidentally score as a collision.
"""

import math

import numpy as np

from ._common import LayoutBranch2D
from ._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig


# ---------------------------------------------------------------------------
# Low-level geometric predicates
# ---------------------------------------------------------------------------

def _segments_share_endpoint(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> bool:
    endpoint_pairs = ((a0, b0), (a0, b1), (a1, b0), (a1, b1))
    return any(np.linalg.norm(point_a - point_b) <= 1e-6 for point_a, point_b in endpoint_pairs)


def _segments_intersect(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orientation(a0, a1, b0)
    o2 = orientation(a0, a1, b1)
    o3 = orientation(b0, b1, a0)
    o4 = orientation(b0, b1, a1)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _segment_distance_um(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> float:
    return min(
        _point_to_segment_distance_um(a0, b0, b1),
        _point_to_segment_distance_um(a1, b0, b1),
        _point_to_segment_distance_um(b0, a0, a1),
        _point_to_segment_distance_um(b1, a0, a1),
    )


def _point_to_segment_distance_um(
    point_um: np.ndarray,
    seg0_um: np.ndarray,
    seg1_um: np.ndarray,
) -> float:
    seg_vec_um = seg1_um - seg0_um
    seg_len_sq_um = float(np.dot(seg_vec_um, seg_vec_um))
    if seg_len_sq_um <= 0.0:
        return float(np.linalg.norm(point_um - seg0_um))
    projection = float(np.dot(point_um - seg0_um, seg_vec_um) / seg_len_sq_um)
    projection = min(max(projection, 0.0), 1.0)
    closest_um = seg0_um + projection * seg_vec_um
    return float(np.linalg.norm(point_um - closest_um))


def _pair_score(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    *,
    margin_um: float,
) -> float:
    if _segments_share_endpoint(a0, a1, b0, b1):
        return 0.0
    if _segments_intersect(a0, a1, b0, b1):
        return 1000.0
    distance_um = _segment_distance_um(a0, a1, b0, b1)
    if distance_um < margin_um:
        return margin_um - distance_um
    return 0.0


# ---------------------------------------------------------------------------
# Spatial-hash index
# ---------------------------------------------------------------------------

class _SegmentSpatialHash:
    """Uniform-grid 2D spatial hash for layout segments.

    Each segment is inserted into every cell its axis-aligned
    bounding box touches. A candidate-segment query gathers every
    segment in the cells the candidate AABB (plus a margin
    inflation) touches, so the caller only pays for "nearby"
    segments instead of the full existing set.

    Parameters
    ----------
    cell_size_um : float
        Size of one grid cell. See :class:`LayoutConfig` for tuning
        notes. Must be strictly positive; callers enforce this
        implicitly by reading from ``LayoutConfig``.
    """

    __slots__ = ("cell_size_um", "_cells", "_layouts", "_segments")

    def __init__(self, cell_size_um: float) -> None:
        if cell_size_um <= 0.0:
            raise ValueError(f"cell_size_um must be > 0, got {cell_size_um!r}.")
        self.cell_size_um = float(cell_size_um)
        self._cells: dict[tuple[int, int], list[int]] = {}
        self._layouts: list[LayoutBranch2D] = []
        self._segments: list[tuple[np.ndarray, np.ndarray]] = []

    def insert(self, layout: LayoutBranch2D) -> None:
        self._layouts.append(layout)
        points = layout.segment_points_um
        for segment_index in range(len(points) - 1):
            p0 = np.asarray(points[segment_index], dtype=float)
            p1 = np.asarray(points[segment_index + 1], dtype=float)
            segment_flat_index = len(self._segments)
            self._segments.append((p0, p1))
            for cell_key in self._iter_segment_cells(p0, p1, pad_um=0.0):
                self._cells.setdefault(cell_key, []).append(segment_flat_index)

    def insert_all(self, layouts: tuple[LayoutBranch2D, ...]) -> None:
        for layout in layouts:
            self.insert(layout)

    def scored_candidate(self, candidate: LayoutBranch2D, margin_um: float) -> float:
        score = 0.0
        candidate_points = candidate.segment_points_um
        for segment_index in range(len(candidate_points) - 1):
            a0 = np.asarray(candidate_points[segment_index], dtype=float)
            a1 = np.asarray(candidate_points[segment_index + 1], dtype=float)
            seen: set[int] = set()
            for cell_key in self._iter_segment_cells(a0, a1, pad_um=margin_um):
                bucket = self._cells.get(cell_key)
                if bucket is None:
                    continue
                for other_flat_index in bucket:
                    if other_flat_index in seen:
                        continue
                    seen.add(other_flat_index)
                    b0, b1 = self._segments[other_flat_index]
                    score += _pair_score(a0, a1, b0, b1, margin_um=margin_um)
        return score

    def _iter_segment_cells(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        pad_um: float,
    ):
        cell_size = self.cell_size_um
        min_x = min(float(p0[0]), float(p1[0])) - pad_um
        max_x = max(float(p0[0]), float(p1[0])) + pad_um
        min_y = min(float(p0[1]), float(p1[1])) - pad_um
        max_y = max(float(p0[1]), float(p1[1])) + pad_um
        cx0 = int(math.floor(min_x / cell_size))
        cx1 = int(math.floor(max_x / cell_size))
        cy0 = int(math.floor(min_y / cell_size))
        cy1 = int(math.floor(max_y / cell_size))
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                yield (cx, cy)


def _build_collision_index(
    existing_layouts: tuple[LayoutBranch2D, ...],
    *,
    layout_config: LayoutConfig | None = None,
) -> _SegmentSpatialHash:
    config = layout_config or DEFAULT_LAYOUT_CONFIG
    index = _SegmentSpatialHash(cell_size_um=config.collision_cell_size_um)
    index.insert_all(existing_layouts)
    return index


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def _layout_collision_score(
    candidate: LayoutBranch2D,
    existing_layouts: tuple[LayoutBranch2D, ...],
    *,
    layout_config: LayoutConfig | None = None,
) -> float:
    """Score a candidate layout against a fixed set of existing layouts.

    Builds a fresh :class:`_SegmentSpatialHash` for every call; prefer
    :func:`_build_collision_index` + :meth:`_SegmentSpatialHash.scored_candidate`
    when scoring many candidates against the same ``existing_layouts``.
    """
    config = layout_config or DEFAULT_LAYOUT_CONFIG
    if not existing_layouts:
        return 0.0
    index = _build_collision_index(existing_layouts, layout_config=config)
    return index.scored_candidate(candidate, margin_um=config.collision_margin_um)


def _polyline_collision_score(
    points_a_um: np.ndarray,
    points_b_um: np.ndarray,
    *,
    margin_um: float,
) -> float:
    """Brute-force pairwise score between two polylines.

    Retained for test doubles and for backends that do not use the
    spatial hash. Equivalent to running the hash over
    ``points_b_um`` and querying with ``points_a_um``.
    """
    score = 0.0
    points_a = np.asarray(points_a_um, dtype=float)
    points_b = np.asarray(points_b_um, dtype=float)
    for segment_a_index in range(len(points_a) - 1):
        a0 = points_a[segment_a_index]
        a1 = points_a[segment_a_index + 1]
        for segment_b_index in range(len(points_b) - 1):
            b0 = points_b[segment_b_index]
            b1 = points_b[segment_b_index + 1]
            score += _pair_score(a0, a1, b0, b1, margin_um=margin_um)
    return score
