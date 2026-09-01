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

Implementation note
-------------------
Every predicate below is written twice: a ``_scalar`` core that takes
loose ``float`` coordinates, and a thin array-taking wrapper that
unpacks and delegates. The scalar core is what the hot loop calls.
These are 2-element vectors, so a :func:`numpy.linalg.norm` call spends
microseconds of dispatch on nanoseconds of arithmetic — scoring one
114-branch morphology issued 17M ``norm`` calls and took 56 seconds.
The scalar rewrite is ~30x faster while producing bit-identical
scores (``math.sqrt(dx * dx + dy * dy)`` is the same sequence of IEEE
operations ``norm`` performs for a 2-vector; ``math.hypot`` is *not*
and is deliberately avoided).
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
    return _segments_share_endpoint_scalar(
        float(a0[0]),
        float(a0[1]),
        float(a1[0]),
        float(a1[1]),
        float(b0[0]),
        float(b0[1]),
        float(b1[0]),
        float(b1[1]),
    )


def _segments_share_endpoint_scalar(
    ax0: float,
    ay0: float,
    ax1: float,
    ay1: float,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> bool:
    for px, py, qx, qy in (
        (ax0, ay0, bx0, by0),
        (ax0, ay0, bx1, by1),
        (ax1, ay1, bx0, by0),
        (ax1, ay1, bx1, by1),
    ):
        dx = px - qx
        dy = py - qy
        if math.sqrt(dx * dx + dy * dy) <= 1e-6:
            return True
    return False


def _segments_intersect(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> bool:
    return _segments_intersect_scalar(
        float(a0[0]),
        float(a0[1]),
        float(a1[0]),
        float(a1[1]),
        float(b0[0]),
        float(b0[1]),
        float(b1[0]),
        float(b1[1]),
    )


def _segments_intersect_scalar(
    ax0: float,
    ay0: float,
    ax1: float,
    ay1: float,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> bool:
    o1 = (ax1 - ax0) * (by0 - ay0) - (ay1 - ay0) * (bx0 - ax0)
    o2 = (ax1 - ax0) * (by1 - ay0) - (ay1 - ay0) * (bx1 - ax0)
    if not (o1 * o2 < 0.0):
        return False
    o3 = (bx1 - bx0) * (ay0 - by0) - (by1 - by0) * (ax0 - bx0)
    o4 = (bx1 - bx0) * (ay1 - by0) - (by1 - by0) * (ax1 - bx0)
    return o3 * o4 < 0.0


def _segment_distance_um(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> float:
    return math.sqrt(
        _segment_distance_sq_scalar(
            float(a0[0]),
            float(a0[1]),
            float(a1[0]),
            float(a1[1]),
            float(b0[0]),
            float(b0[1]),
            float(b1[0]),
            float(b1[1]),
        )
    )


def _segment_distance_sq_scalar(
    ax0: float,
    ay0: float,
    ax1: float,
    ay1: float,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> float:
    # ``sqrt`` is monotonic and correctly rounded, so taking the min of
    # the squared distances and rooting once equals rooting each and
    # taking the min — three fewer square roots per pair.
    return min(
        _point_to_segment_distance_sq_scalar(ax0, ay0, bx0, by0, bx1, by1),
        _point_to_segment_distance_sq_scalar(ax1, ay1, bx0, by0, bx1, by1),
        _point_to_segment_distance_sq_scalar(bx0, by0, ax0, ay0, ax1, ay1),
        _point_to_segment_distance_sq_scalar(bx1, by1, ax0, ay0, ax1, ay1),
    )


def _point_to_segment_distance_um(
    point_um: np.ndarray,
    seg0_um: np.ndarray,
    seg1_um: np.ndarray,
) -> float:
    return math.sqrt(
        _point_to_segment_distance_sq_scalar(
            float(point_um[0]),
            float(point_um[1]),
            float(seg0_um[0]),
            float(seg0_um[1]),
            float(seg1_um[0]),
            float(seg1_um[1]),
        )
    )


def _point_to_segment_distance_sq_scalar(
    px: float,
    py: float,
    sx0: float,
    sy0: float,
    sx1: float,
    sy1: float,
) -> float:
    vx = sx1 - sx0
    vy = sy1 - sy0
    seg_len_sq = vx * vx + vy * vy
    wx = px - sx0
    wy = py - sy0
    if seg_len_sq <= 0.0:
        return wx * wx + wy * wy
    projection = (wx * vx + wy * vy) / seg_len_sq
    if projection < 0.0:
        projection = 0.0
    elif projection > 1.0:
        projection = 1.0
    dx = wx - projection * vx
    dy = wy - projection * vy
    return dx * dx + dy * dy


def _pair_score(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    *,
    margin_um: float,
) -> float:
    return _pair_score_scalar(
        float(a0[0]),
        float(a0[1]),
        float(a1[0]),
        float(a1[1]),
        float(b0[0]),
        float(b0[1]),
        float(b1[0]),
        float(b1[1]),
        margin_um=margin_um,
    )


def _pair_score_scalar(
    ax0: float,
    ay0: float,
    ax1: float,
    ay1: float,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    *,
    margin_um: float,
) -> float:
    # AABB reject first. If the boxes are more than ``margin_um`` apart
    # on either axis then the segments cannot share an endpoint, cannot
    # cross, and are farther apart than the margin — the score is
    # exactly 0.0 and the expensive predicates are skipped. This is the
    # common case: the spatial hash returns whole cells, most of whose
    # occupants are not near the candidate.
    if ax0 < ax1:
        a_min_x, a_max_x = ax0, ax1
    else:
        a_min_x, a_max_x = ax1, ax0
    if bx0 < bx1:
        b_min_x, b_max_x = bx0, bx1
    else:
        b_min_x, b_max_x = bx1, bx0
    if a_min_x - margin_um > b_max_x or b_min_x - margin_um > a_max_x:
        return 0.0
    if ay0 < ay1:
        a_min_y, a_max_y = ay0, ay1
    else:
        a_min_y, a_max_y = ay1, ay0
    if by0 < by1:
        b_min_y, b_max_y = by0, by1
    else:
        b_min_y, b_max_y = by1, by0
    if a_min_y - margin_um > b_max_y or b_min_y - margin_um > a_max_y:
        return 0.0

    if _segments_share_endpoint_scalar(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
        return 0.0
    if _segments_intersect_scalar(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
        return 1000.0
    distance_um = math.sqrt(_segment_distance_sq_scalar(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1))
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
        # Flat ``(x0, y0, x1, y1)`` float tuples rather than ndarray
        # pairs: the scoring loop wants loose floats, and unpacking a
        # tuple is far cheaper than indexing two arrays per pair.
        self._segments: list[tuple[float, float, float, float]] = []

    def insert(self, layout: LayoutBranch2D) -> None:
        self._layouts.append(layout)
        points = np.asarray(layout.segment_points_um, dtype=float)
        for segment_index in range(len(points) - 1):
            x0 = float(points[segment_index, 0])
            y0 = float(points[segment_index, 1])
            x1 = float(points[segment_index + 1, 0])
            y1 = float(points[segment_index + 1, 1])
            segment_flat_index = len(self._segments)
            self._segments.append((x0, y0, x1, y1))
            for cell_key in self._iter_segment_cells(x0, y0, x1, y1, pad_um=0.0):
                self._cells.setdefault(cell_key, []).append(segment_flat_index)

    def insert_all(self, layouts: tuple[LayoutBranch2D, ...]) -> None:
        for layout in layouts:
            self.insert(layout)

    def scored_candidate(self, candidate: LayoutBranch2D, margin_um: float) -> float:
        score = 0.0
        candidate_points = np.asarray(candidate.segment_points_um, dtype=float)
        cells = self._cells
        segments = self._segments
        for segment_index in range(len(candidate_points) - 1):
            ax0 = float(candidate_points[segment_index, 0])
            ay0 = float(candidate_points[segment_index, 1])
            ax1 = float(candidate_points[segment_index + 1, 0])
            ay1 = float(candidate_points[segment_index + 1, 1])
            seen: set[int] = set()
            for cell_key in self._iter_segment_cells(ax0, ay0, ax1, ay1, pad_um=margin_um):
                bucket = cells.get(cell_key)
                if bucket is None:
                    continue
                for other_flat_index in bucket:
                    if other_flat_index in seen:
                        continue
                    seen.add(other_flat_index)
                    bx0, by0, bx1, by1 = segments[other_flat_index]
                    score += _pair_score_scalar(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1, margin_um=margin_um)
        return score

    def _iter_segment_cells(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        pad_um: float,
    ):
        cell_size = self.cell_size_um
        min_x = (x0 if x0 < x1 else x1) - pad_um
        max_x = (x1 if x0 < x1 else x0) + pad_um
        min_y = (y0 if y0 < y1 else y1) - pad_um
        max_y = (y1 if y0 < y1 else y0) + pad_um
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
