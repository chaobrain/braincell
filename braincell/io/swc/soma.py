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
from typing import Iterable

import numpy as np

from .types import _SwcRow

_REL_TOL = 1e-6
_SPECIAL_THREE_POINT_RADIUS_TOL_SCALE = 0.02


def row_point(row: _SwcRow) -> np.ndarray:
    return np.array([row.x, row.y, row.z], dtype=float)


def row_radius(row: _SwcRow) -> float:
    return float(row.radius)


def is_special_three_point_soma(rows: Iterable[_SwcRow]) -> tuple[bool, tuple[_SwcRow, _SwcRow, _SwcRow] | None]:
    soma_rows = tuple(rows)
    if len(soma_rows) != 3:
        return False, None

    center_row, side_a_row, side_b_row = soma_rows
    if side_a_row.parent_id != center_row.node_id or side_b_row.parent_id != center_row.node_id:
        return False, None

    center = row_point(center_row)
    side_a = row_point(side_a_row)
    side_b = row_point(side_b_row)
    abs_tol = max(_SPECIAL_THREE_POINT_RADIUS_TOL_SCALE * row_radius(center_row), _REL_TOL)

    if min(_distance(center, side_a), _distance(center, side_b)) <= 0.0:
        return False, None

    radii = np.array([row_radius(row) for row in soma_rows], dtype=float)
    if not np.allclose(radii, radii[0], rtol=_REL_TOL, atol=abs_tol):
        return False, None

    radius = row_radius(center_row)
    if not math.isclose(_distance(center, side_a), radius, rel_tol=_REL_TOL, abs_tol=abs_tol):
        return False, None
    if not math.isclose(_distance(center, side_b), radius, rel_tol=_REL_TOL, abs_tol=abs_tol):
        return False, None

    return True, soma_rows


def is_contour_soma(rows: Iterable[_SwcRow]) -> bool:
    soma_rows = tuple(rows)
    if len(soma_rows) <= 3:
        return False
    start_row = soma_rows[0]
    end_row = soma_rows[-1]
    mid_row = _curvature_midpoint(soma_rows)
    if mid_row is None:
        return False
    angle = _interior_angle_degrees(
        row_point(start_row),
        row_point(mid_row),
        row_point(end_row),
    )
    return angle <= 90.0 + _REL_TOL


def contour_equivalent_center_radius(rows: Iterable[_SwcRow]) -> tuple[np.ndarray, float]:
    soma_rows = tuple(rows)
    points = np.array([row_point(row) for row in soma_rows], dtype=float)
    center = points.mean(axis=0)
    radius = max(
        np.linalg.norm(point - center) + row_radius(row)
        for point, row in zip(points, soma_rows)
    )
    return center, float(radius)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def _order_rows_along_line(rows: tuple[_SwcRow, _SwcRow, _SwcRow]) -> tuple[_SwcRow, _SwcRow, _SwcRow]:
    points = [row_point(row) for row in rows]
    farthest_pair = max(
        ((i, j) for i in range(3) for j in range(i + 1, 3)),
        key=lambda pair: _distance(points[pair[0]], points[pair[1]]),
    )
    start = points[farthest_pair[0]]
    end = points[farthest_pair[1]]
    direction = end - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        return rows
    projections = [float(np.dot(point - start, direction) / direction_norm) for point in points]
    order = sorted(range(3), key=lambda index: projections[index])
    return tuple(rows[index] for index in order)


def _is_collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray, *, abs_tol: float) -> bool:
    baseline = c - a
    baseline_norm = np.linalg.norm(baseline)
    if baseline_norm == 0.0:
        return False
    point_norm = np.linalg.norm(np.cross(b - a, baseline)) / baseline_norm
    return point_norm <= abs_tol


def _interior_angle_degrees(prev_point: np.ndarray, point: np.ndarray, next_point: np.ndarray) -> float:
    incoming = prev_point - point
    outgoing = next_point - point
    incoming_norm = np.linalg.norm(incoming)
    outgoing_norm = np.linalg.norm(outgoing)
    if incoming_norm == 0.0 or outgoing_norm == 0.0:
        return 180.0
    cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _curvature_midpoint(rows: tuple[_SwcRow, ...]) -> _SwcRow | None:
    if len(rows) <= 2:
        return None
    start_point = row_point(rows[0])
    end_point = row_point(rows[-1])
    best_row: _SwcRow | None = None
    best_score = -math.inf
    for row in rows[1:-1]:
        point = row_point(row)
        score = min(_distance(start_point, point), _distance(point, end_point))
        if score > best_score + _REL_TOL:
            best_row = row
            best_score = score
    return best_row
