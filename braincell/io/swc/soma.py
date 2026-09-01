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
from collections.abc import Iterable, Mapping

import numpy as np

from .types import _SwcRow

_REL_TOL = 1e-6
_SPECIAL_THREE_POINT_RADIUS_TOL_SCALE = 0.02


def row_point(row: _SwcRow) -> np.ndarray:
    return np.array([row.x, row.y, row.z], dtype=float)


def row_radius(row: _SwcRow) -> float:
    return float(row.radius)


def is_special_three_point_soma(
    rows: Iterable[_SwcRow],
    children_by_id: Mapping[int, list[int] | tuple[int, ...]],
) -> tuple[bool, tuple[_SwcRow, _SwcRow, _SwcRow] | None]:
    soma_rows = tuple(rows)
    if len(soma_rows) != 3:
        return False, None

    center_row = None
    side_a_row: _SwcRow
    side_b_row: _SwcRow
    for candidate in soma_rows:
        others = [r for r in soma_rows if r is not candidate]
        if len(others) == 2 and all(r.parent_id == candidate.node_id for r in others):
            center_row = candidate
            side_a_row, side_b_row = others
            break
    if center_row is None:
        return False, None
    if children_by_id.get(side_a_row.node_id) or children_by_id.get(side_b_row.node_id):
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
    # SWC contour-soma recognition is intentionally disabled. Multi-point soma
    # samples are imported as regular soma sections; contour reconstruction is
    # no longer supported during SWC import.
    del rows
    return False


def contour_equivalent_center_radius(rows: Iterable[_SwcRow]) -> tuple[np.ndarray, float]:
    soma_rows = tuple(rows)
    points = np.array([row_point(row) for row in soma_rows], dtype=float)
    center = points.mean(axis=0)
    radius = max(np.linalg.norm(point - center) + row_radius(row) for point, row in zip(points, soma_rows))
    return center, float(radius)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))
