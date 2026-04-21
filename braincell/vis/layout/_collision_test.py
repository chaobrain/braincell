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


import unittest

import numpy as np

from braincell.vis.layout._collision import (
    _SegmentSpatialHash,
    _build_collision_index,
    _layout_collision_score,
    _point_to_segment_distance_um,
    _polyline_collision_score,
    _segment_distance_um,
    _segments_intersect,
    _segments_share_endpoint,
)
from braincell.vis.layout._common import LayoutBranch2D
from braincell.vis.layout._config import DEFAULT_LAYOUT_CONFIG, LayoutConfig

_COLLISION_MARGIN_UM = DEFAULT_LAYOUT_CONFIG.collision_margin_um


def _brute_force_score(
    candidate: LayoutBranch2D,
    existing: tuple[LayoutBranch2D, ...],
    *,
    margin_um: float,
) -> float:
    """Reference implementation of the per-pair collision scoring.

    Iterates every segment pair and sums the legacy pair score.
    Used to verify that the spatial-hash backend produces the same
    numeric output.
    """
    score = 0.0
    candidate_points = candidate.segment_points_um
    for existing_layout in existing:
        score += _polyline_collision_score(
            candidate_points,
            existing_layout.segment_points_um,
            margin_um=margin_um,
        )
    return score


def _layout_from_points(points: list[list[float]], *, index: int = 0) -> LayoutBranch2D:
    pts = np.asarray(points, dtype=float)
    diffs = np.diff(pts, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    directions = diffs / np.where(lengths > 0.0, lengths, 1.0)[:, None]
    normals = np.column_stack([-directions[:, 1], directions[:, 0]])
    return LayoutBranch2D(
        branch_index=index,
        branch_name=f"b{index}",
        branch_type="soma",
        segment_points_um=pts,
        radii_proximal_um=np.ones(len(lengths), dtype=float),
        radii_distal_um=np.ones(len(lengths), dtype=float),
        total_length_um=float(cum[-1]),
        segment_directions_um=directions,
        segment_normals_um=normals,
        cumulative_lengths_um=cum,
    )


class PointToSegmentDistanceTest(unittest.TestCase):
    def test_point_on_segment_returns_zero(self) -> None:
        point = np.array([5.0, 0.0])
        self.assertAlmostEqual(
            _point_to_segment_distance_um(point, np.array([0.0, 0.0]), np.array([10.0, 0.0])),
            0.0,
        )

    def test_point_off_segment_returns_perpendicular(self) -> None:
        point = np.array([5.0, 3.0])
        self.assertAlmostEqual(
            _point_to_segment_distance_um(point, np.array([0.0, 0.0]), np.array([10.0, 0.0])),
            3.0,
        )

    def test_degenerate_segment_uses_endpoint_distance(self) -> None:
        point = np.array([3.0, 4.0])
        self.assertAlmostEqual(
            _point_to_segment_distance_um(point, np.array([0.0, 0.0]), np.array([0.0, 0.0])),
            5.0,
        )


class SegmentIntersectionTest(unittest.TestCase):
    def test_crossing_segments_intersect(self) -> None:
        self.assertTrue(
            _segments_intersect(
                np.array([0.0, 0.0]),
                np.array([10.0, 10.0]),
                np.array([0.0, 10.0]),
                np.array([10.0, 0.0]),
            )
        )

    def test_parallel_segments_do_not_intersect(self) -> None:
        self.assertFalse(
            _segments_intersect(
                np.array([0.0, 0.0]),
                np.array([10.0, 0.0]),
                np.array([0.0, 5.0]),
                np.array([10.0, 5.0]),
            )
        )

    def test_shared_endpoint_is_not_a_collision(self) -> None:
        # Proper intersection requires strictly crossing interiors.
        self.assertFalse(
            _segments_intersect(
                np.array([0.0, 0.0]),
                np.array([10.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([0.0, 10.0]),
            )
        )


class SegmentsShareEndpointTest(unittest.TestCase):
    def test_shared_start_points(self) -> None:
        self.assertTrue(
            _segments_share_endpoint(
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([0.0, 1.0]),
            )
        )

    def test_disjoint_segments(self) -> None:
        self.assertFalse(
            _segments_share_endpoint(
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([10.0, 10.0]),
                np.array([20.0, 20.0]),
            )
        )


class SegmentDistanceTest(unittest.TestCase):
    def test_distance_between_parallel_segments(self) -> None:
        d = _segment_distance_um(
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            np.array([0.0, 4.0]),
            np.array([10.0, 4.0]),
        )
        self.assertAlmostEqual(d, 4.0)


class PolylineCollisionScoreTest(unittest.TestCase):
    def test_non_overlapping_polylines_score_zero(self) -> None:
        a = np.array([[0.0, 0.0], [10.0, 0.0]])
        b = np.array([[0.0, 100.0], [10.0, 100.0]])
        self.assertEqual(_polyline_collision_score(a, b, margin_um=_COLLISION_MARGIN_UM), 0.0)

    def test_crossing_polylines_earn_big_penalty(self) -> None:
        a = np.array([[0.0, 0.0], [10.0, 10.0]])
        b = np.array([[0.0, 10.0], [10.0, 0.0]])
        # 1000.0 penalty per intersecting segment pair.
        self.assertGreaterEqual(
            _polyline_collision_score(a, b, margin_um=_COLLISION_MARGIN_UM),
            1000.0,
        )


class LayoutCollisionScoreTest(unittest.TestCase):
    def test_empty_existing_list_scores_zero(self) -> None:
        candidate = _layout_from_points([[0.0, 0.0], [10.0, 0.0]])
        self.assertEqual(_layout_collision_score(candidate, ()), 0.0)

    def test_well_separated_candidate_scores_zero(self) -> None:
        candidate = _layout_from_points([[0.0, 0.0], [10.0, 0.0]], index=0)
        existing = _layout_from_points([[0.0, 100.0], [10.0, 100.0]], index=1)
        self.assertEqual(_layout_collision_score(candidate, (existing,)), 0.0)

    def test_overlapping_candidate_scores_positive(self) -> None:
        candidate = _layout_from_points([[0.0, 0.0], [10.0, 10.0]], index=0)
        existing = _layout_from_points([[0.0, 10.0], [10.0, 0.0]], index=1)
        self.assertGreater(_layout_collision_score(candidate, (existing,)), 0.0)


class SegmentSpatialHashTest(unittest.TestCase):
    def test_rejects_non_positive_cell_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "cell_size_um must be > 0"):
            _SegmentSpatialHash(cell_size_um=0.0)
        with self.assertRaisesRegex(ValueError, "cell_size_um must be > 0"):
            _SegmentSpatialHash(cell_size_um=-1.0)

    def test_empty_index_scores_zero(self) -> None:
        index = _SegmentSpatialHash(cell_size_um=20.0)
        candidate = _layout_from_points([[0.0, 0.0], [10.0, 0.0]])
        self.assertEqual(index.scored_candidate(candidate, margin_um=2.0), 0.0)

    def test_non_overlapping_layouts_score_zero(self) -> None:
        index = _SegmentSpatialHash(cell_size_um=20.0)
        index.insert(_layout_from_points([[0.0, 100.0], [10.0, 100.0]], index=0))
        candidate = _layout_from_points([[0.0, 0.0], [10.0, 0.0]])
        self.assertEqual(index.scored_candidate(candidate, margin_um=2.0), 0.0)

    def test_intersecting_layouts_score_at_least_1000(self) -> None:
        index = _SegmentSpatialHash(cell_size_um=20.0)
        index.insert(_layout_from_points([[0.0, 10.0], [10.0, 0.0]], index=0))
        candidate = _layout_from_points([[0.0, 0.0], [10.0, 10.0]])
        self.assertGreaterEqual(index.scored_candidate(candidate, margin_um=2.0), 1000.0)


class SpatialHashVsBruteForceTest(unittest.TestCase):
    """The spatial hash must agree with the brute-force reference on
    every candidate/existing pair we care about. This is the safety
    net that lets the stem code swap backends freely."""

    def _make_layouts(self) -> list[LayoutBranch2D]:
        return [
            _layout_from_points([[0.0, 0.0], [10.0, 0.0]], index=0),
            _layout_from_points([[10.0, 0.0], [20.0, 10.0]], index=1),
            _layout_from_points([[20.0, 10.0], [30.0, 25.0]], index=2),
            _layout_from_points([[0.0, 30.0], [50.0, 30.0]], index=3),
            _layout_from_points([[-20.0, -20.0], [-5.0, -5.0]], index=4),
        ]

    def test_hash_matches_brute_force_for_disjoint_candidate(self) -> None:
        existing = tuple(self._make_layouts())
        candidate = _layout_from_points([[100.0, 100.0], [110.0, 100.0]], index=99)
        hashed = _layout_collision_score(candidate, existing)
        reference = _brute_force_score(candidate, existing, margin_um=_COLLISION_MARGIN_UM)
        self.assertAlmostEqual(hashed, reference, places=6)
        self.assertEqual(hashed, 0.0)

    def test_hash_matches_brute_force_for_tangent_candidate(self) -> None:
        existing = tuple(self._make_layouts())
        # Candidate running very close to layout 0 (distance < margin).
        candidate = _layout_from_points([[0.0, 1.0], [10.0, 1.0]], index=99)
        hashed = _layout_collision_score(candidate, existing)
        reference = _brute_force_score(candidate, existing, margin_um=_COLLISION_MARGIN_UM)
        self.assertAlmostEqual(hashed, reference, places=6)
        self.assertGreater(hashed, 0.0)

    def test_hash_matches_brute_force_for_intersecting_candidate(self) -> None:
        existing = tuple(self._make_layouts())
        candidate = _layout_from_points([[5.0, -5.0], [5.0, 5.0]], index=99)
        hashed = _layout_collision_score(candidate, existing)
        reference = _brute_force_score(candidate, existing, margin_um=_COLLISION_MARGIN_UM)
        self.assertAlmostEqual(hashed, reference, places=6)
        self.assertGreater(hashed, 0.0)

    def test_hash_independent_of_cell_size(self) -> None:
        existing = tuple(self._make_layouts())
        candidate = _layout_from_points([[5.0, -5.0], [5.0, 5.0]], index=99)
        configs = (
            LayoutConfig(collision_cell_size_um=5.0),
            LayoutConfig(collision_cell_size_um=20.0),
            LayoutConfig(collision_cell_size_um=100.0),
        )
        scores = [
            _layout_collision_score(candidate, existing, layout_config=config)
            for config in configs
        ]
        for score in scores[1:]:
            self.assertAlmostEqual(score, scores[0], places=6)


class BuildCollisionIndexTest(unittest.TestCase):
    def test_reuses_index_across_candidates(self) -> None:
        existing = (
            _layout_from_points([[0.0, 0.0], [10.0, 0.0]], index=0),
            _layout_from_points([[0.0, 10.0], [10.0, 10.0]], index=1),
        )
        index = _build_collision_index(existing)
        # Two candidates scored against the same index must agree
        # with the brute-force reference.
        for candidate_points in [
            [[5.0, 0.0], [5.0, 10.0]],    # crosses both
            [[0.0, 5.0], [10.0, 5.0]],    # sits between them
            [[100.0, 0.0], [110.0, 0.0]],  # disjoint
        ]:
            candidate = _layout_from_points(candidate_points, index=99)
            hashed = index.scored_candidate(candidate, margin_um=_COLLISION_MARGIN_UM)
            reference = _brute_force_score(candidate, existing, margin_um=_COLLISION_MARGIN_UM)
            self.assertAlmostEqual(hashed, reference, places=6)


if __name__ == "__main__":
    unittest.main()
