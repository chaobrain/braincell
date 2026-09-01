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

"""Tests for :mod:`braincell.io._geometry`.

The contour helpers used to be private methods on ``AscReader`` and could
only be reached by parsing a file. Testing them directly is the point of
the extraction: a circle, an ellipse, and a two-layer stack have known
answers, which a real reconstruction does not.
"""

import unittest

import numpy as np

from braincell.io import _geometry


def _circle(radius: float, *, n: int = 64, center=(0.0, 0.0), z: float = 0.0) -> np.ndarray:
    """A closed, evenly traced circle in the plane ``z``."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full(n, z, dtype=float),
        ]
    )


def _ellipse(major: float, minor: float, *, n: int = 128, z: float = 0.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([major * np.cos(angles), minor * np.sin(angles), np.full(n, z, dtype=float)])


class ShouldCopyAttachPointTest(unittest.TestCase):
    """The one rule where the SWC and ASC readers deliberately disagree."""

    def test_a_veto_wins_over_every_other_input(self) -> None:
        for same_xyz in (True, False):
            for keep_radius_jump in (True, False):
                with self.subTest(same_xyz=same_xyz, keep_radius_jump=keep_radius_jump):
                    self.assertFalse(
                        _geometry.should_copy_attach_point(
                            allow_copy=False,
                            same_xyz=same_xyz,
                            same_radius=False,
                            keep_radius_jump=keep_radius_jump,
                        )
                    )

    def test_a_displaced_first_point_is_always_copied(self) -> None:
        for keep_radius_jump in (True, False):
            with self.subTest(keep_radius_jump=keep_radius_jump):
                self.assertTrue(
                    _geometry.should_copy_attach_point(
                        allow_copy=True,
                        same_xyz=False,
                        same_radius=True,
                        keep_radius_jump=keep_radius_jump,
                    )
                )

    def test_swc_policy_keeps_a_coincident_point_when_the_radius_jumps(self) -> None:
        self.assertTrue(
            _geometry.should_copy_attach_point(
                allow_copy=True,
                same_xyz=True,
                same_radius=False,
                keep_radius_jump=True,
            )
        )

    def test_asc_policy_suppresses_a_coincident_point_whatever_the_radius(self) -> None:
        self.assertFalse(
            _geometry.should_copy_attach_point(
                allow_copy=True,
                same_xyz=True,
                same_radius=False,
                keep_radius_jump=False,
            )
        )

    def test_a_fully_coincident_point_is_never_copied(self) -> None:
        for keep_radius_jump in (True, False):
            with self.subTest(keep_radius_jump=keep_radius_jump):
                self.assertFalse(
                    _geometry.should_copy_attach_point(
                        allow_copy=True,
                        same_xyz=True,
                        same_radius=True,
                        keep_radius_jump=keep_radius_jump,
                    )
                )


class SyntheticSomaGeometryTest(unittest.TestCase):
    def test_builds_a_cylinder_along_x_through_the_centre(self) -> None:
        points, radii = _geometry.synthetic_soma_geometry(np.array([1.0, 2.0, 3.0]), 4.0)

        np.testing.assert_allclose(points, [[-3.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5.0, 2.0, 3.0]])
        np.testing.assert_allclose(radii, [4.0, 4.0, 4.0])

    def test_total_length_is_twice_the_radius(self) -> None:
        points, _ = _geometry.synthetic_soma_geometry(np.zeros(3), 2.5)

        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        self.assertAlmostEqual(length, 5.0)

    def test_the_minimum_radius_floor_keeps_the_branch_non_degenerate(self) -> None:
        # A reconstruction can record radius 0 for the point a stand-in soma
        # is built from; without the floor the branch would have zero length.
        points, _ = _geometry.synthetic_soma_geometry(np.zeros(3), _geometry.MIN_SYNTHETIC_LENGTH_UM)

        self.assertGreater(float(np.linalg.norm(points[-1] - points[0])), 0.0)


class BoundingBoxTest(unittest.TestCase):
    def test_bbox_ignores_z(self) -> None:
        contour = np.array([[0.0, 1.0, 100.0], [4.0, -1.0, -100.0], [2.0, 3.0, 0.0]])

        self.assertEqual(_geometry.bbox_xy(contour), (0.0, 4.0, -1.0, 3.0))

    def test_touching_boxes_count_as_intersecting(self) -> None:
        self.assertTrue(_geometry.bboxes_intersect_xy((0.0, 1.0, 0.0, 1.0), (1.0, 2.0, 0.0, 1.0)))

    def test_disjoint_boxes_do_not_intersect(self) -> None:
        self.assertFalse(_geometry.bboxes_intersect_xy((0.0, 1.0, 0.0, 1.0), (1.5, 2.0, 0.0, 1.0)))
        self.assertFalse(_geometry.bboxes_intersect_xy((0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 5.0, 6.0)))

    def test_a_single_contour_box_is_padded(self) -> None:
        contour = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])

        self.assertEqual(_geometry.loose_bbox_xy((contour,)), (-0.5, 2.5, -0.5, 2.5))

    def test_a_stack_box_is_the_plain_union(self) -> None:
        first = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
        second = np.array([[1.0, -1.0, 1.0], [3.0, 1.0, 1.0]])

        self.assertEqual(_geometry.loose_bbox_xy((first, second)), (0.0, 3.0, -1.0, 2.0))

    def test_point_membership_is_inclusive_and_ignores_z(self) -> None:
        box = (0.0, 1.0, 0.0, 1.0)

        self.assertTrue(_geometry.point_inside_bbox_xy(np.array([0.0, 1.0, 999.0]), box))
        self.assertFalse(_geometry.point_inside_bbox_xy(np.array([1.5, 0.5, 0.0]), box))


class GroupContourStacksTest(unittest.TestCase):
    def test_no_contours_gives_no_stacks(self) -> None:
        self.assertEqual(_geometry.group_contour_stacks(()), ())

    def test_overlapping_contours_form_one_stack(self) -> None:
        stacks = _geometry.group_contour_stacks(
            (_circle(5.0, z=0.0), _circle(4.0, z=1.0), _circle(3.0, z=2.0)),
        )

        self.assertEqual(len(stacks), 1)
        self.assertEqual(len(stacks[0]), 3)

    def test_a_gap_starts_a_new_stack(self) -> None:
        stacks = _geometry.group_contour_stacks(
            (
                _circle(1.0, center=(0.0, 0.0)),
                _circle(1.0, center=(0.5, 0.0), z=1.0),
                _circle(1.0, center=(100.0, 0.0), z=2.0),
            )
        )

        self.assertEqual([len(stack) for stack in stacks], [2, 1])

    def test_grouping_is_by_adjacency_not_by_global_overlap(self) -> None:
        # The middle contour is far away, so the first and third end up in
        # separate stacks even though they overlap each other.
        stacks = _geometry.group_contour_stacks((_circle(1.0), _circle(1.0, center=(50.0, 0.0)), _circle(1.0)))

        self.assertEqual([len(stack) for stack in stacks], [1, 1, 1])


class ConstantZTest(unittest.TestCase):
    def test_a_planar_contour_reports_its_z(self) -> None:
        self.assertEqual(_geometry.constant_z(_circle(2.0, z=7.5)), 7.5)

    def test_a_tilted_contour_reports_none(self) -> None:
        contour = _circle(2.0, z=0.0)
        contour[3, 2] = 1.0

        self.assertIsNone(_geometry.constant_z(contour))

    def test_a_deviation_inside_the_tolerance_is_accepted(self) -> None:
        contour = _circle(2.0, z=0.0)
        contour[3, 2] = 1e-9

        self.assertEqual(_geometry.constant_z(contour), 0.0)


class ContourCenterTest(unittest.TestCase):
    def test_a_circle_centroid_sits_at_its_centre(self) -> None:
        # Contours are stored open, as NEURON expects, so the fixed 101
        # arc-length samples miss the closing segment and leave a bias of
        # roughly one sample spacing. 0.05 um on a radius-3 circle.
        center, resampled = _geometry.contour_center(_circle(3.0, center=(10.0, -4.0), z=2.0))

        np.testing.assert_allclose(center[:2], [10.0, -4.0], atol=0.05)
        self.assertAlmostEqual(center[2], 2.0)
        self.assertEqual(resampled.shape, (101, 3))

    def test_dense_tracing_on_one_side_does_not_pull_the_centroid(self) -> None:
        # This is the whole reason for arc-length resampling: a plain mean
        # over the raw points would be dragged towards the dense arc.
        sparse = _circle(1.0, n=16)
        dense = np.concatenate([sparse, _circle(1.0, n=200)[:60]])
        raw_mean = dense.mean(axis=0)

        center, _ = _geometry.contour_center(dense)

        self.assertLess(float(np.linalg.norm(center[:2])), float(np.linalg.norm(raw_mean[:2])))


class PrincipalAxisSamplingTest(unittest.TestCase):
    def test_an_ellipse_is_sampled_along_its_major_axis(self) -> None:
        xy, diameters = _geometry.principal_axis_sampling(_ellipse(10.0, 2.0))

        self.assertEqual(xy.shape, (21, 2))
        self.assertEqual(diameters.shape, (21,))
        # The samples advance along x (the major axis) and barely move in y.
        self.assertGreater(xy[-1, 0] - xy[0, 0], 15.0)
        np.testing.assert_allclose(xy[:, 1], 0.0, atol=0.05)

    def test_the_widest_diameter_is_near_the_middle(self) -> None:
        _, diameters = _geometry.principal_axis_sampling(_ellipse(10.0, 2.0))

        self.assertAlmostEqual(float(diameters.max()), 4.0, delta=0.05)
        self.assertEqual(int(np.argmax(diameters)), 10)

    def test_the_major_axis_orientation_is_deterministic(self) -> None:
        # The sign of an eigenvector is arbitrary; the reader depends on the
        # convention that fixes it, so reversing the traced direction must
        # not reverse the emitted point order.
        forward, _ = _geometry.principal_axis_sampling(_ellipse(10.0, 2.0))
        reversed_trace, _ = _geometry.principal_axis_sampling(_ellipse(10.0, 2.0)[::-1])

        self.assertGreater(forward[-1, 0], forward[0, 0])
        self.assertGreater(reversed_trace[-1, 0], reversed_trace[0, 0])

    def test_a_circle_yields_a_symmetric_profile(self) -> None:
        # The two end samples are deliberately smoothed against their
        # neighbours, and the open contour biases them slightly differently,
        # so they agree to ~3% rather than exactly.
        _, diameters = _geometry.principal_axis_sampling(_circle(5.0, n=256))

        np.testing.assert_allclose(diameters, diameters[::-1], rtol=0.05)
        np.testing.assert_allclose(diameters[1:-1], diameters[1:-1][::-1], rtol=0.01)


class ContourToCentroidTest(unittest.TestCase):
    def test_returns_twenty_one_points_at_the_contour_z(self) -> None:
        points, radii, center = _geometry.contour_to_centroid(_circle(4.0, n=256, z=3.0))

        self.assertEqual(points.shape, (21, 3))
        self.assertEqual(radii.shape, (21,))
        np.testing.assert_allclose(points[:, 2], 3.0)
        np.testing.assert_allclose(center[:2], 0.0, atol=0.05)

    def test_the_widest_radius_matches_the_circle(self) -> None:
        _, radii, _ = _geometry.contour_to_centroid(_circle(4.0, n=256))

        self.assertAlmostEqual(float(radii.max()), 4.0, delta=0.05)


class ContourStackTest(unittest.TestCase):
    def test_each_layer_becomes_one_cable_point(self) -> None:
        stack = (_circle(3.0, n=256, z=0.0), _circle(3.0, n=256, z=2.0))

        points, radii = _geometry.contour_stack_to_centroid(stack)

        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(points[:, 2], [0.0, 2.0], atol=1e-6)
        # NEURON's estimate is the mean radius plus perimeter / 2*pi, so a
        # radius-3 circle comes out near 3 rather than exactly 3.
        np.testing.assert_allclose(radii, radii[0], atol=1e-6)
        self.assertAlmostEqual(float(radii[0]), 3.0, delta=0.2)

    def test_a_single_layer_stack_centre_is_that_layer(self) -> None:
        center = _geometry.contour_stack_center((_circle(3.0, center=(7.0, 0.0), z=1.0),))

        np.testing.assert_allclose(center, [7.0, 0.0, 1.0], atol=0.05)

    def test_the_stack_centre_is_halfway_along_the_traced_path(self) -> None:
        stack = (
            _circle(1.0, z=0.0),
            _circle(1.0, z=1.0),
            _circle(1.0, z=9.0),
        )

        center = _geometry.contour_stack_center(stack)

        self.assertAlmostEqual(float(center[2]), 4.5, places=6)

    def test_a_degenerate_stack_falls_back_to_the_first_layer(self) -> None:
        # Every layer at the same place: total path length is zero, so
        # interpolating along it would divide by zero.
        stack = (_circle(1.0, z=0.0), _circle(1.0, z=0.0))

        center = _geometry.contour_stack_center(stack)

        np.testing.assert_allclose(center[:2], 0.0, atol=0.05)
        self.assertEqual(float(center[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
