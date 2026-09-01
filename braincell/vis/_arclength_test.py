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

from braincell.vis._arclength import (
    ArcPolyline,
    cumulative_arclength_um,
    interpolate_at,
    ordered_span,
    segment_index_at,
)

# 3-4-5 triangle then a 5 µm leg: cumulative = [0, 5, 10].
_L_SHAPE_2D = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 9.0]])
# A different 3D curve with the same cumulative arc lengths.
_L_SHAPE_3D = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 5.0]])
# The 2D curve embedded in 3D, so the x/y parameterisation must match.
_L_SHAPE_2D_IN_3D = np.column_stack((_L_SHAPE_2D, np.zeros(len(_L_SHAPE_2D))))


class CumulativeArclengthTest(unittest.TestCase):
    def test_starts_at_zero_and_accumulates_segment_lengths(self) -> None:
        np.testing.assert_allclose(cumulative_arclength_um(_L_SHAPE_2D), [0.0, 5.0, 10.0])

    def test_is_dimension_agnostic(self) -> None:
        np.testing.assert_array_equal(
            cumulative_arclength_um(_L_SHAPE_2D),
            cumulative_arclength_um(_L_SHAPE_3D),
        )

    def test_single_point_polyline_has_only_the_zero_entry(self) -> None:
        np.testing.assert_allclose(cumulative_arclength_um(np.array([[1.0, 2.0]])), [0.0])

    def test_empty_polyline_still_returns_the_zero_entry(self) -> None:
        np.testing.assert_allclose(cumulative_arclength_um(np.zeros((0, 2))), [0.0])


class OrderedSpanTest(unittest.TestCase):
    def test_reversed_input_is_normalized(self) -> None:
        self.assertEqual(ordered_span(0.8, 0.2), (0.2, 0.8))

    def test_out_of_range_endpoints_are_clipped(self) -> None:
        self.assertEqual(ordered_span(-0.5, 1.7), (0.0, 1.0))


class SegmentIndexAtTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cumulative_um = cumulative_arclength_um(_L_SHAPE_2D)

    def _index(self, x: float) -> int:
        index, _ = segment_index_at(self.cumulative_um, total_um=10.0, n_segments=2, x=x)
        return index

    def test_fraction_maps_to_the_containing_segment(self) -> None:
        self.assertEqual(self._index(0.0), 0)
        self.assertEqual(self._index(0.25), 0)
        self.assertEqual(self._index(0.75), 1)

    def test_out_of_range_fractions_clamp_to_valid_segments(self) -> None:
        self.assertEqual(self._index(-3.0), 0)
        self.assertEqual(self._index(1.0), 1)
        self.assertEqual(self._index(9.0), 1)

    def test_returns_absolute_arc_length_alongside_the_index(self) -> None:
        _, arc_length_um = segment_index_at(self.cumulative_um, total_um=10.0, n_segments=2, x=0.3)
        self.assertAlmostEqual(arc_length_um, 3.0)

    def test_total_is_taken_from_the_caller_not_the_cumulative_array(self) -> None:
        # A layout branch carries its own ``total_length_um``, which need
        # not equal ``cumulative[-1]``; the helper must honour the caller.
        _, arc_length_um = segment_index_at(self.cumulative_um, total_um=100.0, n_segments=2, x=0.5)
        self.assertAlmostEqual(arc_length_um, 50.0)


class InterpolateAtTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cumulative_um = cumulative_arclength_um(_L_SHAPE_2D)

    def test_midpoint_of_the_first_segment(self) -> None:
        point = interpolate_at(_L_SHAPE_2D, self.cumulative_um, total_um=10.0, x=0.25)
        np.testing.assert_allclose(point, [1.5, 2.0])

    def test_scalar_arrays_interpolate_to_a_single_value(self) -> None:
        radii = np.array([2.0, 1.0, 0.0])
        value = interpolate_at(radii, self.cumulative_um, total_um=10.0, x=0.25)
        self.assertAlmostEqual(float(value), 1.5)

    def test_zero_length_segment_returns_a_copy_of_its_start(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
        cumulative_um = cumulative_arclength_um(points)
        result = interpolate_at(points, cumulative_um, total_um=4.0, x=0.0)
        np.testing.assert_allclose(result, [0.0, 0.0])
        result[0] = 99.0
        np.testing.assert_allclose(points[0], [0.0, 0.0])


class ArcPolylineTest(unittest.TestCase):
    def test_point_at_endpoints_and_middle(self) -> None:
        arc = ArcPolyline(_L_SHAPE_2D)
        np.testing.assert_allclose(arc.point_at(0.0), [0.0, 0.0])
        np.testing.assert_allclose(arc.point_at(0.5), [3.0, 4.0])
        np.testing.assert_allclose(arc.point_at(1.0), [3.0, 9.0])

    def test_out_of_range_fractions_clamp_to_the_endpoints(self) -> None:
        arc = ArcPolyline(_L_SHAPE_2D)
        np.testing.assert_allclose(arc.point_at(-2.0), [0.0, 0.0])
        np.testing.assert_allclose(arc.point_at(4.0), [3.0, 9.0])

    def test_same_parameterisation_in_2d_and_3d(self) -> None:
        arc_2d = ArcPolyline(_L_SHAPE_2D)
        arc_3d = ArcPolyline(_L_SHAPE_2D_IN_3D)
        self.assertEqual(arc_2d.n_dim, 2)
        self.assertEqual(arc_3d.n_dim, 3)
        for x in (0.0, 0.13, 0.5, 0.77, 1.0):
            np.testing.assert_array_equal(arc_2d.point_at(x), arc_3d.point_at(x)[:2])
            self.assertEqual(arc_3d.point_at(x)[2], 0.0)

    def test_supplied_cumulative_is_used_verbatim(self) -> None:
        supplied = np.array([0.0, 1.0, 2.0])
        arc = ArcPolyline(_L_SHAPE_2D, supplied)
        np.testing.assert_array_equal(arc.cumulative_um, supplied)
        self.assertEqual(arc.total_um, 2.0)

    def test_scalar_at_interpolates_per_vertex_values(self) -> None:
        arc = ArcPolyline(_L_SHAPE_2D)
        self.assertAlmostEqual(arc.scalar_at(np.array([2.0, 1.0, 0.0]), 0.25), 1.5)

    def test_scalar_at_falls_back_to_empty_for_an_empty_array(self) -> None:
        arc = ArcPolyline(_L_SHAPE_2D)
        self.assertEqual(arc.scalar_at(np.zeros(0), 0.5, empty=7.0), 7.0)

    def test_subspan_orders_clips_and_selects_interior_vertices(self) -> None:
        arc = ArcPolyline(_L_SHAPE_2D)
        lo, hi, interior = arc.subspan(0.9, 0.1)
        self.assertEqual((lo, hi), (0.1, 0.9))
        # Only the corner at arc length 5 lies strictly inside [1, 9].
        np.testing.assert_array_equal(interior, [False, True, False])

    def test_subspan_endpoints_are_excluded_from_the_interior_mask(self) -> None:
        arc = ArcPolyline(_L_SHAPE_2D)
        _, _, interior = arc.subspan(0.5, 1.0)
        np.testing.assert_array_equal(interior, [False, False, False])


class DegenerateArcPolylineTest(unittest.TestCase):
    """Overlays are user-driven, so an empty branch must not raise mid-render."""

    def test_zero_length_polyline_reports_degenerate_and_returns_its_start(self) -> None:
        arc = ArcPolyline(np.array([[1.0, 2.0], [1.0, 2.0]]))
        self.assertTrue(arc.is_degenerate)
        np.testing.assert_allclose(arc.point_at(0.5), [1.0, 2.0])

    def test_point_at_returns_a_copy_of_the_first_vertex(self) -> None:
        points = np.array([[1.0, 2.0], [1.0, 2.0]])
        arc = ArcPolyline(points)
        result = arc.point_at(0.5)
        result[0] = 99.0
        np.testing.assert_allclose(points[0], [1.0, 2.0])

    def test_empty_polyline_returns_a_zero_vector_of_the_right_width(self) -> None:
        np.testing.assert_allclose(ArcPolyline(np.zeros((0, 2))).point_at(0.5), [0.0, 0.0])
        np.testing.assert_allclose(ArcPolyline(np.zeros((0, 3))).point_at(0.5), [0.0, 0.0, 0.0])

    def test_single_vertex_polyline_is_degenerate(self) -> None:
        arc = ArcPolyline(np.array([[5.0, 6.0]]))
        self.assertTrue(arc.is_degenerate)
        np.testing.assert_allclose(arc.point_at(1.0), [5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
