# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from braincell._morphology_utils import (
    compute_line_ratios,
    calculate_total_resistance_and_area,
    find_ratio_interval,
    generate_interpolated_nodes,
)


class Test_compute_line_ratios:
    def test_straight_line_returns_correct_ratios(self):
        # Points along a straight line
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [3, 0, 0]
        ])

        result = compute_line_ratios(points)
        expected = np.array([0.0, 0.33333333, 1.0])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_straight_line_returns_correct_ratios_v2(self):
        # Points along a straight line
        points = np.array([
            [3, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])

        result = compute_line_ratios(points)
        expected = np.array([0.0, 0.666666666, 1.0])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_straight_line_returns_correct_ratios_v3(self):
        # Points along a straight line
        points = np.array([
            [3, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ])

        result = compute_line_ratios(points)
        expected = np.array([0.0, 0.75, 1.0])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zigzag_line_accounts_for_direction_changes(self):
        # Points with changing directions
        points = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [2, 0, 0],
            [3, 1, 0]
        ])

        result = compute_line_ratios(points)

        # Total length = sqrt(2) + sqrt(2) + sqrt(2) = 3*sqrt(2)
        expected = np.array([0.0, 1 / 3, 2 / 3, 1.0])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_points_in_3d_space_calculated_correctly(self):
        # Points in 3D space
        points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 0]
        ])

        result = compute_line_ratios(points)

        expected = np.array([0.0, 0.5, 1.0])

        print(result)
        print(expected)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_single_point_returns_zero(self):
        # Single point has no length
        points = np.array([[1, 2, 3]])

        result = compute_line_ratios(points)
        expected = np.array([0.0])

        np.testing.assert_array_equal(result, expected)

    def test_duplicate_points_handled_correctly(self):
        # Duplicate points should be treated as zero-length segments
        points = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0]
        ])

        result = compute_line_ratios(points)
        expected = np.array([0.0, 0.0, 1.0])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_all_identical_points_returns_zeros(self):
        # All identical points should return all zeros
        points = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

        result = compute_line_ratios(points)
        expected = np.array([0.0, 0.0, 0.0])

        np.testing.assert_array_equal(result, expected)

    def test_jax_array_input_works(self):
        # Test with JAX array input
        points = jnp.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0]
        ])

        result = compute_line_ratios(points)
        expected = jnp.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-6)

    def test_empty_array_handled_gracefully(self):
        # Empty array edge case
        with pytest.raises(Exception):
            # This should raise an exception as we can't compute ratios on empty points
            compute_line_ratios(np.array([]))


class Test_calculate_total_resistance_and_area:

    def test_simple_straight_cylinder_returns_correct_values(self):
        # Simple cylinder: 10μm length, 1μm diameter
        points = np.array([[0, 0, 0, 1], [10, 0, 0, 1]]) * u.um

        resistance, area = calculate_total_resistance_and_area(points)

        # For a cylinder of length 10μm, diameter 1μm, resistivity 100 Ω·cm:
        # R = (100 Ω·cm) * (10 μm) / (π * (0.5 μm)²)
        # A = π * 1μm * sqrt(10² + 0²) = π * 10μm
        expected_resistance = 100 * u.ohm * u.cm * 10 * u.um / (np.pi * (0.5 * u.um) ** 2)
        expected_area = np.pi * 1 * u.um * 10 * u.um

        assert u.math.allclose(resistance, expected_resistance)
        assert u.math.allclose(area, expected_area)
        assert u.math.allclose(u.get_mantissa(area), u.get_mantissa(expected_area))

    def test_truncated_cone_returns_correct_values(self):
        # Truncated cone: 10μm length, 2μm diameter at start, 1μm at end
        points = np.array([[0, 0, 0, 2], [10, 0, 0, 1]]) * u.um

        resistance, area = calculate_total_resistance_and_area(points)

        # For a truncated cone:
        # R = (100 Ω·cm) * (10 μm) / (π * 1μm * 0.5μm)
        # Slant height = sqrt(10² + (1-0.5)²) = ~10.025μm
        # A = π * (1μm + 0.5μm) * 10.025μm
        expected_resistance = 100 * u.ohm * u.cm * 10 * u.um / (np.pi * 1 * u.um * 0.5 * u.um)
        slant_height = np.sqrt(10 ** 2 + 0.5 ** 2) * u.um
        expected_area = np.pi * (1 + 0.5) * u.um * slant_height

        assert u.math.allclose(resistance, expected_resistance)
        assert u.math.allclose(area, expected_area)

    def test_branched_structure_returns_summed_values(self):
        # Y-shaped structure with 3 segments
        points = np.array([
            [0, 0, 0, 1],  # Start point
            [10, 0, 0, 1],  # Branch point
            [15, 5, 0, 0.8],  # End of branch 1
            [15, -5, 0, 0.8]  # End of branch 2
        ]) * u.um

        # Calculate manually for each segment
        segment1 = points[0:2]  # Start to branch
        segment2 = points[[1, 2]]  # Branch to end 1
        segment3 = points[[2, 3]]  # Branch to end 2

        r1, a1 = calculate_total_resistance_and_area(segment1)
        r2, a2 = calculate_total_resistance_and_area(segment2)
        r3, a3 = calculate_total_resistance_and_area(segment3)

        r_total, a_total = calculate_total_resistance_and_area(points)

        assert u.math.allclose(r_total, r1 + r2 + r3)
        assert u.math.allclose(a_total, a1 + a2 + a3)

    def test_zero_length_segment_returns_zero_resistance_and_area(self):
        # Point with zero length
        points = np.array([[0, 0, 0, 1], [0, 0, 0, 1]]) * u.um

        resistance, area = calculate_total_resistance_and_area(points)

        assert resistance == 0 * u.ohm
        assert area == 0 * u.um ** 2

    def test_single_point_returns_zero_values(self):
        # Single point cannot form a segment
        points = np.array([[0, 0, 0, 1]]) * u.um

        resistance, area = calculate_total_resistance_and_area(points)

        assert resistance == 0 * u.ohm
        assert area == 0 * u.um ** 2

    def test_zero_diameter_handles_gracefully(self):
        # Zero diameter should handle division by zero gracefully
        points = np.array([[0, 0, 0, 0], [10, 0, 0, 0]]) * u.um

        # Should return infinity for resistance due to zero cross-section
        resistance, area = calculate_total_resistance_and_area(points)

        assert u.math.isinf(resistance)
        assert u.math.allclose(area, 0 * u.um ** 2)

    def test_custom_resistivity_scales_resistance_proportionally(self):
        points = np.array([[0, 0, 0, 1], [10, 0, 0, 1]]) * u.um

        resistance1, area1 = calculate_total_resistance_and_area(
            points, resistivity=100.0 * u.ohm * u.cm
        )
        resistance2, area2 = calculate_total_resistance_and_area(
            points, resistivity=200.0 * u.ohm * u.cm
        )

        assert u.math.allclose(resistance2, 2 * resistance1)
        assert u.math.allclose(area1, area2)  # Area should be unchanged

    def test_works_with_jax_arrays(self):
        points_np = np.array([[0, 0, 0, 1], [10, 0, 0, 1]]) * u.um
        points_jax = jnp.array([[0, 0, 0, 1], [10, 0, 0, 1]]) * u.um

        resistance_np, area_np = calculate_total_resistance_and_area(points_np)
        resistance_jax, area_jax = calculate_total_resistance_and_area(points_jax)

        assert u.math.allclose(resistance_np, resistance_jax)
        assert u.math.allclose(area_np, area_jax)


class TestFindRatioInterval(unittest.TestCase):

    def setUp(self):
        self.ratios = np.array([0.0, 0.3, 0.6, 1.0])

    def test_target_ratio_falls_between_values(self):
        lower_idx, upper_idx = find_ratio_interval(self.ratios, 0.4)
        self.assertEqual(lower_idx, 1)
        self.assertEqual(upper_idx, 2)

    def test_target_ratio_exactly_matches_middle_value(self):
        lower_idx, upper_idx = find_ratio_interval(self.ratios, 0.6)
        self.assertEqual(lower_idx, 1)
        self.assertEqual(upper_idx, 2)

    def test_target_ratio_below_minimum(self):
        lower_idx, upper_idx = find_ratio_interval(self.ratios, -0.1)
        self.assertEqual(lower_idx, 0)
        self.assertEqual(upper_idx, 1)

    def test_target_ratio_equals_minimum(self):
        lower_idx, upper_idx = find_ratio_interval(self.ratios, 0.0)
        self.assertEqual(lower_idx, 0)
        self.assertEqual(upper_idx, 1)

    def test_target_ratio_above_maximum(self):
        lower_idx, upper_idx = find_ratio_interval(self.ratios, 1.2)
        self.assertEqual(lower_idx, 2)
        self.assertEqual(upper_idx, 3)

    def test_target_ratio_equals_maximum(self):
        lower_idx, upper_idx = find_ratio_interval(self.ratios, 1.0)
        self.assertEqual(lower_idx, 2)
        self.assertEqual(upper_idx, 3)

    def test_single_point_ratio_list(self):
        ratios = np.array([0.5])
        lower_idx, upper_idx = find_ratio_interval(ratios, 0.3)
        self.assertEqual(lower_idx, 0)
        self.assertEqual(upper_idx, 1)

    def test_works_with_brainunit_quantities(self):
        ratios = u.Quantity(np.array([0.0, 0.3, 0.6, 1.0]), u.cm)
        target = u.Quantity(0.4, u.cm)
        lower_idx, upper_idx = find_ratio_interval(ratios, target)
        self.assertEqual(lower_idx, 1)
        self.assertEqual(upper_idx, 2)

    def test_works_with_jax_arrays(self):
        ratios = jnp.array([0.0, 0.3, 0.6, 1.0])
        target = jnp.array(0.4)
        lower_idx, upper_idx = find_ratio_interval(ratios, target)
        self.assertEqual(lower_idx, 1)
        self.assertEqual(upper_idx, 2)


class TestGenerateInterpolatedNodes(unittest.TestCase):

    def test_basic_interpolation_works(self):
        node_pre = np.array([
            [0, 0, 0, 1],  # x, y, z, diameter
            [10, 0, 0, 2]
        ])
        nseg = 2
        result = generate_interpolated_nodes(node_pre, nseg)
        self.assertEqual(result.shape, (5, 4))  # 2*nseg+1 = 5 points
        np.testing.assert_almost_equal(result[0], [0, 0, 0, 1])
        np.testing.assert_almost_equal(result[2], [5, 0, 0, 1.5])
        np.testing.assert_almost_equal(result[4], [10, 0, 0, 2])

    def test_handles_3d_path_correctly(self):
        node_pre = np.array([
            [0, 0, 0, 1],
            [5, 5, 5, 2],
            [10, 0, 0, 3]
        ])
        nseg = 3
        result = generate_interpolated_nodes(node_pre, nseg)
        self.assertEqual(result.shape, (7, 4))
        np.testing.assert_almost_equal(result[0], [0, 0, 0, 1])
        np.testing.assert_almost_equal(result[-1], [10, 0, 0, 3])

    def test_handles_large_nseg_value(self):
        node_pre = np.array([
            [0, 0, 0, 1],
            [10, 0, 0, 2]
        ])
        nseg = 50
        result = generate_interpolated_nodes(node_pre, nseg)
        self.assertEqual(result.shape, (101, 4))  # 2*50+1 = 101 points

    def test_handles_single_node_input(self):
        node_pre = np.array([[5, 5, 5, 1.5]])
        nseg = 3
        result = generate_interpolated_nodes(node_pre, nseg)
        self.assertEqual(result.shape, (7, 4))
        for i in range(7):
            np.testing.assert_almost_equal(result[i], [5, 5, 5, 1.5])

    def test_works_with_jax_array_input(self):
        node_pre = jnp.array([
            [0, 0, 0, 1],
            [10, 0, 0, 2]
        ])
        nseg = 2
        result = generate_interpolated_nodes(node_pre, nseg)
        self.assertEqual(result.shape, (5, 4))
        np.testing.assert_almost_equal(result[0], [0, 0, 0, 1])
        np.testing.assert_almost_equal(result[-1], [10, 0, 0, 2])

    def test_handles_brainunit_quantity_input(self):
        node_pre = u.Quantity(np.array([
            [0, 0, 0, 1],
            [10, 0, 0, 2]
        ]), unit=u.um)
        nseg = 2
        result = generate_interpolated_nodes(node_pre, nseg)
        self.assertEqual(result.shape, (5, 4))
        self.assertTrue(isinstance(result, u.Quantity))
        self.assertEqual(result.unit, u.um)

    def test_nseg_zero_returns_start_and_end_points(self):
        node_pre = np.array([
            [0, 0, 0, 1],
            [10, 0, 0, 2]
        ])
        nseg = 0
        result = generate_interpolated_nodes(node_pre, nseg)
        self.assertEqual(result.shape, (1, 4))
        np.testing.assert_almost_equal(result[0], [0, 0, 0, 1])

    def test_interpolation_preserves_values_at_original_points(self):
        node_pre = np.array([
            [0, 0, 0, 1],
            [5, 5, 5, 2],
            [10, 10, 10, 3],
            [15, 15, 15, 4]
        ])
        nseg = 3
        result = generate_interpolated_nodes(node_pre, nseg)
        # Original points should be included or closely approximated
        for i, point in enumerate(node_pre):
            position = i * (2 * nseg) / (len(node_pre) - 1)
            closest_idx = int(round(position))
            np.testing.assert_allclose(result[closest_idx], point, rtol=1e-5)
