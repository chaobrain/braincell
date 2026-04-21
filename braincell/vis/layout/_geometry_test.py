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
import unittest

import numpy as np

from braincell.vis.layout._common import LayoutBranch2D, _LayoutSpec2D
from braincell.vis.layout._geometry import (
    _blend_angle_rad,
    _layout_branch_from_points,
    _segment_angles_rad,
    _shortest_angle_delta_rad,
    _smoothstep,
    _vector_angle_rad,
    point_on_layout_branch,
    sample_layout_branch,
    tangent_on_layout_branch,
)


def _make_straight_layout() -> LayoutBranch2D:
    """Build a 2-segment straight-line layout running along +x from the
    origin, lengths 10 µm and 20 µm, uniform radius 1 µm."""
    segment_points_um = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]], dtype=float)
    spec = _LayoutSpec2D(
        segment_lengths_um=np.array([10.0, 20.0], dtype=float),
        radii_proximal_um=np.array([1.0, 1.0], dtype=float),
        radii_distal_um=np.array([1.0, 1.0], dtype=float),
    )

    class _FakeBranch:
        index = 0
        name = "straight"
        type = "soma"

    return _layout_branch_from_points(_FakeBranch(), spec, segment_points_um)


class ShortestAngleDeltaTest(unittest.TestCase):
    def test_values_within_pi_pass_through(self) -> None:
        self.assertAlmostEqual(_shortest_angle_delta_rad(0.5), 0.5)
        self.assertAlmostEqual(_shortest_angle_delta_rad(-1.5), -1.5)

    def test_wraps_around_2pi(self) -> None:
        # +3 rad is equivalent to +3 - 2π = -3.283… which is out of
        # range; atan2(sin,cos) is the canonical wrap and returns the
        # principal value.
        self.assertAlmostEqual(_shortest_angle_delta_rad(3.0), 3.0, places=6)
        self.assertAlmostEqual(_shortest_angle_delta_rad(3.5), 3.5 - 2.0 * math.pi, places=6)
        self.assertAlmostEqual(_shortest_angle_delta_rad(-3.5), -3.5 + 2.0 * math.pi, places=6)


class VectorAngleTest(unittest.TestCase):
    def test_cardinal_directions(self) -> None:
        self.assertAlmostEqual(_vector_angle_rad(np.array([1.0, 0.0])), 0.0)
        self.assertAlmostEqual(_vector_angle_rad(np.array([0.0, 1.0])), math.pi / 2.0)
        self.assertAlmostEqual(_vector_angle_rad(np.array([-1.0, 0.0])), math.pi)
        self.assertAlmostEqual(_vector_angle_rad(np.array([0.0, -1.0])), -math.pi / 2.0)


class SmoothstepTest(unittest.TestCase):
    def test_monotonic(self) -> None:
        xs = np.linspace(0.0, 1.0, 11)
        ys = _smoothstep(xs)
        self.assertTrue(np.all(np.diff(ys) >= 0.0))
        self.assertAlmostEqual(float(ys[0]), 0.0)
        self.assertAlmostEqual(float(ys[-1]), 1.0)

    def test_half_is_half(self) -> None:
        result = _smoothstep(np.array([0.5]))
        self.assertAlmostEqual(float(result[0]), 0.5)


class BlendAngleTest(unittest.TestCase):
    def test_weight_zero_returns_first(self) -> None:
        self.assertAlmostEqual(_blend_angle_rad(0.2, 1.2, 0.0), 0.2)

    def test_weight_one_returns_second(self) -> None:
        self.assertAlmostEqual(_blend_angle_rad(0.2, 1.2, 1.0), 1.2)


class SegmentAnglesRadTest(unittest.TestCase):
    def test_single_segment_gets_target(self) -> None:
        angles = _segment_angles_rad(
            np.array([10.0]),
            attach_angle_rad=0.0,
            target_angle_rad=1.0,
            bend_fraction=0.4,
        )
        self.assertEqual(angles.shape, (1,))
        self.assertAlmostEqual(float(angles[0]), 1.0)

    def test_multi_segment_first_matches_attach_last_matches_target(self) -> None:
        lengths = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)
        angles = _segment_angles_rad(
            lengths,
            attach_angle_rad=0.0,
            target_angle_rad=math.pi / 2.0,
            bend_fraction=0.5,
        )
        self.assertAlmostEqual(float(angles[0]), 0.0, places=6)
        self.assertAlmostEqual(float(angles[-1]), math.pi / 2.0, places=6)


class SamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = _make_straight_layout()

    def test_point_at_zero_is_start(self) -> None:
        point = point_on_layout_branch(self.layout, 0.0)
        self.assertTrue(np.allclose(point, np.array([0.0, 0.0])))

    def test_point_at_one_is_end(self) -> None:
        point = point_on_layout_branch(self.layout, 1.0)
        self.assertTrue(np.allclose(point, np.array([30.0, 0.0])))

    def test_point_in_middle_is_midway(self) -> None:
        point = point_on_layout_branch(self.layout, 0.5)
        # Total length is 30 µm, so x=0.5 lands at 15 µm → (15, 0).
        self.assertTrue(np.allclose(point, np.array([15.0, 0.0]), atol=1e-6))

    def test_tangent_is_straight(self) -> None:
        tangent = tangent_on_layout_branch(self.layout, 0.4)
        self.assertTrue(np.allclose(tangent, np.array([1.0, 0.0])))

    def test_sample_returns_point_and_tangent(self) -> None:
        point, tangent = sample_layout_branch(self.layout, 0.25)
        self.assertTrue(np.allclose(point, np.array([7.5, 0.0]), atol=1e-6))
        self.assertTrue(np.allclose(tangent, np.array([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
