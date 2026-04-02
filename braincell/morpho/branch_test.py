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

from braincell import Branch
from braincell._test_support import FakeBackend, jnp, np, u
from braincell.vis import BackendChooser


class BranchTest(unittest.TestCase):
    def test_branch_init_accepts_valid_quantities(self) -> None:
        branch = Branch(
            lengths=10.0 * u.um,
            radii_proximal=2.0 * u.um,
            radii_distal=1.0 * u.um,
            type="axon",
        )

        self.assertTrue(u.math.array_equal(branch.lengths, np.array([10.0]) * u.um))
        self.assertTrue(u.math.array_equal(branch.radii_proximal, np.array([2.0]) * u.um))
        self.assertTrue(u.math.array_equal(branch.radii_distal, np.array([1.0]) * u.um))
        self.assertEqual(branch.type, "axon")
        self.assertEqual(branch.n_segments, 1)

    def test_branch_init_accepts_valid_point_geometry(self) -> None:
        branch = Branch(
            lengths=[5.0, 5.0] * u.um,
            radii_proximal=[2.0, 1.5] * u.um,
            radii_distal=[1.5, 1.0] * u.um,
            points_proximal=[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)] * u.um,
            points_distal=[(3.0, 4.0, 0.0), (3.0, 4.0, 5.0)] * u.um,
            type="soma",
        )

        self.assertTrue(
            u.math.array_equal(
                branch.points,
                np.array([(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 5.0)]) * u.um,
            )
        )

    def test_branch_init_rejects_point_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "point-derived segment lengths"):
            Branch(
                lengths=[6.0, 5.0] * u.um,
                radii_proximal=[2.0, 1.5] * u.um,
                radii_distal=[1.5, 1.0] * u.um,
                points_proximal=[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)] * u.um,
                points_distal=[(3.0, 4.0, 0.0), (3.0, 4.0, 5.0)] * u.um,
                type="soma",
            )

    def test_from_lengths_accepts_shared_and_paired_radii(self) -> None:
        shared = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii=[3.0, 2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        self.assertTrue(u.math.array_equal(shared.lengths, np.array([10.0, 20.0]) * u.um))
        self.assertTrue(u.math.array_equal(shared.radii_proximal, np.array([3.0, 2.0]) * u.um))
        self.assertTrue(u.math.array_equal(shared.radii_distal, np.array([2.0, 1.0]) * u.um))

        paired = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii_proximal=[3.0, 2.0] * u.um,
            radii_distal=[2.5, 1.0] * u.um,
            type="axon",
        )
        self.assertTrue(u.math.array_equal(paired.lengths, np.array([10.0, 0.0, 20.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired.radii_proximal, np.array([3.0, 2.5, 2.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired.radii_distal, np.array([2.5, 2.0, 1.0]) * u.um))

    def test_from_points_accepts_shared_and_paired_radii(self) -> None:
        shared = Branch.from_points(
            points=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]) * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="soma",
        )
        self.assertTrue(u.math.array_equal(shared.lengths, np.array([5.0, 12.0]) * u.um))
        self.assertTrue(
            u.math.array_equal(
                shared.points_proximal,
                np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]) * u.um,
            )
        )
        self.assertTrue(
            u.math.array_equal(
                shared.points_distal,
                np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]) * u.um,
            )
        )

        paired = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (0.0, 6.0, 8.0)] * u.um,
            radii_proximal=[1.0] * u.um,
            radii_distal=[0.5] * u.um,
            type="basal_dendrite",
        )
        self.assertTrue(u.math.array_equal(paired.lengths, np.array([10.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired.radii_proximal, np.array([1.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired.radii_distal, np.array([0.5]) * u.um))

    def test_from_points_inserts_zero_length_jump_for_radius_discontinuity(self) -> None:
        branch = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)] * u.um,
            radii_proximal=[2.0, 3.0] * u.um,
            radii_distal=[1.0, 0.5] * u.um,
            type="axon",
        )

        self.assertTrue(u.math.array_equal(branch.lengths, np.array([10.0, 0.0, 10.0]) * u.um))
        self.assertTrue(u.math.array_equal(branch.radii, np.array([2.0, 1.0, 3.0, 0.5]) * u.um))
        self.assertTrue(
            u.math.array_equal(
                branch.points,
                np.array(
                    [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
                ) * u.um,
            )
        )

    def test_geometry_properties_compute_correctly(self) -> None:
        branch = Branch.from_lengths(
            lengths=[10.0] * u.um,
            radii_proximal=[2.0] * u.um,
            radii_distal=[1.0] * u.um,
            type="axon",
        )

        expected_area = math.pi * (2.0 + 1.0) * math.sqrt(10.0 ** 2 + (1.0 - 2.0) ** 2)
        expected_volume = math.pi * 10.0 * (2.0 ** 2 + 2.0 * 1.0 + 1.0 ** 2) / 3.0

        self.assertAlmostEqual(branch.length.to_decimal(u.um), 10.0)
        self.assertAlmostEqual(branch.mean_radius.to_decimal(u.um), 1.5)
        self.assertAlmostEqual(float(branch.areas[0].to_decimal(u.um ** 2)), expected_area, places=5)
        self.assertAlmostEqual(float(branch.area.to_decimal(u.um ** 2)), expected_area, places=5)
        self.assertAlmostEqual(float(branch.volumes[0].to_decimal(u.um ** 3)), expected_volume, places=5)
        self.assertAlmostEqual(float(branch.volume.to_decimal(u.um ** 3)), expected_volume, places=5)
        self.assertTrue(u.math.allclose(branch.area, u.math.sum(branch.areas)))
        self.assertTrue(u.math.allclose(branch.volume, u.math.sum(branch.volumes)))

    def test_zero_length_segment_contributes_area_but_not_volume(self) -> None:
        branch = Branch(
            lengths=[10.0, 0.0, 20.0] * u.um,
            radii_proximal=[2.0, 1.0, 4.0] * u.um,
            radii_distal=[1.0, 2.0, 2.0] * u.um,
            type="axon",
        )

        self.assertAlmostEqual(branch.length.to_decimal(u.um), 30.0)
        self.assertAlmostEqual(float(branch.areas[1].to_decimal(u.um ** 2)), math.pi * (2.0 ** 2 - 1.0 ** 2), places=7)
        self.assertAlmostEqual(float(branch.volumes[1].to_decimal(u.um ** 3)), 0.0, places=9)
        self.assertAlmostEqual(branch.mean_radius.to_decimal(u.um), 2.5)

    def test_branch_rejects_zero_total_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "total length must be > 0"):
            Branch.from_lengths(
                lengths=[0.0] * u.um,
                radii_proximal=[1.0] * u.um,
                radii_distal=[2.0] * u.um,
                type="axon",
            )

    def test_shared_view_properties_behave_correctly(self) -> None:
        with_points = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="axon",
        )
        self.assertTrue(u.math.array_equal(with_points.radii, np.array([2.0, 1.5, 1.0]) * u.um))
        self.assertTrue(
            u.math.array_equal(
                with_points.points,
                np.array([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]) * u.um,
            )
        )

        no_points = Branch.from_lengths(
            lengths=[5.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="axon",
        )
        self.assertIsNone(no_points.points)

    def test_branch_init_rejects_missing_or_wrong_units(self) -> None:
        with self.assertRaises(TypeError):
            Branch(lengths=[10.0], radii_proximal=[2.0] * u.um, radii_distal=[1.0] * u.um, type="axon")
        with self.assertRaises(TypeError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0], radii_distal=[1.0] * u.um, type="axon")
        with self.assertRaises(TypeError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0] * u.um, radii_distal=[1.0], type="axon")
        with self.assertRaises(TypeError):
            Branch(
                lengths=[10.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
                points_proximal=[(0.0, 0.0, 0.0)],
                points_distal=[(1.0, 0.0, 0.0)] * u.um,
                type="axon",
            )
        with self.assertRaises(TypeError):
            Branch(lengths=[10.0] * u.ms, radii_proximal=[2.0] * u.um, radii_distal=[1.0] * u.um, type="axon")
        with self.assertRaises(TypeError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0] * u.ms, radii_distal=[1.0] * u.um, type="axon")
        with self.assertRaises(TypeError):
            Branch(
                lengths=[10.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
                points_proximal=[(0.0, 0.0, 0.0)] * u.ms,
                points_distal=[(1.0, 0.0, 0.0)] * u.um,
                type="axon",
            )

    def test_branch_init_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            Branch(lengths=np.array([[[10.0], [20.0]]]) * u.um, radii_proximal=[2.0, 1.0] * u.um,
                   radii_distal=[1.0, 0.5] * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0] * u.um, radii_proximal=np.array([[[2.0], [1.0]]]) * u.um, radii_distal=[1.0] * u.um,
                   type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0] * u.um, radii_distal=np.array([[[1.0], [0.5]]]) * u.um,
                   type="axon")
        with self.assertRaises(ValueError):
            Branch(
                lengths=[10.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
                points_proximal=[(0.0, 0.0)] * u.um,
                points_distal=[(1.0, 0.0, 0.0)] * u.um,
                type="axon",
            )
        with self.assertRaises(ValueError):
            Branch(
                lengths=[10.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
                points_proximal=[(0.0, 0.0, 0.0)] * u.um,
                points_distal=[(1.0, 0.0, 0.0, 0.0)] * u.um,
                type="axon",
            )

    def test_branch_init_rejects_value_bounds_and_type(self) -> None:
        with self.assertRaises(ValueError):
            Branch(lengths=[-1.0] * u.um, radii_proximal=[2.0] * u.um, radii_distal=[1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[0.0] * u.um, radii_distal=[1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0] * u.um, radii_distal=[0.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0] * u.um, radii_distal=[1.0] * u.um, type="unknown")

    def test_branch_init_rejects_cross_parameter_mismatches(self) -> None:
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0, 1.0] * u.um, radii_distal=[1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0] * u.um, radii_proximal=[2.0] * u.um, radii_distal=[1.0, 0.5] * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch(
                lengths=[10.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
                points_proximal=[(0.0, 0.0, 0.0)] * u.um,
                type="axon",
            )
        with self.assertRaises(ValueError):
            Branch(
                lengths=[10.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
                points_distal=[(1.0, 0.0, 0.0)] * u.um,
                type="axon",
            )
        with self.assertRaises(ValueError):
            Branch(
                lengths=[10.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
                points_proximal=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] * u.um,
                points_distal=[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)] * u.um,
                type="axon",
            )

    def test_from_lengths_rejects_method_protocol_errors(self) -> None:
        with self.assertRaisesRegex(TypeError, "required"):
            Branch.from_lengths(lengths=[10.0] * u.um)
        with self.assertRaisesRegex(TypeError, "provided together"):
            Branch.from_lengths(lengths=[10.0] * u.um, radii_proximal=[2.0] * u.um)
        with self.assertRaisesRegex(TypeError, "provided together"):
            Branch.from_lengths(lengths=[10.0] * u.um, radii_distal=[1.0] * u.um)
        with self.assertRaisesRegex(TypeError, "cannot be provided together"):
            Branch.from_lengths(
                lengths=[10.0] * u.um,
                radii=[2.0, 1.0] * u.um,
                radii_proximal=[2.0] * u.um,
                radii_distal=[1.0] * u.um,
            )
        with self.assertRaisesRegex(ValueError, "length 3"):
            Branch.from_lengths(lengths=[10.0, 20.0] * u.um, radii=[3.0, 2.0] * u.um)
        with self.assertRaises(TypeError):
            Branch.from_lengths(lengths=[10.0], radii=[2.0, 1.0] * u.um)
        with self.assertRaises(TypeError):
            Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.ms)

    def test_from_points_rejects_method_protocol_errors(self) -> None:
        with self.assertRaisesRegex(TypeError, "required"):
            Branch.from_points(points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] * u.um)
        with self.assertRaisesRegex(TypeError, "provided together"):
            Branch.from_points(points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] * u.um, radii_proximal=[1.0] * u.um)
        with self.assertRaisesRegex(TypeError, "provided together"):
            Branch.from_points(points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] * u.um, radii_distal=[0.5] * u.um)
        with self.assertRaisesRegex(TypeError, "cannot be provided together"):
            Branch.from_points(
                points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] * u.um,
                radii=[1.0, 0.5] * u.um,
                radii_proximal=[1.0] * u.um,
                radii_distal=[0.5] * u.um,
            )
        with self.assertRaisesRegex(ValueError, "at least two points"):
            Branch.from_points(points=[(0.0, 0.0, 0.0)] * u.um, radii=[1.0] * u.um)
        with self.assertRaisesRegex(ValueError, "length 2"):
            Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[1.0, 0.8, 0.5] * u.um)
        with self.assertRaises(TypeError):
            Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], radii=[1.0, 0.5] * u.um)
        with self.assertRaises(TypeError):
            Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.ms, radii=[1.0, 0.5] * u.um)
        with self.assertRaises(ValueError):
            Branch.from_points(points=[(0.0, 0.0), (10.0, 0.0)] * u.um, radii=[1.0, 0.5] * u.um)

    def test_property_failures_for_discontinuous_or_missing_views(self) -> None:
        no_points = Branch.from_lengths(
            lengths=[5.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="axon",
        )
        self.assertIsNone(no_points.points)

        discontinuous_radii = Branch(
            lengths=[10.0, 10.0] * u.um,
            radii_proximal=[2.0, 3.0] * u.um,
            radii_distal=[1.0, 0.5] * u.um,
            points_proximal=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um,
            points_distal=[(10.0, 0.0, 0.0), (20.0, 0.0, 0.0)] * u.um,
            type="axon",
        )
        with self.assertRaises(ValueError):
            _ = discontinuous_radii.radii

        discontinuous_points = Branch(
            lengths=[10.0, 10.0] * u.um,
            radii_proximal=[2.0, 1.5] * u.um,
            radii_distal=[1.5, 1.0] * u.um,
            points_proximal=[(0.0, 0.0, 0.0), (10.0, 1.0, 0.0)] * u.um,
            points_distal=[(10.0, 0.0, 0.0), (20.0, 1.0, 0.0)] * u.um,
            type="axon",
        )
        with self.assertRaises(ValueError):
            _ = discontinuous_points.points

    def test_branch_accepts_jax_quantity_inputs(self) -> None:
        if jnp is None:
            self.skipTest("jax is not installed")

        branch = Branch.from_lengths(
            lengths=jnp.array([40.0]) * u.um,
            radii=jnp.array([0.8, 0.4]) * u.um,
            type="axon",
        )
        self.assertTrue(u.math.array_equal(branch.lengths, np.array([40.0]) * u.um))

    def test_branch_equality_supports_vector_fields(self) -> None:
        shared = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii=[3.0, 2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        paired = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii_proximal=[3.0, 2.0] * u.um,
            radii_distal=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        self.assertEqual(shared, paired)

    def test_branch_equality_supports_discontinuous_geometries(self) -> None:
        paired = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii_proximal=[3.0, 2.0] * u.um,
            radii_distal=[1.0, 1.0] * u.um,
        )
        shared = Branch.from_lengths(
            lengths=[10.0, 0.0, 20.0] * u.um,
            radii=[3.0, 1.0, 2.0, 1.0] * u.um,
        )
        self.assertEqual(shared, paired)

    def test_branch_equality_supports_point_geometry(self) -> None:
        left = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 12.0)] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="soma",
        )
        right = Branch.from_points(
            points=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]) * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="soma",
        )
        self.assertEqual(left, right)

    def test_branch_equality_rejects_type_geometry_and_point_mismatches(self) -> None:
        base = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii=[3.0, 2.0, 1.0] * u.um,
            type="axon",
        )
        self.assertNotEqual(
            base,
            Branch.from_lengths(
                lengths=[10.0, 20.0] * u.um,
                radii=[3.0, 2.0, 1.0] * u.um,
                type="basal_dendrite",
            ),
        )
        self.assertNotEqual(
            base,
            Branch.from_lengths(
                lengths=[10.0, 21.0] * u.um,
                radii=[3.0, 2.0, 1.0] * u.um,
                type="axon",
            ),
        )
        self.assertNotEqual(
            Branch.from_points(
                points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um,
                radii=[2.0, 1.0] * u.um,
                type="axon",
            ),
            Branch.from_lengths(
                lengths=[10.0] * u.um,
                radii=[2.0, 1.0] * u.um,
                type="axon",
            ),
        )

    def test_branch_equality_handles_other_types_and_hashing(self) -> None:
        branch = Branch.from_lengths(
            lengths=[10.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="axon",
        )
        self.assertFalse(branch == object())
        with self.assertRaises(TypeError):
            hash(branch)

    def test_vis2d_frustum_mode(self) -> None:
        branch = Branch.from_lengths(
            lengths=[10.0, 15.0] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="dendrite",
        )
        backend = FakeBackend()
        request = branch.vis2d(mode="frustum", show=False, chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.mode, "frustum")
        self.assertGreater(len(request.scene.polygons), 0)

    def test_vis2d_tree_mode(self) -> None:
        branch = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[3.0, 3.0] * u.um,
        )
        backend = FakeBackend()
        request = branch.vis2d(mode="tree", show=False, chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.mode, "tree")
        self.assertEqual(len(request.scene.polylines), 1)

    def test_vis2d_projected_mode_requires_points(self) -> None:
        branch = Branch.from_lengths(
            lengths=[10.0] * u.um,
            radii=[2.0, 2.0] * u.um,
        )
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            branch.vis2d(mode="projected", show=False, chooser=BackendChooser(backends=(backend,)))
