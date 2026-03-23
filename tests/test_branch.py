from __future__ import annotations

import unittest

from ._support import jnp, np, u

from braincell import Branch


class BranchTest(unittest.TestCase):
    def test_branch_accepts_single_segment_scalars(self) -> None:
        branch = Branch(
            lengths=10.0 * u.um,
            radii_prox=1.0 * u.um,
            radii_dist=0.5 * u.um,
            type="axon",
        )

        self.assertTrue(u.math.array_equal(branch.lengths, np.array([10.0]) * u.um))
        self.assertTrue(u.math.array_equal(branch.radii_prox, np.array([1.0]) * u.um))
        self.assertTrue(u.math.array_equal(branch.radii_dist, np.array([0.5]) * u.um))

    def test_branch_normalizes_mixed_inputs(self) -> None:
        branch = Branch(
            lengths=np.array([0.02]) * u.mm,
            radii_prox=[1000.0],
            radii_dist=[0.001] * u.mm,
            type="soma",
        )

        self.assertAlmostEqual(branch.total_length.to_decimal(u.um), 20.0)
        self.assertTrue(u.math.array_equal(branch.radii_prox, np.array([1000.0]) * u.um))
        self.assertTrue(u.math.array_equal(branch.radii_dist, np.array([1.0]) * u.um))

    def test_branch_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0], radii_prox=[1.0, 2.0], radii_dist=[1.0], type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[0.0], radii_prox=[1.0], radii_dist=[1.0], type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0], radii_prox=[-1.0], radii_dist=[1.0], type="axon")
        with self.assertRaises(ValueError):
            Branch.lengths_shared(lengths=10.0 * u.um, radii=1.0 * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch.lengths_paired(lengths=10.0 * u.um, radii_pairs=1.0 * u.um, type="axon")
        with self.assertRaises(ValueError):
            Branch(lengths=[10.0], radii_prox=[1.0], radii_dist=[1.0], type="unknown_branch")

    def test_lengths_constructors_cover_shared_and_paired_radii(self) -> None:
        shared = Branch.lengths_shared(
            lengths=[10.0, 20.0],
            radii=[3.0, 2.0, 1.0],
            type="apical_dendrite",
        )
        self.assertTrue(u.math.array_equal(shared.lengths, np.array([10.0, 20.0]) * u.um))
        self.assertTrue(u.math.array_equal(shared.radii_prox, np.array([3.0, 2.0]) * u.um))
        self.assertTrue(u.math.array_equal(shared.radii_dist, np.array([2.0, 1.0]) * u.um))
        self.assertEqual(shared.type, "apical_dendrite")

        paired = Branch.lengths_paired(
            lengths=np.array([10.0, 20.0]) * u.um,
            radii_pairs=[(3.0, 2.5), (2.0, 1.0)],
            type="axon",
        )
        self.assertTrue(u.math.array_equal(paired.radii_prox, np.array([3.0, 2.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired.radii_dist, np.array([2.5, 1.0]) * u.um))

        shared_scalar = Branch.lengths_shared(
            lengths=10.0 * u.um,
            radii=[3.0, 2.0],
            type="axon",
        )
        paired_scalar = Branch.lengths_paired(
            lengths=10.0 * u.um,
            radii_pairs=[(3.0, 2.0)],
            type="axon",
        )
        self.assertTrue(u.math.array_equal(shared_scalar.lengths, np.array([10.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired_scalar.lengths, np.array([10.0]) * u.um))

    def test_xyz_constructors_cover_shared_and_paired_radii(self) -> None:
        shared = Branch.xyz_shared(
            points=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]) * u.um,
            radii=[2.0, 1.5, 1.0],
            type="soma",
        )
        self.assertTrue(u.math.array_equal(shared.lengths, np.array([5.0, 12.0]) * u.um))
        self.assertTrue(
            u.math.array_equal(
                shared.proximal_points,
                np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]) * u.um,
            )
        )
        self.assertTrue(
            u.math.array_equal(
                shared.distal_points,
                np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]) * u.um,
            )
        )

        paired = Branch.xyz_paired(
            points=[(0.0, 0.0, 0.0), (0.0, 6.0, 8.0)],
            radii_pairs=((1.0 * u.um, 0.5 * u.um),),
            type="basal_dendrite",
        )
        self.assertTrue(u.math.array_equal(paired.lengths, np.array([10.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired.radii_prox, np.array([1.0]) * u.um))
        self.assertTrue(u.math.array_equal(paired.radii_dist, np.array([0.5]) * u.um))
        self.assertFalse(hasattr(shared, "x"))
        self.assertFalse(hasattr(shared, "y"))
        self.assertFalse(hasattr(shared, "z"))

    def test_branch_accepts_jax_quantity_inputs(self) -> None:
        if jnp is None:
            self.skipTest("jax is not installed")

        branch = Branch.lengths_shared(
            lengths=jnp.array([40.0]) * u.um,
            radii=jnp.array([0.8, 0.4]) * u.um,
            type="axon",
        )
        self.assertTrue(u.math.array_equal(branch.lengths, np.array([40.0]) * u.um))
