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

"""Tests for :mod:`braincell._discretization.geometry`."""

import unittest

import brainunit as u
import numpy as np

from braincell._discretization._testing import (
    build_geo,
    make_branch,
    make_single_branch_morpho,
)
from braincell._discretization.geometry import (
    _Frustum,
    _GeoCV,
    _axial_factor_per_cm,
    _boundary_radii_um,
    _build_frusta,
    _lateral_area_um2,
    _midpoint_radius_um,
    _split_frusta,
    locate_cv_on_branch as _locate_cv_on_branch,
    validate_bounds as _validate_bounds,
    validate_connectivity as _validate_connectivity,
    validate_morphology as _validate_morpho,
)
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology


def _jump_branch() -> Branch:
    return Branch.from_lengths(
        lengths=np.asarray([10.0, 20.0]) * u.um,
        radii_proximal=np.asarray([2.0, 4.0]) * u.um,
        radii_distal=np.asarray([1.0, 2.0]) * u.um,
        type="dendrite",
    )


# =============================================================================
# Frustum math
# =============================================================================


class BuildFrustaTest(unittest.TestCase):
    def test_full_branch_single_segment(self) -> None:
        branch = make_branch([10.0], [2.0, 3.0])
        frusta = _build_frusta(branch, prox=0.0, dist=1.0)
        self.assertEqual(len(frusta), 1)
        self.assertAlmostEqual(frusta[0].length_um, 10.0)
        self.assertAlmostEqual(frusta[0].r_prox_um, 2.0)
        self.assertAlmostEqual(frusta[0].r_dist_um, 3.0)

    def test_half_branch_clips_length_and_interpolates_radius(self) -> None:
        branch = make_branch([10.0], [2.0, 4.0])
        frusta = _build_frusta(branch, prox=0.0, dist=0.5)
        self.assertEqual(len(frusta), 1)
        self.assertAlmostEqual(frusta[0].length_um, 5.0)
        self.assertAlmostEqual(frusta[0].r_prox_um, 2.0)
        self.assertAlmostEqual(frusta[0].r_dist_um, 3.0)

    def test_multi_segment_branch(self) -> None:
        branch = make_branch([4.0, 6.0], [1.0, 2.0, 3.0])
        frusta = _build_frusta(branch, prox=0.0, dist=1.0)
        self.assertEqual(len(frusta), 2)
        self.assertAlmostEqual(frusta[0].length_um, 4.0)
        self.assertAlmostEqual(frusta[1].length_um, 6.0)

    def test_preserves_zero_length_jump_segment_for_radius_discontinuity(self) -> None:
        branch = _jump_branch()
        frusta = _build_frusta(branch, prox=0.0, dist=1.0)
        self.assertEqual(len(frusta), 3)
        self.assertAlmostEqual(frusta[0].length_um, 10.0)
        self.assertAlmostEqual(frusta[1].length_um, 0.0)
        self.assertAlmostEqual(frusta[2].length_um, 20.0)
        self.assertAlmostEqual(frusta[1].r_prox_um, 1.0)
        self.assertAlmostEqual(frusta[1].r_dist_um, 4.0)

    def test_jump_branch_frusta_preserve_area_without_changing_axial_factor(self) -> None:
        branch = _jump_branch()
        frusta = _build_frusta(branch, prox=0.0, dist=1.0)

        branch_area = float(branch.area.to_decimal(u.um**2))
        self.assertAlmostEqual(_lateral_area_um2(frusta), branch_area, places=4)

        lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
        radii_prox_um = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
        radii_dist_um = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
        positive_mask = lengths_um > 1e-12
        expected_axial_factor = float(
            np.sum(
                (lengths_um[positive_mask] * 1e-4)
                / (np.pi * (radii_prox_um[positive_mask] * 1e-4) * (radii_dist_um[positive_mask] * 1e-4))
            )
        )
        self.assertAlmostEqual(_axial_factor_per_cm(frusta), expected_axial_factor, places=9)

    def test_shared_boundary_counts_jump_once(self) -> None:
        branch = _jump_branch()
        split_x = 10.0 / 30.0
        left = _build_frusta(branch, prox=0.0, dist=split_x)
        right = _build_frusta(branch, prox=split_x, dist=1.0)

        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 1)
        self.assertAlmostEqual(left[-1].length_um, 0.0)
        self.assertAlmostEqual(right[0].length_um, 20.0)
        self.assertAlmostEqual(
            _lateral_area_um2(left) + _lateral_area_um2(right),
            float(branch.area.to_decimal(u.um**2)),
            places=4,
        )

    def test_rejects_reversed_bounds(self) -> None:
        branch = make_branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.6, dist=0.4)

    def test_rejects_equal_bounds(self) -> None:
        branch = make_branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.5, dist=0.5)

    def test_rejects_out_of_range(self) -> None:
        branch = make_branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=-0.1, dist=0.5)
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.5, dist=1.1)

    def test_rejects_nonpositive_radius(self) -> None:
        # ``Branch.from_lengths`` would reject zero radii earlier. Test that
        # ``_build_frusta`` also raises when handed a hand-assembled branch
        # with a non-positive radius — defense in depth.
        import brainstate

        brainstate.environ.set(precision=64)
        branch = Branch.from_lengths(
            lengths=np.asarray([10.0]) * u.um,
            radii_proximal=np.asarray([1e-20]) * u.um,
            radii_distal=np.asarray([1e-20]) * u.um,
            type="dendrite",
        )
        # Now spoof the radii via object.__setattr__ to simulate a corrupt
        # upstream branch — validator should still reject.
        with self.assertRaises(ValueError):
            object.__setattr__(
                branch,
                "radii_proximal",
                np.asarray([0.0]) * u.um,
            )
            _build_frusta(branch, prox=0.0, dist=1.0)


class FrustumScalarsTest(unittest.TestCase):
    def _single(self, length_um: float, r0: float, r1: float) -> tuple[_Frustum, ...]:
        return (
            _Frustum(
                prox=0.0,
                dist=1.0,
                length_um=length_um,
                r_prox_um=r0,
                r_dist_um=r1,
                point_prox_um=None,
                point_dist_um=None,
            ),
        )

    def test_lateral_area_cylinder(self) -> None:
        frusta = self._single(10.0, 2.0, 2.0)
        # cylinder: π·(r0+r1)·slant, slant=L because dr=0
        area = _lateral_area_um2(frusta)
        self.assertAlmostEqual(area, 40.0 * np.pi, places=6)

    def test_axial_factor_uniform(self) -> None:
        frusta = self._single(10.0, 2.0, 2.0)  # 10 μm = 1e-3 cm
        expected = 1e-3 / (np.pi * 2e-4 * 2e-4)
        self.assertAlmostEqual(_axial_factor_per_cm(frusta), expected, places=4)

    def test_midpoint_radius_uniform(self) -> None:
        frusta = self._single(10.0, 2.0, 2.0)
        self.assertAlmostEqual(_midpoint_radius_um(frusta), 2.0)

    def test_midpoint_radius_tapered(self) -> None:
        frusta = self._single(10.0, 2.0, 4.0)
        self.assertAlmostEqual(_midpoint_radius_um(frusta), 3.0)

    def test_boundary_radii(self) -> None:
        frusta = self._single(10.0, 2.0, 4.0)
        r0, r1 = _boundary_radii_um(frusta)
        self.assertAlmostEqual(r0, 2.0)
        self.assertAlmostEqual(r1, 4.0)

    def test_empty_frusta_raises(self) -> None:
        with self.assertRaises(ValueError):
            _boundary_radii_um(())
        with self.assertRaises(ValueError):
            _midpoint_radius_um(())


class SplitFrustaTest(unittest.TestCase):
    def test_split_at_midpoint(self) -> None:
        f = _Frustum(
            prox=0.0,
            dist=1.0,
            length_um=10.0,
            r_prox_um=2.0,
            r_dist_um=4.0,
            point_prox_um=None,
            point_dist_um=None,
        )
        left, right = _split_frusta((f,), x=0.5)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertAlmostEqual(left[0].length_um, 5.0)
        self.assertAlmostEqual(right[0].length_um, 5.0)
        self.assertAlmostEqual(left[0].r_dist_um, 3.0)
        self.assertAlmostEqual(right[0].r_prox_um, 3.0)

    def test_split_at_boundary_puts_all_on_one_side(self) -> None:
        f = _Frustum(
            prox=0.0,
            dist=1.0,
            length_um=10.0,
            r_prox_um=2.0,
            r_dist_um=4.0,
            point_prox_um=None,
            point_dist_um=None,
        )
        left, right = _split_frusta((f,), x=1.0)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 0)

# =============================================================================
# Geometry build
# =============================================================================


class BuildGeoTest(unittest.TestCase):
    def test_single_branch_one_cv(self) -> None:
        morpho = make_single_branch_morpho()
        geos, ids = build_geo(morpho, (((0.0, 1.0),),))
        self.assertEqual(len(geos), 1)
        self.assertEqual(ids, ((0,),))
        g = geos[0]
        self.assertEqual(g.id, 0)
        self.assertEqual(g.branch_id, 0)
        self.assertEqual(g.branch_type, "soma")
        self.assertIsNone(g.parent_cv)
        self.assertEqual(g.children_cv, ())
        self.assertAlmostEqual(g.length_um, 10.0)

    def test_single_branch_two_cvs_chain(self) -> None:
        morpho = make_single_branch_morpho()
        geos, ids = build_geo(morpho, (((0.0, 0.5), (0.5, 1.0)),))
        self.assertEqual(len(geos), 2)
        self.assertIsNone(geos[0].parent_cv)
        self.assertEqual(geos[0].children_cv, (1,))
        self.assertEqual(geos[1].parent_cv, 0)
        self.assertEqual(geos[1].children_cv, ())

    def test_two_branch_parent_pointer(self) -> None:
        soma = make_branch([10.0], [3.0, 3.0], type="soma")
        dend = make_branch([10.0], [2.0, 1.0], type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.d = dend
        geos, _ = build_geo(tree, (((0.0, 0.5), (0.5, 1.0)), ((0.0, 1.0),)))
        self.assertEqual(len(geos), 3)
        self.assertEqual(geos[2].parent_cv, 1)
        self.assertIn(2, geos[1].children_cv)


class LocateCVOnBranchTest(unittest.TestCase):
    def test_raises_on_bad_bounds(self) -> None:
        g = _GeoCV(
            id=0,
            branch_id=0,
            branch_type="soma",
            prox=0.2,
            dist=0.8,
            midpoint=0.5,
            parent_cv=None,
            children_cv=(),
            length_um=6.0,
            lateral_area_um2=1.0,
            axial_factor_total_per_cm=1.0,
            axial_factor_prox_per_cm=0.5,
            axial_factor_dist_per_cm=0.5,
            r_prox_um=1.0,
            r_mid_um=1.0,
            diam_arc_mean_um=2.0,
            r_dist_um=1.0,
        )
        # x=0.9 and x=0.1 are out of [0.2, 0.8] — raise, not snap.
        with self.assertRaises(ValueError):
            _locate_cv_on_branch((0,), (g,), x=0.1)

# =============================================================================
# Validators
# =============================================================================


class ValidateMorphoTest(unittest.TestCase):
    def test_zero_length_branch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_morpho(
                Morphology.from_root(
                    make_branch([1e-12], [2.0, 2.0], type="soma"),
                    name="soma",
                )
            )

    def test_nonpositive_radius_raises(self) -> None:
        # Branch.from_lengths itself rejects zero radii upstream, so to
        # exercise ``_validate_morpho`` we spoof a branch with a corrupted
        # radius field after construction.
        soma = make_branch([10.0], [2.0, 2.0], type="soma")
        object.__setattr__(
            soma,
            "radii_proximal",
            np.asarray([0.0]) * u.um,
        )
        with self.assertRaises(ValueError):
            _validate_morpho(Morphology.from_root(soma, name="soma"))

    def test_valid_morpho_ok(self) -> None:
        _validate_morpho(make_single_branch_morpho())


class ValidateBoundsTest(unittest.TestCase):
    def test_valid_bounds(self) -> None:
        _validate_bounds((((0.0, 0.5), (0.5, 1.0)),), make_single_branch_morpho())

    def test_gap_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.4), (0.6, 1.0)),), make_single_branch_morpho())

    def test_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.6), (0.5, 1.0)),), make_single_branch_morpho())

    def test_missing_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.1, 1.0),),), make_single_branch_morpho())

    def test_missing_end_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.9),),), make_single_branch_morpho())

    def test_empty_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds(((),), make_single_branch_morpho())

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds(
                (((0.0, 1.0),), ((0.0, 1.0),)),
                make_single_branch_morpho(),
            )


class ValidateConnectivityTest(unittest.TestCase):
    def test_valid_connectivity_ok(self) -> None:
        morpho = make_single_branch_morpho()
        geos, ids = build_geo(morpho, (((0.0, 1.0),),))
        _validate_connectivity(geos, ids, morpho)

if __name__ == "__main__":
    unittest.main()
