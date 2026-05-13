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

import brainunit as u
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from braincell._discretization.base import CV, build_discretization
from braincell._discretization.node_build import (
    _EPS_PARAM,
    _locate_branch_cv_by_x,
    build_node_tree_from_cvs as build_node_tree,
    locate_node_on_branch,
)
from braincell._discretization.geometry import (
    CVGeometryResult,
    _Frustum,
    _GeoCV,
    _axial_factor_per_cm,
    _boundary_radii_um,
    _build_frusta,
    _lateral_area_um2,
    _midpoint_radius_um,
    _split_frusta,
    build_cv_geometry,
    locate_cv_on_branch as _locate_cv_on_branch,
    validate_bounds as _validate_bounds,
    validate_connectivity as _validate_connectivity,
    validate_morphology as _validate_morpho,
)
from braincell._discretization.mechanism import (
    PaintRule,
    PlaceRule,
    _DEFAULT_CABLE,
    _MechBucket,
    _RegionCache,
    _apply_density,
    _apply_place,
    _coverage_fraction,
    _resolve_point_name,
    build_cv_mechanisms,
    default_paint_rules,
    merge_paint_rules,
    merge_place_rules,
    normalize_paint_rules,
    normalize_place_rule,
)
from braincell._discretization.policy import CVPerBranch, CVPolicy
from braincell.filter import (
    AllRegion,
    AtLocation,
    BranchSlice,
)
from braincell.mech import (
    CableProperty,
    Channel,
    CurrentClamp,
    Ion,
    StateProbe,
)
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology


def _cable(cm: float = 1.0, ra: float = 100.0, v: float = -65.0) -> CableProperty:
    return CableProperty(
        resting_potential=v * u.mV,
        membrane_capacitance=cm * (u.uF / u.cm ** 2),
        axial_resistivity=ra * (u.ohm * u.cm),
    )


def _branch(lengths: list[float], radii: list[float], type: str = "dendrite") -> Branch:
    return Branch.from_lengths(
        lengths=np.asarray(lengths) * u.um,
        radii=np.asarray(radii) * u.um,
        type=type,
    )


def _jump_branch() -> Branch:
    return Branch.from_lengths(
        lengths=np.asarray([10.0, 20.0]) * u.um,
        radii_proximal=np.asarray([2.0, 4.0]) * u.um,
        radii_distal=np.asarray([1.0, 2.0]) * u.um,
        type="dendrite",
    )


def _single_branch_morpho(type: str = "soma") -> Morphology:
    return Morphology.from_root(
        _branch([10.0], [2.0, 2.0], type=type),
        name=type,
    )


def _build_cvs(morpho, *, policy, paint_rules, place_rules):
    return build_discretization(
        morpho,
        policy=policy,
        paint_rules=paint_rules,
        place_rules=place_rules,
    ).cvs


def _build_geo(morpho, bounds):
    geometry = build_cv_geometry(morpho, bounds)
    return geometry.geos, geometry.branch_to_cv_ids


def _build_mech(
    morpho,
    geos,
    branch_to_cv_ids,
    *,
    paint_rules,
    place_rules,
    cache=None,
):
    del cache
    geometry = CVGeometryResult(
        geos=tuple(geos),
        branch_to_cv_ids=tuple(branch_to_cv_ids),
    )
    return build_cv_mechanisms(
        morpho,
        geometry,
        paint_rules=paint_rules,
        place_rules=place_rules,
    )


# =============================================================================
# Rule dataclasses + normalize / merge
# =============================================================================


class PaintAndPlaceRuleTest(unittest.TestCase):
    def test_paint_rule_is_frozen_and_equal_by_value(self) -> None:
        c = _cable()
        r1 = PaintRule(region=AllRegion(), mechanism=c)
        r2 = PaintRule(region=AllRegion(), mechanism=c)
        self.assertEqual(r1, r2)
        with self.assertRaises(Exception):
            r1.region = AllRegion()  # type: ignore[misc]

    def test_place_rule_default_site_is_mid(self) -> None:
        rule = PlaceRule(
            locset=AtLocation(branch=0, x=0.5),
            mechanisms=(CurrentClamp.step(0.2 * u.nA, 10 * u.ms),),
        )
        self.assertEqual(rule.site, "mid")


class NormalizePaintRulesTest(unittest.TestCase):
    def test_default_has_one_cable_on_all_region(self) -> None:
        rules = default_paint_rules()
        self.assertEqual(len(rules), 1)
        self.assertIsInstance(rules[0].region, AllRegion)
        self.assertIsInstance(rules[0].mechanism, CableProperty)

    def test_rejects_non_region_expr(self) -> None:
        with self.assertRaises(TypeError):
            normalize_paint_rules(
                "not a region",  # type: ignore[arg-type]
                (Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV),),
            )

    def test_rejects_empty_mechanisms(self) -> None:
        with self.assertRaises(ValueError):
            normalize_paint_rules(AllRegion(), ())

    def test_accepts_cable_and_density(self) -> None:
        cable = _cable()
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        rules = normalize_paint_rules(AllRegion(), (cable, ch))
        self.assertEqual(len(rules), 2)
        self.assertIs(rules[0].mechanism, cable)
        self.assertIs(rules[1].mechanism, ch)

    def test_rejects_point_mechanism(self) -> None:
        with self.assertRaises(TypeError):
            normalize_paint_rules(
                AllRegion(),
                (CurrentClamp.step(0.1 * u.nA, 10 * u.ms),),
            )

    def test_accepts_ion(self) -> None:
        rules = normalize_paint_rules(AllRegion(), (Ion("SodiumFixed"),))
        self.assertEqual(len(rules), 1)
        self.assertIsInstance(rules[0].mechanism, Ion)


class NormalizePlaceRuleTest(unittest.TestCase):
    def test_rejects_non_locset(self) -> None:
        with self.assertRaises(TypeError):
            normalize_place_rule(
                "not a locset",  # type: ignore[arg-type]
                (CurrentClamp.step(0.1 * u.nA, 10 * u.ms),),
            )

    def test_rejects_empty_mechanisms(self) -> None:
        with self.assertRaises(ValueError):
            normalize_place_rule(AtLocation(branch=0, x=0.5), ())

    def test_rejects_non_point_mechanism(self) -> None:
        with self.assertRaises(TypeError):
            normalize_place_rule(
                AtLocation(branch=0, x=0.5),
                (Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV),),
            )

    def test_returns_place_rule_with_site_mid(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        rule = normalize_place_rule(AtLocation(branch=0, x=0.5), (clamp,))
        self.assertEqual(rule.site, "mid")
        self.assertEqual(rule.mechanisms, (clamp,))


class MergePaintRulesTest(unittest.TestCase):
    def test_cable_same_region_replaces(self) -> None:
        r1 = PaintRule(region=AllRegion(), mechanism=_cable(cm=1.0))
        r2 = PaintRule(region=AllRegion(), mechanism=_cable(cm=2.0))
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0].mechanism, r2.mechanism)

    def test_cable_different_regions_kept(self) -> None:
        r1 = PaintRule(region=AllRegion(), mechanism=_cable(cm=1.0))
        r2 = PaintRule(
            region=BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            mechanism=_cable(cm=2.0),
        )
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)

    def test_density_same_region_same_name_replaces(self) -> None:
        d1 = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        d2 = Channel("IL", g_max=0.2 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0].mechanism, d2)

    def test_density_same_class_different_names_kept(self) -> None:
        d1 = Channel("IL", name="a", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        d2 = Channel("IL", name="b", g_max=0.2 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)

    def test_density_different_classes_both_kept(self) -> None:
        d1 = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        d2 = Channel(
            "Na_Ba2002", g_max=0.05 * (u.mS / u.cm ** 2), E=50 * u.mV
        )
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)


class MergePlaceRulesTest(unittest.TestCase):
    def test_exact_duplicate_dropped(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        r = normalize_place_rule(AtLocation(branch=0, x=0.5), (clamp,))
        merged = merge_place_rules((r,), (r,))
        self.assertEqual(len(merged), 1)

    def test_different_clamps_both_kept(self) -> None:
        c1 = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        c2 = CurrentClamp.step(0.2 * u.nA, 10 * u.ms)
        r1 = normalize_place_rule(AtLocation(branch=0, x=0.5), (c1,))
        r2 = normalize_place_rule(AtLocation(branch=0, x=0.5), (c2,))
        merged = merge_place_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)


# =============================================================================
# Region cache
# =============================================================================


class RegionCacheTest(unittest.TestCase):
    def test_intervals_returns_same_object_twice(self) -> None:
        morpho = _single_branch_morpho()
        region = AllRegion()
        cache = _RegionCache(morpho)
        a = cache.intervals(region)
        b = cache.intervals(region)
        self.assertIs(a, b)
        self.assertEqual(a, {0: ((0.0, 1.0),)})

    def test_points_cached(self) -> None:
        morpho = _single_branch_morpho()
        locset = AtLocation(branch=0, x=0.5)
        cache = _RegionCache(morpho)
        a = cache.points(locset)
        b = cache.points(locset)
        self.assertIs(a, b)
        self.assertEqual(a, ((0, 0.5, "soma(0.5)"),))


# =============================================================================
# Frustum math
# =============================================================================


class BuildFrustaTest(unittest.TestCase):
    def test_full_branch_single_segment(self) -> None:
        branch = _branch([10.0], [2.0, 3.0])
        frusta = _build_frusta(branch, prox=0.0, dist=1.0)
        self.assertEqual(len(frusta), 1)
        self.assertAlmostEqual(frusta[0].length_um, 10.0)
        self.assertAlmostEqual(frusta[0].r_prox_um, 2.0)
        self.assertAlmostEqual(frusta[0].r_dist_um, 3.0)

    def test_half_branch_clips_length_and_interpolates_radius(self) -> None:
        branch = _branch([10.0], [2.0, 4.0])
        frusta = _build_frusta(branch, prox=0.0, dist=0.5)
        self.assertEqual(len(frusta), 1)
        self.assertAlmostEqual(frusta[0].length_um, 5.0)
        self.assertAlmostEqual(frusta[0].r_prox_um, 2.0)
        self.assertAlmostEqual(frusta[0].r_dist_um, 3.0)

    def test_multi_segment_branch(self) -> None:
        branch = _branch([4.0, 6.0], [1.0, 2.0, 3.0])
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

        branch_area = float(branch.area.to_decimal(u.um ** 2))
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
            float(branch.area.to_decimal(u.um ** 2)),
            places=4,
        )

    def test_rejects_reversed_bounds(self) -> None:
        branch = _branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.6, dist=0.4)

    def test_rejects_equal_bounds(self) -> None:
        branch = _branch([10.0], [2.0, 3.0])
        with self.assertRaises(ValueError):
            _build_frusta(branch, prox=0.5, dist=0.5)

    def test_rejects_out_of_range(self) -> None:
        branch = _branch([10.0], [2.0, 3.0])
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
                branch, "radii_proximal",
                np.asarray([0.0]) * u.um,
            )
            _build_frusta(branch, prox=0.0, dist=1.0)


class FrustumScalarsTest(unittest.TestCase):
    def _single(self, length_um: float, r0: float, r1: float) -> tuple[_Frustum, ...]:
        return (
            _Frustum(
                prox=0.0, dist=1.0, length_um=length_um,
                r_prox_um=r0, r_dist_um=r1,
                point_prox_um=None, point_dist_um=None,
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
            prox=0.0, dist=1.0, length_um=10.0,
            r_prox_um=2.0, r_dist_um=4.0,
            point_prox_um=None, point_dist_um=None,
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
            prox=0.0, dist=1.0, length_um=10.0,
            r_prox_um=2.0, r_dist_um=4.0,
            point_prox_um=None, point_dist_um=None,
        )
        left, right = _split_frusta((f,), x=1.0)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 0)


# =============================================================================
# Geometry build
# =============================================================================


class BuildGeoTest(unittest.TestCase):
    def test_single_branch_one_cv(self) -> None:
        morpho = _single_branch_morpho()
        geos, ids = _build_geo(morpho, (((0.0, 1.0),),))
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
        morpho = _single_branch_morpho()
        geos, ids = _build_geo(morpho, (((0.0, 0.5), (0.5, 1.0)),))
        self.assertEqual(len(geos), 2)
        self.assertIsNone(geos[0].parent_cv)
        self.assertEqual(geos[0].children_cv, (1,))
        self.assertEqual(geos[1].parent_cv, 0)
        self.assertEqual(geos[1].children_cv, ())

    def test_two_branch_parent_pointer(self) -> None:
        soma = _branch([10.0], [3.0, 3.0], type="soma")
        dend = _branch([10.0], [2.0, 1.0], type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.d = dend
        geos, _ = _build_geo(tree, (((0.0, 0.5), (0.5, 1.0)), ((0.0, 1.0),)))
        self.assertEqual(len(geos), 3)
        self.assertEqual(geos[2].parent_cv, 1)
        self.assertIn(2, geos[1].children_cv)


class LocateCVOnBranchTest(unittest.TestCase):
    def test_raises_on_bad_bounds(self) -> None:
        g = _GeoCV(
            id=0, branch_id=0, branch_type="soma",
            prox=0.2, dist=0.8, midpoint=0.5,
            parent_cv=None, children_cv=(),
            length_um=6.0, lateral_area_um2=1.0,
            axial_factor_total_per_cm=1.0,
            axial_factor_prox_per_cm=0.5, axial_factor_dist_per_cm=0.5,
            r_prox_um=1.0, r_mid_um=1.0, r_dist_um=1.0,
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
                    _branch([1e-12], [2.0, 2.0], type="soma"),
                    name="soma",
                )
            )

    def test_nonpositive_radius_raises(self) -> None:
        # Branch.from_lengths itself rejects zero radii upstream, so to
        # exercise ``_validate_morpho`` we spoof a branch with a corrupted
        # radius field after construction.
        soma = _branch([10.0], [2.0, 2.0], type="soma")
        object.__setattr__(
            soma, "radii_proximal", np.asarray([0.0]) * u.um,
        )
        with self.assertRaises(ValueError):
            _validate_morpho(Morphology.from_root(soma, name="soma"))

    def test_valid_morpho_ok(self) -> None:
        _validate_morpho(_single_branch_morpho())


class ValidateBoundsTest(unittest.TestCase):
    def test_valid_bounds(self) -> None:
        _validate_bounds((((0.0, 0.5), (0.5, 1.0)),), _single_branch_morpho())

    def test_gap_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.4), (0.6, 1.0)),), _single_branch_morpho())

    def test_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.6), (0.5, 1.0)),), _single_branch_morpho())

    def test_missing_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.1, 1.0),),), _single_branch_morpho())

    def test_missing_end_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds((((0.0, 0.9),),), _single_branch_morpho())

    def test_empty_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds(((),), _single_branch_morpho())

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_bounds(
                (((0.0, 1.0),), ((0.0, 1.0),)),
                _single_branch_morpho(),
            )


class ValidateConnectivityTest(unittest.TestCase):
    def test_valid_connectivity_ok(self) -> None:
        morpho = _single_branch_morpho()
        geos, ids = _build_geo(morpho, (((0.0, 1.0),),))
        _validate_connectivity(geos, ids, morpho)


# =============================================================================
# Mechanism lowering
# =============================================================================


class CoverageFractionTest(unittest.TestCase):
    def _geo(self) -> tuple[Morphology, _GeoCV]:
        morpho = _single_branch_morpho()
        geos, _ = _build_geo(morpho, (((0.0, 1.0),),))
        return morpho, geos[0]

    def test_full_overlap_fraction_one(self) -> None:
        morpho, geo = self._geo()
        self.assertAlmostEqual(_coverage_fraction(morpho, geo, ((0.0, 1.0),)), 1.0)

    def test_half_overlap(self) -> None:
        morpho, geo = self._geo()
        self.assertAlmostEqual(
            _coverage_fraction(morpho, geo, ((0.0, 0.5),)),
            0.5,
            places=3,
        )

    def test_zero_overlap(self) -> None:
        morpho = _single_branch_morpho()
        # Using a CV that spans [0, 0.5]; an interval at [0.6, 1.0] should
        # produce 0. Geometry build now validates full branch coverage, so
        # construct the partial geo manually.
        g = _GeoCV(
            id=0, branch_id=0, branch_type="soma",
            prox=0.0, dist=0.5, midpoint=0.25,
            parent_cv=None, children_cv=(),
            length_um=5.0, lateral_area_um2=10.0,
            axial_factor_total_per_cm=1.0,
            axial_factor_prox_per_cm=0.5, axial_factor_dist_per_cm=0.5,
            r_prox_um=1.0, r_mid_um=1.0, r_dist_um=1.0,
        )
        self.assertAlmostEqual(
            _coverage_fraction(morpho, g, ((0.6, 1.0),)),
            0.0,
        )


class ApplyDensityTest(unittest.TestCase):
    def test_channel_full_coverage_no_scaling(self) -> None:
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        _apply_density(bucket, ch, region_key=AllRegion(), fraction=1.0)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.coverage_area_fraction, 1.0)

    def test_channel_half_coverage_records_fraction(self) -> None:
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        _apply_density(bucket, ch, region_key=AllRegion(), fraction=0.5)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.coverage_area_fraction, 0.5)

    def test_ion_ignores_coverage(self) -> None:
        ion = Ion("SodiumFixed")
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        _apply_density(bucket, ion, region_key=AllRegion(), fraction=0.5)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.coverage_area_fraction, 1.0)

    def test_same_key_replaces(self) -> None:
        c1 = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        c2 = Channel("IL", g_max=0.2 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        region = AllRegion()
        _apply_density(bucket, c1, region_key=region, fraction=1.0)
        _apply_density(bucket, c2, region_key=region, fraction=1.0)
        self.assertEqual(len(bucket.density_by_key), 1)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.params["g_max"], 0.2 * (u.mS / u.cm ** 2))


class ResolvePointNameTest(unittest.TestCase):
    def test_state_probe_auto_name(self) -> None:
        probe = StateProbe(field="v")
        named = _resolve_point_name(probe, display_name="loc_0")
        self.assertEqual(named.name, "loc_0_v")

    def test_state_probe_keeps_explicit_name(self) -> None:
        probe = StateProbe(field="v", name="my_probe")
        named = _resolve_point_name(probe, display_name="loc_0")
        self.assertEqual(named.name, "my_probe")

    def test_clamp_untouched_when_no_auto_name(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        named = _resolve_point_name(clamp, display_name="loc_0")
        self.assertIs(named, clamp)


class ApplyPlaceTest(unittest.TestCase):
    def test_auto_generated_duplicate_raises(self) -> None:
        seen: set[str] = set()
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        probe = StateProbe(field="v")
        _apply_place(bucket, probe, display_name="loc_0", seen_names=seen)
        with self.assertRaises(ValueError):
            _apply_place(bucket, probe, display_name="loc_0", seen_names=seen)

    def test_user_named_duplicate_allowed(self) -> None:
        seen: set[str] = set()
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        a = StateProbe(field="v", name="dup")
        b = StateProbe(field="v", name="dup")
        _apply_place(bucket, a, display_name="loc_0", seen_names=seen)
        _apply_place(bucket, b, display_name="loc_1", seen_names=seen)
        self.assertEqual(len(bucket.points), 2)


class BuildMechTest(unittest.TestCase):
    def test_paint_cable_and_channel_on_all_region(self) -> None:
        morpho = _single_branch_morpho()
        geos, ids = _build_geo(morpho, (((0.0, 1.0),),))
        cable = _cable(cm=2.0)
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        paint = (
            PaintRule(region=AllRegion(), mechanism=cable),
            PaintRule(region=AllRegion(), mechanism=ch),
        )
        cache = _RegionCache(morpho)
        buckets = _build_mech(
            morpho, geos, ids,
            paint_rules=paint, place_rules=(), cache=cache,
        )
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0].cable.membrane_capacitance, 2.0 * (u.uF / u.cm ** 2))
        self.assertEqual(len(buckets[0].density_by_key), 1)

    def test_place_clamp_attaches_to_one_cv(self) -> None:
        morpho = _single_branch_morpho()
        geos, ids = _build_geo(morpho, (((0.0, 0.5), (0.5, 1.0)),))
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms)
        place = (
            PlaceRule(
                locset=AtLocation(branch=0, x=0.25),
                mechanisms=(clamp,),
            ),
        )
        cache = _RegionCache(morpho)
        buckets = _build_mech(
            morpho, geos, ids,
            paint_rules=(), place_rules=place, cache=cache,
        )
        self.assertEqual([len(b.points) for b in buckets], [1, 0])


class BuildMechCachesFrustaTest(unittest.TestCase):
    """MED-03: frusta for (branch, prox, dist) must be computed at most once per _build_mech call."""

    def test_overlapping_rules_reuse_frusta(self) -> None:
        from unittest.mock import patch

        morpho = _single_branch_morpho()
        # Four CVs so each rule visits four distinct (prox, dist) pairs.
        geos, ids = _build_geo(
            morpho,
            (((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),),
        )
        ch1 = Channel("IL", g_max=0.1 * (u.mS / u.cm ** 2), E=-70 * u.mV)
        ch2 = Channel("IL", g_max=0.2 * (u.mS / u.cm ** 2), E=-60 * u.mV)
        paint = (
            PaintRule(region=AllRegion(), mechanism=ch1),
            PaintRule(region=AllRegion(), mechanism=ch2),
        )
        cache = _RegionCache(morpho)

        calls: dict = {}
        original = _build_frusta

        def counting(branch, *, prox, dist):
            key = (id(branch), round(float(prox), 9), round(float(dist), 9))
            calls[key] = calls.get(key, 0) + 1
            return original(branch, prox=prox, dist=dist)

        with patch("braincell._discretization.mechanism._build_frusta", new=counting):
            _build_mech(
                morpho, geos, ids,
                paint_rules=paint, place_rules=(), cache=cache,
            )

        for key, count in calls.items():
            self.assertEqual(
                count, 1,
                f"_build_frusta was called {count} times for key={key!r}; "
                "expected 1 after caching.",
            )


class LowerSmokeTest(unittest.TestCase):
    def test_single_branch_default_cable(self) -> None:
        morpho = _single_branch_morpho()
        cvs = _build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        self.assertEqual(len(cvs), 2)
        self.assertIsInstance(cvs[0], CV)
        self.assertEqual(cvs[0].id, 0)
        self.assertEqual(cvs[0].branch_id, 0)
        self.assertEqual(cvs[1].parent_cv, 0)
        self.assertAlmostEqual(float(cvs[0].length.to_decimal(u.um)), 5.0)

    def test_rejects_invalid_policy_bounds(self) -> None:
        morpho = _single_branch_morpho()

        class BadPolicy(CVPolicy):
            def resolve_cv_bounds(self, morpho, *, paint_rules=None):
                return (((0.0, 0.5),),)  # missing 0.5..1.0

        with self.assertRaises(ValueError):
            _build_cvs(
                morpho,
                policy=BadPolicy(),
                paint_rules=(),
                place_rules=(),
            )

    def test_rejects_non_policy(self) -> None:
        morpho = _single_branch_morpho()
        with self.assertRaises(TypeError):
            _build_cvs(
                morpho,
                policy="not a policy",  # type: ignore[arg-type]
                paint_rules=(),
                place_rules=(),
            )


# =============================================================================
# Property-based invariants (skipped when hypothesis missing)
# =============================================================================

class LowerPropertyTest(unittest.TestCase):

    @given(cv_count=st.integers(min_value=1, max_value=8))
    @settings(max_examples=25, deadline=None)
    def test_cv_lengths_sum_to_branch_total(self, cv_count: int) -> None:
        morpho = Morphology.from_root(
            _branch([30.0], [3.0, 3.0], type="soma"), name="soma",
        )
        cvs = _build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=cv_count),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        total_um = sum(float(cv.length.to_decimal(u.um)) for cv in cvs)
        self.assertAlmostEqual(total_um, 30.0, places=4)

    @given(cv_count=st.integers(min_value=1, max_value=8))
    @settings(max_examples=25, deadline=None)
    def test_coverage_fraction_of_all_region_is_one_per_cv(
        self, cv_count: int,
    ) -> None:
        morpho = Morphology.from_root(
            _branch([20.0], [2.0, 2.0], type="soma"), name="soma",
        )
        bounds = CVPerBranch(cv_per_branch=cv_count).resolve_cv_bounds(morpho)
        geos, _ = _build_geo(morpho, bounds)
        for g in geos:
            self.assertAlmostEqual(
                _coverage_fraction(morpho, g, ((0.0, 1.0),)), 1.0, places=3,
            )

    @given(cv_count=st.integers(min_value=1, max_value=6))
    @settings(max_examples=25, deadline=None)
    def test_each_cv_id_appears_once_as_child_or_root(self, cv_count: int) -> None:
        soma = _branch([30.0], [3.0, 3.0], type="soma")
        dend = _branch([20.0], [2.0, 1.0], type="basal_dendrite")
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.d = dend
        cvs = _build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=cv_count),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        roots = [cv.id for cv in cvs if cv.parent_cv is None]
        children = [cid for cv in cvs for cid in cv.children_cv]
        self.assertEqual(sorted(roots + children), list(range(len(cvs))))


if __name__ == "__main__":
    unittest.main()
