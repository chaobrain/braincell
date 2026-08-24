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

"""Tests for :mod:`braincell._discretization.mechanism`."""

import unittest

import brainunit as u

from braincell._discretization._testing import (
    build_geo,
    make_cable,
    make_single_branch_morpho,
)
from braincell._discretization.base import CV
from braincell._discretization.geometry import (
    CVGeometryResult,
    _GeoCV,
    _build_frusta,
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
from braincell.filter import AllRegion, AtLocation, BranchSlice
from braincell.mech import CableProperty, Channel, CurrentClamp, Ion, StateProbe
from braincell.morph.morphology import Morphology


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
        c = make_cable()
        r1 = PaintRule(region=AllRegion(), mechanism=c)
        r2 = PaintRule(region=AllRegion(), mechanism=c)
        self.assertEqual(r1, r2)
        with self.assertRaises(Exception):
            r1.region = AllRegion()  # type: ignore[misc]

    def test_place_rule_default_site_is_mid(self) -> None:
        rule = PlaceRule(
            locset=AtLocation(branch=0, x=0.5),
            mechanisms=(CurrentClamp(durations=10 * u.ms, amplitudes=0.2 * u.nA),),
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
                (Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV),),
            )

    def test_rejects_empty_mechanisms(self) -> None:
        with self.assertRaises(ValueError):
            normalize_paint_rules(AllRegion(), ())

    def test_accepts_cable_and_density(self) -> None:
        cable = make_cable()
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        rules = normalize_paint_rules(AllRegion(), (cable, ch))
        self.assertEqual(len(rules), 2)
        self.assertIs(rules[0].mechanism, cable)
        self.assertIs(rules[1].mechanism, ch)

    def test_rejects_point_mechanism(self) -> None:
        with self.assertRaises(TypeError):
            normalize_paint_rules(
                AllRegion(),
                (CurrentClamp(durations=10 * u.ms, amplitudes=0.1 * u.nA),),
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
                (CurrentClamp(durations=10 * u.ms, amplitudes=0.1 * u.nA),),
            )

    def test_rejects_empty_mechanisms(self) -> None:
        with self.assertRaises(ValueError):
            normalize_place_rule(AtLocation(branch=0, x=0.5), ())

    def test_rejects_non_point_mechanism(self) -> None:
        with self.assertRaises(TypeError):
            normalize_place_rule(
                AtLocation(branch=0, x=0.5),
                (Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV),),
            )

    def test_returns_place_rule_with_site_mid(self) -> None:
        clamp = CurrentClamp(durations=10 * u.ms, amplitudes=0.1 * u.nA)
        rule = normalize_place_rule(AtLocation(branch=0, x=0.5), (clamp,))
        self.assertEqual(rule.site, "mid")
        self.assertEqual(rule.mechanisms, (clamp,))


class MergePaintRulesTest(unittest.TestCase):
    def test_cable_same_region_replaces(self) -> None:
        r1 = PaintRule(region=AllRegion(), mechanism=make_cable(cm=1.0))
        r2 = PaintRule(region=AllRegion(), mechanism=make_cable(cm=2.0))
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0].mechanism, r2.mechanism)

    def test_cable_different_regions_kept(self) -> None:
        r1 = PaintRule(region=AllRegion(), mechanism=make_cable(cm=1.0))
        r2 = PaintRule(
            region=BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            mechanism=make_cable(cm=2.0),
        )
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)

    def test_density_same_region_same_name_is_preserved_for_cv_validation(self) -> None:
        d1 = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        d2 = Channel("IL", g_max=0.2 * (u.mS / u.cm**2), E=-70 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)
        self.assertIs(merged[0].mechanism, d1)
        self.assertIs(merged[1].mechanism, d2)

    def test_density_same_class_different_names_kept(self) -> None:
        d1 = Channel("IL", name="a", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        d2 = Channel("IL", name="b", g_max=0.2 * (u.mS / u.cm**2), E=-70 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)

    def test_density_different_classes_both_kept(self) -> None:
        d1 = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        d2 = Channel("Na_Ba2002", g_max=0.05 * (u.mS / u.cm**2), E=50 * u.mV)
        r1 = PaintRule(region=AllRegion(), mechanism=d1)
        r2 = PaintRule(region=AllRegion(), mechanism=d2)
        merged = merge_paint_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)


class MergePlaceRulesTest(unittest.TestCase):
    def test_exact_duplicate_kept_as_independent_declaration(self) -> None:
        clamp = CurrentClamp(durations=10 * u.ms, amplitudes=0.1 * u.nA)
        r = normalize_place_rule(AtLocation(branch=0, x=0.5), (clamp,))
        merged = merge_place_rules((r,), (r,))
        self.assertEqual(len(merged), 2)

    def test_different_clamps_both_kept(self) -> None:
        c1 = CurrentClamp(durations=10 * u.ms, amplitudes=0.1 * u.nA)
        c2 = CurrentClamp(durations=10 * u.ms, amplitudes=0.2 * u.nA)
        r1 = normalize_place_rule(AtLocation(branch=0, x=0.5), (c1,))
        r2 = normalize_place_rule(AtLocation(branch=0, x=0.5), (c2,))
        merged = merge_place_rules((r1,), (r2,))
        self.assertEqual(len(merged), 2)


# =============================================================================
# Region cache
# =============================================================================


class RegionCacheTest(unittest.TestCase):
    def test_intervals_returns_same_object_twice(self) -> None:
        morpho = make_single_branch_morpho()
        region = AllRegion()
        cache = _RegionCache(morpho)
        a = cache.intervals(region)
        b = cache.intervals(region)
        self.assertIs(a, b)
        self.assertEqual(a, {0: ((0.0, 1.0),)})

    def test_points_cached(self) -> None:
        morpho = make_single_branch_morpho()
        locset = AtLocation(branch=0, x=0.5)
        cache = _RegionCache(morpho)
        a = cache.points(locset)
        b = cache.points(locset)
        self.assertIs(a, b)
        self.assertEqual(a, ((0, 0.5, "soma(0.5)"),))


# =============================================================================
# Mechanism lowering
# =============================================================================


class CoverageFractionTest(unittest.TestCase):
    def _geo(self) -> tuple[Morphology, _GeoCV]:
        morpho = make_single_branch_morpho()
        geos, _ = build_geo(morpho, (((0.0, 1.0),),))
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
        morpho = make_single_branch_morpho()
        # Using a CV that spans [0, 0.5]; an interval at [0.6, 1.0] should
        # produce 0. Geometry build now validates full branch coverage, so
        # construct the partial geo manually.
        g = _GeoCV(
            id=0,
            branch_id=0,
            branch_type="soma",
            prox=0.0,
            dist=0.5,
            midpoint=0.25,
            parent_cv=None,
            children_cv=(),
            length_um=5.0,
            lateral_area_um2=10.0,
            axial_factor_total_per_cm=1.0,
            axial_factor_prox_per_cm=0.5,
            axial_factor_dist_per_cm=0.5,
            r_prox_um=1.0,
            r_mid_um=1.0,
            diam_arc_mean_um=2.0,
            r_dist_um=1.0,
        )
        self.assertAlmostEqual(
            _coverage_fraction(morpho, g, ((0.6, 1.0),)),
            0.0,
        )


class ApplyDensityTest(unittest.TestCase):
    def test_channel_full_coverage_no_scaling(self) -> None:
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        _apply_density(bucket, ch, region_key=AllRegion(), fraction=1.0)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.coverage_area_fraction, 1.0)

    def test_channel_half_coverage_records_fraction(self) -> None:
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
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
        c1 = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        c2 = Channel("IL", g_max=0.2 * (u.mS / u.cm**2), E=-70 * u.mV)
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        region = AllRegion()
        _apply_density(bucket, c1, region_key=region, fraction=1.0)
        _apply_density(bucket, c2, region_key=region, fraction=1.0)
        self.assertEqual(len(bucket.density_by_key), 1)
        stored = next(iter(bucket.density_by_key.values()))
        self.assertEqual(stored.params["g_max"], 0.2 * (u.mS / u.cm**2))


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
        clamp = CurrentClamp(durations=10 * u.ms, amplitudes=0.1 * u.nA)
        named = _resolve_point_name(clamp, display_name="loc_0")
        self.assertIs(named, clamp)


class ApplyPlaceTest(unittest.TestCase):
    def test_auto_generated_duplicate_gets_stable_placement_suffix(self) -> None:
        seen: set[str] = set()
        bucket = _MechBucket(cable=_DEFAULT_CABLE, density_by_key={}, points=[])
        probe = StateProbe(field="v")
        _apply_place(bucket, probe, display_name="loc_0", placement_id=0, seen_names=seen)
        _apply_place(bucket, probe, display_name="loc_0", placement_id=1, seen_names=seen)
        self.assertEqual(
            [item.name for item in bucket.points],
            ["loc_0_v", "loc_0_v__placement_1"],
        )

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
        morpho = make_single_branch_morpho()
        geos, ids = build_geo(morpho, (((0.0, 1.0),),))
        cable = make_cable(cm=2.0)
        ch = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        paint = (
            PaintRule(region=AllRegion(), mechanism=cable),
            PaintRule(region=AllRegion(), mechanism=ch),
        )
        cache = _RegionCache(morpho)
        buckets = _build_mech(
            morpho,
            geos,
            ids,
            paint_rules=paint,
            place_rules=(),
            cache=cache,
        )
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0].cable.membrane_capacitance, 2.0 * (u.uF / u.cm**2))
        self.assertEqual(len(buckets[0].density_by_key), 1)

    def test_callable_cable_field_resolves_from_cv_context(self) -> None:
        morpho = make_single_branch_morpho()
        geos, ids = build_geo(
            morpho,
            (((0.0, 0.5), (0.5, 1.0)),),
        )
        cable = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=lambda cv: (1.0 + float(cv.diam_arc_mean.to_decimal(u.um))) * (u.uF / u.cm**2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        buckets = _build_mech(
            morpho,
            geos,
            ids,
            paint_rules=(PaintRule(region=AllRegion(), mechanism=cable),),
            place_rules=(),
        )
        values = [float(bucket.cable.membrane_capacitance.to_decimal(u.uF / u.cm**2)) for bucket in buckets]
        self.assertEqual(values, [5.0, 5.0])

    def test_callable_cable_field_must_return_scalar_quantity(self) -> None:
        morpho = make_single_branch_morpho()
        geos, ids = build_geo(morpho, (((0.0, 1.0),),))
        cable = CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=lambda cv: 1.0,
            axial_resistivity=100.0 * (u.ohm * u.cm),
        )
        with self.assertRaisesRegex(TypeError, "must return a Quantity"):
            _build_mech(
                morpho,
                geos,
                ids,
                paint_rules=(PaintRule(region=AllRegion(), mechanism=cable),),
                place_rules=(),
            )

    def test_place_clamp_attaches_to_one_cv(self) -> None:
        morpho = make_single_branch_morpho()
        geos, ids = build_geo(morpho, (((0.0, 0.5), (0.5, 1.0)),))
        clamp = CurrentClamp(durations=10 * u.ms, amplitudes=0.1 * u.nA)
        place = (
            PlaceRule(
                locset=AtLocation(branch=0, x=0.25),
                mechanisms=(clamp,),
            ),
        )
        cache = _RegionCache(morpho)
        buckets = _build_mech(
            morpho,
            geos,
            ids,
            paint_rules=(),
            place_rules=place,
            cache=cache,
        )
        self.assertEqual([len(b.points) for b in buckets], [1, 0])


class BuildMechCachesFrustaTest(unittest.TestCase):
    """MED-03: frusta for (branch, prox, dist) must be computed at most once per _build_mech call."""

    def test_overlapping_rules_reuse_frusta(self) -> None:
        from unittest.mock import patch

        morpho = make_single_branch_morpho()
        # Four CVs so each rule visits four distinct (prox, dist) pairs.
        geos, ids = build_geo(
            morpho,
            (((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),),
        )
        ch1 = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        ch2 = Channel("IL", g_max=0.2 * (u.mS / u.cm**2), E=-60 * u.mV)
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
            with self.assertRaisesRegex(ValueError, "overlap after discretization"):
                _build_mech(
                    morpho,
                    geos,
                    ids,
                    paint_rules=paint,
                    place_rules=(),
                    cache=cache,
                )

        for key, count in calls.items():
            self.assertEqual(
                count,
                1,
                f"_build_frusta was called {count} times for key={key!r}; expected 1 after caching.",
            )


if __name__ == "__main__":
    unittest.main()
