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

from braincell._test_support import u

import braincell

from braincell import (
    Branch,
    CVPolicy,
    CableProperties,
    Cell,
    CurrentClamp,
    DensityMechanism,
    Morpho,
)
from braincell.filter import BranchSlice, RootLocation


def _build_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morpho.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


class CellFacadeTest(unittest.TestCase):
    def test_default_cell_has_cv_and_default_paint_rules(self) -> None:
        cell = Cell(_build_tree())
        self.assertEqual(cell.n_cv, 2)
        self.assertEqual(len(cell.paint_rules), 1)
        self.assertEqual(cell.summary()["n_paint_rules"], 1)
        cv0 = cell.cv(0)
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), -65.0, places=12)
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm**2), 1.0, places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), 100.0, places=12)
        self.assertAlmostEqual(
            cv0.temp.to_decimal(u.kelvin),
            u.celsius2kelvin(36.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_cell_freezes_morphology_snapshot(self) -> None:
        tree = _build_tree()
        cell = Cell(tree)
        self.assertEqual(cell.n_cv, 2)
        tree.soma.axon = Branch.from_lengths(lengths=[80.0] * u.um, radii=[1.0, 0.6] * u.um, type="axon")
        self.assertEqual(cell.n_cv, 2)

    def test_cable_paint_hits_midpoint_only(self) -> None:
        cell = Cell(_build_tree())
        base = cell.cv(0)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.49),
            CableProperties(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm**2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        cv0 = cell.cv(0)
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm**2), base.cm.to_decimal(u.uF / u.cm**2), places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), base.ra.to_decimal(u.ohm * u.cm), places=12)
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), base.v.to_decimal(u.mV), places=12)
        self.assertAlmostEqual(cv0.temp.to_decimal(u.kelvin), base.temp.to_decimal(u.kelvin), places=12)

        cell.paint(
            BranchSlice(branch_index=0, prox=0.49, dist=0.51),
            CableProperties(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm**2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        cv0 = cell.cv(0)
        self.assertAlmostEqual(cv0.cm.to_decimal(u.uF / u.cm**2), 2.0, places=12)
        self.assertAlmostEqual(cv0.ra.to_decimal(u.ohm * u.cm), 200.0, places=12)
        self.assertAlmostEqual(cv0.v.to_decimal(u.mV), -70.0, places=12)
        self.assertAlmostEqual(cv0.temp.to_decimal(u.kelvin), u.celsius2kelvin(20.0).to_decimal(u.kelvin), places=12)

    def test_cable_paint_compacts_same_region_history(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=0, prox=0.0, dist=1.0)
        cell.paint(
            region,
            CableProperties(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm**2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(20.0),
            ),
        )
        self.assertEqual(len(cell.paint_rules), 2)

        cell.paint(
            region,
            CableProperties(
                resting_potential=-60.0 * u.mV,
                membrane_capacitance=3.0 * (u.uF / u.cm**2),
                axial_resistivity=300.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(30.0),
            ),
        )
        self.assertEqual(len(cell.paint_rules), 2)

        last = next(rule for rule in cell.paint_rules if rule.region == region)
        cable = last.mechanism
        self.assertIsInstance(cable, CableProperties)
        self.assertAlmostEqual(cable.resting_potential.to_decimal(u.mV), -60.0, places=12)
        self.assertAlmostEqual(cable.membrane_capacitance.to_decimal(u.uF / u.cm**2), 3.0, places=12)
        self.assertAlmostEqual(cable.axial_resistivity.to_decimal(u.ohm * u.cm), 300.0, places=12)
        self.assertAlmostEqual(
            cable.temperature.to_decimal(u.kelvin),
            u.celsius2kelvin(30.0).to_decimal(u.kelvin),
            places=12,
        )

    def test_density_paint_channel_scales_by_area_fraction(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            DensityMechanism(channel_type="leaky", params=(("g_max", 4.0 * (u.mS / u.cm**2)),)),
            DensityMechanism(ion_type="sodium", params=(("c0", 12.0),)),
        )
        cv0 = cell.cv(0)
        self.assertEqual(len(cv0.density_mech), 2)
        channel = next(mech for mech in cv0.density_mech if mech.channel_type is not None)
        ion = next(mech for mech in cv0.density_mech if mech.ion_type is not None)
        gmax = dict(channel.params)["g_max"]
        self.assertAlmostEqual(float(gmax.to_decimal(u.mS / u.cm**2)), 2.0, places=12)
        self.assertEqual(dict(ion.params)["c0"], 12.0)

    def test_place_boundary_goes_to_right_cv_and_branch_endpoint_stays_local(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.d = dend
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))

        stim_boundary = CurrentClamp(amplitude=0.1 * u.nA, delay=1.0 * u.ms, duration=1.0 * u.ms)
        stim_parent_end = CurrentClamp(amplitude=0.2 * u.nA, delay=1.0 * u.ms, duration=1.0 * u.ms)
        cell.place(RootLocation(x=0.5), stim_boundary)
        cell.place(RootLocation(x=1.0), stim_parent_end)

        # RootLocation(x=0.5) on branch 0 should map to right CV (id 1).
        self.assertEqual(len(cell.cv(0).point_mech), 0)
        self.assertEqual(len(cell.cv(1).point_mech), 2)
        # parent branch endpoint remains on branch 0 last CV, not child branch first CV.
        self.assertEqual(cell.cv(1).point_mech[-1].amplitude.to_decimal(u.nA), 0.2)
        self.assertEqual(len(cell.cv(2).point_mech), 0)

    def test_cv_geometry_and_split_axial_resistance(self) -> None:
        soma = Branch.from_lengths(
            lengths=[10.0, 20.0] * u.um,
            radii=[1.0, 2.0, 3.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=1))
        cv0 = cell.cv(0)
        self.assertFalse(hasattr(cv0, "mean_radius"))

        area1 = math.pi * (1.0 + 2.0) * math.sqrt(10.0**2 + (2.0 - 1.0) ** 2)
        area2 = math.pi * (2.0 + 3.0) * math.sqrt(20.0**2 + (3.0 - 2.0) ** 2)
        self.assertAlmostEqual(cv0.lateral_area.to_decimal(u.um**2), area1 + area2, places=9)

        expected_total = 100.0 * (
            (10.0e-4) / (math.pi * (1.0e-4) * (2.0e-4))
            + (20.0e-4) / (math.pi * (2.0e-4) * (3.0e-4))
        )
        expected_prox = 100.0 * (
            (10.0e-4) / (math.pi * (1.0e-4) * (2.0e-4))
            + (5.0e-4) / (math.pi * (2.0e-4) * (2.25e-4))
        )
        expected_dist = 100.0 * ((15.0e-4) / (math.pi * (2.25e-4) * (3.0e-4)))
        self.assertAlmostEqual(cv0.r_axial.to_decimal(u.ohm), expected_total, places=9)
        self.assertAlmostEqual(cv0.r_axial_prox.to_decimal(u.ohm), expected_prox, places=9)
        self.assertAlmostEqual(cv0.r_axial_dist.to_decimal(u.ohm), expected_dist, places=9)
        self.assertAlmostEqual(
            cv0.r_axial.to_decimal(u.ohm),
            cv0.r_axial_prox.to_decimal(u.ohm) + cv0.r_axial_dist.to_decimal(u.ohm),
            places=9,
        )

    def test_zero_length_radius_jump_contributes_area_but_not_axial_resistance(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii_proximal=[2.0, 3.0] * u.um,
            radii_distal=[1.0, 0.5] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

        whole = Cell(tree, cv_policy=CVPolicy(cv_per_branch=1)).cv(0)
        split = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))
        cv_left = split.cv(0)
        cv_right = split.cv(1)

        jump_area = math.pi * (3.0**2 - 1.0**2)
        seg1 = math.pi * (2.0 + 1.0) * math.sqrt(10.0**2 + (1.0 - 2.0) ** 2)
        seg2 = math.pi * (3.0 + 0.5) * math.sqrt(10.0**2 + (0.5 - 3.0) ** 2)

        self.assertAlmostEqual(whole.lateral_area.to_decimal(u.um**2), seg1 + jump_area + seg2, places=9)
        self.assertAlmostEqual(cv_left.lateral_area.to_decimal(u.um**2), seg1, places=9)
        self.assertAlmostEqual(cv_right.lateral_area.to_decimal(u.um**2), jump_area + seg2, places=9)
        self.assertAlmostEqual(whole.r_axial.to_decimal(u.ohm), cv_left.r_axial.to_decimal(u.ohm) + cv_right.r_axial.to_decimal(u.ohm), places=9)

    def test_cv_as_branch(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=2))
        branch_slice = cell.cv(1).as_branch()
        self.assertEqual(branch_slice.type, "soma")
        self.assertGreater(float(branch_slice.length.to_decimal(u.um)), 0.0)

    def test_type_validation_and_removed_legacy_exports(self) -> None:
        tree = _build_tree()
        cell = Cell(tree)
        with self.assertRaises(TypeError):
            Cell(tree.soma)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cell.paint(RootLocation(x=0.5), DensityMechanism(channel_type="hh"))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cell.place(BranchSlice(branch_index=0, prox=0.0, dist=1.0), CurrentClamp(  # type: ignore[arg-type]
                amplitude=0.2 * u.nA,
                delay=2.0 * u.ms,
                duration=1.0 * u.ms,
            ))
        with self.assertRaises(ValueError):
            cell.paint(BranchSlice(branch_index=0, prox=0.0, dist=1.0))
        with self.assertRaises(ValueError):
            cell.place(RootLocation(x=0.5))
        self.assertFalse(hasattr(braincell, "DiscretizationPolicy"))

    def test_lazy_rebuild_only_happens_on_query(self) -> None:
        cell = Cell(_build_tree())
        original_cvs = cell._cvs
        self.assertIsNotNone(original_cvs)

        cell.cv_policy = CVPolicy(cv_per_branch=2)
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)

        cell.paint(
            BranchSlice(branch_index=0, prox=0.4, dist=0.6),
            CableProperties(
                resting_potential=-75.0 * u.mV,
                membrane_capacitance=1.2 * (u.uF / u.cm**2),
                axial_resistivity=120.0 * (u.ohm * u.cm),
            ),
        )
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)

        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(amplitude=0.05 * u.nA, delay=1.0 * u.ms, duration=1.0 * u.ms),
        )
        self.assertTrue(cell._dirty)
        self.assertIs(cell._cvs, original_cvs)

        self.assertEqual(cell.n_cv, 4)
        self.assertFalse(cell._dirty)
        self.assertIsNot(cell._cvs, original_cvs)

    def test_density_paint_channel_fallback_records_area_fraction(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morpho.from_root(soma, name="soma")
        cell = Cell(tree)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=0.5),
            DensityMechanism(channel_type="leaky", params=(("g_max", "not-scalable"),)),
            DensityMechanism(channel_type="leaky", params=(("tau", 10.0),)),
        )
        cv0 = cell.cv(0)
        self.assertEqual(len(cv0.density_mech), 2)

        with_gmax = next(
            mech for mech in cv0.density_mech if "g_max" in dict(mech.params)
        )
        self.assertEqual(dict(with_gmax.params)["g_max"], "not-scalable")
        self.assertAlmostEqual(dict(with_gmax.params)["coverage_area_fraction"], 0.5, places=12)

        no_gmax = next(
            mech for mech in cv0.density_mech if "tau" in dict(mech.params)
        )
        self.assertEqual(dict(no_gmax.params)["tau"], 10.0)
        self.assertAlmostEqual(dict(no_gmax.params)["coverage_area_fraction"], 0.5, places=12)

    def test_public_exports_remain_stable(self) -> None:
        from braincell import CVPolicy as PublicCVPolicy
        from braincell import Cell as PublicCell
        from braincell.cell import CV as PublicCV
        from braincell.cell import PaintRule as PublicPaintRule
        from braincell.cell import PlaceRule as PublicPlaceRule

        self.assertIs(PublicCell, Cell)
        self.assertIs(PublicCVPolicy, CVPolicy)
        self.assertEqual(PublicCV.__name__, "CV")
        self.assertEqual(PublicPaintRule.__name__, "PaintRule")
        self.assertEqual(PublicPlaceRule.__name__, "PlaceRule")
