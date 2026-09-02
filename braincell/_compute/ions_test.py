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

"""Tests for :mod:`braincell._compute.ions`."""

import unittest

import braintools
import brainunit as u
import numpy as np

import braincell
from braincell import Cell
from braincell.filter import BranchSlice, at
from ._testing import _build_tree, _quantity_set_at


class RuntimeIonTest(unittest.TestCase):
    """Default, named, dynamic, and kinetic ion behaviour in the cell runtime."""

    def test_default_ions_are_available_with_global_shape(self) -> None:
        import braincell

        cell = Cell(_build_tree())

        cell.init_state()
        rcell = cell

        self.assertIsInstance(rcell.get_ion("na"), braincell.ion.SodiumFixed)
        self.assertIsInstance(rcell.get_ion("k"), braincell.ion.PotassiumFixed)
        self.assertIsInstance(rcell.get_ion("ca"), braincell.ion.CalciumFixed)
        self.assertEqual(rcell.get_ion("na").varshape, (1, 5))
        self.assertEqual(rcell.get_ion("k").varshape, (1, 5))
        self.assertEqual(rcell.get_ion("ca").varshape, (1, 5))

    def test_default_ions_expand_with_population_shape(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2, 3))
        cell.init_state()
        rcell = cell
        self.assertEqual(rcell.get_ion("na").varshape, (2, 3, 5))
        self.assertEqual(rcell.get_ion("k").varshape, (2, 3, 5))
        self.assertEqual(rcell.get_ion("ca").varshape, (2, 3, 5))

    def test_runtime_ions_expose_point_space_geometry_arrays(self) -> None:
        cell = Cell(_build_tree())

        cell.init_state()
        rcell = cell

        na = rcell.get_ion("na")
        self.assertEqual(na.length.shape, (1, 5))
        self.assertEqual(na.area.shape, (1, 5))
        self.assertEqual(na.diam_mid.shape, (1, 5))
        self.assertEqual(na.radius_prox.shape, (1, 5))
        self.assertEqual(na.radius_dist.shape, (1, 5))

        self.assertAlmostEqual(float(na.length[0, 1].to_decimal(u.um)), 20.0, places=12)
        self.assertAlmostEqual(float(na.length[0, 3].to_decimal(u.um)), 100.0, places=12)
        self.assertAlmostEqual(float(na.diam_mid[0, 1].to_decimal(u.um)), 20.0, places=12)
        self.assertAlmostEqual(float(na.diam_mid[0, 3].to_decimal(u.um)), 3.0, places=12)
        self.assertAlmostEqual(float(na.radius_prox[0, 1].to_decimal(u.um)), 10.0, places=12)
        self.assertAlmostEqual(float(na.radius_dist[0, 3].to_decimal(u.um)), 1.0, places=12)
        self.assertAlmostEqual(float(na.area[0, 0].to_decimal(u.um**2)), 0.0, places=12)

    def test_runtime_ion_geometry_expands_with_population_shape(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,))
        cell.init_state()
        rcell = cell
        na = rcell.get_ion("na")
        self.assertEqual(na.length.shape, (2, 5))
        self.assertEqual(na.area.shape, (2, 5))
        self.assertAlmostEqual(float(na.length[0, 1].to_decimal(u.um)), 20.0, places=12)
        self.assertAlmostEqual(float(na.length[1, 3].to_decimal(u.um)), 100.0, places=12)

    def test_single_named_ion_keeps_family_and_class_aliases(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumFixed", name="na_left", E=55.0 * u.mV),
        )

        cell.init_state()
        rcell = cell

        layout = next(layout for layout in rcell.layouts if layout.kind == "ion:SodiumFixed")
        node = rcell.get_runtime_node(layout.id)
        na = rcell.get_ion("na")

        self.assertIs(node, na)
        self.assertIs(rcell.get_ion("SodiumFixed"), na)
        self.assertIs(rcell.get_ion("na_left"), na)
        self.assertIsInstance(na, braincell.ion.SodiumFixed)
        self.assertEqual(layout.point_index.tolist(), [1])
        self.assertAlmostEqual(float(na.E[0, 1].to_decimal(u.mV)), 55.0, places=12)
        self.assertAlmostEqual(float(na.E[0, 3].to_decimal(u.mV)), 50.0, places=12)

    def test_explicit_init_nernst_ion_replaces_default_species_container(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "SodiumInitNernst",
                name="na_pool",
                temp=u.celsius2kelvin(30.0),
                Ci=12.0 * u.mM,
                Co=145.0 * u.mM,
            ),
        )

        cell.init_state()
        rcell = cell

        na = rcell.get_ion("na")
        self.assertIsInstance(na, braincell.ion.SodiumInitNernst)
        self.assertIs(rcell.get_ion("SodiumInitNernst"), na)
        self.assertIs(rcell.get_ion("na_pool"), na)
        self.assertAlmostEqual(
            float(na.temp[0, 1].to_decimal(u.kelvin)), float(u.celsius2kelvin(30.0).to_decimal(u.kelvin)), places=12
        )
        self.assertAlmostEqual(float(na.Ci[0, 1].to_decimal(u.mM)), 12.0, places=12)
        self.assertAlmostEqual(float(na.Co[0, 1].to_decimal(u.mM)), 145.0, places=12)

    def test_multiple_named_ions_make_family_lookup_ambiguous(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion("CalciumFixed", name="ca_hva", E=120.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion("CalciumFixed", name="ca_lva", E=110.0 * u.mV),
        )

        cell.init_state()
        rcell = cell

        self.assertIs(rcell.get_ion("ca_hva"), rcell.get_ion("ca_hva"))
        self.assertIs(rcell.get_ion("ca_lva"), rcell.get_ion("ca_lva"))
        with self.assertRaises(ValueError):
            rcell.get_ion("ca")
        with self.assertRaises(ValueError):
            rcell.get_ion("CalciumFixed")

    def test_dynamic_ion_lifecycle_runs_in_runtime(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "CalciumDetailed",
                name="ca_dyn",
                d=0.5 * u.um,
                tau=10.0 * u.ms,
                C_rest=5e-5 * u.mM,
                Ci_initializer=2.4e-4 * u.mM,
            ),
        )
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "CaT_HM1992",
                ion_name="ca_dyn",
                g_max=2.0 * (u.mS / u.cm**2),
            ),
        )

        cell.init_state()
        rcell = cell
        layout = next(layout for layout in rcell.layouts if layout.kind == "ion:CalciumDetailed")
        ion = rcell.get_ion("ca_dyn")

        self.assertIs(rcell.get_ion("ca"), ion)
        self.assertIs(rcell.get_ion("CalciumDetailed"), ion)

        self.assertIsInstance(ion.Ci, braincell.quad.DiffEqState)
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 2.4e-4, places=12)

        ion.Ci.value = _quantity_set_at(ion.Ci.value, 1, 1.0e-3 * u.mM)
        rcell.reset_state()
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 2.4e-4, places=12)

        rcell.compute_derivative()
        self.assertEqual(ion.Ci.derivative.shape, (1, 5))

        rcell.set_state(layout.id, "Ci_initializer", 7.0e-4 * u.mM)
        rcell.reset_state()
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 7.0e-4, places=12)

    def test_imported_cdp_ion_relaxes_without_channel_in_runtime(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "CdpHVA_SU2015_DCN",
                name="ca_cdp",
                tauCa=70.0 * u.ms,
                caiBase=50e-6 * u.mM,
                depth=0.2 * u.um,
                Ci_initializer=80e-6 * u.mM,
            ),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_cdp")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 80e-6, places=12)

        rcell.compute_derivative()
        expected = -(80e-6 - 50e-6) / 70.0
        self.assertAlmostEqual(float(ion.Ci.derivative[0, 1].to_decimal(u.mM / u.ms)), expected, places=12)

    def test_imported_cdp_ion_and_cahva_channel_run_together(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpHVA_SU2015_DCN",
                name="ca_cdp",
                tauCa=70.0 * u.ms,
                caiBase=50e-6 * u.mM,
                depth=0.2 * u.um,
                Ci_initializer=80e-6 * u.mM,
            ),
        )
        cell.paint(
            region,
            braincell.mech.Channel(
                "CaHVA_SU2015_DCN",
                ion_name="ca_cdp",
                perm=7.5e-6 * (u.cm / u.second),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_cdp", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="CaHVA_SU2015_DCN", field="m"),
            braincell.mech.CurrentProbe(ion="ca_cdp", mechanism="CaHVA_SU2015_DCN"),
        )

        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        self.assertIn("soma(0.5)_ca_cdp_Ci", result.traces)
        self.assertIn("soma(0.5)_CaHVA_SU2015_DCN_m", result.traces)
        self.assertIn("soma(0.5)_CaHVA_SU2015_DCN_current", result.traces)

    def test_imported_cdplva_ion_relaxes_without_channel_in_runtime(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "CdpLVA_SU2015_DCN",
                name="ca_lva",
                tauCal=70.0 * u.ms,
                caliBase=50e-6 * u.mM,
                depth=0.2 * u.um,
                Ci_initializer=80e-6 * u.mM,
            ),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_lva")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 80e-6, places=12)

        rcell.compute_derivative()
        expected = -(80e-6 - 50e-6) / 70.0
        self.assertAlmostEqual(float(ion.Ci.derivative[0, 1].to_decimal(u.mM / u.ms)), expected, places=12)

    def test_imported_cdplva_ion_and_calva_channel_run_together(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpLVA_SU2015_DCN",
                name="ca_lva",
                tauCal=70.0 * u.ms,
                caliBase=50e-6 * u.mM,
                depth=0.2 * u.um,
                Ci_initializer=80e-6 * u.mM,
            ),
        )
        cell.paint(
            region,
            braincell.mech.Channel(
                "CaLVA_SU2015_DCN",
                ion_name="ca_lva",
                perm=1.0 * (u.cm / u.second),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_lva", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="CaLVA_SU2015_DCN", field="m"),
            braincell.mech.MechanismProbe(mechanism="CaLVA_SU2015_DCN", field="h"),
            braincell.mech.CurrentProbe(ion="ca_lva", mechanism="CaLVA_SU2015_DCN"),
        )

        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        self.assertIn("soma(0.5)_ca_lva_Ci", result.traces)
        self.assertIn("soma(0.5)_CaLVA_SU2015_DCN_m", result.traces)
        self.assertIn("soma(0.5)_CaLVA_SU2015_DCN_h", result.traces)
        self.assertIn("soma(0.5)_CaLVA_SU2015_DCN_current", result.traces)

    def test_toy_kinetic_ion_runs_and_exposes_species_probes(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
                Ci_initializer=0.2 * u.mM,
                BC_initializer=0.3 * u.mM,
                Btot=1.0 * u.mM,
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_toy", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_toy", field="BC"),
            braincell.mech.MechanismProbe(mechanism="ca_toy", field="B"),
        )

        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        self.assertIn("soma(0.5)_ca_toy_Ci", result.traces)
        self.assertIn("soma(0.5)_ca_toy_BC", result.traces)
        self.assertIn("soma(0.5)_ca_toy_B", result.traces)

    def test_toy_kinetic_ion_reset_restores_custom_initializers(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
                Ci_initializer=0.2 * u.mM,
                BC_initializer=0.3 * u.mM,
                Btot=1.0 * u.mM,
            ),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_toy")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.BC.value[0, 1].to_decimal(u.mM)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.B.value[0, 1].to_decimal(u.mM)), 0.7, places=12)

        ion.Ci.value = _quantity_set_at(ion.Ci.value, 1, 0.9 * u.mM)
        ion.BC.value = _quantity_set_at(ion.BC.value, 1, 0.8 * u.mM)
        rcell.reset_state()

        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.BC.value[0, 1].to_decimal(u.mM)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.B.value[0, 1].to_decimal(u.mM)), 0.7, places=12)

    def test_toy_source_kinetic_ion_runs_and_exposes_species_probes(self) -> None:
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)

        def _run(ci_source):
            cell = Cell(_build_tree())
            cell.paint(
                region,
                braincell.mech.Ion(
                    "ToyCaBindingSourceKinetic_SU2015_DCN",
                    name="ca_toy_src",
                    Ci_initializer=0.2 * u.mM,
                    BC_initializer=0.3 * u.mM,
                    Btot=1.0 * u.mM,
                    ci_source=ci_source,
                ),
            )
            cell.place(
                at("soma", 0.5),
                braincell.mech.MechanismProbe(mechanism="ca_toy_src", field="Ci"),
                braincell.mech.MechanismProbe(mechanism="ca_toy_src", field="BC"),
                braincell.mech.MechanismProbe(mechanism="ca_toy_src", field="B"),
            )
            return cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)

        baseline = _run(0.0 * u.mM / u.ms)
        result = _run(0.01 * u.mM / u.ms)
        self.assertIn("soma(0.5)_ca_toy_src_Ci", result.traces)
        self.assertIn("soma(0.5)_ca_toy_src_BC", result.traces)
        self.assertIn("soma(0.5)_ca_toy_src_B", result.traces)

        ci_baseline = baseline.traces["soma(0.5)_ca_toy_src_Ci"].to_decimal(u.mM)
        ci = result.traces["soma(0.5)_ca_toy_src_Ci"].to_decimal(u.mM)
        bc = result.traces["soma(0.5)_ca_toy_src_BC"].to_decimal(u.mM)
        b = result.traces["soma(0.5)_ca_toy_src_B"].to_decimal(u.mM)
        self.assertTrue(np.allclose(np.asarray(bc) + np.asarray(b), 1.0, atol=1e-9))
        self.assertGreater(float(np.asarray(ci)[-1, 0]), float(np.asarray(ci_baseline)[-1, 0]))

    def test_toy_ica_source_kinetic_ion_and_cahva_run_together(self) -> None:
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)

        def _run_current_driven(kCa):
            cell = Cell(_build_tree(), V_init=-60.0 * u.mV, solver="staggered")
            cell.paint(
                region,
                braincell.mech.Ion(
                    "ToyCaBindingIcaSourceKinetic_SU2015_DCN",
                    name="ca_toy_ica",
                    Ci_initializer=0.2 * u.mM,
                    BC_initializer=0.3 * u.mM,
                    Btot=1.0 * u.mM,
                    kCa=kCa,
                    depth=0.2 * u.um,
                ),
            )
            cell.paint(
                region,
                braincell.mech.Channel(
                    "CaHVA_SU2015_DCN",
                    ion_name="ca_toy_ica",
                    perm=7.5e-6 * (u.cm / u.second),
                ),
            )
            cell.place(
                at("soma", 0.5),
                braincell.mech.CurrentClamp(delay=0.1 * u.ms, durations=0.8 * u.ms, amplitudes=0.05 * u.nA),
                braincell.mech.MechanismProbe(mechanism="ca_toy_ica", field="Ci"),
                braincell.mech.MechanismProbe(mechanism="ca_toy_ica", field="BC"),
                braincell.mech.MechanismProbe(mechanism="ca_toy_ica", field="B"),
                braincell.mech.MechanismProbe(mechanism="CaHVA_SU2015_DCN", field="m"),
                braincell.mech.CurrentProbe(ion="ca_toy_ica", mechanism="CaHVA_SU2015_DCN"),
            )
            return cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)

        result = _run_current_driven(3.45e-7 / u.coulomb)

        self.assertIn("soma(0.5)_ca_toy_ica_Ci", result.traces)
        self.assertIn("soma(0.5)_ca_toy_ica_BC", result.traces)
        self.assertIn("soma(0.5)_ca_toy_ica_B", result.traces)
        self.assertIn("soma(0.5)_CaHVA_SU2015_DCN_m", result.traces)
        self.assertIn("soma(0.5)_CaHVA_SU2015_DCN_current", result.traces)

        bc = np.asarray(result.traces["soma(0.5)_ca_toy_ica_BC"].to_decimal(u.mM))
        b = np.asarray(result.traces["soma(0.5)_ca_toy_ica_B"].to_decimal(u.mM))
        current = np.asarray(result.traces["soma(0.5)_CaHVA_SU2015_DCN_current"].to_decimal(u.mA / (u.cm**2)))

        self.assertTrue(np.allclose(bc + b, 1.0, atol=1e-9))
        self.assertGreater(float(np.max(np.abs(current))), 0.0)

    def test_toy_factor_kinetic_ion_and_cahva_run_together(self) -> None:
        cell = Cell(_build_tree(), V_init=-60.0 * u.mV, solver="staggered")
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "ToyCaPumpFactorKinetic_SU2015_DCN",
                name="ca_toy_factor",
                Ci_initializer=0.2 * u.mM,
                PumpBound_initializer=0.3 * u.mM * u.um,
                PumpTot=1.0 * u.mM * u.um,
                kCa=3.45e-7 / u.coulomb,
                depth=0.2 * u.um,
            ),
        )
        cell.paint(
            region,
            braincell.mech.Channel(
                "CaHVA_SU2015_DCN",
                ion_name="ca_toy_factor",
                perm=7.5e-6 * (u.cm / u.second),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentClamp(delay=0.1 * u.ms, durations=0.8 * u.ms, amplitudes=0.05 * u.nA),
            braincell.mech.MechanismProbe(mechanism="ca_toy_factor", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_toy_factor", field="PumpBound"),
            braincell.mech.MechanismProbe(mechanism="ca_toy_factor", field="PumpFree"),
            braincell.mech.MechanismProbe(mechanism="CaHVA_SU2015_DCN", field="m"),
            braincell.mech.CurrentProbe(ion="ca_toy_factor", mechanism="CaHVA_SU2015_DCN"),
        )

        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        self.assertIn("soma(0.5)_ca_toy_factor_Ci", result.traces)
        self.assertIn("soma(0.5)_ca_toy_factor_PumpBound", result.traces)
        self.assertIn("soma(0.5)_ca_toy_factor_PumpFree", result.traces)
        self.assertIn("soma(0.5)_CaHVA_SU2015_DCN_m", result.traces)
        self.assertIn("soma(0.5)_CaHVA_SU2015_DCN_current", result.traces)

        pump_bound = np.asarray(result.traces["soma(0.5)_ca_toy_factor_PumpBound"].to_decimal(u.mM * u.um))
        pump_free = np.asarray(result.traces["soma(0.5)_ca_toy_factor_PumpFree"].to_decimal(u.mM * u.um))
        current = np.asarray(result.traces["soma(0.5)_CaHVA_SU2015_DCN_current"].to_decimal(u.mA / (u.cm**2)))

        self.assertTrue(np.allclose(pump_bound + pump_free, 1.0, atol=1e-9))
        self.assertGreater(float(np.max(np.abs(current))), 0.0)

    def test_toy_diam_factor_kinetic_ion_runs_and_exposes_geometry_factor_species(self) -> None:
        cell = Cell(_build_tree(), solver="staggered")
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "ToyDiamFactorKinetic_SU2015_DCN",
                name="ca_diam_factor",
                Ci_initializer=0.2 * u.mM,
                PumpBound_initializer=0.3 * u.mM * u.um,
                PumpTot=1.0 * u.mM * u.um,
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_diam_factor", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_diam_factor", field="PumpBound"),
            braincell.mech.MechanismProbe(mechanism="ca_diam_factor", field="PumpFree"),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_diam_factor")

        self.assertAlmostEqual(float(ion.diam_mid[0, 1].to_decimal(u.um)), 20.0, places=12)
        self.assertAlmostEqual(float(ion.PumpBound.value[0, 1].to_decimal(u.mM * u.um)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.PumpFree.value[0, 1].to_decimal(u.mM * u.um)), 0.7, places=6)

        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        self.assertIn("soma(0.5)_ca_diam_factor_Ci", result.traces)
        self.assertIn("soma(0.5)_ca_diam_factor_PumpBound", result.traces)
        self.assertIn("soma(0.5)_ca_diam_factor_PumpFree", result.traces)

    def test_toy_diam_factor_kinetic_ion_reset_restores_custom_initializers(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "ToyDiamFactorKinetic_SU2015_DCN",
                name="ca_diam_factor",
                Ci_initializer=0.2 * u.mM,
                PumpBound_initializer=0.3 * u.mM * u.um,
                PumpTot=1.0 * u.mM * u.um,
            ),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_diam_factor")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.PumpBound.value[0, 1].to_decimal(u.mM * u.um)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.PumpFree.value[0, 1].to_decimal(u.mM * u.um)), 0.7, places=6)

        ion.Ci.value = _quantity_set_at(ion.Ci.value, 1, 0.9 * u.mM)
        ion.PumpBound.value = _quantity_set_at(ion.PumpBound.value, 1, 0.8 * u.mM * u.um)
        rcell.reset_state()

        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.PumpBound.value[0, 1].to_decimal(u.mM * u.um)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.PumpFree.value[0, 1].to_decimal(u.mM * u.um)), 0.7, places=6)

    def test_cdpstc_goc_runs_and_exposes_species_and_geometry_probes(self) -> None:
        cell = Cell(_build_tree(), solver="staggered")
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpStC_MA2020_GoC",
                name="ca_stc",
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="pump"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="pumpca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM0"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM2C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1N2C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM2N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM2N1C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1C1N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM4"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="vrat"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="parea"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="dsq"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="dsqvol"),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_stc")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 45e-6, places=10)
        self.assertAlmostEqual(float(ion.mg.value[0, 1].to_decimal(u.mM)), 0.59, places=6)
        self.assertAlmostEqual(float(ion.CAM0.value[0, 1].to_decimal(u.mM)), 0.03, places=6)
        self.assertAlmostEqual(float(ion.CAM1C.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM2C.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1N2C.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1N.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM2N.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM2N1C.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1C1N.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM4.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.pump.value[0, 1].to_decimal(u.mol / u.cm**2)), 1e-9, places=15)
        self.assertAlmostEqual(float(ion.pumpca.value[0, 1].to_decimal(u.mol / u.cm**2)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.parea[0, 1].to_decimal(u.um)), float(np.pi * 20.0), places=5)
        self.assertAlmostEqual(float(ion.dsq[0, 1].to_decimal(u.um**2)), 400.0, places=6)

        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        for key in (
            "soma(0.5)_ca_stc_Ci",
            "soma(0.5)_ca_stc_pump",
            "soma(0.5)_ca_stc_pumpca",
            "soma(0.5)_ca_stc_CAM0",
            "soma(0.5)_ca_stc_CAM1C",
            "soma(0.5)_ca_stc_CAM2C",
            "soma(0.5)_ca_stc_CAM1N2C",
            "soma(0.5)_ca_stc_CAM1N",
            "soma(0.5)_ca_stc_CAM2N",
            "soma(0.5)_ca_stc_CAM2N1C",
            "soma(0.5)_ca_stc_CAM1C1N",
            "soma(0.5)_ca_stc_CAM4",
            "soma(0.5)_ca_stc_vrat",
            "soma(0.5)_ca_stc_parea",
            "soma(0.5)_ca_stc_dsq",
            "soma(0.5)_ca_stc_dsqvol",
        ):
            self.assertIn(key, result.traces)

        tracked = {
            "Ci": np.asarray(result.traces["soma(0.5)_ca_stc_Ci"].to_decimal(u.mM)),
            "pump": np.asarray(result.traces["soma(0.5)_ca_stc_pump"].to_decimal(u.mol / u.cm**2)),
            "pumpca": np.asarray(result.traces["soma(0.5)_ca_stc_pumpca"].to_decimal(u.mol / u.cm**2)),
            "CAM0": np.asarray(result.traces["soma(0.5)_ca_stc_CAM0"].to_decimal(u.mM)),
            "CAM1C": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1C"].to_decimal(u.mM)),
            "CAM1N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1N"].to_decimal(u.mM)),
            "CAM2N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM2N"].to_decimal(u.mM)),
            "vrat": np.asarray(result.traces["soma(0.5)_ca_stc_vrat"]),
            "parea": np.asarray(result.traces["soma(0.5)_ca_stc_parea"].to_decimal(u.um)),
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_dsq"].to_decimal(u.um**2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_dsqvol"].to_decimal(u.um**2)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        pump = tracked["pump"]
        pumpca = tracked["pumpca"]
        total = pump + pumpca
        self.assertTrue(np.allclose(total, total[0], atol=1e-15))
        # Imported kinetic-ion traces can shift numerically with solver/runtime
        # details; the contract here is qualitative dynamics plus conservation.
        self.assertAlmostEqual(float(tracked["pump"][-1, 0]), float(tracked["pump"][0, 0]), delta=1e-15)
        self.assertLessEqual(abs(float(tracked["pumpca"][-1, 0])), 1e-12)
        self.assertGreater(float(tracked["Ci"][-1, 0]), float(tracked["Ci"][0, 0]))
        self.assertLess(float(tracked["CAM0"][-1, 0]), float(tracked["CAM0"][0, 0]))
        self.assertGreater(float(tracked["CAM1C"][-1, 0]), float(tracked["CAM1C"][0, 0]))
        self.assertGreater(float(tracked["CAM1N"][-1, 0]), float(tracked["CAM1N"][0, 0]))
        self.assertGreater(float(tracked["CAM2N"][-1, 0]), float(tracked["CAM2N"][0, 0]))

    def test_cdpstc_and_cav3p1_goc_run_together(self) -> None:
        cell = Cell(_build_tree(), solver="staggered", V_init=-60.0 * u.mV)
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpStC_MA2020_GoC",
                name="ca_stc",
                temp=u.celsius2kelvin(25.0),
            ),
        )
        cell.paint(
            region,
            braincell.mech.Channel(
                "Cav3p1_MA2020_GoC",
                ion_name="ca_stc",
                g_max=2.5e-4 * (u.cm / u.second),
                temp=u.celsius2kelvin(25.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentClamp(delay=5.0 * u.ms, durations=20.0 * u.ms, amplitudes=0.005 * u.nA),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="pump"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="pumpca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM0"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM2C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM2N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="vrat"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="parea"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="dsq"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="dsqvol"),
            braincell.mech.MechanismProbe(mechanism="Cav3p1_MA2020_GoC", field="p"),
            braincell.mech.MechanismProbe(mechanism="Cav3p1_MA2020_GoC", field="q"),
            braincell.mech.CurrentProbe(ion="ca_stc", mechanism="Cav3p1_MA2020_GoC"),
        )

        cell.init_state()
        cell.reset_state()
        result = cell.run(dt=0.05 * u.ms, duration=40.0 * u.ms)

        for key in (
            "soma(0.5)_ca_stc_Ci",
            "soma(0.5)_ca_stc_pump",
            "soma(0.5)_ca_stc_pumpca",
            "soma(0.5)_ca_stc_CAM0",
            "soma(0.5)_ca_stc_CAM1C",
            "soma(0.5)_ca_stc_CAM2C",
            "soma(0.5)_ca_stc_CAM1N",
            "soma(0.5)_ca_stc_CAM2N",
            "soma(0.5)_ca_stc_vrat",
            "soma(0.5)_ca_stc_parea",
            "soma(0.5)_ca_stc_dsq",
            "soma(0.5)_ca_stc_dsqvol",
            "soma(0.5)_Cav3p1_MA2020_GoC_p",
            "soma(0.5)_Cav3p1_MA2020_GoC_q",
            "soma(0.5)_Cav3p1_MA2020_GoC_current",
        ):
            self.assertIn(key, result.traces)

        tracked = {
            "Ci": np.asarray(result.traces["soma(0.5)_ca_stc_Ci"].to_decimal(u.mM)),
            "pump": np.asarray(result.traces["soma(0.5)_ca_stc_pump"].to_decimal(u.mol / u.cm**2)),
            "pumpca": np.asarray(result.traces["soma(0.5)_ca_stc_pumpca"].to_decimal(u.mol / u.cm**2)),
            "CAM0": np.asarray(result.traces["soma(0.5)_ca_stc_CAM0"].to_decimal(u.mM)),
            "CAM1C": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1C"].to_decimal(u.mM)),
            "CAM2C": np.asarray(result.traces["soma(0.5)_ca_stc_CAM2C"].to_decimal(u.mM)),
            "CAM1N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1N"].to_decimal(u.mM)),
            "CAM2N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM2N"].to_decimal(u.mM)),
            "vrat": np.asarray(result.traces["soma(0.5)_ca_stc_vrat"]),
            "parea": np.asarray(result.traces["soma(0.5)_ca_stc_parea"].to_decimal(u.um)),
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_dsq"].to_decimal(u.um**2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_dsqvol"].to_decimal(u.um**2)),
            "p": np.asarray(result.traces["soma(0.5)_Cav3p1_MA2020_GoC_p"]),
            "q": np.asarray(result.traces["soma(0.5)_Cav3p1_MA2020_GoC_q"]),
            "current": np.asarray(result.traces["soma(0.5)_Cav3p1_MA2020_GoC_current"].to_decimal(u.mA / (u.cm**2))),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        total = tracked["pump"] + tracked["pumpca"]
        self.assertTrue(np.allclose(total, total[0], atol=1e-18))
        self.assertGreater(float(np.max(np.abs(tracked["current"]))), 0.0)
        self.assertTrue(np.all((tracked["p"] >= 0.0) & (tracked["p"] <= 1.0)))
        self.assertTrue(np.all((tracked["q"] >= 0.0) & (tracked["q"] <= 1.0)))
        self.assertGreater(float(tracked["Ci"][-1, 0]), float(tracked["Ci"][0, 0]))
        self.assertLess(float(tracked["CAM0"][-1, 0]), float(tracked["CAM0"][0, 0]))
        self.assertGreater(float(tracked["CAM1C"][-1, 0]), float(tracked["CAM1C"][0, 0]))
        self.assertGreater(float(tracked["CAM2C"][-1, 0]), float(tracked["CAM2C"][0, 0]))
        self.assertGreater(float(tracked["CAM1N"][-1, 0]), float(tracked["CAM1N"][0, 0]))
        self.assertGreater(float(tracked["CAM2N"][-1, 0]), float(tracked["CAM2N"][0, 0]))

    def test_cdpstc_and_cav2p1_run_together(self) -> None:
        cell = Cell(_build_tree(), solver="staggered", V_init=-60.0 * u.mV)
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpStC_MA2020_GoC",
                name="ca_stc",
                temp=u.celsius2kelvin(25.0),
            ),
        )
        cell.paint(
            region,
            braincell.mech.Channel(
                "Cav2p1_RI2021_SC",
                ion_name="ca_stc",
                g_max=2.2e-4 * (u.cm / u.second),
                temp=u.celsius2kelvin(25.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentClamp(delay=5.0 * u.ms, durations=20.0 * u.ms, amplitudes=0.005 * u.nA),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="pump"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="pumpca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM0"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM2C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM1N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="CAM2N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="vrat"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="parea"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="dsq"),
            braincell.mech.MechanismProbe(mechanism="ca_stc", field="dsqvol"),
            braincell.mech.MechanismProbe(mechanism="Cav2p1_RI2021_SC", field="m"),
            braincell.mech.CurrentProbe(ion="ca_stc", mechanism="Cav2p1_RI2021_SC"),
        )

        cell.init_state()
        cell.reset_state()
        result = cell.run(dt=0.05 * u.ms, duration=40.0 * u.ms)

        for key in (
            "soma(0.5)_ca_stc_Ci",
            "soma(0.5)_ca_stc_pump",
            "soma(0.5)_ca_stc_pumpca",
            "soma(0.5)_ca_stc_CAM0",
            "soma(0.5)_ca_stc_CAM1C",
            "soma(0.5)_ca_stc_CAM2C",
            "soma(0.5)_ca_stc_CAM1N",
            "soma(0.5)_ca_stc_CAM2N",
            "soma(0.5)_ca_stc_vrat",
            "soma(0.5)_ca_stc_parea",
            "soma(0.5)_ca_stc_dsq",
            "soma(0.5)_ca_stc_dsqvol",
            "soma(0.5)_Cav2p1_RI2021_SC_m",
            "soma(0.5)_Cav2p1_RI2021_SC_current",
        ):
            self.assertIn(key, result.traces)

        tracked = {
            "Ci": np.asarray(result.traces["soma(0.5)_ca_stc_Ci"].to_decimal(u.mM)),
            "pump": np.asarray(result.traces["soma(0.5)_ca_stc_pump"].to_decimal(u.mol / u.cm**2)),
            "pumpca": np.asarray(result.traces["soma(0.5)_ca_stc_pumpca"].to_decimal(u.mol / u.cm**2)),
            "CAM0": np.asarray(result.traces["soma(0.5)_ca_stc_CAM0"].to_decimal(u.mM)),
            "CAM1C": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1C"].to_decimal(u.mM)),
            "CAM2C": np.asarray(result.traces["soma(0.5)_ca_stc_CAM2C"].to_decimal(u.mM)),
            "CAM1N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1N"].to_decimal(u.mM)),
            "CAM2N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM2N"].to_decimal(u.mM)),
            "vrat": np.asarray(result.traces["soma(0.5)_ca_stc_vrat"]),
            "parea": np.asarray(result.traces["soma(0.5)_ca_stc_parea"].to_decimal(u.um)),
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_dsq"].to_decimal(u.um**2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_dsqvol"].to_decimal(u.um**2)),
            "m": np.asarray(result.traces["soma(0.5)_Cav2p1_RI2021_SC_m"]),
            "current": np.asarray(result.traces["soma(0.5)_Cav2p1_RI2021_SC_current"].to_decimal(u.mA / (u.cm**2))),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        total = tracked["pump"] + tracked["pumpca"]
        self.assertTrue(np.allclose(total, total[0], atol=1e-18))
        self.assertGreater(float(np.max(np.abs(tracked["current"]))), 0.0)
        self.assertTrue(np.all((tracked["m"] >= 0.0) & (tracked["m"] <= 1.0)))
        self.assertGreater(float(tracked["Ci"][-1, 0]), float(tracked["Ci"][0, 0]))
        self.assertLess(float(tracked["CAM0"][-1, 0]), float(tracked["CAM0"][0, 0]))
        self.assertGreater(float(tracked["CAM1C"][-1, 0]), float(tracked["CAM1C"][0, 0]))
        self.assertGreater(float(tracked["CAM2C"][-1, 0]), float(tracked["CAM2C"][0, 0]))
        self.assertGreater(float(tracked["CAM1N"][-1, 0]), float(tracked["CAM1N"][0, 0]))
        self.assertGreater(float(tracked["CAM2N"][-1, 0]), float(tracked["CAM2N"][0, 0]))

    def test_cdpstc_camonly_goc_runs_and_exposes_species_and_geometry_probes(self) -> None:
        cell = Cell(_build_tree(), solver="staggered")
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpStC_CAMOnly_MA2020_GoC",
                name="ca_stc_camonly",
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM0"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM1C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM2C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM1N2C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM1N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM2N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM2N1C"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM1C1N"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="CAM4"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="vrat"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="dsq"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_camonly", field="dsqvol"),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_stc_camonly")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 45e-6, places=10)
        self.assertAlmostEqual(float(ion.CAM0.value[0, 1].to_decimal(u.mM)), 0.03, places=6)
        self.assertAlmostEqual(float(ion.CAM1C.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1N.value[0, 1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.dsq[0, 1].to_decimal(u.um**2)), 400.0, places=6)

        result = cell.run(dt=0.05 * u.ms, duration=0.2 * u.ms)
        for key in (
            "soma(0.5)_ca_stc_camonly_Ci",
            "soma(0.5)_ca_stc_camonly_CAM0",
            "soma(0.5)_ca_stc_camonly_CAM1C",
            "soma(0.5)_ca_stc_camonly_CAM2C",
            "soma(0.5)_ca_stc_camonly_CAM1N2C",
            "soma(0.5)_ca_stc_camonly_CAM1N",
            "soma(0.5)_ca_stc_camonly_CAM2N",
            "soma(0.5)_ca_stc_camonly_CAM2N1C",
            "soma(0.5)_ca_stc_camonly_CAM1C1N",
            "soma(0.5)_ca_stc_camonly_CAM4",
            "soma(0.5)_ca_stc_camonly_vrat",
            "soma(0.5)_ca_stc_camonly_dsq",
            "soma(0.5)_ca_stc_camonly_dsqvol",
        ):
            self.assertIn(key, result.traces)

        tracked = {
            "Ci": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_Ci"].to_decimal(u.mM)),
            "CAM0": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_CAM0"].to_decimal(u.mM)),
            "CAM1C": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_CAM1C"].to_decimal(u.mM)),
            "CAM1N": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_CAM1N"].to_decimal(u.mM)),
            "CAM2N": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_CAM2N"].to_decimal(u.mM)),
            "vrat": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_vrat"]),
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_dsq"].to_decimal(u.um**2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_dsqvol"].to_decimal(u.um**2)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

    def test_cdpstc_nocam_goc_runs_and_exposes_species_and_geometry_probes(self) -> None:
        cell = Cell(_build_tree(), solver="staggered")
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpStC_NoCAM_MA2020_GoC",
                name="ca_stc_nocam",
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="mg"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="Buff1"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="Buff1_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="Buff2"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="Buff2_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="BTC"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="BTC_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="DMNPE"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="DMNPE_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="PV"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="PV_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="PV_mg"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="pump"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="pumpca"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="vrat"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="parea"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="dsq"),
            braincell.mech.MechanismProbe(mechanism="ca_stc_nocam", field="dsqvol"),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_stc_nocam")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 45e-6, places=10)
        self.assertAlmostEqual(float(ion.mg.value[0, 1].to_decimal(u.mM)), 0.59, places=6)
        self.assertAlmostEqual(float(ion.pump.value[0, 1].to_decimal(u.mol / u.cm**2)), 1e-9, places=15)
        self.assertAlmostEqual(float(ion.pumpca.value[0, 1].to_decimal(u.mol / u.cm**2)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.parea[0, 1].to_decimal(u.um)), float(np.pi * 20.0), places=5)
        self.assertAlmostEqual(float(ion.dsq[0, 1].to_decimal(u.um**2)), 400.0, places=6)

        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        for key in (
            "soma(0.5)_ca_stc_nocam_Ci",
            "soma(0.5)_ca_stc_nocam_mg",
            "soma(0.5)_ca_stc_nocam_Buff1",
            "soma(0.5)_ca_stc_nocam_Buff1_ca",
            "soma(0.5)_ca_stc_nocam_Buff2",
            "soma(0.5)_ca_stc_nocam_Buff2_ca",
            "soma(0.5)_ca_stc_nocam_BTC",
            "soma(0.5)_ca_stc_nocam_BTC_ca",
            "soma(0.5)_ca_stc_nocam_DMNPE",
            "soma(0.5)_ca_stc_nocam_DMNPE_ca",
            "soma(0.5)_ca_stc_nocam_PV",
            "soma(0.5)_ca_stc_nocam_PV_ca",
            "soma(0.5)_ca_stc_nocam_PV_mg",
            "soma(0.5)_ca_stc_nocam_pump",
            "soma(0.5)_ca_stc_nocam_pumpca",
            "soma(0.5)_ca_stc_nocam_vrat",
            "soma(0.5)_ca_stc_nocam_parea",
            "soma(0.5)_ca_stc_nocam_dsq",
            "soma(0.5)_ca_stc_nocam_dsqvol",
        ):
            self.assertIn(key, result.traces)

        tracked = {
            "Ci": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_Ci"].to_decimal(u.mM)),
            "mg": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_mg"].to_decimal(u.mM)),
            "Buff1": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_Buff1"].to_decimal(u.mM)),
            "Buff1_ca": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_Buff1_ca"].to_decimal(u.mM)),
            "Buff2": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_Buff2"].to_decimal(u.mM)),
            "Buff2_ca": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_Buff2_ca"].to_decimal(u.mM)),
            "BTC": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_BTC"].to_decimal(u.mM)),
            "BTC_ca": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_BTC_ca"].to_decimal(u.mM)),
            "DMNPE": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_DMNPE"].to_decimal(u.mM)),
            "DMNPE_ca": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_DMNPE_ca"].to_decimal(u.mM)),
            "PV": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_PV"].to_decimal(u.mM)),
            "PV_ca": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_PV_ca"].to_decimal(u.mM)),
            "PV_mg": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_PV_mg"].to_decimal(u.mM)),
            "pump": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_pump"].to_decimal(u.mol / u.cm**2)),
            "pumpca": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_pumpca"].to_decimal(u.mol / u.cm**2)),
            "vrat": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_vrat"]),
            "parea": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_parea"].to_decimal(u.um)),
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_dsq"].to_decimal(u.um**2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_dsqvol"].to_decimal(u.um**2)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        total = tracked["pump"] + tracked["pumpca"]
        self.assertTrue(np.allclose(total, total[0], atol=1e-18))

    def test_cdpcam_pc_zero_ica_runs_and_exposes_steady_species_and_geometry_probes(self) -> None:
        cell = Cell(_build_tree(), solver="staggered")
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion(
                "CdpCAM_MA2024_PC",
                name="ca_cam",
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="Ci"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="pump"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="pumpca"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CB"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CB_f_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CB_ca_s"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CB_ca_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="PV"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="PV_ca"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="PV_mg"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM0"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM1C"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM2C"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM1N2C"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM1N"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM2N"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM2N1C"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM1C1N"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="CAM4"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="vrat"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="parea"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="dsq"),
            braincell.mech.MechanismProbe(mechanism="ca_cam", field="dsqvol"),
        )

        cell.init_state()
        rcell = cell
        ion = rcell.get_ion("ca_cam")
        self.assertAlmostEqual(float(ion.Ci.value[0, 1].to_decimal(u.mM)), 45e-6, places=10)
        self.assertAlmostEqual(float(ion.CB.value[0, 1].to_decimal(u.mM)), 0.13851901461878652, delta=1e-8)
        self.assertAlmostEqual(float(ion.CB_f_ca.value[0, 1].to_decimal(u.mM)), 0.013185944660826794, delta=1e-8)
        self.assertAlmostEqual(float(ion.CB_ca_s.value[0, 1].to_decimal(u.mM)), 0.007574049472521637, delta=1e-8)
        self.assertAlmostEqual(float(ion.CB_ca_ca.value[0, 1].to_decimal(u.mM)), 0.0007209912478650405, delta=1e-8)
        self.assertAlmostEqual(float(ion.CAM0.value[0, 1].to_decimal(u.mM)), 0.03, delta=1e-8)
        self.assertAlmostEqual(float(ion.pump.value[0, 1].to_decimal(u.mol / u.cm**2)), 1e-9, places=15)
        self.assertAlmostEqual(float(ion.pumpca.value[0, 1].to_decimal(u.mol / u.cm**2)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.parea[0, 1].to_decimal(u.um)), float(np.pi * 20.0), places=5)
        self.assertAlmostEqual(float(ion.dsq[0, 1].to_decimal(u.um**2)), 400.0, places=6)

    def test_cdpcam_pc_ion_params_scatter_with_population_shape(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,), solver="staggered")
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "CdpCAM_MA2024_PC",
                name="ca_cam",
                TotalPump=5.0e-8 * (u.mol / u.cm**2),
            ),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "CdpCAM_MA2024_PC",
                name="ca_cam",
                TotalPump=6.0e-8 * (u.mol / u.cm**2),
            ),
        )

        cell.init_state()
        ion = cell.get_ion("ca_cam")

        self.assertEqual(ion.TotalPump.shape, (2, 5))
        np.testing.assert_allclose(
            np.asarray(ion.TotalPump[:, 1].to_decimal(u.mol / u.cm**2)),
            [5.0e-8, 5.0e-8],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(ion.TotalPump[:, 3].to_decimal(u.mol / u.cm**2)),
            [6.0e-8, 6.0e-8],
            rtol=1e-12,
        )

    def test_constant_quantity_ci_initializer_stays_quantity_with_population_shape(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,), solver="staggered")
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "CdpCAM_MA2024_PC",
                name="ca_cam",
                Ci_initializer=braintools.init.Constant(0.2 * u.mM),
            ),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "CdpCAM_MA2024_PC",
                name="ca_cam",
                Ci_initializer=0.3 * u.mM,
            ),
        )

        cell.init_state()
        ion = cell.get_ion("ca_cam")

        self.assertIsInstance(ion.Ci_initializer, u.Quantity)
        self.assertEqual(ion.Ci_initializer.shape, (2, 5))
        np.testing.assert_allclose(
            np.asarray(ion.Ci_initializer[:, 1].to_decimal(u.mM)),
            [0.2, 0.2],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(ion.Ci_initializer[:, 3].to_decimal(u.mM)),
            [0.3, 0.3],
            rtol=1e-12,
        )

        ion.Ci.value = _quantity_set_at(ion.Ci.value, 1, 1.0 * u.mM)
        cell.reset_state()
        np.testing.assert_allclose(
            np.asarray(ion.Ci.value[:, 1].to_decimal(u.mM)),
            [0.2, 0.2],
            rtol=1e-12,
        )

    def test_nonspecific_placeholder_is_seeded_like_the_other_families(self) -> None:
        # ``build_placeholder_ions`` supplies na/k/ca/no, but the seed loop
        # used to take only the first three, so a channel declaring a
        # NonSpecific owner could not bind: "No ion candidates are registered
        # for family 'no'." ``Kv1p5_MA2020_GrC`` is the one shipped channel
        # that hits it.
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("Kv1p5_MA2020_GrC", name="kv1p5"),
        )

        cell.init_state()

        self.assertEqual(sorted(cell.runtime.ions), ["ca", "k", "na", "no"])
        self.assertIsInstance(cell.get_ion("no"), braincell.ion.NonSpecificFixed)

    def test_placeholder_families_match_the_ion_package_exactly(self) -> None:
        # The seed set is derived from ``build_placeholder_ions`` rather than
        # restated, so the two cannot drift apart again.
        from braincell.ion import build_placeholder_ions

        cell = Cell(_build_tree())
        cell.init_state()

        self.assertEqual(
            sorted(cell.runtime.ions),
            sorted(build_placeholder_ions(size=(1, 5))),
        )

    def test_same_ion_instance_name_cannot_mix_different_classes(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumFixed", name="na_main"),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumInitNernst", name="na_main"),
        )

        with self.assertRaises(ValueError) as ctx:
            cell.init_state()
            rcell = cell

            _ = rcell.layouts
        self.assertIn("cannot denote both", str(ctx.exception))
