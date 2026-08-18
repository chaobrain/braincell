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

"""Tests for :mod:`braincell._compute.bindings`.

``_RuntimeTestTwoOwnerChannel`` is looked up here by registry name only; the
``._testing`` import below is what registers it.
"""

import unittest

import brainunit as u
import numpy as np

import braincell
from braincell import Cell
from braincell.filter import BranchSlice, at
from braincell.quad import get_integrator
from ._testing import _build_tree


class RuntimeBindingTest(unittest.TestCase):
    """Binding of channels to their owning runtime ions."""

    def test_density_mechanism_leaky_builds_runtime_il_node(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm**2), E=-69.0 * u.mV),
        )

        cell.init_state()
        rcell = cell

        layout = rcell.layouts[0]
        node = rcell.get_runtime_node(layout.id)

        self.assertIsInstance(node, braincell.channel.IL)
        self.assertAlmostEqual(float(node.g_max[0, 3].to_decimal(u.mS / u.cm**2)), 4.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0, 2].to_decimal(u.mS / u.cm**2)), 0.0, places=12)

    def test_set_state_syncs_runtime_node_param(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        cell.init_state()
        rcell = cell

        layout = rcell.layouts[0]
        rcell.set_state(layout.id, "g_max", 2.5 * (u.mS / u.cm**2))
        node = rcell.get_runtime_node(layout.id)

        self.assertAlmostEqual(float(node.g_max[0, 1].to_decimal(u.mS / u.cm**2)), 2.5, places=12)
        self.assertAlmostEqual(float(node.g_max[0, 0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)

    def test_single_ion_channel_binds_to_explicit_runtime_ion(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumFixed", name="na_soma", E=55.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumFixed", name="na_dend", E=45.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm**2), ion_name="na_soma"),
        )

        cell.init_state()
        rcell = cell

        channel_layout = next(layout for layout in rcell.layouts if layout.kind == "channel:Na_HH1952")
        na_soma = rcell.get_ion("na_soma")
        na_dend = rcell.get_ion("na_dend")
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertIs(na_soma.channels["Na_HH1952"], node)
        self.assertNotIn("Na_HH1952", na_dend.channels)
        self.assertAlmostEqual(float(na_soma.E[0, 1].to_decimal(u.mV)), 55.0, places=12)
        self.assertAlmostEqual(float(na_dend.E[0, 3].to_decimal(u.mV)), 45.0, places=12)

    def test_same_named_single_ion_channels_in_distinct_layouts_do_not_overwrite(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm**2)),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel("Na_HH1952", g_max=8.0 * (u.mS / u.cm**2)),
        )

        cell.init_state()
        rcell = cell

        layouts = tuple(layout for layout in rcell.layouts if layout.kind == "channel:Na_HH1952")
        nodes = tuple(rcell.get_runtime_node(layout.id) for layout in layouts)
        na = rcell.get_ion("na")
        point_V = rcell._discretization_to_point(rcell.V.value)
        expected = nodes[0].current(point_V, na.pack_info())
        total = na.current(point_V, include_external=False)

        self.assertEqual(len(layouts), 2)
        self.assertEqual(len(set(map(id, nodes))), 1)
        self.assertEqual(len(na.channels), 1)
        self.assertIn("Na_HH1952", na.channels)
        self.assertNotIn("Na_HH1952__layout_1", na.channels)
        self.assertAlmostEqual(float(nodes[0].g_max[0, 1].to_decimal(u.mS / u.cm**2)), 12.0, places=12)
        self.assertAlmostEqual(float(nodes[0].g_max[0, 3].to_decimal(u.mS / u.cm**2)), 8.0, places=12)
        np.testing.assert_allclose(
            np.asarray(total.to_decimal(u.mA / (u.cm**2))),
            np.asarray(expected.to_decimal(u.mA / (u.cm**2))),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_same_named_overlapping_ion_channels_remain_distinct(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm**2)),
        )
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("Na_HH1952", g_max=8.0 * (u.mS / u.cm**2)),
        )

        cell.init_state()
        rcell = cell

        layouts = tuple(layout for layout in rcell.layouts if layout.kind == "channel:Na_HH1952")
        nodes = tuple(rcell.get_runtime_node(layout.id) for layout in layouts)
        na = rcell.get_ion("na")

        self.assertEqual(len(layouts), 2)
        self.assertEqual(len(set(map(id, nodes))), 2)
        self.assertEqual(len(na.channels), 2)
        self.assertIn("Na_HH1952", na.channels)
        self.assertTrue(any(key.startswith("Na_HH1952__layout_") for key in na.channels if key != "Na_HH1952"))

    def test_set_state_syncs_merged_channel_param(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm**2)),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel("Na_HH1952", g_max=8.0 * (u.mS / u.cm**2)),
        )

        cell.init_state()
        rcell = cell

        layouts = tuple(layout for layout in rcell.layouts if layout.kind == "channel:Na_HH1952")
        nodes = tuple(rcell.get_runtime_node(layout.id) for layout in layouts)
        self.assertEqual(len(set(map(id, nodes))), 1)

        rcell.set_state(layouts[0].id, "g_max", 5.0 * (u.mS / u.cm**2))

        node = rcell.get_runtime_node(layouts[0].id)
        self.assertAlmostEqual(float(node.g_max[0, 1].to_decimal(u.mS / u.cm**2)), 5.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0, 3].to_decimal(u.mS / u.cm**2)), 8.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0, 0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)

    def test_single_ion_channel_requires_selector_when_family_is_ambiguous(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion("CalciumFixed", name="ca_hva"),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion("CalciumFixed", name="ca_lva"),
        )
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("CaT_HM1992"),
        )

        with self.assertRaises(ValueError) as ctx:
            cell.init_state()
            rcell = cell

            _ = rcell.layouts
        self.assertIn("ambiguous", str(ctx.exception))

    def test_set_state_on_named_ion_layout_updates_only_that_instance(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumFixed", name="na_left", E=55.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumFixed", name="na_right", E=45.0 * u.mV),
        )

        cell.init_state()
        rcell = cell
        layout = next(
            layout
            for layout in rcell.layouts
            if layout.kind == "ion:SodiumFixed"
            and rcell.runtime.get_layout_mechanism(layout.id).instance_name == "na_left"
        )
        na_left = rcell.get_ion("na_left")
        na_right = rcell.get_ion("na_right")
        rcell.set_state(layout.id, "E", 42.0 * u.mV)

        self.assertAlmostEqual(float(na_left.E[0, 1].to_decimal(u.mV)), 42.0, places=12)
        self.assertAlmostEqual(float(na_right.E[0, 3].to_decimal(u.mV)), 45.0, places=12)

    def test_calva_channel_binds_only_to_explicit_lva_ion_when_multiple_calcium_ions_exist(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(
            region,
            braincell.mech.Ion("CdpHVA_SU2015_DCN", name="ca_hva", Ci_initializer=80e-6 * u.mM),
            braincell.mech.Ion("CdpLVA_SU2015_DCN", name="ca_lva", Ci_initializer=60e-6 * u.mM),
        )
        cell.paint(
            region,
            braincell.mech.Channel("CaLVA_SU2015_DCN", ion_name="ca_lva"),
        )

        cell.init_state()
        rcell = cell

        channel_layout = next(layout for layout in rcell.layouts if layout.kind == "channel:CaLVA_SU2015_DCN")
        ca_hva = rcell.get_ion("ca_hva")
        ca_lva = rcell.get_ion("ca_lva")
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertIs(ca_lva.channels["CaLVA_SU2015_DCN"], node)
        self.assertNotIn("CaLVA_SU2015_DCN", ca_hva.channels)
        self.assertEqual(rcell.runtime.bound_ion_keys[channel_layout.id], ("ca_lva",))
        with self.assertRaises(ValueError):
            rcell.get_ion("ca")

    def test_mixed_ion_channel_binds_per_family_and_uses_owner_ion_bucket(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion("PotassiumFixed", name="k_main", E=-88.0 * u.mV),
            braincell.mech.Ion("CalciumFixed", name="ca_hva", Ci=2e-4 * u.mM),
            braincell.mech.Ion("CalciumFixed", name="ca_lva", Ci=5e-4 * u.mM),
        )
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("Kca3p1_MA2020_GoC", ion_names={"ca": "ca_hva"}),
        )

        cell.init_state()
        rcell = cell

        layout = next(layout for layout in rcell.layouts if layout.kind == "channel:Kca3p1_MA2020_GoC")
        runtime = rcell.runtime
        node = rcell.get_runtime_node(layout.id)
        k_main = rcell.get_ion("k_main")
        ca_hva = rcell.get_ion("ca_hva")

        self.assertEqual(runtime.current_owner_keys[layout.id], "k_main")
        self.assertIn("Kca3p1_MA2020_GoC", k_main.channels)
        self.assertNotIn("Kca3p1_MA2020_GoC", ca_hva.channels)
        self.assertIsInstance(node, braincell.channel.Kca3p1_MA2020_GoC)

    def test_same_named_mixed_ion_channels_in_distinct_layouts_do_not_overwrite_owner_current(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion("PotassiumFixed", name="k_main", E=-88.0 * u.mV),
            braincell.mech.Ion("CalciumFixed", name="ca_hva", Ci=2e-4 * u.mM),
        )
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Kca3p1_MA2020_GoC",
                g_max=100.0 * (u.mS / u.cm**2),
                ion_names={"ca": "ca_hva"},
            ),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Kca3p1_MA2020_GoC",
                g_max=50.0 * (u.mS / u.cm**2),
                ion_names={"ca": "ca_hva"},
            ),
        )

        cell.init_state()
        rcell = cell

        layouts = tuple(layout for layout in rcell.layouts if layout.kind == "channel:Kca3p1_MA2020_GoC")
        nodes = tuple(rcell.get_runtime_node(layout.id) for layout in layouts)
        k_main = rcell.get_ion("k_main")
        ca_hva = rcell.get_ion("ca_hva")
        point_V = rcell._discretization_to_point(rcell.V.value)
        expected = nodes[0].current(point_V, k_main.pack_info(), ca_hva.pack_info())
        total = k_main.current(point_V, include_external=False)

        self.assertEqual(len(layouts), 2)
        self.assertEqual(len(set(map(id, nodes))), 1)
        self.assertEqual(len(k_main.channels), 1)
        self.assertIn("Kca3p1_MA2020_GoC", k_main.channels)
        self.assertNotIn("Kca3p1_MA2020_GoC__layout_1", k_main.channels)
        self.assertEqual(ca_hva.channels, {})
        self.assertAlmostEqual(float(nodes[0].g_max[0, 1].to_decimal(u.mS / u.cm**2)), 100.0, places=12)
        self.assertAlmostEqual(float(nodes[0].g_max[0, 3].to_decimal(u.mS / u.cm**2)), 50.0, places=12)
        np.testing.assert_allclose(
            np.asarray(total.to_decimal(u.mA / (u.cm**2))),
            np.asarray(expected.to_decimal(u.mA / (u.cm**2))),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_mixed_ion_channel_probe_uses_bound_ions_and_owner_total_current(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion("PotassiumFixed", name="k_main", E=-88.0 * u.mV),
            braincell.mech.Ion("CalciumFixed", name="ca_hva", Ci=2e-4 * u.mM),
            braincell.mech.Ion("CalciumFixed", name="ca_lva", Ci=5e-4 * u.mM),
        )
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("Kca3p1_MA2020_GoC", ion_names={"ca": "ca_hva"}),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(mechanism="Kca3p1_MA2020_GoC"),
            braincell.mech.CurrentProbe(ion="k_main"),
        )
        cell.init_state()
        rcell = cell

        samples = rcell.sample_probes()
        runtime = rcell.runtime
        layout = next(layout for layout in rcell.layouts if layout.kind == "channel:Kca3p1_MA2020_GoC")
        node = rcell.get_runtime_node(layout.id)
        point_V = rcell._discretization_to_point(rcell.V.value)
        expected_mechanism = node.current(
            point_V,
            rcell.get_ion("k_main").pack_info(),
            rcell.get_ion("ca_hva").pack_info(),
        )[..., 1]
        expected_total = rcell.get_ion("k_main").current(point_V, include_external=False)[..., 1]

        self.assertEqual(runtime.bound_ion_keys[layout.id], ("k_main", "ca_hva"))
        self.assertEqual(samples["soma(0.5)_Kca3p1_MA2020_GoC_current"], expected_mechanism)
        self.assertEqual(samples["soma(0.5)_k_main_current"], expected_total)

    def test_multi_owner_mixed_ion_channel_exposes_component_currents(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion("PotassiumFixed", name="k_main", E=-88.0 * u.mV),
            braincell.mech.Ion("NonSpecificFixed", name="no"),
        )
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("_RuntimeTestTwoOwnerChannel", ion_names={"k": "k_main", "no": "no"}),
        )

        cell.init_state()
        rcell = cell

        layout = next(layout for layout in rcell.layouts if layout.kind == "channel:_RuntimeTestTwoOwnerChannel")
        point_V = rcell._discretization_to_point(rcell.V.value)
        node = rcell.get_runtime_node(layout.id)
        k_main = rcell.get_ion("k_main")
        no = rcell.get_ion("no")
        total = node.current(point_V, k_main.pack_info(), no.pack_info())
        k_current = k_main.current(point_V, include_external=False)
        no_current = no.current(point_V, include_external=False)

        self.assertEqual(rcell.runtime.current_owner_keys[layout.id], ("k_main", "no"))
        self.assertIn("_RuntimeTestTwoOwnerChannel", k_main.channels)
        self.assertIn("_RuntimeTestTwoOwnerChannel", no.channels)
        self.assertFalse(getattr(k_main.channels["_RuntimeTestTwoOwnerChannel"], "_skip_family_update", False))
        self.assertTrue(getattr(no.channels["_RuntimeTestTwoOwnerChannel"], "_skip_family_update", False))
        np.testing.assert_allclose(
            np.asarray(k_current.to_decimal(u.nA / u.cm**2))[0],
            [0.0, 2.0, 0.0, 2.0, 0.0],
        )
        np.testing.assert_allclose(
            np.asarray(no_current.to_decimal(u.nA / u.cm**2))[0],
            [0.0, 3.0, 0.0, 3.0, 0.0],
        )
        np.testing.assert_allclose(
            np.asarray(total.to_decimal(u.nA / u.cm**2))[..., layout.point_index],
            np.asarray((k_current + no_current).to_decimal(u.nA / u.cm**2))[..., layout.point_index],
        )

    def test_family_order_integrates_mixed_ion_wrapper_channel_only(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Ion("PotassiumFixed", name="k_main", E=-88.0 * u.mV),
            braincell.mech.Ion("CalciumFixed", name="ca_hva", Ci=2e-4 * u.mM),
        )
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("Kca3p1_MA2020_GoC", ion_names={"ca": "ca_hva"}),
        )
        cell.place(at("soma", 0.5), braincell.mech.MechanismProbe(mechanism="Kca3p1_MA2020_GoC", field="p"))
        result = cell.run(dt=0.05 * u.ms, duration=0.1 * u.ms)
        self.assertIn("soma(0.5)_Kca3p1_MA2020_GoC_p", result.traces)

    def test_channel_spec_ina_hh1952_builds_runtime_node_and_binds_to_na(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Na_HH1952",
                g_max=12.0 * (u.mS / u.cm**2),
                V_sh=-50.0 * u.mV,
                temp=u.celsius2kelvin(36.0),
            ),
        )

        cell.init_state()
        rcell = cell
        layout = rcell.layouts[0]
        node = rcell.get_runtime_node(layout.id)
        na = rcell.get_ion("na")

        self.assertIsInstance(node, braincell.channel.Na_HH1952)
        # Channels are now keyed on the declaration's instance name, which
        # defaults to the class name. Users can override with name=.
        self.assertIs(na.channels["Na_HH1952"], node)
        self.assertAlmostEqual(float(node.g_max[0, 1].to_decimal(u.mS / u.cm**2)), 12.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0, 0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.V_sh[0, 1].to_decimal(u.mV)), -50.0, places=12)

    def test_set_state_syncs_runtime_node_param_for_ina_hh1952(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Na_HH1952",
                g_max=12.0 * (u.mS / u.cm**2),
                V_sh=-50.0 * u.mV,
                temp=u.celsius2kelvin(36.0),
            ),
        )

        cell.init_state()
        rcell = cell
        layout = rcell.layouts[0]
        rcell.set_state(layout.id, "g_max", 8.0 * (u.mS / u.cm**2))
        rcell.set_state(layout.id, "V_sh", -42.0 * u.mV)
        node = rcell.get_runtime_node(layout.id)

        self.assertAlmostEqual(float(node.g_max[0, 1].to_decimal(u.mS / u.cm**2)), 8.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0, 0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.V_sh[0, 1].to_decimal(u.mV)), -42.0, places=12)

    def test_unknown_channel_name_raises_key_error(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("__totally_unregistered__", g_max=12.0 * (u.mS / u.cm**2)),
        )

        with self.assertRaises(KeyError) as ctx:
            cell.init_state()
            rcell = cell

            _ = rcell.layouts
        self.assertIn("__totally_unregistered__", str(ctx.exception))


class IsRootLevelRuntimeNodeUnknownClassTest(unittest.TestCase):
    """Task 18 (C6): unknown channel kinds raise rather than silently return False."""

    def test_unknown_channel_kind_raises_value_error(self) -> None:
        from braincell._compute.bindings import _is_root_level_runtime_node

        with self.assertRaises(ValueError) as ctx:
            _is_root_level_runtime_node("channel:__never_registered__")
        self.assertIn("__never_registered__", str(ctx.exception))


class RuntimeSubsolverScheduleTest(unittest.TestCase):
    def test_cell_schedule_applies_to_markov_channels_and_kinetic_ions(self) -> None:
        cell = Cell(_build_tree(), subsolver="rk4", substeps=3)
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(region, braincell.mech.Ion("SodiumFixed", name="na_main"))
        cell.paint(
            region,
            braincell.mech.Channel(
                "Nav1p6_MA2024_PC",
                name="nav",
                ion_name="na_main",
            ),
        )
        cell.paint(
            region,
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
            ),
        )

        cell.init_state()
        markov = next(
            cell.get_runtime_node(layout.id) for layout in cell.layouts if layout.kind == "channel:Nav1p6_MA2024_PC"
        )
        kinetic = cell.get_ion("ca_toy")
        self.assertIs(markov.solver, get_integrator("rk4"))
        self.assertEqual(markov.substeps, 3)
        self.assertIs(kinetic.solver, get_integrator("rk4"))
        self.assertEqual(kinetic.substeps, 3)

    def test_local_override_has_priority_over_cell_schedule(self) -> None:
        cell = Cell(_build_tree(), subsolver="euler", substeps=4)
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(region, braincell.mech.Ion("SodiumFixed", name="na_main"))
        cell.paint(
            region,
            braincell.mech.Channel(
                "Nav1p6_MA2024_PC",
                ion_name="na_main",
                solver="rk4",
                substeps=2,
            ),
        )
        cell.paint(
            region,
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
            ),
        )

        cell.init_state()
        markov = next(
            cell.get_runtime_node(layout.id) for layout in cell.layouts if layout.kind == "channel:Nav1p6_MA2024_PC"
        )
        kinetic = cell.get_ion("ca_toy")
        self.assertIs(markov.solver, get_integrator("rk4"))
        self.assertEqual(markov.substeps, 2)
        self.assertIs(kinetic.solver, get_integrator("euler"))
        self.assertEqual(kinetic.substeps, 4)

    def test_one_kinetic_override_applies_to_the_shared_named_runtime(self) -> None:
        cell = Cell(_build_tree(), subsolver="euler", substeps=4)
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
                solver="rk4",
                substeps=2,
            ),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
            ),
        )

        cell.init_state()
        kinetic = cell.get_ion("ca_toy")
        self.assertIs(kinetic.solver, get_integrator("rk4"))
        self.assertEqual(kinetic.substeps, 2)

    def test_conflicting_shared_kinetic_overrides_are_rejected(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
                solver="euler",
                substeps=2,
            ),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Ion(
                "ToyCaBindingKinetic_SU2015_DCN",
                name="ca_toy",
                solver="rk4",
                substeps=2,
            ),
        )

        with self.assertRaisesRegex(ValueError, "conflicting solver/substeps"):
            cell.init_state()

    def test_non_independent_density_override_is_rejected(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "IL",
                solver="rk4",
                substeps=2,
                g_max=0.1 * u.mS / u.cm**2,
                E=-70 * u.mV,
            ),
        )
        with self.assertRaisesRegex(ValueError, "neither a Markov channel"):
            cell.init_state()

    def test_different_markov_overrides_are_not_merged(self) -> None:
        cell = Cell(_build_tree())
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(region, braincell.mech.Ion("SodiumFixed", name="na_main"))
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Nav1p6_MA2024_PC",
                name="nav",
                ion_name="na_main",
                solver="euler",
                substeps=2,
            ),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Nav1p6_MA2024_PC",
                name="nav",
                ion_name="na_main",
                solver="rk4",
                substeps=3,
            ),
        )

        cell.init_state()
        layouts = [layout for layout in cell.layouts if layout.kind == "channel:Nav1p6_MA2024_PC"]
        nodes = [cell.get_runtime_node(layout.id) for layout in layouts]
        self.assertEqual(len(nodes), 2)
        self.assertIsNot(nodes[0], nodes[1])
        self.assertEqual({node.substeps for node in nodes}, {2, 3})
