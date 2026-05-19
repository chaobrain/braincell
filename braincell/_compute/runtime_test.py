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

import braincell
from braincell import (
    Branch,
    CVPerBranch,
    Cell,
    CurrentClamp,
    FunctionClamp,
    Morphology,
    SineClamp,
)
from braincell.filter import BranchSlice, RootLocation, at


def _build_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


class CellRuntimeStateTest(unittest.TestCase):
    def test_density_mechanism_builds_dense_layout_with_global_shape(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )

        self.assertEqual(cell.n_cv, 2)
        cell.init_state(); rcell = cell

        self.assertEqual(len(rcell.node_tree.nodes), 5)
        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "dense")
        self.assertEqual(layout.target, "density")
        self.assertEqual(layout.kind, "channel:leaky")
        self.assertEqual(layout.n_active, 2)
        self.assertEqual(layout.source_cv_ids, (0, 1))
        self.assertEqual(layout.point_index.tolist(), [1, 3])
        self.assertEqual(rcell.expected_state_shape(layout.id, "g_max"), (5,))
        self.assertEqual(rcell.voltage_shape, (5,))
        self.assertEqual(rcell.get_state(layout.id, "g_max").shape, (5,))
        self.assertTrue(all(value == 4.0 * (u.mS / u.cm ** 2) for value in rcell.get_state(layout.id, "g_max")))
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(0)), ())
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(1)), (layout.id,))
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(3)), (layout.id,))

    def test_point_mechanism_builds_sparse_layout_with_local_shape(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp.step(0.1 * u.nA, 2.0 * u.ms, delay=1.0 * u.ms),
        )

        cell.init_state(); rcell = cell

        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "sparse")
        self.assertEqual(layout.target, "point")
        self.assertEqual(layout.kind, "CurrentClamp")
        self.assertEqual(layout.n_active, 1)
        self.assertEqual(layout.point_index.tolist(), [1])
        self.assertIsNone(layout.point_mask)
        self.assertEqual(rcell.expected_state_shape(layout.id, "amplitudes"), (1, 1))
        self.assertEqual(len(rcell.get_state(layout.id, "amplitudes")), 1)
        self.assertEqual(tuple(item.to_decimal(u.nA) for item in rcell.get_state(layout.id, "amplitudes")[0]), (0.1,))
        self.assertEqual(tuple(item.to_decimal(u.ms) for item in rcell.get_state(layout.id, "durations")[0]), (2.0,))
        self.assertEqual(rcell.get_state(layout.id, "start")[0], 1.0 * u.ms)
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(1)), (layout.id,))
        self.assertEqual(rcell.get_point_layouts(1), (layout,))

    def test_point_mechanism_can_land_on_root_endpoint(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.0),
            CurrentClamp.step(0.1 * u.nA, 2.0 * u.ms, delay=1.0 * u.ms),
        )

        cell.init_state(); rcell = cell

        root_node_id = rcell.node_tree.root_node_id
        midpoint_id = int(rcell.node_tree.cv_to_mid_node_id[0])
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "sparse")
        self.assertEqual(layout.target, "point")
        self.assertEqual(layout.point_index.tolist(), [root_node_id])
        self.assertEqual(tuple(item.id for item in rcell.get_point_layouts(root_node_id)), (layout.id,))
        self.assertEqual(tuple(item.id for item in rcell.get_point_layouts(midpoint_id)), ())
        self.assertEqual(tuple(item.id for item in rcell.get_cv_layouts(0)), (layout.id,))

    def test_channel_spec_builds_dense_layout_with_global_shape(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state(); rcell = cell

        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "dense")
        self.assertEqual(layout.kind, "channel:IL")
        self.assertEqual(rcell.expected_state_shape(layout.id, "g_max"), (5,))
        self.assertEqual(rcell.expected_state_shape(layout.id, "E"), (5,))
        self.assertEqual(rcell.get_point_state(1)[layout.id]["g_max"], 4.0 * (u.mS / u.cm ** 2))
        self.assertEqual(rcell.get_point_state(3)[layout.id]["E"], -68.0 * u.mV)
        node = rcell.get_runtime_node(layout.id)
        self.assertIsInstance(node, braincell.channel.IL)
        self.assertEqual(node.varshape, (5,))
        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm ** 2)), 4.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm ** 2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.E[1].to_decimal(u.mV)), -68.0, places=12)

    def test_named_channel_spec_merges_across_regions_when_identity_matches(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_main", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_main", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state(); rcell = cell

        self.assertEqual(len(rcell.layouts), 1)
        self.assertEqual(rcell.layouts[0].point_index.tolist(), [1, 3])

    def test_same_class_different_names_build_distinct_layouts(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_a", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
            braincell.mech.Channel("IL", name="leak_b", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state(); rcell = cell

        self.assertEqual(len(rcell.layouts), 2)
        self.assertEqual({layout.kind for layout in rcell.layouts}, {"channel:IL"})
        self.assertTrue(all(layout.point_index.tolist() == [1, 3] for layout in rcell.layouts))

    def test_runtime_state_keeps_dense_and_sparse_layouts_together(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )
        clamp = CurrentClamp.step(0.1 * u.nA, 2.0 * u.ms, delay=1.0 * u.ms)
        cell.place(RootLocation(x=0.5), clamp)

        cell.init_state(); rcell = cell

        self.assertEqual(len(rcell.layouts), 2)
        dense = next(layout for layout in rcell.layouts if layout.layout == "dense")
        sparse = next(layout for layout in rcell.layouts if layout.layout == "sparse")
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(1)), (dense.id, sparse.id))
        self.assertEqual(tuple(layout.id for layout in rcell.get_point_layouts(3)), (dense.id,))
        self.assertEqual(tuple(layout.id for layout in rcell.get_cv_layouts(0)), (dense.id, sparse.id))
        self.assertEqual(tuple(layout.id for layout in rcell.get_cv_layouts(1)), (dense.id,))
        point_state = rcell.get_point_state(1)
        self.assertEqual(point_state[dense.id]["g_max"], 4.0 * (u.mS / u.cm ** 2))
        self.assertEqual(tuple(item.to_decimal(u.nA) for item in point_state[sparse.id]["amplitudes"]), (0.1,))
        self.assertEqual(rcell.get_cv_state(0)[dense.id]["g_max"], 4.0 * (u.mS / u.cm ** 2))
        self.assertEqual({name for name in ("na", "k", "ca")}, {"na", "k", "ca"})

    def test_rebuild_after_place_produces_new_runtime(self) -> None:
        cell = Cell(_build_tree())

        cell.init_state()
        first = cell.layouts
        self.assertEqual(len(first), 0)

        cell.reset()
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp.step(0.1 * u.nA, 2.0 * u.ms, delay=1.0 * u.ms),
        )
        cell.init_state()
        second = cell.layouts

        self.assertIsNot(first, second)
        self.assertEqual(len(second), 1)

    def test_state_mutation_updates_buffer_without_rebuild(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp.step(0.1 * u.nA, 2.0 * u.ms, delay=1.0 * u.ms),
        )

        cell.init_state(); rcell = cell

        layout = rcell.layouts[0]
        rcell.set_state(layout.id, "amplitudes", (0.25 * u.nA, 0.05 * u.nA))
        rcell.set_state(layout.id, "durations", (1.5 * u.ms, 2.5 * u.ms))

        self.assertEqual(tuple(item.to_decimal(u.nA) for item in rcell.get_state(layout.id, "amplitudes")[0]),
                         (0.25, 0.05))
        self.assertEqual(tuple(item.to_decimal(u.ms) for item in rcell.get_point_state(1)[layout.id]["durations"]),
                         (1.5, 2.5))

    def test_runtime_evaluates_step_sine_and_function_clamps_on_target_points(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(start=0.0 * u.ms, durations=(2.0 * u.ms, 2.0 * u.ms), amplitudes=(0.0 * u.nA, 0.3 * u.nA)),
            SineClamp(amplitude=0.2 * u.nA, frequency=500.0 * u.Hz, offset=0.1 * u.nA, duration=4.0 * u.ms),
            FunctionClamp(fn=lambda local_t: 0.4 * u.nA if local_t < 1.0 * u.ms else 0.0 * u.nA, duration=3.0 * u.ms),
        )
        cell.init_state(); rcell = cell

        runtime = rcell.runtime

        current_early = runtime.evaluate_point_clamps(t=0.5 * u.ms)
        current_late = runtime.evaluate_point_clamps(t=2.5 * u.ms)

        self.assertEqual(current_early.shape, (len(rcell.node_tree.nodes),))
        self.assertAlmostEqual(float(current_early[1].to_decimal(u.nA)), 0.7, places=6)
        self.assertAlmostEqual(float(current_early[0].to_decimal(u.nA)), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[1].to_decimal(u.nA)), 0.6, places=6)

    def test_probe_layouts_are_sparse_and_allocate_no_state_buffers(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="Na_HH1952", field="p"),
            braincell.mech.CurrentProbe(ion="na", mechanism="Na_HH1952"),
        )

        cell.init_state(); rcell = cell

        self.assertEqual(len(rcell.layouts), 3)
        self.assertTrue(all(layout.layout == "sparse" for layout in rcell.layouts))
        self.assertTrue(all(layout.target == "point" for layout in rcell.layouts))
        resolved_names = []
        for layout in rcell.layouts:
            self.assertEqual(layout.point_index.tolist(), [1])
            self.assertEqual(rcell.get_point_state(1)[layout.id], {})
            declaration = rcell.runtime.get_layout_mechanism(layout.id)
            resolved_names.append(declaration.name)
        self.assertEqual(
            sorted(resolved_names),
            ["soma(0.5)_Na_HH1952_current", "soma(0.5)_Na_HH1952_p", "soma(0.5)_v"],
        )

    def test_sample_probe_reads_voltage_and_channel_gate_state(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Na_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                temp=u.celsius2kelvin(36.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="Na_HH1952", field="p"),
        )
        cell.init_state(); rcell = cell

        samples = rcell.sample_probes()
        channel_layout = next(
            layout for layout in rcell.layouts
            if isinstance(rcell.runtime.get_layout_mechanism(layout.id), braincell.mech.Channel)
        )
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertEqual(samples["soma(0.5)_v"], rcell.V.value[0])
        self.assertEqual(samples["soma(0.5)_Na_HH1952_p"], node.p.value[1])
        self.assertEqual(rcell.sample_probe("soma(0.5)_Na_HH1952_p"), node.p.value[1])

    def test_sample_probe_reads_mechanism_and_total_ion_current(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "K_Kv_test",
                g_max=0.1 * (u.mS / u.cm ** 2),
                v12=25.0 * u.mV,
                q=9.0,
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(ion="k", mechanism="K_Kv_test"),
            braincell.mech.CurrentProbe(ion="k"),
        )
        cell.init_state(); rcell = cell

        samples = rcell.sample_probes()
        ion = rcell.get_ion("k")
        node = ion.channels["K_Kv_test"]
        point_V = rcell._discretization_to_point(rcell.V.value)
        expected_mechanism = node.current(point_V, ion.pack_info())[1]
        expected_total = ion.current(point_V, include_external=False)[1]

        self.assertEqual(samples["soma(0.5)_K_Kv_test_current"], expected_mechanism)
        self.assertEqual(samples["soma(0.5)_k_current"], expected_total)

    def test_sample_probe_reads_pure_channel_current_without_ion_selector(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "IL",
                g_max=0.1 * (u.mS / u.cm ** 2),
                E=-68.0 * u.mV,
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(mechanism="IL"),
        )
        cell.init_state(); rcell = cell

        samples = rcell.sample_probes()
        channel_layout = next(
            layout for layout in rcell.layouts
            if isinstance(rcell.runtime.get_layout_mechanism(layout.id), braincell.mech.Channel)
        )
        node = rcell.get_runtime_node(channel_layout.id)
        point_V = rcell._discretization_to_point(rcell.V.value)
        expected_current = node.current(point_V)[1]

        self.assertEqual(samples["soma(0.5)_IL_current"], expected_current)

    def test_sample_probe_reads_plain_field_and_rejects_unknown_mechanism(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Na_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                temp=u.celsius2kelvin(36.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="Na_HH1952", field="g_max"),
            braincell.mech.MechanismProbe(mechanism="missing", field="p"),
        )
        cell.init_state(); rcell = cell

        self.assertEqual(
            rcell.sample_probe("soma(0.5)_Na_HH1952_g_max"),
            rcell.get_ion("na").channels["Na_HH1952"].g_max[1],
        )
        with self.assertRaises(KeyError):
            rcell.sample_probe("soma(0.5)_missing_p")

    def test_sample_probes_requires_unique_names(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Na_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                temp=u.celsius2kelvin(36.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(name="dup"),
            braincell.mech.MechanismProbe(name="dup", mechanism="Na_HH1952", field="p"),
        )
        cell.init_state(); rcell = cell

        with self.assertRaises(ValueError):
            rcell.sample_probes()

    def test_density_mechanism_leaky_builds_runtime_il_node(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "leaky", g_max=4.0 * (u.mS / u.cm ** 2), E=-69.0 * u.mV
            ),
        )

        cell.init_state(); rcell = cell

        layout = rcell.layouts[0]
        node = rcell.get_runtime_node(layout.id)

        self.assertIsInstance(node, braincell.channel.IL)
        self.assertAlmostEqual(float(node.g_max[3].to_decimal(u.mS / u.cm ** 2)), 4.0, places=12)
        self.assertAlmostEqual(float(node.g_max[2].to_decimal(u.mS / u.cm ** 2)), 0.0, places=12)

    def test_set_state_syncs_runtime_node_param(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )

        cell.init_state(); rcell = cell

        layout = rcell.layouts[0]
        rcell.set_state(layout.id, "g_max", 2.5 * (u.mS / u.cm ** 2))
        node = rcell.get_runtime_node(layout.id)

        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm ** 2)), 2.5, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm ** 2)), 0.0, places=12)

    def test_default_ions_are_available_with_global_shape(self) -> None:
        import braincell

        cell = Cell(_build_tree())

        cell.init_state(); rcell = cell

        self.assertIsInstance(rcell.get_ion("na"), braincell.ion.SodiumFixed)
        self.assertIsInstance(rcell.get_ion("k"), braincell.ion.PotassiumFixed)
        self.assertIsInstance(rcell.get_ion("ca"), braincell.ion.CalciumFixed)
        self.assertEqual(rcell.get_ion("na").varshape, (5,))
        self.assertEqual(rcell.get_ion("k").varshape, (5,))
        self.assertEqual(rcell.get_ion("ca").varshape, (5,))

    def test_runtime_ions_expose_point_space_geometry_arrays(self) -> None:
        cell = Cell(_build_tree())

        cell.init_state(); rcell = cell

        na = rcell.get_ion("na")
        self.assertEqual(na.length.shape, (5,))
        self.assertEqual(na.area.shape, (5,))
        self.assertEqual(na.diam_mid.shape, (5,))
        self.assertEqual(na.radius_prox.shape, (5,))
        self.assertEqual(na.radius_dist.shape, (5,))

        self.assertAlmostEqual(float(na.length[1].to_decimal(u.um)), 20.0, places=12)
        self.assertAlmostEqual(float(na.length[3].to_decimal(u.um)), 100.0, places=12)
        self.assertAlmostEqual(float(na.diam_mid[1].to_decimal(u.um)), 20.0, places=12)
        self.assertAlmostEqual(float(na.diam_mid[3].to_decimal(u.um)), 3.0, places=12)
        self.assertAlmostEqual(float(na.radius_prox[1].to_decimal(u.um)), 10.0, places=12)
        self.assertAlmostEqual(float(na.radius_dist[3].to_decimal(u.um)), 1.0, places=12)
        self.assertAlmostEqual(float(na.area[0].to_decimal(u.um ** 2)), 0.0, places=12)

    def test_single_named_ion_keeps_family_and_class_aliases(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Ion("SodiumFixed", name="na_left", E=55.0 * u.mV),
        )

        cell.init_state(); rcell = cell

        layout = next(layout for layout in rcell.layouts if layout.kind == "ion:SodiumFixed")
        node = rcell.get_runtime_node(layout.id)
        na = rcell.get_ion("na")

        self.assertIs(node, na)
        self.assertIs(rcell.get_ion("SodiumFixed"), na)
        self.assertIs(rcell.get_ion("na_left"), na)
        self.assertIsInstance(na, braincell.ion.SodiumFixed)
        self.assertEqual(layout.point_index.tolist(), [1])
        self.assertAlmostEqual(float(na.E[1].to_decimal(u.mV)), 55.0, places=12)
        self.assertAlmostEqual(float(na.E[3].to_decimal(u.mV)), 50.0, places=12)

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

        cell.init_state(); rcell = cell

        na = rcell.get_ion("na")
        self.assertIsInstance(na, braincell.ion.SodiumInitNernst)
        self.assertIs(rcell.get_ion("SodiumInitNernst"), na)
        self.assertIs(rcell.get_ion("na_pool"), na)
        self.assertAlmostEqual(float(na.temp[1].to_decimal(u.kelvin)),
                               float(u.celsius2kelvin(30.0).to_decimal(u.kelvin)), places=12)
        self.assertAlmostEqual(float(na.Ci[1].to_decimal(u.mM)), 12.0, places=12)
        self.assertAlmostEqual(float(na.Co[1].to_decimal(u.mM)), 145.0, places=12)

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

        cell.init_state(); rcell = cell

        self.assertIs(rcell.get_ion("ca_hva"), rcell.get_ion("ca_hva"))
        self.assertIs(rcell.get_ion("ca_lva"), rcell.get_ion("ca_lva"))
        with self.assertRaises(ValueError):
            rcell.get_ion("ca")
        with self.assertRaises(ValueError):
            rcell.get_ion("CalciumFixed")

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
            braincell.mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm ** 2), ion_name="na_soma"),
        )

        cell.init_state(); rcell = cell

        channel_layout = next(layout for layout in rcell.layouts if layout.kind == "channel:Na_HH1952")
        na_soma = rcell.get_ion("na_soma")
        na_dend = rcell.get_ion("na_dend")
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertIs(na_soma.channels["Na_HH1952"], node)
        self.assertNotIn("Na_HH1952", na_dend.channels)
        self.assertAlmostEqual(float(na_soma.E[1].to_decimal(u.mV)), 55.0, places=12)
        self.assertAlmostEqual(float(na_dend.E[3].to_decimal(u.mV)), 45.0, places=12)

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
            cell.init_state(); rcell = cell

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

        cell.init_state(); rcell = cell
        layout = next(
            layout
            for layout in rcell.layouts
            if layout.kind == "ion:SodiumFixed"
            and rcell.runtime.get_layout_mechanism(layout.id).instance_name == "na_left"
        )
        na_left = rcell.get_ion("na_left")
        na_right = rcell.get_ion("na_right")
        rcell.set_state(layout.id, "E", 42.0 * u.mV)

        self.assertAlmostEqual(float(na_left.E[1].to_decimal(u.mV)), 42.0, places=12)
        self.assertAlmostEqual(float(na_right.E[3].to_decimal(u.mV)), 45.0, places=12)

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
                g_max=2.0 * (u.mS / u.cm ** 2),
            ),
        )

        cell.init_state(); rcell = cell
        layout = next(layout for layout in rcell.layouts if layout.kind == "ion:CalciumDetailed")
        ion = rcell.get_ion("ca_dyn")

        self.assertIs(rcell.get_ion("ca"), ion)
        self.assertIs(rcell.get_ion("CalciumDetailed"), ion)

        self.assertIsInstance(ion.Ci, braincell.quad.DiffEqState)
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 2.4e-4, places=12)

        ion.Ci.value = ion.Ci.value.at[1].set(1.0e-3 * u.mM)
        rcell.reset_state()
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 2.4e-4, places=12)

        rcell.compute_derivative()
        self.assertEqual(ion.Ci.derivative.shape, (5,))

        rcell.set_state(layout.id, "Ci_initializer", 7.0e-4 * u.mM)
        rcell.reset_state()
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 7.0e-4, places=12)

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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_cdp")
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 80e-6, places=12)

        rcell.compute_derivative()
        expected = -(80e-6 - 50e-6) / 70.0
        self.assertAlmostEqual(float(ion.Ci.derivative[1].to_decimal(u.mM / u.ms)), expected, places=12)

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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_lva")
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 80e-6, places=12)

        rcell.compute_derivative()
        expected = -(80e-6 - 50e-6) / 70.0
        self.assertAlmostEqual(float(ion.Ci.derivative[1].to_decimal(u.mM / u.ms)), expected, places=12)

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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_toy")
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.BC.value[1].to_decimal(u.mM)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.B.value[1].to_decimal(u.mM)), 0.7, places=12)

        ion.Ci.value = ion.Ci.value.at[1].set(0.9 * u.mM)
        ion.BC.value = ion.BC.value.at[1].set(0.8 * u.mM)
        rcell.reset_state()

        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.BC.value[1].to_decimal(u.mM)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.B.value[1].to_decimal(u.mM)), 0.7, places=12)

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
        self.assertGreater(float(np.asarray(ci)[-1]), float(np.asarray(ci_baseline)[-1]))

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
                braincell.mech.CurrentClamp.step(0.05 * u.nA, 0.8 * u.ms, delay=0.1 * u.ms),
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
        current = np.asarray(result.traces["soma(0.5)_CaHVA_SU2015_DCN_current"].to_decimal(u.mA / (u.cm ** 2)))

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
            braincell.mech.CurrentClamp.step(0.05 * u.nA, 0.8 * u.ms, delay=0.1 * u.ms),
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
        current = np.asarray(result.traces["soma(0.5)_CaHVA_SU2015_DCN_current"].to_decimal(u.mA / (u.cm ** 2)))

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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_diam_factor")

        self.assertAlmostEqual(float(ion.diam_mid[1].to_decimal(u.um)), 20.0, places=12)
        self.assertAlmostEqual(float(ion.PumpBound.value[1].to_decimal(u.mM * u.um)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.PumpFree.value[1].to_decimal(u.mM * u.um)), 0.7, places=6)

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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_diam_factor")
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.PumpBound.value[1].to_decimal(u.mM * u.um)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.PumpFree.value[1].to_decimal(u.mM * u.um)), 0.7, places=6)

        ion.Ci.value = ion.Ci.value.at[1].set(0.9 * u.mM)
        ion.PumpBound.value = ion.PumpBound.value.at[1].set(0.8 * u.mM * u.um)
        rcell.reset_state()

        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 0.2, places=12)
        self.assertAlmostEqual(float(ion.PumpBound.value[1].to_decimal(u.mM * u.um)), 0.3, places=12)
        self.assertAlmostEqual(float(ion.PumpFree.value[1].to_decimal(u.mM * u.um)), 0.7, places=6)

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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_stc")
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 45e-6, places=10)
        self.assertAlmostEqual(float(ion.mg.value[1].to_decimal(u.mM)), 0.59, places=6)
        self.assertAlmostEqual(float(ion.CAM0.value[1].to_decimal(u.mM)), 0.03, places=6)
        self.assertAlmostEqual(float(ion.CAM1C.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM2C.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1N2C.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1N.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM2N.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM2N1C.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1C1N.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM4.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.pump.value[1].to_decimal(u.mol / u.cm ** 2)), 1e-9, places=15)
        self.assertAlmostEqual(float(ion.pumpca.value[1].to_decimal(u.mol / u.cm ** 2)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.parea[1].to_decimal(u.um)), float(np.pi * 20.0), places=5)
        self.assertAlmostEqual(float(ion.dsq[1].to_decimal(u.um ** 2)), 400.0, places=6)

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
            "pump": np.asarray(result.traces["soma(0.5)_ca_stc_pump"].to_decimal(u.mol / u.cm ** 2)),
            "pumpca": np.asarray(result.traces["soma(0.5)_ca_stc_pumpca"].to_decimal(u.mol / u.cm ** 2)),
            "CAM0": np.asarray(result.traces["soma(0.5)_ca_stc_CAM0"].to_decimal(u.mM)),
            "CAM1C": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1C"].to_decimal(u.mM)),
            "CAM1N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM1N"].to_decimal(u.mM)),
            "CAM2N": np.asarray(result.traces["soma(0.5)_ca_stc_CAM2N"].to_decimal(u.mM)),
            "vrat": np.asarray(result.traces["soma(0.5)_ca_stc_vrat"]),
            "parea": np.asarray(result.traces["soma(0.5)_ca_stc_parea"].to_decimal(u.um)),
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_dsq"].to_decimal(u.um ** 2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_dsqvol"].to_decimal(u.um ** 2)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        pump = tracked["pump"]
        pumpca = tracked["pumpca"]
        total = pump + pumpca
        self.assertTrue(np.allclose(total, total[0], atol=1e-18))
        self.assertAlmostEqual(float(tracked["Ci"][-1]), 4.096707925782539e-05, delta=1e-9)
        self.assertAlmostEqual(float(tracked["pump"][-1]), 9.999999717180685e-10, delta=1e-18)
        self.assertLessEqual(abs(float(tracked["pumpca"][-1])), 1e-15)
        self.assertAlmostEqual(float(tracked["CAM0"][-1]), 0.029932903407929307, delta=1e-9)
        self.assertAlmostEqual(float(tracked["CAM1C"][-1]), 5.981674114812124e-06, delta=1e-10)
        self.assertAlmostEqual(float(tracked["CAM1N"][-1]), 6.089851644030247e-05, delta=1e-9)
        self.assertAlmostEqual(float(tracked["CAM2N"][-1]), 2.0599083597580873e-07, delta=1e-10)

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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_stc_camonly")
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 45e-6, places=10)
        self.assertAlmostEqual(float(ion.CAM0.value[1].to_decimal(u.mM)), 0.03, places=6)
        self.assertAlmostEqual(float(ion.CAM1C.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.CAM1N.value[1].to_decimal(u.mM)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.dsq[1].to_decimal(u.um ** 2)), 400.0, places=6)

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
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_dsq"].to_decimal(u.um ** 2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_camonly_dsqvol"].to_decimal(u.um ** 2)),
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

        cell.init_state(); rcell = cell
        ion = rcell.get_ion("ca_stc_nocam")
        self.assertAlmostEqual(float(ion.Ci.value[1].to_decimal(u.mM)), 45e-6, places=10)
        self.assertAlmostEqual(float(ion.mg.value[1].to_decimal(u.mM)), 0.59, places=6)
        self.assertAlmostEqual(float(ion.pump.value[1].to_decimal(u.mol / u.cm ** 2)), 1e-9, places=15)
        self.assertAlmostEqual(float(ion.pumpca.value[1].to_decimal(u.mol / u.cm ** 2)), 0.0, places=15)
        self.assertAlmostEqual(float(ion.parea[1].to_decimal(u.um)), float(np.pi * 20.0), places=5)
        self.assertAlmostEqual(float(ion.dsq[1].to_decimal(u.um ** 2)), 400.0, places=6)

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
            "pump": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_pump"].to_decimal(u.mol / u.cm ** 2)),
            "pumpca": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_pumpca"].to_decimal(u.mol / u.cm ** 2)),
            "vrat": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_vrat"]),
            "parea": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_parea"].to_decimal(u.um)),
            "dsq": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_dsq"].to_decimal(u.um ** 2)),
            "dsqvol": np.asarray(result.traces["soma(0.5)_ca_stc_nocam_dsqvol"].to_decimal(u.um ** 2)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        total = tracked["pump"] + tracked["pumpca"]
        self.assertTrue(np.allclose(total, total[0], atol=1e-18))

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

        cell.init_state(); rcell = cell

        channel_layout = next(layout for layout in rcell.layouts if layout.kind == "channel:CaLVA_SU2015_DCN")
        ca_hva = rcell.get_ion("ca_hva")
        ca_lva = rcell.get_ion("ca_lva")
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertIs(ca_lva.channels["CaLVA_SU2015_DCN"], node)
        self.assertNotIn("CaLVA_SU2015_DCN", ca_hva.channels)
        self.assertEqual(rcell.runtime.bound_ion_keys[channel_layout.id], ("ca_lva",))
        with self.assertRaises(ValueError):
            rcell.get_ion("ca")

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
            cell.init_state(); rcell = cell

            _ = rcell.layouts
        self.assertIn("cannot mix classes", str(ctx.exception))

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
            braincell.mech.Channel("Kca3p1_MA2020", ion_names={"ca": "ca_hva"}),
        )

        cell.init_state(); rcell = cell

        layout = next(layout for layout in rcell.layouts if layout.kind == "channel:Kca3p1_MA2020")
        runtime = rcell.runtime
        node = rcell.get_runtime_node(layout.id)
        k_main = rcell.get_ion("k_main")
        ca_hva = rcell.get_ion("ca_hva")

        self.assertEqual(runtime.current_owner_keys[layout.id], "k_main")
        self.assertIn("Kca3p1_MA2020", k_main.channels)
        self.assertNotIn("Kca3p1_MA2020", ca_hva.channels)
        self.assertIsInstance(node, braincell.channel.Kca3p1_MA2020)

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
            braincell.mech.Channel("Kca3p1_MA2020", ion_names={"ca": "ca_hva"}),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(mechanism="Kca3p1_MA2020"),
            braincell.mech.CurrentProbe(ion="k_main"),
        )
        cell.init_state(); rcell = cell

        samples = rcell.sample_probes()
        runtime = rcell.runtime
        layout = next(layout for layout in rcell.layouts if layout.kind == "channel:Kca3p1_MA2020")
        node = rcell.get_runtime_node(layout.id)
        point_V = rcell._discretization_to_point(rcell.V.value)
        expected_mechanism = node.current(
            point_V,
            rcell.get_ion("k_main").pack_info(),
            rcell.get_ion("ca_hva").pack_info(),
        )[1]
        expected_total = rcell.get_ion("k_main").current(point_V, include_external=False)[1]

        self.assertEqual(runtime.bound_ion_keys[layout.id], ("k_main", "ca_hva"))
        self.assertEqual(samples["soma(0.5)_Kca3p1_MA2020_current"], expected_mechanism)
        self.assertEqual(samples["soma(0.5)_k_main_current"], expected_total)

    def test_channel_spec_ina_hh1952_builds_runtime_node_and_binds_to_na(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Na_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                temp=u.celsius2kelvin(36.0),
            ),
        )

        cell.init_state(); rcell = cell
        layout = rcell.layouts[0]
        node = rcell.get_runtime_node(layout.id)
        na = rcell.get_ion("na")

        self.assertIsInstance(node, braincell.channel.Na_HH1952)
        # Channels are now keyed on the declaration's instance name, which
        # defaults to the class name. Users can override with name=.
        self.assertIs(na.channels["Na_HH1952"], node)
        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm ** 2)), 12.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm ** 2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.V_sh[1].to_decimal(u.mV)), -50.0, places=12)

    def test_set_state_syncs_runtime_node_param_for_ina_hh1952(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "Na_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                temp=u.celsius2kelvin(36.0),
            ),
        )

        cell.init_state(); rcell = cell
        layout = rcell.layouts[0]
        rcell.set_state(layout.id, "g_max", 8.0 * (u.mS / u.cm ** 2))
        rcell.set_state(layout.id, "V_sh", -42.0 * u.mV)
        node = rcell.get_runtime_node(layout.id)

        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm ** 2)), 8.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm ** 2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.V_sh[1].to_decimal(u.mV)), -42.0, places=12)

    def test_unknown_channel_name_raises_key_error(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("__totally_unregistered__", g_max=12.0 * (u.mS / u.cm ** 2)),
        )

        with self.assertRaises(KeyError) as ctx:
            cell.init_state(); rcell = cell

            _ = rcell.layouts
        self.assertIn("__totally_unregistered__", str(ctx.exception))


class EvaluatePointClampsJitTest(unittest.TestCase):
    """Task 19: evaluate_point_clamps compiles under JAX without object dtype."""

    def test_evaluate_point_clamps_jit_compiles(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            CurrentClamp(start=0.0 * u.ms, durations=(2.0 * u.ms,), amplitudes=(0.1 * u.nA,)),
        )
        cell.init_state()
        runtime = cell.runtime
        compiled = jax.jit(lambda t: runtime.evaluate_point_clamps(t=t))
        out = compiled(0.5 * u.ms)
        self.assertEqual(out.mantissa.shape, (runtime.n_point,))


class DensityLayoutMaskingUnderJit(unittest.TestCase):
    """Task 19 (C5-adjacent): density mantissa is JAX-friendly, no object dtype."""

    def test_state_buffer_mantissa_sums_under_jit(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        mantissa = cell.runtime.state_buffers[(layout.id, "g_max")].mantissa
        total = float(jax.jit(lambda x: jnp.asarray(x).sum())(mantissa))
        self.assertGreater(total, 0.0)


class IsRootLevelRuntimeNodeUnknownClassTest(unittest.TestCase):
    """Task 18 (C6): unknown channel kinds raise rather than silently return False."""

    def test_unknown_channel_kind_raises_value_error(self) -> None:
        from braincell._compute.runtime import _is_root_level_runtime_node
        with self.assertRaises(ValueError) as ctx:
            _is_root_level_runtime_node("channel:__never_registered__")
        self.assertIn("__never_registered__", str(ctx.exception))


class CellLifecycleInlineTest(unittest.TestCase):
    """Task 14: init_state / reset own the install/uninstall work directly."""

    def test_init_state_installs_runtime_attributes_directly(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )
        cell.init_state()
        for name in ("_in_size", "_out_size", "ion_channels", "C"):
            self.assertTrue(hasattr(cell, name), f"Cell should have {name} after init_state.")
        self.assertEqual(cell._in_size, (cell.n_cv,))
        self.assertEqual(cell._out_size, (cell.n_cv,))

    def test_reset_clears_runtime_attributes(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )
        cell.init_state()
        cell.reset()
        for name in ("_in_size", "_out_size", "ion_channels", "C"):
            self.assertFalse(hasattr(cell, name), f"Cell should not have {name} after reset.")

    def test_init_reset_init_is_idempotent(self) -> None:
        cell = Cell(_build_tree())
        cell.init_state()
        layouts_a = cell.layouts
        cell.reset()
        cell.init_state()
        layouts_b = cell.layouts
        self.assertEqual(len(layouts_a), len(layouts_b))


# ---------------------------------------------------------------------------
# ClampActiveTable tests (absorbed from former clamp_table_test.py)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

from braincell._compute.runtime import (
    CLAMP_KINDS,
    ClampActiveTable,
    build_clamp_active_table,
)


@_dataclass
class _ClampStubLayout:
    target: str
    kind: str
    point_index: np.ndarray | None


@_dataclass
class _ClampStubCV:
    id: int
    area: object  # brainunit Quantity in cm^2


@_dataclass
class _ClampStubNodeTree:
    cv_to_mid_node_id: np.ndarray


def _clamp_node_tree(n_cv: int) -> _ClampStubNodeTree:
    return _ClampStubNodeTree(
        cv_to_mid_node_id=np.arange(n_cv, dtype=np.int32),
    )


def _clamp_cv(cv_id: int, area_cm2: float) -> _ClampStubCV:
    return _ClampStubCV(id=cv_id, area=area_cm2 * u.cm ** 2)


class TestBuildClampActiveTable(unittest.TestCase):

    def test_no_clamp_layouts_returns_none(self):
        layouts = (
            _ClampStubLayout(
                target="density",
                kind="IL",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        table = build_clamp_active_table(
            layouts=layouts,
            cvs=[_clamp_cv(0, 1e-6)],
            node_tree=_clamp_node_tree(1),
            n_point=1,
        )
        self.assertIsNone(table)

    def test_current_clamp_builds_table(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([1], dtype=np.int32),
            ),
        )
        table = build_clamp_active_table(
            layouts=layouts,
            cvs=[_clamp_cv(0, 1e-6), _clamp_cv(1, 2e-6)],
            node_tree=_clamp_node_tree(2),
            n_point=2,
        )
        self.assertIsInstance(table, ClampActiveTable)
        np.testing.assert_array_equal(table.ids, np.asarray([1], dtype=np.int32))
        np.testing.assert_allclose(table.area, np.asarray([2e-6]))

    def test_each_clamp_kind_is_recognized(self):
        self.assertEqual(
            CLAMP_KINDS, frozenset({"CurrentClamp", "SineClamp", "FunctionClamp"})
        )

    def test_ids_are_sorted_and_unique(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([3, 1], dtype=np.int32),
            ),
            _ClampStubLayout(
                target="point",
                kind="SineClamp",
                point_index=np.asarray([1, 2], dtype=np.int32),
            ),
        )
        table = build_clamp_active_table(
            layouts=layouts,
            cvs=[_clamp_cv(i, 1e-6 * (i + 1)) for i in range(4)],
            node_tree=_clamp_node_tree(4),
            n_point=4,
        )
        np.testing.assert_array_equal(
            table.ids, np.asarray([1, 2, 3], dtype=np.int32)
        )

    def test_zero_area_raises(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        with self.assertRaises(ValueError):
            build_clamp_active_table(
                layouts=layouts,
                cvs=[_clamp_cv(0, 0.0)],
                node_tree=_clamp_node_tree(1),
                n_point=1,
            )

    def test_endpoint_clamp_is_excluded_from_area_table(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="CurrentClamp",
                point_index=np.asarray([2], dtype=np.int32),
            ),
        )
        table = build_clamp_active_table(
            layouts=layouts,
            cvs=[_clamp_cv(0, 1e-6)],
            node_tree=_clamp_node_tree(1),
            n_point=3,
        )
        self.assertIsNone(table)

    def test_non_clamp_point_layout_ignored(self):
        layouts = (
            _ClampStubLayout(
                target="point",
                kind="Synapse",
                point_index=np.asarray([0], dtype=np.int32),
            ),
        )
        self.assertIsNone(
            build_clamp_active_table(
                layouts=layouts,
                cvs=[_clamp_cv(0, 1e-6)],
                node_tree=_clamp_node_tree(1),
                n_point=1,
            )
        )


# ---------------------------------------------------------------------------
# State-buffer storage tests (Task 12: Quantity(jnp.ndarray) rectangular).
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


class StateBufferStorageTest(unittest.TestCase):
    """Task 12: rectangular params live as Quantity(jnp.ndarray, unit)."""

    def test_density_buffer_is_quantity_backed_by_jax(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm ** 2)),
        )
        cell.init_state()
        layout = cell.layouts[0]
        buffer = cell.runtime.state_buffers[(layout.id, "g_max")]
        self.assertTrue(hasattr(buffer, "unit"))
        self.assertTrue(hasattr(buffer, "mantissa"))
        self.assertTrue(isinstance(buffer.mantissa, (np.ndarray, jnp.ndarray)))

    def test_set_state_broadcast_scalar_and_readback(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        cell.runtime.set_state(layout.id, "g_max", 7.5 * (u.mS / u.cm ** 2))
        new = cell.runtime.get_state(layout.id, "g_max")
        self.assertAlmostEqual(
            float(new[1].to_decimal(u.mS / u.cm ** 2)), 7.5, places=12
        )

    def test_set_state_shape_mismatch_raises(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        with self.assertRaises(ValueError):
            cell.runtime.set_state(
                layout.id, "g_max",
                u.Quantity(jnp.ones((99,)), u.mS / u.cm ** 2),
            )

    def test_no_object_dtype_in_density_buffer(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
        )
        cell.init_state()
        layout = cell.layouts[0]
        buffer = cell.runtime.state_buffers[(layout.id, "g_max")]
        self.assertNotEqual(buffer.mantissa.dtype, np.dtype("O"))


class RaggedCurrentClampBufferTest(unittest.TestCase):
    """Task 13: CurrentClamp durations/amplitudes packed into padded + mask."""

    def test_identical_lambda_bodies_produce_one_layout(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            FunctionClamp(fn=lambda t: 0.1 * u.nA, duration=3.0 * u.ms, start=0.0 * u.ms),
        )
        cell.place(
            at("soma", 0.75),
            FunctionClamp(fn=lambda t: 0.1 * u.nA, duration=3.0 * u.ms, start=0.0 * u.ms),
        )
        cell.init_state()
        fn_clamp_layouts = [layout for layout in cell.layouts if layout.kind == "FunctionClamp"]
        self.assertEqual(len(fn_clamp_layouts), 1)

    def test_three_clamps_with_varying_step_counts_pad_and_mask(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.25),
            CurrentClamp(start=0.0 * u.ms, durations=(2.0 * u.ms,), amplitudes=(0.1 * u.nA,)),
        )
        cell.place(
            at("soma", 0.5),
            CurrentClamp(
                start=0.0 * u.ms,
                durations=(1.0 * u.ms, 1.0 * u.ms),
                amplitudes=(0.1 * u.nA, 0.2 * u.nA),
            ),
        )
        cell.place(
            at("soma", 0.75),
            CurrentClamp(
                start=0.0 * u.ms,
                durations=(0.5 * u.ms, 0.5 * u.ms, 1.0 * u.ms),
                amplitudes=(0.1 * u.nA, 0.2 * u.nA, 0.3 * u.nA),
            ),
        )
        cell.init_state()

        current_clamp_layouts = [
            layout for layout in cell.layouts if layout.kind == "CurrentClamp"
        ]
        self.assertGreaterEqual(len(current_clamp_layouts), 1)
        for layout in current_clamp_layouts:
            dur = cell.runtime.state_buffers[(layout.id, "durations")]
            amp = cell.runtime.state_buffers[(layout.id, "amplitudes")]
            self.assertTrue(hasattr(dur, "unit"))
            self.assertEqual(dur.mantissa.ndim, 2)
            self.assertEqual(amp.mantissa.shape, dur.mantissa.shape)
            mask_key = (layout.id, "_mask_durations")
            self.assertIn(mask_key, cell.runtime.state_buffers)
            self.assertEqual(cell.runtime.state_buffers[mask_key].shape, dur.mantissa.shape)


class FnFingerprintWarnsOnOpaqueClosureTest(unittest.TestCase):
    """MED-08: fingerprinting a lambda with opaque closure emits RuntimeWarning."""

    def test_opaque_closure_emits_warning(self) -> None:
        import warnings

        from braincell._compute.runtime import _fn_fingerprint, _opaque_warned

        class _Opaque:
            __slots__ = ("x",)

            def __init__(self) -> None:
                self.x = object()

        opaque = _Opaque()

        def _make():
            return lambda t: opaque

        fn = _make()

        # Drop any prior entry for this call-site so the test is independent
        # of test ordering.
        _opaque_warned.discard((fn.__code__.co_filename, fn.__code__.co_firstlineno))

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _fn_fingerprint(fn)
        self.assertTrue(
            any(issubclass(w.category, RuntimeWarning) for w in captured),
            "expected RuntimeWarning for opaque closure cell",
        )


class CellRuntimeStateIsMutableTest(unittest.TestCase):
    """ARCH-07: CellRuntimeState is a mutable dataclass; callers must use plain setattr."""

    def test_cell_runtime_state_is_not_frozen(self) -> None:
        from braincell._compute.runtime import CellRuntimeState

        self.assertFalse(
            CellRuntimeState.__dataclass_params__.frozen,
            msg="CellRuntimeState must remain a mutable @dataclass",
        )

    def test_no_object_setattr_on_runtime_in_hot_paths(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        for rel in (
            "_multi_compartment/cell.py",
            "quad/_staggered.py",
        ):
            text = (root / rel).read_text()
            self.assertNotIn(
                "object.__setattr__(runtime",
                text,
                f"{rel} still uses object.__setattr__ on runtime",
            )
            self.assertNotIn(
                "object.__setattr__(self._runtime",
                text,
                f"{rel} still uses object.__setattr__ on self._runtime",
            )


class RuntimeModuleAllTest(unittest.TestCase):
    """Pre-split contract: braincell._compute.runtime.__all__ pins public names."""

    def test_all_contains_expected_names(self) -> None:
        from braincell._compute import runtime as rt

        expected = {
            "MechanismLayout",
            "ClampActiveTable",
            "build_clamp_active_table",
            "CellRuntimeState",
            "build_placeholder_ions",
            "clone_morpho",
            "cv_value_vector",
            "mechanism_signature",
        }
        actual = set(getattr(rt, "__all__", []))
        missing = expected - actual
        self.assertFalse(missing, msg=f"missing public symbols: {missing}")


class RuntimeSplitReexportTest(unittest.TestCase):
    """ARCH-02: runtime.py partitions into layouts / state / ions / bindings."""

    def test_mechanism_layout_lives_in_layouts(self) -> None:
        from braincell._compute import layouts, runtime

        self.assertIs(runtime.MechanismLayout, layouts.MechanismLayout)
        self.assertIs(runtime.ClampActiveTable, layouts.ClampActiveTable)
        self.assertIs(
            runtime.build_clamp_active_table,
            layouts.build_clamp_active_table,
        )

    def test_cell_runtime_state_lives_in_state(self) -> None:
        from braincell._compute import runtime, state

        self.assertIs(runtime.CellRuntimeState, state.CellRuntimeState)

    def test_ion_helpers_live_in_ions(self) -> None:
        from braincell._compute import ions, runtime

        self.assertIs(runtime._build_runtime_ions, ions._build_runtime_ions)
        self.assertIs(runtime._build_default_ions, ions._build_default_ions)

    def test_binding_helpers_live_in_bindings(self) -> None:
        from braincell._compute import bindings, runtime

        self.assertIs(
            runtime._resolve_channel_runtime_bindings,
            bindings._resolve_channel_runtime_bindings,
        )
        self.assertIs(
            runtime._instantiate_runtime_node,
            bindings._instantiate_runtime_node,
        )
        self.assertIs(
            runtime._BoundIonChannelRuntime,
            bindings._BoundIonChannelRuntime,
        )
