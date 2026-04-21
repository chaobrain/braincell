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

        self.assertEqual(len(rcell.point_tree().points), 5)
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

        self.assertEqual(current_early.shape, (len(rcell.point_tree().points),))
        self.assertAlmostEqual(float(current_early[1].to_decimal(u.nA)), 0.7, places=6)
        self.assertAlmostEqual(float(current_early[0].to_decimal(u.nA)), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[1].to_decimal(u.nA)), 0.6, places=6)

    def test_probe_layouts_are_sparse_and_allocate_no_state_buffers(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="INa_HH1952", field="p"),
            braincell.mech.CurrentProbe(ion="na", mechanism="INa_HH1952"),
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
            ["soma(0.5)_INa_HH1952_current", "soma(0.5)_INa_HH1952_p", "soma(0.5)_v"],
        )

    def test_sample_probe_reads_voltage_and_channel_gate_state(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "INa_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                T=u.celsius2kelvin(36.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="INa_HH1952", field="p"),
        )
        cell.init_state(); rcell = cell

        samples = rcell.sample_probes()
        channel_layout = next(
            layout for layout in rcell.layouts
            if isinstance(rcell.runtime.get_layout_mechanism(layout.id), braincell.mech.Channel)
        )
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertEqual(samples["soma(0.5)_v"], rcell.V.value[0])
        self.assertEqual(samples["soma(0.5)_INa_HH1952_p"], node.p.value[1])
        self.assertEqual(rcell.sample_probe("soma(0.5)_INa_HH1952_p"), node.p.value[1])

    def test_sample_probe_reads_mechanism_and_total_ion_current(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "IK_Kv_test",
                g_max=0.1 * (u.mS / u.cm ** 2),
                v12=25.0 * u.mV,
                q=9.0,
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(ion="k", mechanism="IK_Kv_test"),
            braincell.mech.CurrentProbe(ion="k"),
        )
        cell.init_state(); rcell = cell

        samples = rcell.sample_probes()
        ion = rcell.get_ion("k")
        node = ion.channels["IK_Kv_test"]
        point_V = rcell._cv_to_point(rcell.V.value)
        expected_mechanism = node.current(point_V, ion.pack_info())[1]
        expected_total = ion.current(point_V, include_external=False)[1]

        self.assertEqual(samples["soma(0.5)_IK_Kv_test_current"], expected_mechanism)
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
        point_V = rcell._cv_to_point(rcell.V.value)
        expected_current = node.current(point_V)[1]

        self.assertEqual(samples["soma(0.5)_IL_current"], expected_current)

    def test_sample_probe_rejects_non_state_field_and_unknown_mechanism(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "INa_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                T=u.celsius2kelvin(36.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="INa_HH1952", field="g_max"),
            braincell.mech.MechanismProbe(mechanism="missing", field="p"),
        )
        cell.init_state(); rcell = cell

        with self.assertRaises(ValueError):
            rcell.sample_probe("soma(0.5)_INa_HH1952_g_max")
        with self.assertRaises(KeyError):
            rcell.sample_probe("soma(0.5)_missing_p")

    def test_sample_probes_requires_unique_names(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "INa_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                T=u.celsius2kelvin(36.0),
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(name="dup"),
            braincell.mech.MechanismProbe(name="dup", mechanism="INa_HH1952", field="p"),
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
            braincell.mech.Channel("INa_HH1952", g_max=12.0 * (u.mS / u.cm ** 2), ion_name="na_soma"),
        )

        cell.init_state(); rcell = cell

        channel_layout = next(layout for layout in rcell.layouts if layout.kind == "channel:INa_HH1952")
        na_soma = rcell.get_ion("na_soma")
        na_dend = rcell.get_ion("na_dend")
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertIs(na_soma.channels["INa_HH1952"], node)
        self.assertNotIn("INa_HH1952", na_dend.channels)
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
            braincell.mech.Channel("ICaT_HM1992"),
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
                "ICaT_HM1992",
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
            braincell.mech.Channel("IKca3_1_Ma2020", ion_names={"ca": "ca_hva"}),
        )

        cell.init_state(); rcell = cell

        layout = next(layout for layout in rcell.layouts if layout.kind == "channel:IKca3_1_Ma2020")
        runtime = rcell.runtime
        node = rcell.get_runtime_node(layout.id)
        k_main = rcell.get_ion("k_main")
        ca_hva = rcell.get_ion("ca_hva")

        self.assertEqual(runtime.current_owner_keys[layout.id], "k_main")
        self.assertIn("IKca3_1_Ma2020", k_main.channels)
        self.assertNotIn("IKca3_1_Ma2020", ca_hva.channels)
        self.assertIsInstance(node, braincell.channel.IKca3_1_Ma2020)

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
            braincell.mech.Channel("IKca3_1_Ma2020", ion_names={"ca": "ca_hva"}),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(mechanism="IKca3_1_Ma2020"),
            braincell.mech.CurrentProbe(ion="k_main"),
        )
        cell.init_state(); rcell = cell

        samples = rcell.sample_probes()
        runtime = rcell.runtime
        layout = next(layout for layout in rcell.layouts if layout.kind == "channel:IKca3_1_Ma2020")
        node = rcell.get_runtime_node(layout.id)
        point_V = rcell._cv_to_point(rcell.V.value)
        expected_mechanism = node.current(
            point_V,
            rcell.get_ion("k_main").pack_info(),
            rcell.get_ion("ca_hva").pack_info(),
        )[1]
        expected_total = rcell.get_ion("k_main").current(point_V, include_external=False)[1]

        self.assertEqual(runtime.bound_ion_keys[layout.id], ("k_main", "ca_hva"))
        self.assertEqual(samples["soma(0.5)_IKca3_1_Ma2020_current"], expected_mechanism)
        self.assertEqual(samples["soma(0.5)_k_main_current"], expected_total)

    def test_channel_spec_ina_hh1952_builds_runtime_node_and_binds_to_na(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "INa_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                T=u.celsius2kelvin(36.0),
            ),
        )

        cell.init_state(); rcell = cell
        layout = rcell.layouts[0]
        node = rcell.get_runtime_node(layout.id)
        na = rcell.get_ion("na")

        self.assertIsInstance(node, braincell.channel.INa_HH1952)
        # Channels are now keyed on the declaration's instance name, which
        # defaults to the class name. Users can override with name=.
        self.assertIs(na.channels["INa_HH1952"], node)
        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm ** 2)), 12.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm ** 2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.V_sh[1].to_decimal(u.mV)), -50.0, places=12)

    def test_set_state_syncs_runtime_node_param_for_ina_hh1952(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "INa_HH1952",
                g_max=12.0 * (u.mS / u.cm ** 2),
                V_sh=-50.0 * u.mV,
                T=u.celsius2kelvin(36.0),
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
import numpy as np

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
class _ClampStubPointTree:
    cv_midpoint_point_id: np.ndarray


def _clamp_point_tree(n_cv: int) -> _ClampStubPointTree:
    return _ClampStubPointTree(
        cv_midpoint_point_id=np.arange(n_cv, dtype=np.int32),
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
            point_tree=_clamp_point_tree(1),
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
            point_tree=_clamp_point_tree(2),
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
            point_tree=_clamp_point_tree(4),
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
                point_tree=_clamp_point_tree(1),
                n_point=1,
            )

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
                point_tree=_clamp_point_tree(1),
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
