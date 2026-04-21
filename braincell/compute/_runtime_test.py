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
        rcell = cell.build()

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

        rcell = cell.build()

        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "sparse")
        self.assertEqual(layout.target, "point")
        self.assertEqual(layout.kind, "CurrentClamp")
        self.assertEqual(layout.n_active, 1)
        self.assertEqual(layout.point_index.tolist(), [1])
        self.assertIsNone(layout.point_mask)
        self.assertEqual(rcell.expected_state_shape(layout.id, "amplitudes"), (1,))
        self.assertEqual(rcell.get_state(layout.id, "amplitudes").shape, (1,))
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

        rcell = cell.build()

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

        rcell = cell.build()

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

        rcell = cell.build()

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

        rcell = cell.build()

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

        first = cell.build().layouts
        self.assertEqual(len(first), 0)

        cell.place(
            RootLocation(x=0.5),
            CurrentClamp.step(0.1 * u.nA, 2.0 * u.ms, delay=1.0 * u.ms),
        )
        second = cell.build().layouts

        self.assertIsNot(first, second)
        self.assertEqual(len(second), 1)

    def test_state_mutation_updates_buffer_without_rebuild(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp.step(0.1 * u.nA, 2.0 * u.ms, delay=1.0 * u.ms),
        )

        rcell = cell.build()

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
        rcell = cell.build()

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

        rcell = cell.build()

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
        rcell = cell.build()

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
        rcell = cell.build()

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
        rcell = cell.build()

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
        rcell = cell.build()

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
        rcell = cell.build()

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

        rcell = cell.build()

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

        rcell = cell.build()

        layout = rcell.layouts[0]
        rcell.set_state(layout.id, "g_max", 2.5 * (u.mS / u.cm ** 2))
        node = rcell.get_runtime_node(layout.id)

        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm ** 2)), 2.5, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm ** 2)), 0.0, places=12)

    def test_default_ions_are_available_with_global_shape(self) -> None:
        import braincell

        cell = Cell(_build_tree())

        rcell = cell.build()

        self.assertIsInstance(rcell.get_ion("na"), braincell.ion.SodiumFixed)
        self.assertIsInstance(rcell.get_ion("k"), braincell.ion.PotassiumFixed)
        self.assertIsInstance(rcell.get_ion("ca"), braincell.ion.CalciumFixed)
        self.assertEqual(rcell.get_ion("na").varshape, (5,))
        self.assertEqual(rcell.get_ion("k").varshape, (5,))
        self.assertEqual(rcell.get_ion("ca").varshape, (5,))

    def test_runtime_ions_expose_point_space_geometry_arrays(self) -> None:
        cell = Cell(_build_tree())

        rcell = cell.build()

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

        rcell = cell.build()

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

        rcell = cell.build()

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

        rcell = cell.build()

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

        rcell = cell.build()

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
            rcell = cell.build()

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

        rcell = cell.build()
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

        rcell = cell.build()
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
            rcell = cell.build()

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

        rcell = cell.build()

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
        rcell = cell.build()

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

        rcell = cell.build()
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

        rcell = cell.build()
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
            rcell = cell.build()

            _ = rcell.layouts
        self.assertIn("__totally_unregistered__", str(ctx.exception))


class _StubCell(braincell.HHTypedNeuron):
    """Minimal :class:`HHTypedNeuron` double for install/uninstall tests."""

    __module__ = "braincell.compute._runtime_test"

    def __init__(self, V_th=-55.0 * u.mV, n_cv: int = 3):
        from braincell.compute._runtime import build_placeholder_ions

        braincell.HHTypedNeuron.__init__(
            self, size=(1,), name="stub", **build_placeholder_ions()
        )
        self._V_th = V_th

        class _FakeCV:
            def __init__(self, cm):
                self.cm = cm

        self._cvs = tuple(_FakeCV(1.0 * u.uF / u.cm ** 2) for _ in range(n_cv))

    @property
    def V_th(self):
        return self._V_th

    @V_th.setter
    def V_th(self, value):
        self._V_th = value

    @property
    def cvs(self):
        return self._cvs

    @property
    def varshape(self):
        return (len(self._cvs),)


class TestInstallCellRuntime(unittest.TestCase):

    def _runtime_double(self, n_cv: int):
        from unittest.mock import MagicMock
        from braincell.compute._runtime import CellRuntimeState

        runtime = MagicMock(spec=CellRuntimeState)
        runtime.n_cv = n_cv
        runtime.ions = {}
        runtime.layouts = ()
        runtime.runtime_nodes = {}
        return runtime

    def test_install_returns_tuple_of_installed_plain_attr_names(self):
        from braincell.compute._runtime import install_cell_runtime

        cell = _StubCell(n_cv=4)
        installed = install_cell_runtime(cell, self._runtime_double(n_cv=4))

        self.assertIsInstance(installed, tuple)
        self.assertEqual(
            set(installed),
            {"_in_size", "_out_size", "ion_channels", "C"},
        )
        for name in installed:
            self.assertTrue(
                hasattr(cell, name),
                f"install_cell_runtime should have set attribute {name!r}.",
            )
        # V_th still reachable via the property setter (not in returned list).
        self.assertTrue(hasattr(cell, "V_th"))


class TestUninstallCellRuntime(unittest.TestCase):

    def _runtime_double(self, n_cv: int):
        from unittest.mock import MagicMock
        from braincell.compute._runtime import CellRuntimeState

        runtime = MagicMock(spec=CellRuntimeState)
        runtime.n_cv = n_cv
        runtime.ions = {}
        runtime.layouts = ()
        runtime.runtime_nodes = {}
        return runtime

    def test_uninstall_round_trip(self):
        from braincell._base import IonChannel
        from braincell.compute._runtime import (
            install_cell_runtime,
            uninstall_cell_runtime,
        )

        cell = _StubCell(n_cv=2)
        runtime = self._runtime_double(n_cv=2)

        installed = install_cell_runtime(cell, runtime)
        for name in installed:
            self.assertTrue(hasattr(cell, name))

        uninstall_cell_runtime(cell, installed)

        for name in installed:
            self.assertFalse(
                hasattr(cell, name),
                f"uninstall_cell_runtime should have removed {name!r}.",
            )
        self.assertEqual(
            dict(cell.nodes(IonChannel, allowed_hierarchy=(1, 1))),
            {},
            "No IonChannel nodes should remain after uninstall.",
        )
