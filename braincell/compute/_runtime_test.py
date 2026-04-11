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

from braincell import Branch, CVPerBranch, Cell, CurrentClamp, DensityMechanism, FunctionClamp, Morphology, SineClamp
from braincell.filter import BranchSlice, RootLocation


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
            DensityMechanism(channel_type="leaky", params=(("g_max", 4.0 * (u.mS / u.cm**2)),)),
        )

        self.assertEqual(cell.n_cv, 2)
        self.assertEqual(len(cell.point_tree().points), 5)
        self.assertEqual(len(cell.layouts), 1)
        layout = cell.layouts[0]
        self.assertEqual(layout.layout, "dense")
        self.assertEqual(layout.target, "density")
        self.assertEqual(layout.kind, "channel:leaky")
        self.assertEqual(layout.n_active, 2)
        self.assertEqual(layout.source_cv_ids, (0, 1))
        self.assertEqual(layout.point_index.tolist(), [1, 3])
        self.assertEqual(cell.expected_state_shape(layout.id, "g_max"), (5,))
        self.assertEqual(cell.voltage_shape, (5,))
        self.assertEqual(cell.get_state(layout.id, "g_max").shape, (5,))
        self.assertTrue(all(value == 4.0 * (u.mS / u.cm**2) for value in cell.get_state(layout.id, "g_max")))
        self.assertEqual(tuple(layout.id for layout in cell.get_point_layouts(0)), ())
        self.assertEqual(tuple(layout.id for layout in cell.get_point_layouts(1)), (layout.id,))
        self.assertEqual(tuple(layout.id for layout in cell.get_point_layouts(3)), (layout.id,))

    def test_point_mechanism_builds_sparse_layout_with_local_shape(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(amplitude=0.1 * u.nA, delay=1.0 * u.ms, duration=2.0 * u.ms),
        )

        self.assertEqual(len(cell.layouts), 1)
        layout = cell.layouts[0]
        self.assertEqual(layout.layout, "sparse")
        self.assertEqual(layout.target, "point")
        self.assertEqual(layout.kind, "current_clamp")
        self.assertEqual(layout.n_active, 1)
        self.assertEqual(layout.point_index.tolist(), [1])
        self.assertIsNone(layout.point_mask)
        self.assertEqual(cell.expected_state_shape(layout.id, "amplitudes"), (1,))
        self.assertEqual(cell.get_state(layout.id, "amplitudes").shape, (1,))
        self.assertEqual(tuple(item.to_decimal(u.nA) for item in cell.get_state(layout.id, "amplitudes")[0]), (0.1,))
        self.assertEqual(tuple(item.to_decimal(u.ms) for item in cell.get_state(layout.id, "durations")[0]), (2.0,))
        self.assertEqual(cell.get_state(layout.id, "start")[0], 1.0 * u.ms)
        self.assertEqual(tuple(layout.id for layout in cell.get_point_layouts(1)), (layout.id,))
        self.assertEqual(cell.get_point_layouts(1), (layout,))

    def test_channel_spec_builds_dense_layout_with_global_shape(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        self.assertEqual(len(cell.layouts), 1)
        layout = cell.layouts[0]
        self.assertEqual(layout.layout, "dense")
        self.assertEqual(layout.kind, "channel:IL")
        self.assertEqual(cell.expected_state_shape(layout.id, "g_max"), (5,))
        self.assertEqual(cell.expected_state_shape(layout.id, "E"), (5,))
        self.assertEqual(cell.get_point_state(1)[layout.id]["g_max"], 4.0 * (u.mS / u.cm**2))
        self.assertEqual(cell.get_point_state(3)[layout.id]["E"], -68.0 * u.mV)
        node = cell.get_runtime_node(layout.id)
        self.assertIsInstance(node, braincell.channel.IL)
        self.assertEqual(node.varshape, (5,))
        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm**2)), 4.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.E[1].to_decimal(u.mV)), -68.0, places=12)

    def test_named_channel_spec_merges_across_regions_when_identity_matches(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=0, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_main", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_main", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        self.assertEqual(len(cell.layouts), 1)
        self.assertEqual(cell.layouts[0].point_index.tolist(), [1, 3])

    def test_same_class_different_names_build_distinct_layouts(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", name="leak_a", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
            braincell.mech.Channel("IL", name="leak_b", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        self.assertEqual(len(cell.layouts), 2)
        self.assertEqual({layout.kind for layout in cell.layouts}, {"channel:IL"})
        self.assertTrue(all(layout.point_index.tolist() == [1, 3] for layout in cell.layouts))

    def test_runtime_state_keeps_dense_and_sparse_layouts_together(self) -> None:
        cell = Cell(_build_tree(), cv_policy=CVPerBranch())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            DensityMechanism(channel_type="leaky", params=(("g_max", 4.0 * (u.mS / u.cm**2)),)),
        )
        clamp = CurrentClamp(amplitude=0.1 * u.nA, delay=1.0 * u.ms, duration=2.0 * u.ms)
        cell.place(RootLocation(x=0.5), clamp)

        self.assertEqual(len(cell.layouts), 2)
        dense = next(layout for layout in cell.layouts if layout.layout == "dense")
        sparse = next(layout for layout in cell.layouts if layout.layout == "sparse")
        self.assertEqual(tuple(layout.id for layout in cell.get_point_layouts(1)), (dense.id, sparse.id))
        self.assertEqual(tuple(layout.id for layout in cell.get_point_layouts(3)), (dense.id,))
        self.assertEqual(tuple(layout.id for layout in cell.get_cv_layouts(0)), (dense.id, sparse.id))
        self.assertEqual(tuple(layout.id for layout in cell.get_cv_layouts(1)), (dense.id,))
        point_state = cell.get_point_state(1)
        self.assertEqual(point_state[dense.id]["g_max"], 4.0 * (u.mS / u.cm**2))
        self.assertEqual(tuple(item.to_decimal(u.nA) for item in point_state[sparse.id]["amplitudes"]), (0.1,))
        self.assertEqual(cell.get_cv_state(0)[dense.id]["g_max"], 4.0 * (u.mS / u.cm**2))
        self.assertEqual({name for name in ("na", "k", "ca")}, {"na", "k", "ca"})

    def test_runtime_state_cache_is_invalidated_by_new_mapping(self) -> None:
        cell = Cell(_build_tree())

        first = cell.layouts
        self.assertEqual(len(first), 0)

        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(amplitude=0.1 * u.nA, delay=1.0 * u.ms, duration=2.0 * u.ms),
        )
        second = cell.layouts

        self.assertIsNot(first, second)
        self.assertEqual(len(second), 1)

    def test_state_mutation_updates_buffer_without_rebuild(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(amplitude=0.1 * u.nA, delay=1.0 * u.ms, duration=2.0 * u.ms),
        )

        layout = cell.layouts[0]
        cell.set_state(layout.id, "amplitudes", (0.25 * u.nA, 0.05 * u.nA))
        cell.set_state(layout.id, "durations", (1.5 * u.ms, 2.5 * u.ms))

        self.assertEqual(tuple(item.to_decimal(u.nA) for item in cell.get_state(layout.id, "amplitudes")[0]), (0.25, 0.05))
        self.assertEqual(tuple(item.to_decimal(u.ms) for item in cell.get_point_state(1)[layout.id]["durations"]), (1.5, 2.5))

    def test_runtime_evaluates_step_sine_and_function_clamps_on_target_points(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(start=0.0 * u.ms, durations=(2.0 * u.ms, 2.0 * u.ms), amplitudes=(0.0 * u.nA, 0.3 * u.nA)),
            SineClamp(amplitude=0.2 * u.nA, frequency=500.0 * u.Hz, offset=0.1 * u.nA, duration=4.0 * u.ms),
            FunctionClamp(fn=lambda local_t: 0.4 * u.nA if local_t < 1.0 * u.ms else 0.0 * u.nA, duration=3.0 * u.ms),
        )
        runtime = cell._ensure_runtime_compiled()

        current_early = runtime.evaluate_point_clamps(t=0.5 * u.ms)
        current_late = runtime.evaluate_point_clamps(t=2.5 * u.ms)

        self.assertEqual(current_early.shape, (len(cell.point_tree().points),))
        self.assertAlmostEqual(float(current_early[1].to_decimal(u.nA)), 0.7, places=6)
        self.assertAlmostEqual(float(current_early[0].to_decimal(u.nA)), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[1].to_decimal(u.nA)), 0.6, places=6)

    def test_density_mechanism_leaky_builds_runtime_il_node(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            DensityMechanism(
                channel_type="leaky",
                params=(
                    ("g_max", 4.0 * (u.mS / u.cm**2)),
                    ("E", -69.0 * u.mV),
                ),
            ),
        )

        layout = cell.layouts[0]
        node = cell.get_runtime_node(layout.id)

        self.assertIsInstance(node, braincell.channel.IL)
        self.assertAlmostEqual(float(node.g_max[3].to_decimal(u.mS / u.cm**2)), 4.0, places=12)
        self.assertAlmostEqual(float(node.g_max[2].to_decimal(u.mS / u.cm**2)), 0.0, places=12)

    def test_set_state_syncs_runtime_node_param(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV),
        )

        layout = cell.layouts[0]
        cell.set_state(layout.id, "g_max", 2.5 * (u.mS / u.cm**2))
        node = cell.get_runtime_node(layout.id)

        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm**2)), 2.5, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)

    def test_default_ions_are_available_with_global_shape(self) -> None:
        import braincell

        cell = Cell(_build_tree())

        self.assertIsInstance(cell.get_ion("na"), braincell.ion.SodiumFixed)
        self.assertIsInstance(cell.get_ion("k"), braincell.ion.PotassiumFixed)
        self.assertIsInstance(cell.get_ion("ca"), braincell.ion.CalciumFixed)
        self.assertEqual(cell.get_ion("na").varshape, (5,))
        self.assertEqual(cell.get_ion("k").varshape, (5,))
        self.assertEqual(cell.get_ion("ca").varshape, (5,))

    def test_channel_spec_ina_hh1952_builds_runtime_node_and_binds_to_na(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "INa_HH1952",
                g_max=12.0 * (u.mS / u.cm**2),
                V_sh=-50.0 * u.mV,
                T=u.celsius2kelvin(36.0),
            ),
        )

        layout = cell.layouts[0]
        node = cell.get_runtime_node(layout.id)
        na = cell.get_ion("na")

        self.assertIsInstance(node, braincell.channel.INa_HH1952)
        self.assertIs(na.channels["INa"], node)
        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm**2)), 12.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.V_sh[1].to_decimal(u.mV)), -50.0, places=12)

    def test_set_state_syncs_runtime_node_param_for_ina_hh1952(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "INa_HH1952",
                g_max=12.0 * (u.mS / u.cm**2),
                V_sh=-50.0 * u.mV,
                T=u.celsius2kelvin(36.0),
            ),
        )

        layout = cell.layouts[0]
        cell.set_state(layout.id, "g_max", 8.0 * (u.mS / u.cm**2))
        cell.set_state(layout.id, "V_sh", -42.0 * u.mV)
        node = cell.get_runtime_node(layout.id)

        self.assertAlmostEqual(float(node.g_max[1].to_decimal(u.mS / u.cm**2)), 8.0, places=12)
        self.assertAlmostEqual(float(node.g_max[0].to_decimal(u.mS / u.cm**2)), 0.0, places=12)
        self.assertAlmostEqual(float(node.V_sh[1].to_decimal(u.mV)), -42.0, places=12)

    def test_unsupported_runtime_channel_raises(self) -> None:
        import braincell

        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("IK_Kv_test", g_max=12.0 * (u.mS / u.cm**2)),
        )

        with self.assertRaises(NotImplementedError):
            _ = cell.layouts
