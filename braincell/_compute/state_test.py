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

"""Tests for :mod:`braincell._compute.state` and the cell lifecycle that builds it."""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import (
    CVPerBranch,
    Cell,
    CurrentClamp,
)
from braincell.filter import BranchSlice, RootLocation, at
from ._testing import _build_tree


class CellRuntimeStateTest(unittest.TestCase):
    """Runtime state construction, mutation, and probe sampling."""

    def test_synapse_mechanism_builds_sparse_runtime_node_and_event_buffer(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            braincell.mech.Synapse(
                "ExpSyn",
                tau=2.0 * u.ms,
                e=0.0 * u.mV,
                name="exp_soma",
            ),
        )

        cell.init_state()
        rcell = cell

        self.assertEqual(len(rcell.layouts), 1)
        layout = rcell.layouts[0]
        self.assertEqual(layout.layout, "sparse")
        self.assertEqual(layout.target, "point")
        self.assertEqual(layout.kind, "synapse:ExpSyn")
        self.assertEqual(layout.point_index.tolist(), [1])
        node = rcell.get_runtime_node(layout.id)
        self.assertIsInstance(node, braincell.synapse.ExpSyn)
        self.assertEqual(node.varshape, (1,))
        np.testing.assert_allclose(rcell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [0.0])

    def test_synapse_event_buffer_is_private_runtime_state(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            braincell.mech.Synapse(
                "ExpSyn",
                tau=2.0 * u.ms,
                e=0.0 * u.mV,
                name="exp_soma",
            ),
        )

        cell.init_state()
        rcell = cell
        layout = rcell.layouts[0]
        with self.assertRaises(KeyError):
            rcell.get_state(layout.id, "pre_spike")
        rcell.runtime.event_buffers[layout.id].value = np.asarray([1.0]) * u.uS
        np.testing.assert_allclose(rcell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [1.0])

    def test_rebuild_after_place_produces_new_runtime(self) -> None:
        cell = Cell(_build_tree())

        cell.init_state()
        first = cell.layouts
        self.assertEqual(len(first), 0)

        cell.reset()
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
        )
        cell.init_state()
        second = cell.layouts

        self.assertIsNot(first, second)
        self.assertEqual(len(second), 1)

    def test_clamp_routing_area_matches_the_cv_that_owns_each_midpoint(self) -> None:
        """The routing table reads the runtime point-area vector, not ``cv.area``.

        ``build_clamp_routing_table`` no longer walks the CV sequence; it slices
        the ``point_area`` vector the runtime already built from each node's
        first role. That is only correct because a CV midpoint reports its own
        CV as ``roles[0]``, which this pins against a real discretization.
        """
        cell = Cell(_build_tree(), cv_policy=CVPerBranch(cv_per_branch=3))
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
        )
        cell.init_state()

        midpoint_ids = np.asarray(cell.runtime.node_tree.cv_to_mid_node_id, dtype=np.int64)
        np.testing.assert_allclose(
            np.asarray(cell.runtime.point_area.to_decimal(u.cm**2))[midpoint_ids],
            [float(np.asarray(cv.area.to_decimal(u.cm**2))) for cv in cell.cvs],
            rtol=0.0,
        )
        table = cell.runtime.clamp_routing_table
        clamped_cv = int(np.flatnonzero(midpoint_ids == table.midpoint_ids[0])[0])
        np.testing.assert_allclose(
            table.midpoint_area,
            [float(np.asarray(cell.cvs[clamped_cv].area.to_decimal(u.cm**2)))],
            rtol=0.0,
        )

    def test_state_mutation_updates_buffer_without_rebuild(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
        )

        cell.init_state()
        rcell = cell

        layout = rcell.layouts[0]
        rcell.set_state(layout.id, "amplitudes", (0.25 * u.nA, 0.05 * u.nA))
        rcell.set_state(layout.id, "durations", (1.5 * u.ms, 2.5 * u.ms))

        self.assertEqual(
            tuple(item.to_decimal(u.nA) for item in rcell.get_state(layout.id, "amplitudes")[0]), (0.25, 0.05)
        )
        self.assertEqual(
            tuple(item.to_decimal(u.ms) for item in rcell.get_point_state(1)[layout.id]["durations"]), (1.5, 2.5)
        )

    def test_sample_probe_reads_voltage_and_channel_gate_state(self) -> None:
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
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(),
            braincell.mech.MechanismProbe(mechanism="Na_HH1952", field="p"),
        )
        cell.init_state()
        rcell = cell

        samples = rcell.sample_probes()
        channel_layout = next(
            layout
            for layout in rcell.layouts
            if isinstance(rcell.runtime.get_layout_mechanism(layout.id), braincell.mech.Channel)
        )
        node = rcell.get_runtime_node(channel_layout.id)

        self.assertEqual(samples["soma(0.5)_v"], rcell.V.value[..., 0])
        self.assertEqual(samples["soma(0.5)_Na_HH1952_p"], node.p.value[..., 1])
        self.assertEqual(rcell.sample_probe("soma(0.5)_Na_HH1952_p"), node.p.value[..., 1])

    def test_sample_probe_reads_mechanism_and_total_ion_current(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "K_Kv_test",
                g_max=0.1 * (u.mS / u.cm**2),
                v12=25.0 * u.mV,
                q=9.0,
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(ion="k", mechanism="K_Kv_test"),
            braincell.mech.CurrentProbe(ion="k"),
        )
        cell.init_state()
        rcell = cell

        samples = rcell.sample_probes()
        ion = rcell.get_ion("k")
        node = ion.channels["K_Kv_test"]
        point_V = rcell._cv_to_point(rcell.V.value)
        expected_mechanism = node.current(point_V, ion.pack_info())[..., 1]
        expected_total = ion.current(point_V, include_external=False)[..., 1]

        self.assertEqual(samples["soma(0.5)_K_Kv_test_current"], expected_mechanism)
        self.assertEqual(samples["soma(0.5)_k_current"], expected_total)

    def test_sample_probe_reads_pure_channel_current_without_ion_selector(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel(
                "IL",
                g_max=0.1 * (u.mS / u.cm**2),
                E=-68.0 * u.mV,
            ),
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentProbe(mechanism="IL"),
        )
        cell.init_state()
        rcell = cell

        samples = rcell.sample_probes()
        channel_layout = next(
            layout
            for layout in rcell.layouts
            if isinstance(rcell.runtime.get_layout_mechanism(layout.id), braincell.mech.Channel)
        )
        node = rcell.get_runtime_node(channel_layout.id)
        point_V = rcell._cv_to_point(rcell.V.value)
        expected_current = node.current(point_V)[..., 1]

        self.assertEqual(samples["soma(0.5)_IL_current"], expected_current)

    def test_sample_probe_reads_plain_field_and_rejects_unknown_mechanism(self) -> None:
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
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(mechanism="Na_HH1952", field="g_max"),
            braincell.mech.MechanismProbe(mechanism="missing", field="p"),
        )
        cell.init_state()
        rcell = cell

        self.assertEqual(
            rcell.sample_probe("soma(0.5)_Na_HH1952_g_max"),
            rcell.get_ion("na").channels["Na_HH1952"].g_max[..., 1],
        )
        with self.assertRaises(KeyError):
            rcell.sample_probe("soma(0.5)_missing_p")

    def test_sample_probes_requires_unique_names(self) -> None:
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
        cell.place(
            at("soma", 0.5),
            braincell.mech.StateProbe(name="dup"),
            braincell.mech.MechanismProbe(name="dup", mechanism="Na_HH1952", field="p"),
        )
        cell.init_state()
        rcell = cell

        with self.assertRaises(ValueError):
            rcell.sample_probes()


class CellLifecycleInlineTest(unittest.TestCase):
    """Task 14: init_state / reset own the install/uninstall work directly."""

    def test_init_state_installs_runtime_attributes_directly(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm**2)),
        )
        cell.init_state()
        for name in ("_in_size", "_out_size", "ion_channels", "C"):
            self.assertTrue(hasattr(cell, name), f"Cell should have {name} after init_state.")
        self.assertEqual(cell._in_size, cell.pop_size + (cell.n_cv,))
        self.assertEqual(cell._out_size, cell.pop_size + (cell.n_cv,))

    def test_reset_clears_runtime_attributes(self) -> None:
        cell = Cell(_build_tree())
        cell.paint(
            BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
            braincell.mech.Channel("leaky", g_max=4.0 * (u.mS / u.cm**2)),
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


class PopulationResponseHeterogeneityTest(unittest.TestCase):
    def test_population_cells_can_have_different_current_clamp_responses(self) -> None:
        cell = Cell(
            _build_tree(),
            pop_size=(2,),
            solver="staggered",
            V_init=-60.0 * u.mV,
        )
        cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
        cell.place(
            at("soma", 0.5),
            CurrentClamp(delay=0.1 * u.ms, durations=0.8 * u.ms, amplitudes=u.Quantity(jnp.asarray([0.0, 0.2]), u.nA)),
        )
        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        trace = result.traces["v"]
        self.assertEqual(trace.shape[1], 2)
        peak0 = float(jnp.max(trace[:, 0].to_decimal(u.mV)))
        peak1 = float(jnp.max(trace[:, 1].to_decimal(u.mV)))
        self.assertGreater(peak1, peak0)

    def test_two_dimensional_population_can_run_with_population_specific_clamp(self) -> None:
        cell = Cell(
            _build_tree(),
            pop_size=(2, 2),
            solver="staggered",
            V_init=-60.0 * u.mV,
        )
        amp = u.Quantity(jnp.asarray([[0.00, 0.03], [0.06, 0.12]]), u.nA)
        cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
        cell.place(
            at("soma", 0.5),
            CurrentClamp(delay=0.1 * u.ms, durations=0.8 * u.ms, amplitudes=amp),
        )
        result = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        trace = result.traces["v"]
        self.assertEqual(trace.shape, (20, 2, 2))

    def test_population_cells_can_have_different_current_clamp_delays(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,))
        cell.place(
            at("soma", 0.5),
            CurrentClamp(
                delay=u.Quantity(np.asarray([0.1, 0.3]), u.ms),
                durations=0.1 * u.ms,
                amplitudes=0.2 * u.nA,
            ),
        )
        cell.init_state()
        layout = next(layout for layout in cell.layouts if layout.kind == "CurrentClamp")
        delay = cell.runtime.state_buffers[(layout.id, "delay")]
        self.assertEqual(delay.mantissa.shape, (2, 1))

        current_early = cell.runtime.evaluate_point_clamps(t=0.15 * u.ms).to_decimal(u.nA)
        current_late = cell.runtime.evaluate_point_clamps(t=0.35 * u.ms).to_decimal(u.nA)
        self.assertAlmostEqual(float(current_early[0, 1]), 0.2, places=6)
        self.assertAlmostEqual(float(current_early[1, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[1, 1]), 0.2, places=6)

    def test_population_delay_works_with_multistep_current_clamp(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,))
        cell.place(
            at("soma", 0.5),
            CurrentClamp(
                delay=u.Quantity(np.asarray([0.1, 0.3]), u.ms),
                durations=(0.1 * u.ms, 0.1 * u.ms),
                amplitudes=(0.2 * u.nA, 0.0 * u.nA),
            ),
        )
        cell.init_state()

        current0 = cell.runtime.evaluate_point_clamps(t=0.15 * u.ms).to_decimal(u.nA)
        current1 = cell.runtime.evaluate_point_clamps(t=0.35 * u.ms).to_decimal(u.nA)
        self.assertAlmostEqual(float(current0[0, 1]), 0.2, places=6)
        self.assertAlmostEqual(float(current0[1, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(current1[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(current1[1, 1]), 0.2, places=6)

    def test_current_clamp_delay_uses_active_point_axis(self) -> None:
        cell = Cell(_build_tree())
        cell.place(
            RootLocation(x=0.5) | at(1, 0.5),
            CurrentClamp(
                delay=u.Quantity(np.asarray([0.1, 0.3]), u.ms),
                durations=0.1 * u.ms,
                amplitudes=0.2 * u.nA,
            ),
        )
        cell.init_state()
        layout = next(layout for layout in cell.layouts if layout.kind == "CurrentClamp")
        self.assertEqual(layout.n_active, 2)

        current_early = cell.runtime.evaluate_point_clamps(t=0.15 * u.ms).to_decimal(u.nA)
        current_late = cell.runtime.evaluate_point_clamps(t=0.35 * u.ms).to_decimal(u.nA)
        first_point, second_point = layout.point_index.tolist()
        self.assertAlmostEqual(float(current_early[0, first_point]), 0.2, places=6)
        self.assertAlmostEqual(float(current_early[0, second_point]), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[0, first_point]), 0.0, places=6)
        self.assertAlmostEqual(float(current_late[0, second_point]), 0.2, places=6)

    def test_unbroadcastable_current_clamp_delay_raises(self) -> None:
        cell = Cell(_build_tree(), pop_size=(2,))
        cell.place(
            at("soma", 0.5),
            CurrentClamp(
                delay=u.Quantity(np.asarray([0.1, 0.2, 0.3]), u.ms),
                durations=0.1 * u.ms,
                amplitudes=0.2 * u.nA,
            ),
        )
        with self.assertRaises(ValueError):
            cell.init_state()


class PointSynapseRuntimeTest(unittest.TestCase):
    def test_synapse_compute_derivative_populates_ode_state(self) -> None:
        cell = Cell(
            _build_tree(),
            solver="staggered",
            V_init=-60.0 * u.mV,
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.Synapse(
                "ExpSyn",
                tau=2.0 * u.ms,
                e=0.0 * u.mV,
                name="exp_soma",
            ),
        )
        cell.init_state()
        runtime = cell.runtime
        layout = next(layout for layout in runtime.layouts if layout.kind == "synapse:ExpSyn")
        node = runtime.get_runtime_node(layout.id)

        node.g.value = np.asarray([1.0]) * u.uS
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            cell.compute_derivative()

        self.assertIsNotNone(node.g.derivative)
        self.assertLess(float(node.g.derivative[0].to_decimal(u.uS / u.ms)), 0.0)

    def test_unregistered_synapse_type_fails_at_build_with_a_suggestion(self) -> None:
        # ``mech.Synapse`` defers name resolution to Cell build, like
        # ``mech.Channel``/``mech.Ion``. The retired receptor names are
        # ordinary unknown names on that path, not a special case.
        for model in ("AMPA", "GABAa", "NMDA", "ExpSynn"):
            cell = Cell(_build_tree(), solver="staggered", V_init=-60.0 * u.mV)
            cell.place(at("soma", 0.5), braincell.mech.Synapse(model, name="syn"))
            with self.assertRaisesRegex(KeyError, "No 'synapse' mechanism registered"):
                cell.init_state()

    def test_expsyn_drive_jumps_g_and_then_decays(self) -> None:
        cell = Cell(
            _build_tree(),
            solver="staggered",
            V_init=-60.0 * u.mV,
        )
        cell.place(at("soma", 0.5), braincell.mech.MechanismProbe(name="g", mechanism="exp_syn", field="g"))
        cell.place(at("soma", 0.5), braincell.mech.CurrentProbe(name="i_syn", mechanism="exp_syn"))
        cell.place(
            at("soma", 0.5),
            braincell.mech.Synapse(
                "ExpSyn",
                tau=2.0 * u.ms,
                e=0.0 * u.mV,
                name="exp_syn",
            ),
        )
        cell.init_state()
        runtime = cell.runtime
        layout = next(layout for layout in runtime.layouts if layout.kind == "synapse:ExpSyn")
        node = runtime.get_runtime_node(layout.id)

        runtime.event_buffers[layout.id].value = np.asarray([1.0]) * u.uS
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            cell.update()
        g_after_event = float(node.g.value[0].to_decimal(u.uS))

        with brainstate.environ.context(t=0.05 * u.ms, dt=0.05 * u.ms):
            cell.update()
        g_after_decay = float(node.g.value[0].to_decimal(u.uS))
        current = float(cell.sample_probe("i_syn").to_decimal(u.nA)[0])

        self.assertAlmostEqual(g_after_event, float(np.exp(-0.05 / 2.0)))
        self.assertAlmostEqual(g_after_decay, float(np.exp(-0.1 / 2.0)))
        self.assertNotEqual(current, 0.0)

    def test_exp2syn_drive_updates_A_B_and_current(self) -> None:
        cell = Cell(
            _build_tree(),
            solver="staggered",
            V_init=-60.0 * u.mV,
        )
        cell.place(at("soma", 0.5), braincell.mech.MechanismProbe(name="A", mechanism="exp2_syn", field="A"))
        cell.place(at("soma", 0.5), braincell.mech.MechanismProbe(name="B", mechanism="exp2_syn", field="B"))
        cell.place(at("soma", 0.5), braincell.mech.CurrentProbe(name="i_syn", mechanism="exp2_syn"))
        cell.place(
            at("soma", 0.5),
            braincell.mech.Synapse(
                "Exp2Syn",
                tau1=0.5 * u.ms,
                tau2=5.0 * u.ms,
                e=0.0 * u.mV,
                name="exp2_syn",
            ),
        )
        cell.init_state()
        runtime = cell.runtime
        layout = next(layout for layout in runtime.layouts if layout.kind == "synapse:Exp2Syn")
        node = runtime.get_runtime_node(layout.id)

        runtime.event_buffers[layout.id].value = np.asarray([1.0]) * u.uS
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            cell.update()

        A = float(node.A.value[0].to_decimal(u.uS))
        B = float(node.B.value[0].to_decimal(u.uS))
        g_after_event = float(node.g[0].to_decimal(u.uS))
        current = float(cell.sample_probe("i_syn").to_decimal(u.nA)[0])

        self.assertGreater(A, 0.0)
        self.assertGreater(B, 0.0)
        self.assertGreater(g_after_event, 0.0)
        self.assertNotEqual(current, 0.0)

        with brainstate.environ.context(t=0.05 * u.ms, dt=0.05 * u.ms):
            cell.update()

        g_after_decay = float(node.g[0].to_decimal(u.uS))
        current = float(cell.sample_probe("i_syn").to_decimal(u.nA)[0])

        self.assertGreater(g_after_decay, 0.0)
        self.assertNotEqual(current, 0.0)

    def test_netstim_can_drive_expsyn_through_cell_run(self) -> None:
        cell = Cell(
            _build_tree(),
            solver="staggered",
            V_init=-65.0 * u.mV,
        )
        cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
        cell.place(at("soma", 0.5), braincell.mech.MechanismProbe(name="g", mechanism="exp_syn", field="g"))
        cell.place(at("soma", 0.5), braincell.mech.CurrentProbe(name="i_syn", mechanism="exp_syn"))
        exp = braincell.mech.Synapse("ExpSyn", tau=2.0 * u.ms, e=0.0 * u.mV, name="exp_syn")
        cell.place(at("soma", 0.5), exp)
        braincell.connect(
            "stim_to_exp",
            source=braincell.NetStim(name="stim", start=1.0 * u.ms, number=1, interval=10.0 * u.ms),
            synapse=cell.synapses[exp],
            weight=1.0 * u.uS,
        )
        result = cell.run(dt=0.05 * u.ms, duration=10.0 * u.ms)
        self.assertGreater(float(np.max(result.traces["g"].to_decimal(u.uS))), 0.0)
        self.assertGreater(float(np.max(np.abs(result.traces["i_syn"].to_decimal(u.nA)))), 0.0)

    def test_synapse_input_preparation_sums_manual_netstim_and_bound_drive(self) -> None:
        cell = Cell(
            _build_tree(),
            solver="staggered",
            V_init=-65.0 * u.mV,
        )
        exp = braincell.mech.Synapse("ExpSyn", tau=2.0 * u.ms, e=0.0 * u.mV, name="exp_syn")
        cell.place(at("soma", 0.5), exp)
        braincell.connect(
            "stim_to_exp",
            source=braincell.NetStim(name="stim", start=1.0 * u.ms, number=1, interval=10.0 * u.ms),
            synapse=cell.synapses[exp],
            weight=2.0 * u.uS,
        )
        cell.bind_synapse_input("exp_syn", source=lambda: np.asarray([3.0]), weight=0.5 * u.uS)
        cell.init_state()
        runtime = cell.runtime
        layout = next(layout for layout in runtime.layouts if layout.kind == "synapse:ExpSyn")
        runtime.event_buffers[layout.id].value = np.asarray([1.0]) * u.uS

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        self.assertAlmostEqual(float(np.asarray(runtime.get_event_buffer(layout.id).to_decimal(u.uS))[0]), 4.5)

    def test_expsyn_discrete_event_applies_at_begin_step_not_post_integral(self) -> None:
        cell = Cell(
            _build_tree(),
            solver="staggered",
            V_init=-65.0 * u.mV,
        )
        cell.place(
            at("soma", 0.5),
            braincell.mech.Synapse(
                "ExpSyn",
                tau=2.0 * u.ms,
                e=0.0 * u.mV,
                name="exp_syn",
            ),
        )
        cell.init_state()
        runtime = cell.runtime
        layout = next(layout for layout in runtime.layouts if layout.kind == "synapse:ExpSyn")
        node = runtime.get_runtime_node(layout.id)
        runtime.event_buffers[layout.id].value = np.asarray([1.0]) * u.uS

        node.post_integral(cell._cv_to_point(cell.V.value)[..., layout.point_index])
        self.assertAlmostEqual(float(node.g.value[0].to_decimal(u.uS)), 0.0)

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            cell._begin_step()

        self.assertAlmostEqual(float(node.g.value[0].to_decimal(u.uS)), 1.0)


class CellRuntimeStateIsMutableTest(unittest.TestCase):
    """ARCH-07: CellRuntimeState is a mutable dataclass; callers must use plain setattr."""

    def test_cell_runtime_state_is_not_frozen(self) -> None:
        from braincell._compute.state import CellRuntimeState

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
