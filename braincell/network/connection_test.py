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

import gc
import unittest

import brainunit as u
import brainstate
import jax
import numpy as np

import braincell
from braincell import (
    Branch,
    Cell,
    ConnectionView,
    CVPerBranch,
    EventSequence,
    EventTable,
    Morphology,
    NetStim,
    Network,
    connect,
)
from braincell._random_test import default_key_data, drawn_seeds
from braincell.filter import AllRegion, Terminals, at
from braincell.network import connection as rules
from braincell.morph._testing import make_tree_s


def _population(size=3):
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morpho = Morphology.from_root(soma, name="soma")
    cell = Cell(morpho, cv_policy=CVPerBranch(), pop_size=(size,))
    exp = braincell.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
    cell.place(at("soma", 0.5), exp)
    return cell, exp


def _hh_population(name, *, with_clamp=False):
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = Cell(
        Morphology.from_root(soma, name="soma"),
        cv_policy=CVPerBranch(),
        pop_size=(1,),
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
        name=name,
    )
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm**2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        ),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("Na_HH1952", g_max=120.0 * (u.mS / u.cm**2)),
        braincell.mech.Channel("K_HH1952", g_max=36.0 * (u.mS / u.cm**2)),
        braincell.mech.Channel("IL", g_max=0.3 * (u.mS / u.cm**2), E=-54.387 * u.mV),
    )
    if with_clamp:
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentClamp(delay=1.0 * u.ms, durations=0.1 * u.ms, amplitudes=1.0 * u.nA),
        )
    return cell


def _step_up_solver(cell):
    cell.V.value = cell.V.value + 40.0 * u.mV


class ConnectionTest(unittest.TestCase):
    def test_connect_does_not_treat_cell_as_population_event_output(self) -> None:
        cell, exp = _population(1)

        with self.assertRaisesRegex(TypeError, "EventSource or EventSourceView"):
            connect("invalid", source=cell, synapse=cell.synapses[exp])

    def test_reset_state_restores_exact_initialized_hh_states(self) -> None:
        cell = _hh_population("reset_baseline", with_clamp=True)
        cell.init_state()
        initial = {path: state.value for path, state in brainstate.graph.states(cell).items()}

        cell.run(dt=0.025 * u.ms, duration=0.2 * u.ms)
        cell.reset_state()

        restored = brainstate.graph.states(cell)
        for path, expected in initial.items():
            actual = restored[path].value
            if isinstance(expected, u.Quantity):
                np.testing.assert_allclose(actual.to_decimal(expected.unit), expected.to_decimal(expected.unit))
            else:
                np.testing.assert_allclose(actual, expected)

    def test_named_queries_boolean_masks_and_repr(self) -> None:
        cell, exp = _population(3)
        fast_source = NetStim(size=3, name="fast_stim")
        slow_source = NetStim(size=3, name="slow_stim")
        fast = connect(
            "pf_fast",
            source=fast_source,
            synapse=cell.synapses[exp],
            delay=np.asarray([0.0, 2.0, 3.0]) * u.ms,
        )
        connect("pf_slow", source=slow_source, synapse=cell.synapses[exp], delay=4.0 * u.ms)

        self.assertEqual(cell.connections.connect_names, ("pf_fast", "pf_slow"))
        np.testing.assert_array_equal(cell.connections["pf_fast"].id, fast.id)
        np.testing.assert_array_equal(cell.connections.by_connect_name("pf_fast").id, fast.id)
        np.testing.assert_array_equal(cell.connections.by_source(fast_source).id, fast.id)
        self.assertEqual(len(cell.connections.by_source_type("NetStim")), 6)
        np.testing.assert_array_equal(cell.connections.by_source_name("fast_stim").id, fast.id)
        self.assertEqual(len(cell.connections.by_synapse_type("ExpSyn")), 6)
        self.assertEqual(len(cell.connections.by_synapse_name("exp")), 6)
        self.assertEqual(len(cell.connections[cell.connections.delay > 1.0 * u.ms]), 5)

        display = repr(cell.connections)
        self.assertIn("rows=6, connects=2", display)
        self.assertIn("pf_fast: NetStim(fast_stim) -> ExpSyn(exp), rows=3", display)
        self.assertIn("pf_slow: NetStim(slow_stim) -> ExpSyn(exp), rows=3", display)

    def test_mixed_synapse_types_require_type_filter_for_weight(self) -> None:
        cell, exp = _population(2)
        exp2 = braincell.mech.Synapse("Exp2Syn", name="slow")
        cell.place(at("soma", 0.25), exp2)
        connect("fast", source=NetStim(size=2), synapse=cell.synapses[exp])
        connect("slow", source=NetStim(size=2), synapse=cell.synapses[exp2])

        with self.assertRaisesRegex(TypeError, "by_synapse_type"):
            _ = cell.connections.weight
        self.assertEqual(len(cell.connections.by_synapse_type("ExpSyn").weight), 2)
        self.assertEqual(len(cell.connections.by_synapse_type("Exp2Syn").weight), 2)

    def test_connect_rejects_duplicate_name_and_mixed_synapse_group(self) -> None:
        cell, exp = _population(2)
        other = braincell.mech.Synapse("ExpSyn", name="other")
        cell.place(at("soma", 0.25), other)
        connect("input", source=NetStim(size=2), synapse=cell.synapses[exp])

        with self.assertRaisesRegex(ValueError, "already used"):
            connect("input", source=NetStim(size=2), synapse=cell.synapses[exp])
        with self.assertRaisesRegex(ValueError, "one synapse type and one synapse name"):
            connect("mixed", source=NetStim(size=4), synapse=cell.synapses)

    def test_many_connect_calls_warn_once_and_repr_is_bounded(self) -> None:
        cell, exp = _population(1)
        target = cell.synapses[exp]
        with self.assertWarnsRegex(RuntimeWarning, "batched connect"):
            for index in range(257):
                connect(f"input_{index}", source=NetStim(), synapse=target)

        display = repr(cell.connections)
        self.assertIn("connects=257", display)
        self.assertIn("... +1 more", display)
        self.assertEqual(len(cell.connections.connect_names), 257)

    def test_connect_aligns_equal_and_singleton_endpoint_views(self) -> None:
        cell, exp = _population(3)
        target = cell.synapses[exp]
        zipped = connect("zipped", source=NetStim(size=3), synapse=target)
        fanout = connect("fanout", source=NetStim(), synapse=target)
        fanin = connect("fanin", source=NetStim(size=3), synapse=target[0])

        self.assertIsInstance(zipped, ConnectionView)
        np.testing.assert_array_equal(zipped.source_index, [0, 1, 2])
        np.testing.assert_array_equal(zipped.synapse_id, target.id)
        np.testing.assert_array_equal(fanout.source_index, [0, 0, 0])
        np.testing.assert_array_equal(fanin.synapse_id, [target.id[0]] * 3)

    def test_connect_uses_duplicate_views_for_arbitrary_rows(self) -> None:
        cell, exp = _population(3)
        source = NetStim(size=3)
        target = cell.synapses[exp]
        connections = connect(
            "arbitrary",
            source=source[[2, 0, 2, 1]],
            synapse=target[[1, 2, 1, 0]],
            weight=np.asarray([100.0, 200.0, -50.0, 300.0]) * u.nS,
        )

        np.testing.assert_array_equal(connections.source_index, [2, 0, 2, 1])
        np.testing.assert_array_equal(connections.synapse_id, target.id[[1, 2, 1, 0]])
        np.testing.assert_allclose(connections.weight.to_decimal(u.uS), [0.1, 0.2, -0.05, 0.3])

    def test_connect_rejects_unaligned_non_singleton_views(self) -> None:
        cell, exp = _population(3)
        with self.assertRaisesRegex(ValueError, "duplicate-preserving"):
            connect("bad_shape", source=NetStim(size=2), synapse=cell.synapses[exp])

    def test_connection_weight_contract_default_and_errors(self) -> None:
        cell, exp = _population(2)
        default = connect("default", source=NetStim(size=2), synapse=cell.synapses[exp])
        np.testing.assert_allclose(default.weight.to_decimal(u.uS), [1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "incompatible unit"):
            connect("bad_unit", source=NetStim(size=2), synapse=cell.synapses[exp], weight=np.ones(2) * u.nA)
        with self.assertRaisesRegex(TypeError, "cannot be None"):
            connect("bad_none", source=NetStim(size=2), synapse=cell.synapses[exp], weight=None)

    def test_connection_view_set_remove_and_nonreused_ids(self) -> None:
        cell, exp = _population(3)
        connections = connect("first", source=NetStim(size=3), synapse=cell.synapses[exp], delay=0.5 * u.ms)
        selected = connections[[2, 0]]
        with self.assertRaises(RuntimeError) as caught:
            selected.set(weight=[0.2, -0.1] * u.uS)
        self.assertEqual(
            str(caught.exception),
            "ConnectionView.set() requires init_state() first. Before init_state(), give the values to connect().",
        )
        connections[1].remove()
        np.testing.assert_array_equal(connections.id, [0, 2])
        later = connect("later", source=NetStim(), synapse=cell.synapses[exp][0])
        np.testing.assert_array_equal(later.id, [3])
        cell.init_state()
        connections = cell.connections["first"]
        selected = connections[[1, 0]]
        selected.set(weight=[0.2, -0.1] * u.uS)
        np.testing.assert_allclose(connections.weight.to_decimal(u.uS), [-0.1, 0.2])
        with self.assertRaisesRegex(RuntimeError, "delay"):
            selected.set(delay=1 * u.ms)
        np.testing.assert_allclose(connections.delay.to_decimal(u.ms), [0.5, 0.5])

    def test_multi_call_view_groups_calls_and_round_trips_weights(self) -> None:
        # ``weight_for``/``set_weight`` index each call's weight vector by row
        # offset and ``_call_views`` groups in one pass, so a view spanning
        # several calls in shuffled order is the case that pins all three.
        cell, exp = _population(6)
        first = connect("a", source=NetStim(size=2), synapse=cell.synapses[exp][0:2], weight=[0.1, 0.2] * u.uS)
        second = connect("b", source=NetStim(size=2), synapse=cell.synapses[exp][2:4], weight=[0.3, 0.4] * u.uS)
        third = connect("c", source=NetStim(size=2), synapse=cell.synapses[exp][4:6], weight=[0.5, 0.6] * u.uS)
        np.testing.assert_array_equal(first.id, [0, 1])
        np.testing.assert_array_equal(second.id, [2, 3])
        np.testing.assert_array_equal(third.id, [4, 5])
        cell.init_state()
        shuffled = cell.connections[[5, 0, 3, 2, 4, 1]]
        np.testing.assert_allclose(shuffled.weight.to_decimal(u.uS), [0.6, 0.1, 0.4, 0.3, 0.5, 0.2])

        views = shuffled._call_views()
        # One view per call, in the order the calls first appear in ``shuffled``.
        self.assertEqual([np.asarray(view.id).tolist() for view in views], [[5, 4], [0, 1], [3, 2]])

        shuffled.set(weight=[-0.6, -0.1, -0.4, -0.3, -0.5, -0.2] * u.uS)
        np.testing.assert_allclose(
            cell.connections.weight.to_decimal(u.uS),
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        )

    def test_multi_call_weight_gathers_each_call_once_in_view_order(self) -> None:
        cell, exp = _population(4)
        connect("a", source=NetStim(size=2), synapse=cell.synapses[exp][0:2], weight=[0.1, 0.2] * u.uS)
        connect("b", source=NetStim(size=2), synapse=cell.synapses[exp][2:4], weight=[30.0, 40.0] * u.nS)
        np.testing.assert_allclose(cell.connections[[3, 0, 2]].weight.to_decimal(u.uS), [0.04, 0.1, 0.03], rtol=1e-6)
        np.testing.assert_allclose(cell.connections[[1]].weight.to_decimal(u.uS), [0.2], rtol=1e-6)

    def test_event_sequence_drives_connection_runtime(self) -> None:
        cell, exp = _population(2)
        sequence = EventSequence(
            size=2,
            events=EventTable(source_index=[0, 1], time=[1.0, 2.0] * u.ms),
        )
        connect("sequence", source=sequence, synapse=cell.synapses[exp], weight=[0.2, -0.1] * u.uS)
        cell.init_state()
        layout = next(item for item in cell.runtime.layouts if item.kind == "synapse:ExpSyn")
        cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))
        np.testing.assert_allclose(cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [0.2, 0.0])

    def test_live_cell_event_output_drives_target_without_replay(self) -> None:
        pre = _hh_population("pre", with_clamp=True)
        post = _hh_population("post")
        exp = braincell.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        post.place(at("soma", 0.5), exp)
        post.loc(at("soma", 0.5)).record("g", braincell.observe.synapse(name="exp").state("g"))
        connect("pre_spike", source=pre.event_outputs["spike"], synapse=post.synapses[exp], weight=0.1 * u.uS)
        network = Network(name="live_direct_connection")
        network.add_population("pre", pre)
        network.add_population("post", post)

        result = network.run(dt=0.025 * u.ms, duration=5.0 * u.ms)
        spike_times = result.events["pre"]["spike"].time.to_decimal(u.ms)
        spike_rows = np.searchsorted(result.time.to_decimal(u.ms), spike_times)
        conductance = np.asarray(result.traces["post"]["g"].to_decimal(u.uS)).reshape(-1)
        conductance_rows = np.flatnonzero(conductance > 0.0)

        self.assertEqual(len(spike_rows), 1)
        self.assertGreater(len(conductance_rows), 0)
        # A recording samples before each step: the event delivered on the
        # step that starts at the spike row first shows one sample later.
        self.assertEqual(int(conductance_rows[0]) - 1, int(spike_rows[0]))

    def test_network_rejects_live_source_outside_execution_scope(self) -> None:
        pre, _ = _population(1)
        post, exp = _population(1)
        connect("outside", source=pre.event_outputs["spike"], synapse=post.synapses[exp])
        network = Network(name="closed_scope")
        network.add_population("post", post)

        with self.assertRaisesRegex(RuntimeError, "outside this Network execution scope"):
            network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

    def test_network_direct_live_delay_survives_split_run(self) -> None:
        def build_network():
            pre, _ = _population(1)
            pre.V_init = -10.0 * u.mV
            pre.V_th = 0.0 * u.mV
            pre.solver = _step_up_solver
            post, exp = _population(1)
            post.loc(at("soma", 0.5)).record("g", braincell.observe.synapse(name="exp").state("g"))
            connect(
                "live_delay",
                source=pre.event_outputs["spike"],
                synapse=post.synapses[exp],
                weight=0.2 * u.uS,
                delay=0.3 * u.ms,
            )
            network = Network(name="live_delay")
            network.add_population("pre", pre)
            network.add_population("post", post)
            return network

        continuous = build_network().run(dt=0.1 * u.ms, duration=0.8 * u.ms, event_backend="scatter")
        split_network = build_network()
        first = split_network.run(dt=0.1 * u.ms, duration=0.2 * u.ms, event_backend="scatter")
        second = split_network.run(dt=0.1 * u.ms, duration=0.6 * u.ms, event_backend="scatter")
        split_g = np.concatenate(
            [
                np.asarray(first.traces["post"]["g"].to_decimal(u.uS)),
                np.asarray(second.traces["post"]["g"].to_decimal(u.uS)),
            ]
        )

        np.testing.assert_allclose(split_g, continuous.traces["post"]["g"].to_decimal(u.uS), rtol=1e-6)
        first_nonzero = int(np.flatnonzero(np.asarray(continuous.traces["post"]["g"].to_decimal(u.uS)))[0])
        # A recording samples before each step: the event delivered on the
        # step that starts at 0.3 ms first shows one sample later.
        self.assertAlmostEqual(float(continuous.time[first_nonzero - 1].to_decimal(u.ms)), 0.3)

    def test_live_source_delay_history_supports_zero_and_multiple_steps(self) -> None:
        pre, _ = _population(1)
        post, exp = _population(2)
        connections = connect(
            "delayed_spike",
            source=pre.event_outputs["spike"],
            synapse=post.synapses[exp],
            delay=np.asarray([0.0, 0.2]) * u.ms,
        )
        pre.init_state()
        connections.prepare_runtime(0.1 * u.ms)

        pre.spike.value = np.asarray([[1.0]])
        first = connections.event_count(t=0.0 * u.ms, dt=0.1 * u.ms)
        pre.spike.value = np.asarray([[0.0]])
        second = connections.event_count(t=0.1 * u.ms, dt=0.1 * u.ms)
        third = connections.event_count(t=0.2 * u.ms, dt=0.1 * u.ms)

        np.testing.assert_array_equal(first, [1.0, 0.0])
        np.testing.assert_array_equal(second, [0.0, 0.0])
        np.testing.assert_array_equal(third, [0.0, 1.0])

    def test_equal_sizes_zip_and_cell_owns_connections(self) -> None:
        cell, exp = _population(3)
        connection = connect(
            "zipped",
            source=NetStim(size=3),
            synapse=cell.synapses[exp],
            weight=np.asarray([1.0, -2.0, 3.0]) * u.uS,
        )

        np.testing.assert_array_equal(connection.source_index, [0, 1, 2])
        np.testing.assert_array_equal(connection.target_index, [0, 1, 2])
        np.testing.assert_array_equal(connection.id, [0, 1, 2])
        np.testing.assert_array_equal(cell.connections.id, connection.id)

    def test_fanout_convergence_explicit_pairs_and_nonreused_ids(self) -> None:
        cell, exp = _population(3)
        fanout = connect("fanout", source=NetStim(), synapse=cell.synapses[exp])
        convergence = connect("convergence", source=NetStim(size=2), synapse=cell.synapses[exp][0])
        source = NetStim(size=2)
        target = cell.synapses[exp]
        explicit = connect(
            "explicit",
            source=source[[1, 0]],
            synapse=target[[2, 0]],
        )
        fanout.remove()

        np.testing.assert_array_equal(convergence.target_index, [0, 0])
        np.testing.assert_array_equal(explicit.source_index, [1, 0])
        np.testing.assert_array_equal(explicit.target_index, [0, 1])
        np.testing.assert_array_equal(explicit.synapse_id, target.id[[2, 0]])
        self.assertEqual(convergence.id.tolist(), [3, 4])
        self.assertEqual(explicit.id.tolist(), [5, 6])
        self.assertTrue(fanout.removed)
        self.assertEqual(cell.connections.connect_names, ("convergence", "explicit"))

    def test_incompatible_shapes_require_pairs(self) -> None:
        cell, exp = _population(3)
        with self.assertRaisesRegex(ValueError, "duplicate-preserving"):
            connect("bad", source=NetStim(size=2), synapse=cell.synapses[exp])

    def test_heterogeneous_events_scatter_to_aligned_population_targets(self) -> None:
        cell, exp = _population(3)
        connect(
            "heterogeneous",
            source=NetStim(
                size=3,
                start=np.asarray([1.0, 2.0, 1.0]) * u.ms,
                interval=10.0 * u.ms,
            ),
            synapse=cell.synapses[exp],
            weight=np.asarray([1.0, 2.0, -3.0]) * u.uS,
        )
        cell.init_state()
        layout = next(item for item in cell.runtime.layouts if item.kind == "synapse:ExpSyn")
        cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        np.testing.assert_allclose(cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [1.0, 0.0, -3.0])

    def test_target_owned_contacts_run_inside_network_population(self) -> None:
        cell, exp = _population(3)
        cell.loc(at("soma", 0.5)).record("g", braincell.observe.synapse(name="exp").state("g"))
        source = NetStim(size=3, start=0.5 * u.ms, interval=10.0 * u.ms)
        connect(
            "network_input",
            source=source,
            synapse=cell.synapses[exp],
            weight=np.asarray([0.1, 0.2, 0.3]) * u.uS,
        )
        network = Network(name="netstim_population")
        network.add_population("stim", source)
        network.add_population("post", cell)

        result = network.run(dt=0.05 * u.ms, duration=2.0 * u.ms)

        self.assertGreater(float(np.max(result.traces["post"]["g"].to_decimal(u.uS))), 0.0)

    def test_non_grid_event_and_delay_arrive_on_nearest_step_across_runs(self) -> None:
        cell, exp = _population(1)
        cell.loc(at("soma", 0.5)).record("g", braincell.observe.synapse(name="exp").state("g"))
        connect(
            "nongrid",
            source=NetStim(start=1.03 * u.ms, interval=10.0 * u.ms),
            synapse=cell.synapses[exp],
            weight=0.2 * u.uS,
            delay=0.10 * u.ms,
        )

        first = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        second = cell.run(dt=0.05 * u.ms, duration=0.5 * u.ms)
        first_g = first.traces["g"].to_decimal(u.uS)
        second_g = second.traces["g"].to_decimal(u.uS)

        np.testing.assert_allclose(first_g, 0.0)
        first_nonzero = int(np.flatnonzero(second_g > 0.0)[0])
        # A recording samples before each step: the event delivered on the
        # step that starts at 1.15 ms first shows one sample later.
        self.assertAlmostEqual(float(second.time[first_nonzero - 1].to_decimal(u.ms)), 1.15)

    def test_non_grid_event_uses_nearest_not_ceil_step(self) -> None:
        source = NetStim(start=1.021 * u.ms, interval=10.0 * u.ms)

        at_nearest = source.event_count(
            np.asarray([0]),
            t=1.0 * u.ms,
            delay=np.asarray([0.0]) * u.ms,
            dt=0.05 * u.ms,
        )
        at_ceil = source.event_count(
            np.asarray([0]),
            t=1.05 * u.ms,
            delay=np.asarray([0.0]) * u.ms,
            dt=0.05 * u.ms,
        )

        np.testing.assert_array_equal(at_nearest, [1])
        np.testing.assert_array_equal(at_ceil, [0])

    def test_half_step_neighborhood_is_stable_across_precisions(self) -> None:
        for precision in (32, 64):
            with self.subTest(precision=precision):
                with brainstate.environ.context(precision=precision):
                    source = NetStim(
                        size=3,
                        start=np.asarray([1.024999, 1.025, 1.025001]) * u.ms,
                        interval=10.0 * u.ms,
                    )

                    earlier = source.event_count(
                        np.asarray([0, 1, 2]),
                        t=1.0 * u.ms,
                        delay=np.zeros(3) * u.ms,
                        dt=0.05 * u.ms,
                    )
                    later = source.event_count(
                        np.asarray([0, 1, 2]),
                        t=1.05 * u.ms,
                        delay=np.zeros(3) * u.ms,
                        dt=0.05 * u.ms,
                    )
                    compiled_count = jax.jit(
                        lambda t_ms: source.event_count(
                            np.asarray([0, 1, 2]),
                            t=t_ms * u.ms,
                            delay=np.zeros(3) * u.ms,
                            dt=0.05 * u.ms,
                        )
                    )

                    np.testing.assert_array_equal(earlier, [1, 0, 0])
                    np.testing.assert_array_equal(later, [0, 1, 1])
                    np.testing.assert_array_equal(compiled_count(1.0), earlier)
                    np.testing.assert_array_equal(compiled_count(1.05), later)

    def test_multiple_connections_add_on_one_synapse(self) -> None:
        cell, exp = _population(1)
        target = cell.synapses[exp]
        connect("positive", source=NetStim(start=1.0 * u.ms), synapse=target, weight=0.2 * u.uS)
        connect("negative", source=NetStim(start=1.0 * u.ms), synapse=target, weight=-0.05 * u.uS)
        cell.init_state()
        layout = next(item for item in cell.runtime.layouts if item.kind == "synapse:ExpSyn")
        cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        np.testing.assert_allclose(cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [0.15])


def _extra_basal() -> Branch:
    return Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")


def _two_rule_network(target: str, *, pop_size: int = 1):
    """Tree S post Cell with ExpSyn "a" on Terminals() and "b" at soma(0.5), driven by a pre Cell.

    Terminals of tree S are dend's and axon's ends, so the logical synapses
    are a, a, b (ids 0, 1, 2 for one member).
    """
    morpho = make_tree_s()
    post = Cell(morpho, pop_size=(pop_size,))
    post.place(Terminals(), braincell.mech.Synapse("ExpSyn", name="a", tau=2.0 * u.ms, e=0.0 * u.mV))
    post.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="b", tau=2.0 * u.ms, e=0.0 * u.mV))
    pre = Cell(make_tree_s(), pop_size=(pop_size,))
    network = Network("n")
    pre_population = network.add_population("pre", pre)
    post_population = network.add_population("post", post)
    network.connect(
        "drive",
        source=pre_population.event_outputs["spike"],
        synapse=post_population.synapses[target],
        weight=0.1 * u.uS,
    )
    return network, post, morpho


def _targets(post: Cell) -> list[tuple[str, str, float, int]]:
    """(synapse name, branch name, x, member) of every connection row of ``post``."""
    synapses = post._get_synapse_store()
    rows = [synapses._row_by_id[int(item)] for item in post._get_connection_store().synapse_id.tolist()]
    names = [post.morpho.branch(index=int(synapses.branch_id[row])).name for row in rows]
    return [
        (str(synapses.name[row]), name, float(synapses.branch_x[row]), int(synapses.population_index[row]))
        for row, name in zip(rows, names)
    ]


class ConnectionFollowsItsSynapseTest(unittest.TestCase):
    """MORPH-15: a connection row reaches the synapse it was connected to after a rebuild.

    Logical synapse ids are numbered across every place rule, so a rebuild
    that adds a site to an earlier rule shifts the later ids; the rows used
    to keep the old id and deliver to whatever synapse holds it now.
    """

    def test_a_row_follows_its_synapse_when_an_earlier_rule_gains_a_site(self) -> None:
        network, post, morpho = _two_rule_network("b")
        self.assertEqual(post._get_connection_store().synapse_id.tolist(), [2])

        morpho.soma.extra = _extra_basal()
        network.init_state()

        self.assertEqual(post._get_connection_store().synapse_id.tolist(), [3])
        self.assertEqual(_targets(post), [("b", "soma", 0.5, 0)])

    def test_every_member_follows_its_synapse(self) -> None:
        network, post, morpho = _two_rule_network("b", pop_size=2)
        morpho.soma.extra = _extra_basal()
        network.init_state()

        self.assertEqual(_targets(post), [("b", "soma", 0.5, 0), ("b", "soma", 0.5, 1)])

    def test_a_cell_initialized_on_its_own_follows_too(self) -> None:
        morpho = make_tree_s()
        post = Cell(morpho)
        post.place(Terminals(), braincell.mech.Synapse("ExpSyn", name="a", tau=2.0 * u.ms, e=0.0 * u.mV))
        post.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="b", tau=2.0 * u.ms, e=0.0 * u.mV))
        source = NetStim(start=0.1 * u.ms, interval=1.0 * u.ms, number=1)
        connect("drive", source=source, synapse=post.synapses["b"], weight=0.1 * u.uS)

        morpho.soma.extra = _extra_basal()
        post.init_state()

        self.assertEqual(_targets(post), [("b", "soma", 0.5, 0)])

    def test_a_row_follows_a_reused_branch_name(self) -> None:
        # The synapse is identified by branch name: removing the axon and
        # attaching another branch named axon gives the row that branch.
        morpho = make_tree_s()
        post = Cell(morpho)
        post.place(at("axon", 0.5), braincell.mech.Synapse("ExpSyn", name="b", tau=2.0 * u.ms, e=0.0 * u.mV))
        connect("drive", source=NetStim(start=0.1 * u.ms, number=1), synapse=post.synapses["b"], weight=0.1 * u.uS)

        morpho.remove("axon")
        morpho.soma.axon = Branch.from_lengths(lengths=[60.0] * u.um, radii=[0.5, 0.5] * u.um, type="axon")
        post.init_state()

        self.assertEqual(_targets(post), [("b", "axon", 0.5, 0)])

    def test_a_new_site_in_the_connected_group_raises(self) -> None:
        network, post, morpho = _two_rule_network("a")
        self.assertEqual(post._get_connection_store().synapse_id.tolist(), [0, 1])

        morpho.soma.extra = _extra_basal()

        pattern = (
            r"Connection 'drive' .*'a'.*3 sites \(axon\(1\), dend\(1\), extra\(1\)\).*when connected it resolved to 2"
        )
        with self.assertRaisesRegex(RuntimeError, pattern):
            network.init_state()
        # The check runs before any Cell of the Network initializes.
        self.assertFalse(post._initialized)
        self.assertFalse(network.populations["pre"].cell._initialized)
        self.assertEqual(post._get_connection_store().synapse_id.tolist(), [0, 1])

        # The recoveries the message names: connect the group again ...
        post.connections["drive"].remove()
        network.connect(
            "drive_all",
            source=network.populations["pre"].event_outputs["spike"],
            synapse=network.populations["post"].synapses["a"],
            weight=0.1 * u.uS,
        )
        network.init_state()
        self.assertEqual(
            [target for target in _targets(post)][-3:],
            [("a", "dend", 1.0, 0), ("a", "axon", 1.0, 0), ("a", "extra", 1.0, 0)],
        )

    def test_undoing_the_edit_lets_the_connection_initialize(self) -> None:
        network, post, morpho = _two_rule_network("a")
        morpho.soma.extra = _extra_basal()
        with self.assertRaisesRegex(RuntimeError, "Connection 'drive'"):
            network.init_state()

        morpho.remove("extra")
        network.init_state()

        self.assertEqual(_targets(post), [("a", "dend", 1.0, 0), ("a", "axon", 1.0, 0)])

    def test_an_unchanged_declaration_keeps_the_rows(self) -> None:
        network, post, _ = _two_rule_network("b")
        post.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2))
        network.init_state()

        self.assertEqual(post._get_connection_store().synapse_id.tolist(), [2])
        self.assertEqual(_targets(post), [("b", "soma", 0.5, 0)])


if __name__ == "__main__":
    unittest.main()


def _reference_connect_names(view):
    return tuple(dict.fromkeys(str(view._store.call(value).name) for value in view.connect_id.tolist()))


def _reference_by_connect_name(view, name):
    names = np.asarray([view._store.call(value).name for value in view.connect_id.tolist()], dtype=object)
    return ConnectionView(view._store, view._active_ids[names == name])


def _reference_view_repr(view):
    # ``ConnectionView.__repr__`` before CELL-18d: one scan of every row per connect name.
    from braincell._multi_compartment.synapses import _cell_label
    from braincell.network.connection import _CONNECT_CALL_WARNING_THRESHOLD as limit

    names = _reference_connect_names(view)
    lines = [
        f"ConnectionView(cell={_cell_label(view.cell, view.synapse.population_index)}, "
        f"rows={len(view)}, connects={len(names)})"
    ]
    for name in names[:limit]:
        selected = _reference_by_connect_name(view, name)
        call = selected._require_single_call("format connection")
        lines.append(
            f"  {name}: {call.source.source_type}({call.source.source_name}) -> "
            f"{call.synapse_type}({call.synapse_name}), rows={len(selected)}"
        )
    if len(names) > limit:
        lines.append(f"  ... +{len(names) - limit} more")
    return "\n".join(lines)


def _reference_describe(network, target, name):
    view = _reference_by_connect_name(network.populations[target].connections, name)
    call = view._require_single_call("format connection")
    source_name = call.source.source_name
    if not source_name:
        owner = call.source if call.source.is_scheduled else call.source.execution_owner
        source_name = next(
            (population.name for population in network.populations.values() if population.model is owner),
            call.source.source_type,
        )
    return f"{name}: {call.source.source_type}({source_name}) -> {call.synapse_type}({call.synapse_name}), rows={len(view)}"


def _reference_targets(network):
    return tuple(
        name
        for name, population in network.populations.items()
        if population.kind == "cell" and len(population.connections) > 0
    )


def _reference_network_connections_repr(network):
    targets = _reference_targets(network)
    views = {target: network.populations[target].connections for target in targets}
    lines = [
        f"NetworkConnections(targets={len(targets)}, "
        f"connections={sum(len(_reference_connect_names(view)) for view in views.values())}, "
        f"rows={sum(len(view) for view in views.values())})"
    ]
    for target, view in views.items():
        names = _reference_connect_names(view)
        lines.append(f"  {target}: connections={len(names)}, rows={len(view)}")
        lines.extend(f"    {_reference_describe(network, target, name)}" for name in names)
    return "\n".join(lines)


def _reference_network_str(network):
    targets = _reference_targets(network)
    views = {target: network.populations[target].connections for target in targets}
    lines = [
        f"Network(name={network.name!r}, populations={len(network.populations)}, "
        f"connections={sum(len(_reference_connect_names(view)) for view in views.values())}, "
        f"rows={sum(len(view) for view in views.values())})",
        "  populations:",
    ]
    for name, population in network.populations.items():
        initialized = bool(getattr(population.model, "_initialized", False))
        lines.append(
            f"    {name}: size={population.size}, kind={population.kind}, "
            f"model={type(population.model).__name__}, initialized={initialized}"
        )
    if not network.populations:
        lines.append("    <none>")
    lines.append("  connections:")
    for target, view in views.items():
        names = _reference_connect_names(view)
        lines.append(f"    {target}: connections={len(names)}, rows={len(view)}")
        lines.extend(f"      {_reference_describe(network, target, name)}" for name in names)
    if not views:
        lines.append("    <none>")
    return "\n".join(lines)


class ConnectionSummaryTest(unittest.TestCase):
    """CELL-18d: summaries read every row once, and say what they said before."""

    def _network(self):
        from braincell.network._testing import make_recording_post_cell, make_threshold_cell

        network = Network("summary")
        live = network.add_population("pre", make_threshold_cell(3))
        unnamed = network.add_population("stim", NetStim(size=3))
        named = network.add_population("drive", NetStim(size=3, name="named_drive"))
        sequence = network.add_population(
            "seq",
            EventSequence(size=2, events=EventTable(source_index=[0, 1], time=[1.0, 2.0] * u.ms)),
        )
        post = network.add_population("post", make_recording_post_cell(4))
        other = network.add_population("other", make_recording_post_cell(2))
        network.add_population("idle", make_recording_post_cell(1))
        spikes, synapses = live.event_outputs["spike"], post.synapses["exp"]
        network.connect("live", source=spikes[[0, 1, 2, 0]], synapse=synapses)
        network.connect("unnamed", source=unnamed.event_outputs["spike"][0], synapse=synapses[[1, 3]])
        network.connect("named", source=named.event_outputs["spike"][[2, 1]], synapse=other.synapses["exp"])
        network.connect("sequence", source=sequence.event_outputs["event"][[1, 0, 1]], synapse=synapses[[0, 2, 3]])
        network.connect("gone", source=unnamed.event_outputs["spike"][[0, 1]], synapse=synapses[[0, 1]])
        network.connect("partly", source=spikes[[2, 1, 0]], synapse=synapses[[3, 2, 1]])
        network.connect("late", source=named.event_outputs["spike"][1], synapse=synapses[2])
        post.connections["gone"].remove()
        post.connections["partly"][[1]].remove()
        return network

    def _assert_view(self, view) -> None:
        self.assertEqual(repr(view), _reference_view_repr(view))
        self.assertEqual(view.connect_names, _reference_connect_names(view))
        np.testing.assert_array_equal(
            view.connect_name,
            np.asarray([view._store.call(value).name for value in view.connect_id.tolist()], dtype=object),
        )
        self.assertEqual(view.connect_name.dtype, np.dtype(object))
        for name in view.connect_names:
            np.testing.assert_array_equal(view.by_connect_name(name).id, _reference_by_connect_name(view, name).id)

    def test_summaries_match_one_scan_per_connect_name(self) -> None:
        network = self._network()
        post = network.populations["post"].connections
        views = {
            "all rows": post,
            "reordered, repeated rows": post[[7, 0, 9, 0, 3, 7]],
            "one call": post["sequence"],
            "no rows": post[np.asarray([], dtype=np.int64)],
            "another target": network.populations["other"].connections,
            "a target with none": network.populations["idle"].connections,
        }
        for name, view in views.items():
            with self.subTest(name):
                self._assert_view(view)
        self.assertEqual(repr(network.connections), _reference_network_connections_repr(network))
        self.assertEqual(str(network), _reference_network_str(network))
        self.assertEqual(network.connections.n_connections, 6)
        for target in _reference_targets(network):
            for name in _reference_connect_names(network.populations[target].connections):
                self.assertEqual(network.connections.describe(target, name), _reference_describe(network, target, name))

    def test_an_empty_network_summarises_nothing(self) -> None:
        network = Network("empty")
        self.assertEqual(repr(network.connections), _reference_network_connections_repr(network))
        self.assertEqual(str(network), _reference_network_str(network))

    def test_a_view_of_more_connects_than_it_lists_matches(self) -> None:
        cell, exp = _population(2)
        target = cell.synapses[exp]
        with self.assertWarnsRegex(RuntimeWarning, "batched connect"):
            for index in range(258):
                connect(f"input_{index}", source=NetStim(size=2), synapse=target[[index % 2]])
        self._assert_view(cell.connections)
        self._assert_view(cell.connections[::-3])


_STORE_COLUMNS = ("id", "connect_id", "source_index", "synapse_id", "target_index", "active", "_row_of_id")


class ConnectionStoreColumnsTest(unittest.TestCase):
    """CELL-18e: the routing columns, however connects, reads and removals interleave."""

    def _recorded_cell(self):
        from unittest import mock

        from braincell.network.connection import _ConnectionStore

        calls = []
        original = _ConnectionStore.add

        def add(store, **fields):
            row_ids = original(store, **fields)
            calls.append(fields)
            return row_ids

        patcher = mock.patch.object(_ConnectionStore, "add", autospec=True, side_effect=add)
        patcher.start()
        self.addCleanup(patcher.stop)
        cell, exp = _population(4)
        return cell, cell.synapses[exp], calls

    @staticmethod
    def _reference_columns(calls, removed):
        # ``_ConnectionStore.add`` before CELL-18e: every column re-concatenated per call.
        columns = {name: np.asarray([], dtype=np.int64) for name in _STORE_COLUMNS}
        columns["active"] = np.asarray([], dtype=bool)
        delay = np.asarray([], dtype=np.float64) * u.ms
        next_row = 0
        for connect_id, fields in enumerate(calls):
            count = len(fields["source_index"])
            row_ids = np.arange(next_row, next_row + count, dtype=np.int64)
            start = len(columns["id"])
            next_row += count
            columns["id"] = np.concatenate((columns["id"], row_ids))
            columns["connect_id"] = np.concatenate((columns["connect_id"], np.full(count, connect_id, dtype=np.int64)))
            for name in ("source_index", "synapse_id", "target_index"):
                columns[name] = np.concatenate((columns[name], np.asarray(fields[name], dtype=np.int64)))
            delay = u.Quantity(
                np.concatenate((np.asarray(delay.to_decimal(u.ms)), np.asarray(fields["delay"].to_decimal(u.ms)))),
                u.ms,
            )
            columns["active"] = np.concatenate((columns["active"], np.ones(count, dtype=bool)))
            columns["_row_of_id"] = np.concatenate(
                (columns["_row_of_id"], np.arange(start, start + count, dtype=np.int64))
            )
        columns["active"][np.asarray(sorted(removed), dtype=np.int64)] = False
        return columns, delay

    def _assert_columns(self, cell, calls, removed) -> None:
        store = cell._get_connection_store()
        columns, delay = self._reference_columns(calls, removed)
        for name in _STORE_COLUMNS:
            actual = getattr(store, name)
            self.assertEqual(actual.dtype, columns[name].dtype, name)
            np.testing.assert_array_equal(actual, columns[name], err_msg=name)
        self.assertEqual(u.get_unit(store.delay), u.ms)
        self.assertEqual(np.asarray(store.delay.mantissa).dtype, np.asarray(delay.mantissa).dtype)
        np.testing.assert_array_equal(np.asarray(store.delay.mantissa), np.asarray(delay.mantissa))

    def test_columns_match_one_concatenation_per_connect(self) -> None:
        cell, target, calls = self._recorded_cell()
        rng = np.random.default_rng(18)
        removed = set()
        connect("first", source=NetStim(size=4), synapse=target)
        self._assert_columns(cell, calls, removed)
        connect("micro", source=NetStim(size=4), synapse=target, delay=np.asarray([100.0, 250.0, 0.5, 7.0]) * u.us)
        connect("single", source=NetStim(size=1), synapse=target[2], delay=0.3 * u.ms)
        removed.update(cell.connections["micro"].id[[1, 2]].tolist())
        cell.connections["micro"][[1, 2]].remove()
        connect("after removal", source=NetStim(size=2), synapse=target[[3, 0]], delay=np.float32(0.1) * u.ms)
        self._assert_columns(cell, calls, removed)
        for index in range(40):
            size = int(rng.integers(1, 5))
            delay = rng.uniform(0.0, 3.0, size) * (u.ms if index % 3 else u.us)
            connect(f"many_{index}", source=NetStim(size=size), synapse=target[rng.permutation(4)[:size]], delay=delay)
            if index % 9 == 0:
                np.testing.assert_array_equal(
                    cell.connections.id, np.flatnonzero(self._reference_columns(calls, removed)[0]["active"])
                )
            if index == 20:
                removed.update(cell.connections[f"many_{index}"].id.tolist())
                cell.connections[f"many_{index}"].remove()
        self._assert_columns(cell, calls, removed)
        self.assertEqual(len(cell.connections), len(self._reference_columns(calls, removed)[0]["id"]) - len(removed))

    def test_an_empty_store_has_empty_columns(self) -> None:
        cell, target, calls = self._recorded_cell()
        self._assert_columns(cell, calls, set())
        self.assertEqual(len(cell.connections), 0)


def _sampled_rows(pairing) -> np.ndarray:
    """Rows of a direct ``connect`` from six periodic sources to four lumped members, one ExpSyn each."""
    cell = Cell.point(4, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
    rows = connect("drive", source=NetStim(size=6, number=1), synapse=cell.synapses["syn"], pairing=pairing)
    return np.stack([np.asarray(rows.source_index), np.asarray(rows.target_index)])


_RULES = {
    "independent(5)": lambda **seed: rules.independent(5, **seed),
    "source_first(5)": lambda **seed: rules.source_first(5, **seed),
    "synapse_first(5)": lambda **seed: rules.synapse_first(5, **seed),
    "by_source(2)": lambda **seed: rules.by_source(2, **seed),
    "by_synapse(2)": lambda **seed: rules.by_synapse(2, **seed),
    "match_degrees(2, 3)": lambda **seed: rules.match_degrees(2, 3, **seed),
}


class DirectConnectSeedTest(unittest.TestCase):
    """CELL-11b: a rule without a seed in a direct ``connect`` draws one from the BrainState default stream."""

    def test_a_seedless_rule_draws_the_seed_it_then_uses(self) -> None:
        # Acceptance 6. Before CELL-11b every seedless rule used seed 613549620.
        seed = drawn_seeds(42, 1)[0]
        for name, rule in _RULES.items():
            with self.subTest(rule=name):
                with brainstate.random.seed_context(42):
                    drawn = _sampled_rows(rule())
                np.testing.assert_array_equal(drawn, _sampled_rows(rule(seed=seed)))
                with brainstate.random.seed_context(43):
                    self.assertFalse(np.array_equal(_sampled_rows(rule()), drawn))

    def test_only_a_seedless_rule_advances_the_default_stream(self) -> None:
        for name, pairing, advances in (
            ("independent(5)", rules.independent(5), True),
            ("independent(5, seed=8)", rules.independent(5, seed=8), False),
            ("no pairing", None, False),
        ):
            with self.subTest(pairing=name):
                cell = Cell.point(6, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
                cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn"))
                stim = NetStim(size=6, number=1)
                with brainstate.random.seed_context(5):
                    before = default_key_data()
                    connect("drive", source=stim, synapse=cell.synapses["syn"], pairing=pairing)
                    self.assertEqual(not np.array_equal(default_key_data(), before), advances)

    def test_a_seedless_rule_cannot_draw_under_a_transformation(self) -> None:
        # Acceptance 7. Before CELL-11b the same call failed later, when the
        # rows were sampled (TracerArrayConversionError), as a seeded rule still does.
        cell = Cell.point(4, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn"))
        stim = NetStim(size=6, number=1)

        def sampled(pairing):
            return lambda: len(connect("drive", source=stim, synapse=cell.synapses["syn"], pairing=pairing))

        brainstate.random.seed(1)
        before = default_key_data()
        with self.assertRaisesRegex(TypeError, "connect.*traced"):
            brainstate.transform.jit(sampled(rules.independent(4)))()
        np.testing.assert_array_equal(default_key_data(), before)
        with self.assertRaises(jax.errors.TracerArrayConversionError):
            brainstate.transform.jit(sampled(rules.independent(4, seed=1)))()


# ---------------------------------------------------------------------------
# ConnectionView.record rejections (CELL-12c): each names its stage and type,
# and a rejection at record leaves the Cell's recordings unchanged.


def _rule(**kwargs):
    params = dict(A_plus=0.01 * u.uS, A_minus=0.0105 * u.uS, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms)
    params.update(kwargs)
    return braincell.mech.ConnectionRule("PairSTDP", **params)


def _rule_cell(n_sites=1):
    cell = Cell.point(1, V_init=-65.0 * u.mV)
    for _ in range(n_sites):
        cell.place(braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms))
    return cell


def _times(*times_ms):
    return EventSequence.from_times(tuple(np.asarray(row, dtype=float) * u.ms for row in times_ms))


class _MillisecondTrace(braincell.ConnectionRule):
    """A rule whose ``x_pre`` is in ms, so it cannot share a recording with PairSTDP's unitless trace."""

    def init_state(self, weight):
        return {"w": weight, "x_pre": np.zeros(self.size) * u.ms}

    def update(self, state, *, pre, dt):
        return dict(state)


class RecordRuleStateRejectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = _rule_cell()
        self.ruled = connect(
            "c", source=_times([0.0]), synapse=self.cell.synapses["s"], weight=0.05 * u.uS, plasticity=_rule()
        )
        self.plain = connect("p", source=_times([1.0]), synapse=self.cell.synapses["s"], weight=0.05 * u.uS)

    def _assert_rejected_at_record(self, error, pattern, record):
        before = dict(self.cell.recordings)
        with self.assertRaisesRegex(error, pattern):
            record()
        self.assertEqual(dict(self.cell.recordings), before)

    def test_a_row_without_a_rule_raises_type_error(self) -> None:
        self._assert_rejected_at_record(TypeError, "'p' has no plasticity rule", lambda: self.plain.record("w", "w"))
        self._assert_rejected_at_record(
            TypeError, "'p' has no plasticity rule", lambda: self.cell.connections.record("w", "w")
        )

    def test_an_unknown_state_raises_key_error_listing_the_states(self) -> None:
        self._assert_rejected_at_record(
            KeyError, r"no state 'z'; its states are \('w', 'x_pre', 'y_post'\)", lambda: self.ruled.record("z", "z")
        )

    def test_states_of_different_dimensions_raise_type_error(self) -> None:
        cell = _rule_cell(n_sites=2)
        connect("a", source=_times([0.0]), synapse=cell.synapses["s"][0], weight=0.05 * u.uS, plasticity=_rule())
        connect(
            "b",
            source=_times([0.0]),
            synapse=cell.synapses["s"][1],
            weight=0.05 * u.uS,
            plasticity=braincell.mech.ConnectionRule(_MillisecondTrace),
        )
        with self.assertRaisesRegex(TypeError, "record them separately"):
            cell.connections.record("x", "x_pre")
        self.assertEqual(dict(cell.recordings), {})
        cell.connections.record("w", "w")
        samples = cell.run(dt=0.1 * u.ms, duration=1.0 * u.ms).samples
        self.assertEqual(samples["w"].values.shape, (10, 2))
        self.assertEqual([row.mechanism_type for row in samples["w"].schema.rows], ["PairSTDP", "_MillisecondTrace"])

    def test_an_empty_view_raises_value_error(self) -> None:
        self._assert_rejected_at_record(ValueError, "empty ConnectionView", lambda: self.ruled[0:0].record("w", "w"))

    def test_a_state_that_is_not_a_name_raises_value_error(self) -> None:
        for state in (1, ""):
            self._assert_rejected_at_record(ValueError, "rule state", lambda: self.ruled.record("w", state))

    def test_a_used_name_and_a_bad_period_raise_value_error(self) -> None:
        self.ruled.record("w", "w")
        self._assert_rejected_at_record(ValueError, "already", lambda: self.ruled.record("w", "x_pre"))
        self._assert_rejected_at_record(ValueError, "period", lambda: self.ruled.record("p", "w", period=-1.0 * u.ms))

    def test_after_init_state_raises_runtime_error_until_reset(self) -> None:
        self.cell.init_state()
        # The view selected before init_state is stale; select it again.
        self._assert_rejected_at_record(
            RuntimeError,
            r"after init_state\(\); call reset\(\) first",
            lambda: self.cell.connections["c"].record("w", "w"),
        )
        self.cell.reset()
        self.cell.connections["c"].record("w", "w")
        self.assertEqual(self.cell.run(dt=0.1 * u.ms, duration=1.0 * u.ms).samples["w"].values.shape, (10, 1))

    def test_cell_record_rejects_the_rule_observable(self) -> None:
        observable = self.ruled.record("w", "w").observable
        other = _rule_cell()
        with self.assertRaises(TypeError):
            other.record("w", observable)
        self.assertEqual(dict(other.recordings), {})

    def test_cell_run_on_a_cell_its_gone_network_initialized_raises_runtime_error(self) -> None:
        cell = _rule_cell()
        net = Network("gone")
        source = net.add_population("pre", _times([0.0]))
        net.add_population("post", cell)
        net.connect("c", source=source, synapse=cell.synapses["s"], weight=0.05 * u.uS, plasticity=_rule())
        cell.connections["c"].record("w", "w")
        net.run(dt=0.1 * u.ms, duration=1.0 * u.ms)
        del net, source
        gc.collect()
        self.assertIsNone(cell.network_owner)
        # The Network's Cell keeps no copy of the state before each update.
        with self.assertRaisesRegex(
            RuntimeError, r"a Network initialized this Cell. Call reset\(\) and init_state\(\)"
        ):
            cell.run(dt=0.1 * u.ms, duration=1.0 * u.ms)
        cell.reset()
        cell.init_state()
        w = cell.run(dt=0.1 * u.ms, duration=1.0 * u.ms).samples["w"].values
        np.testing.assert_allclose(w.to_decimal(u.uS), np.full((10, 1), 0.05), rtol=1e-6)

    def test_batches_and_live_rows_on_cell_run_are_still_rejected(self) -> None:
        self.ruled.record("w", "w")
        with self.assertRaisesRegex(NotImplementedError, "batch"):
            self.cell.init_state(batch_size=2)
        cell = _rule_cell()
        pre = _rule_cell()
        connect(
            "c", source=pre.event_outputs["spike"], synapse=cell.synapses["s"], weight=0.05 * u.uS, plasticity=_rule()
        )
        cell.connections["c"].record("w", "w")
        with self.assertRaisesRegex(NotImplementedError, "Network"):
            cell.run(dt=0.1 * u.ms, duration=1.0 * u.ms)
