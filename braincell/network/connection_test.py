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
from braincell.filter import AllRegion, at


def _population(size=3):
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morpho = Morphology.from_root(soma, name="soma")
    cell = Cell(morpho, cv_policy=CVPerBranch(), pop_size=(size,))
    exp = braincell.mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
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
        exp2 = braincell.mech.SynapseSpec("Exp2Syn", name="slow")
        cell.place(at("soma", 0.25), exp2)
        connect("fast", source=NetStim(size=2), synapse=cell.synapses[exp])
        connect("slow", source=NetStim(size=2), synapse=cell.synapses[exp2])

        with self.assertRaisesRegex(TypeError, "by_synapse_type"):
            _ = cell.connections.weight
        self.assertEqual(len(cell.connections.by_synapse_type("ExpSyn").weight), 2)
        self.assertEqual(len(cell.connections.by_synapse_type("Exp2Syn").weight), 2)

    def test_connect_rejects_duplicate_name_and_mixed_synapse_group(self) -> None:
        cell, exp = _population(2)
        other = braincell.mech.SynapseSpec("ExpSyn", name="other")
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
        selected.set(weight=[0.2, -0.1] * u.uS, delay=[2.0, 1.0] * u.ms)

        np.testing.assert_allclose(connections.weight.to_decimal(u.uS), [-0.1, 1.0, 0.2])
        np.testing.assert_allclose(connections.delay.to_decimal(u.ms), [1.0, 0.5, 2.0])
        connections[1].remove()
        np.testing.assert_array_equal(connections.id, [0, 2])
        later = connect("later", source=NetStim(), synapse=cell.synapses[exp][0])
        np.testing.assert_array_equal(later.id, [3])

    def test_event_sequence_drives_connection_runtime(self) -> None:
        cell, exp = _population(2)
        sequence = EventSequence(
            size=2,
            events=EventTable(source_index=[0, 1], time=[1.0, 2.0] * u.ms),
        )
        connect("sequence", source=sequence, synapse=cell.synapses[exp], weight=[0.2, -0.1] * u.uS)
        cell.init_state()
        layout = next(item for item in cell.runtime.layouts if item.kind == "synapse:ExpSyn")
        node = cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))
        np.testing.assert_allclose(cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [0.2, 0.0])

    def test_live_cell_event_output_drives_target_without_replay(self) -> None:
        pre = _hh_population("pre", with_clamp=True)
        post = _hh_population("post")
        exp = braincell.mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        post.place(
            at("soma", 0.5),
            exp,
            braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"),
        )
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
        self.assertEqual(int(conductance_rows[0]), int(spike_rows[0]))

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
            post.place(
                at("soma", 0.5),
                braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"),
            )
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
        self.assertAlmostEqual(float(continuous.time[first_nonzero].to_decimal(u.ms)), 0.3)

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
        node = cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        np.testing.assert_allclose(cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [1.0, 0.0, -3.0])

    def test_target_owned_contacts_run_inside_network_population(self) -> None:
        cell, exp = _population(3)
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"),
        )
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
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"),
        )
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
        self.assertAlmostEqual(float(second.time[first_nonzero].to_decimal(u.ms)), 1.15)

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
        node = cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        np.testing.assert_allclose(cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS), [0.15])


if __name__ == "__main__":
    unittest.main()
