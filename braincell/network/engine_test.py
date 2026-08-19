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

"""Tests for :mod:`braincell.network.engine`."""

import unittest

import brainunit as u
import numpy as np

from braincell.network import (
    Connection,
    Network,
    dense,
    pairs,
)
from braincell.network._testing import (
    make_post_cell,
    make_probe_cell,
    make_spiking_cell,
    step_up_solver,
)


class NetworkRuntimeTest(unittest.TestCase):
    def test_init_state_initializes_population_cells_and_is_idempotent(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        returned = net.add_population("E", pre)
        net.add_population("I", post)

        self.assertIs(returned.cell, pre)
        self.assertFalse(pre._initialized)
        self.assertFalse(post._initialized)

        self.assertIs(net.init_state(), net)
        self.assertTrue(pre._initialized)
        self.assertTrue(post._initialized)

        pre_runtime = pre.runtime
        post_runtime = post.runtime
        self.assertIs(net.init_state(), net)
        self.assertIs(pre.runtime, pre_runtime)
        self.assertIs(post.runtime, post_runtime)

    def test_reset_state_initializes_uninitialized_cells_and_preserves_topology(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        edges = net.add_edges(
            name="E_to_I",
            pre="E",
            post="I",
            method=pairs([(0, 1)]),
        )
        projection = net.add_projection(name="E_to_I_exp", edges="E_to_I", synapse="exp")
        connection = net.add_connection(Connection("E", "I", [1], [0], "exp"))

        self.assertFalse(pre._initialized)
        self.assertFalse(post._initialized)

        self.assertIs(net.reset_state(), net)

        self.assertTrue(pre._initialized)
        self.assertTrue(post._initialized)
        self.assertIs(net.populations["E"].cell, pre)
        self.assertIs(net.edge_sets["E_to_I"], edges)
        self.assertIs(net.projections["E_to_I_exp"], projection)
        self.assertIs(net.connections[0], connection)

    def test_reset_state_restarts_run_from_initial_cell_state(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        first = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)
        net.reset_state()
        second = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        np.testing.assert_allclose(
            np.asarray(first.traces["I"]["g"].to_decimal(u.uS)),
            np.asarray(second.traces["I"]["g"].to_decimal(u.uS)),
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            np.asarray(first.spikes["E"]),
            np.asarray(second.spikes["E"]),
        )
    def test_cross_population_delivery_arrives_on_next_step(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        result = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertEqual(g.shape, (3, 2))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertGreater(float(g[1, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 0]), 0.0)
        self.assertIn("E", result.spikes)
        self.assertEqual(result.spikes["E"].shape, (3, 2, 1))

    def test_spike_recording_population_returns_cell_level_spikes(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        result = net.run(
            dt=0.1 * u.ms,
            duration=0.3 * u.ms,
            spike_recording="population",
        )

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertGreater(float(g[1, 1]), 0.0)
        self.assertEqual(result.spikes["E"].shape, (3, 2))
        self.assertGreater(int(np.asarray(result.spikes["E"]).sum()), 0)

    def test_spike_recording_none_omits_spike_traces_but_delivers_events(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        result = net.run(
            dt=0.1 * u.ms,
            duration=0.3 * u.ms,
            spike_recording="none",
        )

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertGreater(float(g[1, 1]), 0.0)
        self.assertEqual(result.spikes, {})

    def test_spike_recording_rejects_unknown_value(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        with self.assertRaisesRegex(ValueError, "spike_recording"):
            net.run(
                dt=0.1 * u.ms,
                duration=0.1 * u.ms,
                spike_recording="cell",
            )

    def test_run_setup_cache_reuses_repeated_configuration(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(net._run_setup_cache), 1)
        first_setup = next(iter(net._run_setup_cache.values()))

        pre.reset_state()
        post.reset_state()
        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertEqual(len(net._run_setup_cache), 1)
        self.assertIs(next(iter(net._run_setup_cache.values())), first_setup)

    def test_run_loop_cache_reuses_repeated_configuration(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(net._network_run_loop_cache), 1)
        first_loop = next(iter(net._network_run_loop_cache.values()))

        pre.reset_state()
        post.reset_state()
        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertEqual(len(net._network_run_loop_cache), 1)
        self.assertIs(next(iter(net._network_run_loop_cache.values())), first_loop)

    def test_run_setup_cache_is_cleared_on_topology_change(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))
        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(net._run_setup_cache), 1)

        net.add_connection(Connection("E", "I", [1], [0], "exp"))

        self.assertEqual(len(net._run_setup_cache), 0)
        self.assertEqual(len(net._network_run_loop_cache), 0)

    def test_same_population_recurrent_delivery_uses_next_step(self) -> None:
        cell = make_post_cell(size=2)
        cell.solver = step_up_solver
        cell.V_init = -10.0 * u.mV
        net = Network()
        net.add_population("E", cell)
        net.add_connection(Connection("E", "E", [0], [1], "exp"))

        result = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        g = np.asarray(result.traces["E"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertGreater(float(g[1, 1]), 0.0)

    def test_multi_step_delay_arrives_after_requested_steps(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp", delay=0.2 * u.ms))

        result = net.run(dt=0.1 * u.ms, duration=0.4 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 1]), 0.0)
        self.assertGreater(float(g[2, 1]), 0.0)

    def test_non_integer_delay_uses_ceil_by_default(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp", delay=0.15 * u.ms))

        result = net.run(dt=0.1 * u.ms, duration=0.4 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[1, 1]), 0.0)
        self.assertGreater(float(g[2, 1]), 0.0)

    def test_non_integer_delay_can_use_floor_quantization(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp", delay=0.15 * u.ms))

        result = net.run(
            dt=0.1 * u.ms,
            duration=0.3 * u.ms,
            delay_quantization="floor",
        )

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertGreater(float(g[1, 1]), 0.0)

    def test_per_edge_heterogeneous_delays_arrive_at_different_steps(self) -> None:
        pre = make_spiking_cell(size=2)
        post = make_post_cell(size=2)
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(
            Connection(
                "E",
                "I",
                [0, 1],
                [1, 1],
                "exp",
                weight=[0.25, 0.75] * u.uS,
                delay=[0.0, 0.2] * u.ms,
            )
        )

        result = net.run(
            dt=0.1 * u.ms,
            duration=0.4 * u.ms,
            event_backend="scatter",
        )

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 1]), 0.25, places=6)
        self.assertAlmostEqual(float(g[2, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(g[3, 1]), 1.0, places=6)

    def test_event_backend_auto_matches_scatter(self) -> None:
        def run(backend):
            pre = make_spiking_cell(size=2)
            post = make_post_cell(size=2)
            net = Network()
            net.add_population("E", pre)
            net.add_population("I", post)
            net.add_connection(
                Connection(
                    "E",
                    "I",
                    [0, 1],
                    [1, 1],
                    "exp",
                    weight=[0.25, 0.75] * u.uS,
                    delay=[0.0, 0.2] * u.ms,
                )
            )
            result = net.run(dt=0.1 * u.ms, duration=0.4 * u.ms, event_backend=backend)
            return np.asarray(result.traces["I"]["g"].to_decimal(u.uS))

        np.testing.assert_allclose(run("auto"), run("scatter"), rtol=1e-9, atol=1e-9)

    def test_event_backend_brainevent_jax_raw_matches_scatter_when_available(self) -> None:
        try:
            import brainevent
        except Exception:
            return
        if not hasattr(brainevent, "coomv"):
            return

        def run(backend, *, brainevent_backend="jax_raw"):
            pre = make_spiking_cell(size=2)
            post = make_post_cell(size=2)
            net = Network()
            net.add_population("E", pre)
            net.add_population("I", post)
            net.add_connection(
                Connection(
                    "E",
                    "I",
                    [0, 1],
                    [1, 1],
                    "exp",
                    weight=[0.25, 0.75] * u.uS,
                    delay=[0.0, 0.2] * u.ms,
                )
            )
            result = net.run(
                dt=0.1 * u.ms,
                duration=0.4 * u.ms,
                event_backend=backend,
                brainevent_backend=brainevent_backend,
            )
            return np.asarray(result.traces["I"]["g"].to_decimal(u.uS))

        np.testing.assert_allclose(
            run("brainevent", brainevent_backend="jax_raw"),
            run("scatter"),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_event_backend_rejects_unknown_value(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        with self.assertRaisesRegex(ValueError, "event_backend"):
            net.run(dt=0.1 * u.ms, duration=0.1 * u.ms, event_backend="dense")
    def test_multiple_pre_populations_sum_arrivals_to_same_synapse_layout(self) -> None:
        pre_a = make_spiking_cell(size=1)
        pre_b = make_spiking_cell(size=1)
        post = make_post_cell()
        net = Network()
        net.add_population("A", pre_a)
        net.add_population("B", pre_b)
        net.add_population("I", post)
        net.add_connection(Connection("A", "I", [0], [1], "exp", weight=0.25 * u.uS))
        net.add_connection(Connection("B", "I", [0], [1], "exp", weight=0.75 * u.uS))

        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 1]), 1.0, places=6)


class NetworkDisplayTest(unittest.TestCase):
    def test_network_repr_and_str_show_topology_summary(self) -> None:
        net = Network(name="demo")
        net.add_population("E", make_probe_cell(size=3))
        net.add_population("I", make_probe_cell(size=2))
        net.add_edges(name="E_to_I", pre="E", post="I", method=pairs([(0, 1)]))
        net.add_projection(name="E_to_I_exp", edges="E_to_I", synapse="exp")
        net.add_connection(Connection("E", "I", [1], [0], "exp"))

        compact = repr(net)
        detailed = str(net)

        self.assertEqual(
            compact,
            "Network(name='demo', populations=2, edge_sets=1, projections=1, connections=1)",
        )
        self.assertIn("populations:", detailed)
        self.assertIn("E: size=3", detailed)
        self.assertIn("edge_sets:", detailed)
        self.assertIn("E_to_I: E -> I, n_edge=1", detailed)
        self.assertIn("projections:", detailed)
        self.assertIn("E_to_I_exp: edges='E_to_I', synapse='exp'", detailed)
        self.assertIn("direct connections:", detailed)
        self.assertIn("[0]: E -> I, synapse='exp', n_contact=1", detailed)
        self.assertNotIn("array(", detailed)


if __name__ == "__main__":
    unittest.main()
