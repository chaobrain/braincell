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

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell.filter import AllRegion
from braincell.network import Network
from braincell.network._testing import make_runtime_network, make_threshold_cell


class NetworkRuntimeTest(unittest.TestCase):
    def test_preparation_requires_cell_and_keeps_configuration_static(self):
        net = Network("source_only")
        net.add_population("input", braincell.NetStim())
        with self.assertRaisesRegex(ValueError, "at least one Cell"):
            net.prepare_run(dt=0.1 * u.ms)
        net = make_runtime_network()
        net.prepare_run(dt=0.1 * u.ms, event_backend="scatter")
        with self.assertRaisesRegex(RuntimeError, "configuration is fixed"):
            net.prepare_run(dt=0.2 * u.ms, event_backend="scatter")

    def test_cell_with_trainable_bindings_is_aggregated_without_copying(self) -> None:
        cell = make_threshold_cell()
        cell.paint(AllRegion(), braincell.mech.Channel("IL", name="leak"))
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="factor"))
        network = Network("network")
        network.add_population("cell", cell)
        self.assertIs(
            network.trainables.parameters().states()["cell.factor"], cell.trainables.parameters().states()["factor"]
        )

    def test_prepared_network_keeps_weight_and_threshold_gradients(self):
        for backend in ("scatter", "brainevent"):
            with self.subTest(backend=backend):
                network = make_runtime_network(delay=0.2 * u.ms)
                post = network.populations["post"].cell
                pre = network.populations["pre"].cell
                post.connections["drive"].trainable(weight=braincell.trainable.scale(name="w"))
                pre.event_outputs["spike"].trainable(threshold=braincell.trainable.parameter(group_by="all", name="th"))
                network.prepare_run(dt=0.1 * u.ms, event_backend=backend)
                synapse = post.runtime.get_runtime_node(post.synapses["exp"]._store.layout_id("ExpSyn"))

                def observe():
                    network.reset_state()
                    brainstate.transform.for_loop(lambda _: network.update(), np.arange(5))
                    return synapse.g.value.to_decimal(u.uS).sum()

                run = brainstate.transform.jit(observe)
                grad = brainstate.transform.jit(
                    brainstate.transform.grad(observe, grad_states=network.trainables.parameters().states())
                )
                first = run()
                gradients = grad()
                self.assertGreater(float(first), 0.0)
                self.assertNotEqual(float(gradients["post.w"]), 0.0)
                self.assertNotEqual(float(u.get_mantissa(gradients["pre.th"])), 0.0)
                values = network.trainables.parameters().physical_values()
                values["post.w"] = 2.0
                network.trainables.parameters().set_physical_values(values)
                np.testing.assert_allclose(run(), 2 * first, rtol=1e-6)

    def test_prepare_required_and_update_matches_run(self):
        network = make_runtime_network(delay=0.2 * u.ms)
        with self.assertRaisesRegex(RuntimeError, "prepare_run"):
            network.update()
        network.prepare_run(dt=0.1 * u.ms, event_backend="scatter")
        post = network.populations["post"].cell
        synapse = post.runtime.get_runtime_node(post.synapses["exp"]._store.layout_id("ExpSyn"))

        def step(_):
            network.update()
            return synapse.g.value

        actual = brainstate.transform.for_loop(step, np.arange(5))
        reference = make_runtime_network(delay=0.2 * u.ms).run(
            dt=0.1 * u.ms, duration=0.5 * u.ms, event_backend="scatter"
        )
        # This fixture records before the step; update() exposes post-step state.
        np.testing.assert_allclose(
            actual.to_decimal(u.uS)[:-1], reference.samples["post"]["g"].values.to_decimal(u.uS)[1:], rtol=1e-6
        )

    def test_cell_has_one_network_execution_owner(self) -> None:
        cell = make_threshold_cell()
        first = Network("first")
        second = Network("second")
        first.add_population("pre", cell)
        with self.assertRaisesRegex(ValueError, "more than one Network population"):
            first.add_population("alias", cell)
        with self.assertRaisesRegex(RuntimeError, "already belongs"):
            second.add_population("pre", cell)

    def test_initialization_freezes_topology(self) -> None:
        network = make_runtime_network()
        network.init_state()
        self.assertIs(network.init_state(), network)
        with self.assertRaisesRegex(RuntimeError, "after Network initialization"):
            network.add_population("extra", braincell.NetStim())
        with self.assertRaisesRegex(RuntimeError, "after Network initialization"):
            network.connect(
                "late",
                source=network.populations["pre"],
                synapse=network.populations["post"].synapses["exp"],
            )

    def test_batch_size_is_explicitly_unsupported(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "batch execution"):
            make_runtime_network().init_state(batch_size=2)

    def test_zero_delay_delivers_on_source_boundary(self) -> None:
        result = make_runtime_network().run(dt=0.1 * u.ms, duration=0.4 * u.ms)
        conductance = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))
        first_nonzero = int(np.flatnonzero(conductance[:, 0] > 0.0)[0])
        self.assertEqual(first_nonzero, 1)

    def test_heterogeneous_delay_routes_rows_independently(self) -> None:
        network = make_runtime_network(delay=[0.0, 0.3] * u.ms)
        result = network.run(dt=0.1 * u.ms, duration=0.6 * u.ms)
        conductance = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))
        self.assertEqual(int(np.flatnonzero(conductance[:, 0] > 0.0)[0]), 1)
        self.assertEqual(int(np.flatnonzero(conductance[:, 1] > 0.0)[0]), 4)

    def test_split_run_preserves_events_in_flight(self) -> None:
        continuous = make_runtime_network(delay=0.3 * u.ms).run(dt=0.1 * u.ms, duration=0.7 * u.ms)
        split_network = make_runtime_network(delay=0.3 * u.ms)
        split = braincell.NetworkResult.concat(
            (
                split_network.run(dt=0.1 * u.ms, duration=0.2 * u.ms),
                split_network.run(dt=0.1 * u.ms, duration=0.5 * u.ms),
            )
        )
        np.testing.assert_allclose(
            split.samples["post"]["g"].values.to_decimal(u.uS),
            continuous.samples["post"]["g"].values.to_decimal(u.uS),
            rtol=1e-6,
        )

    def test_reset_state_restarts_and_discards_events_in_flight(self) -> None:
        network = make_runtime_network(delay=0.3 * u.ms)
        first = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        network.reset_state()
        second = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        np.testing.assert_allclose(
            first.samples["post"]["g"].values.to_decimal(u.uS),
            second.samples["post"]["g"].values.to_decimal(u.uS),
        )

    def test_event_backend_auto_matches_scatter(self) -> None:
        auto = make_runtime_network(delay=[0.0, 0.2] * u.ms).run(
            dt=0.1 * u.ms,
            duration=0.5 * u.ms,
            event_backend="auto",
        )
        scatter = make_runtime_network(delay=[0.0, 0.2] * u.ms).run(
            dt=0.1 * u.ms,
            duration=0.5 * u.ms,
            event_backend="scatter",
        )
        np.testing.assert_allclose(
            auto.samples["post"]["g"].values.to_decimal(u.uS),
            scatter.samples["post"]["g"].values.to_decimal(u.uS),
            rtol=1e-6,
        )

    def test_run_setup_and_compiled_loop_are_reused(self) -> None:
        network = make_runtime_network()
        network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        setup_count = len(network._run_setup_cache)
        loop_count = len(network._network_run_loop_cache)
        network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(network._run_setup_cache), setup_count)
        self.assertEqual(len(network._network_run_loop_cache), loop_count)


if __name__ == "__main__":
    unittest.main()
