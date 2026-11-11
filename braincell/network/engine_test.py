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
from types import ModuleType
from unittest.mock import patch

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell._multi_compartment.run_test import (
    DT,
    WEIGHT,
    _float64,
    assert_same_trajectory,
    make_event_cell,
    split_steps,
)
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
        network = Network("network")
        network.add_population("cell", cell)
        network.init_state()
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="factor"))
        self.assertIs(
            network.trainables.parameters().states()["cell.factor"], cell.trainables.parameters().states()["factor"]
        )

    def test_prepared_network_keeps_weight_and_threshold_gradients(self):
        self._check_prepared_network_gradients("scatter")

    def test_prepared_network_keeps_weight_and_threshold_gradients_brainevent(self):
        self._check_prepared_network_gradients("brainevent")

    def test_event_backends_agree_elementwise(self):
        traces = {}
        for backend in ("scatter", "brainevent", "auto"):
            result = make_runtime_network(delay=0.2 * u.ms).run(
                dt=0.1 * u.ms, duration=1.0 * u.ms, event_backend=backend
            )
            traces[backend] = result.samples["post"]["g"].values.to_decimal(u.uS)
        self.assertGreater(float(np.max(traces["scatter"])), 0.0)
        np.testing.assert_allclose(traces["brainevent"], traces["scatter"], rtol=1e-6)
        # "auto" resolves to the scatter path, so it is bit-identical.
        np.testing.assert_array_equal(traces["auto"], traces["scatter"])

    def test_brainevent_without_csrmv_is_rejected_at_prepare_run(self):
        network = make_runtime_network(delay=0.2 * u.ms)
        with patch.dict("sys.modules", {"brainevent": ModuleType("brainevent")}):
            with self.assertRaisesRegex(RuntimeError, "event_backend='scatter'"):
                network.prepare_run(dt=0.1 * u.ms, event_backend="brainevent")

    def _check_prepared_network_gradients(self, backend):
        network = make_runtime_network(delay=0.2 * u.ms)
        network.init_state()
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


class SplitNetworkRunTest(unittest.TestCase):
    """CELL-34: ``Network.run`` split at any step, event times included, matches one run."""

    def setUp(self):
        self.enterContext(_float64())

    @staticmethod
    def _run(solver, source_kwargs, steps, *, delay=0.0 * u.ms):
        cell, _ = make_event_cell(solver)
        network = Network("split")
        source = network.add_population("stim", braincell.NetStim(**source_kwargs))
        target = network.add_population("post", cell)
        network.connect("drive", source=source, synapse=target.synapses["syn"], weight=WEIGHT, delay=delay)
        parts = tuple(network.run(dt=DT, duration=n * DT) for n in steps)
        return braincell.NetworkResult.concat(parts).samples["post"]

    def _check(self, source_kwargs, steps, **kwargs):
        for solver in ("staggered", "rk4"):
            with self.subTest(solver=solver, steps=steps, **kwargs):
                whole = self._run(solver, source_kwargs, (sum(steps),), **kwargs)
                split = self._run(solver, source_kwargs, steps, **kwargs)
                self.assertGreater(float(np.max(split["g"].values.to_decimal(u.uS))), 0.0)
                assert_same_trajectory(split, whole)

    def test_fixed_interval_events_inside_and_on_segment_ends(self):
        # Events every 0.5 ms from 0.5 ms; 1 ms segments put every other one
        # on a segment end. A 0.5 ms delay moves the arrivals by 20 steps.
        source = dict(start=0.5 * u.ms, interval=0.5 * u.ms, number=9)
        for delay in (0.0, 0.5):
            self._check(source, (40,) * 5, delay=delay * u.ms)

    def test_noisy_events_on_segment_ends(self):
        source = dict(start=0.5 * u.ms, interval=0.5 * u.ms, number=9, noise=0.5, seed=34)
        times = np.asarray(braincell.NetStim(**source).event_times.to_decimal(u.ms))[0]
        event_steps = np.unique(np.floor(times / 0.025 + 0.5).astype(int))
        # Split on the first three event steps, plus one step inside a segment.
        points = (*event_steps[:3], event_steps[0] + 7)
        self._check(source, split_steps(200, points))

    def test_random_split_points_match_one_run(self):
        rng = np.random.default_rng(34)
        event_steps = np.arange(10, 200, 10)
        points = (*rng.choice(event_steps, 2, replace=False), rng.choice(np.setdiff1d(np.arange(1, 200), event_steps)))
        self._check(dict(start=0.25 * u.ms, interval=0.25 * u.ms, number=19), split_steps(200, points))



def _resting_one_cv_cell(kind, solver, pop_size=2):
    """A soma 20 um x 10 um with a leak at its reversal, -65 mV."""
    if kind == "lumped":
        cell = braincell.Cell.point(pop_size, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver=solver)
    else:
        morphology = braincell.Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um)
        cell = braincell.Cell(
            morphology, pop_size=pop_size, cv_policy=braincell.CVPerBranch(), V_init=-65.0 * u.mV, solver=solver
        )
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-65.0 * u.mV))
    return cell


_DELTA_PATHS = (("lumped", "rk4"), ("cable", "rk4"), ("cable", "staggered"))


class NetworkDeltaInputTest(unittest.TestCase):
    """CELL-33: Network-managed cells add delta inputs; a later input drops the compiled loops."""

    def test_delta_projection_drives_update_in_successive_loops(self):
        # On main the lumped and "staggered" cells never consumed a kick (the
        # second loop raised ValueError: key already defined), and the second
        # loop of the "rk4" cable cell raised UnexpectedTracerError: update()
        # reused the loop traced with the first loop's input.
        import brainpy

        kicks = np.asarray([[1.0, 0.5], [2.0, 0.0], [0.0, 1.5]])
        for kind, solver in _DELTA_PATHS:
            with self.subTest(kind=kind, solver=solver), _float64():
                finals = []
                for managed in (True, False):
                    cell = _resting_one_cv_cell(kind, solver)
                    shape = (2,) if kind == "lumped" else (2, 1)
                    proj = brainpy.state.DeltaProj(comm=lambda x, shape=shape: x.reshape(shape) * u.mV, post=cell)
                    if managed:
                        network = Network("kicked")
                        network.add_population("cells", cell)
                        network.prepare_run(dt=DT)
                        advance = network.update
                    else:
                        cell.init_state()
                        advance = cell.update

                    def step(index, kick, proj=proj, advance=advance):
                        with brainstate.environ.context(t=index * DT):
                            proj(kick)
                            advance()
                        return index

                    with brainstate.environ.context(dt=DT):
                        for first in (0, 3):
                            brainstate.transform.for_loop(step, np.arange(first, first + 3), kicks)
                    finals.append(cell.V.value.to_decimal(u.mV).reshape(-1))
                # Six kicks: 3 + 2 mV per member, the leak relaxing them a little.
                np.testing.assert_allclose(finals[0], [-59.0, -61.0], atol=0.2)
                np.testing.assert_allclose(finals[0], finals[1], rtol=0.0, atol=1e-12)

    def test_run_includes_inputs_added_after_a_run(self):
        # On main the second Network.run reused the loop traced without them.
        def kick():
            return 0.25 * u.mV

        def drive(V):
            return u.math.full_like(u.get_mantissa(V), 10.0) * u.uA / u.cm**2

        for kind, solver in _DELTA_PATHS:
            for entry in ("delta", "current"):
                with self.subTest(kind=kind, solver=solver, input=entry):
                    cell = _resting_one_cv_cell(kind, solver)
                    network = Network("late_input")
                    network.add_population("cells", cell)
                    network.run(dt=DT, duration=0.5 * u.ms)
                    np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -65.0)
                    if entry == "delta":
                        cell.add_delta_input("kick", kick)
                    else:
                        cell.add_current_input("drive", drive)
                    network.run(dt=DT, duration=0.5 * u.ms)
                    # 20 kicks of 0.25 mV, or 10 uA/cm^2 for 0.5 ms, less the leak.
                    self.assertGreater(float(np.min(cell.V.value.to_decimal(u.mV))), -61.0)


if __name__ == "__main__":
    unittest.main()
