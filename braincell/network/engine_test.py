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
import re
import unittest
import warnings
from types import ModuleType
from unittest.mock import patch

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import braincell
from braincell._random_test import default_key_data, drawn_seeds
from braincell._multi_compartment.lifecycle_test import loop_compiles
from braincell._multi_compartment.run_test import (
    DT,
    WEIGHT,
    _float64,
    _rule_cell,
    _rule_updates,
    assert_same_trajectory,
    axial_builds,
    hh_y_cable,
    long_run_cell,
    make_event_cell,
    split_steps,
    update_loop_voltage,
)
from braincell.filter import AllRegion, at
from braincell.filter import RandomSamples
from braincell.network import Network
from braincell.network.connection import independent
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
        # The spike of step 0 acts at the start of step 1, so sample 1 (t_1) has no g yet (CELL-81; main 51f669f: 1).
        self.assertEqual(first_nonzero, 2)

    def test_heterogeneous_delay_routes_rows_independently(self) -> None:
        network = make_runtime_network(delay=[0.0, 0.3] * u.ms)
        result = network.run(dt=0.1 * u.ms, duration=0.6 * u.ms)
        conductance = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))
        # Sample 1 + 1 + k for k delay steps, as NEURON (CELL-75, CELL-81; main 51f669f: 1 for the delay-0 row).
        self.assertEqual(int(np.flatnonzero(conductance[:, 0] > 0.0)[0]), 2)
        self.assertEqual(int(np.flatnonzero(conductance[:, 1] > 0.0)[0]), 5)

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


#: The steps of the three events of ``_clocked_network``: 100, 409.6 and 800 ms
#: at dt 0.025 ms, whole multiples of dt in float64.
_EVENT_STEPS = (4000, 16384, 32000)
_CLOCK_STEPS = 32010


def _clock_cell():
    """A passive soma (20 um x 10 um, leak 0.1 mS/cm^2 at -65 mV) with an ExpSyn (tau 2 ms, e 0 mV); records V and g."""
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1), V_init=-65.0 * u.mV
    )
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at(0, 0.5), braincell.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV))
    cell.record("V", "V")
    cell.record("g", braincell.observe.synapse(name="exp").state("g"))
    return cell


def _clocked_network():
    """``_clock_cell`` driven by an EventSequence at ``_EVENT_STEPS`` (weight 0.001 uS, no delay); ``(network, cell)``."""
    cell = _clock_cell()
    network = Network("clock")
    post = network.add_population("post", cell)
    stim = network.add_population("stim", braincell.EventSequence.from_times([np.asarray(_EVENT_STEPS) * DT]))
    network.connect("drive", source=stim, synapse=post.synapses["exp"], weight=0.001 * u.uS)
    return network, cell


def _clock_samples(results):
    """``(t in ms, V in mV, g in uS)`` per step of one or more ``Network.run`` results of ``_clocked_network``."""
    result = braincell.NetworkResult.concat(tuple(results))
    samples = result.samples["post"]
    return (
        np.asarray(result.time.to_decimal(u.ms)),
        np.asarray(samples["V"].values.to_decimal(u.mV)).reshape(-1),
        np.asarray(samples["g"].values.to_decimal(u.uS)).reshape(-1),
    )


def _update_loop(network, cell, n_steps):
    """``(t before each update in ms, V and g after it)``, from one compiled loop of ``n_steps`` ``update()`` calls."""
    (synapse,) = cell.runtime_objects(braincell.synapse.ExpSyn).values()

    def step(_):
        t = cell.current_time.to_decimal(u.ms)
        network.update()
        return t, cell.V.value.to_decimal(u.mV).reshape(-1)[0], synapse.g.value.to_decimal(u.uS).reshape(-1)[0]

    loop = brainstate.transform.jit(lambda: brainstate.transform.for_loop(step, np.arange(n_steps)))
    return tuple(np.asarray(values) for values in loop())


def _jumps(g):
    """Indices where ``g`` jumps up by an event."""
    return np.flatnonzero(np.diff(g, prepend=0.0) > 1e-4).tolist()


class NetworkClockTest(unittest.TestCase):
    """CELL-45: one run, split runs and ``update()`` loops step at ``t = step·dt``, with no accumulated time."""

    def test_update_loops_split_runs_and_one_run_step_at_the_same_times(self):
        # main: in float32, 32010 update() calls ended at 800.443176 ms and
        # the events acted at updates 4000, 16387 and 31992.
        for precision, float_type, rtol in ((32, np.float32, 1e-6), (64, np.float64, 1e-12)):
            context = _float64() if precision == 64 else brainstate.environ.context(precision=32)
            with self.subTest(precision=precision), context:
                network, _ = _clocked_network()
                whole = _clock_samples((network.run(dt=DT, duration=_CLOCK_STEPS * DT),))
                network, _ = _clocked_network()
                steps = (800,) * (_CLOCK_STEPS // 800) + (_CLOCK_STEPS % 800,)
                split = _clock_samples(network.run(dt=DT, duration=n * DT) for n in steps)
                network, cell = _clocked_network()
                network.prepare_run(dt=DT)
                with brainstate.environ.context(dt=DT):
                    times, V, g = _update_loop(network, cell, _CLOCK_STEPS)

                expected_times = np.arange(_CLOCK_STEPS).astype(float_type) * float_type(0.025)
                for t in (whole[0], split[0], times):
                    np.testing.assert_array_equal(t, expected_times)
                np.testing.assert_array_equal(cell._clock_step.value, _CLOCK_STEPS)
                np.testing.assert_array_equal(
                    cell.current_time.to_decimal(u.ms), float_type(_CLOCK_STEPS) * float_type(0.025)
                )
                # A sample is taken before its step, update() is read after it.
                self.assertEqual(_jumps(whole[2]), [step + 1 for step in _EVENT_STEPS])
                self.assertEqual(_jumps(g), list(_EVENT_STEPS))
                for name, actual, expected in (("V", split[1], whole[1]), ("g", split[2], whole[2])):
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0, err_msg=name)
                for name, actual, expected in (("V", V[:-1], whole[1][1:]), ("g", g[:-1], whole[2][1:])):
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0, err_msg=name)

    def test_reset_state_restarts_the_clock_of_update(self):
        network, cell = _clocked_network()
        network.prepare_run(dt=DT)
        with brainstate.environ.context(dt=DT):
            first = _update_loop(network, cell, 50)
            network.reset_state()
            np.testing.assert_array_equal(cell.current_time.to_decimal(u.ms), 0.0)
            again = _update_loop(network, cell, 50)
        np.testing.assert_array_equal(again[0], first[0])
        np.testing.assert_array_equal(cell.current_time.to_decimal(u.ms), np.float32(50) * np.float32(0.025))
        # run() after the update() loop continues its count.
        result = network.run(dt=DT, duration=10 * DT)
        np.testing.assert_array_equal(
            result.time.to_decimal(u.ms), np.arange(50, 60).astype(np.float32) * np.float32(0.025)
        )

    def test_the_step_count_stops_at_its_int32_limit(self):
        network, cell = _clocked_network()
        network.run(dt=DT, duration=DT)
        cell._clock_step.value = np.int32(2**31 - 10)
        before = np.asarray(cell.current_time.to_decimal(u.ms))
        with self.assertRaisesRegex(ValueError, r"2147483647 steps \(14\.9 h at dt"):
            network.run(dt=DT, duration=20 * DT)
        np.testing.assert_array_equal(cell._clock_step.value, 2**31 - 10)
        np.testing.assert_array_equal(cell.current_time.to_decimal(u.ms), before)
        network.run(dt=DT, duration=9 * DT)
        np.testing.assert_array_equal(cell._clock_step.value, 2**31 - 1)
        network.reset_state()
        network.run(dt=DT, duration=20 * DT)
        np.testing.assert_array_equal(cell._clock_step.value, 20)

        network, cell = _clocked_network()
        network.prepare_run(dt=DT)
        cell._clock_step.value = np.int32(2**31 - 1)
        with brainstate.environ.context(dt=DT), self.assertRaisesRegex(ValueError, "2147483647"):
            network.update()

        network = Network("sources")
        network.add_population("stim", braincell.NetStim())
        network._source_step = 2**31 - 10
        with self.assertRaisesRegex(ValueError, "2147483647"):
            network.run(dt=DT, duration=20 * DT)

    def test_segments_reuse_the_compiled_loops(self):
        # The step count is a state, not a key of the loop caches, nor a
        # constant of their traces (L40): counted with the step body's calls.
        def counted(cell):
            calls = [0]
            begin_step = cell._begin_step

            def counting(*args, **kwargs):
                calls[0] += 1
                return begin_step(*args, **kwargs)

            cell._begin_step = counting
            return calls

        network, cell = _clocked_network()
        network.init_state()
        calls = counted(cell)
        for index in range(5):
            if index == 3:
                network.reset_state()
            network.run(dt=DT, duration=20 * DT)
        self.assertEqual(calls[0], 1)
        self.assertEqual(len(network._network_run_loop_cache), 1)
        np.testing.assert_array_equal(cell._clock_step.value, 40)

        network, cell = _clocked_network()
        network.prepare_run(dt=DT)
        calls = counted(cell)
        with brainstate.environ.context(dt=DT):
            for index in range(5):
                if index == 3:
                    network.reset_state()
                network.update()
        self.assertEqual(calls[0], 1)
        np.testing.assert_array_equal(cell._clock_step.value, 2)

    def test_a_cell_run_alone_before_joining_keeps_its_time(self):
        # A Cell that ran on its own at another dt joins with its time: the
        # Network counts from there.
        cell = _clock_cell()
        cell.run(dt=2 * DT, duration=6 * DT)
        network = Network("late")
        network.add_population("post", cell)
        result = network.run(dt=DT, duration=4 * DT)
        np.testing.assert_allclose(result.time.to_decimal(u.ms), 0.15 + 0.025 * np.arange(4), rtol=1e-6)
        np.testing.assert_allclose(cell.current_time.to_decimal(u.ms), 0.25, rtol=1e-6)
        np.testing.assert_array_equal(cell._clock_step.value, 4)

        # Two such Cells at the same time on different clocks share the first's.
        first, second = _clock_cell(), _clock_cell()
        first.run(dt=DT, duration=10 * DT)
        second.run(dt=2 * DT, duration=10 * DT)
        network = Network("joined")
        network.add_population("first", first)
        network.add_population("second", second)
        network.run(dt=DT, duration=4 * DT)
        for cell in (first, second):
            np.testing.assert_array_equal(cell._clock_step.value, 14)
            np.testing.assert_array_equal(cell.current_time.to_decimal(u.ms), np.float32(14) * np.float32(0.025))
        # At different times they cannot run together.
        late = _clock_cell()
        late.run(dt=DT, duration=12 * DT)
        network = Network("apart")
        network.add_population("first", _clock_cell())
        network.add_population("late", late)
        with self.assertRaisesRegex(ValueError, "same current_time"):
            network.run(dt=DT, duration=4 * DT)


#: CELL-58: the events of the off-grid probe and the boundaries (ms) they act on
#: from t = 0 at dt 0.05 ms, floor(T/dt + 0.5) by hand: 6, 10.25 -> 10, 12,
#: 16.75 -> 17, 20 and 20.25 -> 20 steps, with the arrivals at each.
_LATE_EVENTS = (0.3, 0.5125, 0.6, 0.8375, 1.0, 1.0125)
_LATE_BOUNDARIES = ([0.3, 0.5, 0.6, 0.85, 1.0], [1, 1, 1, 1, 2])
_NET_DT = 2 * DT
_NET_STEPS = 22


def _late_cell(segments, *, jitted=False):
    """``(cell, source)``: a Cell and its EventSequence, the Cell run alone for ``segments`` (``(dt, steps)`` pairs).

    The Cell is a passive 1-CV cell at rest whose ExpSyn (tau 1e9 ms,
    e = rest) gains 1 uS per arrival of ``_LATE_EVENTS``. ``segments=None``
    leaves it uninitialized; ``jitted`` runs each segment inside
    ``brainstate.transform.jit``.
    """
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1), V_init=-65.0 * u.mV
    )
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at(0, 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=1e9 * u.ms, e=-65.0 * u.mV))
    cell.record("g", braincell.observe.synapse(name="s").state("g"))
    cell.record("V", "V")
    source = braincell.EventSequence.from_times([np.asarray(_LATE_EVENTS) * u.ms])
    braincell.connect("c", source=source, synapse=cell.synapses["s"], weight=1.0 * u.uS)
    if segments is None:
        return cell, source
    cell.init_state()
    for dt, steps in segments:
        if jitted:
            brainstate.transform.jit(lambda: cell.run(dt=dt, duration=steps * dt).stop_time)()
        else:
            cell.run(dt=dt, duration=steps * dt)
    return cell, source


def _late_network(segments):
    """A Cell run alone for ``segments`` (``(dt, steps)`` pairs), then a Network of it and its EventSequence.

    The Cell is ``_late_cell``'s; ``(network, cell)``.
    """
    cell, source = _late_cell(segments)
    network = Network("late")
    network.add_population("post", cell)
    network.add_population("stim", source)
    return network, cell


def _synapse_g(cell) -> float:
    (synapse,) = cell.runtime_objects(braincell.synapse.ExpSyn).values()
    return float(np.asarray(synapse.g.value.to_decimal(u.uS), dtype=np.float64).reshape(-1)[0])


def _late_state(network, cell) -> dict:
    """Every state a rejected ``run`` or ``prepare_run`` must leave as it found it."""
    (synapse,) = cell.runtime_objects(braincell.synapse.ExpSyn).values()
    return {
        "origin": np.asarray(cell._clock_origin.value.to_decimal(u.ms)).copy(),
        "step": np.asarray(cell._clock_step.value).copy(),
        "clock dt": np.asarray(cell._clock_dt.value.to_decimal(u.ms)).copy(),
        "V": np.asarray(cell.V.value.to_decimal(u.mV)).copy(),
        "g": np.asarray(synapse.g.value.to_decimal(u.uS)).copy(),
        "prepared": np.asarray(cell._next_synapse_inputs_prepared.value).copy(),
        "float64 clock": getattr(cell, "_run_float64_clock", None),
        "initialized": network._initialized,
        "runtime config": network._runtime_config,
        "setups": len(network._run_setup_cache),
        "loops": len(network._network_run_loop_cache),
        "prepared run": network._prepared_run,
    }


class NetworkOffGridStartTest(unittest.TestCase):
    """CELL-58: a Cell that ran alone to a time off the Network's dt grid cannot start it; on the grid it joins."""

    def _contexts(self):
        return (("float32", brainstate.environ.context(precision=32)), ("float64", _float64()))

    def test_a_cell_off_the_grid_is_rejected_at_entry_and_nothing_changes(self):
        # main ran these: 0 of 22 samples recorded, and the 0.3 ms event acted
        # at 0.275 ms (half a step early), with no error.
        histories = {"0.075 ms": ((DT, 3),), "0.175 ms": ((DT, 3), (_NET_DT, 2))}
        entries = {
            "run": lambda network: network.run(dt=_NET_DT, duration=_NET_STEPS * _NET_DT),
            "prepare_run": lambda network: network.prepare_run(dt=_NET_DT),
            "direct": lambda network: network.run(
                dt=_NET_DT, duration=_NET_STEPS * _NET_DT, scheduled_delivery="direct"
            ),
        }
        for precision, context in self._contexts():
            with context:
                for time, segments in histories.items():
                    for entry, call in entries.items():
                        with self.subTest(precision=precision, start=time, entry=entry):
                            network, cell = _late_network(segments)
                            before = _late_state(network, cell)
                            message = (
                                rf"cannot start population 'post' at t = {re.escape(time[:-3])} ms, which is not "
                                r"a multiple of dt = 0\.05 ms.*Network\.reset_state\(\).*before adding it"
                            )
                            with self.assertRaisesRegex(ValueError, message):
                                call(network)
                            after = _late_state(network, cell)
                            for key, value in before.items():
                                np.testing.assert_array_equal(after[key], value, err_msg=key)

    def test_the_ways_out_of_the_rejection(self):
        for precision, context in self._contexts():
            with self.subTest(precision=precision), context:
                fresh, _ = _late_network(())
                expected = fresh.run(dt=_NET_DT, duration=_NET_STEPS * _NET_DT)
                # reset_state() starts the rejected Network from t = 0.
                network, cell = _late_network(((DT, 3),))
                with self.assertRaisesRegex(ValueError, "not a multiple of dt"):
                    network.run(dt=_NET_DT, duration=_NET_STEPS * _NET_DT)
                network.reset_state()
                result = network.run(dt=_NET_DT, duration=_NET_STEPS * _NET_DT)
                np.testing.assert_allclose(result.time.to_decimal(u.ms), expected.time.to_decimal(u.ms), rtol=1e-12)
                for name in ("g", "V"):
                    np.testing.assert_allclose(
                        result.samples["post"][name].values.mantissa,
                        expected.samples["post"][name].values.mantissa,
                        rtol=1e-12,
                        err_msg=name,
                    )
                # A Cell run alone to a multiple of dt (4 steps of 0.025 ms) joins.
                network, cell = _late_network(((DT, 4),))
                self.assertEqual(self._boundaries(network, cell, "run"), _LATE_BOUNDARIES)

    def _boundaries(self, network, cell, driver):
        """The boundaries (ms) of the steps whose arrivals raised g, and the arrivals at each.

        Read from the synapse state (L93): ``update`` reads it after each
        step; the runs read the recording, after checking it holds one
        sample per step and that g rose by one per event over the run.
        """
        g0 = _synapse_g(cell)
        if driver == "update":
            network.prepare_run(dt=_NET_DT)
            times, rises, previous = [], [], g0
            for _ in range(_NET_STEPS):
                time = float(np.asarray(cell.current_time.to_decimal(u.ms), dtype=np.float64))
                network.update()
                g = _synapse_g(cell)
                if g - previous > 0.5:
                    times.append(round(time, 6))
                    rises.append(int(np.rint(g - previous)))
                previous = g
        else:
            options = {"scheduled_delivery": "direct"} if driver == "direct" else {}
            result = network.run(dt=_NET_DT, duration=_NET_STEPS * _NET_DT, **options)
            g = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS), dtype=np.float64).reshape(-1)
            self.assertEqual(g.shape, (_NET_STEPS,))
            # The sample at t_k precedes the arrivals of boundary t_k.
            steps_ms = np.asarray(result.time.to_decimal(u.ms), dtype=np.float64)
            rise = np.diff(np.append(g, _synapse_g(cell)))
            jumps = np.flatnonzero(rise > 0.5)
            times = np.round(steps_ms[jumps], 6).tolist()
            rises = np.rint(rise[jumps]).astype(int).tolist()
        self.assertEqual(round(_synapse_g(cell) - g0), len(_LATE_EVENTS))
        return times, rises

    def test_starts_on_the_grid_deliver_each_event_once(self):
        # Fresh, and 10 steps of 0.025 ms alone (0.25 ms, 5 steps of 0.05 ms):
        # as on main, each event acts once on its boundary from t = 0.
        for precision, context in self._contexts():
            with context:
                for start, segments in (("fresh", ()), ("0.25 ms", ((DT, 10),))):
                    for driver in ("run", "update", "direct"):
                        with self.subTest(precision=precision, start=start, driver=driver):
                            network, cell = _late_network(segments)
                            self.assertEqual(self._boundaries(network, cell, driver), _LATE_BOUNDARIES)

    def test_the_grid_test_agrees_with_exact_arithmetic(self):
        # Time reached alone / Network dt, exactly: 0.075/0.05 = 1.5, 0.3/0.05
        # = 6, 0.32/0.03 = 10.67, 0.3/0.025 = 12; in float32 800.25/0.05 =
        # 16005 and 800.275/0.05 = 16005.5, after 32010 and 32011 steps.
        cases = (
            (64, ((0.025, 3),), 0.05, False),
            (64, ((0.1, 3),), 0.05, True),
            (64, ((0.02, 16),), 0.03, False),
            (64, ((0.05, 2), (0.1, 2)), 0.025, True),
            (32, ((0.025, 32010),), 0.05, True),
            (32, ((0.025, 32011),), 0.05, False),
        )
        for precision, segments, dt, on_grid in cases:
            context = _float64() if precision == 64 else brainstate.environ.context(precision=32)
            with self.subTest(precision=precision, segments=segments, dt=dt), context:
                cell = braincell.Cell.point(1, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
                cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
                cell.init_state()
                for segment_dt, steps in segments:
                    cell.run(dt=segment_dt * u.ms, duration=steps * segment_dt * u.ms)
                network = Network("late")
                network.add_population("post", cell)
                if on_grid:
                    network.prepare_run(dt=dt * u.ms)
                else:
                    with self.assertRaisesRegex(ValueError, "not a multiple of dt"):
                        network.prepare_run(dt=dt * u.ms)

    def test_another_dt_after_the_first_run_still_raises_its_runtime_error(self):
        # After 3 steps of 0.05 ms the Network is at 0.15 ms, 1.5 steps of 0.1
        # ms: the fixed dt is the error, as before.
        network, _ = _late_network(())
        network.run(dt=_NET_DT, duration=3 * _NET_DT)
        with self.assertRaisesRegex(RuntimeError, "configuration is fixed"):
            network.run(dt=2 * _NET_DT, duration=2 * _NET_DT)
        with self.assertRaisesRegex(RuntimeError, "configuration is fixed"):
            network.prepare_run(dt=2 * _NET_DT)


def _apart_network(*cells):
    """A Network of ``_late_cell`` pairs ``(cell, source)``, as populations ``c0``, ``c1``, … and ``s0``, ``s1``, …."""
    network = Network("apart")
    for index, (cell, source) in enumerate(cells):
        network.add_population(f"c{index}", cell)
        network.add_population(f"s{index}", source)
    return network


def _apart_state(network, cells) -> dict:
    """``_late_state`` of each Cell (only that it is not initialized, for one that is not) and of the Network."""
    state = {
        "initialized": network._initialized,
        "runtime config": network._runtime_config,
        "setups": len(network._run_setup_cache),
        "loops": len(network._network_run_loop_cache),
        "prepared run": network._prepared_run,
        "plans": network._scheduled_plans,
    }
    for index, cell in enumerate(cells):
        state[f"c{index} initialized"] = cell._initialized
        if cell._initialized:
            state.update({f"c{index} {key}": value for key, value in _late_state(network, cell).items()})
    return state


#: CELL-60: 1000 steps of 100 ms alone reach 100000 ms; one more of 0.5 ms, 100000.5 ms.
_AT_100000 = ((100.0 * u.ms, 1000),)
_AT_100000_5 = ((100.0 * u.ms, 1000), (0.5 * u.ms, 1))


class NetworkSameTimeTest(unittest.TestCase):
    """CELL-60: Cells at different times, beyond the run precision, cannot run together; within it they join."""

    def _contexts(self):
        return (("float32", brainstate.environ.context(precision=32)), ("float64", _float64()))

    def _assert_refused(self, network, cells, call, message):
        """``call(network)`` raises ``ValueError`` matching ``message``, and every state is as it was."""
        before = _apart_state(network, cells)
        with self.assertRaises(ValueError) as caught:
            call(network)
        after = _apart_state(network, cells)
        for key, value in before.items():
            np.testing.assert_array_equal(after[key], value, err_msg=key)
        self.assertRegex(str(caught.exception), message)

    def test_cells_at_different_times_are_refused_before_anything_changes(self):
        # main ran 100000 and 100000.5 ms together: the second Cell went back
        # to the first's clock, 20 steps of 0.025 ms, and took again an event
        # it had taken. With a Cell not yet initialized, main raised after
        # initializing it and fixing the configuration.
        entries = {
            "run": lambda network: network.run(dt=DT, duration=4 * DT),
            "prepare_run": lambda network: network.prepare_run(dt=DT),
            "direct": lambda network: network.run(dt=DT, duration=4 * DT, scheduled_delivery="direct"),
        }
        cases = (
            ("100000 ms first", (_AT_100000, _AT_100000_5), ("100000", "100000.5"), tuple(entries)),
            ("100000.5 ms first", (_AT_100000_5, _AT_100000), ("100000.5", "100000"), ("run",)),
            ("a new Cell first", (None, ((DT, 12),)), ("0", "0.3"), ("run", "prepare_run")),
        )
        for precision, context in self._contexts():
            with context:
                for label, histories, times, names in cases:
                    for entry in names:
                        with self.subTest(precision=precision, case=label, entry=entry):
                            cells = [_late_cell(segments) for segments in histories]
                            network = _apart_network(*cells)
                            message = (
                                r"requires its Cell populations at the same current_time: population 'c0' is at "
                                rf"t = {re.escape(times[0])} ms and population 'c1' at t = {re.escape(times[1])} ms\."
                                r".*Network\.reset_state\(\).*before adding them"
                            )
                            self._assert_refused(network, [cell for cell, _ in cells], entries[entry], message)

    def test_times_that_agree_to_the_run_precision_join_on_the_first_clock(self):
        # 32010 steps of 0.025 ms, eager and jitted (L77). The eager run
        # keeps its clock at float64; a jitted one keeps none, so in float32
        # its dt is the float32 0.025 (0.025000000372529), another dt: the
        # times, 800.25 and 800.2500119 ms, agree to the run precision. In
        # float64 both count 0.025 from 0 and have counted the same step.
        # float64, 12 steps of 0.025 ms (0.30000000000000004 ms) and 4 of
        # 0.075 ms (0.3 ms): 1 ulp apart.
        cases = (
            ("float32, jitted", brainstate.environ.context(precision=32), ((DT, 32010),), True, 32010),
            ("float64, jitted", _float64(), ((DT, 32010),), True, 32010),
            ("float64, 1 ulp", _float64(), ((0.075 * u.ms, 4),), False, 12),
        )
        for label, context, second, jitted, step in cases:
            with self.subTest(case=label), context:
                first_cell = _late_cell(((DT, step),))
                second_cell = _late_cell(second, jitted=jitted)
                network = _apart_network(first_cell, second_cell)
                network.run(dt=DT, duration=4 * DT)
                for cell, _ in (first_cell, second_cell):
                    np.testing.assert_array_equal(cell._clock_step.value, step + 4)
                    np.testing.assert_array_equal(cell._clock_dt.value.mantissa, first_cell[0]._clock_dt.value.mantissa)
                    np.testing.assert_array_equal(cell._clock_origin.value.mantissa, 0.0)

    def test_the_tolerance_is_four_eps_of_the_time(self):
        # At 800.25 ms (3201 steps of 0.25 ms alone) 4·eps·800.25 ms is
        # 3.8158e-4 ms in float32 (eps 2**-23) and 7.1078e-13 ms in float64
        # (eps 2**-52), by hand: one more step of 3e-4 or 5e-13 ms joins, one
        # of 5e-4 or 1e-12 ms is refused (main ran them all).
        at = ((0.25 * u.ms, 3201),)
        cases = (
            ("float32", brainstate.environ.context(precision=32), 3e-4, 5e-4, "800.2505"),
            ("float64", _float64(), 5e-13, 1e-12, "800.250000000001"),
        )
        run = lambda network: network.run(dt=DT, duration=DT)  # noqa: E731
        for precision, context, joins, refused, shown in cases:
            with self.subTest(precision=precision), context:
                cells = [_late_cell(at), _late_cell(at + ((joins * u.ms, 1),))]
                run(_apart_network(*cells))
                # Both count from the first's 800.25 ms, at the Network dt.
                for cell, _ in cells:
                    np.testing.assert_array_equal(cell._clock_origin.value.to_decimal(u.ms), 800.25)
                    np.testing.assert_array_equal(cell._clock_step.value, 1)
                cells = [_late_cell(at), _late_cell(at + ((refused * u.ms, 1),))]
                message = rf"'c0' is at t = 800\.25 ms and population 'c1' at t = {re.escape(shown)} ms"
                self._assert_refused(_apart_network(*cells), [cell for cell, _ in cells], run, message)

    def test_clocks_counting_one_dt_compare_their_steps(self):
        # 1000 steps of 100 ms alone, then 1 or 2 of 0.025 ms: both clocks
        # count 0.025 ms from 100000 ms, at steps 1 and 2. 999 steps of 100
        # ms and 4002 of 0.025 ms: from 99900 ms, also on the grid, at step
        # 4002, 100000.05 ms. One step of 0.025 ms is within 4·eps·1e5 ms
        # in float32 (0.0477 ms), but these clocks counted different steps.
        after = lambda origin, steps: ((100.0 * u.ms, origin), (DT, steps))  # noqa: E731
        run = lambda network: network.run(dt=DT, duration=DT)  # noqa: E731
        refused = (
            ("same origin", after(1000, 1), after(1000, 2), ("100000.025", "100000.05")),
            ("origins on the grid", after(1000, 1), after(999, 4002), ("100000.025", "100000.05")),
        )
        for precision, context in self._contexts():
            with context:
                for label, first, second, times in refused:
                    with self.subTest(precision=precision, case=label):
                        cells = [_late_cell(first), _late_cell(second)]
                        message = (
                            rf"'c0' is at t = {re.escape(times[0])} ms and population 'c1' at "
                            rf"t = {re.escape(times[1])} ms"
                        )
                        self._assert_refused(_apart_network(*cells), [cell for cell, _ in cells], run, message)
                with self.subTest(precision=precision, case="the same step from origins on the grid"):
                    cells = [_late_cell(after(1000, 2)), _late_cell(after(999, 4002))]
                    run(_apart_network(*cells))
                    for cell, _ in cells:
                        np.testing.assert_array_equal(cell._clock_step.value, 3)

    def test_a_network_that_ran_runs_again(self):
        # float32: 1000 steps of 100 ms alone (100000 ms), and the same then
        # one of 0.025 ms (100000.025 ms). Their clocks count other dt, and
        # one step is within 4·eps·1e5 ms (0.0477 ms): they join, on the
        # first's clock, the run precision's limit. The second Cell keeps
        # the float64 time it reached beside the shared clock; the members'
        # equal clock states still let the next run start.
        with brainstate.environ.context(precision=32):
            cells = [_late_cell(_AT_100000), _late_cell(_AT_100000 + ((DT, 1),))]
            network = _apart_network(*cells)
            network.run(dt=DT, duration=4 * DT)
            network.run(dt=DT, duration=4 * DT)
            for cell, _ in cells:
                np.testing.assert_array_equal(cell._clock_origin.value.to_decimal(u.ms), 100000.0)
                np.testing.assert_array_equal(cell._clock_step.value, 8)

    def test_the_ways_out_of_the_refusal(self):
        for precision, context in self._contexts():
            with self.subTest(precision=precision), context:
                # Network.reset_state() starts every Cell from t = 0.
                cells = [_late_cell(_AT_100000), _late_cell(_AT_100000_5)]
                network = _apart_network(*cells)
                with self.assertRaisesRegex(ValueError, "same current_time"):
                    network.run(dt=DT, duration=4 * DT)
                network.reset_state()
                network.run(dt=DT, duration=4 * DT)
                for cell, _ in cells:
                    np.testing.assert_array_equal(cell._clock_step.value, 4)
                # A Cell not yet initialized: Network.init_state() first.
                cells = [_late_cell(None), _late_cell(((DT, 12),))]
                network = _apart_network(*cells)
                with self.assertRaisesRegex(ValueError, "same current_time"):
                    network.run(dt=DT, duration=4 * DT)
                network.init_state()
                network.reset_state()
                network.run(dt=DT, duration=4 * DT)
                for cell, _ in cells:
                    np.testing.assert_array_equal(cell._clock_step.value, 4)
                # Or run the earlier Cell on its own to the later one's time.
                cells = [_late_cell(_AT_100000 + ((0.5 * u.ms, 1),)), _late_cell(_AT_100000_5)]
                _apart_network(*cells).run(dt=DT, duration=4 * DT)
                for cell, _ in cells:
                    np.testing.assert_array_equal(
                        cell.current_time.to_decimal(u.ms), cells[0][0].current_time.to_decimal(u.ms)
                    )


#: CELL-57: events in the half steps of 0.025 and 0.05 ms on both sides of 0.25 ms (L94).
_NEAR_EVENTS = (0.2255, 0.2385, 0.245, 0.25, 0.255, 0.2615, 0.27, 0.3)


def _counting_cell(source):
    """``(cell, (source,))``: a passive 1-CV cell at rest whose ExpSyn (tau 1e9 ms, e = rest) gains 1 uS per arrival."""
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1), V_init=-65.0 * u.mV
    )
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at(0, 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=1e9 * u.ms, e=-65.0 * u.mV))
    cell.record("g", braincell.observe.synapse(name="s").state("g"))
    braincell.connect("c", source=source, synapse=cell.synapses["s"], weight=1.0 * u.uS)
    return cell, (source,)


def _sequence(times=_NEAR_EVENTS):
    return braincell.EventSequence.from_times([np.asarray(times) * u.ms])


def _join_network(cell, sources):
    network = Network("join")
    network.add_population("post", cell)
    for index, source in enumerate(sources):
        network.add_population(f"source{index}", source)
    return network


def _now_ms(cell) -> float:
    return round(float(np.asarray(cell.current_time.to_decimal(u.ms), dtype=np.float64)), 6)


def _network_steps(network, cell, driver, dt_ms, n_steps, rises, *, prepare=True):
    """Drive ``network`` ``n_steps`` steps of ``dt_ms``, adding each step's rise of g to ``rises`` (boundary -> count).

    ``"run"``, ``"direct"`` and ``"padded"`` call ``Network.run`` once per
    step; ``"update"`` calls ``update()`` (``"direct update"`` with a
    ``"direct"`` plan) after ``prepare_run`` (unless not ``prepare``), and
    ``"loop"`` runs the ``update()`` calls in one compiled loop; ``"one
    run"`` is one ``Network.run``, read from its recording. g is read from
    the synapse after every step (L93).
    """
    dt = dt_ms * u.ms
    if prepare and driver in ("update", "loop", "direct update"):
        network.prepare_run(dt=dt, scheduled_delivery="direct" if driver == "direct update" else None)
    previous = _synapse_g(cell)
    if driver == "loop":
        start = _now_ms(cell)
        _, _, g = _update_loop(network, cell, n_steps)
        steps = [round(start + k * dt_ms, 6) for k in range(n_steps)]
        values = np.asarray(g, dtype=np.float64)
    elif driver == "one run":
        result = network.run(dt=dt, duration=n_steps * dt)
        steps = np.round(np.asarray(result.time.to_decimal(u.ms), dtype=np.float64), 6).tolist()
        sampled = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS), dtype=np.float64).reshape(-1)
        # The sample at t_k precedes the arrivals of boundary t_k.
        values = np.append(sampled[1:], _synapse_g(cell))
    else:
        steps, values = [], []
        options = {"scheduled_delivery": driver} if driver in ("direct", "padded") else {}
        for _ in range(n_steps):
            steps.append(_now_ms(cell))
            if driver in ("update", "direct update"):
                network.update()
            else:
                network.run(dt=dt, duration=dt, **options)
            values.append(_synapse_g(cell))
    for step, value in zip(steps, values):
        if value - previous > 0.5:
            rises[step] = rises.get(step, 0) + int(np.rint(value - previous))
        previous = value
    return rises


def _cell_steps(cell, dt_ms, n_steps, rises):
    """``Cell.run`` ``n_steps`` steps of ``dt_ms``, one per call, adding each step's rise of g to ``rises``."""
    previous = _synapse_g(cell)
    for _ in range(n_steps):
        step = _now_ms(cell)
        cell.run(dt=dt_ms * u.ms, duration=dt_ms * u.ms)
        value = _synapse_g(cell)
        if value - previous > 0.5:
            rises[step] = rises.get(step, 0) + int(np.rint(value - previous))
        previous = value
    return rises


class NetworkHandOverTest(unittest.TestCase):
    """CELL-57, CELL-55: a Cell that ran alone joins a Network, or leaves one, and each event arrives once.

    A run on its own and a Network step each prepare the boundary at the
    step's start and consume it within the step. So the Network's first step
    prepares the boundary at the time the Cell reached; at another dt it skips
    the events the Cell's run took and delivers the others on its own grid,
    none before that boundary, as in a Network run from t = 0. A Cell run
    after a Network skips what the Network took in the same way. The expected
    boundaries are worked out by hand in each test.
    """

    #: Cell.run 10 x 0.025 ms (its last boundary, 0.225 ms, takes [0.2125,
    #: 0.2375)), then the Network: at 0.025 ms 0.25 ms takes [0.2375, 0.2625)
    #: and the rest are on their steps (0.27 at 0.275 ms); at 0.05 ms 0.25 ms
    #: takes [0.225, 0.275), 0.2385 to 0.27 ms. main bbb7b0a, where the Cell's
    #: run prepared 0.25 ms at its end, put 0.27 ms (5.4 steps) at 0.3 ms at
    #: 0.05 ms: {0.225: 1, 0.25: 5, 0.3: 2}.
    JOINS = {
        0.025: (10, {0.225: 1, 0.25: 5, 0.275: 1, 0.3: 1}),
        0.05: (5, {0.225: 1, 0.25: 6, 0.3: 1}),
    }

    @staticmethod
    def _precision(precision):
        return _float64() if precision == 64 else brainstate.environ.context(precision=32)

    def test_a_cell_that_ran_alone_joins_and_each_event_arrives_once(self):
        for precision in (32, 64):
            for net_dt, (n_steps, expected) in self.JOINS.items():
                for driver in ("run", "one run", "update", "loop", "direct", "padded"):
                    with self.subTest(precision=precision, dt=net_dt, driver=driver), self._precision(precision):
                        cell, sources = _counting_cell(_sequence())
                        cell.init_state()
                        rises = _cell_steps(cell, 0.025, 10, {})
                        network = _join_network(cell, sources)
                        _network_steps(network, cell, driver, net_dt, n_steps, rises)
                        self.assertEqual(rises, expected)
                        self.assertEqual(round(_synapse_g(cell)), len(_NEAR_EVENTS))

    def test_a_netstim_planned_across_a_join_arrives_once(self):
        # NetStim at 0.2255 + 0.013 k ms (k < 6): alone, 0.2255 ms acts at
        # 0.225 ms. At 0.025 ms 0.2385 and 0.2515 ms act at 0.25 ms, 0.2645
        # and 0.2775 ms (10.58 and 11.1 steps) at 0.275 ms, 0.2905 ms (11.62)
        # at 0.3 ms; at 0.05 ms 0.2385, 0.2515 and 0.2645 ms (4.77, 5.03,
        # 5.29 steps) at 0.25 ms, 0.2775 and 0.2905 ms (5.55, 5.81) at 0.3 ms.
        # main: {0.225: 1, 0.25: 2, 0.3: 3} at 0.05 ms.
        expected = {0.025: {0.225: 1, 0.25: 2, 0.275: 2, 0.3: 1}, 0.05: {0.225: 1, 0.25: 3, 0.3: 2}}
        for precision in (32, 64):
            for net_dt, rises_expected in expected.items():
                for driver in ("run", "direct", "padded", "update"):
                    with self.subTest(precision=precision, dt=net_dt, driver=driver), self._precision(precision):
                        stim = braincell.NetStim(size=1, start=0.2255 * u.ms, interval=0.013 * u.ms, number=6)
                        cell, sources = _counting_cell(stim)
                        cell.init_state()
                        rises = _cell_steps(cell, 0.025, 10, {})
                        network = _join_network(cell, sources)
                        _network_steps(network, cell, driver, net_dt, self.JOINS[net_dt][0], rises)
                        self.assertEqual(rises, rises_expected)

    def test_a_rule_updates_once_per_boundary_across_a_join(self):
        # The run alone updates the rule at its steps' boundaries, 0 to 0.225
        # ms; the Network at its own, from 0.25 ms: once per boundary.
        for precision in (32, 64):
            for net_dt, (n_steps, expected) in self.JOINS.items():
                with (
                    self.subTest(precision=precision, dt=net_dt),
                    self._precision(precision),
                    _rule_updates() as updates,
                ):
                    cell, sources = _rule_cell(_NEAR_EVENTS)
                    cell.run(dt=DT, duration=10 * DT)
                    network = _join_network(cell, sources)
                    network.run(dt=net_dt * u.ms, duration=n_steps * net_dt * u.ms)
                    jax.effects_barrier()
                    alone = [round(k * 0.025, 6) for k in range(10)]
                    joined = [round(0.25 + k * net_dt, 6) for k in range(n_steps)]
                    self.assertEqual([round(t, 6) for t, _, _ in updates], alone + joined)
                    self.assertEqual({round(t, 6): int(n) for t, n, _ in updates if n}, expected)
                    self.assertEqual({round(t, 6): int(n) for t, _, n in updates if n}, expected)

    def test_a_joined_cell_with_its_own_live_row_receives_each_event_once(self):
        # The Cell's own spike port feeds its synapse with a 1 ms delay: in the
        # Network that row has a delay queue, which writes each step's (empty)
        # arrivals into the synapse's buffer. Before CELL-57 that write
        # replaced the boundary the run alone had prepared (3 of 8). The Cell
        # never spikes, so g counts the scheduled events, and an ExpSynSTD
        # beside it their number.
        for precision in (32, 64):
            for net_dt, (n_steps, expected) in self.JOINS.items():
                with self.subTest(precision=precision, dt=net_dt), self._precision(precision):
                    cell, sources = _counting_cell(_sequence())
                    cell.place(
                        at(0, 0.5),
                        braincell.mech.Synapse("ExpSynSTD", name="d", tau=1e9 * u.ms, U=0.5, tau_rec=1e9 * u.ms),
                    )
                    braincell.connect("d", source=sources[0], synapse=cell.synapses["d"], weight=1.0 * u.uS)
                    for name in ("s", "d"):
                        braincell.connect(
                            f"self {name}",
                            source=cell.event_outputs["spike"],
                            synapse=cell.synapses[name],
                            weight=1.0 * u.uS,
                            delay=1.0 * u.ms,
                        )
                    cell.init_state()
                    rises = _cell_steps(cell, 0.025, 10, {})
                    network = _join_network(cell, sources)
                    _network_steps(network, cell, "run", net_dt, n_steps, rises)
                    self.assertEqual(rises, expected)
                    (depressing,) = cell.runtime_objects(braincell.synapse.ExpSynSTD).values()
                    x = float(np.asarray(u.get_mantissa(depressing.x.value), dtype=np.float64).reshape(-1)[0])
                    self.assertAlmostEqual(-np.log2(x), len(_NEAR_EVENTS), places=4)

    def test_a_cell_that_leaves_a_network_runs_on_and_each_event_arrives_once(self):
        # The Network's last step consumed its boundary, so the Cell alone
        # prepares 0.25 ms itself, without the events the Network took.
        # 0.2375 ms is a tie, 9.5 steps of 0.025 ms: step 10, which the Network
        # did not reach. main: 10 x 0.025 then 0.05 delivered 0.2255 ms again
        # at 0.25 ms (10 of 9); 5 x 0.05 then 0.025 lost it (8 of 9).
        expected = {
            (0.025, 0.025): {0.225: 1, 0.25: 6, 0.275: 1, 0.3: 1},
            (0.025, 0.05): {0.225: 1, 0.25: 7, 0.3: 1},
            (0.05, 0.025): {0.25: 7, 0.275: 1, 0.3: 1},
            (0.05, 0.05): {0.25: 8, 0.3: 1},
        }
        times = sorted(_NEAR_EVENTS + (0.2375,))
        for precision in (32, 64):
            for (net_dt, cell_dt), rises_expected in expected.items():
                for driver in ("run", "direct"):
                    with (
                        self.subTest(precision=precision, network=net_dt, cell=cell_dt, driver=driver),
                        self._precision(precision),
                    ):
                        cell, sources = _counting_cell(_sequence(times))
                        network = _join_network(cell, sources)
                        network.init_state()
                        rises = _network_steps(network, cell, driver, net_dt, round(0.25 / net_dt), {})
                        del network
                        gc.collect()
                        _cell_steps(cell, cell_dt, 4, rises)
                        self.assertEqual(rises, rises_expected)

    def test_reset_state_after_a_join_counts_from_time_zero(self):
        # From t = 0 at 0.05 ms the events count from time 0 again: 0.2255 to
        # 0.27 ms are 4.51 to 5.4 steps (0.25 ms), 0.3 ms 6.
        for driver in ("update", "direct update", "run", "direct"):
            with self.subTest(driver=driver):
                cell, sources = _counting_cell(_sequence())
                cell.init_state()
                _cell_steps(cell, 0.025, 10, {})
                network = _join_network(cell, sources)
                _network_steps(network, cell, driver, 0.05, 3, {})
                network.reset_state()
                rises = _network_steps(network, cell, driver, 0.05, 8, {}, prepare=False)
                self.assertEqual(rises, {0.25: 7, 0.3: 1})

    def test_a_network_of_fresh_cells_compiles_the_loop_it_did(self):
        # Nothing to take over: one loop, keyed as before. After a join at
        # another dt the runs share one loop, compiled for the hand-over: no
        # boundary is left prepared, so the first run needs no loop of its own
        # (main bbb7b0a: 2 loops).
        cell, sources = _counting_cell(_sequence())
        network = _join_network(cell, sources)
        for _ in range(3):
            network.run(dt=_NET_DT, duration=2 * _NET_DT)
        self.assertEqual([len(key) for key in network._network_run_loop_cache], [4])
        cell, sources = _counting_cell(_sequence())
        cell.run(dt=DT, duration=10 * DT)
        network = _join_network(cell, sources)
        for _ in range(3):
            network.run(dt=_NET_DT, duration=2 * _NET_DT)
        self.assertEqual(len(network._network_run_loop_cache), 1)


def _resting_one_cv_cell(kind, solver, pop_size=2):
    """A soma 20 um x 10 um with a leak at its reversal, -65 mV."""
    if kind == "lumped":
        cell = braincell.Cell.point(
            pop_size, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver=solver
        )
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


def _two_branch_tree() -> braincell.Morphology:
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
    tree = braincell.Morphology.from_root(soma, name="soma")
    tree.soma.dend = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite"
    )
    return tree


class NetworkInitSnapshotTest(unittest.TestCase):
    """MORPH-14: Network cells run on the morphology as it was at ``init_state``."""

    def test_editing_a_shared_morphology_after_init_warns_once_and_runs(self):
        tree = _two_branch_tree()
        first, second = braincell.Cell(tree), braincell.Cell(tree)
        network = Network("shared")
        network.add_population("first", first)
        network.add_population("second", second)
        network.init_state()
        self.assertIsNot(first.morpho, tree)
        self.assertIsNot(second.morpho, first.morpho)

        tree.soma.extra = braincell.Branch.from_lengths(
            lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite"
        )

        def snapshot_warnings():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                network.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
            return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

        (message,) = snapshot_warnings()
        self.assertRegex(message, r"populations 'first', 'second' run on the snapshot.*new Network")
        self.assertEqual(snapshot_warnings(), [])
        self.assertEqual((first.n_cv, second.n_cv), (2, 2))


def _leak(name="leak"):
    return braincell.mech.Channel("IL", name=name, g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV)


def _leak_cell(kind="cable"):
    """A 40 um branch named soma, radius 2 um, two CVs (one on a lumped Cell), with a leak."""
    branch = braincell.Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.0, 2.0] * u.um)
    policy = braincell.Lumped() if kind == "lumped" else braincell.CVPerBranch(2)
    cell = braincell.Cell(braincell.Morphology.from_root(branch, name="soma"), cv_policy=policy)
    cell.paint(AllRegion(), _leak())
    return cell


def _failing_cell():
    """A Cell whose init_state raises: a sodium ion named after the potassium family."""
    cell = _leak_cell()
    cell.paint(AllRegion(), braincell.mech.Ion("SodiumFixed", name="k"))
    return cell


_INIT_ERROR = "conflicts with canonical family key"


class NetworkInitRollbackTest(unittest.TestCase):
    """CELL-37: when a Cell fails init_state, the Cells this call initialized go back to declaring."""

    @staticmethod
    def _declaration(cell):
        # What the Cell is before Network.init_state: its declaration tree,
        # rules and the grid built from them.
        return {
            "morpho": cell.morpho,
            "paint rules": cell._paint_rules,
            "place rules": cell._place_rules,
            "grid": cell._discretization,
        }

    def test_the_cells_before_a_failing_cell_are_declaring_as_before(self):
        for kind in ("cable", "lumped"):
            with self.subTest(kind=kind):
                a, b = _leak_cell(kind), _failing_cell()
                net = Network("n")
                net.add_population("a", a)
                net.add_population("b", b)
                declared = a.morpho
                before = self._declaration(a)
                generation = a._view_generation
                leak = a.channels["leak"]

                with self.assertRaisesRegex(ValueError, _INIT_ERROR):
                    net.init_state()

                self.assertEqual((a._initialized, b._initialized, net._initialized), (False, False, False))
                self.assertIs(a.morpho, declared)
                for (name, now), then in zip(self._declaration(a).items(), before.values()):
                    self.assertTrue(now is then, f"{name} changed")
                self.assertEqual(a._view_generation, generation)
                self.assertEqual(leak.names, ("leak",))  # a selection made before the call is still live
                # The recovery: a declares again, and the Network takes new connections onto it.
                a.paint(AllRegion(), _leak("extra"))
                net.connect(
                    "c",
                    source=net.populations["b"].event_outputs["spike"],
                    target=net.populations["a"],
                    synapse=braincell.mech.Synapse("ExpSyn", name="s"),
                    locations=at("soma", 0.5),
                )
                self.assertFalse(a._initialized)

    def test_a_retry_after_the_failure_initializes_and_runs(self):
        a, b = _leak_cell(), _leak_cell()
        net = Network("n")
        net.add_population("a", a)
        net.add_population("b", b)
        boom = RuntimeError("boom")
        init_runtime = braincell.Cell._init_runtime
        failed = []

        def fail_once(cell, batch_size):
            if cell is b and not failed:
                failed.append(cell)
                raise boom
            return init_runtime(cell, batch_size)

        with patch.object(braincell.Cell, "_init_runtime", autospec=True, side_effect=fail_once):
            with self.assertRaises(RuntimeError) as caught:
                net.init_state()
            self.assertIs(caught.exception, boom)  # the original error, not wrapped
            self.assertEqual((a._initialized, b._initialized, net._initialized), (False, False, False))
            net.init_state()
        self.assertEqual((a._initialized, b._initialized, net._initialized), (True, True, True))
        result = net.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(_voltages(a), -65.0, atol=1e-6)

    def test_rolls_back_in_reverse_order_and_leaves_cells_initialized_earlier(self):
        early = _leak_cell()
        early.init_state()
        early_snapshot = early.morpho
        first, second, failing = _leak_cell(), _leak_cell(), _failing_cell()
        net = Network("n")
        for name, cell in (("first", first), ("early", early), ("second", second), ("failing", failing)):
            net.add_population(name, cell)
        reset = braincell.Cell.reset
        order = []

        def tracked(cell):
            order.append(cell)
            return reset(cell)

        with patch.object(braincell.Cell, "reset", autospec=True, side_effect=tracked):
            with self.assertRaisesRegex(ValueError, _INIT_ERROR):
                net.init_state()

        self.assertEqual([id(cell) for cell in order], [id(second), id(first)])
        self.assertTrue(early._initialized)  # not initialized by this call: left as it was
        self.assertIs(early.morpho, early_snapshot)
        self.assertEqual((first._initialized, second._initialized), (False, False))


def _add_value(cell, entry):
    if entry == "delta":
        cell.add_delta_input("kick", 5.0 * u.mV)
    else:
        cell.add_current_input("pulse", 10.0 * u.uA / u.cm**2)


def _voltages(cell):
    return np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1)


class NetworkValueInputTest(unittest.TestCase):
    """CELL-36: a value input on a population Cell acts on one step of Network.run and Network.update."""

    def _update_loop(self, kind, solver, entry):
        cell = _resting_one_cv_cell(kind, solver)
        cell.init_state()
        _add_value(cell, entry)
        finals = []
        with brainstate.environ.context(dt=DT):
            for step in range(8):
                with brainstate.environ.context(t=step * DT):
                    cell.update()
                if step in (3, 7):
                    finals.append(_voltages(cell))
        return finals

    def test_run_and_update_match_the_cell_update_loop(self):
        # Acceptance 4. On main both loops added the value at every step (a
        # constant of their trace), and so did every later call that reused
        # them: Network.run gave -45.22 mV after four steps of a 5 mV delta.
        for kind, solver in _DELTA_PATHS:
            for entry in ("delta", "current"):
                with self.subTest(kind=kind, solver=solver, input=entry), _float64():
                    expected = self._update_loop(kind, solver, entry)
                    cell = _resting_one_cv_cell(kind, solver)
                    network = Network("valued")
                    network.add_population("cells", cell)
                    network.init_state()
                    _add_value(cell, entry)
                    actual = []
                    for _ in range(2):
                        network.run(dt=DT, duration=4 * DT)
                        actual.append(_voltages(cell))
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

                    cell = _resting_one_cv_cell(kind, solver)
                    network = Network("stepped")
                    network.add_population("cells", cell)
                    network.prepare_run(dt=DT)
                    _add_value(cell, entry)
                    stepped = []
                    with brainstate.environ.context(dt=DT):
                        for step in range(8):
                            network.update()
                            if step in (3, 7):
                                stepped.append(_voltages(cell))
                    np.testing.assert_allclose(stepped, expected, rtol=1e-12, atol=0.0)
                    if entry == "delta":
                        np.testing.assert_allclose(actual[0], -60.11, atol=0.01)

    def test_value_registered_between_runs_does_not_retrace(self):
        with _float64():
            cell = _resting_one_cv_cell("cable", "rk4")
            network = Network("late_value")
            network.add_population("cells", cell)
            network.run(dt=DT, duration=4 * DT)
            loops = dict(network._network_run_loop_cache)
            _add_value(cell, "delta")
            self.assertEqual(dict(network._network_run_loop_cache), loops)
            network.run(dt=DT, duration=4 * DT)
            self.assertGreater(float(np.min(_voltages(cell))), -60.2)
            self.assertNotIn("kick", cell.delta_inputs)

    def test_a_run_that_fails_keeps_its_value_inputs(self):
        cell = _resting_one_cv_cell("lumped", "rk4")
        network = Network("failing")
        network.add_population("cells", cell)
        network.prepare_run(dt=DT)
        cell.add_current_input("pulse", np.ones((3,)) * 10.0 * u.uA / u.cm**2)
        for advance in (lambda: network.update(), lambda: network.run(dt=DT, duration=4 * DT)):
            with brainstate.environ.context(dt=DT), self.assertRaisesRegex(ValueError, "pulse"):
                advance()
            self.assertIn("pulse", cell.current_inputs)


def _event_times(stim) -> np.ndarray:
    model = stim.model if hasattr(stim, "model") else stim
    return np.asarray(model.event_times.to_decimal(u.ms))


def _soma_dend_cell(name, *, random):
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    morpho = braincell.Morphology.from_root(soma, name="soma")
    morpho.soma.dend = braincell.Branch.from_lengths(lengths=[200.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
    v_init = braintools.init.Uniform(-70.0 * u.mV, -60.0 * u.mV) if random else -65.0 * u.mV
    cell = braincell.Cell(morpho, pop_size=3, cv_policy=braincell.CVPerBranch(4), V_init=v_init, name=name)
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV))
    cell.loc(at("soma", 0.5)).record("v", braincell.observe.state("V"))
    if not random:
        cell.place(
            at("soma", 0.5),
            braincell.mech.CurrentClamp(delay=1.0 * u.ms, durations=5.0 * u.ms, amplitudes=0.05 * u.nA),
        )
        return cell, None
    sites = braincell.filter.sample(AllRegion(), number=3)
    cell.place(sites, braincell.mech.Synapse("ExpSyn", name="dend", tau=3.0 * u.ms, e=0.0 * u.mV))
    # A user callback drawing from the default stream at every step.
    cell.place(
        at("soma", 0.5),
        braincell.mech.FunctionClamp(fn=lambda t: brainstate.random.uniform(0.0, 0.02) * u.nA),
    )
    return cell, sites


def _context_experiment() -> dict:
    """Every source of randomness of a Network experiment, none given a seed (CELL-11a acceptance 2)."""
    net = Network("demo", seed=None)
    stim = net.add_population(
        "stim", braincell.NetStim(size=4, number=5, noise=1.0, start=1.0 * u.ms, interval=2.0 * u.ms)
    )
    cell, sites = _soma_dend_cell("post", random=True)
    det, _ = _soma_dend_cell("det", random=False)
    post = net.add_population("post", cell)
    net.add_population("det", det)
    rows = net.connect("drive", source=stim, synapse=post.synapses["ampa"], weight=0.002 * u.uS, pairing=independent(6))
    out = {
        "root": np.asarray(net.seed),
        "netstim": _event_times(stim),
        "pairing": np.stack([np.asarray(rows.source_index), np.asarray(rows.target_index)]),
        "sample_seed": np.asarray(sites.seed),
        "sites": np.asarray(sites.evaluate(cell.morpho).points, dtype=float),
    }
    result = net.run(dt=DT, duration=6.0 * u.ms)
    out["v_post"] = np.asarray(result.samples["post"]["v"].values.to_decimal(u.mV))
    out["v_det"] = np.asarray(result.samples["det"]["v"].values.to_decimal(u.mV))
    return out


def _two_stim_network(order) -> dict:
    net = Network("ordered", seed=None)
    made = {name: net.add_population(name, braincell.NetStim(size=2, number=4, noise=1.0)) for name in order}
    return {name: _event_times(stim) for name, stim in made.items()}


def _point_cell(pop_size=4, name="post"):
    """A lumped soma (20 um x 10 um, leak at -65 mV) with one ExpSyn."""
    cell = braincell.Cell.point(pop_size, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, name=name)
    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
    cell.loc(at("soma", 0.5)).record("v", braincell.observe.state("V"))
    return cell


def _direct_rows(pairing) -> np.ndarray:
    """Rows of a direct ``braincell.connect`` from six periodic sources to four lumped members."""
    cell = _point_cell()
    rows = braincell.connect(
        "drive",
        source=braincell.NetStim(size=6, number=1),
        synapse=cell.synapses["syn"],
        pairing=pairing,
        weight=0.001 * u.uS,
    )
    return np.stack([np.asarray(rows.source_index), np.asarray(rows.target_index)])


def _default_roots() -> dict:
    """Everything CELL-11b gives a drawn root when no seed is written."""
    net = Network("demo")
    registered = net.add_population("stim", braincell.NetStim(size=2, number=3, noise=1.0))
    return {
        "Network() root": np.asarray(net.seed),
        "Network() NetStim": _event_times(registered),
        "standalone NetStim": _event_times(braincell.NetStim(size=2, number=3, noise=1.0)),
        "direct connect": _direct_rows(independent(5)),
    }


def _explicit_seed_draws() -> dict:
    """Explicit seeds and independent generators: local streams that no context controls."""
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    morpho = braincell.Morphology.from_root(soma, name="soma")
    network_stim = Network(seed=7).add_population("s", braincell.NetStim(size=3, number=4, noise=1.0))
    return {
        "NetStim(seed=9)": _event_times(braincell.NetStim(size=3, number=4, noise=1.0, seed=9)),
        "Network(seed=7)": _event_times(network_stim),
        "sample(seed=3)": np.asarray(
            braincell.filter.sample(AllRegion(), number=4, seed=3).evaluate(morpho).points, dtype=float
        ),
        "RandomSamples(seed=123)": np.asarray(RandomSamples(AllRegion(), 4, 123).evaluate(morpho).points, dtype=float),
        "independent(5, seed=8)": _direct_rows(independent(5, seed=8)),
        "RandomState(5)": np.asarray(brainstate.random.RandomState(5).normal(size=3)),
        "jax.random.key(3)": np.asarray(jax.random.normal(jax.random.key(3), (3,))),
        "default_rng(1)": np.random.default_rng(1).random(3),
    }


class NetworkContextSeedTest(unittest.TestCase):
    """CELL-11a: ``Network(seed=None)`` draws its root from the BrainState default stream."""

    def test_seed_none_draws_the_root_by_the_definition(self) -> None:
        # Acceptance 1. On main ``Network(seed=None)`` raised TypeError.
        for seed in (0, 42):
            with self.subTest(seed=seed):
                with brainstate.random.seed_context(seed):
                    roots = [Network(seed=None).seed, Network("second", seed=None).seed]
                self.assertEqual(roots, drawn_seeds(seed, 2))
                self.assertNotEqual(roots[0], roots[1])
                self.assertIs(type(roots[0]), int)
        roots = []
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision), brainstate.random.seed_context(42):
                roots.append(Network(seed=None).seed)
        self.assertEqual(roots[0], roots[1])

    def test_one_context_seed_controls_every_random_stream_of_an_experiment(self) -> None:
        # Acceptance 2: root, NetStim noise, pairing, sampled locations,
        # initial values and a callback; the Cell without randomness keeps its V.
        with brainstate.random.seed_context(42):
            first = _context_experiment()
        with brainstate.random.seed_context(42):
            again = _context_experiment()
        with brainstate.random.seed_context(43):
            other = _context_experiment()
        for key, value in first.items():
            with self.subTest(stream=key):
                np.testing.assert_array_equal(again[key], value)
                if key == "v_det":
                    np.testing.assert_allclose(other[key], value, rtol=1e-12, atol=0.0)
                else:
                    self.assertFalse(np.array_equal(other[key], value), f"{key} did not change with the seed")

    def test_the_recorded_root_reproduces_the_network_streams(self) -> None:
        # Acceptance 3: record ``net.seed``, pass it back as an explicit root.
        with brainstate.random.seed_context(42):
            drawn = Network(seed=None)
            stim = drawn.add_population("stim", braincell.NetStim(size=2, number=3, noise=1.0))
        replayed = Network(seed=drawn.seed).add_population("stim", braincell.NetStim(size=2, number=3, noise=1.0))
        np.testing.assert_array_equal(_event_times(replayed), _event_times(stim))

    def test_registration_order_does_not_change_the_streams_under_a_drawn_root(self) -> None:
        # Acceptance 4: a Network takes one draw, then derives by path.
        with brainstate.random.seed_context(5):
            forward = _two_stim_network(("a", "b"))
        with brainstate.random.seed_context(5):
            backward = _two_stim_network(("b", "a"))
        for name in ("a", "b"):
            np.testing.assert_array_equal(backward[name], forward[name])
        self.assertFalse(np.array_equal(forward["a"], forward["b"]))

    def test_reset_keeps_the_drawn_streams_and_outside_a_context_the_global_stream_is_used(self) -> None:
        # Acceptance 5. No persistent RNG: the schedule and the rows are concrete.
        with brainstate.random.seed_context(3):
            net = Network("reset", seed=None)
            stim = net.add_population("stim", braincell.NetStim(size=3, number=2, noise=1.0))
            soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            cell = braincell.Cell(braincell.Morphology.from_root(soma, name="soma"), pop_size=2, V_init=-65.0 * u.mV)
            cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV))
            post = net.add_population("c", cell)
            rows = net.connect("d", source=stim, synapse=post.synapses["ampa"], weight=WEIGHT, pairing=independent(5))
            events = _event_times(stim)
            pairs = (np.asarray(rows.source_index).copy(), np.asarray(rows.target_index).copy())
            net.run(dt=DT, duration=1.0 * u.ms)
            net.reset_state()
        fresh = net.connections["c", "d"]
        np.testing.assert_array_equal(_event_times(stim), events)
        np.testing.assert_array_equal(np.asarray(fresh.source_index), pairs[0])
        np.testing.assert_array_equal(np.asarray(fresh.target_index), pairs[1])

        with brainstate.random.seed_context(4):
            inside = Network(seed=None).seed
        brainstate.random.seed(4)
        self.assertEqual(Network(seed=None).seed, inside)

    def test_invalid_seeds_still_raise_at_construction(self) -> None:
        # Acceptance 6; main raised the same TypeError.
        for seed in (True, 1.5, "3"):
            with self.subTest(seed=seed), self.assertRaisesRegex(TypeError, "Network seed must be an integer"):
                Network(seed=seed)

    def test_seed_none_under_a_transformation_raises_and_leaves_the_default_stream(self) -> None:
        # Acceptance 6: inside a trace the default stream has no concrete key.
        def in_loop(fn):
            return lambda: brainstate.transform.for_loop(lambda i: fn(), np.arange(2))

        for name, transform in (("jit", brainstate.transform.jit), ("for_loop", in_loop)):
            with self.subTest(transform=name):
                brainstate.random.seed(1)
                expected = float(brainstate.random.uniform())
                brainstate.random.seed(1)
                before = default_key_data()
                with self.assertRaisesRegex(TypeError, "traced"):
                    transform(lambda: Network(seed=None).seed)()
                np.testing.assert_array_equal(default_key_data(), before)
                self.assertEqual(float(brainstate.random.uniform()), expected)
        # The suggested fix, an integer seed, builds inside jit.
        self.assertEqual(int(brainstate.transform.jit(lambda: Network(seed=3).seed)()), 3)

    def test_explicit_seeds_and_independent_generators_ignore_the_context(self) -> None:
        # Acceptance 7.
        with brainstate.random.seed_context(1):
            one = _explicit_seed_draws()
        with brainstate.random.seed_context(2):
            two = _explicit_seed_draws()
        for key, value in one.items():
            with self.subTest(stream=key):
                np.testing.assert_array_equal(two[key], value)


class NetworkDefaultSeedTest(unittest.TestCase):
    """CELL-11b: without a seed the roots come from the BrainState default stream."""

    def test_a_network_without_a_seed_draws_its_root(self) -> None:
        # Acceptance 6. Before CELL-11b ``Network()`` had the root 0.
        for seed in (0, 42):
            with self.subTest(seed=seed), brainstate.random.seed_context(seed):
                self.assertEqual(Network().seed, drawn_seeds(seed, 1)[0])

    def test_one_context_controls_every_default_root(self) -> None:
        # Before CELL-11b all four were the same under seed_context(1) and (2).
        with brainstate.random.seed_context(1):
            first = _default_roots()
        with brainstate.random.seed_context(1):
            again = _default_roots()
        with brainstate.random.seed_context(2):
            other = _default_roots()
        for name, value in first.items():
            with self.subTest(stream=name):
                np.testing.assert_array_equal(again[name], value)
                self.assertFalse(np.array_equal(other[name], value))

    def test_an_explicit_network_seed_derives_the_new_noise_stream(self) -> None:
        stim = Network(seed=7).add_population("stim", braincell.NetStim(size=2, number=3, noise=1.0))
        # Before CELL-11b: [6.56192421913147, 15.912121534347534, 47.53369450569153].
        np.testing.assert_allclose(
            _event_times(stim)[0], [8.318834060570685, 8.727936488554263, 12.538289521217408], rtol=1e-12
        )

    def test_only_a_network_without_a_seed_advances_the_default_stream(self) -> None:
        for name, build, advances in (
            ("Network()", Network, True),
            ("Network(seed=7)", lambda: Network(seed=7), False),
        ):
            with self.subTest(network=name), brainstate.random.seed_context(5):
                before = default_key_data()
                build()
                self.assertEqual(not np.array_equal(default_key_data(), before), advances)

    def test_outside_a_context_the_global_stream_gives_the_root(self) -> None:
        brainstate.random.seed(4)
        outside = (Network().seed, _event_times(braincell.NetStim(size=2, number=3, noise=1.0)))
        with brainstate.random.seed_context(4):
            inside = (Network().seed, _event_times(braincell.NetStim(size=2, number=3, noise=1.0)))
        self.assertEqual(outside[0], inside[0])
        np.testing.assert_allclose(outside[1], inside[1], rtol=1e-12)

    def test_a_network_without_a_seed_cannot_be_built_under_a_transformation(self) -> None:
        # Acceptance 7. Before CELL-11b it was built with the root 0.
        brainstate.random.seed(1)
        before = default_key_data()
        with self.assertRaisesRegex(TypeError, r"Network\(seed=None\).*traced"):
            brainstate.transform.jit(lambda: Network().seed)()
        np.testing.assert_array_equal(default_key_data(), before)
        self.assertEqual(int(brainstate.transform.jit(lambda: Network(seed=3).seed)()), 3)

    def test_split_runs_and_reset_replay_the_noise(self) -> None:
        # Acceptance 5: 40 events are generated in full, 10**9 on demand.
        def network(number):
            with brainstate.random.seed_context(3):
                net = Network("split")
                stim = net.add_population(
                    "stim", braincell.NetStim(size=2, number=number, interval=0.5 * u.ms, noise=1.0)
                )
                post = net.add_population("post", _point_cell(pop_size=2))
                net.connect("drive", source=stim, synapse=post.synapses["syn"], weight=0.01 * u.uS)
            return net

        def voltage(result):
            return np.asarray(result.samples["post"]["v"].values.to_decimal(u.mV))

        for number in (40, 10**9):
            with self.subTest(number=number):
                net = network(number)
                whole = net.run(dt=DT, duration=10.0 * u.ms)
                self.assertEqual(len(whole.events["stim"]["spike"].time), 46)
                split_net = network(number)
                parts = [split_net.run(dt=DT, duration=d * u.ms) for d in (2.5, 0.025, 4.475, 3.0)]
                np.testing.assert_allclose(
                    np.concatenate([voltage(part) for part in parts]), voltage(whole), rtol=1e-12
                )
                net.reset_state()
                again = net.run(dt=DT, duration=10.0 * u.ms)
                np.testing.assert_allclose(voltage(again), voltage(whole), rtol=1e-12)
                np.testing.assert_allclose(
                    again.events["stim"]["spike"].time.to_decimal(u.ms),
                    whole.events["stim"]["spike"].time.to_decimal(u.ms),
                    rtol=1e-12,
                )

    def test_batch_members_receive_the_same_events(self) -> None:
        cell = _point_cell(pop_size=2)
        stim = braincell.NetStim(size=2, number=20, interval=0.5 * u.ms, noise=1.0, seed=4)
        braincell.connect("drive", source=stim, synapse=cell.synapses["syn"], weight=0.01 * u.uS)
        cell.init_state(batch_size=3)
        v = np.asarray(cell.run(dt=DT, duration=5.0 * u.ms).samples["v"].values.to_decimal(u.mV))
        self.assertEqual(v.shape, (200, 3, 2))
        self.assertGreater(float(v.max()), -60.0)
        np.testing.assert_allclose(v[:, 1], v[:, 0], rtol=1e-12)
        np.testing.assert_allclose(v[:, 2], v[:, 0], rtol=1e-12)


class NetworkAxialOperatorTest(unittest.TestCase):
    """CELL-50: the network run loop builds each Cell's axial operator once, before its steps."""

    N = 40

    def test_run_matches_the_cell_update_loop(self):
        for solver in ("staggered", "rk4"):
            with self.subTest(solver=solver), _float64():
                looped = hh_y_cable(solver)
                looped.init_state()
                expected = np.asarray(update_loop_voltage(looped, self.N))
                network = Network("cable")
                network.add_population("cells", hh_y_cable(solver))
                result = network.run(dt=DT, duration=self.N * DT)
                actual = np.asarray(result.samples["cells"]["v"].values.to_decimal(u.mV))
                np.testing.assert_allclose(actual, expected.reshape(actual.shape), rtol=1e-12, atol=0.0)
                self.assertGreater(float(actual[-1].max()), 20.0)

    def test_the_operator_is_built_once_before_the_steps(self):
        for solver in ("staggered", "rk4"):
            with self.subTest(solver=solver), _float64():
                network = Network("cable")
                network.add_population("cells", hh_y_cable(solver))
                self.assertEqual(axial_builds(lambda: network.run(dt=DT, duration=4 * DT)), (0, 1))


class NetworkSecondRunCompileTest(unittest.TestCase):
    """CELL-54: a Network's second run reuses its loop."""

    @staticmethod
    def _network():
        # An event every 0.1 ms from 0.1 ms: each 1 ms window plans 9 or 10
        # arrivals, so no path's arrays change shape between the runs.
        cell, _ = make_event_cell()
        network = Network("again")
        source = network.add_population("stim", braincell.NetStim(start=0.1 * u.ms, interval=0.1 * u.ms, number=1000))
        target = network.add_population("post", cell)
        network.connect("drive", source=source, synapse=target.synapses["syn"], weight=WEIGHT)
        return network, cell

    def test_a_second_run_does_not_compile_the_loop_again(self):
        # On main 29d6828 V, spike and the synapse's g were weak-typed after
        # init_state() and reset_state() and strong after a run, which jit
        # keys on: on every path the loop compiled at runs 1 and 2,
        # [1, 1, 0, 0, 0].
        for delivery in (None, "direct", "padded"):
            with self.subTest(scheduled_delivery=delivery):
                network, cell = self._network()
                network.init_state()
                options = {} if delivery is None else {"scheduled_delivery": delivery}
                counts, conductances = [], []
                for index in range(5):
                    if index == 3:
                        network.reset_state()
                    with loop_compiles() as names:
                        network.run(dt=DT, duration=40 * DT, **options)
                    counts.append(names.count("jit(jaxpr_call)"))
                    conductances.append(_synapse_g(cell))
                self.assertGreaterEqual(counts[0], 1)
                self.assertEqual(counts[1:], [0, 0, 0, 0])
                self.assertTrue(all(np.isfinite(g) and g > 0.0 for g in conductances))

    def test_a_jitted_update_loop_after_prepare_run_compiles_once(self):
        # A loop of the user's own over update(), compiled with
        # brainstate.transform.jit: twice after init_state(), twice after
        # reset_state() (main 29d6828: [1, 1, 0, 0]).
        network, cell = self._network()
        network.init_state()
        network.prepare_run(dt=DT)

        def step(_):
            network.update()
            return cell.V.value

        loop = brainstate.transform.jit(lambda: brainstate.transform.for_loop(step, np.arange(40)))
        counts, conductances = [], []
        for index in range(4):
            if index == 2:
                network.reset_state()
            with loop_compiles() as names:
                jax.block_until_ready(loop())
            counts.append(names.count("jit(jaxpr_call)"))
            conductances.append(_synapse_g(cell))
        self.assertGreaterEqual(counts[0], 1)
        self.assertEqual(counts[1:], [0, 0, 0])
        self.assertTrue(all(np.isfinite(g) and g > 0.0 for g in conductances))

    def test_a_jitted_update_loop_after_prepare_run_with_a_duration_compiles_once(self):
        # CELL-66: the stop is a constant of the user's trace, as are the
        # events, not a State (L40). The duration covers the loop's two
        # calls before reset_state() and the two after.
        network, cell = self._network()
        network.init_state()
        network.prepare_run(dt=DT, duration=80 * DT)

        def step(_):
            network.update()
            return cell.V.value

        loop = brainstate.transform.jit(lambda: brainstate.transform.for_loop(step, np.arange(40)))
        counts, conductances = [], []
        for index in range(4):
            if index == 2:
                network.reset_state()
            with loop_compiles() as names:
                jax.block_until_ready(loop())
            counts.append(names.count("jit(jaxpr_call)"))
            conductances.append(_synapse_g(cell))
        self.assertGreaterEqual(counts[0], 1)
        self.assertEqual(counts[1:], [0, 0, 0])
        self.assertTrue(all(np.isfinite(g) and g > 0.0 for g in conductances))
        # A second prepare_run extends the duration; the loop compiled
        # before it keeps the old stop, so its steps past it are NaN.
        network.prepare_run(dt=DT, duration=80 * DT)
        stale = np.asarray(u.get_mantissa(loop()))
        self.assertTrue(np.all(np.isnan(stale)))
        self.assertTrue(np.isnan(_synapse_g(cell)))
        # A new trace runs the new duration, here 120 steps from t = 0
        # (the stop is now step 160), past the old stop.
        network.reset_state()
        traced = brainstate.transform.jit(lambda: brainstate.transform.for_loop(step, np.arange(120)))
        fresh, fresh_cell = self._network()
        fresh.init_state()
        fresh.prepare_run(dt=DT, duration=120 * DT)

        def fresh_step(_):
            fresh.update()
            return fresh_cell.V.value

        expected = np.asarray(u.get_mantissa(brainstate.transform.for_loop(fresh_step, np.arange(120))))
        np.testing.assert_array_equal(np.asarray(u.get_mantissa(traced())), expected)
        self.assertTrue(np.all(np.isfinite(expected)))


class NetworkScheduleGrowthCompileTest(unittest.TestCase):
    """CELL-74: the default path compiles its loop once per doubling of a NetStim schedule."""

    @staticmethod
    def _network():
        # 2**21 events: too many to generate at construction, so the
        # schedule grows with the runs.
        cell, _ = make_event_cell()
        network = Network("grow")
        stim = braincell.NetStim(start=0.1 * u.ms, interval=0.1 * u.ms, number=2**21)
        source = network.add_population("stim", stim)
        target = network.add_population("post", cell)
        network.connect("drive", source=source, synapse=target.synapses["syn"], weight=WEIGHT)
        return network, stim

    def test_runs_of_one_length_compile_once_per_doubling(self):
        # The loop holds the generated events as constants, and the
        # schedule's width keys it. Run k (1 ms) generates up to
        # until = k ms + dt: width max(floor((until - 0.1) / 0.1) + 2,
        # 2 * width) whenever the last event is not past until. This pins
        # the behaviour of main 36dd9cd.
        network, stim = self._network()
        self.assertFalse(stim._schedule_complete)
        counts, widths, parts = [], [], []
        for index in range(32):
            if index == 16:
                network.reset_state()
            with loop_compiles() as names:
                result = network.run(dt=DT, duration=40 * DT)
            g = np.asarray(u.get_mantissa(result.samples["post"]["g"].values))
            self.assertEqual(g.shape[0], 40)
            parts.append(g)
            counts.append(names.count("jit(jaxpr_call)"))
            widths.append(stim._event_times_ms.shape[1])
        self.assertEqual(widths[:16], [11, 22, 44, 44] + [88] * 4 + [176] * 8)
        self.assertEqual(widths[16:], [176] * 16)
        # Each width change at least doubles the schedule and compiles the
        # loop: 4 = log2(16) times after the first run; none after reset_state.
        self.assertGreaterEqual(counts[0], 1)
        self.assertEqual(counts[1:16], [1, 1, 0, 1, 0, 0, 0, 1] + [0] * 7)
        self.assertEqual(counts[16:], [0] * 16)

        split = np.concatenate(parts[:16])
        self.assertTrue(np.all(np.isfinite(split)))
        self.assertGreater(np.count_nonzero(np.diff(split, axis=0)), 0)
        whole, _ = self._network()
        once = np.asarray(u.get_mantissa(whole.run(dt=DT, duration=640 * DT).samples["post"]["g"].values))
        np.testing.assert_array_equal(split, once)
        np.testing.assert_array_equal(np.concatenate(parts[16:]), once)


class NetworkLongRunSampleTest(unittest.TestCase):
    """CELL-59a: a Network run samples in whole steps, however long the Network has run."""

    def test_a_long_run_takes_each_sample_once(self):
        # On main 73136f6 the float time mask took extra samples from about
        # 1e6 steps on: 500002 and 250002 in 1e6 steps in float64 (510938 and
        # 255471 in float32), and 21 and 16 in the 21 steps after them, for
        # 500000, 250000, 11 and 6.
        for precision in (32, 64):
            with self.subTest(precision=precision), brainstate.environ.context(precision=precision):
                cell = long_run_cell()
                network = Network("long")
                network.add_population("post", cell)
                long = network.run(dt=DT, duration=1_000_000 * DT)
                self.assertEqual(cell._clock_host()[1], 1_000_000)
                self.assertEqual([long.samples["post"][n].values.shape[0] for n in ("p2", "p4")], [500_000, 250_000])
                short = network.run(dt=DT, duration=21 * DT)
                self.assertEqual(cell._clock_host()[1], 1_000_021)
                self.assertEqual([short.samples["post"][n].values.shape[0] for n in ("p2", "p4")], [11, 6])


if __name__ == "__main__":
    unittest.main()
