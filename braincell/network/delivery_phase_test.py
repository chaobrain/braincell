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

"""The phase of live and zero-delay deliveries on every driver (CELL-81).

The model is a NEURON-matched HH soma (staggered, V_th 0 mV, a 0.1 nA clamp
from 0.25 ms; dt 0.025 ms, float64) that first crosses threshold in step
``m = 98``, with an ExpSyn (tau 2 ms, e 0 mV) at soma(0.5) and a
self-connection of 0.002 uS and ``k`` steps of delay. As in NEURON, the
event acts at the start of step ``b = m + 1 + k``: the sample of ``g`` at
``t_b`` (the state before step ``b``) is 0, the sample at ``t_{b+1}`` is
``w exp(-dt / tau)``. Sample ``j`` of an ``update()`` loop is ``g`` read
after update ``j - 1``.
"""

import gc
import unittest

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell import mech
from braincell._multi_compartment.lifecycle_test import loop_compiles
from braincell.filter import AllRegion, at

DT_MS, N, M, W, TAU = 0.025, 200, 98, 0.002, 2.0
DT = DT_MS * u.ms
#: NEURON 9.0.2, g of the ExpSyn one step after the delivery boundary: w exp(-dt / tau).
G_AFTER = 0.0019751556009877627
#: The same with the weight set to 0.004 uS between two runs split at the boundary.
G_AFTER_SET = 0.0039503112019755255
DRIVERS = ("Cell.run", "update", "Network.run", "Network.update")
DELAYS = (0, 1, 3)


def _float64(test):
    def wrapped(self, *args, **kwargs):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            return test(self, *args, **kwargs)

    wrapped.__name__ = test.__name__
    wrapped.__doc__ = test.__doc__
    return wrapped


def _cell(k, *, plasticity=None, scheduled_ms=None):
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"),
        cv_policy=braincell.CVPerBranch(),
        solver="staggered",
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
    )
    cell.paint(
        AllRegion(),
        mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm**2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        ),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", g_max=120.0 * (u.mS / u.cm**2)),
        mech.Channel("K_HH1952", g_max=36.0 * (u.mS / u.cm**2)),
        mech.Channel("IL", g_max=0.3 * (u.mS / u.cm**2), E=-54.387 * u.mV),
    )
    cell.place(at("soma", 0.5), mech.CurrentClamp(delay=0.25 * u.ms, durations=1e3 * u.ms, amplitudes=0.1 * u.nA))
    cell.place(at("soma", 0.5), mech.Synapse("ExpSyn", name="syn", tau=TAU * u.ms, e=0.0 * u.mV))
    cell.record("v", braincell.observe.state("V"))
    cell.record("g", braincell.observe.synapse(name="syn").state("g"))
    if scheduled_ms is not None:
        braincell.connect(
            "scheduled",
            source=braincell.EventSequence.from_times([np.asarray([scheduled_ms]) * u.ms]),
            synapse=cell.synapses["syn"],
            weight=0.001 * u.uS,
        )
    if plasticity is None:
        braincell.connect(
            "self", source=cell.event_outputs["spike"], synapse=cell.synapses["syn"], weight=W * u.uS, delay=k * DT
        )
    return cell


def _rule():
    # No pairing changes the weight, so the row delivers what set() wrote.
    return mech.ConnectionRule(
        "PairSTDP", A_plus=0.0 * u.uS, A_minus=0.0 * u.uS, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms
    )


def _g_now(cell):
    return float(np.asarray(cell.synapses["syn"].g.to_decimal(u.uS), dtype=np.float64).reshape(-1)[0])


def _samples(block):
    v = np.asarray(block["v"].values.to_decimal(u.mV), dtype=np.float64).reshape(-1)
    g = np.asarray(block["g"].values.to_decimal(u.uS), dtype=np.float64).reshape(-1)
    return v, g


class _Driver:
    """Steps one model with one driver; ``go(n)`` returns the V and g samples of the next ``n`` steps."""

    def __init__(self, driver, k, *, plasticity=None, scheduled_ms=None):
        self.driver, self.step = driver, 0
        self.cell = _cell(k, plasticity=plasticity, scheduled_ms=scheduled_ms)
        if driver.startswith("Network"):
            self.net = braincell.Network("phase")
            pop = self.net.add_population("c", self.cell)
            if plasticity is not None:
                self.net.connect(
                    "self",
                    source=pop.event_outputs["spike"],
                    synapse=self.cell.synapses["syn"],
                    weight=W * u.uS,
                    delay=k * DT,
                    plasticity=plasticity,
                )
            self.net.init_state()
            if driver == "Network.update":
                self.net.prepare_run(dt=DT)
        else:
            self.cell.init_state()
            if driver == "update":
                self.cell.connections.prepare_runtime(DT)

    def go(self, n):
        first, self.step = self.step, self.step + n
        if self.driver == "Cell.run":
            return _samples(self.cell.run(dt=DT, duration=n * DT).samples)
        if self.driver == "Network.run":
            return _samples(self.net.run(dt=DT, duration=n * DT).samples["c"])
        cell = self.cell

        def read():
            return cell.V.value.to_decimal(u.mV).reshape(-1)[0], cell.synapses["syn"].g.to_decimal(u.uS).reshape(-1)[0]

        def body(step):
            if self.driver == "Network.update":
                self.net.update()
            else:
                with brainstate.environ.context(t=step * DT):
                    cell.update()
            return read()

        before = [np.asarray(x, dtype=np.float64) for x in read()]
        with brainstate.environ.context(dt=DT):
            v, g = brainstate.transform.for_loop(body, np.arange(first, first + n))
        # Sample j is the state before step j: the read before the loop, then each read but the last.
        v = np.concatenate([[before[0]], np.asarray(v, dtype=np.float64)[:-1]])
        g = np.concatenate([[before[1]], np.asarray(g, dtype=np.float64)[:-1]])
        return v, g


def _whole(driver, k, n=N, **kwargs):
    run = _Driver(driver, k, **kwargs)
    v, g = run.go(n)
    return v, g, run


class PhaseTest(unittest.TestCase):
    """Acceptance 1: the event acts at the start of step ``m + 1 + k`` on every driver."""

    @_float64
    def test_every_driver_delivers_at_the_boundary_after_the_delay(self):
        for driver in DRIVERS:
            for k in DELAYS:
                with self.subTest(driver=driver, k=k):
                    v, g, _ = _whole(driver, k)
                    self.assertEqual(g.shape, (N,))
                    first = int(np.flatnonzero(v >= 0.0)[0])
                    self.assertEqual(first - 1, M)  # V of the step m end is the sample m + 1
                    b = M + 1 + k
                    self.assertEqual(g[b], 0.0)  # main: 0.002 on Cell.run, update and Network k = 0
                    np.testing.assert_allclose(g[b + 1], G_AFTER, rtol=1e-12, atol=0.0)
                    self.assertTrue(np.all(g[:b] == 0.0))

    @_float64
    def test_the_drivers_agree_sample_by_sample(self):
        for k in DELAYS:
            with self.subTest(k=k):
                v0, g0, _ = _whole("Cell.run", k)
                for driver in DRIVERS[1:]:
                    v, g, _ = _whole(driver, k)
                    np.testing.assert_allclose(v, v0, rtol=1e-12, atol=1e-12, err_msg=driver)
                    np.testing.assert_allclose(g, g0, rtol=1e-12, atol=1e-18, err_msg=driver)


class WeightAcrossRunsTest(unittest.TestCase):
    """Acceptance 2: an event delivered on the first boundary of a run takes the weight set before it."""

    def _check(self, driver, k, plasticity=None):
        b = M + 1 + k
        run = _Driver(driver, k, plasticity=plasticity)
        _, g1 = run.go(b)
        run.cell.connections.set(weight=2 * W * u.uS)
        _, g2 = run.go(N - b)
        g = np.concatenate([g1, g2])
        self.assertEqual(g[b], 0.0)
        # main: G_AFTER, the old weight, on Cell.run and on the Network for k = 0.
        np.testing.assert_allclose(g[b + 1], G_AFTER_SET, rtol=1e-12, atol=0.0)

    @_float64
    def test_every_driver_delivers_the_new_weight(self):
        for driver in DRIVERS:
            for k in DELAYS:
                with self.subTest(driver=driver, k=k):
                    self._check(driver, k)

    @_float64
    def test_a_zero_delay_rule_row_delivers_the_new_weight(self):
        for driver in ("Network.run", "Network.update"):
            with self.subTest(driver=driver):
                self._check(driver, 0, plasticity=_rule())


class HandoverTest(unittest.TestCase):
    """Acceptance 4: a zero-delay event detected in the last step of a segment arrives once, in the next."""

    SPLIT = M + 1

    def _reference(self, **kwargs):
        v, g, _ = _whole("Cell.run", 0, n=2 * self.SPLIT, **kwargs)
        return v, g

    def _assert_once(self, v, g, reference, scheduled=0.0):
        v0, g0 = reference
        self.assertEqual(g[self.SPLIT], 0.0)
        np.testing.assert_allclose(g[self.SPLIT + 1], G_AFTER * (1 + scheduled / W), rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(v, v0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g, g0, rtol=1e-12, atol=1e-18)

    def _cell_run(self, cell):
        return _samples(cell.run(dt=DT, duration=self.SPLIT * DT).samples)

    def _network_run(self, cell):
        net = braincell.Network("next")
        net.add_population("c", cell)
        samples = _samples(net.run(dt=DT, duration=self.SPLIT * DT).samples["c"])
        del net
        gc.collect()
        return samples

    @_float64
    def test_each_transition_delivers_the_event_once(self):
        reference = self._reference()
        for first, second in (
            ("Cell.run", "Cell.run"),
            ("Cell.run", "Network"),
            ("Network", "Cell.run"),
            ("update", "Cell.run"),
        ):
            with self.subTest(first=first, second=second):
                if first == "Network":
                    run = _Driver("Network.run", 0)
                    v1, g1 = run.go(self.SPLIT)
                    cell = run.cell
                    del run.net, run
                    gc.collect()
                    self.assertIsNone(cell.network_owner)
                else:
                    run = _Driver(first, 0)
                    v1, g1 = run.go(self.SPLIT)
                    cell = run.cell
                    if first == "update":
                        cell._set_current_time(self.SPLIT * DT)
                v2, g2 = (self._network_run if second == "Network" else self._cell_run)(cell)
                self._assert_once(np.concatenate([v1, v2]), np.concatenate([g1, g2]), reference)

    @_float64
    def test_a_scheduled_event_on_the_same_boundary_arrives_once_too(self):
        at_split = self.SPLIT * DT_MS
        reference = self._reference(scheduled_ms=at_split)
        cell = _cell(0, scheduled_ms=at_split)
        cell.init_state()
        v1, g1 = self._cell_run(cell)
        v2, g2 = self._cell_run(cell)
        self._assert_once(np.concatenate([v1, v2]), np.concatenate([g1, g2]), reference, scheduled=0.001)

    @_float64
    def test_a_new_dt_takes_the_pending_zero_delay_event_once(self):
        # Acceptance 5 too: zero-delay rows only, so the new dt is accepted.
        cell = _cell(0)
        cell.init_state()
        self._cell_run(cell)
        half = DT_MS / 2
        _, g = _samples(cell.run(dt=half * u.ms, duration=4 * half * u.ms).samples)
        np.testing.assert_allclose(g, [0.0, *(W * np.exp(-np.arange(1, 4) * half / TAU))], rtol=1e-12, atol=0.0)


def _fingerprint(cell):
    states = {
        str(path): [np.asarray(leaf).copy() for leaf in jax.tree.leaves(state.value)]
        for path, state in brainstate.graph.states(cell).items()
    }
    buffers = {
        key: np.asarray(u.get_mantissa(cell._runtime.get_event_buffer(key))).copy()
        for key in cell._runtime.event_buffers
    }
    calls = [
        {
            name: (np.asarray(value).copy() if isinstance(value, np.ndarray) else repr(value))
            for name, value in vars(call).items()
            if name != "live_history"
        }
        for call in cell._get_connection_store()._calls
    ]
    histories = [
        None if call.live_history is None else np.asarray(call.live_history.value).copy()
        for call in cell._get_connection_store()._calls
    ]
    return states, buffers, calls, histories, _g_now(cell), float(cell.current_time.to_decimal(u.ms))


class InFlightTest(unittest.TestCase):
    """Acceptance 5: a delayed event detected in the last step is in flight across a dt change."""

    def _assert_same(self, before, after):
        self.assertEqual(before[0].keys(), after[0].keys())
        for key, leaves in before[0].items():
            for leaf, other in zip(leaves, after[0][key]):
                np.testing.assert_array_equal(leaf, other, err_msg=key)
        self.assertEqual(before[1].keys(), after[1].keys())
        for key, buffer in before[1].items():
            np.testing.assert_array_equal(buffer, after[1][key], err_msg=key)
        for call, other in zip(before[2], after[2]):
            self.assertEqual(call.keys(), other.keys())
            for name, value in call.items():
                np.testing.assert_array_equal(value, other[name], err_msg=name)
        for history, other in zip(before[3], after[3]):
            np.testing.assert_array_equal(history, other)
        self.assertEqual(before[4:], after[4:])

    @_float64
    def test_changing_dt_with_an_event_from_the_last_step_raises(self):
        for k in (1, 3):
            with self.subTest(k=k):
                cell = _cell(k)
                cell.init_state()
                cell.run(dt=DT, duration=(M + 1) * DT)
                before = _fingerprint(cell)
                with self.assertRaisesRegex(ValueError, "in flight"):
                    cell.run(dt=DT / 2, duration=DT)
                self._assert_same(before, _fingerprint(cell))
                # The same dt goes on and delivers the event k steps later.
                _, g = _samples(cell.run(dt=DT, duration=(k + 2) * DT).samples)
                self.assertEqual(g[k], 0.0)
                np.testing.assert_allclose(g[k + 1], G_AFTER, rtol=1e-12, atol=0.0)


class CompileTest(unittest.TestCase):
    """The step-start delivery adds no compile: runs 1 and 2, and a run after ``reset_state()``."""

    @_float64
    def test_runs_reuse_their_loop(self):
        for driver in ("Cell.run", "Network.run"):
            for k in (0, 1):
                with self.subTest(driver=driver, k=k):
                    run = _Driver(driver, k)
                    owner = run.net if driver == "Network.run" else run.cell
                    counts = []
                    for index in range(3):
                        if index == 2:
                            owner.reset_state()
                        with loop_compiles() as names:
                            run.go(20)
                        counts.append(len(names))
                    self.assertGreaterEqual(counts[0], 1)
                    self.assertEqual(counts[1:], [0, 0])


if __name__ == "__main__":
    unittest.main()
