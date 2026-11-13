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

"""The time of a Cell's spike: the end of the step that crosses the threshold (CELL-80).

A crossing in the step from ``t_m`` to ``t_{m+1}`` is reported at ``t_{m+1}``
by ``RunResult.spikes`` and ``NetworkResult.events``, as NEURON's
``NetCon.record`` in fixed step, as a plasticity rule pairs it, and as a
delayed connection delivers it (at ``t_{m+1+k}``). The soma is the one
``validation/neuron/synapse/spike_times.py`` compares with NEURON.
"""

import contextlib
import unittest

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell import mech
from braincell._multi_compartment.lifecycle_test import loop_compiles
from braincell.filter import AllRegion, at

DT_MS = 0.025
DT = DT_MS * u.ms

# NEURON 9.0.2's NetCon.record times of the soma below (0.1 nA, 50 ms), from
# validation/neuron/synapse/spike_times.py: 2.4750000001, 18.8000000001 and
# 34.9250000001 ms. NEURON stamps a crossing with the end of its step and
# accumulates t, about 1e-10 ms off the grid.
NEURON_TIMES_MS = (2.475, 18.8, 34.925)
# The first crossing is in step 98: V(t_98) < 0 mV <= V(t_99).
FIRST_CROSSING_STEP = 98


@contextlib.contextmanager
def _float64():
    with jax.enable_x64(True), brainstate.environ.context(precision=64):
        yield


def _soma(amplitude=0.1 * u.nA, name=None, *, synapse=False):
    """One 20 um x 20 um HH soma, staggered, V_th 0 mV, a clamp from 0.25 ms; ExpSyn "s" (tau 2 ms) if asked."""
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch, name="soma"),
        cv_policy=braincell.CVPerBranch(),
        solver="staggered",
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
        name=name,
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
    cell.place(at("soma", 0.5), mech.CurrentClamp(delay=0.25 * u.ms, durations=1e3 * u.ms, amplitudes=amplitude))
    if synapse:
        cell.place(at("soma", 0.5), mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV))
    return cell


def _target(name=None):
    """A lumped HH target (rk4, default channel parameters) with an ExpSyn "syn" (tau 2 ms, e 0 mV), recording V."""
    cell = braincell.Cell.point(1, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4", name=name)
    cell.paint(
        AllRegion(),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("IL"),
        mech.Channel("Na_HH1952"),
        mech.Channel("K_HH1952"),
    )
    cell.place(mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
    cell.record("v", braincell.observe.state("V"))
    return cell


def _ms(value):
    return np.asarray(u.Quantity(value).to_decimal(u.ms), dtype=np.float64).reshape(-1)


def _alone(cell, *, n_steps):
    cell.init_state()
    return cell.run(dt=DT, duration=n_steps * DT)


def _in_network(cell, *, n_steps):
    network = braincell.Network("spike_time")
    network.add_population("c", cell)
    network.init_state()
    return network, network.run(dt=DT, duration=n_steps * DT)


def _f32(value):
    """Mantissas in ms, as stored (float32 in a float32 run)."""
    return np.asarray(u.get_mantissa(u.Quantity(value).in_unit(u.ms))).reshape(-1)


def _v_after_steps(cell, block, n_steps):
    """V after each step: samples 1 .. n - 1, then the final state."""
    v = np.asarray(block.values.to_decimal(u.mV), dtype=np.float64).reshape(n_steps)
    last = float(np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1)[0])
    return np.concatenate([v[1:], [last]])


class NeuronSpikeTimeTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(_float64())

    def test_spikes_are_at_neurons_times(self):
        # main 999a21e reported 2.45, 18.775 and 34.9 ms: each 0.025 ms early.
        n_steps = 2000
        alone = _alone(_soma(), n_steps=n_steps).spikes
        _, result = _in_network(_soma(), n_steps=n_steps)
        for series in (alone, result.events["c"]["spike"]):
            np.testing.assert_allclose(_ms(series.time), NEURON_TIMES_MS, rtol=0.0, atol=1e-9)
            np.testing.assert_array_equal(series.source_id, [0, 0, 0])

    def test_a_spike_is_at_the_time_axis_entry_after_its_step(self):
        n_steps = 200
        cell = _soma()
        cell.record("v", braincell.observe.state("V"))
        result = _alone(cell, n_steps=n_steps)
        v = _v_after_steps(cell, result.samples["v"], n_steps)
        before = np.concatenate([[-65.0], v[:-1]])
        (m,) = np.flatnonzero((before < 0.0) & (v >= 0.0))
        self.assertEqual(m, FIRST_CROSSING_STEP)
        # Bit for bit the entry of result.time: no recomputed t + dt.
        np.testing.assert_array_equal(_ms(result.spikes.time), _ms(result.time)[[m + 1]])


class PlasticityPostTimeTest(unittest.TestCase):
    """PairSTDP's post time is the reported spike time."""

    A_PLUS, TAU, T_PRE, W0 = 0.01, 20.0, 1.0, 1e-9

    def setUp(self):
        self.enterContext(_float64())

    def _rule(self):
        return mech.ConnectionRule(
            "PairSTDP",
            A_plus=self.A_PLUS * u.uS,
            A_minus=0.0105 * u.uS,
            tau_plus=self.TAU * u.ms,
            tau_minus=self.TAU * u.ms,
        )

    def _pre(self):
        return braincell.EventSequence.from_times([np.asarray([self.T_PRE]) * u.ms])

    def _dw(self, cell):
        return float(np.asarray(cell.connections["c"].rule_state("w").to_decimal(u.uS)).reshape(-1)[0]) - self.W0

    def test_the_weight_change_pairs_the_reported_spike(self):
        n_steps = 400
        cell = _soma(synapse=True)
        braincell.connect(
            "c", source=self._pre(), synapse=cell.synapses["s"], weight=self.W0 * u.uS, plasticity=self._rule()
        )
        runs = {"Cell.run": (cell, _alone(cell, n_steps=n_steps).spikes)}
        cell = _soma(synapse=True)
        network = braincell.Network("stdp")
        post = network.add_population("post", cell)
        pre = network.add_population("pre", self._pre())
        network.connect("c", source=pre, synapse=post.synapses["s"], weight=self.W0 * u.uS, plasticity=self._rule())
        network.init_state()
        runs["Network"] = (cell, network.run(dt=DT, duration=n_steps * DT).events["post"]["spike"])
        for driver, (cell, spikes) in runs.items():
            with self.subTest(driver=driver):
                (t_post,) = _ms(spikes.time)
                expected = self.A_PLUS * np.exp(-(t_post - self.T_PRE) / self.TAU)
                np.testing.assert_allclose(self._dw(cell), expected, rtol=1e-12)
                # 0.01 exp(-1.475 / 20) uS; main 999a21e reported 2.45 ms, 1.25e-3 off in relative terms.
                np.testing.assert_allclose(expected, 0.009289038907881565, rtol=1e-12)


class ReplayTest(unittest.TestCase):
    """Spikes replayed as an EventSequence act in the step the live connection does."""

    N_STEPS = 200

    def setUp(self):
        self.enterContext(_float64())

    def test_replay_and_live_connection_act_in_the_same_step(self):
        unconnected = _target()
        unconnected.init_state()
        reference = _v_after_steps(
            unconnected, unconnected.run(dt=DT, duration=self.N_STEPS * DT).samples["v"], self.N_STEPS
        )
        spikes = _alone(_soma(), n_steps=self.N_STEPS).spikes
        np.testing.assert_allclose(_ms(spikes.time), [2.475], rtol=1e-12)
        # main 999a21e: the replay acted in steps 98, 99 and 101, one step early.
        for k, first in ((0, 99), (1, 100), (3, 102)):
            with self.subTest(k=k):
                network = braincell.Network("live")
                source = network.add_population("src", _soma(name="src"))
                target = network.add_population("t", _target(name="t"))
                network.connect("l", source=source, synapse=target.synapses["syn"], weight=0.002 * u.uS, delay=k * DT)
                network.init_state()
                live = network.run(dt=DT, duration=self.N_STEPS * DT).samples["t"]["v"]
                live_first = np.flatnonzero(_v_after_steps(target.cell, live, self.N_STEPS) != reference)[0]

                replayed = _target()
                braincell.connect(
                    "r",
                    source=braincell.EventSequence.from_times([spikes.time]),
                    synapse=replayed.synapses["syn"],
                    weight=0.002 * u.uS,
                    delay=k * DT,
                )
                replayed.init_state()
                samples = replayed.run(dt=DT, duration=self.N_STEPS * DT).samples["v"]
                replay_first = np.flatnonzero(_v_after_steps(replayed, samples, self.N_STEPS) != reference)[0]
                self.assertEqual((live_first, replay_first), (first, first))


class WindowTest(unittest.TestCase):
    """A run reports the spikes of its own steps, at times in (start_time, stop_time]."""

    def setUp(self):
        self.enterContext(_float64())

    def test_a_crossing_in_the_last_step_belongs_to_that_run(self):
        first = FIRST_CROSSING_STEP + 1  # the first run ends with the crossing step
        whole = _alone(_soma(), n_steps=800)
        cell = _soma()
        cell.init_state()
        parts = (cell.run(dt=DT, duration=first * DT), cell.run(dt=DT, duration=(800 - first) * DT))
        np.testing.assert_array_equal(_ms(parts[0].spikes.time), _ms(parts[0].stop_time))
        np.testing.assert_allclose(_ms(parts[0].stop_time), [2.475], rtol=1e-12)
        self.assertTrue(np.all(_ms(parts[1].spikes.time) > _ms(parts[1].start_time)))
        joined = braincell.RunResult.concat(parts).spikes
        np.testing.assert_array_equal(_ms(joined.time), _ms(whole.spikes.time))
        np.testing.assert_array_equal(joined.source_id, whole.spikes.source_id)
        cell.reset_state()
        again = cell.run(dt=DT, duration=800 * DT).spikes
        np.testing.assert_array_equal(_ms(again.time), _ms(whole.spikes.time))

        network, whole_net = _in_network(_soma(), n_steps=800)
        network, _ = _in_network(_soma(), n_steps=first)
        network.reset_state()
        net_parts = (network.run(dt=DT, duration=first * DT), network.run(dt=DT, duration=(800 - first) * DT))
        np.testing.assert_array_equal(_ms(net_parts[0].events["c"]["spike"].time), _ms(net_parts[0].stop_time))
        joined = braincell.NetworkResult.concat(net_parts).events["c"]["spike"]
        np.testing.assert_array_equal(_ms(joined.time), _ms(whole_net.events["c"]["spike"].time))
        np.testing.assert_array_equal(_ms(joined.time), _ms(whole.spikes.time))



class Float32WindowTest(unittest.TestCase):
    def test_float32_segments_label_as_one_run(self):
        # 30000 steps (750 ms) in float32, where t = n dt has a few ulp of
        # rounding: each label is the next entry of its segment's time axis,
        # or its stop time, and three segments give the one run's events (L77).
        def build():
            cell = _soma()
            cell.init_state()
            return cell

        with brainstate.environ.context(precision=32):
            whole = build().run(dt=DT, duration=30000 * DT)
            steps = np.rint(_ms(whole.spikes.time) / DT_MS).astype(int) - 1
            self.assertGreater(len(steps), 40)
            # The first segment ends with a crossing step past step 10000.
            first = int(steps[steps > 10000][0]) + 1
            cell = build()
            parts = [cell.run(dt=DT, duration=n * DT) for n in (first, 7001, 30000 - first - 7001)]
        for part in parts:
            ends = np.concatenate([_f32(part.time), _f32(part.stop_time)])
            self.assertEqual(ends.dtype, np.float32)
            self.assertEqual(_f32(part.spikes.time).dtype, np.float32)
            times = _f32(part.spikes.time)
            self.assertGreater(len(times), 0)
            self.assertTrue(np.all(np.isin(times, ends[1:])))
        self.assertEqual(_f32(parts[0].spikes.time)[-1], _f32(parts[0].stop_time)[0])
        joined = braincell.RunResult.concat(parts).spikes
        np.testing.assert_array_equal(_f32(joined.time), _f32(whole.spikes.time))
        np.testing.assert_array_equal(joined.source_id, whole.spikes.source_id)


class CompileCountTest(unittest.TestCase):
    """The labels are assembled on the host: no compile after the first run."""

    def test_runs_after_the_first_compile_nothing(self):
        with _float64():
            cell = _soma()
            cell.init_state()
            network, _ = _in_network(_soma(), n_steps=40)
            for driver, run, reset in (
                ("Cell.run", lambda: cell.run(dt=DT, duration=40 * DT).spikes, cell.reset_state),
                ("Network", lambda: network.run(dt=DT, duration=40 * DT).events["c"]["spike"], network.reset_state),
            ):
                with self.subTest(driver=driver):
                    counts = []
                    for index in range(3):
                        if index == 2:
                            reset()
                        with loop_compiles() as names:
                            run()
                        counts.append(len(names))
                    self.assertEqual(counts[1:], [0, 0])


if __name__ == "__main__":
    unittest.main()
