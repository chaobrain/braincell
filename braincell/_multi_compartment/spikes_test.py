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

"""Tests for :mod:`braincell._multi_compartment.spikes`: the spikes of ``Cell.run`` and their output paths."""

import contextlib
import unittest
from unittest import mock

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell import mech
from braincell._multi_compartment import run_test
from braincell._multi_compartment import spikes as spikes_module
from braincell._multi_compartment.run import RunResult
from braincell._multi_compartment.run_test import (
    DT,
    _detector_cell,
    _float64,
    _hh_cell,
    _ms,
    _netstim,
    _paint_hh,
    _reduced_cell,
    assert_same_events,
)
from braincell.filter import AllRegion, RootLocation


def _ms_scalar(time):
    return float(np.asarray(u.Quantity(time).to_decimal(u.ms)))


class RunSpikesTest(unittest.TestCase):
    """CELL-26: ``RunResult.spikes`` carries the Cell's spikes as ``NetworkResult.events`` would."""

    DURATION = 20.0 * u.ms

    def setUp(self):
        self.enterContext(_float64())

    def _alone_and_in_network(self, build, *, dt=DT, duration=DURATION, stim=None):
        cell = build()
        if stim is not None:
            braincell.connect("drive", source=stim(), synapse=cell.synapses["input"], weight=0.2 * u.uS)
        alone = cell.run(dt=dt, duration=duration).spikes
        network = braincell.Network()
        post = network.add_population("post", build())
        if stim is not None:
            source = network.add_population("stim", stim())
            network.connect("drive", source=source, synapse=post.synapses["input"], weight=0.2 * u.uS)
        events = network.run(dt=dt, duration=duration).events["post"]["spike"]
        return alone, events

    def test_spikes_match_network_events(self):
        cases = {
            "lumped": (lambda: _hh_cell("lumped"), {}, ([3.2, 4.55, 19.45], [2, 1, 2])),
            "cable": (lambda: _hh_cell("cable"), {}, ([3.9, 6.35], [2, 1])),
            # The NetStim's events at 0.3, 1.3 and 2.3 ms act in the steps
            # from there; the model's events of those steps are at their ends.
            "reduction": (
                _reduced_cell,
                dict(dt=0.1 * u.ms, duration=4.0 * u.ms, stim=_netstim),
                ([0.4, 1.4, 2.4], [0, 0, 0]),
            ),
        }
        for name, (build, options, (times, ids)) in cases.items():
            with self.subTest(name):
                alone, events = self._alone_and_in_network(build, **options)
                assert_same_events(alone, events)
                self.assertEqual(dict(alone.metadata).keys(), {"port", "cv_id"})
                self.assertEqual(alone.metadata["port"], events.metadata["port"])
                np.testing.assert_array_equal(alone.metadata["cv_id"], events.metadata["cv_id"])
                # This machine's values (2b02a98, one step later from CELL-80); the comparison above is the contract.
                np.testing.assert_allclose(_ms(alone), times, rtol=1e-9)
                np.testing.assert_array_equal(alone.source_id, ids)
                np.testing.assert_array_equal(alone.count, np.ones(len(ids)))

    def test_events_sit_at_the_step_whose_end_crosses_the_threshold(self):
        for kind, V_th in (("lumped", 0.0), ("cable", 0.0), ("lumped", -20.0)):
            with self.subTest(kind=kind, V_th=V_th):
                cell = _hh_cell(kind, V_th=V_th * u.mV)
                cell.loc(RootLocation(0.5)).record("v", "V")
                result = cell.run(dt=DT, duration=self.DURATION)
                v = np.asarray(result.samples["v"].values.to_decimal(u.mV))
                t = np.asarray(result.time.to_decimal(u.ms))
                rows = result.samples["v"].schema.rows
                for member in range(3):
                    (column,) = [i for i, row in enumerate(rows) if row.population_index == member]
                    crossed = np.flatnonzero((v[:-1, column] < V_th) & (v[1:, column] >= V_th))
                    spikes = result.spikes
                    # The event of the step from t[j] to t[j + 1] is at t[j + 1] (CELL-80).
                    np.testing.assert_array_equal(_ms(spikes)[spikes.source_id == member], t[crossed + 1])
                    self.assertLessEqual(len(crossed), 3)
                self.assertGreater(len(result.spikes.source_id), 0)

    def test_members_beyond_the_first_byte_report_their_own_crossings(self):
        # Each step packs eight members to a byte; 20 members span three bytes.
        drive = np.linspace(0.0, 0.2, 20) * u.nA
        cell = _hh_cell("lumped", n=20, amplitudes=None)
        cell.add_current_input("drive", lambda V: drive)
        cell.loc(RootLocation(0.5)).record("v", "V")
        result = cell.run(dt=DT, duration=self.DURATION)
        v = np.asarray(result.samples["v"].values.to_decimal(u.mV))
        crossed = (v[:-1] < 0.0) & (v[1:] >= 0.0)
        rows = [row.population_index for row in result.samples["v"].schema.rows]
        step, column = np.nonzero(crossed)
        order = np.lexsort((np.asarray(rows)[column], step))
        spikes = result.spikes
        np.testing.assert_array_equal(spikes.source_id, np.asarray(rows)[column][order])
        np.testing.assert_array_equal(_ms(spikes), np.asarray(result.time.to_decimal(u.ms))[step[order] + 1])
        self.assertGreater(len(set(spikes.source_id.tolist()) & set(range(16, 20))), 0)

    def test_a_lower_threshold_fires_earlier(self):
        at_zero = _hh_cell("lumped").run(dt=DT, duration=self.DURATION).spikes
        at_minus_20 = _hh_cell("lumped", V_th=-20.0 * u.mV).run(dt=DT, duration=self.DURATION).spikes
        np.testing.assert_array_equal(at_minus_20.source_id, at_zero.source_id)
        self.assertTrue(np.all(_ms(at_minus_20) < _ms(at_zero)))

    def test_batch_rows_are_batch_major_with_their_member_and_batch(self):
        drive = np.asarray([[0.0, 0.05, 0.1], [0.1, 0.0, 0.05]]) * u.nA
        cell = _hh_cell("lumped", amplitudes=None)
        cell.add_current_input("drive", lambda V: drive)
        cell.init_state(batch_size=2)
        batched = cell.run(dt=DT, duration=self.DURATION).spikes
        batch_index = batched.metadata["batch_index"]
        population_index = batched.metadata["population_index"]
        np.testing.assert_array_equal(batch_index, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(population_index, [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(batched.metadata["cv_id"], np.zeros(6))
        for b in range(2):
            single = _hh_cell("lumped", amplitudes=None)
            single.add_current_input("drive", lambda V, b=b: drive[b])
            expected = single.run(dt=DT, duration=self.DURATION).spikes
            rows = batch_index[batched.source_id] == b
            np.testing.assert_array_equal(population_index[batched.source_id][rows], expected.source_id)
            np.testing.assert_array_equal(batched.source_id[rows], 3 * b + expected.source_id)
            np.testing.assert_allclose(_ms(batched)[rows], _ms(expected), rtol=1e-12)
            self.assertGreater(len(expected.source_id), 0)

    def test_a_batched_cable_cell_reports_each_batch(self):
        # One member: staggered with pop_size > 1 and a batch fails on main (a separate fix).
        cell = _hh_cell("cable", n=1, amplitudes=np.asarray([0.3]) * u.nA)
        cell.init_state(batch_size=2)
        batched = cell.run(dt=DT, duration=self.DURATION).spikes
        expected = _hh_cell("cable", n=1, amplitudes=np.asarray([0.3]) * u.nA).run(dt=DT, duration=self.DURATION).spikes
        self.assertGreater(len(expected.source_id), 0)
        for b in range(2):
            rows = batched.metadata["batch_index"][batched.source_id] == b
            np.testing.assert_array_equal(batched.source_id[rows], np.full(len(expected.source_id), b))
            np.testing.assert_allclose(_ms(batched)[rows], _ms(expected), rtol=1e-12)

    def test_a_run_split_at_a_spike_step_concatenates_to_one_run(self):
        for kind in ("lumped", "cable"):
            with self.subTest(kind):
                whole = _hh_cell(kind).run(dt=DT, duration=self.DURATION)
                # The first spike is at the end of its step: step m ends at t_{m+1}.
                m = int(round(_ms(whole.spikes)[0] / 0.025)) - 1
                # Split after step m: its spike belongs to the first run, at its stop time (CELL-80).
                cell = _hh_cell(kind)
                parts = (cell.run(dt=DT, duration=(m + 1) * DT), cell.run(dt=DT, duration=(799 - m) * DT))
                np.testing.assert_array_equal(_ms(parts[0].spikes), [_ms_scalar(parts[0].stop_time)])
                self.assertGreater(_ms(parts[1].spikes)[0], _ms_scalar(parts[0].stop_time))
                assert_same_events(RunResult.concat(parts).spikes, whole.spikes)
                # Split before step m: the second run's first step has it, at the first run's end plus dt.
                cell = _hh_cell(kind)
                parts = (cell.run(dt=DT, duration=m * DT), cell.run(dt=DT, duration=(800 - m) * DT))
                self.assertEqual(len(parts[0].spikes.source_id), 0)
                np.testing.assert_array_equal(_ms(parts[1].spikes)[0], np.asarray(parts[1].time.to_decimal(u.ms))[1])
                assert_same_events(RunResult.concat(parts).spikes, whole.spikes)

    def test_concat_without_spikes_in_a_part_has_none(self):
        cell = _hh_cell("lumped")
        first = cell.run(dt=DT, duration=1.0 * u.ms)
        second = cell.run(dt=DT, duration=1.0 * u.ms)
        stripped = RunResult(
            time=second.time,
            traces=second.traces,
            samples=second.samples,
            start_time=second.start_time,
            stop_time=second.stop_time,
            dt=second.dt,
        )
        self.assertIsNotNone(RunResult.concat((first, second)).spikes)
        self.assertIsNone(RunResult.concat((first, stripped)).spikes)

    def test_reset_state_gives_the_same_spikes_again(self):
        cell = _hh_cell("lumped")
        first = cell.run(dt=DT, duration=self.DURATION).spikes
        cell.reset_state()
        assert_same_events(cell.run(dt=DT, duration=self.DURATION).spikes, first)

    def test_a_new_grid_after_reset_moves_the_port(self):
        cell = _hh_cell("cable")
        self.assertEqual(cell.run(dt=DT, duration=1.0 * u.ms).spikes.metadata["cv_id"].tolist(), [1, 1, 1])
        cell.reset()
        cell.cv_policy = braincell.CVPerBranch(5)
        cell.init_state()
        spikes = cell.run(dt=DT, duration=self.DURATION).spikes
        self.assertEqual(spikes.metadata["cv_id"].tolist(), [2, 2, 2])
        network = braincell.Network()
        twin = _hh_cell("cable")
        twin.cv_policy = braincell.CVPerBranch(5)
        network.add_population("post", twin)
        assert_same_events(spikes, network.run(dt=DT, duration=self.DURATION).events["post"]["spike"])

    def test_a_traced_run_has_no_spikes(self):
        cell = _hh_cell("lumped")
        cell.record("v", "V")
        cell.init_state()
        seen = {}

        @brainstate.transform.jit
        def traced():
            result = cell.run(dt=DT, duration=1.0 * u.ms)
            seen["spikes"] = result.spikes
            return result.samples["v"].values

        self.assertEqual(traced().shape, (40, 3))
        self.assertIsNone(seen["spikes"])
        self.assertIsNotNone(cell.run(dt=DT, duration=1.0 * u.ms).spikes)

    def test_a_two_dimensional_population_flattens_in_c_order(self):
        drive = np.asarray([[0.0, 0.05, 0.1], [0.1, 0.0, 0.05]]) * u.nA
        grid = braincell.Cell.point((2, 3), length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4")
        flat = _hh_cell("lumped", n=6, amplitudes=None)
        _paint_hh(grid)
        grid.add_current_input("drive", lambda V: drive)
        flat.add_current_input("drive", lambda V: drive.reshape(6))
        on_grid = grid.run(dt=DT, duration=self.DURATION).spikes
        assert_same_events(on_grid, flat.run(dt=DT, duration=self.DURATION).spikes)
        self.assertEqual(on_grid.metadata["cv_id"].shape, (6,))
        self.assertGreater(len(on_grid.source_id), 0)


SPIKE_PATHS = {
    "per step": {"_BLOCKED_SPIKE_OUTPUT": False, "_COMPACT_SPIKE_COPY": False},
    "blocked": {"_BLOCKED_SPIKE_OUTPUT": True, "_COMPACT_SPIKE_COPY": False},
    "compacted": {
        "_BLOCKED_SPIKE_OUTPUT": True,
        "_COMPACT_SPIKE_COPY": True,
        "_COMPACT_MIN_BYTES": 0,
        "_COMPACT_CAPACITY_DIVISOR": 1,
    },
}


@contextlib.contextmanager
def _spike_path(name):
    """Plans built inside take the spike output path ``name`` of :data:`SPIKE_PATHS`."""
    with contextlib.ExitStack() as stack:
        for switch, value in SPIKE_PATHS[name].items():
            stack.enter_context(mock.patch.object(spikes_module, switch, value))
        yield


def _high_firing_cell(n=100, solver="rk4"):
    """A population a quarter of whose rows spike at every step from step 2 (CELL-49).

    ``Cell.point(n, length=20 um, radius=10 um, V_init=-65 mV, V_th=-65 mV)``
    on ``solver`` with only ``IL`` (0.3 mS/cm^2, E -65 mV), driven by 1 nA
    sin(2 pi t / 0.1 ms + phi_m), phi_m = (m mod 4) pi / 2 + pi / 4: at
    dt = 0.025 ms a period is 4 steps, and on rk4 each member's V crosses
    -65 mV upward once in each. ``V`` is recorded as ``"V"``.
    """
    cell = braincell.Cell.point(
        n, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, V_th=-65.0 * u.mV, solver=solver
    )
    cell.paint(AllRegion(), mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-65.0 * u.mV))
    phase = (np.arange(n) % 4) * (np.pi / 2) + np.pi / 4
    cell.add_current_input(
        "drive", lambda V: 1.0 * u.nA * u.math.sin(2 * np.pi * (brainstate.environ.get("t") / (0.1 * u.ms)) + phase)
    )
    cell.record("V", "V")
    return cell


def _clamped_hh_cell(n=100):
    """CELL-39's population: ``_hh_cell``'s lumped HH (rk4) under a 10 uA/cm^2 clamp from 5 to 45 ms."""
    area = 2.0 * u.math.pi * (10.0 * u.um) * (20.0 * u.um)
    cell = _hh_cell("lumped", n=n, amplitudes=None)
    cell.place(
        mech.CurrentClamp(
            delay=5.0 * u.ms, durations=40.0 * u.ms, amplitudes=(10.0 * u.uA / u.cm**2 * area).in_unit(u.nA)
        )
    )
    cell.record("V", "V")
    return cell


def _segments(*segments):
    """A ``run(cell)`` of the ``(steps, dt)`` (or ``steps`` at ``DT``) segments, one after another."""
    segments = tuple(segment if isinstance(segment, tuple) else (segment, DT) for segment in segments)

    def run(cell):
        return tuple(cell.run(dt=dt, duration=n * dt) for n, dt in segments)

    return run


def _outcome(build, run):
    """``(results, clock, ports)``: ``run(cell)``'s results of a fresh ``build()``, its clock, its plans' spike ports."""
    cell = build()
    with mock.patch.object(spikes_module, "_grid_events", wraps=spikes_module._grid_events) as whole:
        results = run(cell)
    ports = tuple(plan.spike_port for plan in cell._run_loop_cache.values())
    if any(port.capacity for port in ports):
        # A compacted copy with room for all of the grid is decoded as it is.
        np.testing.assert_equal(whole.call_count, 0)
    return results, cell._clock_host(), ports


def assert_same_runs(actual, expected):
    """Runs pairwise equal: spikes (time, source_id, count, metadata), times, recordings; floats to rtol 1e-12."""
    np.testing.assert_equal(len(actual), len(expected))
    for one, other in zip(actual, expected):
        _assert_same_run(one, other)


def _assert_same_run(actual, expected):
    assert_same_events(actual.spikes, expected.spikes)
    np.testing.assert_equal(dict(actual.spikes.metadata), dict(expected.spikes.metadata))
    np.testing.assert_allclose(_ms(actual), _ms(expected), rtol=1e-12, atol=0.0)
    np.testing.assert_equal(tuple(actual.samples), tuple(expected.samples))
    for name, block in expected.samples.items():
        values = actual.samples[name].values
        np.testing.assert_equal(u.get_unit(values), u.get_unit(block.values))
        np.testing.assert_allclose(u.get_mantissa(values), u.get_mantissa(block.values), rtol=1e-12, atol=0.0)


class SpikeOutputPathTest(unittest.TestCase):
    """CELL-49: each spike output path of ``Cell.run`` gives the result of packing the bits at every step."""

    def setUp(self):
        self.enterContext(_float64())

    def assert_paths_match(self, build, run):
        """Each path's ``run`` of a fresh ``build()`` gives the per-step path's results and clock.

        Returns those results and the spike ports of each path's plans.
        """
        ports = {}
        with _spike_path("per step"):
            expected, clock, ports["per step"] = _outcome(build, run)
        for name in tuple(SPIKE_PATHS)[1:]:
            with self.subTest(path=name), _spike_path(name):
                actual, actual_clock, ports[name] = _outcome(build, run)
                assert_same_runs(actual, expected)
                self.assertEqual(actual_clock, clock)
        self.assertTrue(all(port.block_steps is None and port.capacity is None for port in ports["per step"]))
        return expected, ports

    def test_a_quarter_of_the_high_firing_rows_spike_at_every_step(self):
        with _spike_path("per step"):
            spikes = _high_firing_cell().run(dt=DT, duration=64 * DT).spikes
        # A step's spikes are at its end: step m at (m + 1) dt (CELL-80).
        per_step = np.bincount(np.rint(_ms(spikes) / 0.025).astype(int) - 1, minlength=64)
        np.testing.assert_array_equal(per_step[2:], np.full(62, 25))

    def test_step_counts_with_and_without_a_block_size(self):
        # Blocks of 25, 32, 31 and 23 steps; 37 and 7 steps have no divisor in [8, 32].
        for n_steps, block_steps in ((2000, 25), (64, 32), (31, 31), (69, 23), (37, None), (7, None)):
            with self.subTest(n_steps=n_steps):
                (expected,), ports = self.assert_paths_match(_high_firing_cell, _segments(n_steps))
                self.assertGreater(len(expected.spikes.source_id), 0)
                for name in ("blocked", "compacted"):
                    (port,) = ports[name]
                    self.assertEqual(port.block_steps, block_steps)
                self.assertEqual(ports["compacted"][0].capacity, -(-n_steps * 13 // 4))

    def test_the_clamped_population(self):
        for precision in (32, 64):
            context = contextlib.nullcontext() if precision == 64 else brainstate.environ.context(precision=32)
            with self.subTest(precision=precision), context:
                (expected,), _ = self.assert_paths_match(_clamped_hh_cell, _segments(2000))
                self.assertGreater(len(expected.spikes.source_id), 100)

    def test_a_run_split_in_blocks_of_other_sizes_matches_one_run(self):
        # 31 steps make one block of 31, 69 three of 23; one run of 100 makes four of 25.
        with _spike_path("per step"):
            expected, clock, _ = _outcome(_high_firing_cell, _segments(100))
        for name in SPIKE_PATHS:
            with self.subTest(path=name), _spike_path(name):
                actual, actual_clock, _ = _outcome(_high_firing_cell, _segments(31, 69))
                assert_same_runs((RunResult.concat(actual),), expected)
                self.assertEqual(actual_clock, clock)

    def test_a_cable_and_a_batch(self):
        def batched():
            drive = np.asarray([[0.0, 0.05, 0.1], [0.1, 0.0, 0.05]]) * u.nA
            cell = _hh_cell("lumped", amplitudes=None)
            cell.add_current_input("drive", lambda V: drive)
            cell.init_state(batch_size=2)
            return cell

        def cable():
            cell = _hh_cell("cable")
            cell.loc(RootLocation(0.5)).record("v", "V")
            return cell

        for name, build in (("cable", cable), ("batch", batched)):
            with self.subTest(name):
                (expected,), _ = self.assert_paths_match(build, _segments(800))
                self.assertGreater(len(expected.spikes.source_id), 0)

    def test_a_reduction_model(self):
        def reduced():
            cell = _reduced_cell()
            braincell.connect("drive", source=_netstim(), synapse=cell.synapses["input"], weight=0.2 * u.uS)
            return cell

        (expected,), ports = self.assert_paths_match(reduced, _segments((40, 0.1 * u.ms)))
        np.testing.assert_allclose(_ms(expected.spikes), [0.4, 1.4, 2.4], rtol=1e-9)
        # A reduction model's event counts are not packed: each step stores them.
        for name in SPIKE_PATHS:
            (port,) = ports[name]
            self.assertEqual((port.packed, port.block_steps, port.capacity), (False, None, None))

    def test_a_value_input_acts_on_the_first_step(self):
        # A 20 nA pulse for one step of 0.025 ms lifts the soma (12.6 pF) by
        # about 40 mV: every member fires at once, in the first run only.
        def pulsed(cell):
            cell.add_current_input("pulse", 20.0 * u.nA)
            return (cell.run(dt=DT, duration=64 * DT), cell.run(dt=DT, duration=64 * DT))

        (pulsed_run, later), _ = self.assert_paths_match(lambda: _hh_cell("lumped", amplitudes=None), pulsed)
        np.testing.assert_array_equal(pulsed_run.spikes.source_id, [0, 1, 2])
        self.assertEqual(len(later.spikes.source_id), 0)

    def test_a_detector_reads_the_voltage_before_each_step(self):
        (expected,), _ = self.assert_paths_match(_detector_cell, _segments(800))
        g = np.asarray(expected.samples["g"].values.to_decimal(u.uS))
        self.assertEqual(int((np.diff(g, axis=0) > 0.0).sum()), 3)

    def test_the_arrivals_after_a_change_of_dt(self):
        # CELL-48's runs (run_test.CellArrivalStepTest): segments of 3 then 22 steps
        # (no block, then blocks of 22), of 10 then 20 (blocks of 10 and 20),
        # and of 3, 5 and 13 steps (no block).
        cases = {
            "off the new grid": ((0.06,) + run_test.CellArrivalStepTest.TIMES, ((3, DT), (22, 2 * DT))),
            "on the new grid": (run_test.CellArrivalStepTest.TIMES, ((10, DT), (20, 2 * DT))),
            "two changes": ((0.425, 0.455, 0.475, 0.6, 1.06, 1.375), ((3, DT), (5, 2 * DT), (13, 4 * DT))),
        }
        for name, (times, segments) in cases.items():
            with self.subTest(name):
                (*_, last), _ = self.assert_paths_match(
                    lambda times=times: run_test.CellArrivalStepTest._counting_cell(times), _segments(*segments)
                )
                self.assertGreater(float(np.asarray(last.samples["g"].values.to_decimal(u.uS)).max()), 4.5)

    def test_a_count_over_the_capacity_copies_the_whole_grid(self):
        # Room for one of the 4-byte groups: 64 steps of the high-firing grid have about 200 nonzero.
        with _spike_path("per step"):
            expected, _, _ = _outcome(_high_firing_cell, _segments(64))
        with (
            _spike_path("compacted"),
            mock.patch.object(spikes_module, "_COMPACT_CAPACITY_DIVISOR", 10**9),
            mock.patch.object(spikes_module, "_grid_events", wraps=spikes_module._grid_events) as whole,
        ):
            cell = _high_firing_cell()
            actual = cell.run(dt=DT, duration=64 * DT)
        (plan,) = cell._run_loop_cache.values()
        port = plan.spike_port
        self.assertEqual((port.block_steps, port.capacity), (32, 1))
        self.assertEqual(whole.call_count, 1)
        assert_same_runs((actual,), expected)

    def test_a_grid_of_1_mib_is_compacted(self):
        # 100 rows pack into 13 bytes a step; 80660 steps make 1048580 bytes.
        cell = _high_firing_cell()
        cell.init_state()
        for n_steps, capacity in ((80659, None), (80660, 1048580 // 4 // 64)):
            with self.subTest(n_steps=n_steps):
                port = spikes_module._SpikePort.of(cell, n_steps=n_steps, path=(False, True))
                self.assertEqual((port.block_steps, port.capacity), (None, capacity))
        self.assertIsNone(spikes_module._SpikePort.of(cell, n_steps=80660, path=(True, False)).capacity)
        self.assertEqual(spikes_module._SpikePort.of(cell, n_steps=80660, path=(True, False)).block_steps, 20)

    def test_the_paths_follow_the_platform_of_the_spike_state(self):
        cell = _high_firing_cell()
        cell.init_state()
        reduced = _reduced_cell()
        reduced.init_state()
        for on_gpu in (False, True):
            with self.subTest(on_gpu=on_gpu), mock.patch.object(spikes_module, "_on_gpu", return_value=on_gpu):
                self.assertEqual(spikes_module._spike_output_path(cell), (on_gpu, on_gpu))
                self.assertEqual(spikes_module._spike_output_path(reduced), (False, False))
                with _spike_path("blocked"):
                    self.assertEqual(spikes_module._spike_output_path(cell), (True, False))
        self.assertFalse(spikes_module._on_gpu(jax.device_put(np.zeros(3), jax.devices("cpu")[0])))
        self.assertFalse(spikes_module._on_gpu(np.zeros(3) * u.mV))
