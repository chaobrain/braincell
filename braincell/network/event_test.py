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

import contextlib
import gc
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell as bc
from braincell import Branch, CVPerBranch, Cell, EventSequence, EventTable, Morphology, NetStim
from braincell.network.event import (
    _ARRIVAL_ORIGIN_KEY,
    _MAX_STEP,
    _SCHEDULE_HORIZON_KEY,
    EventSourceView,
    VoltageCrossingSource,
    _ArrivalPlan,
    _resolve_cell_location_cv,
    _round_half_up_steps,
    arrival_origin,
    boundary_step,
    round_half_up_steps_host,
)
from braincell._misc import is_traced_value
from braincell.filter import AllRegion, LocsetMask, RootLocation, at
from braincell._random_test import default_key_data, drawn_seeds
from braincell.morph._testing import make_tree_u


def _two_cv_population(size=2):
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
    morpho = Morphology.from_root(soma, name="soma")
    morpho.soma.attach(dend, name="dend")
    return Cell(
        morpho,
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        V_th=-20.0 * u.mV,
    )


def _definition_draw(seed: int, source: int, column: int) -> float:
    """Draw ``column`` of ``source`` by the definition of the NetStim noise stream (CELL-11b), one element at a time."""
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(seed, impl="threefry2x32"), source), column)
    high, low = (int(word) for word in np.asarray(jax.random.key_data(key)))
    uniform = (((high << 32) | low) >> 11) * 2.0**-53
    return float(-np.log1p(-uniform))


def _times(source) -> np.ndarray:
    return np.asarray(source.event_times.to_decimal(u.ms))


#: The event times of ``NetStim(size=2, start=[1, 2] ms, number=3, interval=[3, 5] ms, noise=[1, 0.25], seed=11)``.
_FORMULA_EVENTS_MS = np.asarray(
    [
        [17.691200921786546, 19.996961869727215, 22.878349547471345],
        [2.0861805660256603, 6.476189290384145, 11.361936217420482],
    ]
)


class NetStimTest(unittest.TestCase):
    def test_scalar_parameters_broadcast_over_size(self) -> None:
        source = NetStim(size=3, start=1.0 * u.ms, number=2, interval=4.0 * u.ms)

        np.testing.assert_allclose(source.start.to_decimal(u.ms), [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(source.number, [2, 2, 2])
        np.testing.assert_allclose(
            source.event_times.to_decimal(u.ms),
            [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
        )

    def test_heterogeneous_parameters_produce_independent_schedules(self) -> None:
        source = NetStim(
            size=3,
            start=np.asarray([0.0, 1.0, 2.0]) * u.ms,
            number=np.asarray([1, 2, 3]),
            interval=np.asarray([2.0, 3.0, 4.0]) * u.ms,
        )

        np.testing.assert_allclose(
            source.event_times.to_decimal(u.ms),
            [[0.0, 2.0, 4.0], [1.0, 4.0, 7.0], [2.0, 6.0, 10.0]],
        )
        np.testing.assert_array_equal(
            source._event_mask,
            [[True, False, False], [True, True, False], [True, True, True]],
        )

    def test_noisy_schedule_is_seeded_and_source_local(self) -> None:
        first = NetStim(size=2, number=4, interval=5.0 * u.ms, noise=1.0, seed=7)
        second = NetStim(size=2, number=4, interval=5.0 * u.ms, noise=1.0, seed=7)
        other = NetStim(size=2, number=4, interval=5.0 * u.ms, noise=1.0, seed=8)

        np.testing.assert_allclose(first.event_times.to_decimal(u.ms), second.event_times.to_decimal(u.ms))
        self.assertFalse(np.allclose(first.event_times.to_decimal(u.ms), other.event_times.to_decimal(u.ms)))

    def test_noisy_schedule_matches_neuron_first_event_and_interval_formula(self) -> None:
        # CELL-11b acceptance 1: draw j of source i is a function of (seed, i, j) only.
        seed = 11
        start_ms = np.asarray([1.0, 2.0])
        interval_ms = np.asarray([3.0, 5.0])
        noise = np.asarray([1.0, 0.25])
        source = NetStim(
            size=2,
            start=start_ms * u.ms,
            number=3,
            interval=interval_ms * u.ms,
            noise=noise,
            seed=seed,
        )

        exponential = np.asarray([[_definition_draw(seed, i, j) for j in range(3)] for i in range(2)])
        np.testing.assert_allclose(
            exponential,
            [
                [5.563733640595515, 0.7685869826468897, 0.9604625592480438],
                [0.06894445282052826, 0.5120069794867872, 0.9085975416290698],
            ],
            rtol=1e-12,
        )
        first = start_ms + noise * interval_ms * exponential[:, 0]
        gaps = (1.0 - noise[:, None]) * interval_ms[:, None] + noise[:, None] * interval_ms[:, None] * exponential[
            :, 1:
        ]
        expected = np.concatenate(
            [first[:, None], first[:, None] + np.cumsum(gaps, axis=1)],
            axis=1,
        )

        np.testing.assert_allclose(_times(source), expected, rtol=1e-12)
        # Before CELL-11b: [[3.8273154497146606, 3.8274499208055204, 5.919950428637094],
        # [2.848641097545624, 7.851643532514572, 11.910894017666578]].
        np.testing.assert_allclose(_times(source), _FORMULA_EVENTS_MS, rtol=1e-12)

    def test_each_source_has_its_own_stream(self) -> None:
        # Acceptance 2. Before CELL-11b one (size, max(number)) draw fed every
        # source, so a longer source 0 moved source 1 by up to 16.21 ms.
        base = dict(size=2, number=4, interval=5.0 * u.ms, noise=1.0, seed=3)
        reference = _times(NetStim(**base))[1]
        np.testing.assert_allclose(
            reference, [2.57321927476678, 5.188461452910328, 6.7670104637006245, 8.431462481683619], rtol=1e-12
        )
        variants = {
            "size=5": dict(base, size=5),
            "number=9": dict(base, number=np.asarray([9, 4])),
            "noise=0": dict(base, noise=np.asarray([0.0, 1.0])),
            "start=7 ms": dict(base, start=np.asarray([7.0, 0.0]) * u.ms),
            "interval=2 ms": dict(base, interval=np.asarray([2.0, 5.0]) * u.ms),
        }
        for name, kwargs in variants.items():
            with self.subTest(variant=name):
                np.testing.assert_allclose(_times(NetStim(**kwargs))[1][:4], reference, rtol=1e-12)

    def test_a_longer_train_keeps_the_earlier_events(self) -> None:
        # Acceptance 3: before CELL-11b number 4 -> 50 moved source 1's first events.
        short = _times(NetStim(size=2, number=4, interval=5.0 * u.ms, noise=1.0, seed=3))
        long = _times(NetStim(size=2, number=50, interval=5.0 * u.ms, noise=1.0, seed=3))
        np.testing.assert_allclose(long[:, :4], short, rtol=1e-12)

    def test_the_stream_does_not_depend_on_precision_or_generator_settings(self) -> None:
        # Acceptance 4. Before CELL-11b the draws took the float precision's
        # dtype (events moved by up to 4.897 ms) and needed the partitionable
        # threefry generator to be generated on demand.
        def events():
            return _times(NetStim(size=2, number=4, interval=5.0 * u.ms, noise=1.0, seed=3))

        reference = events()
        for precision in (32, 64):
            with self.subTest(precision=precision), brainstate.environ.context(precision=precision):
                np.testing.assert_allclose(events(), reference, rtol=1e-12)
        previous = jax.config.jax_threefry_partitionable
        jax.config.update("jax_threefry_partitionable", not previous)
        try:
            np.testing.assert_allclose(events(), reference, rtol=1e-12)
        finally:
            jax.config.update("jax_threefry_partitionable", previous)

    def test_single_noisy_event_is_randomized_but_deterministic_one_is_not(self) -> None:
        noisy = NetStim(start=1.0 * u.ms, number=1, interval=3.0 * u.ms, noise=1.0, seed=11)
        deterministic = NetStim(
            start=1.0 * u.ms,
            number=1,
            interval=3.0 * u.ms,
            noise=0.0,
            seed=11,
        )

        self.assertGreater(float(noisy.event_times[0, 0].to_decimal(u.ms)), 1.0)
        self.assertEqual(float(deterministic.event_times[0, 0].to_decimal(u.ms)), 1.0)

    def test_validates_shape_range_and_units(self) -> None:
        with self.assertRaises(ValueError):
            NetStim(size=2, start=np.asarray([0.0, 1.0, 2.0]) * u.ms)
        with self.assertRaises(ValueError):
            NetStim(interval=0.0 * u.ms)
        with self.assertRaises(ValueError):
            NetStim(noise=1.1)
        with self.assertRaises(TypeError):
            NetStim(start=1.0)


def _exp_syn_cell():
    """One-CV passive Cell with one ``ExpSyn``, recording ``v``."""
    branch = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = Cell(Morphology.from_root(branch, name="soma"), cv_policy=CVPerBranch(1), V_init=-65.0 * u.mV)
    cell.paint(bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(RootLocation(0.5), bc.mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
    cell.record("v", bc.observe.state("V"))
    return cell


def _cell_run_v(stim, *, durations, dt=0.1 * u.ms, extra_sources=()):
    """Voltage of a standalone ``Cell.run`` split into ``durations``."""
    cell = _exp_syn_cell()
    for index, source in enumerate((stim, *extra_sources)):
        bc.connect(f"in{index}", source=source, synapse=cell.synapses["syn"], weight=0.01 * u.uS, delay=1.0 * u.ms)
    runs = [cell.run(dt=dt, duration=duration) for duration in durations]
    return np.concatenate([np.asarray(run.samples["v"].values.to_decimal(u.mV)) for run in runs])


def _network_run(stim, *, duration=200.0 * u.ms, dt=0.1 * u.ms):
    """Voltage and NetStim event times of one ``Network.run``."""
    net = bc.Network("large_number")
    source = net.add_population("stim", stim)
    post = net.add_population("post", _exp_syn_cell())
    net.connect("in", source=source, synapse=post.synapses["syn"], weight=0.01 * u.uS)
    result = net.run(dt=dt, duration=duration)
    return (
        np.asarray(result.samples["post"]["v"].values.to_decimal(u.mV)),
        np.asarray(result.events["stim"]["spike"].time.to_decimal(u.ms)),
    )


def _large_number_memory_report() -> dict:
    """Peak-RSS growth of a 200 ms ``Cell.run`` and ``Network.run`` at ``number=10**9``.

    Run in a fresh process so that other tests do not set the peak.
    """
    import resource

    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    _exp_syn_cell()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    stim = NetStim(number=10**9, interval=25.0 * u.ms)
    _cell_run_v(stim, durations=[200.0 * u.ms], dt=0.025 * u.ms)
    _, events = _network_run(NetStim(number=10**9, interval=25.0 * u.ms), dt=0.025 * u.ms)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    return dict(
        growth_mb=(after - before) / 2**20,
        peak_mb=after / 2**20,
        width=int(stim._event_times_ms.shape[1]),
        events=events.tolist(),
    )


class NetStimLargeNumberTest(unittest.TestCase):
    """``number`` far beyond the run: memory independent of it, events unchanged."""

    @unittest.skipIf(sys.platform == "win32", "resource.getrusage is POSIX only")
    def test_run_memory_does_not_grow_with_number(self) -> None:
        code = (
            "import json; from braincell.network.event_test import _large_number_memory_report; "
            "print(json.dumps(_large_number_memory_report()))"
        )
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        # A full (1, 10**9) schedule needs tens of GB; 10**7 already adds 1.2 GB.
        self.assertLess(report["growth_mb"], 512.0, report)
        self.assertLessEqual(report["width"], 16)
        self.assertEqual(report["events"], [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0])

    def test_cell_and_network_runs_match_finite_number_bit_for_bit(self) -> None:
        def stim(number):
            return NetStim(number=number, interval=25.0 * u.ms)

        expected = _cell_run_v(stim(8), durations=[200.0 * u.ms])
        large = stim(10**9)
        np.testing.assert_array_equal(_cell_run_v(large, durations=[200.0 * u.ms]), expected)
        self.assertFalse(large._schedule_complete)
        self.assertLessEqual(large._event_times_ms.shape[1], 16)
        # Split runs extend the schedule, doubling its width.
        np.testing.assert_array_equal(_cell_run_v(stim(10**9), durations=[50.0 * u.ms] * 4), expected)

        v_small, events_small = _network_run(stim(8))
        v_large, events_large = _network_run(stim(10**9))
        np.testing.assert_array_equal(v_large, v_small)
        np.testing.assert_array_equal(events_large, events_small)
        self.assertEqual(events_large.tolist(), [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0])

    def test_cell_run_prepares_each_netstim_once_beside_other_sources(self) -> None:
        def v(number):
            stim = NetStim(number=number, interval=25.0 * u.ms)
            sequence = EventSequence.from_times([np.asarray([12.5, 60.0]) * u.ms])
            return _cell_run_v(stim, durations=[100.0 * u.ms], extra_sources=(stim, sequence)), stim

        expected, _ = v(8)
        actual, large = v(10**9)
        np.testing.assert_array_equal(actual, expected)
        self.assertLessEqual(large._event_times_ms.shape[1], 16)

    def test_traced_cell_run_generates_every_event(self) -> None:
        number = (1 << 20) + 1
        cell = _exp_syn_cell()
        stim = NetStim(number=number, interval=25.0 * u.ms)
        bc.connect("in", source=stim, synapse=cell.synapses["syn"], weight=0.01 * u.uS, delay=1.0 * u.ms)
        cell.init_state()

        @brainstate.transform.jit
        def run():
            return cell.run(dt=0.1 * u.ms, duration=60.0 * u.ms).samples["v"].values.to_decimal(u.mV)

        traced = np.asarray(run())
        self.assertTrue(stim._schedule_complete)
        self.assertEqual(stim._event_times_ms.shape, (1, number))
        eager = _cell_run_v(NetStim(number=8, interval=25.0 * u.ms), durations=[60.0 * u.ms])
        np.testing.assert_allclose(traced, eager, rtol=1e-6, atol=1e-6)

    def test_schedule_width_doubles_until_it_passes_the_horizon(self) -> None:
        source = NetStim(number=10**9, interval=25.0 * u.ms)
        with mock.patch("braincell.network.event._estimated_width", return_value=1):
            source._ensure_schedule(until_ms=200.0)
        self.assertEqual(source._event_times_ms.shape, (1, 16))
        np.testing.assert_array_equal(source._event_times_ms[0], np.arange(16) * 25.0)

    def test_noisy_events_are_a_prefix_of_the_full_schedule(self) -> None:
        # One source: the first 8 events of an unlimited train.
        small = NetStim(number=8, interval=25.0 * u.ms, noise=0.5, seed=5)
        large = NetStim(number=10**9, interval=25.0 * u.ms, noise=0.5, seed=5)
        table = large._event_table(until_ms=400.0)
        np.testing.assert_allclose(table.time.to_decimal(u.ms)[:8], _times(small)[0], rtol=1e-12)

        # Several sources (CELL-11b acceptance 3): the events generated up to
        # 200 ms are the full schedule of that width.
        kwargs = dict(
            size=3,
            start=np.asarray([0.0, 5.0, 10.0]) * u.ms,
            interval=np.asarray([25.0, 10.0, 5.0]) * u.ms,
            noise=np.asarray([0.0, 0.5, 1.0]),
            seed=11,
        )
        source = NetStim(number=np.asarray([400_000, 350_000, 10**9]), **kwargs)
        self.assertFalse(source._schedule_complete)
        source._ensure_schedule(until_ms=200.0)
        width = source._event_times_ms.shape[1]
        self.assertLess(width, 100)
        full = NetStim(number=width, **kwargs)
        np.testing.assert_allclose(source._event_times_ms, full._event_times_ms, rtol=1e-12)
        np.testing.assert_array_equal(source._event_mask, full._event_mask)
        self.assertTrue(np.all(source._event_times_ms[:, -1] > 200.0))

    def test_entries_without_a_run_end_generate_every_event(self) -> None:
        def lazy():
            return NetStim(size=2, number=600_000, interval=1.0 * u.ms, noise=0.25, seed=3)

        expected = lazy().event_times.to_decimal(u.ms)
        self.assertEqual(expected.shape, (2, 600_000))

        source = lazy()
        count = source.event_count(
            np.asarray([0, 1]),
            t=u.math.asarray(expected[:, -1][0]) * u.ms,
            delay=0.0 * u.ms,
            dt=0.025 * u.ms,
        )
        self.assertTrue(source._schedule_complete)
        self.assertGreaterEqual(int(np.asarray(count)[0]), 1)
        np.testing.assert_array_equal(source._event_times_ms, expected)
        self.assertEqual(len(lazy().events), 1_200_000)

    def test_scheduled_sources_only_network_reads_events_up_to_the_end(self) -> None:
        net = bc.Network("sources")
        stim = net.add_population("stim", NetStim(number=10**9, interval=25.0 * u.ms))
        first = net.run(dt=0.1 * u.ms, duration=200.0 * u.ms)
        second = net.run(dt=0.1 * u.ms, duration=100.0 * u.ms)
        np.testing.assert_array_equal(first.events["stim"]["spike"].time.to_decimal(u.ms), np.arange(8) * 25.0)
        np.testing.assert_array_equal(second.events["stim"]["spike"].time.to_decimal(u.ms), 200.0 + np.arange(4) * 25.0)
        self.assertFalse(stim.model._schedule_complete)

    def test_network_seed_binding_restarts_a_lazy_schedule(self) -> None:
        source = NetStim(number=10**9, interval=25.0 * u.ms, noise=1.0)
        source._ensure_schedule(until_ms=100.0)
        net = bc.Network("bound")
        net.add_population("stim", source)
        self.assertEqual(source._event_times_ms.shape, (1, 0))
        self.assertNotEqual(source._schedule_seed, 0)
        reference = NetStim(number=8, interval=25.0 * u.ms, noise=1.0, seed=source._schedule_seed)
        np.testing.assert_array_equal(
            source._event_table(until_ms=1000.0).time.to_decimal(u.ms)[:8],
            reference.event_times.to_decimal(u.ms)[0],
        )


#: dt of the result-event tests and their run lengths in steps: the window bounds are exact in float32.
_WINDOW_DT_MS = 0.125
_WINDOW_STEPS = (8, 8, 8, 1, 7, 8, 1000)
#: Event starts on window bounds (0 to 5 ms) and half a step either side of them.
_WINDOW_STARTS_MS = np.asarray([0.0, 1.0, 0.9375, 1.0625, 0.5, 2.0, 3.9375, 4.0625, 3.125, 5.0])


class _CountingTimes(np.ndarray):
    """Event times that count the elements read through indexing."""

    reads = 0

    def __getitem__(self, index):
        result = np.asarray(super().__getitem__(index))
        _CountingTimes.reads += result.size
        return result


def _window_stim(noise):
    """Ten sources every 1 ms, numbers 0 to ``10**9``; sources 2, 5 and 8 have ``noise``."""
    return NetStim(
        size=10,
        start=_WINDOW_STARTS_MS * u.ms,
        interval=1.0 * u.ms,
        number=np.asarray([0, 1, 3, 50, 10**9, 7, 10**9, 2, 10**9, 40]),
        noise=np.where(np.arange(10) % 3 == 2, noise, 0.0),
        seed=5,
    )


def _events_in(table, start_ms, stop_ms):
    """Sources and times of the rows of ``table`` in ``[start_ms, stop_ms)``, by time, then source."""
    time_ms = np.asarray(table.time.to_decimal(u.ms))
    rows = np.flatnonzero((time_ms >= start_ms) & (time_ms < stop_ms))
    rows = rows[np.lexsort((table.source_index[rows], time_ms[rows]))]
    return table.source_index[rows], time_ms[rows]


def _hh_points():
    """Three HH point Cells clamped from 1 ms at 0, 0.2 and 0.4 nA: the last two fire repetitively."""
    cell = Cell.point(3, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
    cell.paint(
        bc.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        bc.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        bc.mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
        bc.mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2),
        bc.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    amplitudes = np.asarray([0.0, 0.2, 0.4]) * u.nA
    cell.place(RootLocation(0.5), bc.mech.CurrentClamp(delay=1.0 * u.ms, durations=1.0e3 * u.ms, amplitudes=amplitudes))
    return cell


class ResultEventsTest(unittest.TestCase):
    """``Network.run`` reports the events of its window, reading only those and compiling nothing (CELL-68)."""

    def assert_series(self, series, source, time_ms) -> None:
        self.assertIs(type(series.time.mantissa), np.ndarray)
        self.assertEqual(series.time.mantissa.dtype, np.float64)
        self.assertEqual(series.time.unit, u.ms)
        np.testing.assert_array_equal(series.time.mantissa, time_ms)
        self.assertEqual(series.source_id.dtype, np.int64)
        np.testing.assert_array_equal(series.source_id, source)
        np.testing.assert_array_equal(series.count, np.ones(len(source), dtype=np.int64))

    def test_run_events_are_the_generated_events_of_the_window(self) -> None:
        # The EventSequence holds the periodic sources' events generated up to 50 ms, shuffled.
        periodic = _window_stim(0.0)._event_table(until_ms=50.0)
        order = np.random.default_rng(2).permutation(len(periodic))
        sequence_table = EventTable(source_index=periodic.source_index[order], time=periodic.time[order])
        for bits in (32, 64):
            for noise in (0.0, 0.5):
                with self.subTest(bits=bits, noise=noise), _precision(bits):
                    net = bc.Network("window_events")
                    stim = net.add_population("stim", _window_stim(noise))
                    sequence = net.add_population("sequence", EventSequence(size=10, events=sequence_table))
                    post = net.add_population("post", _exp_syn_cell())
                    rows = np.zeros(10, dtype=int)
                    net.connect("in", source=stim, synapse=post.synapses["syn"][rows], weight=0.01 * u.uS)
                    net.connect("seq", source=sequence, synapse=post.synapses["syn"][rows], weight=0.01 * u.uS)
                    on_start = on_stop = 0
                    for steps in _WINDOW_STEPS:
                        result = net.run(dt=_WINDOW_DT_MS * u.ms, duration=steps * _WINDOW_DT_MS * u.ms)
                        start_ms = float(result.start_time.to_decimal(u.ms))
                        stop_ms = float(result.stop_time.to_decimal(u.ms))
                        table = stim.model._event_table(until_ms=stop_ms)
                        self.assert_series(result.events["stim"]["spike"], *_events_in(table, start_ms, stop_ms))
                        self.assert_series(
                            result.events["sequence"]["event"], *_events_in(sequence_table, start_ms, stop_ms)
                        )
                        table_ms = np.asarray(table.time.to_decimal(u.ms))
                        on_start += int(np.count_nonzero(table_ms == start_ms))
                        on_stop += int(np.count_nonzero(table_ms == stop_ms))
                    # Generated events on a window's start (in it) and on its stop (in the next): sources 1,
                    # 5 (four), 8 and 9 when periodic, 1 and 9 when sources 5 and 8 are noisy.
                    self.assertEqual((on_start, on_stop), (2, 2) if noise else (7, 7))

    def test_run_events_read_the_window_not_the_schedule(self) -> None:
        # 2 sources x 2**19 events, generated in full when built; a 100-step window holds about 200.
        dt = 0.5 * u.ms
        for cells in (False, True):
            with self.subTest(cells=cells):
                net = bc.Network("window_reads")
                stim = net.add_population(
                    "stim", NetStim(size=2, start=[1.0, 3.0] * u.ms, interval=0.5 * u.ms, number=2**19)
                )
                model = stim.model
                self.assertTrue(model._schedule_complete)
                if cells:
                    post = net.add_population("post", _exp_syn_cell())
                    net.connect("in", source=stim, synapse=post.synapses["syn"][[0, 0]], weight=0.01 * u.uS)
                reads = []
                result_events = net._result_events

                def counted(**kwargs):
                    times = model._event_times_ms
                    object.__setattr__(model, "_event_times_ms", times.view(_CountingTimes))
                    _CountingTimes.reads = 0
                    try:
                        return result_events(**kwargs)
                    finally:
                        object.__setattr__(model, "_event_times_ms", times)
                        reads.append(_CountingTimes.reads)

                net._result_events = counted
                sizes = [
                    len(net.run(dt=dt, duration=steps * dt).events["stim"]["spike"].source_id)
                    for steps in (100, 100, 1000, 1, 100)
                ]
                self.assertEqual(sizes, [192, 200, 2000, 2, 200])
                # Two bisections per source of at most log2(2**19) + 1 reads each, then one read per event;
                # a table of every event reads 2**20.
                for read, size in zip(reads, sizes, strict=True):
                    self.assertLessEqual(read, size + 2 * model.size * 20)

    def test_result_events_compile_nothing(self) -> None:
        # Runs of a length no other test uses: on the device, each new spike count compiled a gather.
        net = bc.Network("compiles", seed=1)
        net.add_population("post", _hh_points())
        stim = net.add_population("stim", NetStim(size=2, start=[0.5, 1.5] * u.ms, interval=2.0 * u.ms, number=10**9))
        net.add_population("sequence", EventSequence.from_times([np.asarray([0.25, 3.0]) * u.ms] * 2))
        net.add_population("other", _exp_syn_cell())
        net.connect("in", source=stim, synapse=net.populations["other"].synapses["syn"][[0, 0]], weight=0.01 * u.uS)
        compiles = []

        def listener(event, duration, **kwargs):
            if event == "/jax/core/compile/backend_compile_duration":
                compiles[-1] += 1

        result_events = net._result_events

        def counted(**kwargs):
            compiles.append(0)
            jax.monitoring.register_event_duration_secs_listener(listener)
            try:
                return result_events(**kwargs)
            finally:
                jax.monitoring.unregister_event_duration_listener(listener)

        net._result_events = counted
        dt = 0.025 * u.ms
        results = [net.run(dt=dt, duration=steps * dt) for steps in (333, 333, 666, 999)]
        self.assertEqual(compiles, [0, 0, 0, 0])
        for result in results:
            spikes = result.events["post"]["spike"]
            step_ms = np.asarray(u.get_mantissa(result.time))
            self.assertGreater(len(spikes.source_id), 0)
            # Host arrays, like a standalone Cell.run's spikes: the steps' times, in the run's dtype.
            self.assertIs(type(spikes.time.mantissa), np.ndarray)
            self.assertEqual(spikes.time.mantissa.dtype, step_ms.dtype)
            self.assertEqual(spikes.time.unit, u.get_unit(result.time))
            self.assertTrue(np.all(np.isin(spikes.time.mantissa, step_ms)))

    def test_first_at_or_after_is_a_left_search_of_each_prefix(self) -> None:
        from braincell.network.event import _first_at_or_after

        rng = np.random.default_rng(3)
        times = np.sort(rng.integers(0, 20, (50, 33)).astype(float), axis=1)
        valid = rng.integers(0, 34, 50)
        valid[:2] = (0, 33)
        # Entries past a row's prefix are not events: not in order, and never read.
        times[np.arange(33) >= valid[:, None]] = -1.0
        for value in (-1.0, 0.0, 5.0, 5.5, 19.0, 20.0):
            expected = [np.searchsorted(row[:n], value, side="left") for row, n in zip(times, valid, strict=True)]
            np.testing.assert_array_equal(_first_at_or_after(times, valid, value), expected)
        np.testing.assert_array_equal(_first_at_or_after(np.empty((3, 0)), np.zeros(3, dtype=np.int64), 1.0), 0)


#: dt of the update() horizon tests, and the time of the boundary one step past the step limit.
_HORIZON_DT_MS = 0.5
_HORIZON_MS = 2**31 * _HORIZON_DT_MS
#: 2 * 2**21 entries: generated on demand (more than 2**20).
_HORIZON_NUMBER = 2**21


def _horizon_stim(noise=0.0, number=_HORIZON_NUMBER):
    return NetStim(size=2, start=[1.0, 3.0] * u.ms, interval=1.0e4 * u.ms, number=number, noise=noise, seed=11)


def _horizon_network(noise=0.0, number=_HORIZON_NUMBER):
    """Two NetStim sources driving one ExpSyn: arrivals at 1.3 and 3.3 ms, then one per 10 s."""
    cell = Cell.point(1, V_init=-65.0 * u.mV)
    cell.place(at("soma", 0.5), bc.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV))
    net = bc.Network("horizon", seed=1)
    post = net.add_population("post", cell)
    stim = net.add_population("stim", _horizon_stim(noise, number))
    net.connect(
        "drive",
        source=stim,
        synapse=post.synapses["s"][np.zeros(2, dtype=int)],
        weight=0.001 * u.uS,
        delay=0.3 * u.ms,
    )
    return net, cell, stim.model


def _updates_g(net, cell, n_steps) -> np.ndarray:
    """``g`` before each of ``n_steps`` update() calls, in one ``for_loop``: the samples a run takes."""

    def step(_):
        g = cell.synapses["s"].get("g")
        net.update()
        return g

    gs = brainstate.transform.for_loop(step, jnp.arange(n_steps))
    return np.asarray(u.get_mantissa(gs)).reshape(n_steps, -1)


def _run_g(net, n_steps) -> np.ndarray:
    result = net.run(dt=_HORIZON_DT_MS * u.ms, duration=n_steps * _HORIZON_DT_MS * u.ms)
    g = np.asarray(u.get_mantissa(result.samples["post"]["g"].values))
    return g.reshape(g.shape[0], -1)


def _horizon_run_g(n_steps) -> np.ndarray:
    """``g`` of one ``n_steps`` run of a new ``_horizon_network``."""
    net, cell, _ = _horizon_network()
    cell.record("g", bc.observe.synapse(name="s").state("g"))
    return _run_g(net, n_steps)


class NetStimUpdateHorizonTest(unittest.TestCase):
    """CELL-56: ``prepare_run`` generates NetStim events as far as the step limit, which ``update()`` never passes."""

    def _plan_windows(self, run):
        windows = []
        build = _ArrivalPlan.build.__func__

        def spy(cls, **kwargs):
            plan = build(cls, **kwargs)
            windows.append(plan.window)
            return plan

        with mock.patch.object(_ArrivalPlan, "build", classmethod(spy)):
            run()
        return windows

    def test_prepare_run_generates_events_up_to_the_step_limit(self) -> None:
        net, _, source = _horizon_network()
        net.prepare_run(dt=_HORIZON_DT_MS * u.ms)
        times = source._event_times_ms
        width = times.shape[1]
        # floor((H - start) / interval) + 1 = 107375 events of each source
        # are at or before H, for both starts; on main every one of 2**21.
        self.assertLess(width, _HORIZON_NUMBER)
        self.assertTrue(np.all(times[:, 107374] <= _HORIZON_MS))
        self.assertTrue(np.all(times[:, 107375] > _HORIZON_MS))
        np.testing.assert_array_equal(times, _times(_horizon_stim())[:, :width])

    def test_prepare_run_generates_a_train_that_ends_before_the_limit_in_full(self) -> None:
        # Moved from test_entries_without_a_run_end_generate_every_event.
        net = bc.Network("prepared")
        stim = net.add_population("stim", NetStim(number=2_000_000, interval=1.0 * u.ms))
        post = net.add_population("post", _exp_syn_cell())
        net.connect("in", source=stim, synapse=post.synapses["syn"], weight=0.01 * u.uS)
        net.prepare_run(dt=0.1 * u.ms)
        self.assertTrue(stim.model._schedule_complete)

    def test_update_reads_only_the_arrivals_of_one_step(self) -> None:
        # On main 69a8072 the step limit, to which later arrivals are
        # clipped, held 3979554 (noise 0) and 3979943 (noise 1) of them.
        for noise in (0.0, 1.0):
            with self.subTest(noise=noise):
                net, _, _ = _horizon_network(noise)
                net.prepare_run(dt=_HORIZON_DT_MS * u.ms)
                windows = self._plan_windows(net.update)
                self.assertEqual(len(windows), 1)
                self.assertLessEqual(windows[0], 2)

    def test_arrivals_past_the_step_limit_are_counted_apart(self) -> None:
        far = (2**31 + 5) * _HORIZON_DT_MS
        dt = _HORIZON_DT_MS * u.ms
        plan = _ArrivalPlan.build(
            event_source=np.asarray([0, 0, 0, 0, 1]),
            event_time_ms=np.asarray([1.0, far, far + 3.0, far + 7.0, 2.0]),
            source_index=np.asarray([0, 1, 0]),
            t=0.0 * u.ms,
            delay=0.0 * u.ms,
            dt=dt,
        )
        # Steps 2 and 4, and three arrivals clipped to the limit.
        np.testing.assert_array_equal(plan.steps, [2, 4, _MAX_STEP, _MAX_STEP, _MAX_STEP])
        self.assertEqual(plan.window, 1)
        t = 0.0 * u.ms
        for step, expected in ((2, [1, 0, 1]), (3, [0, 0, 0]), (4, [0, 1, 0]), (_MAX_STEP, [3, 0, 3])):
            with boundary_step(t, step):
                np.testing.assert_array_equal(np.asarray(plan.count(t=t, dt=dt)), expected, err_msg=str(step))

    def test_update_delivers_as_run(self) -> None:
        def updates(precision):
            with brainstate.environ.context(precision=precision):
                net, cell, _ = _horizon_network()
                net.prepare_run(dt=_HORIZON_DT_MS * u.ms)
                return _updates_g(net, cell, 20), _horizon_run_g(20)

        for precision in (64, 32):
            with self.subTest(precision=precision):
                g_update, g_run = updates(precision)
                gc.collect()
                self.assertEqual(g_run.shape, (20, 1))  # one sample per step (L93)
                np.testing.assert_array_equal(g_update, g_run)
                # Arrivals at 1.3 and 3.3 ms: boundaries floor(T / dt + 0.5) =
                # 3 and 7, which act in the step there, seen in the next sample.
                np.testing.assert_array_equal(np.flatnonzero(np.diff(g_update[:, 0]) > 0) + 1, [4, 8])

    def test_update_after_a_run_and_after_reset_state_delivers_as_run(self) -> None:
        expected = _horizon_run_g(20)
        net, cell, source = _horizon_network()
        cell.record("g", bc.observe.synapse(name="s").state("g"))
        first = _run_g(net, 4)
        net.prepare_run(dt=_HORIZON_DT_MS * u.ms)
        width = source._event_times_ms.shape[1]
        self.assertLess(width, _HORIZON_NUMBER)
        np.testing.assert_array_equal(np.concatenate([first, _updates_g(net, cell, 16)]), expected)
        net.reset_state()
        np.testing.assert_array_equal(_updates_g(net, cell, 20), expected)
        self.assertEqual(source._event_times_ms.shape[1], width)

    def test_an_input_registered_after_prepare_run_keeps_the_schedule(self) -> None:
        # The update loop is rebuilt for the input and traced again (L40),
        # reading the events prepare_run generated.
        def bias(V):
            return u.Quantity(np.full(V.shape, 0.01), u.nA)

        expected = _horizon_run_g(20)
        twin, twin_cell, _ = _horizon_network()
        twin.prepare_run(dt=_HORIZON_DT_MS * u.ms)
        _updates_g(twin, twin_cell, 20)
        net, cell, source = _horizon_network()
        net.prepare_run(dt=_HORIZON_DT_MS * u.ms)
        width = source._event_times_ms.shape[1]
        before = _updates_g(net, cell, 6)
        cell.add_current_input("bias", bias)
        after = []
        windows = self._plan_windows(lambda: after.append(_updates_g(net, cell, 14)))
        self.assertEqual(len(windows), 1)
        self.assertLessEqual(windows[0], 2)
        self.assertEqual(source._event_times_ms.shape[1], width)
        np.testing.assert_array_equal(np.concatenate([before, after[0]]), expected)
        v, v_twin = (float(np.asarray(c.V.value.to_decimal(u.mV)).reshape(-1)[0]) for c in (cell, twin_cell))
        self.assertGreater(v, v_twin)

    def test_a_loop_reused_by_a_later_run_delivers_its_events(self) -> None:
        # Loops are cached by step count and schedule width, not by the time
        # they run to: the loop of one run serves a later one of as many
        # steps, while the schedule is wide enough for it.
        def network():
            net = bc.Network("split", seed=1)
            stim = net.add_population("stim", NetStim(number=10**9, interval=25.0 * u.ms, noise=1.0, seed=4))
            post = net.add_population("post", _exp_syn_cell())
            net.connect("in", source=stim, synapse=post.synapses["syn"], weight=0.01 * u.uS)
            post.cell.record("g", bc.observe.synapse(name="syn").state("g"))
            return net, stim.model

        def jumps(g):
            return (np.flatnonzero(np.diff(g.reshape(-1)) > 0) + 1).tolist()

        net, source = network()
        split, widths = [], []
        for _ in range(3):
            split.append(net.run(dt=0.1 * u.ms, duration=100.0 * u.ms).samples["post"]["g"].values.mantissa)
            widths.append(source._event_times_ms.shape[1])
        self.assertEqual(widths[1], widths[2])
        whole, _ = network()
        expected = jumps(
            np.asarray(whole.run(dt=0.1 * u.ms, duration=300.0 * u.ms).samples["post"]["g"].values.mantissa)
        )
        self.assertGreater(len([step for step in expected if step >= 1000]), 2)
        self.assertEqual(jumps(np.concatenate([np.asarray(g) for g in split])), expected)

    def test_update_reusing_the_loop_of_a_one_step_run_delivers_its_events(self) -> None:
        # Of a schedule generated in full, the loops of a one-step run and
        # of update() are one.
        net, cell, _ = _horizon_network(number=5)
        cell.record("g", bc.observe.synapse(name="s").state("g"))
        first = _run_g(net, 1)
        net.prepare_run(dt=_HORIZON_DT_MS * u.ms)
        self.assertEqual(len(net._network_run_loop_cache), 1)
        np.testing.assert_array_equal(np.concatenate([first, _updates_g(net, cell, 19)]), _horizon_run_g(20))


class NetStimDefaultSeedTest(unittest.TestCase):
    """CELL-11b: a noisy NetStim without a seed draws it from the BrainState default stream."""

    def test_a_seedless_noisy_source_draws_its_seed_when_built(self) -> None:
        # Acceptance 6. Before CELL-11b it used the root 0 in every context.
        with brainstate.random.seed_context(42):
            drawn = NetStim(size=2, number=3, noise=1.0)
        expected = NetStim(size=2, number=3, noise=1.0, seed=drawn_seeds(42, 1)[0])
        np.testing.assert_allclose(_times(drawn), _times(expected), rtol=1e-12)
        np.testing.assert_allclose(
            _times(drawn)[0], [18.175300151525736, 33.58823782228345, 34.59078348544616], rtol=1e-12
        )
        self.assertIsNone(drawn.seed)

        def built_in(seed):
            with brainstate.random.seed_context(seed):
                return _times(NetStim(size=2, number=3, noise=1.0))

        np.testing.assert_allclose(built_in(1), built_in(1), rtol=1e-12)
        self.assertFalse(np.allclose(built_in(1), built_in(2)))

    def test_only_a_seedless_noisy_source_advances_the_default_stream(self) -> None:
        cases = {
            "NetStim(noise=1)": (lambda: NetStim(size=2, number=3, noise=1.0), True),
            "NetStim(noise=[0, 0.5])": (lambda: NetStim(size=2, number=3, noise=[0.0, 0.5]), True),
            "NetStim(noise=0)": (lambda: NetStim(size=2, number=3), False),
            "NetStim(noise=1, seed=3)": (lambda: NetStim(size=2, number=3, noise=1.0, seed=3), False),
        }
        for name, (build, advances) in cases.items():
            with self.subTest(source=name), brainstate.random.seed_context(5):
                before = default_key_data()
                build()
                self.assertEqual(not np.array_equal(default_key_data(), before), advances)

    def test_a_seedless_noisy_source_cannot_be_built_under_a_transformation(self) -> None:
        # Acceptance 7. Before CELL-11b it was built with the root 0.
        def in_loop(fn):
            return lambda: brainstate.transform.for_loop(lambda i: fn(), np.arange(2))

        for name, transform in (("jit", brainstate.transform.jit), ("for_loop", in_loop)):
            with self.subTest(transform=name):
                brainstate.random.seed(1)
                before = default_key_data()
                with self.assertRaisesRegex(TypeError, r"NetStim\(seed=None\).*traced"):
                    transform(lambda: NetStim(size=2, number=3, noise=1.0).size)()
                np.testing.assert_array_equal(default_key_data(), before)
        # The suggested fix, an integer seed, and a periodic source build inside jit.
        self.assertEqual(int(brainstate.transform.jit(lambda: NetStim(size=2, number=3, noise=1.0, seed=3).size)()), 2)
        self.assertEqual(int(brainstate.transform.jit(lambda: NetStim(size=2, number=3).size)()), 2)


def _reference_arrival_steps(arrivals_ms, delay, dt):
    """Arrival steps of an arrival matrix: at float64 on the host (CELL-46), in the run precision when traced."""
    if is_traced_value(delay) or is_traced_value(dt) or is_traced_value(arrivals_ms):
        return _round_half_up_steps(u.math.asarray(arrivals_ms) / u.math.asarray(dt.to_decimal(u.ms)))
    return round_half_up_steps_host(np.asarray(arrivals_ms, dtype=np.float64) / float(dt.to_decimal(u.ms)))


def _reference_netstim_count(source, source_index, *, t, delay, dt):
    """``NetStim.event_count`` as the full arrival matrix every call (before CELL-18a), rounded as CELL-46 does."""
    source._ensure_schedule(until_ms=brainstate.environ.get(_SCHEDULE_HORIZON_KEY, None))
    source_index = np.asarray(source_index, dtype=np.int32)
    mask = source._event_mask[source_index]
    times = np.where(mask, source._event_times_ms[source_index], 0.0)
    delay_ms = delay.to_decimal(u.ms)
    if not is_traced_value(delay_ms):
        delay_ms = np.asarray(delay_ms, dtype=np.float64)
    if getattr(delay_ms, "ndim", 0) == 0:
        delay_ms = u.math.broadcast_to(delay_ms, source_index.shape)
    arrival_steps = _reference_arrival_steps(times + delay_ms[..., None], delay, dt)
    current_step = _round_half_up_steps(u.math.asarray(t.to_decimal(u.ms)) / u.math.asarray(dt.to_decimal(u.ms)))
    return u.math.sum(u.math.asarray(mask) & (u.math.asarray(arrival_steps) == current_step), axis=-1)


def _reference_flat_count(table, source_index, *, t, delay, dt):
    """``EventSequence.event_count`` as written before CELL-18a, rounded as CELL-46 does."""
    source_index = np.asarray(source_index, dtype=np.int64).reshape(-1)
    delay_ms = delay.to_decimal(u.ms)
    if not is_traced_value(delay_ms):
        delay_ms = np.asarray(delay_ms, dtype=np.float64)
    if getattr(delay_ms, "ndim", 0) == 0:
        delay_ms = u.math.broadcast_to(delay_ms, source_index.shape)
    event_times = np.asarray(table.time.to_decimal(u.ms), dtype=np.float64)
    arrivals = event_times[None, :] + delay_ms[:, None]
    selected_source = np.asarray(table.source_index)[None, :] == source_index[:, None]
    arrival_steps = _reference_arrival_steps(arrivals, delay, dt)
    current_step = _round_half_up_steps(u.math.asarray(t.to_decimal(u.ms)) / u.math.asarray(dt.to_decimal(u.ms)))
    return u.math.sum(u.math.asarray(selected_source) & (u.math.asarray(arrival_steps) == current_step), axis=-1)


def _half_tie_times(dt_ms: float, count: int) -> np.ndarray:
    """Event times whose float32 step ratio sits a few ulp from a half-step tie.

    Each one rounds to a different step when ``time / dt`` is evaluated as a
    division than as a product with ``1 / dt``, so these pin the arithmetic
    itself, not only its usual outcome.
    """
    rng = np.random.default_rng(18)
    dt32 = np.float32(dt_ms)
    ratios = rng.integers(1, 400, size=200_000).astype(np.float32) + np.float32(0.5)
    offsets = rng.integers(-6, 7, size=ratios.size)
    for step in range(6):
        toward = np.where(offsets > 0, np.inf, -np.inf).astype(np.float32)
        ratios = np.where(np.abs(offsets) > step, np.nextafter(ratios, toward), ratios)
    times = (ratios * dt32).astype(np.float32)

    def rounded(values):
        half = np.floor(values) + np.float32(0.5)
        ulp = np.spacing(np.abs(values))
        return np.floor(np.where(np.abs(values - half) <= 4 * ulp, half, values) + np.float32(0.5))

    divided, multiplied = rounded(times / dt32), rounded(times * (np.float32(1.0) / dt32))
    return times[divided != multiplied][:count].astype(np.float64)


class ScheduledEventCountTest(unittest.TestCase):
    """CELL-18a: scheduled ``event_count`` returns the counts of the full arrival matrix.

    Each case is evaluated at every step of a compiled loop, as ``Cell.run``
    does, and eagerly at a few times, against the matrix formula above.
    """

    DT = 0.025 * u.ms

    def _assert_matches(self, count, reference, steps, any_event=True):
        times = np.arange(steps) * self.DT

        def stepped(fn):
            with brainstate.environ.context(dt=self.DT):
                return brainstate.transform.for_loop(lambda t: (fn(t), reference(t)), times)

        actual, expected = stepped(count)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        self.assertEqual(int(np.sum(np.asarray(expected))) > 0, any_event)
        for t in (0.0 * u.ms, 0.5 * u.ms, 0.5125 * u.ms, 1.0499 * u.ms):
            eager, eager_expected = count(t), reference(t)
            self.assertEqual(np.shape(eager), np.shape(eager_expected))
            np.testing.assert_array_equal(np.asarray(eager), np.asarray(eager_expected))

    def _netstim_cases(self):
        heterogeneous = NetStim(
            size=4,
            start=[0.0, 0.3, 1.025, 2.0] * u.ms,
            number=[5, 0, 7, 3],
            interval=[0.5, 1.0, 0.025, 0.7] * u.ms,
            noise=[0.0, 0.0, 0.5, 1.0],
            seed=18,
        )
        rows = np.asarray([0, 0, 1, 2, 2, 3, 3, 2, -1])
        delays = np.asarray([0.0, 0.0125, 0.0, 0.5, 0.5, 1.0, 0.075, 0.0375, 0.2]) * u.ms
        ties = _half_tie_times(0.025, 64)
        tie_source = NetStim(size=ties.size, start=ties * u.ms, number=2, interval=0.0125 * u.ms)
        return (
            ("heterogeneous rows and delays", heterogeneous, rows, delays),
            ("scalar delay", heterogeneous, rows, 0.25 * u.ms),
            ("scalar index", heterogeneous, np.asarray(2), 0.0 * u.ms),
            ("two-dimensional index", heterogeneous, np.asarray([[0, 2, 3], [3, 3, 2]]), 0.05 * u.ms),
            ("delay in microseconds", heterogeneous, rows[:3], np.asarray([0.0, 12.5, 25.0]) * u.us),
            ("half-step ties", tie_source, np.arange(ties.size), 0.0 * u.ms),
            ("half-step ties shifted by a delay", tie_source, np.arange(ties.size), 0.1 * u.ms),
        )

    def test_netstim_counts_match_the_arrival_matrix(self) -> None:
        for name, source, rows, delay in self._netstim_cases():
            with self.subTest(name):
                self._assert_matches(
                    lambda t: source.event_count(rows, t=t, delay=delay, dt=self.DT),
                    lambda t: _reference_netstim_count(source, rows, t=t, delay=delay, dt=self.DT),
                    steps=480,
                )

    def test_event_sequence_counts_match_the_arrival_matrix(self) -> None:
        ties = _half_tie_times(0.025, 32)
        source_index = np.concatenate([[0, 0, 0, 3, 3, 4, 4, 4], np.full(ties.size, 5)])
        times = np.concatenate([[2.0, 0.5, 0.5, 1.0125, 7.5, 0.0, 0.025, 0.0375], ties])
        sequence = EventSequence(size=6, events=EventTable(source_index=source_index, time=times * u.ms))
        rows = np.asarray([4, 0, 0, 3, 1, 4, 2, 5, 5])
        cases = (
            ("heterogeneous delays", rows, np.asarray([0.0, 0.0125, 0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.1]) * u.ms),
            ("scalar delay", rows, 0.25 * u.ms),
            ("rows without events", np.asarray([1, 2]), 0.0 * u.ms),
        )
        for name, index, delay in cases:
            with self.subTest(name):
                self._assert_matches(
                    lambda t: sequence.event_count(index, t=t, delay=delay, dt=self.DT),
                    lambda t: _reference_flat_count(sequence.events, index, t=t, delay=delay, dt=self.DT),
                    steps=480,
                    any_event=name != "rows without events",
                )

    def test_counts_match_at_float64(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            self.test_netstim_counts_match_the_arrival_matrix()
            self.test_event_sequence_counts_match_the_arrival_matrix()

    def test_empty_schedules_count_nothing(self) -> None:
        silent = NetStim(size=3, number=0)
        empty = EventSequence(size=2, events=EventTable(source_index=np.zeros(0, int), time=np.zeros(0) * u.ms))
        for count, reference in (
            (
                lambda t: silent.event_count(np.asarray([0, 2]), t=t, delay=0.0 * u.ms, dt=self.DT),
                lambda t: _reference_netstim_count(silent, np.asarray([0, 2]), t=t, delay=0.0 * u.ms, dt=self.DT),
            ),
            (
                lambda t: empty.event_count(np.asarray([1, 0, 1]), t=t, delay=0.0 * u.ms, dt=self.DT),
                lambda t: _reference_flat_count(empty.events, np.asarray([1, 0, 1]), t=t, delay=0.0 * u.ms, dt=self.DT),
            ),
        ):
            for t in (0.0 * u.ms, 1.0 * u.ms):
                actual, expected = count(t), reference(t)
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_traced_delay_counts_match_the_arrival_matrix(self) -> None:
        source = NetStim(size=2, start=[0.0, 0.5] * u.ms, number=[4, 3], interval=[0.25, 0.1] * u.ms)
        sequence = EventSequence.from_times([[0.0, 0.25] * u.ms, [0.5] * u.ms])
        rows = np.asarray([0, 1, 1])

        @jax.jit
        def counts(delay_ms, t_ms):
            delay, t = delay_ms * u.ms, t_ms * u.ms
            return (
                source.event_count(rows, t=t, delay=delay, dt=self.DT),
                _reference_netstim_count(source, rows, t=t, delay=delay, dt=self.DT),
                sequence.event_count(rows, t=t, delay=delay, dt=self.DT),
                _reference_flat_count(sequence.events, rows, t=t, delay=delay, dt=self.DT),
            )

        seen = 0
        for t_ms in np.arange(0, 60) * 0.025:
            netstim, netstim_expected, flat, flat_expected = counts(np.asarray([0.0, 0.1, 0.0125]), t_ms)
            np.testing.assert_array_equal(netstim, netstim_expected)
            np.testing.assert_array_equal(flat, flat_expected)
            seen += int(np.sum(netstim_expected)) + int(np.sum(flat_expected))
        self.assertGreater(seen, 0)

    def test_traced_dt_counts_match_the_arrival_matrix(self) -> None:
        # CELL-46: a traced dt has no host value, so the arrivals are rounded
        # in the run precision; the count runs without error.
        source = NetStim(size=2, start=[0.0, 0.5] * u.ms, number=[4, 3], interval=[0.25, 0.1] * u.ms)
        sequence = EventSequence.from_times([[0.0, 0.25] * u.ms, [0.5] * u.ms])
        rows = np.asarray([0, 1, 1])
        for delay in (np.asarray([0.0, 0.1, 0.0125]) * u.ms, 0.0125 * u.ms):

            @jax.jit
            def counts(dt_ms, t_ms):
                dt, t = dt_ms * u.ms, t_ms * u.ms
                return (
                    source.event_count(rows, t=t, delay=delay, dt=dt),
                    _reference_netstim_count(source, rows, t=t, delay=delay, dt=dt),
                    sequence.event_count(rows, t=t, delay=delay, dt=dt),
                    _reference_flat_count(sequence.events, rows, t=t, delay=delay, dt=dt),
                )

            seen = 0
            for t_ms in np.arange(0, 60) * 0.025:
                netstim, netstim_expected, flat, flat_expected = counts(0.025, t_ms)
                np.testing.assert_array_equal(netstim, netstim_expected)
                np.testing.assert_array_equal(flat, flat_expected)
                seen += int(np.sum(netstim_expected)) + int(np.sum(flat_expected))
            self.assertGreater(seen, 0)

    def test_delays_that_do_not_fit_the_rows_take_the_matrix_path(self) -> None:
        source = NetStim(size=2, start=[0.0, 0.5] * u.ms, number=[4, 3], interval=[0.25, 0.1] * u.ms)
        sequence = EventSequence.from_times([[0.0, 0.25] * u.ms, [0.5] * u.ms])
        rows = np.asarray([0, 1, 1])
        one_delay = np.asarray([0.1]) * u.ms
        for t in (0.0 * u.ms, 0.1 * u.ms, 0.6 * u.ms):
            np.testing.assert_array_equal(
                sequence.event_count(rows, t=t, delay=one_delay, dt=self.DT),
                _reference_flat_count(sequence.events, rows, t=t, delay=one_delay, dt=self.DT),
            )
        # A t that is not a scalar also takes the matrix path, here with a scalar delay.
        seen = 0
        for t_ms in (0.0, 0.25, 0.6, 0.85):
            t = np.asarray([t_ms]) * u.ms
            for actual, expected in (
                (
                    source.event_count(rows, t=t, delay=0.25 * u.ms, dt=self.DT),
                    _reference_netstim_count(source, rows, t=t, delay=0.25 * u.ms, dt=self.DT),
                ),
                (
                    sequence.event_count(rows, t=t, delay=0.25 * u.ms, dt=self.DT),
                    _reference_flat_count(sequence.events, rows, t=t, delay=0.25 * u.ms, dt=self.DT),
                ),
            ):
                np.testing.assert_array_equal(actual, expected)
                seen += int(np.sum(expected))
        self.assertGreater(seen, 0)
        two_delays = np.asarray([0.0, 0.1]) * u.ms
        with self.assertRaises(ValueError) as expected:
            _reference_netstim_count(source, rows, t=0.0 * u.ms, delay=two_delays, dt=self.DT)
        with self.assertRaises(type(expected.exception)):
            source.event_count(rows, t=0.0 * u.ms, delay=two_delays, dt=self.DT)


DT_MS = 0.025

#: Model A of the CELL-46 acceptance: event times of the ``EventSequence`` rows
#: (half-step, quarter-step, round and delayed-half-step times) and their delays.
_SEQUENCE_ROWS = (
    0.0125 + 0.075 * np.arange(14),
    0.03125 + 0.075 * np.arange(14),
    np.asarray([100.0, 409.6, 800.0, 999.9875]),
    0.5 + 0.1 * np.arange(20),
)
_SEQUENCE_DELAY = np.asarray([0.0, 0.0, 0.0, 0.0125])
_N_STIM = 100
_STIM_DELAY = np.linspace(0.0, 1.0, _N_STIM)


def _float64():
    stack = contextlib.ExitStack()
    stack.enter_context(jax.enable_x64(True))
    stack.enter_context(brainstate.environ.context(precision=64))
    return stack


def _precision(bits):
    return _float64() if bits == 64 else brainstate.environ.context(precision=32)


def _counting_cell(n_synapses):
    """A passive cell at rest whose ExpSyn (tau 1e9 ms, e = rest) g counts the arrivals of each synapse."""
    cell = Cell(
        Morphology.from_root(Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)),
        cv_policy=CVPerBranch(1),
        V_init=-65.0 * u.mV,
    )
    cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(
        LocsetMask.from_columns(np.zeros(n_synapses, dtype=int), np.linspace(0.05, 0.95, n_synapses)),
        bc.mech.Synapse("ExpSyn", name="s", tau=1e9 * u.ms, e=-65.0 * u.mV),
    )
    cell.record("g", bc.observe.synapse(name="s").state("g"))
    return cell


def _delivered(g):
    """Per synapse, the sample of every arrival, read off the per-sample increments of ``g`` (1 uS each)."""
    g = np.asarray(g, dtype=np.float64)
    counts = np.rint(np.diff(g, axis=0, prepend=np.zeros((1, g.shape[1])))).astype(np.int64)
    return [np.repeat(np.arange(g.shape[0]), counts[:, j]) for j in range(g.shape[1])]


def _host_steps(times_ms, delay_ms, below):
    steps = round_half_up_steps_host((np.asarray(times_ms, dtype=np.float64) + delay_ms) / DT_MS).astype(np.int64)
    return np.sort(steps[steps < below])


class ArrivalStepPrecisionTest(unittest.TestCase):
    """CELL-46: float32 and float64 runs deliver every scheduled event at the step of the host float64 rule.

    On main ``3e59b9f`` float32 rounded the arrivals in the run precision:
    with the NetStim below, 146 of the 20342 arrivals in 1 s and 16146 of
    the 200079 in 10 s fell one step late, and in model A 415 of 49970
    arrivals were delivered one step late by ``Network.run``, an ``update()``
    loop and ``Cell.run`` alike.
    """

    def test_plan_steps_equal_the_host_rule(self):
        for seconds in (1.0, 10.0):
            stop = seconds * 1000.0
            for delay_ms in (np.zeros(200), np.linspace(0.0, 1.0, 200)):
                for bits in (32, 64):
                    with self.subTest(seconds=seconds, delay=float(delay_ms[-1]), bits=bits), _precision(bits):
                        source = NetStim(
                            size=200,
                            start=0.0 * u.ms,
                            interval=10.0 * u.ms,
                            number=int(stop / 10.0 * 1.2) + 20,
                            noise=1.0,
                            seed=3,
                        )
                        source._ensure_schedule()
                        rows, column = np.nonzero(source._event_mask)
                        times = source._event_times_ms[rows, column]
                        keep = times < stop
                        rows, times = rows[keep], times[keep]
                        plan = _ArrivalPlan.build(
                            event_source=rows,
                            event_time_ms=times,
                            source_index=np.arange(200),
                            t=0.0 * u.ms,
                            delay=delay_ms * u.ms,
                            dt=DT_MS * u.ms,
                        )
                        self.assertEqual(plan.steps.dtype, np.int32)
                        self.assertEqual(plan.steps.size, times.size)
                        expected = round_half_up_steps_host((times + delay_ms[rows]) / DT_MS)
                        np.testing.assert_array_equal(np.sort(plan.steps), np.sort(expected).astype(np.int32))

    def _model_a(self, mode, bits):
        """Delivered samples of every arrival in model A over 1 s, and the host steps of its events."""
        n_steps = 40000
        starts = np.random.default_rng(11).uniform(0.0, 2.0, _N_STIM)
        with _precision(bits):
            cell = _counting_cell(_N_STIM + len(_SEQUENCE_ROWS))
            stim = NetStim(size=_N_STIM, start=starts * u.ms, interval=2.0 * u.ms, number=600, noise=1.0, seed=11)
            sequence = EventSequence.from_times([row * u.ms for row in _SEQUENCE_ROWS])
            stim_rows = cell.synapses["s"][np.arange(_N_STIM)]
            sequence_rows = cell.synapses["s"][_N_STIM + np.arange(len(_SEQUENCE_ROWS))]
            if mode == "Cell.run":
                bc.connect("drive", source=stim, synapse=stim_rows, weight=1.0 * u.uS, delay=_STIM_DELAY * u.ms)
                bc.connect(
                    "seq", source=sequence, synapse=sequence_rows, weight=1.0 * u.uS, delay=_SEQUENCE_DELAY * u.ms
                )
                g = cell.run(dt=DT_MS * u.ms, duration=n_steps * DT_MS * u.ms).samples["g"].values
            else:
                net = bc.Network("arrivals", seed=11)
                post = net.add_population("post", cell)
                net.add_population("stim", stim)
                net.add_population("seq", sequence)
                net.connect(
                    "drive",
                    source=stim,
                    synapse=post.synapses["s"][np.arange(_N_STIM)],
                    weight=1.0 * u.uS,
                    delay=_STIM_DELAY * u.ms,
                )
                net.connect(
                    "seq",
                    source=sequence,
                    synapse=post.synapses["s"][_N_STIM + np.arange(4)],
                    weight=1.0 * u.uS,
                    delay=_SEQUENCE_DELAY * u.ms,
                )
                if mode == "Network.run":
                    g = net.run(dt=DT_MS * u.ms, duration=n_steps * DT_MS * u.ms).samples["post"]["g"].values
                else:
                    net.prepare_run(dt=DT_MS * u.ms)
                    (synapse,) = cell.runtime_objects(bc.synapse.ExpSyn).values()

                    def step(_):
                        net.update()
                        return synapse.g.value.reshape(-1)

                    with brainstate.environ.context(dt=DT_MS * u.ms):
                        loop = brainstate.transform.jit(lambda: brainstate.transform.for_loop(step, np.arange(n_steps)))
                        g = loop()
        delivered = _delivered(g.to_decimal(u.uS))
        stim._ensure_schedule()
        expected = [
            _host_steps(stim._event_times_ms[j][stim._event_mask[j]], _STIM_DELAY[j], n_steps) for j in range(_N_STIM)
        ]
        expected += [_host_steps(row, _SEQUENCE_DELAY[r], n_steps) for r, row in enumerate(_SEQUENCE_ROWS)]
        return delivered, expected

    def test_runs_deliver_each_event_at_the_host_step_in_both_precisions(self):
        # Samples are taken after a step: Network.run and Cell.run show an
        # arrival at step s in sample s + 1; the update() loop, in sample s.
        for mode, offset in (("Network.run", 1), ("update loop", 0), ("Cell.run", 1)):
            delivered = {}
            for bits in (32, 64):
                with self.subTest(mode=mode, bits=bits):
                    delivered[bits], expected = self._model_a(mode, bits)
                    self.assertEqual(sum(e.size for e in expected), 49970 + 51)
                    for got, want in zip(delivered[bits], expected):
                        np.testing.assert_array_equal(got - offset, want)
            for got32, got64 in zip(delivered[32], delivered[64]):
                np.testing.assert_array_equal(got32, got64)

    def test_large_step_counts_deliver_at_the_host_step(self):
        # On main, float32 put 40 of 260 events a step late at 2**17, 120, 120
        # and 88 a step early at 2**20, 1.5 * 2**20 and 2**22, and at
        # 2**25 + 3, where t / dt cannot tell neighbouring steps apart,
        # delivered the 260 events 1046 times.
        fractions = np.asarray([0.0, 0.2, 0.25, 0.3, 0.4, 0.45, 0.49, 0.5, 0.51, 0.55, 0.6, 0.75, 0.8])
        bases = (2**17, 2**20, 3 * 2**19, 2**22, 2**25 + 3)
        n_steps = 160
        rows = [np.concatenate([(base + 7 * np.arange(20) + frac) * DT_MS for base in bases]) for frac in fractions]
        for bits in (32, 64):
            with self.subTest(bits=bits), _precision(bits):
                cell = _counting_cell(len(fractions))
                net = bc.Network("large")
                post = net.add_population("post", cell)
                sequence = net.add_population("seq", EventSequence.from_times([row * u.ms for row in rows]))
                net.connect(
                    "seq", source=sequence, synapse=post.synapses["s"][np.arange(len(fractions))], weight=1.0 * u.uS
                )
                net.run(dt=DT_MS * u.ms, duration=DT_MS * u.ms)
                for base in bases:
                    cell._clock_step.value = jnp.asarray(base - 1, dtype=jnp.int32)
                    before = net.run(dt=DT_MS * u.ms, duration=DT_MS * u.ms).samples["post"]["g"].values
                    g = net.run(dt=DT_MS * u.ms, duration=n_steps * DT_MS * u.ms).samples["post"]["g"].values
                    g = np.concatenate([before.to_decimal(u.uS)[-1:], g.to_decimal(u.uS)]) - before.to_decimal(u.uS)[-1]
                    for frac, got in zip(fractions, _delivered(g[1:])):
                        times = (base + 7 * np.arange(20) + frac) * DT_MS
                        steps = _host_steps(times, 0.0, base + n_steps)
                        np.testing.assert_array_equal(got + base - 1, steps[steps >= base], err_msg=f"{base} {frac}")

    def test_segments_build_the_arrival_plans_once(self):
        # The plans are built when the step is traced; later segments and a
        # run after reset_state reuse the compiled loop (L40).
        calls = []
        original = _ArrivalPlan.build.__func__

        def counted(cls, **kwargs):
            calls.append(True)
            return original(cls, **kwargs)

        cell = _counting_cell(3)
        net = bc.Network("segments")
        post = net.add_population("post", cell)
        stim = net.add_population("stim", NetStim(size=2, interval=1.0 * u.ms, number=50, noise=1.0, seed=2))
        sequence = net.add_population("seq", EventSequence.from_times([[0.3, 0.6] * u.ms]))
        net.connect(
            "drive", source=stim, synapse=post.synapses["s"][np.arange(2)], weight=1.0 * u.uS, delay=[0.0, 0.1] * u.ms
        )
        net.connect("seq", source=sequence, synapse=post.synapses["s"][[2]], weight=1.0 * u.uS)
        with mock.patch.object(_ArrivalPlan, "build", classmethod(counted)):
            seen = []
            for index in range(5):
                if index == 3:
                    net.reset_state()
                net.run(dt=DT_MS * u.ms, duration=20 * DT_MS * u.ms)
                seen.append(len(calls))
        self.assertEqual(seen, [2, 2, 2, 2, 2])

    def test_an_event_past_the_step_limit_is_never_delivered(self):
        far = (2**31 + 5) * DT_MS
        plan = _ArrivalPlan.build(
            event_source=np.asarray([0, 0]),
            event_time_ms=np.asarray([0.05, far]),
            source_index=np.asarray([0]),
            t=0.0 * u.ms,
            delay=0.0 * u.ms,
            dt=DT_MS * u.ms,
        )
        np.testing.assert_array_equal(plan.steps, [2, _MAX_STEP])
        cell = _counting_cell(1)
        net = bc.Network("far")
        post = net.add_population("post", cell)
        sequence = net.add_population("seq", EventSequence.from_times([[0.05, far] * u.ms]))
        net.connect("seq", source=sequence, synapse=post.synapses["s"], weight=1.0 * u.uS)
        g = net.run(dt=DT_MS * u.ms, duration=10 * DT_MS * u.ms).samples["post"]["g"].values.to_decimal(u.uS)
        np.testing.assert_array_equal(np.asarray(g).reshape(-1), [0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    def test_a_caller_t_rounds_in_the_run_precision(self):
        # The loop's step is used only for the very t object it was set
        # with; any other t is rounded as t / dt.
        sequence = EventSequence.from_times([np.asarray([7, 9]) * DT_MS * u.ms])
        rows = np.asarray([0])
        dt = DT_MS * u.ms
        t_loop = 9 * dt
        with brainstate.environ.context(t=t_loop, dt=dt):
            with boundary_step(t_loop, 7):
                np.testing.assert_array_equal(sequence.event_count(rows, t=t_loop, delay=0.0 * u.ms, dt=dt), [1])
                other = 9 * dt
                np.testing.assert_array_equal(sequence.event_count(rows, t=other, delay=0.0 * u.ms, dt=dt), [1])
                self.assertIsNot(other, t_loop)
                np.testing.assert_array_equal(sequence.event_count(rows, t=8 * dt, delay=0.0 * u.ms, dt=dt), [0])
            with boundary_step(t_loop, None):
                np.testing.assert_array_equal(sequence.event_count(rows, t=t_loop, delay=0.0 * u.ms, dt=dt), [1])
        # With step 8 for the loop's t there is no event, though t / dt = 9
        # has one: the step comes from the loop, not from t.
        with boundary_step(t_loop, 8):
            np.testing.assert_array_equal(sequence.event_count(rows, t=t_loop, delay=0.0 * u.ms, dt=dt), [0])

    def test_arrivals_count_from_an_arrival_origin(self):
        # CELL-48: steps of 0.05 ms from 0.075 ms, whose boundary a run loop
        # numbers 2. An arrival at 0.3 ms is (0.3 - 0.075) / 0.05 = 4.5 steps
        # after it, a tie, so it is due at boundary 2 + 5 = 7 (0.325 ms); one
        # at 0.5125 ms, 8.75 steps after, at 2 + 9 = 11 (0.525 ms). From time
        # 0 they were due at 6 and 10 (0.3 / 0.05, 0.5125 / 0.05 = 10.25).
        # The plan, a t the caller chose and a traced delay agree.
        sequence = EventSequence.from_times([np.asarray([0.3, 0.5125]) * u.ms])
        rows = np.asarray([0])
        dt = 0.05 * u.ms
        expected = {6: 0, 7: 1, 10: 0, 11: 1, 12: 0}

        @jax.jit
        def traced(delay_ms, t_ms, step):
            t = t_ms * u.ms
            with boundary_step(t, step):
                return sequence.event_count(rows, t=t, delay=delay_ms * u.ms, dt=dt)

        with arrival_origin(0.075, 2):
            for step, count in expected.items():
                t = (0.075 + (step - 2) * 0.05) * u.ms
                with self.subTest(step=step):
                    with boundary_step(t, step):
                        np.testing.assert_array_equal(sequence.event_count(rows, t=t, delay=0.0 * u.ms, dt=dt), [count])
                    np.testing.assert_array_equal(sequence.event_count(rows, t=t, delay=0.0 * u.ms, dt=dt), [count])
                    np.testing.assert_array_equal(traced(np.zeros(1), t.to_decimal(u.ms), step), [count])
        # Outside the context, and with an origin of None, steps count from time 0.
        t = 0.3 * u.ms
        for context in (contextlib.nullcontext(), arrival_origin(None, 2)):
            with context, boundary_step(t, 6):
                self.assertIsNone(brainstate.environ.get(_ARRIVAL_ORIGIN_KEY, None))
                np.testing.assert_array_equal(sequence.event_count(rows, t=t, delay=0.0 * u.ms, dt=dt), [1])

    def test_a_subclass_with_the_old_signature_works_in_a_network(self):
        # The public signature is unchanged: a subclass that overrides
        # event_count(source_index, *, t, delay, dt) receives the same
        # keywords. One that passes its own t on gets t / dt rounded.
        received = []

        class Recording(EventSequence):
            def event_count(self, source_index, *, t, delay, dt):
                received.append(("t", "delay", "dt"))
                return super().event_count(source_index, t=t, delay=delay, dt=dt)

        class Leading(EventSequence):
            def event_count(self, source_index, *, t, delay, dt):
                return super().event_count(source_index, t=t + dt, delay=delay, dt=dt)

        times = [np.asarray([0.3, 0.5125, 2.0]) * u.ms]
        g = {}
        for bits in (32, 64):
            for name, kind in (("plain", EventSequence), ("recording", Recording), ("leading", Leading)):
                with _precision(bits):
                    cell = _counting_cell(1)
                    net = bc.Network("custom")
                    post = net.add_population("post", cell)
                    sequence = net.add_population("seq", kind.from_times(times))
                    self.assertIsInstance(sequence.model, kind)
                    net.connect("seq", source=sequence, synapse=post.synapses["s"], weight=1.0 * u.uS)
                    values = net.run(dt=DT_MS * u.ms, duration=100 * DT_MS * u.ms).samples["post"]["g"].values
                    increments = np.diff(np.asarray(values.to_decimal(u.uS)).reshape(-1), prepend=0.0)
                    g[bits, name] = np.flatnonzero(increments > 0.5)
        self.assertTrue(received)
        for bits in (32, 64):
            # 0.3, 0.5125 (a half-step tie, to the later step) and 2.0 ms: steps 12, 21, 80, seen a sample later.
            np.testing.assert_array_equal(g[bits, "plain"], [13, 22, 81])
            np.testing.assert_array_equal(g[bits, "recording"], g[bits, "plain"])
            np.testing.assert_array_equal(g[bits, "leading"], g[bits, "plain"] - 1)


class EventSourceTest(unittest.TestCase):
    def test_threshold_equality_directions_and_surrogate_override(self):
        cell = _two_cv_population(size=1)
        rising = VoltageCrossingSource(cell, threshold=0.0 * u.mV)
        falling = VoltageCrossingSource(cell, threshold=0.0 * u.mV, direction="falling")
        custom = VoltageCrossingSource(cell, threshold=0.0 * u.mV, spk_fun=braintools.surrogate.Sigmoid())
        cell.init_state()
        for last in (-1.0, 0.0, 1.0):
            for new in (-1.0, 0.0, 1.0):
                cell._event_previous_V.value = np.full((1, 2), last) * u.mV
                cell.V.value = np.full((1, 2), new) * u.mV
                for detector, expected in (
                    (rising, last < 0.0 <= new),
                    (falling, last > 0.0 >= new),
                    (custom, last < 0.0 <= new),
                ):
                    np.testing.assert_array_equal(detector.current_event_count([0]), [float(expected)])
        cell.reset_state()
        np.testing.assert_array_equal(rising.current_event_count([0]), [0.0])
        with self.assertRaises(TypeError):
            VoltageCrossingSource(cell, spk_fun="not callable")

    def test_event_source_view_preserves_order_and_duplicates(self) -> None:
        source = NetStim(size=3)
        view = source[[2, 0, 2]]

        self.assertIsInstance(view, EventSourceView)
        self.assertIs(view.owner, source)
        np.testing.assert_array_equal(view.source_id, [2, 0, 2])
        np.testing.assert_array_equal(view[[2, 0]].source_id, [2, 2])

    def test_event_sequence_uses_flat_event_table(self) -> None:
        sequence = EventSequence(
            size=3,
            events=EventTable(
                source_index=[0, 0, 2],
                time=np.asarray([1.0, 2.0, 1.5]) * u.ms,
            ),
        )

        self.assertEqual(len(sequence.events), 3)
        np.testing.assert_array_equal(sequence.events.event_id, [0, 1, 2])
        np.testing.assert_array_equal(
            sequence.event_count(
                np.asarray([0, 1, 2]),
                t=1.5 * u.ms,
                delay=np.zeros(3) * u.ms,
                dt=0.1 * u.ms,
            ),
            [0, 0, 1],
        )

    def test_netstim_exposes_generated_flat_event_table(self) -> None:
        source = NetStim(size=2, start=[1.0, 2.0] * u.ms, number=[2, 1], interval=3.0 * u.ms)

        np.testing.assert_array_equal(source.events.source_index, [0, 0, 1])
        np.testing.assert_allclose(source.events.time.to_decimal(u.ms), [1.0, 4.0, 2.0])

    def test_cell_default_spike_output_gathers_root_cv(self) -> None:
        cell = _two_cv_population()
        cell.init_state()
        cell.spike.value = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        source = cell.event_outputs["spike"]

        self.assertEqual(len(source), 2)
        self.assertEqual(source.owner.cv_id, 0)
        np.testing.assert_array_equal(source.owner.current_event_count(source.source_id), [1.0, 0.0])

    def test_voltage_crossing_source_uses_selected_cv_and_threshold(self) -> None:
        cell = _two_cv_population()
        source = VoltageCrossingSource(
            cell[[1, 0]],
            location=at("dend", 0.5),
            threshold=-40.0 * u.mV,
        )
        cell.init_state()
        cell._event_previous_V.value = np.asarray([[-65.0, -50.0], [-65.0, -45.0]]) * u.mV
        cell.V.value = np.asarray([[-65.0, -35.0], [-65.0, -42.0]]) * u.mV

        np.testing.assert_array_equal(source.population_index, [1, 0])
        np.testing.assert_array_equal(source.location_index, [0, 0])
        np.testing.assert_array_equal(source.cv_id, [1, 1])
        np.testing.assert_array_equal(source.current_event_count([0, 1]), [False, True])

    def test_voltage_crossing_source_defaults_to_root_and_cell_threshold(self) -> None:
        cell = _two_cv_population()
        source = VoltageCrossingSource(cell)
        cell.init_state()
        cell.spike.value = np.asarray([[1.0, 0.0], [0.0, 1.0]])

        np.testing.assert_array_equal(source.population_index, [0, 1])
        np.testing.assert_array_equal(source.location_index, [0, 0])
        np.testing.assert_array_equal(source.cv_id, [0, 0])
        np.testing.assert_array_equal(source.current_event_count(source.ids), [1.0, 0.0])

    def test_voltage_crossing_source_expands_all_cv_endpoints_population_major(self) -> None:
        cell = _two_cv_population()
        source = VoltageCrossingSource(cell, location=cell.cv_midpoints, name="all_cv")
        cell.init_state()
        cell.spike.value = np.asarray([[1.0, 0.0], [0.0, 1.0]])

        self.assertEqual(source.size, 4)
        np.testing.assert_array_equal(source.population_index, [0, 0, 1, 1])
        np.testing.assert_array_equal(source.location_index, [0, 1, 0, 1])
        np.testing.assert_array_equal(source.cv_id, [0, 1, 0, 1])
        np.testing.assert_array_equal(source.current_event_count(source.ids), [1.0, 0.0, 0.0, 1.0])

    def test_voltage_crossing_source_falling_uses_heterogeneous_cell_threshold(self) -> None:
        cell = _two_cv_population()
        cell.init_state()
        cell[0].V_th = -40.0 * u.mV
        cell[1].V_th = -30.0 * u.mV
        source = VoltageCrossingSource(cell, location=at("dend", 0.5), direction="falling")
        cell._event_previous_V.value = np.asarray([[-65.0, -35.0], [-65.0, -25.0]]) * u.mV
        cell.V.value = np.asarray([[-65.0, -45.0], [-65.0, -35.0]]) * u.mV

        np.testing.assert_array_equal(source.current_event_count(source.ids), [True, True])

    def test_voltage_crossing_source_explicit_threshold_broadcasts_over_locations(self) -> None:
        cell = _two_cv_population()
        source = VoltageCrossingSource(
            cell,
            location=cell.cv_midpoints,
            threshold=np.asarray([[-60.0, -40.0], [-50.0, -30.0]]) * u.mV,
        )
        cell.init_state()
        cell._event_previous_V.value = np.asarray([[-65.0, -45.0], [-55.0, -35.0]]) * u.mV
        cell.V.value = np.asarray([[-55.0, -35.0], [-45.0, -25.0]]) * u.mV

        np.testing.assert_array_equal(source.current_event_count(source.ids), [True, True, True, True])

    def test_voltage_crossing_source_broadcasts_population_threshold_vector(self) -> None:
        cell = _two_cv_population()
        source = VoltageCrossingSource(
            cell,
            location=at("dend", 0.5),
            threshold=np.asarray([-40.0, -30.0]) * u.mV,
        )
        cell.init_state()
        cell._event_previous_V.value = np.asarray([[-65.0, -45.0], [-65.0, -35.0]]) * u.mV
        cell.V.value = np.asarray([[-65.0, -35.0], [-65.0, -25.0]]) * u.mV

        np.testing.assert_array_equal(source.current_event_count(source.ids), [True, True])

    def test_voltage_crossing_source_rejects_explicit_none_threshold(self) -> None:
        with self.assertRaisesRegex(TypeError, "voltage quantity"):
            VoltageCrossingSource(_two_cv_population(), threshold=None)

    def test_event_output_is_available_before_cell_initialization(self) -> None:
        cell = _two_cv_population()
        source = cell[1].event_outputs["spike"]

        np.testing.assert_array_equal(source.source_id, [1])
        with self.assertRaisesRegex(RuntimeError, "init_state"):
            source.owner.current_event_count(source.source_id)


class VoltageCrossingIndexHeldLocationTest(unittest.TestCase):
    """A detector located by a resolved mask raises once a removal shifts its indices.

    Tree U: soma 0, dend 1, axon 2, apical 3, one CV per branch. Removing
    ``dend`` puts apical on index 2, so the axon mask ``(2, 0.5)`` would
    silently watch apical's CV.
    """

    def test_a_mask_location_raises_at_the_next_cv_read_after_a_removal(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5).evaluate(morpho))
        np.testing.assert_array_equal(detector.cv_id, [2])

        morpho.remove("dend")

        pattern = r"VoltageCrossingSource location .*LocsetMask.*revision \d+.*at\(\"axon\", 0\.5\)"
        with self.assertRaisesRegex(RuntimeError, pattern):
            detector.cv_id
        with self.assertRaisesRegex(RuntimeError, pattern):
            detector.cv_id

    def test_an_expression_location_follows_a_removal(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5))

        morpho.remove("dend")

        (cv_id,) = detector.cv_id
        self.assertEqual(morpho.branch(index=cell.cvs[cv_id].branch_id).name, "axon")

    def test_an_expression_on_a_removed_branch_names_the_detector(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at("dend", 0.5))

        morpho.remove("dend")

        pattern = r"VoltageCrossingSource location AtLocation\(branch='dend'.*names branch 'dend'.*does not have"
        with self.assertRaisesRegex(KeyError, pattern):
            detector.cv_id
        # Reusing the name gives the expression a branch to follow again.
        morpho.soma.dend = Branch.from_lengths(lengths=[40.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")
        (cv_id,) = detector.cv_id
        self.assertEqual(morpho.branch(index=cell.cvs[cv_id].branch_id).name, "dend")

    def test_an_unknown_branch_name_names_the_detector_at_construction(self) -> None:
        cell = Cell(make_tree_u())
        with self.assertRaisesRegex(KeyError, r"VoltageCrossingSource location .*names branch 'dned'"):
            VoltageCrossingSource(cell, location=at("dned", 0.5))

    def test_a_mask_location_survives_an_append(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5).evaluate(morpho))

        morpho.soma.extra = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")

        np.testing.assert_array_equal(detector.cv_id, [2])
        self.assertEqual(cell.n_cv, 5)

    def test_a_removal_before_init_is_still_reported_after_init(self) -> None:
        # MORPH-14: the initialized Cell runs on a copy of its tree; the
        # detector's indices belong to the tree it was declared on.
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5).evaluate(morpho))

        morpho.remove("dend")
        cell.init_state()

        with self.assertRaisesRegex(RuntimeError, r"VoltageCrossingSource location .*LocsetMask"):
            detector.cv_id

    def test_a_removal_after_init_waits_for_reset(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5).evaluate(morpho))
        cell.init_state()

        morpho.remove("dend")

        np.testing.assert_array_equal(detector.cv_id, [2])
        cell.reset()
        with self.assertRaisesRegex(RuntimeError, r"VoltageCrossingSource location .*LocsetMask"):
            detector.cv_id

    def test_a_mask_resolved_on_the_snapshot_is_checked_after_reset(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.init_state()
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5).evaluate(cell.morpho))

        morpho.remove("dend")

        np.testing.assert_array_equal(detector.cv_id, [2])
        cell.reset()
        with self.assertRaisesRegex(RuntimeError, r"VoltageCrossingSource location .*LocsetMask"):
            detector.cv_id

    def test_a_mask_on_a_morphology_reassigned_away_is_not_checked(self) -> None:
        first = make_tree_u()
        cell = Cell(first)
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5).evaluate(first))
        cell.morphology = make_tree_u()

        first.remove("dend")

        np.testing.assert_array_equal(detector.cv_id, [2])


class VoltageCrossingIndexExpressionTest(unittest.TestCase):
    """MORPH-17: a detector at ``at(2, 0.5)`` raises once an edit shifts index 2.

    Tree U: soma 0, dend 1, axon 2, apical 3, one CV per branch; removing
    ``dend`` puts apical on index 2, which the expression used to follow.
    """

    _STALE = r"VoltageCrossingSource location selects AtLocation\(branch=2.*by default index 2, .*removed or moved"

    @staticmethod
    def _synapse():
        return bc.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV)

    def test_the_next_cv_read_after_a_removal_raises(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at(2, 0.5))
        np.testing.assert_array_equal(detector.cv_id, [2])

        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, self._STALE):
            detector.cv_id

    def test_a_removal_before_init_is_still_reported_after_init(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at(2, 0.5))

        morpho.remove("dend")
        cell.init_state()

        with self.assertRaisesRegex(ValueError, self._STALE):
            detector.cv_id

    def test_a_removal_after_init_waits_for_reset(self) -> None:
        for declared in ("before init", "on the snapshot"):
            with self.subTest(declared):
                morpho = make_tree_u()
                cell = Cell(morpho)
                detector = VoltageCrossingSource(cell, location=at(2, 0.5)) if declared == "before init" else None
                cell.init_state()
                if detector is None:
                    detector = VoltageCrossingSource(cell, location=at(2, 0.5))

                morpho.remove("dend")

                np.testing.assert_array_equal(detector.cv_id, [2])
                cell.reset()
                with self.assertRaisesRegex(ValueError, self._STALE):
                    detector.cv_id

    def test_a_detector_by_index_survives_an_append(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        detector = VoltageCrossingSource(cell, location=at(2, 0.5))

        morpho.soma.extra = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")

        np.testing.assert_array_equal(detector.cv_id, [2])

    def test_a_connected_detector_raises_before_init(self) -> None:
        # Without the check at init_state the Cell initialized and the first
        # run raised, when a Cell owned by a Network can no longer reset.
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("soma", 0.5), self._synapse())
        bc.connect("c", source=VoltageCrossingSource(cell, location=at(2, 0.5)), synapse=cell.synapses["s"])

        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, self._STALE):
            cell.init_state()
        self.assertFalse(cell._initialized)
        # The recovery the message names: a new detector, connected in place of the old one.
        cell.connections["c"].remove()
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5))
        bc.connect("c2", source=detector, synapse=cell.synapses["s"])
        cell.init_state()
        np.testing.assert_array_equal(detector.cv_id, [1])

    def test_a_network_with_a_connected_detector_initializes_no_cell(self) -> None:
        morpho = make_tree_u()
        pre, post = Cell(morpho), Cell(make_tree_u())
        post.place(at("soma", 0.5), self._synapse())
        network = bc.Network("n")
        network.add_population("pre", pre)
        target = network.add_population("post", post)
        detector = VoltageCrossingSource(pre, location=at(2, 0.5), name="det")
        network.connect("c", source=detector, synapse=target.synapses["s"], weight=0.1 * u.uS)

        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, self._STALE):
            network.init_state()
        self.assertFalse(pre._initialized)
        self.assertFalse(post._initialized)

    def test_a_connected_mask_detector_raises_before_init_too(self) -> None:
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("soma", 0.5), self._synapse())
        detector = VoltageCrossingSource(cell, location=at("axon", 0.5).evaluate(morpho))
        bc.connect("c", source=detector, synapse=cell.synapses["s"])

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, r"VoltageCrossingSource location .*LocsetMask"):
            cell.init_state()
        self.assertFalse(cell._initialized)


class ResolveCellLocationCvTest(unittest.TestCase):
    """The one place a presynaptic location becomes a CV id.

    ``braincell.network.lowering`` used to carry a second copy of this,
    ``resolve_source_cv``, which had no production caller -- only its own
    test. These assertions came from there.
    """

    def _named_dend_cell(self) -> Cell:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.dend = Branch.from_lengths(
            lengths=[100.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="dendrite",
        )
        return Cell(morpho, cv_policy=CVPerBranch(), pop_size=(2,))

    def test_a_single_location_resolves_to_its_owning_cv(self) -> None:
        cell = self._named_dend_cell()
        self.assertEqual(_resolve_cell_location_cv(cell, RootLocation(0.5)), 0)
        self.assertEqual(_resolve_cell_location_cv(cell, at("dend", 0.5)), 1)

    def test_a_multi_point_locset_is_rejected(self) -> None:
        cell = self._named_dend_cell()
        with self.assertRaisesRegex(ValueError, "resolve to one point"):
            _resolve_cell_location_cv(cell, at("soma", 0.5) | at("dend", 0.5))


if __name__ == "__main__":
    unittest.main()
