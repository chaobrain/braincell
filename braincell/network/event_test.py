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

import json
import os
import subprocess
import sys
import unittest

import brainstate
import braintools
import brainunit as u
import jax
import numpy as np

import braincell as bc
from braincell import Branch, CVPerBranch, Cell, EventSequence, EventTable, Morphology, NetStim
from braincell.network.event import (
    EventSourceView,
    VoltageCrossingSource,
    _build_event_schedule,
    _counter_draws_supported,
    _resolve_cell_location_cv,
)
from braincell.filter import RootLocation, at


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

        rng = brainstate.random.RandomState(seed)
        exponential = np.asarray(rng.exponential(scale=1.0, size=(2, 3)), dtype=np.float64)
        first = start_ms + noise * interval_ms * exponential[:, 0]
        gaps = (1.0 - noise[:, None]) * interval_ms[:, None] + noise[:, None] * interval_ms[:, None] * exponential[
            :, 1:
        ]
        expected = np.concatenate(
            [first[:, None], first[:, None] + np.cumsum(gaps, axis=1)],
            axis=1,
        )

        np.testing.assert_allclose(source.event_times.to_decimal(u.ms), expected)

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
    cell.record("v", bc.observe.state("v"))
    return cell


def _cell_run_v(stim, *, durations, dt=0.1 * u.ms):
    """Voltage of a standalone ``Cell.run`` split into ``durations``."""
    cell = _exp_syn_cell()
    bc.connect("in", source=stim, synapse=cell.synapses["syn"], weight=0.01 * u.uS, delay=1.0 * u.ms)
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

    def test_noisy_events_are_a_prefix_of_the_full_schedule(self) -> None:
        # One source: the full draw of a (1, n) matrix is the same for every n.
        small = NetStim(number=8, interval=25.0 * u.ms, noise=0.5, seed=5)
        large = NetStim(number=10**9, interval=25.0 * u.ms, noise=0.5, seed=5)
        table = large._event_table(until_ms=400.0)
        np.testing.assert_array_equal(table.time.to_decimal(u.ms)[:8], small.event_times.to_decimal(u.ms)[0])

        # Several sources: rows after the first depend on max(number), so
        # compare against the full draw of the same shape.
        kwargs = dict(
            size=3,
            start=np.asarray([0.0, 5.0, 10.0]) * u.ms,
            number=np.asarray([400_000, 350_000, 3]),
            interval=np.asarray([25.0, 10.0, 5.0]) * u.ms,
            noise=np.asarray([0.0, 0.5, 1.0]),
            seed=11,
        )
        for precision in (32, 64):
            with self.subTest(precision=precision), brainstate.environ.context(precision=precision):
                self.assertTrue(_counter_draws_supported(np.dtype(brainstate.environ.dftype()).name))
                source = NetStim(**kwargs)
                self.assertFalse(source._schedule_complete)
                source._ensure_schedule(until_ms=200.0)
                width = source._event_times_ms.shape[1]
                self.assertLess(width, 100)
                full_times, full_mask = _build_event_schedule(
                    start_ms=np.asarray([0.0, 5.0, 10.0]),
                    number=source.number,
                    interval_ms=np.asarray([25.0, 10.0, 5.0]),
                    noise=source.noise,
                    seed=11,
                )
                np.testing.assert_array_equal(source._event_times_ms, full_times[:, :width])
                np.testing.assert_array_equal(source._event_mask, full_mask[:, :width])
                self.assertTrue(np.all(source._event_times_ms[:2, -1] > 200.0))

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

        net = bc.Network("prepared")
        stim = net.add_population("stim", NetStim(number=2_000_000, interval=1.0 * u.ms))
        post = net.add_population("post", _exp_syn_cell())
        net.connect("in", source=stim, synapse=post.synapses["syn"], weight=0.01 * u.uS)
        net.prepare_run(dt=0.1 * u.ms)
        self.assertTrue(stim.model._schedule_complete)

    def test_scheduled_sources_only_network_reads_events_up_to_the_end(self) -> None:
        net = bc.Network("sources")
        stim = net.add_population("stim", NetStim(number=10**9, interval=25.0 * u.ms))
        first = net.run(dt=0.1 * u.ms, duration=200.0 * u.ms)
        second = net.run(dt=0.1 * u.ms, duration=100.0 * u.ms)
        np.testing.assert_array_equal(first.events["stim"]["spike"].time.to_decimal(u.ms), np.arange(8) * 25.0)
        np.testing.assert_array_equal(second.events["stim"]["spike"].time.to_decimal(u.ms), 200.0 + np.arange(4) * 25.0)
        self.assertFalse(stim.model._schedule_complete)

    def test_noisy_draws_fall_back_to_the_full_matrix_without_partitionable_threefry(self) -> None:
        previous = jax.config.jax_threefry_partitionable
        jax.config.update("jax_threefry_partitionable", False)
        try:
            self.assertFalse(_counter_draws_supported("float32"))
            source = NetStim(size=2, number=600_000, interval=1.0 * u.ms, noise=1.0, seed=2)
            source._ensure_schedule(until_ms=20.0)
            width = source._event_times_ms.shape[1]
            full_times, _ = _build_event_schedule(
                start_ms=np.zeros(2),
                number=source.number,
                interval_ms=np.ones(2),
                noise=source.noise,
                seed=2,
            )
            np.testing.assert_array_equal(source._event_times_ms, full_times[:, :width])
        finally:
            jax.config.update("jax_threefry_partitionable", previous)

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
