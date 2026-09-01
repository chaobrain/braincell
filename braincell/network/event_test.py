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
from braincell import Branch, CVPerBranch, Cell, EventSequence, EventTable, Morphology, NetStim
from braincell.network.event import EventSourceView, VoltageCrossingSource
from braincell.filter import at


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


class EventSourceTest(unittest.TestCase):
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
        cell[0].V_th = -40.0 * u.mV
        cell[1].V_th = -30.0 * u.mV
        source = VoltageCrossingSource(cell, location=at("dend", 0.5), direction="falling")
        cell.init_state()
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


if __name__ == "__main__":
    unittest.main()
