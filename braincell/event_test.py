import unittest

import brainstate
import brainunit as u
import numpy as np

from braincell import NetStim


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


if __name__ == "__main__":
    unittest.main()
