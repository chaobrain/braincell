"""Tests for the heterogeneous population protocol dataset example."""

from __future__ import annotations

import unittest

import brainstate
import numpy as np

import heterogeneous_protocol_dataset as dataset


def _calibration() -> dataset.Calibration:
    negative = (-0.2, -0.4, -0.6, -0.8, -1.0, -1.2)
    intervals = {count: (0.02 + 0.08 * count, 0.05 + 0.08 * count) for count in range(6)}
    return dataset.Calibration(
        negative_amplitudes_na={site: negative for site in dataset.SITES},
        positive_intervals_na={site: dict(intervals) for site in dataset.SITES},
        negative_soma_minima_mv={site: (-70.0, -80.0, -90.0, -100.0, -110.0, -120.0) for site in dataset.SITES},
    )


class ProtocolCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = dataset.build_protocol_catalog(_calibration())

    def test_catalog_has_expected_counts(self) -> None:
        self.assertEqual(len(self.catalog), 144)
        self.assertEqual(
            {family: sum(protocol.family == family for protocol in self.catalog) for family in dataset.FAMILIES},
            {"dc": 72, "paired": 48, "sine": 24},
        )
        self.assertEqual(
            {site: sum(protocol.injection_site == site for protocol in self.catalog) for site in dataset.SITES},
            {"soma": 48, "dend_a": 48, "dend_b": 48},
        )
        self.assertEqual(
            {
                split: sum(protocol.split == split for protocol in self.catalog)
                for split in ("train", "validation", "test")
            },
            {"train": 108, "validation": 18, "test": 18},
        )

    def test_currents_are_zero_outside_stimulus_window(self) -> None:
        currents = dataset.protocol_currents(self.catalog)
        self.assertEqual(currents.shape, (144, dataset.N_STEPS, 3))
        self.assertFalse(np.any(currents[:, : dataset.EVOKED_START_INDEX]))
        self.assertFalse(np.any(currents[:, dataset.STIMULUS_STOP_INDEX :]))

    def test_finite_window_sines_have_negligible_mean(self) -> None:
        sine = tuple(protocol for protocol in self.catalog if protocol.family == "sine")
        currents = dataset.protocol_currents(sine)
        active = currents[:, dataset.EVOKED_START_INDEX : dataset.STIMULUS_STOP_INDEX]
        site_indices = np.asarray([dataset.SITES.index(protocol.injection_site) for protocol in sine])
        selected = active[np.arange(len(sine)), :, site_indices]
        np.testing.assert_allclose(selected.mean(axis=1), 0.0, atol=3e-4)

    def test_pair_orders_are_balanced(self) -> None:
        paired = [protocol for protocol in self.catalog if protocol.family == "paired"]
        negative_positive = sum(protocol.amplitudes_na[0] < 0.0 for protocol in paired)
        positive_negative = sum(protocol.amplitudes_na[1] < 0.0 for protocol in paired)
        self.assertEqual((negative_positive, positive_negative), (24, 24))


class HeterogeneousModelTest(unittest.TestCase):
    def test_model_has_nine_distinct_conductance_parameters(self) -> None:
        protocol = dataset.build_protocol_catalog(_calibration())[0]
        cell = dataset.build_cell((protocol,), trainable=True)
        parameters = dataset.find_conductance_parameters(cell)
        self.assertEqual(cell.n_cv, 3)
        self.assertEqual(len(parameters), 9)
        self.assertEqual(len({id(parameter) for parameter in parameters.values()}), 9)

    def test_small_population_rollout_is_finite(self) -> None:
        protocols = tuple(
            protocol for protocol in dataset.build_protocol_catalog(_calibration()) if protocol.family == "dc"
        )[:2]
        with brainstate.environ.context(dt=dataset.DT):
            result = dataset.simulate_batch(protocols, probe_names=("soma",))
        self.assertEqual(result.voltages_mv.shape, (2, dataset.N_STEPS, 1))
        self.assertTrue(np.all(np.isfinite(result.voltages_mv)))


class SpikeDetectionTest(unittest.TestCase):
    def test_detects_only_upward_crossings(self) -> None:
        voltage = np.asarray(((-1.0, 1.0, -1.0, 1.0), (-1.0, -0.5, -0.25, -0.1)))
        mask = dataset.spike_mask(voltage)
        np.testing.assert_array_equal(mask[0], np.asarray([False, True, False, True]))
        self.assertFalse(np.any(mask[1]))

    def test_rejects_non_matrix_input(self) -> None:
        with self.assertRaises(ValueError):
            dataset.spike_mask(np.zeros(10))


if __name__ == "__main__":
    unittest.main()
