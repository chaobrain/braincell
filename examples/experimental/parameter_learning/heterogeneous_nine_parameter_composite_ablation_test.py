"""Tests for heterogeneous nine-parameter Composite loss ablations."""

from __future__ import annotations

from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import heterogeneous_nine_parameter_composite_ablation as ablation
import heterogeneous_protocol_dataset as dataset


class LossConfigurationTest(unittest.TestCase):
    def test_approved_component_weights_are_exact(self) -> None:
        expected = {
            "voltage_count": [1.0, 0.0, 0.0, 0.0, 0.4, 0.0],
            "without_count_composite": [1.0, 0.1, 0.25, 0.75, 0.0, 2.0],
            "full_composite": [1.0, 0.1, 0.25, 0.75, 0.4, 2.0],
        }
        self.assertEqual(set(ablation.CONFIGURATIONS), set(expected))
        for name, weights in expected.items():
            np.testing.assert_allclose(ablation.loss_configuration(name).weights, weights)

    def test_unknown_configuration_and_all_zero_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ablation.loss_configuration("unknown")
        with self.assertRaises(ValueError):
            ablation.validate_component_weights(np.zeros(6))

    def test_output_directories_are_independent(self) -> None:
        directories = {ablation.output_directory(name) for name in ablation.CONFIGURATIONS}
        self.assertEqual(len(directories), 3)

    def test_output_directories_use_local_experiment_plot(self) -> None:
        expected_root = Path(__file__).resolve().parent / "plot"
        self.assertEqual(dataset.ARTIFACT_ROOT, expected_root)
        self.assertEqual(dataset.DEFAULT_OUTPUT_DIR.parent, expected_root)
        self.assertEqual(ablation.baseline.DEFAULT_OUTPUT_DIR.parent, expected_root)
        self.assertEqual(ablation.DEFAULT_OUTPUT_ROOT, expected_root)


class CompositeComponentTest(unittest.TestCase):
    def test_identical_traces_have_zero_loss_components(self) -> None:
        voltage = jnp.full((2, dataset.N_STEPS, 3), -65.0)
        mask = jnp.ones((2, dataset.N_STEPS))
        components = ablation.raw_loss_components(voltage, voltage, mask)
        np.testing.assert_allclose(components, 0.0, atol=1e-7)

    def test_dendrite_only_error_does_not_change_soma_components(self) -> None:
        target = jnp.full((1, dataset.N_STEPS, 3), -65.0)
        prediction = target.at[:, 1000:1600, 1].add(5.0)
        mask = jnp.ones((1, dataset.N_STEPS))
        components = np.asarray(ablation.raw_loss_components(prediction, target, mask))[0]
        self.assertTrue(np.all(components[:3] > 0.0))
        np.testing.assert_allclose(components[3:], 0.0, atol=1e-7)

    def test_smooth_count_component_has_nonzero_gradient_near_threshold(self) -> None:
        time = jnp.arange(dataset.N_STEPS)
        target = jnp.full((1, dataset.N_STEPS, 3), -5.0)
        waveform = jnp.exp(-(((time - 1600.0) / 60.0) ** 2))
        mask = jnp.ones((1, dataset.N_STEPS))

        def count_loss(amplitude):
            prediction = target.at[0, :, 0].set(-1.0 + amplitude * 2.0 * waveform)
            return ablation.raw_loss_components(prediction, target, mask)[0, 4]

        gradient = float(jax.grad(count_loss)(1.0))
        self.assertTrue(np.isfinite(gradient))
        self.assertNotEqual(gradient, 0.0)

    def test_normalized_objective_uses_only_active_weighted_components(self) -> None:
        raw = jnp.asarray([[2.0, 100.0, 100.0, 100.0, 4.0, 100.0]])
        normalizers = jnp.asarray([[2.0, 1.0, 1.0, 1.0, 2.0, 1.0]])
        weights = ablation.loss_configuration("voltage_count").weights
        objective = ablation.normalized_component_objective(raw, normalizers, weights)
        expected = (1.0 * 1.0 + 0.4 * 2.0) / 1.4
        np.testing.assert_allclose(objective, [expected])


if __name__ == "__main__":
    unittest.main()
