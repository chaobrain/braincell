"""Tests for the example-local three-conductance training core."""

from __future__ import annotations

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import conductance_learning_core as core


class ConductanceLearningProblemTest(unittest.TestCase):
    """Verify target construction, parameter ownership, and differentiation."""

    @classmethod
    def setUpClass(cls):
        """Build one reusable deterministic problem and gradient."""
        cls.problem = core.make_training_problem()
        grad_fn = brainstate.transform.grad(
            lambda: cls.problem.loss_with_aux("composite"),
            grad_states=cls.problem.param_states,
            has_aux=True,
            return_value=True,
        )
        cls.gradients, cls.loss, cls.components = grad_fn()

    def test_target_has_subthreshold_segments_and_one_spike(self):
        """Keep the intended mixed subthreshold/spiking task stable."""
        target_mv = np.asarray(self.problem.target.voltages.to_decimal(u.mV))[:, 0]
        spikes = core.hard_spike_indices(self.problem.target.voltages)
        self.assertEqual(spikes.size, 1)
        self.assertAlmostEqual(spikes[0] * core.DT.to_decimal(u.ms), 72.675, places=2)
        for start, stop in ((10.0, 20.0), (30.0, 40.0), (50.0, 60.0)):
            sample = target_mv[int(start / core.DT.to_decimal(u.ms)) : int(stop / core.DT.to_decimal(u.ms))]
            self.assertLess(float(np.max(sample)), 0.0)

    def test_selects_exactly_three_trainable_shared_scalars(self):
        """Select only leak, sodium, and potassium optimizer states."""
        self.assertEqual(tuple(self.problem.param_states), core.PARAMETER_NAMES)
        for state in self.problem.param_states.values():
            self.assertIsInstance(state, brainstate.ParamState)
            self.assertEqual(np.shape(state.value), ())
        target_parameters = core.find_parameters(self.problem.target_cell)
        for parameter in target_parameters:
            self.assertNotIsInstance(parameter.val, brainstate.ParamState)

    def test_reset_produces_identical_voltage_and_event_traces(self):
        """Reset voltage, HH gates, probes, and time before every rollout."""
        first = core.simulate(self.problem.fitted_cell)
        second = core.simulate(self.problem.fitted_cell)
        np.testing.assert_array_equal(
            first.voltages.to_decimal(u.mV),
            second.voltages.to_decimal(u.mV),
        )
        np.testing.assert_array_equal(first.smooth_events, second.smooth_events)

    def test_all_initial_gradients_are_finite_and_nonzero(self):
        """Differentiate the composite loss through every conductance."""
        self.assertAlmostEqual(float(np.asarray(self.loss)), 1.0, places=5)
        np.testing.assert_allclose(np.asarray(self.components), np.ones(6), rtol=1e-5, atol=1e-5)
        for name in core.PARAMETER_NAMES:
            gradient = float(np.asarray(self.gradients[name]))
            self.assertTrue(np.isfinite(gradient))
            self.assertGreater(abs(gradient), 1e-6)


class ConductanceTrainingTest(unittest.TestCase):
    """Verify stage construction and a short compiled training run."""

    def test_stage_masks_release_one_two_then_three_parameters(self):
        """Open leak, then leak plus potassium, then all parameters."""
        masks = np.asarray(core.make_stage_masks(6, staged=True))
        np.testing.assert_array_equal(masks[:1], np.asarray([[1.0, 0.0, 0.0]]))
        np.testing.assert_array_equal(masks[1:2], np.asarray([[1.0, 0.0, 1.0]]))
        np.testing.assert_array_equal(masks[2:], np.ones((4, 3)))
        np.testing.assert_array_equal(core.make_stage_masks(2, staged=False), np.ones((2, 3)))

    def test_rejects_nonpositive_epoch_count(self):
        """Reject an experiment without optimizer updates."""
        with self.assertRaisesRegex(ValueError, "n_epochs must be positive"):
            core.make_stage_masks(0, staged=True)

    def test_short_staged_run_obeys_masks_and_stays_bounded(self):
        """Update only parameters released by each epoch's stage mask."""
        result = core.run_training(core.TrainingConfig(name="short", n_epochs=3, learning_rate=0.01, staged=True))
        trajectory = np.asarray(result.parameter_trajectory)
        self.assertEqual(trajectory.shape, (4, 3))
        self.assertNotEqual(trajectory[1, 0], trajectory[0, 0])
        self.assertEqual(trajectory[1, 1], trajectory[0, 1])
        self.assertEqual(trajectory[1, 2], trajectory[0, 2])
        self.assertEqual(trajectory[2, 1], trajectory[1, 1])
        self.assertNotEqual(trajectory[2, 2], trajectory[1, 2])
        self.assertNotEqual(trajectory[3, 1], trajectory[2, 1])
        self.assertTrue(np.all(np.isfinite(np.asarray(result.gradients))))
        self.assertTrue(np.all(np.asarray(result.fitted_parameters) > core.LOWER_BOUNDS))
        self.assertTrue(np.all(np.asarray(result.fitted_parameters) < core.UPPER_BOUNDS))


class ConductanceValidationTest(unittest.TestCase):
    """Verify local adapter and input validation."""

    def test_shared_adapter_rejects_empty_active_mask(self):
        """Reject a runtime layout with no painted conductance."""
        with self.assertRaisesRegex(ValueError, "at least one active point"):
            core.ExplorationTrainableLeak(size=2, g_max=jnp.zeros(2) * core.CONDUCTANCE_UNIT)

    def test_build_cell_rejects_wrong_parameter_shape(self):
        """Require one scalar for each of the three channels."""
        with self.assertRaisesRegex(ValueError, "shape .*3"):
            core.build_cell([0.2, 70.0], trainable=True)


if __name__ == "__main__":
    unittest.main()
