"""Tests for heterogeneous nine-parameter minibatch training."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import brainstate
import braintools
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import heterogeneous_nine_parameter_training as training
import heterogeneous_protocol_dataset as dataset


class InitialGridTest(unittest.TestCase):
    def test_grid_scales_channel_families_across_regions(self) -> None:
        starts = training.make_initial_grid()
        self.assertEqual(starts.shape, (8, 9))
        ratios = starts / training.TARGET_PARAMETERS
        self.assertEqual(set(np.unique(ratios).tolist()), {0.5, 1.5})
        np.testing.assert_allclose(ratios[:, 0], ratios[:, 3])
        np.testing.assert_allclose(ratios[:, 0], ratios[:, 6])
        np.testing.assert_allclose(ratios[:, 1], ratios[:, 4])
        np.testing.assert_allclose(ratios[:, 1], ratios[:, 7])
        np.testing.assert_allclose(ratios[:, 2], ratios[:, 5])
        np.testing.assert_allclose(ratios[:, 2], ratios[:, 8])

    def test_grid_is_strictly_inside_parameter_bounds(self) -> None:
        starts = training.make_initial_grid()
        self.assertTrue(np.all(starts > training.LOWER_BOUNDS))
        self.assertTrue(np.all(starts < training.UPPER_BOUNDS))

    def test_relative_rms_has_percentage_interpretation(self) -> None:
        starts = training.make_initial_grid()
        distances = training.relative_parameter_rms(starts, training.TARGET_PARAMETERS)
        np.testing.assert_allclose(distances, 0.5)
        self.assertEqual(
            training.relative_parameter_rms(training.TARGET_PARAMETERS, training.TARGET_PARAMETERS),
            0.0,
        )

    def test_relative_rms_rejects_zero_target(self) -> None:
        with self.assertRaises(ValueError):
            training.relative_parameter_rms(np.ones(2), np.asarray([1.0, 0.0]))

    def test_relative_parameter_steps_use_target_scaling(self) -> None:
        target = np.asarray([2.0, 4.0])
        trajectories = np.asarray(
            [
                [[2.0, 4.0], [4.0, 4.0], [4.0, 8.0]],
                [[1.0, 2.0], [1.0, 2.0], [2.0, 4.0]],
            ]
        )
        steps = training.relative_parameter_step_rms(trajectories, target)
        self.assertEqual(steps.shape, (2, 2))
        np.testing.assert_allclose(steps[0], np.sqrt(0.5))
        np.testing.assert_allclose(steps[1], [0.0, 0.5])

    def test_signed_relative_error_preserves_direction(self) -> None:
        values = np.asarray([[1.5, 3.0], [0.5, 5.0]])
        target = np.asarray([1.0, 4.0])
        np.testing.assert_allclose(
            training.signed_relative_parameter_error(values, target),
            [[0.5, -0.25], [-0.5, 0.25]],
        )


class BatchScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = training.load_dataset()

    def test_schedule_is_deterministic_and_covers_train_split(self) -> None:
        first = training.make_batch_schedule(self.data, 2, seed=13)
        second = training.make_batch_schedule(self.data, 2, seed=13)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 6, 18))
        expected = np.sort(self.data.indices("train"))
        for epoch in first:
            np.testing.assert_array_equal(np.sort(epoch.ravel()), expected)

    def test_every_batch_has_exact_site_family_composition(self) -> None:
        schedule = training.make_batch_schedule(self.data, 1)[0]
        expected = {"dc": 3, "paired": 2, "sine": 1}
        for batch in schedule:
            for site in dataset.SITES:
                local = batch[self.data.sites[batch] == site]
                self.assertEqual(local.size, 6)
                self.assertEqual(
                    {family: int(np.sum(self.data.families[local] == family)) for family in dataset.FAMILIES},
                    expected,
                )


class SpikeMaskTest(unittest.TestCase):
    def test_zero_spike_trace_keeps_unit_weights(self) -> None:
        voltage = np.full((1, 20, 3), -65.0)
        np.testing.assert_array_equal(training.target_weight_masks(voltage), np.ones((1, 20)))

    def test_boundary_spike_window_is_clipped(self) -> None:
        n_steps = int(round(5.0 / dataset.DT_MS))
        voltage = np.full((1, n_steps, 3), -1.0)
        voltage[0, 1:, 0] = 1.0
        mask = training.target_weight_masks(voltage)[0]
        self.assertEqual(mask[0], training.SPIKE_WINDOW_WEIGHT)
        self.assertEqual(mask[-1], 1.0)


class HardSpikeQualityTest(unittest.TestCase):
    def test_separates_count_and_timing_failures(self) -> None:
        target = np.full((2, 100, 3), -1.0)
        target[1, (20, 60), 0] = 1.0
        prediction = np.broadcast_to(target, (3,) + target.shape).copy()
        prediction[1, 1, 60, 0] = -1.0
        prediction[2, 1, 20, 0] = -1.0
        prediction[2, 1, 45, 0] = 1.0
        count_error, count_exact, timing_exact = training.hard_spike_protocol_quality(prediction, target)
        np.testing.assert_array_equal(count_error[:, 0], 0)
        np.testing.assert_array_equal(count_exact[:, 0], True)
        np.testing.assert_array_equal(timing_exact[:, 0], True)
        self.assertEqual(count_error[1, 1], -1)
        self.assertFalse(count_exact[1, 1])
        self.assertFalse(timing_exact[1, 1])
        self.assertTrue(count_exact[2, 1])
        self.assertFalse(timing_exact[2, 1])

    def test_rejects_negative_timing_tolerance(self) -> None:
        voltage = np.full((1, 20, 3), -1.0)
        with self.assertRaises(ValueError):
            training.hard_spike_protocol_quality(voltage, voltage, tolerance_ms=-0.1)

    def test_jax_metric_matches_numpy_metric(self) -> None:
        target = np.full((2, 100, 3), -1.0)
        target[1, (20, 60), 0] = 1.0
        prediction = target.copy()
        prediction[1, 20, 0] = -1.0
        prediction[1, 45, 0] = 1.0
        _, expected_count, expected_timing = training.hard_spike_protocol_quality(prediction, target)
        actual_count, actual_timing = training._hard_spike_quality_jax(
            jnp.asarray(prediction),
            jnp.asarray(target),
            max_spikes=2,
        )
        np.testing.assert_array_equal(actual_count, expected_count)
        np.testing.assert_array_equal(actual_timing, expected_timing)

    def test_max_timing_error_handles_silent_and_count_mismatch_protocols(self) -> None:
        target = np.full((3, 100, 3), -1.0)
        target[1, (20, 60), 0] = 1.0
        target[2, 30, 0] = 1.0
        prediction = target.copy()
        prediction[1, (20, 60), 0] = -1.0
        prediction[1, (24, 58), 0] = 1.0
        prediction[2, 30, 0] = -1.0
        error = training.max_ordered_spike_timing_error_ms(prediction, target)
        self.assertEqual(error.shape, (3,))
        self.assertEqual(error[0], 0.0)
        self.assertAlmostEqual(error[1], 4 * dataset.DT_MS)
        self.assertTrue(np.isnan(error[2]))


class EndpointMetricTest(unittest.TestCase):
    def test_protocol_voltage_rmse_averages_time_and_all_probes(self) -> None:
        target = np.zeros((2, 4, 3))
        prediction = np.zeros((2, 2, 4, 3))
        prediction[0, 0] = 1.0
        prediction[0, 1, :, 0] = 3.0
        rmse = training.protocol_voltage_rmse(prediction, target)
        self.assertEqual(rmse.shape, (2, 2))
        self.assertEqual(rmse[0, 0], 1.0)
        self.assertAlmostEqual(rmse[0, 1], np.sqrt(3.0))
        np.testing.assert_array_equal(rmse[1], 0.0)


class PlaybackTest(unittest.TestCase):
    def test_one_batch_huber_loss_backpropagates_and_updates_all_parameters(self) -> None:
        data = training.load_dataset()
        indices = training.make_batch_schedule(data, 1, training.RANDOM_SEED)[0, 0]
        masks = training.target_weight_masks(data.voltages_mv)
        problem = training.PlaybackProblem()
        problem.set_physical_parameters(training.make_initial_grid()[0])
        problem.set_batch(
            data,
            jnp.asarray(indices),
            np.ones(data.currents_na.shape[0]),
            masks,
        )
        optimizer = braintools.optim.Adam(lr=0.01, grad_clip_norm=1.0)
        optimizer.register_trainable_weights(problem.parameter_states)
        grad_fn = brainstate.transform.grad(
            problem.loss_with_aux,
            grad_states=problem.parameter_states,
            has_aux=True,
            return_value=True,
        )

        with brainstate.environ.context(dt=dataset.DT):
            before = np.asarray(problem.physical_parameters())
            gradients, loss, raw_losses = grad_fn()
            optimizer.update(gradients)
            after = np.asarray(problem.physical_parameters())

        gradient_vector = np.asarray([gradients[name] for name in training.PARAMETER_NAMES], dtype=float)
        self.assertEqual(raw_losses.shape, (training.BATCH_SIZE,))
        self.assertTrue(np.isfinite(float(np.asarray(loss))))
        self.assertTrue(np.all(np.isfinite(gradient_vector)))
        self.assertGreater(float(np.linalg.norm(gradient_vector)), 0.0)
        self.assertEqual(np.count_nonzero(after != before), len(training.PARAMETER_NAMES))

    def test_target_parameters_replay_saved_currents(self) -> None:
        data = training.load_dataset()
        problem = training.PlaybackProblem()
        indices = data.indices("validation")
        masks = training.target_weight_masks(data.voltages_mv)
        problem.set_physical_parameters(training.TARGET_PARAMETERS)
        problem.set_batch(data, indices, np.ones(data.currents_na.shape[0]), masks)
        with brainstate.environ.context(dt=dataset.DT):
            prediction = np.asarray(problem.simulate())
        error = prediction - data.voltages_mv[indices]
        self.assertLess(float(np.sqrt(np.mean(error**2))), 0.02)
        self.assertLess(float(np.max(np.abs(error))), 0.1)


class TracePlotTest(unittest.TestCase):
    def test_trace_atlas_has_no_initial_dotted_series(self) -> None:
        data = training.load_dataset()
        result = training.load_saved_training_result()
        figure = training._plot_trace_atlas(data, result, 0)
        try:
            self.assertEqual(len(figure.axes), 18)
            for axis in figure.axes:
                self.assertEqual(len(axis.lines), 6)
                self.assertEqual({line.get_linestyle() for line in axis.lines}, {"-", "--"})
        finally:
            plt.close(figure)


class ErrorBreakdownPlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = training.load_dataset()
        cls.result = training.load_saved_training_result()
        n_checkpoint = cls.result.parameter_trajectories.shape[1]
        n_test = cls.data.indices("test").size
        parameter_distance = training.relative_parameter_rms(
            cls.result.parameter_trajectories,
            cls.result.target_parameters,
        )
        count_error, count_exact, timing_exact = training.hard_spike_protocol_quality(
            cls.result.fitted_test_traces,
            cls.result.target_test_traces,
        )
        cls.quality = training.QualityDiagnostics(
            parameter_relative_rms=parameter_distance,
            relative_parameter_step_rms=training.relative_parameter_step_rms(
                cls.result.parameter_trajectories,
                cls.result.target_parameters,
            ),
            best_parameter_relative_rms=training.relative_parameter_rms(
                cls.result.best_parameters,
                cls.result.target_parameters,
            ),
            best_parameter_signed_relative_error=training.signed_relative_parameter_error(
                cls.result.best_parameters,
                cls.result.target_parameters,
            ),
            validation_voltage_rmse_mv=np.ones((8, n_checkpoint)),
            validation_spike_count_exact=np.ones((8, n_checkpoint, 18), dtype=bool),
            validation_spike_timing_exact=np.ones((8, n_checkpoint, 18), dtype=bool),
            test_voltage_rmse_mv=np.ones(8),
            test_protocol_voltage_rmse_mv=training.protocol_voltage_rmse(
                cls.result.fitted_test_traces,
                cls.result.target_test_traces,
            ),
            test_spike_count_error=count_error,
            test_spike_count_exact=count_exact,
            test_spike_timing_exact=timing_exact,
            test_spike_max_timing_error_ms=training.max_ordered_spike_timing_error_ms(
                cls.result.fitted_test_traces,
                cls.result.target_test_traces,
            ),
        )
        assert n_test == 18

    def test_parameter_distance_figure_has_distance_and_step_panels(self) -> None:
        figure = training._plot_parameter_distance(self.result, self.quality)
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(figure.axes[1].get_yscale(), "symlog")
        finally:
            plt.close(figure)

    def test_quality_archive_schema_contains_endpoint_breakdowns(self) -> None:
        arrays = training._quality_array_values(self.quality)
        expected_shapes = {
            "relative_parameter_step_rms": (8, 30),
            "test_protocol_voltage_rmse_mv": (8, 18),
            "best_parameter_signed_relative_error": (8, 9),
            "test_spike_max_timing_error_ms": (8, 18),
        }
        for name, shape in expected_shapes.items():
            self.assertIn(name, arrays)
            self.assertEqual(arrays[name].shape, shape)

    def test_endpoint_breakdown_figures_construct(self) -> None:
        figures = (
            training._plot_test_protocol_voltage_rmse(self.data, self.quality),
            training._plot_best_parameter_signed_error(self.quality),
            training._plot_test_spike_timing_error(self.data, self.quality),
            training._plot_endpoint_pareto(self.quality),
        )
        try:
            self.assertEqual(len(figures[0].axes), 2)
            self.assertEqual(len(figures[1].axes), 2)
            self.assertEqual(len(figures[2].axes), 2)
            self.assertEqual(len(figures[3].axes), 2)
            self.assertEqual(len(figures[3].axes[0].texts), 8)
        finally:
            for figure in figures:
                plt.close(figure)

    def test_pareto_labels_do_not_overlap_for_clustered_endpoints(self) -> None:
        clustered = replace(
            self.quality,
            best_parameter_relative_rms=np.linspace(0.20, 0.207, 8),
            test_voltage_rmse_mv=np.linspace(10.0, 10.07, 8),
        )
        figure = training._plot_endpoint_pareto(clustered)
        try:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            boxes = [label.get_window_extent(renderer) for label in figure.axes[0].texts]
            overlaps = [first.overlaps(second) for index, first in enumerate(boxes) for second in boxes[index + 1 :]]
            self.assertFalse(any(overlaps))
        finally:
            plt.close(figure)


class LandscapePlotTest(unittest.TestCase):
    def test_landscape_omits_spike_count_boundary(self) -> None:
        result = training.load_saved_training_result()
        landscape = training.LandscapeResult(
            axis_1=np.eye(9)[0],
            axis_2=np.eye(9)[1],
            x_values=np.asarray([-1.0, 1.0]),
            y_values=np.asarray([-1.0, 1.0]),
            plane_losses=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            plane_spike_mismatch=np.asarray([[0, 1], [1, 0]]),
            projected_trajectories=np.zeros((8, 2, 2)),
            best_projection=np.asarray([0.5, 0.5]),
            profile_parameters=np.zeros((2, 9, 1, 9)),
            profile_losses=np.zeros((2, 9, 1)),
            profile_spike_mismatch=np.zeros((2, 9, 1)),
        )
        with patch("matplotlib.axes.Axes.contour") as line_contour:
            figure = training._plot_landscape(result, landscape)
        try:
            line_contour.assert_not_called()
            self.assertEqual(len(figure.axes[0].lines), 8)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
