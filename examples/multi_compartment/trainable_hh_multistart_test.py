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

"""Smoke tests for the one-CV HH multistart training example."""

from contextlib import redirect_stdout
import io
import unittest

import brainunit as u
import numpy as np

from examples.experimental.optim_parameter_fitting.diagnostics import plot_diagnostics
from examples.multi_compartment.trainable_hh_multistart import (
    SCALE_BOUNDS,
    plot_result,
    run_experiment,
)


class TrainableHHMultistartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            cls.result = run_experiment(
                num_starts=3,
                num_epochs=2,
                duration=10.0 * u.ms,
                log_every=1,
            )
        cls.training_log = output.getvalue()

    def test_smoke_training_is_finite_bounded_and_improves(self) -> None:
        losses = np.asarray(self.result.loss_history)
        factors = np.asarray(self.result.factor_history)

        self.assertEqual(losses.shape, (3, 3))
        self.assertEqual(factors.shape, (3, 3, 3))
        self.assertGreaterEqual(self.result.target_spike_count, 1)
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(np.isfinite(factors).all())
        self.assertTrue((factors >= SCALE_BOUNDS[0]).all())
        self.assertTrue((factors <= SCALE_BOUNDS[1]).all())
        self.assertLess(losses[-1].mean(), losses[0].mean())

    def test_training_reports_requested_epochs(self) -> None:
        self.assertIn("epoch 1/2", self.training_log)
        self.assertIn("epoch 2/2", self.training_log)

    def test_diagnostics_align_parameter_states_with_optimizer_updates(self) -> None:
        diagnostics = self.result.diagnostics
        self.assertEqual(diagnostics.num_states, 3)
        self.assertEqual(diagnostics.num_updates, 2)
        self.assertEqual(diagnostics.num_starts, 3)
        self.assertEqual(diagnostics.gradient_norm.shape, (2, 3))
        self.assertEqual(diagnostics.optimizer_step_norm.shape, (2, 3))
        self.assertEqual(diagnostics.metrics["signed_count_error/step"].shape, (3, 3))
        self.assertEqual(len(self.result.diagnostic_summary["starts"]), 3)

    def test_best_archives_select_aligned_history_states(self) -> None:
        archives = self.result.archives
        losses = np.asarray(self.result.loss_history)
        expected_epochs = np.argmin(losses, axis=0)
        np.testing.assert_array_equal(archives.continuous.epoch, expected_epochs)
        np.testing.assert_allclose(
            archives.continuous.losses["total"],
            losses[expected_epochs, np.arange(losses.shape[1])],
        )
        feasible = np.asarray(archives.spike_feasible.valid)
        feasible_epochs = np.asarray(archives.spike_feasible.epoch)[feasible]
        starts = np.flatnonzero(feasible)
        signed_error = np.asarray(self.result.diagnostics.metrics["signed_count_error/step"])
        np.testing.assert_array_equal(signed_error[feasible_epochs, starts], 0)

    def test_plot_contains_voltage_loss_and_factor_axes(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plot_result(self.result)
        self.assertEqual(len(figure.axes), 3)
        diagnostic_figure = plot_diagnostics(self.result.diagnostics)
        self.assertGreaterEqual(len(diagnostic_figure.axes), 4)
        plt.close(figure)
        plt.close(diagnostic_figure)


if __name__ == "__main__":
    unittest.main()
