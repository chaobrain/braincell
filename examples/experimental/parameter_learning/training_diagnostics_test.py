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

"""Tests for the experiment-local parameter-learning diagnostics."""

import json
from pathlib import Path
import tempfile
import unittest

import braincell
import brainunit as u
import jax.numpy as jnp
import numpy as np
from braincell.filter import AllRegion, at

from examples.experimental.parameter_learning.training_diagnostics import (
    DiagnosticConfig,
    StateSignals,
    UpdateSignals,
    evaluate_voltage_protocols,
    extract_best_archives,
    finalize_history,
    save_artifacts,
    summarize_history,
    voltage_mse_objective,
)


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_best_archives_separate_loss_best_from_spike_feasible_best(self) -> None:
        states = StateSignals(
            optimizer_values={"theta": jnp.array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]])},
            physical_values={"theta": jnp.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])},
            losses={"total": jnp.array([[5.0, 4.0, 6.0], [3.0, 3.0, 5.0]])},
            metrics={
                "signed_count_error/weak": jnp.array([[0, 1, -1], [0, 0, -1]]),
                "signed_count_error/strong": jnp.array([[0, 1, 1], [0, 1, 1]]),
                "finite/weak": jnp.ones((2, 3), dtype=bool),
                "finite/strong": jnp.ones((2, 3), dtype=bool),
            },
        )
        endpoint = StateSignals(
            optimizer_values={"theta": jnp.array([0.2, 1.2, 2.2])},
            physical_values={"theta": jnp.array([12.0, 22.0, 32.0])},
            losses={"total": jnp.array([1.0, 2.0, 4.0])},
            metrics={
                "signed_count_error/weak": jnp.array([1, 0, -1]),
                "signed_count_error/strong": jnp.array([1, 0, 1]),
                "finite/weak": jnp.ones((3,), dtype=bool),
                "finite/strong": jnp.ones((3,), dtype=bool),
            },
        )
        updates = UpdateSignals(
            gradients={"theta": jnp.ones((2, 3))},
            learning_rate=jnp.full((2,), 0.01),
        )
        history = finalize_history(states, endpoint, updates)

        archives = extract_best_archives(history)

        np.testing.assert_array_equal(archives.continuous.valid, [True, True, True])
        np.testing.assert_array_equal(archives.continuous.epoch, [2, 2, 2])
        np.testing.assert_allclose(archives.continuous.losses["total"], [1.0, 2.0, 4.0])
        np.testing.assert_array_equal(archives.spike_feasible.valid, [True, True, False])
        np.testing.assert_array_equal(archives.spike_feasible.epoch, [1, 2, -1])
        np.testing.assert_allclose(archives.spike_feasible.physical_values["theta"][:2], [11.0, 22.0])
        self.assertTrue(np.isnan(archives.spike_feasible.physical_values["theta"][2]))

    def test_soma_dendrite_protocol_suite_uses_the_same_contract(self) -> None:
        targets = {
            "subthreshold": _run_dendritic_protocol(0.02 * u.nA),
            "spiking": _run_dendritic_protocol(0.05 * u.nA),
        }
        predictions = {name: jnp.stack((target, target + 0.5), axis=1) for name, target in targets.items()}

        total, components = voltage_mse_objective(predictions, targets)
        metrics = evaluate_voltage_protocols(predictions, targets, spike_probe=0)

        self.assertEqual(set(components), {"voltage_mse/subthreshold", "voltage_mse/spiking"})
        np.testing.assert_allclose(total, [0.0, 0.25], rtol=1e-5)
        self.assertEqual(metrics["voltage_rmse/spiking"].shape, (2, 2))
        self.assertEqual(int(metrics["spike_count/subthreshold"][0]), 0)
        self.assertEqual(int(metrics["spike_count/spiking"][0]), 1)

    def test_named_protocol_objective_and_metrics_support_multiple_probes(self) -> None:
        targets = {
            "weak": jnp.array([[-1.0, -2.0], [1.0, -1.0], [-1.0, 1.0]]),
            "strong": jnp.array([[-1.0, -1.0], [1.0, 1.0]]),
        }
        predictions = {name: jnp.stack((target, target + 1.0), axis=1) for name, target in targets.items()}

        total, components = voltage_mse_objective(predictions, targets)
        metrics = evaluate_voltage_protocols(predictions, targets, spike_probe={"weak": 0, "strong": 1})

        np.testing.assert_allclose(total, [0.0, 1.0])
        self.assertEqual(set(components), {"voltage_mse/weak", "voltage_mse/strong"})
        self.assertEqual(metrics["voltage_rmse/weak"].shape, (2, 2))
        np.testing.assert_array_equal(metrics["signed_count_error/strong"], [0, -1])

    def test_history_keeps_state_and_update_axes_separate(self) -> None:
        states = StateSignals(
            optimizer_values={"theta": jnp.array([[0.0, 1.0], [0.5, 0.5]])},
            physical_values={"theta": jnp.array([[0.5, 1.5], [0.8, 1.2]])},
            losses={"total": jnp.array([[4.0, 3.0], [2.0, 2.5]])},
            metrics={
                "signed_count_error/step": jnp.array([[-1, 1], [0, 1]]),
                "finite/step": jnp.ones((2, 2), dtype=bool),
            },
        )
        endpoint = StateSignals(
            optimizer_values={"theta": jnp.array([1.0, 0.0])},
            physical_values={"theta": jnp.array([1.0, 1.0])},
            losses={"total": jnp.array([1.0, 2.0])},
            metrics={
                "signed_count_error/step": jnp.array([0, 1]),
                "finite/step": jnp.ones((2,), dtype=bool),
            },
        )
        updates = UpdateSignals(
            gradients={"theta": jnp.array([[2.0, -1.0], [1.0, -1.0]])},
            learning_rate=jnp.array([0.1, 0.1]),
        )

        history = finalize_history(states, endpoint, updates, bounds={"theta": (0.0, 2.0)})

        self.assertEqual(history.num_states, 3)
        self.assertEqual(history.num_updates, 2)
        self.assertEqual(history.num_starts, 2)
        np.testing.assert_allclose(history.gradient_norm, [[2.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(history.gradient_cosine[1], [1.0, 1.0])
        np.testing.assert_allclose(history.bound_positions["theta"][-1], [0.5, 0.5])
        self.assertEqual(history.optimizer_step_norm.shape, (2, 2))
        self.assertEqual(history.physical_step_norm.shape, (2, 2))

        summary = summarize_history(
            history,
            target_parameters={"theta": 1.0},
            config=DiagnosticConfig(window=1),
        )
        self.assertEqual(summary["starts"][0]["region_state"], "feasible")
        self.assertEqual(summary["starts"][1]["region_state"], "extra")
        self.assertEqual(summary["starts"][0]["parameter_relative_error"], 0.0)
        self.assertEqual(summary["counts"]["degraded_from_best"], 0)
        self.assertEqual(summary["config"]["window"], 1)

    def test_artifacts_are_split_between_arrays_metadata_and_summary(self) -> None:
        states = StateSignals(
            optimizer_values={"theta": jnp.array([[0.0]])},
            physical_values={"theta": jnp.array([[1.0]])},
            losses={"total": jnp.array([[1.0]])},
            metrics={"finite/step": jnp.array([[True]])},
        )
        endpoint = StateSignals(
            optimizer_values={"theta": jnp.array([0.1])},
            physical_values={"theta": jnp.array([1.1])},
            losses={"total": jnp.array([0.5])},
            metrics={"finite/step": jnp.array([True])},
        )
        updates = UpdateSignals(
            gradients={"theta": jnp.array([[0.5]])},
            learning_rate=jnp.array([0.01]),
        )
        history = finalize_history(states, endpoint, updates)
        archives = extract_best_archives(history)
        summary = summarize_history(history, archives=archives)

        with tempfile.TemporaryDirectory() as directory:
            output = save_artifacts(directory, history, summary, metadata={"seed": 7}, archives=archives)
            files = {path.name for path in Path(output).iterdir()}
            self.assertEqual(files, {"history.npz", "metadata.json", "summary.json"})
            with (Path(output) / "metadata.json").open(encoding="utf-8") as file:
                self.assertEqual(json.load(file)["seed"], 7)
            with np.load(Path(output) / "history.npz") as arrays:
                self.assertIn("loss/total", arrays)
                self.assertIn("derived/gradient_norm", arrays)
                self.assertIn("archive/continuous/epoch", arrays)
                self.assertIn("archive/spike_feasible/valid", arrays)


def _run_dendritic_protocol(amplitude):
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dendrite = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dend = dendrite
    cell = braincell.Cell(
        morphology,
        cv_policy=braincell.CVPerBranch(),
        pop_size=(1,),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2),
        braincell.mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        braincell.mech.Channel("K_HH1952", name="k", g_max=10.0 * u.mS / u.cm**2),
    )
    cell.place(
        at("soma", 0.5),
        braincell.mech.CurrentClamp(
            delay=1.0 * u.ms,
            durations=8.0 * u.ms,
            amplitudes=amplitude,
        ),
    )
    cell.soma.record("soma_v", braincell.observe.state("v"))
    cell.basal_dendrite.record("dendrite_v", braincell.observe.state("v"))
    cell.init_state()
    result = cell.run(dt=0.025 * u.ms, duration=10.0 * u.ms)
    soma_mv = result.samples["soma_v"].values.to_decimal(u.mV)[:, 0]
    dendrite_mv = result.samples["dendrite_v"].values.to_decimal(u.mV)[:, 0]
    return jnp.stack((soma_mv, dendrite_mv), axis=1)


if __name__ == "__main__":
    unittest.main()
