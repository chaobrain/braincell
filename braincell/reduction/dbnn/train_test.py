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
from unittest import mock
from pathlib import Path
import tempfile

import brainunit as u
import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dbnn import train
from braincell.reduction.dbnn.dataset import DatasetBatch, save_dataset
from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.reduction.dbnn.model import DBNN, DBNNGIF
from braincell.reduction.dbnn.stimulus import generate_multichannel_protocol
from braincell.mech import Synapse


def _layout(n_channels=1):
    indices = tuple(range(n_channels))
    return ChannelLayout((Synapse("ExpSyn", name="E"),), indices, indices, indices, (0,) * n_channels)


def _layout_with_reference_weight(reference_weight_us):
    return ChannelLayout(
        (Synapse("ExpSyn", name="E"),),
        (0,),
        (0,),
        (0,),
        (0,),
        (reference_weight_us,),
    )


class TrainBoundaryTest(unittest.TestCase):
    def test_spike_time_offset_uses_median_signed_validation_error(self):
        metrics = {
            "matches": (
                ((7.28, 1.0), (16.28, 10.0)),
                ((26.28, 20.0),),
            )
        }
        offset, count, timing_mae = train.spike_time_alignment(metrics)
        self.assertAlmostEqual(offset, 6.28)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(timing_mae, 6.28)

    def test_spike_time_offset_uses_zero_when_validation_has_no_matches(self):
        offset, count, timing_mae = train.spike_time_alignment({"matches": ((), ())})
        self.assertEqual(offset, 0.0)
        self.assertEqual(count, 0)
        self.assertIsNone(timing_mae)

    def test_spike_time_offset_uses_mean_of_middle_errors_for_even_matches(self):
        metrics = {"matches": (((7.0, 1.0), (10.0, 2.0)),)}
        offset, count, _ = train.spike_time_alignment(metrics)
        self.assertEqual(offset, 7.0)
        self.assertEqual(count, 2)

    def test_exposes_shared_training_functions(self):
        self.assertIn("dbnn_forward", train.__all__)
        self.assertIn("build_fit_mask", train.__all__)

    def test_build_fit_mask_excludes_crossing_window(self):
        targets = np.asarray([[-70.0, -60.0, -10.0, -5.0, -60.0]])
        actual = train.build_fit_mask(
            targets,
            1.0,
            spike_threshold_mv=-20.0,
            spike_window_pre_ms=1.0,
            spike_window_post_ms=1.0,
        )
        np.testing.assert_array_equal(actual, [[True, False, False, False, True]])

    def test_batches_preserve_selected_order(self):
        inputs = np.arange(3)[:, None, None]
        targets = np.arange(3)[:, None]
        masks = np.ones((3, 1), dtype=bool)
        result = list(train.batches(inputs, targets, masks, 2, indices=np.asarray([2, 0])))
        np.testing.assert_array_equal(result[0][0][:, 0, 0], [2, 0])

    def test_spike_metrics_include_f1_and_one_to_one_counts(self):
        actual = train.compute_spike_metrics([10.0, 20.0, 40.0], [11.0, 19.0, 80.0], window_ms=2.0)
        self.assertEqual((actual["tp"], actual["fp"], actual["fn"]), (2, 1, 1))
        self.assertAlmostEqual(actual["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(actual["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(actual["f1"], 2.0 / 3.0)

    def test_spike_matching_keeps_float32_values_on_closed_window_boundary(self):
        predicted = np.asarray([10.7], dtype=np.float64)
        target = np.asarray([0.7], dtype=np.float32)
        self.assertEqual(
            train.match_spike_times(predicted, target, window_ms=10.0),
            ((10.7, float(target[0])),),
        )

    def test_f1_is_undefined_when_both_spike_sets_are_empty(self):
        actual = train.compute_spike_metrics([], [])
        self.assertTrue(np.isnan(actual["precision"]))
        self.assertTrue(np.isnan(actual["recall"]))
        self.assertTrue(np.isnan(actual["f1"]))

    def test_train_step_updates_open_loop_parameters(self):
        model = DBNN(_layout(), dt=1.0 * u.ms)
        trainer = train.DBNNTrainer(model, learning_rate=0.01, loss="mse")
        inputs = jnp.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32)
        targets = jnp.asarray([[-70.0, -60.0, -60.0, -60.0]], dtype=jnp.float32)
        before = np.asarray(model.omega.value).copy()
        bias_before = np.asarray(model.bias.value).copy()
        threshold_before = np.asarray(model.v_th.value).copy()
        info = trainer.train_step(inputs, targets)
        self.assertTrue(np.isfinite(float(info["loss"])))
        self.assertFalse(np.array_equal(before, np.asarray(model.omega.value)))
        np.testing.assert_array_equal(model.bias.value, bias_before)
        np.testing.assert_array_equal(model.v_th.value, threshold_before)

    def test_trainer_rejects_already_calibrated_gif_model(self):
        with self.assertRaisesRegex(TypeError, "model must be a DBNN"):
            train.DBNNTrainer(DBNNGIF(_layout(), dt=1.0 * u.ms))

    def test_load_data_rejects_same_size_different_layout(self):
        source_layout = _layout_with_reference_weight(1.0)
        plan = generate_multichannel_protocol(
            source_layout,
            n_traces=1,
            duration_ms=2.0,
            dt_ms=1.0,
            rate_hz=0.0,
        )
        batch = DatasetBatch(
            plan,
            np.asarray([[-70.0, -70.0, -70.0]], dtype=np.float32),
            (np.asarray([], dtype=np.float32),),
            np.arange(3, dtype=np.float32),
            {"input_alignment": "target-step"},
        )
        trainer = train.DBNNTrainer(DBNN(_layout_with_reference_weight(2.0), dt=1.0 * u.ms))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            save_dataset(path, batch)
            with self.assertRaisesRegex(ValueError, "layout fingerprint"):
                trainer.load_data(path)

    def test_load_data_binds_and_enforces_source_fingerprint(self):
        channel_layout = _layout()
        plan = generate_multichannel_protocol(
            channel_layout,
            n_traces=1,
            duration_ms=2.0,
            dt_ms=1.0,
            rate_hz=0.0,
        )
        trainer = train.DBNNTrainer(DBNN(channel_layout, dt=1.0 * u.ms))
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.npz"
            second_path = Path(directory) / "second.npz"
            for path, source in ((first_path, "cell-a"), (second_path, "cell-b")):
                save_dataset(
                    path,
                    DatasetBatch(
                        plan,
                        np.full((1, 3), -70.0, dtype=np.float32),
                        (np.asarray([], dtype=np.float32),),
                        np.arange(3, dtype=np.float32),
                        {"input_alignment": "target-step", "source_fingerprint": source},
                    ),
                )
            trainer.load_data(first_path)
            self.assertEqual(trainer.source_fingerprint, "cell-a")
            with self.assertRaisesRegex(ValueError, "source fingerprint"):
                trainer.load_data(second_path)

    def test_split_data_handles_per_trace_spike_tuples(self):
        trainer = train.DBNNTrainer(DBNN(_layout(), dt=1.0 * u.ms), seed=3)
        data = {
            "inputs": np.zeros((10, 1, 2), dtype=np.float32),
            "targets": np.zeros((10, 2), dtype=np.float32),
            "spike_times_ms": tuple(np.asarray([trace], dtype=np.float32) for trace in range(10)),
            "metadata": {"input_alignment": "target-step"},
        }
        splits = trainer.split_data(data, train_fraction=0.6, validation_fraction=0.2)
        spike_splits = tuple(splits[name]["spike_times_ms"] for name in ("train", "validation", "test"))
        self.assertEqual(tuple(map(len, spike_splits)), (6, 2, 2))
        self.assertEqual(splits["train"]["metadata"], data["metadata"])

    def test_fit_lowers_validation_mse_on_synthetic_dbnn_data(self):
        teacher = DBNN(_layout(), dt=1.0 * u.ms)
        teacher_params = teacher.get_params()
        teacher_params["omega"] = jnp.asarray([5.0])
        teacher.set_params(teacher_params)
        inputs = jnp.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        data = {"inputs": inputs, "targets": teacher.predict(inputs)["voltage"]}
        trainer = train.DBNNTrainer(
            DBNN(_layout(), dt=1.0 * u.ms),
            learning_rate=0.05,
            loss="mse",
            initialization_search=False,
        )
        before = trainer.evaluate(data)["mse"]
        trainer.fit(data, validation_data=data, epochs=8, batch_size=4, shuffle=False)
        after = trainer.evaluate(data)["mse"]
        self.assertLess(after, before)

    def test_fit_does_not_calibrate_spike_parameters(self):
        model = DBNN(_layout(), dt=1.0 * u.ms)
        inputs = jnp.zeros((2, 1, 4), dtype=jnp.float32)
        data = {
            "inputs": inputs,
            "targets": model.predict(inputs)["voltage"],
            "spike_times_ms": (np.asarray([]), np.asarray([])),
        }
        trainer = train.DBNNTrainer(model, learning_rate=0.01, loss="mse", initialization_search=False)

        trainer.fit(data, validation_data=data, epochs=1, batch_size=2, shuffle=False)

        self.assertAlmostEqual(float(model.v_th.value), -55.0)

    def test_repeated_fit_restores_historical_best_params_and_optimizer(self):
        model = DBNN(_layout(), dt=1.0 * u.ms)
        inputs = jnp.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32)
        data = {"inputs": inputs, "targets": jnp.asarray([[-70.0, -60.0, -60.0, -60.0]])}
        trainer = train.DBNNTrainer(model, learning_rate=0.01, loss="mse", initialization_search=False)
        trainer.fit(data, validation_data=data, epochs=1, batch_size=1, shuffle=False)
        best_params = {name: np.asarray(value).copy() for name, value in model.get_params().items()}
        best_optimizer = {
            path: jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), state.value)
            for path, state in brainstate.graph.states(trainer.optimizer).items()
            if not isinstance(state, brainstate.ParamState)
        }
        trainer.best_validation_loss = -1.0

        trainer.fit(data, validation_data=data, epochs=1, batch_size=1, shuffle=False)

        for name, value in best_params.items():
            np.testing.assert_array_equal(model.get_params()[name], value)
        for path, value in best_optimizer.items():
            actual_leaves = jax.tree_util.tree_leaves(brainstate.graph.states(trainer.optimizer)[path].value)
            expected_leaves = jax.tree_util.tree_leaves(value)
            for actual, expected in zip(actual_leaves, expected_leaves):
                np.testing.assert_array_equal(actual, expected)

    def test_fit_runs_default_initialization_only_once(self):
        model = DBNN(_layout(), dt=1.0 * u.ms)
        trainer = train.DBNNTrainer(model, learning_rate=0.01, loss="mse")
        data = {
            "inputs": jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float32),
            "targets": jnp.asarray([[-70.0, -69.0, -69.0]], dtype=jnp.float32),
        }

        def initialize(_):
            trainer.initialization_completed = True
            trainer.initialization_report = {"backend": "test"}
            return trainer.initialization_report

        with mock.patch.object(trainer, "search_initialization", side_effect=initialize) as search:
            trainer.fit(data, epochs=1, batch_size=1, shuffle=False)
            trainer.fit(data, epochs=1, batch_size=1, shuffle=False)
        search.assert_called_once_with(data)

    def test_fit_uses_epoch_step_learning_rate_schedule(self):
        model = DBNN(_layout(), dt=1.0 * u.ms)
        trainer = train.DBNNTrainer(
            model,
            learning_rate=0.01,
            lr_step_size=2,
            lr_gamma=0.5,
            loss="mse",
            initialization_search=False,
        )
        data = {
            "inputs": jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float32),
            "targets": jnp.asarray([[-70.0, -69.0, -69.0]], dtype=jnp.float32),
        }

        history = trainer.fit(data, epochs=3, batch_size=1, shuffle=False)

        np.testing.assert_allclose(history["learning_rates"], [0.01, 0.01, 0.005])

    def test_scipy_search_initialization_fits_packed_ridge_weights(self):
        teacher = DBNN(_layout(2), dt=1.0 * u.ms)
        teacher_params = teacher.get_params()
        teacher_params.update(
            tau_rise=jnp.asarray([3.0, 3.0]),
            tau_decay=jnp.asarray([10.0, 10.0]),
            omega=jnp.asarray([2.5, 2.5]),
            quadratic_weight_upper=jnp.asarray([0.6]),
        )
        teacher.set_params(teacher_params)
        inputs = jnp.asarray(
            [
                [[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        data = {"inputs": inputs, "targets": teacher.predict(inputs)["voltage"]}
        model = DBNN(_layout(2), dt=1.0 * u.ms)
        trainer = train.DBNNTrainer(model, initialization_search=False, seed=2)

        report = trainer.search_initialization(
            data,
            backend="scipy",
            samples=4,
            time_stride=1,
            maxiter=1,
            popsize=2,
            max_pairs=0,
            ridge=1e-4,
        )

        self.assertTrue(trainer.initialization_completed)
        self.assertEqual(report["backend"], "scipy")
        self.assertLessEqual(report["calibration_mse"], report["default_calibration_mse"])
        self.assertGreater(abs(float(model.quadratic_weight_upper.value[0])), 1e-6)
        self.assertTrue(bool(jnp.all(model.omega.value >= 0.0)))

    def test_calibrate_gif_returns_new_model_without_mutating_dbnn(self):
        model = DBNN(_layout(), dt=1.0 * u.ms)
        params = model.get_params()
        params.update(
            tau_rise=jnp.asarray([1.0]),
            tau_decay=jnp.asarray([5.0]),
            omega=jnp.asarray([10.0]),
        )
        model.set_params(params)
        inputs = jnp.asarray([[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]], dtype=jnp.float32)
        truth = DBNNGIF.from_dbnn(model)
        truth_params = truth.get_params()
        truth_params.update(
            v_th=-69.0,
            threshold_increment_mv=2.0,
            threshold_tau_ms=10.0,
        )
        truth.set_params(truth_params)
        target_spikes = (np.flatnonzero(np.asarray(truth.predict(inputs)["spike"][0])).astype(float),)
        trainer = train.DBNNTrainer(model)
        original_params = {name: np.asarray(value).copy() for name, value in model.get_params().items()}

        gif, report = trainer.calibrate_gif(
            {"inputs": inputs, "spike_times_ms": target_spikes},
            thresholds_mv=[-69.0, -60.0],
            threshold_increments_mv=[2.0, 10.0],
            threshold_taus_ms=[10.0],
            candidate_batch_size=1,
        )

        self.assertIsInstance(gif, DBNNGIF)
        self.assertIsNot(gif, model)
        self.assertEqual(report["gif"]["candidate_count"], 4)
        self.assertAlmostEqual(float(gif.v_th.value), -69.0)
        self.assertAlmostEqual(float(gif.threshold_increment_mv.value), 2.0)
        self.assertAlmostEqual(report["gif"]["validation_metrics"]["f1"], 1.0)
        self.assertIn("spike_time_offset_ms", report["gif"])
        self.assertEqual(report["gif"]["matched_count"], len(target_spikes[0]))
        self.assertTrue(np.isfinite(gif.spike_time_offset_ms))
        np.testing.assert_array_equal(gif.predict(inputs)["spike"], truth.predict(inputs)["spike"])
        for name, value in original_params.items():
            np.testing.assert_array_equal(model.get_params()[name], value)

    def test_recurrent_gif_calibration_uses_recurrent_backend(self):
        model = DBNN(_layout(), mode="r", dt=1.0 * u.ms)
        trainer = train.DBNNTrainer(model)
        inputs = jnp.asarray([[[1.0, 0.0, 0.0, 1.0, 0.0]]], dtype=jnp.float32)
        with mock.patch.object(train, "dbnn_forward", side_effect=AssertionError("FFT path used")):
            gif, _ = trainer.calibrate_gif(
                {"inputs": inputs, "spike_times_ms": (np.asarray([1.0]),)},
                thresholds_mv=[-69.0],
                threshold_increments_mv=[0.0],
                threshold_taus_ms=[10.0],
                candidate_batch_size=1,
            )
        self.assertEqual(gif.mode, "r")

    def test_calibrate_gif_rejects_validation_data_without_spikes(self):
        trainer = train.DBNNTrainer(DBNN(_layout(), dt=1.0 * u.ms))
        with self.assertRaisesRegex(ValueError, "at least one validation spike"):
            trainer.calibrate_gif(
                {
                    "inputs": jnp.zeros((1, 1, 3), dtype=jnp.float32),
                    "spike_times_ms": (np.asarray([]),),
                }
            )

    def test_calibrate_gif_uses_zero_offset_when_best_candidate_has_no_matches(self):
        trainer = train.DBNNTrainer(DBNN(_layout(), dt=1.0 * u.ms))
        inputs = jnp.zeros((1, 1, 4), dtype=jnp.float32)
        raw_metrics = {
            "precision": float("nan"),
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "match_window_ms": 10.0,
            "matches": ((),),
        }
        with mock.patch.object(
            train,
            "_search_gif_candidates",
            return_value=(np.asarray([1000.0, 0.0, 10.0], dtype=np.float32), raw_metrics),
        ):
            gif, report = trainer.calibrate_gif(
                {"inputs": inputs, "spike_times_ms": (np.asarray([1.0]),)},
                thresholds_mv=[1000.0],
                threshold_increments_mv=[0.0],
                threshold_taus_ms=[10.0],
            )
        self.assertEqual(gif.spike_time_offset_ms, 0.0)
        self.assertEqual(report["gif"]["matched_count"], 0)
        self.assertIsNone(report["gif"]["raw_timing_mae_ms"])
