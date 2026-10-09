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
from pathlib import Path
import tempfile
import json

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dbnn import asset
from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.reduction.dbnn.model import DBNN, DBNNGIF, SpikeAlignmentEvidence
from braincell.reduction.dbnn.train import DBNNTrainer
from braincell.mech import Synapse


def _layout():
    specs = (
        Synapse("ExpSyn", name="E"),
        Synapse("ExpSyn", name="I", e=-80.0 * u.mV),
    )
    return ChannelLayout(specs, (0, 1), (0, 1), (0, 0), (0, 1))


def _single_layout(reference_weight_us):
    return ChannelLayout(
        (Synapse("ExpSyn", name="E"),),
        (0,),
        (0,),
        (0,),
        (0,),
        (reference_weight_us,),
    )


class AssetBoundaryTest(unittest.TestCase):
    def test_model_metadata_cannot_override_source_fingerprint(self):
        model = DBNN(_layout(), dt=1.0 * u.ms)
        model.source_fingerprint = "teacher-a"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must match"):
                asset.save_model(
                    Path(directory) / "model.npz",
                    model,
                    metadata={"source_fingerprint": "teacher-b"},
                )

    def test_model_loader_rejects_version_seven_asset(self):
        model = DBNNGIF(_layout(), dt=1.0 * u.ms)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            asset.save_model(path, model, metadata={"source_fingerprint": "test"})
            with np.load(path, allow_pickle=False) as saved:
                values = {name: np.array(saved[name], copy=True) for name in saved.files}
            manifest = json.loads(str(values["manifest_json"]))
            manifest["format_version"] = 7
            values["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                asset.load_model(path)

    def test_model_loader_accepts_legacy_version_eight_without_alignment_data(self):
        model = DBNNGIF(_layout(), dt=1.0 * u.ms)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            asset.save_model(path, model, metadata={"source_fingerprint": "test"})
            with np.load(path, allow_pickle=False) as saved:
                values = {name: np.array(saved[name], copy=True) for name in saved.files}
            manifest = json.loads(str(values["manifest_json"]))
            manifest["format_version"] = 8
            manifest.pop("spike_alignment_evidence_path", None)
            values["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
            np.savez_compressed(path, **values)
            restored = asset.load_model(path)

        self.assertIsInstance(restored, DBNNGIF)
        self.assertIsNone(restored.spike_alignment_evidence)

    def test_module_is_importable(self):
        self.assertIn("asset", asset.__doc__)

    def test_host_params_returns_numpy_copies(self):
        actual = asset.host_params({"omega": jnp.asarray([1.0, 2.0])})
        self.assertIsInstance(actual["omega"], np.ndarray)
        np.testing.assert_array_equal(actual["omega"], [1.0, 2.0])

    def test_model_asset_round_trip(self):
        model = DBNN(_layout(), mode="r", input_sign_mode="channel_type", dt=0.2 * u.ms)
        inputs = jnp.ones((1, 2, 4), dtype=jnp.float32)
        expected = model.predict(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            asset.save_model(path, model, metadata={"source_fingerprint": "test"})
            restored = asset.load_model(path)
        self.assertEqual(restored.mode, "r")
        self.assertEqual(restored.input_sign_mode, "channel_type")
        self.assertEqual(restored.asset_metadata, {"source_fingerprint": "test"})
        self.assertEqual(restored.layout, model.layout)
        np.testing.assert_allclose(restored.predict(inputs)["voltage"], expected["voltage"])

    def test_model_and_alignment_paths_accept_missing_npz_suffix(self):
        model = DBNNGIF(_layout(), dt=1.0 * u.ms)
        model.spike_alignment_evidence = SpikeAlignmentEvidence(
            teacher_voltage_mv=np.asarray([[-70.0, -60.0]], dtype=np.float32),
            raw_spike_times_ms=(np.asarray([], dtype=np.float32),),
            dt_ms=1.0,
            match_window_ms=10.0,
            layout_fingerprint=model.layout.fingerprint,
            dynamics_fingerprint="dynamics",
            validation_seeds=(1,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            asset.save_model(path, model, metadata={"source_fingerprint": "test"})
            self.assertTrue((Path(directory) / "model.npz").is_file())
            self.assertTrue((Path(directory) / "model_spike_alignment.npz").is_file())
            restored = asset.load_model(path)
        self.assertIsNotNone(restored.spike_alignment_evidence)

    def test_training_checkpoint_restores_progress(self):
        model = DBNN(_single_layout(1.0), dt=1.0 * u.ms)
        trainer = DBNNTrainer(model)
        trainer.epoch = 3
        trainer.best_validation_loss = 2.5
        trainer.initialization_completed = True
        trainer.initialization_report = {"backend": "scipy", "calibration_mse": 1.5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            asset.save_training_checkpoint(path, trainer)
            trainer.epoch = 0
            trainer.best_validation_loss = float("inf")
            asset.load_training_checkpoint(path, trainer)
        self.assertEqual(trainer.epoch, 3)
        self.assertEqual(trainer.best_validation_loss, 2.5)
        self.assertTrue(trainer.initialization_completed)
        self.assertEqual(trainer.initialization_report["backend"], "scipy")

    def test_training_checkpoint_round_trips_historical_best_snapshot(self):
        trainer = DBNNTrainer(
            DBNN(_single_layout(1.0), dt=1.0 * u.ms),
            loss="mse",
            initialization_search=False,
        )
        inputs = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float32)
        data = {"inputs": inputs, "targets": jnp.asarray([[-70.0, -60.0, -60.0]])}
        trainer.fit(data, validation_data=data, epochs=1, batch_size=1, shuffle=False)
        expected = {name: np.asarray(value).copy() for name, value in trainer._best_params.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_checkpoint.npz"
            asset.save_training_checkpoint(path, trainer)
            restored = DBNNTrainer(
                DBNN(_single_layout(1.0), dt=1.0 * u.ms),
                loss="mse",
                initialization_search=False,
            )
            asset.load_training_checkpoint(path, restored)
        self.assertIsNotNone(restored._best_params)
        for name, value in expected.items():
            np.testing.assert_array_equal(restored._best_params[name], value)

    def test_training_checkpoint_rejects_same_size_different_layout(self):
        source = DBNNTrainer(DBNN(_single_layout(1.0), dt=1.0 * u.ms))
        target = DBNNTrainer(DBNN(_single_layout(2.0), dt=1.0 * u.ms))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            asset.save_training_checkpoint(path, source)
            with self.assertRaisesRegex(ValueError, "channel layout"):
                asset.load_training_checkpoint(path, target)

    def test_training_checkpoint_rejects_incompatible_trainer_config(self):
        source = DBNNTrainer(DBNN(_single_layout(1.0), dt=1.0 * u.ms), loss="mse")
        target = DBNNTrainer(DBNN(_single_layout(1.0), dt=1.0 * u.ms), loss="masked_mse")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            asset.save_training_checkpoint(path, source)
            with self.assertRaisesRegex(ValueError, "loss"):
                asset.load_training_checkpoint(path, target)

    def test_gif_model_asset_round_trip_preserves_model_type(self):
        model = DBNNGIF(_layout(), mode="r", dt=0.2 * u.ms)
        model.threshold_increment_mv.value = jnp.asarray(7.0)
        model.set_spike_time_offset(6.28)
        inputs = jnp.ones((1, 2, 4), dtype=jnp.float32)
        expected = model.predict(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gif_model.npz"
            asset.save_model(path, model, metadata={"source_fingerprint": "test"})
            restored = asset.load_model(path)
        self.assertIsInstance(restored, DBNNGIF)
        self.assertAlmostEqual(float(restored.threshold_increment_mv.value), 7.0)
        self.assertAlmostEqual(restored.spike_time_offset_ms, 6.28)
        np.testing.assert_allclose(restored.predict(inputs)["voltage"], expected["voltage"])

    def test_gif_model_asset_round_trip_uses_separate_alignment_asset(self):
        model = DBNNGIF(_layout(), dt=1.0 * u.ms)
        model.spike_alignment_evidence = SpikeAlignmentEvidence(
            teacher_voltage_mv=np.asarray([[-70.0, -60.0, -50.0]], dtype=np.float32),
            raw_spike_times_ms=(np.asarray([2.0]),),
            dt_ms=1.0,
            match_window_ms=10.0,
            layout_fingerprint=model.layout.fingerprint,
            dynamics_fingerprint="dynamics",
            validation_seeds=(3,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gif_model.npz"
            asset.save_model(path, model, metadata={"source_fingerprint": "test"})
            with np.load(path, allow_pickle=False) as saved:
                manifest = json.loads(str(saved["manifest_json"]))
                self.assertNotIn("teacher_voltage_mv", saved.files)
            evidence_path = path.parent / manifest["spike_alignment_evidence_path"]
            self.assertTrue(evidence_path.is_file())
            restored = asset.load_model(path)

        actual = restored.spike_alignment_evidence
        np.testing.assert_array_equal(actual.teacher_voltage_mv, model.spike_alignment_evidence.teacher_voltage_mv)
        np.testing.assert_array_equal(actual.raw_spike_times_ms[0], [2.0])
        self.assertEqual(actual.validation_seeds, (3,))

    def test_gif_model_asset_round_trip_preserves_optional_reset(self):
        model = DBNNGIF(_layout(), dt=0.2 * u.ms).enable_reset(amplitude_mv=4.0, tau_ms=5.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gif_reset_model.npz"
            asset.save_model(path, model, metadata={"source_fingerprint": "test"})
            restored = asset.load_model(path)
        self.assertTrue(restored.reset_enabled)
        self.assertAlmostEqual(float(restored.reset_amp.value), 4.0)
        self.assertAlmostEqual(float(restored.tau_reset.value), 5.0)
