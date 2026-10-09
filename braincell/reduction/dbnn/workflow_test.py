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
import json
from pathlib import Path
import tempfile

import brainunit as u
import numpy as np

from braincell.mech import Synapse
from braincell.reduction.dbnn.dataset import DatasetBatch
from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.reduction.dbnn.model import DBNNGIF
from braincell.reduction.dbnn.stimulus import SparseEvents, StimulusPlan, generate_multichannel_protocol
from braincell.reduction.dbnn.workflow import _stratified_split_indices, fit_dbnn_gif
from braincell.reduction.dbnn import fit_dbnn_gif as public_fit_dbnn_gif


def _layout():
    return ChannelLayout((Synapse("ExpSyn", name="E"),), (0,), (0,), (0,), (0,))


def _batch(plan: StimulusPlan) -> DatasetBatch:
    voltage = np.tile(np.asarray([-65.0, -64.0, -10.0, -63.0, -62.0], dtype=np.float32), (plan.n_traces, 1))
    spikes = tuple(np.asarray([2.0], dtype=np.float32) for _ in range(plan.n_traces))
    return DatasetBatch(
        plan,
        voltage,
        spikes,
        np.arange(5, dtype=np.float32),
        {
            "dt_ms": 1.0,
            "input_alignment": "target-step",
            "source_fingerprint": "workflow-test",
            "spike_threshold_mv": -20.0,
        },
    )


def _pool(n_traces=6):
    trace_ids = np.repeat(np.arange(n_traces, dtype=np.int64), 2)
    plan = StimulusPlan(
        _layout().fingerprint,
        n_traces,
        SparseEvents(
            trace_ids,
            np.zeros(len(trace_ids), dtype=np.int64),
            np.tile(np.asarray([0.0, 3.0], dtype=np.float32), n_traces),
        ),
        np.ones((n_traces, 1), dtype=np.float32),
        4.0,
        tuple(range(100, 100 + n_traces)),
        ("multichannel",) * n_traces,
        "pool",
    )
    return _batch(plan)


def _fit_kwargs():
    return {
        "train_traces": 4,
        "validation_traces": 1,
        "test_traces": 1,
        "epochs": 1,
        "batch_size": 2,
        "initialization_search": False,
        "spike_window_pre_ms": 0.0,
        "spike_window_post_ms": 0.0,
        "gif_thresholds_mv": [-70.0],
        "gif_threshold_increments_mv": [0.0],
        "gif_threshold_taus_ms": [10.0],
        "gif_candidate_batch_size": 1,
    }


class FitDBNNGIFTest(unittest.TestCase):
    def test_fits_from_unified_dataset_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            gif, mse, variance_explained = fit_dbnn_gif(
                _layout(), dataset_pool=_pool(), output_dir=directory, **_fit_kwargs()
            )
            self.assertTrue((Path(directory) / "dbnn_gif_model.npz").is_file())
            self.assertTrue((Path(directory) / "data" / "spike_alignment_validation.npz").is_file())
            self.assertTrue((Path(directory) / "metrics.json").is_file())
            report = json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))

        self.assertIsInstance(gif, DBNNGIF)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(variance_explained, float)
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(np.isfinite(variance_explained))
        self.assertEqual(set(gif.test_spike_metrics), {"raw", "aligned"})
        self.assertIsNotNone(gif.spike_alignment_evidence)
        self.assertEqual(len(gif.spike_alignment_evidence.validation_seeds), 1)
        self.assertTrue(set(gif.spike_alignment_evidence.validation_seeds).issubset(gif.dataset_trace_seeds))
        self.assertEqual(
            report["test_spikes"]["aligned"]["tp"],
            gif.test_spike_metrics["aligned"]["tp"],
        )

    def test_is_exported_from_public_package(self):
        self.assertIs(public_fit_dbnn_gif, fit_dbnn_gif)

    def test_generates_pool_with_factory_and_acceptance_callback(self):
        accepted_batches = []
        fit_layout = ChannelLayout(
            _layout().specs,
            (0,),
            (0,),
            (0,),
            (0,),
            training_ranges=((0.25, 0.5),),
        )
        original_seed = generate_multichannel_protocol(
            fit_layout,
            n_traces=4,
            duration_ms=4.0,
            dt_ms=1.0,
            rate_hz=40.0,
            seed=20260828,
            split="pool_candidate_000",
        ).seeds[0]

        class FakeTeacherSession:
            def __init__(self, teacher, layout, *, n_traces):
                self.teacher = teacher
                self.layout = layout
                self.n_traces = n_traces

            def run(self, plan):
                accepted_batches.append(plan)
                covered_plan = StimulusPlan(
                    plan.layout_fingerprint,
                    plan.n_traces,
                    SparseEvents(
                        np.arange(plan.n_traces, dtype=np.int64),
                        np.zeros(plan.n_traces, dtype=np.int64),
                        np.zeros(plan.n_traces, dtype=np.float32),
                    ),
                    plan.channel_weights,
                    plan.duration_ms,
                    plan.seeds,
                    plan.protocol_labels,
                    plan.split,
                )
                return _batch(covered_plan)

        def accept_even_rows(batch):
            return np.arange(0, batch.plan.n_traces, 2, dtype=np.int64)

        with mock.patch("braincell.reduction.dbnn.workflow.TeacherSession", FakeTeacherSession):
            gif, mse, variance_explained = fit_dbnn_gif(
                fit_layout,
                cell_factory=lambda pop_size: None,
                rate_hz=40.0,
                acceptance=accept_even_rows,
                duration_ms=4.0,
                dt=1.0 * u.ms,
                teacher_batch_size=4,
                source_fingerprint="workflow-test",
                forbidden_trace_seeds=(original_seed,),
                **_fit_kwargs(),
            )

        self.assertEqual(len(accepted_batches), 3)
        self.assertIsInstance(gif, DBNNGIF)
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(np.isfinite(variance_explained))
        self.assertNotIn(original_seed, gif.dataset_trace_seeds)
        self.assertNotEqual(gif.data_seed_roots[0], 20260828)
        self.assertEqual(len(gif.dataset_trace_seeds), 6)
        self.assertEqual(len(gif.consumed_trace_seeds), 12)
        self.assertTrue(all(np.max(plan.channel_weights) <= 0.5 for plan in accepted_batches))

    def test_generation_rejects_layout_without_positive_event_amplitude(self):
        invalid_layout = ChannelLayout(
            _layout().specs,
            (0,),
            (0,),
            (0,),
            (0,),
            training_ranges=((0.0, 0.0),),
        )
        with self.assertRaisesRegex(ValueError, "no positive event amplitude"):
            fit_dbnn_gif(
                invalid_layout,
                cell_factory=lambda pop_size: None,
                rate_hz=40.0,
                duration_ms=4.0,
                dt=1.0 * u.ms,
                teacher_batch_size=4,
                source_fingerprint="workflow-test",
                **_fit_kwargs(),
            )

    def test_pool_trace_seeds_cannot_overlap_forbidden_data(self):
        with self.assertRaisesRegex(ValueError, "overlap forbidden"):
            fit_dbnn_gif(
                _layout(),
                dataset_pool=_pool(),
                forbidden_trace_seeds=(100,),
                **_fit_kwargs(),
            )

    def test_requires_exactly_one_data_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            fit_dbnn_gif(_layout(), **_fit_kwargs())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            fit_dbnn_gif(
                _layout(),
                cell_factory=lambda pop_size: None,
                dataset_pool=_pool(),
                rate_hz=40.0,
                **_fit_kwargs(),
            )

    def test_validates_generation_and_pool_boundaries(self):
        with self.assertRaisesRegex(ValueError, "rate_hz is required"):
            fit_dbnn_gif(_layout(), cell_factory=lambda pop_size: None, **_fit_kwargs())
        with self.assertRaisesRegex(ValueError, "source_fingerprint is required"):
            fit_dbnn_gif(
                _layout(), cell_factory=lambda pop_size: None, rate_hz=40.0, **_fit_kwargs()
            )
        with self.assertRaisesRegex(ValueError, "rate_hz must be finite"):
            fit_dbnn_gif(
                _layout(),
                cell_factory=lambda pop_size: None,
                rate_hz=np.nan,
                source_fingerprint="workflow-test",
                **_fit_kwargs(),
            )
        with self.assertRaisesRegex(ValueError, "exactly 7 traces"):
            fit_dbnn_gif(
                _layout(),
                dataset_pool=_pool(),
                train_traces=5,
                validation_traces=1,
                test_traces=1,
                **{key: value for key, value in _fit_kwargs().items() if not key.endswith("_traces")},
            )
        with self.assertRaisesRegex(ValueError, "incompatible with dataset_pool metadata"):
            fit_dbnn_gif(
                _layout(), dataset_pool=_pool(), spike_threshold_mv=-10.0, **_fit_kwargs()
            )
        pool = _pool()
        metadata = dict(pool.metadata)
        metadata.pop("spike_threshold_mv")
        missing_threshold = DatasetBatch(
            pool.plan, pool.voltage_mv, pool.spike_times_ms, pool.time_ms, metadata
        )
        with self.assertRaisesRegex(ValueError, "must declare spike_threshold_mv"):
            fit_dbnn_gif(_layout(), dataset_pool=missing_threshold, **_fit_kwargs())

    def test_rejects_pool_without_channel_coverage(self):
        pool = _pool()
        empty_plan = StimulusPlan(
            pool.plan.layout_fingerprint,
            pool.plan.n_traces,
            SparseEvents(
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=np.float32),
            ),
            pool.plan.channel_weights,
            pool.plan.duration_ms,
            pool.plan.seeds,
            pool.plan.protocol_labels,
            pool.plan.split,
        )
        uncovered = DatasetBatch(
            empty_plan, pool.voltage_mv, pool.spike_times_ms, pool.time_ms, pool.metadata
        )

        with self.assertRaisesRegex(ValueError, "holdout-safe"):
            fit_dbnn_gif(_layout(), dataset_pool=uncovered, **_fit_kwargs())
        zero_weight_plan = StimulusPlan(
            pool.plan.layout_fingerprint,
            pool.plan.n_traces,
            pool.plan.events,
            np.zeros_like(pool.plan.channel_weights),
            pool.plan.duration_ms,
            pool.plan.seeds,
            pool.plan.protocol_labels,
            pool.plan.split,
        )
        zero_weight = DatasetBatch(
            zero_weight_plan, pool.voltage_mv, pool.spike_times_ms, pool.time_ms, pool.metadata
        )
        with self.assertRaisesRegex(ValueError, "holdout-safe"):
            fit_dbnn_gif(_layout(), dataset_pool=zero_weight, **_fit_kwargs())

    def test_stratified_split_keeps_a_spiking_validation_trace(self):
        pool = _pool()
        mixed = DatasetBatch(
            pool.plan,
            pool.voltage_mv,
            (np.asarray([]),) * 3 + pool.spike_times_ms[3:],
            pool.time_ms,
            pool.metadata,
        )

        indices = _stratified_split_indices(
            mixed, {"train": 4, "validation": 1, "test": 1}, seed=42
        )

        self.assertTrue(any(len(mixed.spike_times_ms[index]) for index in indices["validation"]))

    def test_rejects_pair_coverage_that_can_be_consumed_by_holdouts(self):
        layout = ChannelLayout(
            (Synapse("ExpSyn", name="E"),),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 0),
        )
        plan = StimulusPlan(
            layout.fingerprint,
            6,
            SparseEvents(
                np.asarray([0, 0, 1, 2, 3, 4, 5]),
                np.asarray([0, 1, 0, 0, 0, 0, 0]),
                np.zeros(7, dtype=np.float32),
            ),
            np.ones((6, 2), dtype=np.float32),
            4.0,
            tuple(range(6)),
            ("multichannel",) * 6,
            "pool",
        )
        pool = _batch(plan)

        with self.assertRaisesRegex(ValueError, "holdout-safe"):
            fit_dbnn_gif(layout, dataset_pool=pool, **_fit_kwargs())

    def test_rejects_invalid_acceptance_rows(self):
        class FakeTeacherSession:
            def __init__(self, teacher, layout, *, n_traces):
                self.n_traces = n_traces

            def run(self, plan):
                return _batch(plan)

        with mock.patch("braincell.reduction.dbnn.workflow.TeacherSession", FakeTeacherSession):
            with self.assertRaisesRegex(ValueError, "duplicate rows"):
                fit_dbnn_gif(
                    _layout(),
                    cell_factory=lambda pop_size: None,
                    rate_hz=40.0,
                    acceptance=lambda batch: np.asarray([0, 0]),
                    duration_ms=4.0,
                    dt=1.0 * u.ms,
                    teacher_batch_size=4,
                    max_teacher_batches=1,
                    source_fingerprint="workflow-test",
                    **_fit_kwargs(),
                )


if __name__ == "__main__":
    unittest.main()
