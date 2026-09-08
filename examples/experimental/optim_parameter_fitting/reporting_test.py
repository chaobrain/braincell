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

import csv
import time

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.experimental.optim_parameter_fitting.config import ExperimentConfig, InitializationConfig
from examples.experimental.optim_parameter_fitting.datasets import DatasetBundle, Protocol, feature_step_dataset
from examples.experimental.optim_parameter_fitting.losses import raw_voltage_mse
from examples.experimental.optim_parameter_fitting.models import hh_1cv_classic_bounded_direct
from examples.experimental.optim_parameter_fitting.optimizers import adam
from examples.experimental.optim_parameter_fitting.reporting import (
    _transition,
    archive_completed_run,
    save_run,
)
from examples.experimental.optim_parameter_fitting.training import run_pipeline


def test_small_run_writes_queryable_config_metrics_report_and_figures(tmp_path) -> None:
    config = ExperimentConfig(
        name="artifact_smoke",
        model=hh_1cv_classic_bounded_direct(),
        dataset=feature_step_dataset(),
        loss=raw_voltage_mse(),
        initialization=InitializationConfig(seed=0, num_candidates=2),
        stages=(adam(epochs=2, checkpoint_every=1),),
    )
    splits = ("train",) * 5 + ("validation",) * 2 + ("test",)
    protocols = tuple(Protocol(f"step_{i}", split, "smoke", 0.0, 0) for i, split in enumerate(splits))
    current = np.zeros((8, 3))
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        target = np.asarray(config.model.simulate(current, dt_ms=0.025))
        dataset = DatasetBundle(protocols, np.arange(3) * 0.025, current, target, 0.025, 0.025, 0.05)
        result = run_pipeline(config, dataset)
        source = tmp_path / "input_config.py"
        source.write_text("# test config\n", encoding="utf-8")
        summary = save_run(tmp_path / "result", config, source, result, run_started_at=time.perf_counter())

    root = tmp_path / "result"
    assert summary["num_starts"] == 2
    assert (root / "config.py").exists()
    assert (root / "resolved_config.json").exists()
    assert (root / "REPORT.md").exists()
    assert (root / "figures" / "loss_curves.png").exists()
    with (root / "metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    test_rows = [row for row in rows if row["split"] == "test"]
    assert {int(row["epoch"]) for row in test_rows} == {2}
    assert len(test_rows) == 2
    archive = archive_completed_run(config, root)
    assert archive["test_used_for_selection"] is False
    assert (root / "validation_archive.npz").exists()


def test_success_transition_labels_all_paired_states() -> None:
    assert _transition(True, True) == "both"
    assert _transition(False, True) == "gained"
    assert _transition(True, False) == "lost"
    assert _transition(False, False) == "neither"
