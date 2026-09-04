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

# ruff: noqa: E402

"""Compare random, Sobol, and differential-evolution starts before RTRL training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from examples.experimental.optim.gradients import (
    build_rollout_value_and_grad,
    build_trajectory_value_and_grad,
)
from examples.experimental.optim_initialization import dc_protocol_dataset as dataset_module

METHODS = ("direct_random", "random_screen", "sobol", "two_points_de")
SCREEN_METHODS = ("random_screen", "sobol", "two_points_de")
PARAMETER_NAMES = dataset_module.parameter_names()
LOG_SCALE_LOW = math.log(0.5)
LOG_SCALE_HIGH = math.log(1.5)
DEFAULT_OUTPUT_DIR = dataset_module.ARTIFACT_ROOT / "comparison"


@dataclass(frozen=True)
class ExperimentConfig:
    """Static search, training, and checkpoint settings."""

    screen_candidates: int = 1024
    screen_batch_size: int = 64
    selected_starts: int = 16
    updates: int = 180
    learning_rate: float = 0.01
    gradient_clip_norm: float = 1.0
    checkpoint_every: int = 10
    validation_rmse_threshold_mv: float = 5.0
    parameter_relative_rms_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.screen_candidates < self.selected_starts:
            raise ValueError("screen_candidates must be at least selected_starts.")
        if self.screen_candidates % self.screen_batch_size:
            raise ValueError("screen_candidates must be divisible by screen_batch_size.")
        if self.screen_candidates & (self.screen_candidates - 1):
            raise ValueError("screen_candidates must be a power of two for Sobol random_base2().")
        if self.selected_starts < 1 or self.updates < 1:
            raise ValueError("selected_starts and updates must be positive.")
        if self.updates % self.checkpoint_every:
            raise ValueError("updates must be divisible by checkpoint_every.")


class CandidateMetrics(NamedTuple):
    """Forward-only metrics with one leading candidate axis."""

    total_mse: object
    protocol_mse: object
    spike_counts: object
    spike_timing_error_ms: object


@dataclass(frozen=True)
class SearchResult:
    """All evaluated coordinates and the selected training starts."""

    method: str
    seed: int
    log_coordinates: np.ndarray
    scales: np.ndarray
    metrics: CandidateMetrics
    selected_indices: np.ndarray

    @property
    def selected_scales(self) -> np.ndarray:
        return self.scales[self.selected_indices]


def coordinates_to_scales(coordinates) -> object:
    """Map log-relative search coordinates to strict physical scale bounds."""
    return jnp.clip(jnp.exp(jnp.asarray(coordinates)), 0.5 + 1e-6, 1.5 - 1e-6)


def scales_to_optimizer_roots(scales) -> tuple[object, ...]:
    """Invert ``SigmoidT(0.5, 1.5)`` for a ``(..., 9)`` scale matrix."""
    values = jnp.clip(jnp.asarray(scales), 0.5 + 1e-6, 1.5 - 1e-6)
    raw = jnp.log((values - 0.5) / (1.5 - values))
    return tuple(raw[..., index] for index in range(raw.shape[-1]))


def optimizer_roots_to_scales(roots) -> object:
    """Map named optimizer roots back to physical relative scales."""
    raw = jnp.stack(tuple(roots), axis=-1)
    return 0.5 + jax.nn.sigmoid(raw)


def clip_gradient_rows(gradient, max_norm: float) -> tuple[object, object]:
    """Clip each start independently and return clipped rows and original norms."""
    values = jnp.asarray(gradient)
    if values.ndim != 2:
        raise ValueError(f"gradient must have shape (start, parameter), got {values.shape!r}.")
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")
    norms = jnp.sqrt(jnp.sum(values * values, axis=1))
    factor = jnp.minimum(1.0, max_norm / jnp.maximum(norms, 1e-30))
    return values * factor[:, None], norms


class CandidateEvaluator:
    """Compile forward-only batched voltage metrics for one protocol split."""

    def __init__(
        self,
        protocols: tuple[dataset_module.DcProtocol, ...],
        target_voltage_mv,
        *,
        batch_size: int,
        num_steps: int = dataset_module.N_STEPS,
    ) -> None:
        target = jnp.asarray(target_voltage_mv)
        expected = (len(protocols), num_steps, 3)
        if target.shape != expected:
            raise ValueError(f"Expected target shape {expected!r}, got {target.shape!r}.")
        self.protocols = protocols
        self.target = target
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.times_ms = jnp.arange(num_steps, dtype=target.dtype) * dataset_module.DT_MS
        self.cell = dataset_module.build_cell(protocols, trainable=True)

        def observation_step(time_ms):
            voltage = self.cell.V.value.to_decimal(u.mV)
            with brainstate.environ.context(t=time_ms * u.ms):
                self.cell.update()
            return voltage

        engine = build_trajectory_value_and_grad(
            self.cell,
            step=observation_step,
            loss=lambda observations, _times: jnp.mean(observations * 0.0),
            method="rtrl",
        )
        engine.prepare(self.times_ms[0])
        if engine.parameter_names != PARAMETER_NAMES:
            raise RuntimeError(f"Unexpected parameter order {engine.parameter_names!r}.")
        self.engine = engine
        target_counts, target_indices = _spike_summary(target[..., 0], max_spikes=8)

        def evaluate_one(roots):
            time_leading = engine._observation_rollout(roots, self.times_ms)
            prediction = jnp.moveaxis(time_leading, 0, 1)
            protocol_mse = jnp.mean((prediction - target) ** 2, axis=(1, 2))
            counts, indices = _spike_summary(prediction[..., 0], max_spikes=8)
            timing = _ordered_timing_error(counts, indices, target_counts, target_indices)
            return CandidateMetrics(jnp.mean(protocol_mse), protocol_mse, counts, timing)

        def trace_one(roots):
            return jnp.moveaxis(engine._observation_rollout(roots, self.times_ms), 0, 1)

        self._evaluate = jax.jit(jax.vmap(evaluate_one))
        self._trace = jax.jit(jax.vmap(trace_one))

    def evaluate_scales(self, scales) -> CandidateMetrics:
        """Evaluate any candidate count through fixed-size padded chunks."""
        values = np.asarray(scales, dtype=np.float64)
        _validate_scale_matrix(values)
        chunks = []
        for offset in range(0, values.shape[0], self.batch_size):
            chunk = values[offset : offset + self.batch_size]
            size = chunk.shape[0]
            if size < self.batch_size:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], self.batch_size - size, axis=0)), axis=0)
            output = self._evaluate(scales_to_optimizer_roots(jnp.asarray(chunk)))
            _block_until_ready(output)
            chunks.append(jax.tree.map(lambda value, n=size: np.asarray(value)[:n], output))
        return jax.tree.map(lambda *parts: np.concatenate(parts, axis=0), *chunks)

    def traces(self, scales) -> np.ndarray:
        """Return full traces for a candidate collection."""
        values = np.asarray(scales, dtype=np.float64)
        _validate_scale_matrix(values)
        outputs = []
        for offset in range(0, values.shape[0], self.batch_size):
            chunk = values[offset : offset + self.batch_size]
            size = chunk.shape[0]
            if size < self.batch_size:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], self.batch_size - size, axis=0)), axis=0)
            output = self._trace(scales_to_optimizer_roots(jnp.asarray(chunk)))
            output.block_until_ready()
            outputs.append(np.asarray(output)[:size])
        return np.concatenate(outputs, axis=0)


class RtrlTrainer:
    """Compile one 16-lane exact-RTRL MSE gradient kernel."""

    def __init__(
        self,
        protocols: tuple[dataset_module.DcProtocol, ...],
        target_voltage_mv,
        *,
        num_starts: int,
        num_steps: int = dataset_module.N_STEPS,
        method: str = "rtrl",
    ) -> None:
        if method not in {"bptt", "rtrl"}:
            raise ValueError("method must be 'bptt' or 'rtrl'.")
        target = jnp.asarray(target_voltage_mv)
        self.num_starts = num_starts
        self.num_steps = num_steps
        self.cell = dataset_module.build_cell(protocols, trainable=True)
        times_ms = jnp.arange(num_steps, dtype=target.dtype) * dataset_module.DT_MS

        def rollout_step(data):
            time_ms, target_mv = data
            voltage = self.cell.V.value.to_decimal(u.mV)
            local_loss = jnp.mean((voltage - target_mv) ** 2) / num_steps
            with brainstate.environ.context(t=time_ms * u.ms):
                self.cell.update()
            return local_loss

        engine = build_rollout_value_and_grad(self.cell, step=rollout_step, method=method)
        engine.prepare((times_ms[0], target[:, 0]))
        if engine.parameter_names != PARAMETER_NAMES:
            raise RuntimeError(f"Unexpected parameter order {engine.parameter_names!r}.")
        step_data = (times_ms, jnp.moveaxis(target, 0, 1))

        def one_start(roots):
            result = engine._rtrl(roots, step_data) if method == "rtrl" else engine._bptt(roots, step_data)
            return result.loss, jnp.stack(tuple(result.gradients[name] for name in PARAMETER_NAMES))

        self._gradient = jax.jit(jax.vmap(one_start))

    def gradient(self, scales) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(scales, dtype=np.float64)
        if values.shape != (self.num_starts, len(PARAMETER_NAMES)):
            raise ValueError(f"Expected scale shape {(self.num_starts, len(PARAMETER_NAMES))!r}, got {values.shape!r}.")
        loss, gradient = self._gradient(scales_to_optimizer_roots(jnp.asarray(values)))
        _block_until_ready((loss, gradient))
        return np.asarray(loss), np.asarray(gradient)

    def gradient_roots(self, roots) -> tuple[object, object]:
        loss, gradient = self._gradient(roots)
        _block_until_ready((loss, gradient))
        return loss, gradient


def run_search(
    method: str,
    seed: int,
    evaluator: CandidateEvaluator,
    config: ExperimentConfig,
) -> SearchResult:
    """Generate/evolve candidates, evaluate them, and select 16 starts."""
    if method not in METHODS:
        raise ValueError(f"Unknown search method {method!r}.")
    count = config.selected_starts if method == "direct_random" else config.screen_candidates
    if method in ("direct_random", "random_screen"):
        random = brainstate.random.RandomState(seed)
        coordinates = np.asarray(random.uniform(LOG_SCALE_LOW, LOG_SCALE_HIGH, size=(count, len(PARAMETER_NAMES))))
        scales = np.asarray(coordinates_to_scales(coordinates))
        metrics = evaluator.evaluate_scales(scales)
    elif method == "sobol":
        sampler = qmc.Sobol(d=len(PARAMETER_NAMES), scramble=True, seed=seed)
        unit = sampler.random_base2(int(math.log2(config.screen_candidates)))
        coordinates = qmc.scale(unit, LOG_SCALE_LOW, LOG_SCALE_HIGH)
        scales = np.asarray(coordinates_to_scales(coordinates))
        metrics = evaluator.evaluate_scales(scales)
    else:
        coordinates, metrics = _run_nevergrad(seed, evaluator, config)
        scales = np.asarray(coordinates_to_scales(coordinates))
    finite_loss = np.where(np.isfinite(metrics.total_mse), metrics.total_mse, np.inf)
    selected = np.argsort(finite_loss, kind="stable")[: config.selected_starts]
    if not np.isfinite(finite_loss[selected]).all():
        raise RuntimeError(f"Search method {method!r} produced fewer than {config.selected_starts} finite candidates.")
    return SearchResult(method, seed, coordinates, scales, metrics, selected.astype(np.int32))


def train_selected_starts(
    search: SearchResult,
    trainer: RtrlTrainer,
    evaluators: dict[str, CandidateEvaluator],
    config: ExperimentConfig,
) -> dict[str, np.ndarray]:
    """Train selected starts and retain validation-best checkpoints."""
    initial_scales = np.asarray(search.selected_scales)
    roots = tuple(brainstate.ParamState(value) for value in scales_to_optimizer_roots(jnp.asarray(initial_scales)))
    parameter_states = {name: state for name, state in zip(PARAMETER_NAMES, roots)}
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(parameter_states)

    update_loss = []
    gradient_history = []
    gradient_norm = []
    delta_history = []
    scale_history = [initial_scales]
    checkpoint_updates = []
    checkpoint_scales = []
    checkpoint_metrics = {split: [] for split in ("train", "validation")}
    best_loss = np.full((config.selected_starts,), np.inf)
    best_scales = initial_scales.copy()
    best_update = np.zeros((config.selected_starts,), dtype=np.int32)

    def checkpoint(update: int, scales: np.ndarray) -> None:
        nonlocal best_loss, best_scales, best_update
        metrics = {split: evaluators[split].evaluate_scales(scales) for split in checkpoint_metrics}
        checkpoint_updates.append(update)
        checkpoint_scales.append(scales.copy())
        for split, value in metrics.items():
            checkpoint_metrics[split].append(value)
        validation_loss = np.asarray(metrics["validation"].total_mse)
        improved = validation_loss < best_loss
        best_loss = np.where(improved, validation_loss, best_loss)
        best_scales = np.where(improved[:, None], scales, best_scales)
        best_update = np.where(improved, update, best_update)

    checkpoint(0, initial_scales)
    for update in range(1, config.updates + 1):
        before_roots = tuple(state.value for state in parameter_states.values())
        loss, flat_gradient = trainer.gradient_roots(before_roots)
        clipped, norms = clip_gradient_rows(flat_gradient, config.gradient_clip_norm)
        gradients = {name: clipped[:, index] for index, name in enumerate(PARAMETER_NAMES)}
        optimizer.update(gradients)
        after_roots = tuple(state.value for state in parameter_states.values())
        _block_until_ready(after_roots)
        scales = np.asarray(optimizer_roots_to_scales(after_roots))
        update_loss.append(np.asarray(loss))
        gradient_history.append(np.asarray(flat_gradient))
        gradient_norm.append(np.asarray(norms))
        delta_history.append(np.asarray(jnp.stack(after_roots, axis=1) - jnp.stack(before_roots, axis=1)))
        scale_history.append(scales)
        if update % config.checkpoint_every == 0:
            checkpoint(update, scales)

    final_scales = scale_history[-1]
    final_metrics = {split: evaluator.evaluate_scales(final_scales) for split, evaluator in evaluators.items()}
    best_metrics = {split: evaluator.evaluate_scales(best_scales) for split, evaluator in evaluators.items()}
    target = np.ones((len(PARAMETER_NAMES),), dtype=np.float64)
    parameter_relative_rms = np.sqrt(np.mean(((best_scales - target) / target) ** 2, axis=1))
    validation_rmse = np.sqrt(np.asarray(best_metrics["validation"].total_mse))
    validation_counts = np.asarray(best_metrics["validation"].spike_counts)
    expected_counts, _ = _spike_summary(np.asarray(evaluators["validation"].target)[..., 0], max_spikes=8)
    count_exact = np.all(validation_counts == np.asarray(expected_counts)[None, :], axis=1)
    trace_success = count_exact & (validation_rmse <= config.validation_rmse_threshold_mv)
    parameter_success = parameter_relative_rms <= config.parameter_relative_rms_threshold

    arrays = {
        "initial_scales": initial_scales,
        "update_loss": np.stack(update_loss),
        "gradient": np.stack(gradient_history),
        "gradient_norm": np.stack(gradient_norm),
        "optimizer_delta": np.stack(delta_history),
        "scale_history": np.stack(scale_history),
        "checkpoint_updates": np.asarray(checkpoint_updates),
        "checkpoint_scales": np.stack(checkpoint_scales),
        "best_scales": best_scales,
        "best_update": best_update,
        "best_validation_loss": best_loss,
        "parameter_relative_rms": parameter_relative_rms,
        "trace_success": trace_success,
        "parameter_success": parameter_success,
        "joint_success": trace_success & parameter_success,
    }
    for state_name, metric_group in (
        ("checkpoint", checkpoint_metrics),
        ("final", final_metrics),
        ("best", best_metrics),
    ):
        for split, metrics in metric_group.items():
            if isinstance(metrics, list):
                arrays[f"{state_name}_{split}_total_mse"] = np.stack([np.asarray(item.total_mse) for item in metrics])
                arrays[f"{state_name}_{split}_protocol_mse"] = np.stack(
                    [np.asarray(item.protocol_mse) for item in metrics]
                )
                arrays[f"{state_name}_{split}_spike_counts"] = np.stack(
                    [np.asarray(item.spike_counts) for item in metrics]
                )
                arrays[f"{state_name}_{split}_timing_error_ms"] = np.stack(
                    [np.asarray(item.spike_timing_error_ms) for item in metrics]
                )
            else:
                arrays[f"{state_name}_{split}_total_mse"] = np.asarray(metrics.total_mse)
                arrays[f"{state_name}_{split}_protocol_mse"] = np.asarray(metrics.protocol_mse)
                arrays[f"{state_name}_{split}_spike_counts"] = np.asarray(metrics.spike_counts)
                arrays[f"{state_name}_{split}_timing_error_ms"] = np.asarray(metrics.spike_timing_error_ms)
    arrays["initial_test_trace"] = evaluators["test"].traces(initial_scales)
    arrays["best_train_trace"] = evaluators["train"].traces(best_scales)
    arrays["best_validation_trace"] = evaluators["validation"].traces(best_scales)
    arrays["best_test_trace"] = evaluators["test"].traces(best_scales)
    arrays["final_test_trace"] = evaluators["test"].traces(final_scales)
    return arrays


def run_worker(
    *,
    stage: str,
    output_dir: Path,
    config: ExperimentConfig,
    resume: bool = False,
) -> dict[str, object]:
    """Run one complete worker under a continuous x64 lifecycle."""
    with jax.enable_x64(True), brainstate.environ.context(dt=dataset_module.DT_MS * u.ms, precision=64):
        return _run_worker(stage=stage, output_dir=output_dir, config=config, resume=resume)


def _run_worker(
    *,
    stage: str,
    output_dir: Path,
    config: ExperimentConfig,
    resume: bool,
) -> dict[str, object]:
    """Generate/load data, run all search methods, and train selected starts."""
    if stage not in {"pilot", "formal"}:
        raise ValueError("stage must be 'pilot' or 'formal'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": asdict(config),
        "dt_ms": dataset_module.DT_MS,
        "duration_ms": dataset_module.DURATION_MS,
        "parameter_names": list(PARAMETER_NAMES),
    }
    manifest_path = output_dir / "manifest.json"
    if resume and manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing manifest does not match the requested experiment configuration.")
    _write_json(manifest_path, manifest)
    data_dir = output_dir / "dataset"
    data = (
        dataset_module.load_dataset(data_dir)
        if (data_dir / "dataset.npz").exists()
        else dataset_module.generate_dataset(data_dir)
    )
    split_data = {split: data.subset(split) for split in dataset_module.SPLITS}
    evaluators = {
        split: CandidateEvaluator(protocols, voltage, batch_size=config.screen_batch_size)
        for split, (protocols, voltage) in split_data.items()
    }
    trainer = RtrlTrainer(*split_data["train"], num_starts=config.selected_starts)
    seeds = (0,) if stage == "pilot" else (0, 1, 2)
    rows = []
    for seed in seeds:
        for method in METHODS:
            stem = f"{method}_seed{seed}"
            metadata_path = output_dir / f"training_{stem}.json"
            history_path = output_dir / f"training_{stem}.npz"
            search_path = output_dir / f"search_{stem}.npz"
            if resume and metadata_path.exists() and history_path.exists() and search_path.exists():
                row = json.loads(metadata_path.read_text(encoding="utf-8"))
                with np.load(history_path) as training:
                    row.update(_training_summary_fields(training))
                rows.append(row)
                _write_json(metadata_path, row)
                continue
            started = time.perf_counter()
            search = run_search(method, seed, evaluators["train"], config)
            search_seconds = time.perf_counter() - started
            _write_search(search_path, search)
            started = time.perf_counter()
            training = train_selected_starts(search, trainer, evaluators, config)
            training_seconds = time.perf_counter() - started
            np.savez_compressed(history_path, **training)
            row = {
                "method": method,
                "seed": seed,
                "screen_evaluations": int(search.scales.shape[0]),
                "search_seconds": search_seconds,
                "training_seconds": training_seconds,
                "initial_mse_min": float(np.min(search.metrics.total_mse[search.selected_indices])),
                "best_validation_mse_median": float(np.median(training["best_validation_loss"])),
                "trace_success": int(np.sum(training["trace_success"])),
                "parameter_success": int(np.sum(training["parameter_success"])),
                "joint_success": int(np.sum(training["joint_success"])),
                "starts": config.selected_starts,
            }
            row.update(_training_summary_fields(training))
            rows.append(row)
            _write_json(metadata_path, row)
    summary = {
        "status": "ok",
        "stage": stage,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "config": asdict(config),
        "rows": rows,
        "aggregates": _aggregate_rows(output_dir, rows),
    }
    _write_summary(output_dir, summary)
    _plot_summary(output_dir / "comparison.png", output_dir, rows, config)
    return summary


def run_subprocess(
    *,
    stage: str,
    output_dir: Path,
    config: ExperimentConfig,
    gpu: int,
    python_executable: Path,
    resume: bool,
) -> None:
    """Launch the experiment in an isolated CUDA process."""
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAX_PLATFORMS": "cuda",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "worker",
        "--stage",
        stage,
        "--output-dir",
        str(output_dir),
        "--config",
        json.dumps(asdict(config)),
    ]
    if resume:
        command.append("--resume")
    completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    (output_dir / "worker.log").write_text(
        completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"Experiment worker failed; see {output_dir / 'worker.log'}.")


def _run_nevergrad(seed: int, evaluator: CandidateEvaluator, config: ExperimentConfig):
    bounds = {name: (LOG_SCALE_LOW, LOG_SCALE_HIGH) for name in PARAMETER_NAMES}
    metric_batches = []

    def batched_loss(**values):
        coordinates = np.stack([np.asarray(values[name]) for name in PARAMETER_NAMES], axis=1)
        metrics = evaluator.evaluate_scales(np.asarray(coordinates_to_scales(coordinates)))
        metric_batches.append(metrics)
        return np.asarray(metrics.total_mse)

    optimizer = braintools.optim.NevergradOptimizer(
        batched_loss,
        bounds,
        n_sample=config.screen_batch_size,
        method="TwoPointsDE",
        budget=config.screen_candidates,
        num_workers=config.screen_batch_size,
    )
    optimizer.parametrization.random_state.seed(seed)
    optimizer.minimize(n_iter=config.screen_candidates // config.screen_batch_size, verbose=False)
    coordinates = np.asarray(
        [[candidate[name] for name in PARAMETER_NAMES] for candidate in optimizer.candidates],
        dtype=np.float64,
    )
    metrics = jax.tree.map(lambda *parts: np.concatenate(tuple(map(np.asarray, parts)), axis=0), *metric_batches)
    np.testing.assert_allclose(metrics.total_mse, optimizer.errors, rtol=1e-9, atol=1e-9)
    return coordinates, metrics


def _spike_summary(voltage_mv, *, max_spikes: int):
    values = jnp.asarray(voltage_mv)
    crossings = (values[..., :-1] < 0.0) & (values[..., 1:] >= 0.0)
    counts = jnp.sum(crossings, axis=-1)

    def indices_one(mask):
        return jnp.nonzero(mask, size=max_spikes, fill_value=mask.shape[0])[0] + 1

    indices = jax.vmap(indices_one)(crossings)
    return counts, indices


def _ordered_timing_error(counts, indices, target_counts, target_indices):
    active = jnp.arange(indices.shape[-1])[None, :] < target_counts[:, None]
    difference = jnp.max(jnp.where(active, jnp.abs(indices - target_indices), 0), axis=-1)
    return jnp.where(counts == target_counts, difference * dataset_module.DT_MS, jnp.nan)


def _validate_scale_matrix(values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(f"Scale matrix must have shape (candidate, {len(PARAMETER_NAMES)}), got {values.shape!r}.")
    if values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Scale matrix must contain finite candidates.")
    if np.any(values <= 0.5) or np.any(values >= 1.5):
        raise ValueError("Scale candidates must be strictly inside (0.5, 1.5).")


def _write_search(path: Path, result: SearchResult) -> None:
    np.savez_compressed(
        path,
        log_coordinates=result.log_coordinates,
        scales=result.scales,
        total_mse=np.asarray(result.metrics.total_mse),
        protocol_mse=np.asarray(result.metrics.protocol_mse),
        spike_counts=np.asarray(result.metrics.spike_counts),
        spike_timing_error_ms=np.asarray(result.metrics.spike_timing_error_ms),
        selected_indices=result.selected_indices,
    )


def _write_summary(output_dir: Path, summary: dict[str, object]) -> None:
    _write_json(output_dir / "summary.json", summary)
    rows = summary["rows"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_summary_fields(training) -> dict[str, float]:
    relative_rms = np.asarray(training["parameter_relative_rms"])
    return {
        "parameter_relative_rms_min": float(np.min(relative_rms)),
        "parameter_relative_rms_median": float(np.median(relative_rms)),
    }


def _aggregate_rows(output_dir: Path, rows) -> dict[str, dict[str, float | int]]:
    aggregates = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        relative_rms = []
        validation_loss = []
        for row in selected:
            with np.load(output_dir / f"training_{method}_seed{row['seed']}.npz") as values:
                relative_rms.append(np.asarray(values["parameter_relative_rms"]))
                validation_loss.append(np.asarray(values["best_validation_loss"]))
        relative_rms = np.concatenate(relative_rms)
        validation_loss = np.concatenate(validation_loss)
        starts = int(sum(row["starts"] for row in selected))
        trace_success = int(sum(row["trace_success"] for row in selected))
        parameter_success = int(sum(row["parameter_success"] for row in selected))
        joint_success = int(sum(row["joint_success"] for row in selected))
        aggregates[method] = {
            "repeats": len(selected),
            "starts": starts,
            "trace_success": trace_success,
            "trace_success_fraction": trace_success / starts,
            "parameter_success": parameter_success,
            "parameter_success_fraction": parameter_success / starts,
            "joint_success": joint_success,
            "joint_success_fraction": joint_success / starts,
            "best_validation_mse_median": float(np.median(validation_loss)),
            "parameter_relative_rms_min": float(np.min(relative_rms)),
            "parameter_relative_rms_median": float(np.median(relative_rms)),
            "search_seconds": float(sum(row["search_seconds"] for row in selected)),
            "training_seconds": float(sum(row["training_seconds"] for row in selected)),
        }
    return aggregates


def _plot_summary(path: Path, output_dir: Path, rows, config: ExperimentConfig) -> None:
    import matplotlib.pyplot as plt

    methods = list(METHODS)
    labels = ("direct", "random-1024", "sobol-1024", "DE-1024")
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.24
    for offset, field in enumerate(("trace_success", "parameter_success", "joint_success")):
        values = np.asarray([sum(row[field] for row in rows if row["method"] == method) for method in methods])
        totals = np.asarray([sum(row["starts"] for row in rows if row["method"] == method) for method in methods])
        axes[0, 0].bar(x + (offset - 1) * width, values / totals, width=width, label=field.replace("_", " "))
    axes[0, 0].set(
        title="Convergence criteria", ylabel="success fraction", ylim=(0.0, 1.0), xticks=x, xticklabels=labels
    )
    axes[0, 0].legend(frameon=False)

    selected_initial = []
    validation_loss = []
    parameter_error = []
    for method in methods:
        method_initial = []
        method_validation = []
        method_parameter = []
        for row in rows:
            if row["method"] != method:
                continue
            with np.load(output_dir / f"search_{method}_seed{row['seed']}.npz") as search:
                method_initial.extend(search["total_mse"][search["selected_indices"]].tolist())
            with np.load(output_dir / f"training_{method}_seed{row['seed']}.npz") as training:
                method_validation.extend(training["best_validation_loss"].tolist())
                method_parameter.extend(training["parameter_relative_rms"].tolist())
        selected_initial.append(method_initial)
        validation_loss.append(method_validation)
        parameter_error.append(method_parameter)

    axes[0, 1].boxplot(selected_initial, tick_labels=labels, showfliers=False)
    axes[0, 1].set(title="Selected initial train MSE", ylabel="MSE (mV squared)", yscale="log")
    axes[1, 0].boxplot(validation_loss, tick_labels=labels, showfliers=False)
    axes[1, 0].axhline(config.validation_rmse_threshold_mv**2, color="tab:red", linestyle="--", label="5 mV RMSE")
    axes[1, 0].set(title="Validation-best MSE", ylabel="MSE (mV squared)", yscale="log")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].boxplot(parameter_error, tick_labels=labels, showfliers=False)
    axes[1, 1].axhline(config.parameter_relative_rms_threshold, color="tab:red", linestyle="--", label="10% threshold")
    axes[1, 1].set(title="Validation-best parameter error", ylabel="relative RMS")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(True, axis="y")
        axis.tick_params(axis="x", rotation=15)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--stage", choices=("pilot", "formal"), default="pilot")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--gpu", type=int, default=7)
    run.add_argument("--python", type=Path, default=Path("/home/swl/anaconda3/envs/braincell_311/bin/python"))
    run.add_argument("--config", default=json.dumps(asdict(ExperimentConfig())))
    run.add_argument("--resume", action="store_true")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--stage", choices=("pilot", "formal"), required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--config", required=True)
    worker.add_argument("--resume", action="store_true")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    config = ExperimentConfig(**json.loads(args.config))
    if args.command == "worker":
        summary = run_worker(stage=args.stage, output_dir=args.output_dir, config=config, resume=args.resume)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    run_subprocess(
        stage=args.stage,
        output_dir=args.output_dir,
        config=config,
        gpu=args.gpu,
        python_executable=args.python,
        resume=args.resume,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
