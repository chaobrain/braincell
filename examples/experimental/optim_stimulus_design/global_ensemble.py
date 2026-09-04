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

"""Forward-only global low-loss ensemble for the seven-CV stimulus dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from examples.experimental.optim.gradients import build_trajectory_value_and_grad
from examples.experimental.optim_stimulus_design import dataset as dataset_module
from examples.experimental.optim_stimulus_design.robust_oed import prior_scales, scales_to_raw_roots

ENSEMBLE_SEED = 20260831
ENSEMBLE_SIZE = 16384
TOP_SIZE = 256
BATCH_SIZE = 64
NORMALIZER_FLOOR_MV2 = 1.0
PROFILE_POINTS = 101
PLANE_POINTS = 41


@dataclass(frozen=True)
class EnsembleScores:
    """Candidate-leading aggregate metrics for three frozen splits."""

    raw_train: np.ndarray
    raw_validation: np.ndarray
    raw_test: np.ndarray
    normalized_train: np.ndarray
    normalized_validation: np.ndarray
    normalized_test: np.ndarray
    spike_distance_train: np.ndarray
    spike_distance_validation: np.ndarray
    spike_distance_test: np.ndarray
    spike_exact_train: np.ndarray
    spike_exact_validation: np.ndarray
    spike_exact_test: np.ndarray
    parameter_relative_rms: np.ndarray


class ForwardEnsembleEvaluator:
    """Evaluate voltage MSE and hard spike counts without differentiating."""

    def __init__(
        self,
        dataset: dataset_module.StimulusDataset,
        *,
        batch_size: int = BATCH_SIZE,
        num_steps: int = dataset_module.N_STEPS,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if num_steps < 1 or num_steps > dataset_module.N_STEPS:
            raise ValueError(f"num_steps must lie within [1, {dataset_module.N_STEPS}].")
        self.dataset = dataset
        self.batch_size = batch_size
        self.target = jnp.asarray(dataset.voltage_mv[:, :num_steps])
        self.cell = dataset_module.build_cell(dataset.current_na, trainable=True)
        self.times_ms = jnp.arange(num_steps, dtype=jnp.float64) * dataset_module.DT_MS

        def observation_step(time_ms):
            voltage = self.cell.V.value.to_decimal(u.mV)
            with brainstate.environ.context(t=time_ms * u.ms):
                self.cell.update()
            return voltage

        engine = build_trajectory_value_and_grad(
            self.cell,
            step=observation_step,
            loss=lambda observation, _time: jnp.mean(observation * 0.0),
            method="rtrl",
        )
        engine.prepare(self.times_ms[0])
        if engine.parameter_names != dataset_module.PARAMETER_NAMES:
            raise RuntimeError(f"Unexpected parameter order {engine.parameter_names!r}.")
        self.engine = engine

        def one_candidate(raw_roots):
            time_leading = engine._observation_rollout(raw_roots, self.times_ms)
            prediction = jnp.moveaxis(time_leading, 0, 1)
            protocol_mse = jnp.mean((prediction - self.target) ** 2, axis=(1, 2))
            soma = prediction[..., 0]
            counts = jnp.sum((soma[:, :-1] < 0.0) & (soma[:, 1:] >= 0.0), axis=1)
            return protocol_mse, counts

        self._compiled = jax.jit(jax.vmap(one_candidate))

    def evaluate(self, scales) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate arbitrary candidate count through fixed-size chunks."""
        values = np.asarray(scales, dtype=np.float64)
        _validate_scales(values)
        losses = []
        counts = []
        for offset in range(0, values.shape[0], self.batch_size):
            chunk = values[offset : offset + self.batch_size]
            size = chunk.shape[0]
            if size < self.batch_size:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], self.batch_size - size, axis=0)), axis=0)
            with jax.enable_x64(True):
                loss, count = self._compiled(scales_to_raw_roots(jnp.asarray(chunk, dtype=jnp.float64)))
            loss.block_until_ready()
            losses.append(np.asarray(loss)[:size])
            counts.append(np.asarray(count)[:size])
        return np.concatenate(losses), np.concatenate(counts)


def ensemble_scales(size: int = ENSEMBLE_SIZE) -> np.ndarray:
    """Return deterministic scrambled Sobol scales without injecting target."""
    if size < 1 or size & (size - 1):
        raise ValueError("size must be a positive power of two.")
    sampler = qmc.Sobol(d=len(dataset_module.PARAMETER_NAMES), scramble=True, seed=ENSEMBLE_SEED)
    unit = sampler.random_base2(int(math.log2(size)))
    values = qmc.scale(unit, math.log(0.5), math.log(1.5))
    return np.exp(values)


def protocol_normalizers(prior_protocol_mse) -> np.ndarray:
    """Return fixed prior-median protocol normalizers with a 1 mV squared floor."""
    values = np.asarray(prior_protocol_mse, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 60:
        raise ValueError(f"prior_protocol_mse must have shape (reference,60), got {values.shape!r}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("prior protocol losses must be finite and non-negative.")
    return np.maximum(np.median(values, axis=0), NORMALIZER_FLOOR_MV2)


def aggregate_scores(dataset, scales, protocol_mse, spike_counts, normalizer) -> EnsembleScores:
    """Aggregate raw/normalized voltage and spike metrics by frozen split."""
    values = np.asarray(scales)
    mse = np.asarray(protocol_mse)
    counts = np.asarray(spike_counts)
    expected_mse = (values.shape[0], len(dataset.protocols))
    if mse.shape != expected_mse or counts.shape != expected_mse:
        raise ValueError(f"Metric arrays must have shape {expected_mse!r}.")
    normalizer = np.asarray(normalizer)
    if normalizer.shape != (len(dataset.protocols),):
        raise ValueError("normalizer must have one value per protocol.")
    target_counts = np.asarray(dataset.target_spike_counts)

    def split_metrics(split):
        indices = dataset.indices(split)
        raw = np.mean(mse[:, indices], axis=1)
        normalized = np.mean(mse[:, indices] / normalizer[indices], axis=1)
        difference = np.abs(counts[:, indices] - target_counts[indices][None, :])
        return raw, normalized, np.sum(difference, axis=1), np.all(difference == 0, axis=1)

    train = split_metrics("train")
    validation = split_metrics("validation")
    test = split_metrics("test")
    parameter_rms = np.sqrt(np.mean((values - 1.0) ** 2, axis=1))
    return EnsembleScores(
        train[0],
        validation[0],
        test[0],
        train[1],
        validation[1],
        test[1],
        train[2],
        validation[2],
        test[2],
        train[3],
        validation[3],
        test[3],
        parameter_rms,
    )


def top_indices(score, *, size: int = TOP_SIZE) -> np.ndarray:
    """Return stable finite lowest-score indices."""
    values = np.asarray(score)
    if values.ndim != 1 or values.size < size:
        raise ValueError(f"score must contain at least {size} candidates.")
    finite = np.where(np.isfinite(values), values, np.inf)
    selected = np.argsort(finite, kind="stable")[:size]
    if not np.isfinite(finite[selected]).all():
        raise RuntimeError(f"Fewer than {size} candidates have finite scores.")
    return selected.astype(np.int32)


def parameter_geometry(log_scales, selected_indices, fim_eigenvectors) -> dict[str, np.ndarray | float]:
    """Compute PCA and variance in ascending FIM eigenvector coordinates."""
    values = np.asarray(log_scales)[np.asarray(selected_indices)]
    centered = values - np.mean(values, axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    denominator = max(values.shape[0] - 1, 1)
    pca_variance = singular**2 / denominator
    covariance = centered.T @ centered / denominator
    fim_vectors = np.asarray(fim_eigenvectors)
    projected = values @ fim_vectors
    fim_variance = np.var(projected, axis=0, ddof=1)
    cosine = float(abs(np.dot(vh[0], fim_vectors[:, 0])))
    return {
        "mean_log_scale": np.mean(values, axis=0),
        "covariance": covariance,
        "pca_components": vh,
        "pca_variance": pca_variance,
        "fim_projection": projected,
        "fim_variance": fim_variance,
        "pc1_weak_cosine": cosine,
    }


def target_fim_geometry(information_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending target FIM eigenvalues/eigenvectors from all train candidates."""
    with np.load(information_path) as values:
        information = np.asarray(values["information"])
    target_fim = np.sum(information[0], axis=0)
    return np.linalg.eigh(target_fim)


def direction_interval(direction, *, low: float = math.log(0.5), high: float = math.log(1.5)) -> tuple[float, float]:
    """Return the full alpha interval keeping ``alpha * direction`` in log bounds."""
    vector = np.asarray(direction, dtype=np.float64)
    lower = -np.inf
    upper = np.inf
    for value in vector:
        if value > 0.0:
            lower = max(lower, low / value)
            upper = min(upper, high / value)
        elif value < 0.0:
            lower = max(lower, high / value)
            upper = min(upper, low / value)
    if not lower < upper:
        raise ValueError("Direction does not intersect a finite scale interval.")
    return float(lower), float(upper)


def evaluate_direction_profiles(evaluator, dataset, normalizer, eigenvectors) -> dict[str, np.ndarray]:
    """Evaluate weakest/strongest lines and the two-weakest target plane."""
    outputs = {}
    for name, direction in (("weak", eigenvectors[:, 0]), ("strong", eigenvectors[:, -1])):
        lower, upper = direction_interval(direction)
        margin = 1e-9 * (upper - lower)
        half = (PROFILE_POINTS - 1) // 2
        alpha = np.concatenate(
            (
                np.linspace(lower + margin, 0.0, half + 1),
                np.linspace(0.0, upper - margin, half + 1)[1:],
            )
        )
        scales = np.exp(alpha[:, None] * direction[None, :])
        mse, counts = evaluator.evaluate(scales)
        scores = aggregate_scores(dataset, scales, mse, counts, normalizer)
        outputs[f"{name}_alpha"] = alpha
        outputs[f"{name}_scales"] = scales
        for field, value in scores.__dict__.items():
            outputs[f"{name}_{field}"] = np.asarray(value)

    first = eigenvectors[:, 0]
    second = eigenvectors[:, 1]
    first_range = direction_interval(first)
    second_range = direction_interval(second)
    axis_0 = np.linspace(*first_range, PLANE_POINTS)
    axis_1 = np.linspace(*second_range, PLANE_POINTS)
    mesh_0, mesh_1 = np.meshgrid(axis_0, axis_1, indexing="ij")
    log_scales = mesh_0[..., None] * first + mesh_1[..., None] * second
    valid = np.all((log_scales > math.log(0.5)) & (log_scales < math.log(1.5)), axis=-1)
    flat_valid = np.flatnonzero(valid.ravel())
    valid_scales = np.exp(log_scales.reshape((-1, 6))[flat_valid])
    mse, counts = evaluator.evaluate(valid_scales)
    scores = aggregate_scores(dataset, valid_scales, mse, counts, normalizer)
    outputs["plane_axis_0"] = axis_0
    outputs["plane_axis_1"] = axis_1
    outputs["plane_valid"] = valid
    outputs["plane_scales"] = valid_scales
    for field, value in scores.__dict__.items():
        grid = np.full((PLANE_POINTS * PLANE_POINTS,), np.nan)
        grid[flat_valid] = np.asarray(value)
        outputs[f"plane_{field}"] = grid.reshape((PLANE_POINTS, PLANE_POINTS))
    return outputs


def run_ensemble(
    dataset_dir: Path = dataset_module.ARTIFACT_ROOT / "dataset",
    information_path: Path = dataset_module.ARTIFACT_ROOT / "oed" / "information.npz",
    output_dir: Path = dataset_module.ARTIFACT_ROOT / "global_ensemble",
    *,
    size: int = ENSEMBLE_SIZE,
    batch_size: int = BATCH_SIZE,
) -> dict[str, object]:
    """Evaluate, analyse, and persist the forward-only global ensemble."""
    dataset = dataset_module.load_dataset(dataset_dir)
    scales = ensemble_scales(size)
    prior = prior_scales()
    output_dir.mkdir(parents=True, exist_ok=True)
    with jax.enable_x64(True), brainstate.environ.context(dt=dataset_module.DT_MS * u.ms, precision=64):
        evaluator = ForwardEnsembleEvaluator(dataset, batch_size=batch_size)
        started = time.perf_counter()
        example_roots = scales_to_raw_roots(jnp.asarray(scales[:batch_size]))
        compiled = evaluator._compiled.lower(example_roots).compile()
        compile_seconds = time.perf_counter() - started
        evaluator._compiled = compiled
        started = time.perf_counter()
        prior_mse, prior_counts = evaluator.evaluate(prior)
        normalizer = protocol_normalizers(prior_mse)
        protocol_mse, spike_counts = evaluator.evaluate(scales)
        evaluation_seconds = time.perf_counter() - started
        memory = compiled.memory_analysis()
    scores = aggregate_scores(dataset, scales, protocol_mse, spike_counts, normalizer)
    selected_size = min(TOP_SIZE, size)
    raw_top = top_indices(scores.raw_train, size=selected_size)
    normalized_top = top_indices(scores.normalized_train, size=selected_size)
    eigenvalues, eigenvectors = target_fim_geometry(information_path)
    raw_geometry = parameter_geometry(np.log(scales), raw_top, eigenvectors)
    normalized_geometry = parameter_geometry(np.log(scales), normalized_top, eigenvectors)
    started = time.perf_counter()
    profiles = evaluate_direction_profiles(evaluator, dataset, normalizer, eigenvectors)
    profile_seconds = time.perf_counter() - started
    np.savez_compressed(
        output_dir / "ensemble.npz",
        scales=scales,
        protocol_mse=protocol_mse,
        spike_counts=spike_counts,
        target_spike_counts=dataset.target_spike_counts,
        normalizer=normalizer,
        prior_scales=prior,
        prior_protocol_mse=prior_mse,
        prior_spike_counts=prior_counts,
        raw_top_indices=raw_top,
        normalized_top_indices=normalized_top,
        fim_eigenvalues=eigenvalues,
        fim_eigenvectors=eigenvectors,
        **{name: np.asarray(value) for name, value in scores.__dict__.items()},
        **{f"raw_geometry_{name}": value for name, value in raw_geometry.items()},
        **{f"normalized_geometry_{name}": value for name, value in normalized_geometry.items()},
        **profiles,
    )
    _write_top_csv(output_dir / "top_raw.csv", scales, scores, raw_top, eigenvectors)
    _write_top_csv(output_dir / "top_normalized.csv", scales, scores, normalized_top, eigenvectors)
    overlap = np.intersect1d(raw_top, normalized_top)
    summary = _summary(
        scores,
        raw_top,
        normalized_top,
        raw_geometry,
        normalized_geometry,
        overlap,
        selected_size,
        compile_seconds,
        evaluation_seconds,
        profile_seconds,
        memory,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_heldout(output_dir / "train_heldout_scatter.png", scores, raw_top, normalized_top)
    _plot_geometry(
        output_dir / "parameter_geometry.png", scales, raw_top, normalized_top, raw_geometry, normalized_geometry
    )
    _plot_profiles(output_dir / "weak_direction_profiles.png", profiles)
    _plot_spikes(output_dir / "spike_signature_summary.png", scores, raw_top, normalized_top)
    return summary


def _write_top_csv(path, scales, scores, indices, eigenvectors) -> None:
    rows = []
    projections = np.log(scales[indices]) @ eigenvectors
    for rank, (index, projection) in enumerate(zip(indices, projections), start=1):
        row = {
            "rank": rank,
            "candidate_index": int(index),
            **{name: float(scales[index, offset]) for offset, name in enumerate(dataset_module.PARAMETER_NAMES)},
            "raw_train": float(scores.raw_train[index]),
            "raw_validation": float(scores.raw_validation[index]),
            "raw_test": float(scores.raw_test[index]),
            "normalized_train": float(scores.normalized_train[index]),
            "normalized_validation": float(scores.normalized_validation[index]),
            "normalized_test": float(scores.normalized_test[index]),
            "spike_distance_train": int(scores.spike_distance_train[index]),
            "spike_distance_validation": int(scores.spike_distance_validation[index]),
            "spike_distance_test": int(scores.spike_distance_test[index]),
            "parameter_relative_rms": float(scores.parameter_relative_rms[index]),
            "weak_projection": float(projection[0]),
            "second_weak_projection": float(projection[1]),
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    scores,
    raw_top,
    normalized_top,
    raw_geometry,
    normalized_geometry,
    overlap,
    selected_size,
    compile_s,
    eval_s,
    profile_s,
    memory,
):
    def selected_summary(indices):
        return {
            "raw_train_median": float(np.median(scores.raw_train[indices])),
            "raw_validation_median": float(np.median(scores.raw_validation[indices])),
            "raw_test_median": float(np.median(scores.raw_test[indices])),
            "normalized_train_median": float(np.median(scores.normalized_train[indices])),
            "normalized_validation_median": float(np.median(scores.normalized_validation[indices])),
            "normalized_test_median": float(np.median(scores.normalized_test[indices])),
            "parameter_relative_rms_median": float(np.median(scores.parameter_relative_rms[indices])),
            "validation_spike_exact": int(np.sum(scores.spike_exact_validation[indices])),
            "test_spike_exact": int(np.sum(scores.spike_exact_test[indices])),
        }

    return {
        "status": "ok",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "candidate_count": int(scores.raw_train.size),
        "top_size": selected_size,
        "compile_seconds": compile_s,
        "evaluation_seconds": eval_s,
        "profile_seconds": profile_s,
        "temporary_bytes": int(memory.temp_size_in_bytes),
        "raw_top": selected_summary(raw_top),
        "normalized_top": selected_summary(normalized_top),
        "top_overlap": int(overlap.size),
        "top_jaccard": float(overlap.size / np.union1d(raw_top, normalized_top).size),
        "raw_pc1_weak_cosine": float(raw_geometry["pc1_weak_cosine"]),
        "normalized_pc1_weak_cosine": float(normalized_geometry["pc1_weak_cosine"]),
        "all_raw_train_median": float(np.median(scores.raw_train)),
        "all_normalized_train_median": float(np.median(scores.normalized_train)),
        "raw_train_validation_correlation": _finite_correlation(scores.raw_train, scores.raw_validation),
        "normalized_train_validation_correlation": _finite_correlation(
            scores.normalized_train, scores.normalized_validation
        ),
    }


def _finite_correlation(left, right) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(np.asarray(left)[mask], np.asarray(right)[mask])[0, 1])


def _plot_heldout(path, scores, raw_top, normalized_top) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), constrained_layout=True)
    fields = (
        (scores.raw_train, scores.raw_validation, "Raw train vs validation"),
        (scores.raw_train, scores.raw_test, "Raw train vs test"),
        (scores.normalized_train, scores.normalized_validation, "Normalized train vs validation"),
        (scores.normalized_train, scores.normalized_test, "Normalized train vs test"),
    )
    for axis, (x, y, title) in zip(axes.flat, fields):
        axis.scatter(x, y, s=4, alpha=0.12, color="0.5")
        axis.scatter(x[raw_top], y[raw_top], s=12, alpha=0.7, label="raw top")
        axis.scatter(x[normalized_top], y[normalized_top], s=12, alpha=0.7, label="normalized top")
        axis.set(xlabel="train", ylabel="held-out", title=title, xscale="log", yscale="log")
        axis.grid(True)
        axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_geometry(path, scales, raw_top, normalized_top, raw_geometry, normalized_geometry) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    for indices, geometry, label, color in (
        (raw_top, raw_geometry, "raw top", "tab:blue"),
        (normalized_top, normalized_geometry, "normalized top", "tab:orange"),
    ):
        centered = np.log(scales[indices]) - geometry["mean_log_scale"]
        pca = centered @ geometry["pca_components"].T
        axes[0, 0].scatter(pca[:, 0], pca[:, 1], s=12, alpha=0.6, label=label, color=color)
        fim = geometry["fim_projection"]
        axes[0, 1].scatter(fim[:, 0], fim[:, 1], s=12, alpha=0.6, label=label, color=color)
        axes[1, 0].plot(np.arange(1, 7), geometry["pca_variance"], marker="o", label=label, color=color)
        axes[1, 1].plot(np.arange(1, 7), geometry["fim_variance"], marker="o", label=label, color=color)
    axes[0, 0].set(title="Top ensemble PCA", xlabel="PC1", ylabel="PC2")
    axes[0, 1].set(title="FIM weak coordinates", xlabel="weak 1", ylabel="weak 2")
    axes[1, 0].set(title="PCA variance", xlabel="component", ylabel="variance", yscale="log")
    axes[1, 1].set(title="Variance along FIM eigenvectors", xlabel="weak to strong", ylabel="variance", yscale="log")
    for axis in axes.flat:
        axis.grid(True)
        axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_profiles(path, profiles) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    for row, name in enumerate(("weak", "strong")):
        alpha = profiles[f"{name}_alpha"]
        axes[row, 0].semilogy(alpha, profiles[f"{name}_raw_train"], label="train")
        axes[row, 0].semilogy(alpha, profiles[f"{name}_raw_validation"], label="validation")
        axes[row, 0].semilogy(alpha, profiles[f"{name}_raw_test"], label="test")
        axes[row, 0].set(title=f"{name} direction: raw", xlabel="alpha", ylabel="MSE")
        axes[row, 1].semilogy(alpha, profiles[f"{name}_normalized_train"], label="train")
        axes[row, 1].semilogy(alpha, profiles[f"{name}_normalized_validation"], label="validation")
        axes[row, 1].semilogy(alpha, profiles[f"{name}_normalized_test"], label="test")
        axes[row, 1].set(title=f"{name} direction: normalized", xlabel="alpha", ylabel="loss")
    for axis in axes.flat:
        axis.grid(True)
        axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_spikes(path, scores, raw_top, normalized_top) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, split in zip(axes, ("train", "validation", "test")):
        distance = getattr(scores, f"spike_distance_{split}")
        bins = np.arange(0, min(int(np.max(distance)), 30) + 2) - 0.5
        axis.hist(distance[raw_top], bins=bins, alpha=0.6, label="raw top")
        axis.hist(distance[normalized_top], bins=bins, alpha=0.6, label="normalized top")
        axis.set(title=f"{split} spike distance", xlabel="sum abs count error", ylabel="candidates")
        axis.grid(True)
        axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _validate_scales(values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] < 1:
        raise ValueError(f"scales must have shape (candidate,6), got {values.shape!r}.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.5) or np.any(values >= 1.5):
        raise ValueError("scales must be finite and strictly inside (0.5,1.5).")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=dataset_module.ARTIFACT_ROOT / "dataset")
    parser.add_argument("--information", type=Path, default=dataset_module.ARTIFACT_ROOT / "oed" / "information.npz")
    parser.add_argument("--output-dir", type=Path, default=dataset_module.ARTIFACT_ROOT / "global_ensemble")
    parser.add_argument("--size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = run_ensemble(
        args.dataset_dir,
        args.information,
        args.output_dir,
        size=args.size,
        batch_size=args.batch_size,
    )
    print(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
