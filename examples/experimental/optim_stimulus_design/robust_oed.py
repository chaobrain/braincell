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

"""Robust D-optimal ordering for the seven-CV stimulus candidate dataset."""

from __future__ import annotations

import argparse
import csv
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

PRIOR_SEED = 20260830
PRIOR_SOBOL_POINTS = 16
SCALE_BOUNDS = (0.5, 1.5)
RIDGE = 1e-8
RANK_RTOL = 1e-8


def prior_scales() -> np.ndarray:
    """Return target plus sixteen log-space Sobol scale vectors."""
    sampler = qmc.Sobol(d=len(dataset_module.PARAMETER_NAMES), scramble=True, seed=PRIOR_SEED)
    unit = sampler.random_base2(int(math.log2(PRIOR_SOBOL_POINTS)))
    log_values = qmc.scale(unit, math.log(SCALE_BOUNDS[0]), math.log(SCALE_BOUNDS[1]))
    return np.concatenate((np.ones((1, len(dataset_module.PARAMETER_NAMES))), np.exp(log_values)), axis=0)


def scales_to_raw_roots(scales) -> tuple[object, ...]:
    """Invert the six ``SigmoidT(0.5, 1.5)`` trainable roots."""
    values = jnp.asarray(scales)
    if values.shape[-1] != len(dataset_module.PARAMETER_NAMES):
        raise ValueError(f"Expected final scale axis of size 6, got {values.shape!r}.")
    values = jnp.clip(values, SCALE_BOUNDS[0] + 1e-9, SCALE_BOUNDS[1] - 1e-9)
    raw = jnp.log((values - SCALE_BOUNDS[0]) / (SCALE_BOUNDS[1] - values))
    return tuple(raw[..., index] for index in range(raw.shape[-1]))


def raw_to_log_scale_factors(scales) -> object:
    """Return ``dz / dlog(scale)`` for the bounded sigmoid coordinates."""
    values = jnp.asarray(scales)
    derivative = (values - SCALE_BOUNDS[0]) * (SCALE_BOUNDS[1] - values)
    return values / derivative


class ObservationInformationEngine:
    """Accumulate one protocol FIM per parameter reference point."""

    def __init__(self, current_na, *, num_steps: int = dataset_module.N_STEPS) -> None:
        current = np.asarray(current_na, dtype=np.float64)
        if current.ndim != 3 or current.shape[1:] != (dataset_module.N_STEPS, len(dataset_module.SITES)):
            raise ValueError(
                f"current_na must have shape (protocol, {dataset_module.N_STEPS}, 3), got {current.shape!r}."
            )
        if num_steps < 1 or num_steps > dataset_module.N_STEPS:
            raise ValueError(f"num_steps must lie within [1, {dataset_module.N_STEPS}].")
        self.current = current
        self.num_steps = num_steps
        self.cell = dataset_module.build_cell(current, trainable=True)
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
        self._compiled = jax.jit(jax.vmap(self._one_reference))

    def information(self, scales) -> np.ndarray:
        """Return ``(reference,protocol,parameter,parameter)`` information."""
        values = np.asarray(scales, dtype=np.float64)
        expected = (values.shape[0], len(dataset_module.PARAMETER_NAMES))
        if values.ndim != 2 or values.shape != expected:
            raise ValueError(f"scales must have shape (reference,6), got {values.shape!r}.")
        output = self._compiled(scales_to_raw_roots(jnp.asarray(values)), jnp.asarray(values))
        output.block_until_ready()
        return np.asarray(output)

    def _one_reference(self, raw_roots, scales):
        values, tangents = self.engine._initial_full_carry(raw_roots)
        factors = raw_to_log_scale_factors(scales)
        tangents = jax.tree.map(lambda value: _scale_direction_tangent(value, factors), tangents)
        count = len(dataset_module.PARAMETER_NAMES)
        fim = jnp.zeros((self.current.shape[0], count, count), dtype=jnp.float64)

        def scan_step(carry, time_ms):
            current_values, current_tangents, current_fim = carry

            def transition(state_values):
                return self.engine._functional_step.call(state_values, time_ms)

            (next_values, _voltage), linear_map = jax.linearize(transition, current_values)
            next_tangents, voltage_tangents = jax.vmap(linear_map)(current_tangents)
            increment = jnp.einsum("kpc,lpc->pkl", voltage_tangents, voltage_tangents)
            return (next_values, next_tangents, current_fim + increment), None

        (_, _, fim), _ = jax.lax.scan(scan_step, (values, tangents, fim), self.times_ms)
        return fim / float(self.num_steps * self.cell.n_cv)


def _scale_direction_tangent(value, factors):
    if getattr(value, "dtype", None) == jax.dtypes.float0:
        return value
    shape = (factors.shape[0],) + (1,) * (value.ndim - 1)
    return value * factors.reshape(shape)


def robust_greedy_order(information, protocol_ids: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a stable worst-reference D-opt ordering and prefix metrics."""
    matrices = np.asarray(information, dtype=np.float64)
    if matrices.ndim != 4 or matrices.shape[2:] != (6, 6):
        raise ValueError(f"information must have shape (reference,protocol,6,6), got {matrices.shape!r}.")
    if matrices.shape[1] != len(protocol_ids):
        raise ValueError("protocol_ids length does not match the protocol axis.")
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    finite = np.isfinite(symmetric).all(axis=(0, 2, 3))
    safe_symmetric = np.where(finite[None, :, None, None], symmetric, 0.0)
    eigen = np.linalg.eigvalsh(safe_symmetric)
    valid = finite & (np.min(eigen, axis=(0, 2)) >= -1e-8)
    remaining = [index for index in range(len(protocol_ids)) if valid[index]]
    invalid = np.flatnonzero(~valid)
    selected = []
    accumulated = np.zeros((matrices.shape[0], 6, 6), dtype=np.float64)
    prefix = []
    identity = np.eye(6)
    while remaining:
        scores = []
        for index in remaining:
            candidate = accumulated + symmetric[:, index]
            sign, logdet = np.linalg.slogdet(candidate + RIDGE * identity)
            scores.append(float(np.min(np.where(sign > 0, logdet, -np.inf))))
        winner_position = int(np.argmax(np.asarray(scores)))
        winner = remaining.pop(winner_position)
        selected.append(winner)
        accumulated = accumulated + symmetric[:, winner]
        prefix.append(_prefix_metrics(accumulated))
    metrics = {name: np.asarray([item[name] for item in prefix]) for name in prefix[0]} if prefix else {}
    metrics["invalid_indices"] = invalid.astype(np.int32)
    metrics["valid_mask"] = valid
    return np.asarray(selected, dtype=np.int32), metrics


def _prefix_metrics(accumulated: np.ndarray) -> dict[str, object]:
    eigen = np.linalg.eigvalsh(accumulated)
    maximum = eigen[:, -1]
    threshold = RANK_RTOL * maximum
    ranks = np.sum(eigen > threshold[:, None], axis=1)
    minimum = eigen[:, 0]
    condition = np.where(ranks == 6, maximum / np.maximum(minimum, np.finfo(float).tiny), np.inf)
    sign, logdet = np.linalg.slogdet(accumulated + RIDGE * np.eye(6))
    diagonal = np.diagonal(accumulated, axis1=-2, axis2=-1)
    denominator = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
    correlation = np.divide(accumulated, denominator, out=np.zeros_like(accumulated), where=denominator > 0.0)
    off_diagonal = correlation - np.eye(6)[None, :, :]
    return {
        "worst_rank": int(np.min(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "minimum_eigenvalue": float(np.min(minimum)),
        "maximum_eigenvalue": float(np.max(maximum)),
        "worst_condition": float(np.max(condition)),
        "mean_logdet": float(np.mean(np.where(sign > 0, logdet, -np.inf))),
        "worst_logdet": float(np.min(np.where(sign > 0, logdet, -np.inf))),
        "worst_abs_correlation": float(np.max(np.abs(off_diagonal))),
        "target_correlation": correlation[0],
    }


def run_oed(
    dataset_dir: Path = dataset_module.ARTIFACT_ROOT / "dataset",
    output_dir: Path = dataset_module.ARTIFACT_ROOT / "oed",
) -> dict[str, object]:
    """Compute all train FIMs, rank protocols, and persist review artifacts."""
    data = dataset_module.load_dataset(dataset_dir)
    train_indices = data.indices("train")
    protocol_ids = tuple(data.protocols[index].protocol_id for index in train_indices)
    scales = prior_scales()
    output_dir.mkdir(parents=True, exist_ok=True)
    with jax.enable_x64(True), brainstate.environ.context(dt=dataset_module.DT_MS * u.ms, precision=64):
        engine = ObservationInformationEngine(data.current_na[train_indices])
        started = time.perf_counter()
        compiled = engine._compiled.lower(scales_to_raw_roots(jnp.asarray(scales)), jnp.asarray(scales)).compile()
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        information = compiled(scales_to_raw_roots(jnp.asarray(scales)), jnp.asarray(scales))
        information.block_until_ready()
        execution_seconds = time.perf_counter() - started
        information = np.asarray(information)
        memory = compiled.memory_analysis()
    ordering, metrics = robust_greedy_order(information, protocol_ids)
    np.savez_compressed(
        output_dir / "information.npz",
        prior_scales=scales,
        information=information,
        protocol_id=np.asarray(protocol_ids),
        ordering=ordering,
        **metrics,
    )
    rows = _ordering_rows(data, train_indices, ordering, metrics)
    with (output_dir / "ordering.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "ok",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "reference_points": int(scales.shape[0]),
        "train_candidates": int(len(train_indices)),
        "valid_candidates": int(np.sum(metrics["valid_mask"])),
        "invalid_candidates": int(len(metrics["invalid_indices"])),
        "compile_seconds": compile_seconds,
        "execution_seconds": execution_seconds,
        "temporary_bytes": int(memory.temp_size_in_bytes),
        "ordering": [protocol_ids[index] for index in ordering],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_oed(output_dir / "conditioning.png", metrics)
    return summary


def _ordering_rows(data, train_indices, ordering, metrics) -> list[dict[str, object]]:
    rows = []
    for rank, candidate_index in enumerate(ordering, start=1):
        protocol = data.protocols[int(train_indices[candidate_index])]
        rows.append(
            {
                "rank": rank,
                "protocol_id": protocol.protocol_id,
                "family": protocol.family,
                "feature": protocol.feature,
                "injection_site": protocol.injection_site,
                "worst_rank": int(metrics["worst_rank"][rank - 1]),
                "minimum_eigenvalue": float(metrics["minimum_eigenvalue"][rank - 1]),
                "worst_condition": float(metrics["worst_condition"][rank - 1]),
                "mean_logdet": float(metrics["mean_logdet"][rank - 1]),
                "worst_logdet": float(metrics["worst_logdet"][rank - 1]),
                "worst_abs_correlation": float(metrics["worst_abs_correlation"][rank - 1]),
            }
        )
    return rows


def _plot_oed(path: Path, metrics) -> None:
    import matplotlib.pyplot as plt

    count = len(metrics["worst_rank"])
    x = np.arange(1, count + 1)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    axes[0, 0].plot(x, metrics["worst_rank"], marker="o", markersize=3)
    axes[0, 0].set(title="Worst-reference rank", ylabel="rank", ylim=(0, 6.2))
    axes[0, 1].semilogy(x, np.maximum(metrics["minimum_eigenvalue"], np.finfo(float).tiny))
    axes[0, 1].set(title="Worst minimum eigenvalue", ylabel="eigenvalue")
    finite_condition = np.where(np.isfinite(metrics["worst_condition"]), metrics["worst_condition"], np.nan)
    axes[1, 0].semilogy(x, finite_condition)
    axes[1, 0].set(title="Worst condition number", ylabel="condition")
    axes[1, 1].plot(x, metrics["mean_logdet"], label="mean")
    axes[1, 1].plot(x, metrics["worst_logdet"], label="worst")
    axes[1, 1].set(title="Regularized log determinant", ylabel="logdet")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel("greedy prefix size")
        axis.grid(True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=dataset_module.ARTIFACT_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=dataset_module.ARTIFACT_ROOT / "oed")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = run_oed(args.dataset_dir, args.output_dir)
    print(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
