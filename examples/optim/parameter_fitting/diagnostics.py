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

"""Composable, experiment-local diagnostics for multistart parameter fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, NamedTuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np


class StateSignals(NamedTuple):
    """Signals evaluated at one parameter state before an optimizer update."""

    optimizer_values: dict[str, object]
    physical_values: dict[str, object]
    losses: dict[str, object]
    metrics: dict[str, object]


class UpdateSignals(NamedTuple):
    """Signals associated with the update leaving one parameter state."""

    gradients: dict[str, object]
    learning_rate: object


@dataclass(frozen=True)
class DiagnosticConfig:
    window: int = 20
    relative_improvement: float = 0.005
    flat_gradient_ratio: float = 0.1
    movement_tolerance: float = 1e-3
    bound_margin: float = 0.02
    final_degradation: float = 0.1
    trace_loss_threshold: float | None = None
    parameter_error_threshold: float = 0.1


@dataclass(frozen=True)
class TrainingHistory:
    """Aligned state and update histories produced by a compiled training loop."""

    optimizer_values: dict[str, object]
    physical_values: dict[str, object]
    bound_positions: dict[str, object]
    losses: dict[str, object]
    metrics: dict[str, object]
    gradients: dict[str, object]
    learning_rate: object
    gradient_norm: object
    gradient_cosine: object
    optimizer_step_norm: object
    physical_step_norm: object | None

    @property
    def num_states(self) -> int:
        return int(self.losses["total"].shape[0])

    @property
    def num_updates(self) -> int:
        return int(self.gradient_norm.shape[0])

    @property
    def num_starts(self) -> int:
        return int(self.losses["total"].shape[1])


@dataclass(frozen=True)
class BestCheckpointArchive:
    """One fixed-shape best checkpoint per multistart lane."""

    valid: object
    epoch: object
    optimizer_values: dict[str, object]
    physical_values: dict[str, object]
    losses: dict[str, object]
    metrics: dict[str, object]


@dataclass(frozen=True)
class BestArchives:
    """Unconstrained-loss and spike-count-feasible best checkpoints."""

    continuous: BestCheckpointArchive
    spike_feasible: BestCheckpointArchive


def voltage_mse_objective(
    predictions: Mapping[str, object],
    targets: Mapping[str, object],
) -> tuple[object, dict[str, object]]:
    """Return per-start mean MSE and one component per named protocol."""
    _require_same_keys(predictions, targets, label="protocol")
    components = {}
    for name in predictions:
        prediction = jnp.asarray(predictions[name])
        target = jnp.asarray(targets[name])
        _require_voltage_shapes(prediction, target, name=name)
        components[f"voltage_mse/{name}"] = jnp.mean(
            (prediction - target[:, None, :]) ** 2,
            axis=(0, 2),
        )
    total = jnp.mean(jnp.stack(tuple(components.values())), axis=0)
    return total, components


def evaluate_voltage_protocols(
    predictions: Mapping[str, object],
    targets: Mapping[str, object],
    *,
    spike_probe: int | Mapping[str, int] = 0,
    threshold: float = 0.0,
) -> dict[str, object]:
    """Evaluate fixed-shape voltage and hard upward-crossing metrics."""
    _require_same_keys(predictions, targets, label="protocol")
    metrics = {}
    for name in predictions:
        prediction = jnp.asarray(predictions[name])
        target = jnp.asarray(targets[name])
        _require_voltage_shapes(prediction, target, name=name)
        probe = int(spike_probe[name] if isinstance(spike_probe, Mapping) else spike_probe)
        if not 0 <= probe < prediction.shape[2]:
            raise IndexError(f"Protocol {name!r} spike probe {probe} is out of range.")
        error = prediction - target[:, None, :]
        predicted_count = upward_crossing_count(prediction[:, :, probe], threshold=threshold)
        target_count = upward_crossing_count(target[:, probe], threshold=threshold)
        metrics[f"spike_count/{name}"] = predicted_count
        metrics[f"signed_count_error/{name}"] = predicted_count - target_count
        metrics[f"voltage_rmse/{name}"] = jnp.sqrt(jnp.mean(error**2, axis=0))
        metrics[f"finite/{name}"] = jnp.all(jnp.isfinite(prediction), axis=(0, 2))
    return metrics


def upward_crossing_count(voltage, *, threshold: float = 0.0):
    voltage = jnp.asarray(voltage)
    return jnp.sum((voltage[:-1] < threshold) & (voltage[1:] >= threshold), axis=0)


def capture_state(
    parameters,
    *,
    total_loss,
    components: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
) -> StateSignals:
    """Capture one pre-update parameter state without retaining gradient edges."""
    losses = {"total": total_loss, **dict(components or {})}
    return StateSignals(
        optimizer_values=_stop_mapping(parameters.optimizer_values()),
        physical_values=_stop_mapping(parameters.physical_values()),
        losses=_stop_mapping(losses),
        metrics=_stop_mapping(metrics or {}),
    )


def capture_update(gradients: Mapping[str, object], *, learning_rate) -> UpdateSignals:
    return UpdateSignals(
        gradients=_stop_mapping(gradients),
        learning_rate=jax.lax.stop_gradient(jnp.asarray(learning_rate)),
    )


def finalize_history(
    states: StateSignals,
    endpoint: StateSignals,
    updates: UpdateSignals,
    *,
    bounds: Mapping[str, tuple[object, object]] | None = None,
) -> TrainingHistory:
    """Append the post-training endpoint and derive per-start diagnostics."""
    optimizer_values = _append_mapping(states.optimizer_values, endpoint.optimizer_values)
    physical_values = _append_mapping(states.physical_values, endpoint.physical_values)
    losses = _append_mapping(states.losses, endpoint.losses)
    metrics = _append_mapping(states.metrics, endpoint.metrics)
    num_starts = int(losses["total"].shape[1])
    bound_positions = _bound_positions(physical_values, bounds or {})
    gradient_norm, gradient_cosine = _tree_norm_and_cosine(updates.gradients, num_starts=num_starts)
    optimizer_step_norm = _tree_step_norm(optimizer_values, num_starts=num_starts)
    physical_step_norm = _tree_step_norm(bound_positions, num_starts=num_starts) if bound_positions else None
    return TrainingHistory(
        optimizer_values=optimizer_values,
        physical_values=physical_values,
        bound_positions=bound_positions,
        losses=losses,
        metrics=metrics,
        gradients=dict(updates.gradients),
        learning_rate=updates.learning_rate,
        gradient_norm=gradient_norm,
        gradient_cosine=gradient_cosine,
        optimizer_step_norm=optimizer_step_norm,
        physical_step_norm=physical_step_norm,
    )


def extract_best_archives(history: TrainingHistory) -> BestArchives:
    """Extract per-start best checkpoints without changing the training trajectory."""
    finite = _finite_state_mask(history)
    continuous = _extract_archive(history, finite)
    signed_errors = [value for name, value in history.metrics.items() if name.startswith("signed_count_error/")]
    if signed_errors:
        feasible = finite
        for value in signed_errors:
            feasible = feasible & _reduce_state_start(jnp.asarray(value) == 0)
    else:
        feasible = jnp.zeros_like(finite)
    return BestArchives(
        continuous=continuous,
        spike_feasible=_extract_archive(history, feasible),
    )


def summarize_history(
    history: TrainingHistory,
    *,
    archives: BestArchives | None = None,
    target_parameters: Mapping[str, object] | None = None,
    config: DiagnosticConfig = DiagnosticConfig(),
) -> dict[str, object]:
    """Return host-side per-start state classifications and aggregate counts."""
    archives = extract_best_archives(history) if archives is None else archives
    loss = np.asarray(history.losses["total"], dtype=float)
    gradient_norm = np.asarray(history.gradient_norm, dtype=float)
    gradient_cosine = np.asarray(history.gradient_cosine, dtype=float)
    movement = np.asarray(
        history.physical_step_norm if history.physical_step_norm is not None else history.optimizer_step_norm,
        dtype=float,
    )
    rows = []
    for start in range(history.num_starts):
        finite = bool(np.isfinite(loss[:, start]).all()) and _metrics_finite(history, start)
        best_epoch = int(np.asarray(archives.continuous.epoch)[start])
        best_loss = (
            float(np.asarray(archives.continuous.losses["total"])[start])
            if bool(np.asarray(archives.continuous.valid)[start])
            else None
        )
        feasible_epoch = int(np.asarray(archives.spike_feasible.epoch)[start])
        feasible_loss = (
            float(np.asarray(archives.spike_feasible.losses["total"])[start])
            if bool(np.asarray(archives.spike_feasible.valid)[start])
            else None
        )
        final_vs_best = (
            float((loss[-1, start] - best_loss) / max(abs(best_loss), 1e-12)) if best_loss is not None else None
        )
        optimization_state = _optimization_state(
            loss[:, start],
            gradient_norm[:, start],
            gradient_cosine[:, start],
            movement[:, start],
            finite=finite,
            config=config,
        )
        parameter_error = _parameter_error(history, target_parameters, start=start)
        row = {
            "start": start,
            "initial_loss": float(loss[0, start]),
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "feasible_best_loss": feasible_loss,
            "feasible_best_epoch": feasible_epoch,
            "final_loss": float(loss[-1, start]),
            "final_vs_best": final_vs_best,
            "degraded_from_best": (None if final_vs_best is None else bool(final_vs_best > config.final_degradation)),
            "region_state": _region_state(history, start),
            "optimization_state": optimization_state,
            "near_bound": _near_bound(history, start, margin=config.bound_margin),
            "parameter_relative_error": parameter_error,
            "trace_success": (
                None if config.trace_loss_threshold is None else bool(loss[-1, start] <= config.trace_loss_threshold)
            ),
            "parameter_success": (
                None if parameter_error is None else bool(parameter_error <= config.parameter_error_threshold)
            ),
        }
        rows.append(row)
    counts = {}
    for field in ("region_state", "optimization_state"):
        for value in sorted({row[field] for row in rows}):
            counts[f"{field}/{value}"] = sum(row[field] == value for row in rows)
    for field in ("degraded_from_best", "near_bound", "trace_success", "parameter_success"):
        available = [row[field] for row in rows if row[field] is not None]
        if available:
            counts[field] = sum(available)
    counts["spike_feasible_archive"] = int(np.asarray(archives.spike_feasible.valid).sum())
    return {
        "num_starts": history.num_starts,
        "num_updates": history.num_updates,
        "config": asdict(config),
        "counts": counts,
        "starts": rows,
    }


def format_summary(summary: Mapping[str, object]) -> str:
    """Format a compact lane-by-lane diagnostic table."""
    lines = ["start  initial      best     final  best@  feasible@  region       optimization  param_err"]
    for row in summary["starts"]:
        error = "-" if row["parameter_relative_error"] is None else f"{row['parameter_relative_error']:.3f}"
        best = "-" if row["best_loss"] is None else f"{row['best_loss']:.3f}"
        lines.append(
            f"{row['start']:>5}  {row['initial_loss']:>7.3f}  {best:>8}  "
            f"{row['final_loss']:>8.3f}  {row['best_epoch']:>5}  "
            f"{row['feasible_best_epoch']:>9}  "
            f"{row['region_state']:<11}  {row['optimization_state']:<12}  {error:>9}"
        )
    return "\n".join(lines)


def save_artifacts(
    directory,
    history: TrainingHistory,
    summary: Mapping[str, object],
    *,
    metadata: Mapping[str, object],
    archives: BestArchives | None = None,
) -> Path:
    """Save inspectable array histories plus JSON metadata and summary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    units = {}
    _store_mapping(arrays, units, "optimizer", history.optimizer_values)
    _store_mapping(arrays, units, "physical", history.physical_values)
    _store_mapping(arrays, units, "bound_position", history.bound_positions)
    _store_mapping(arrays, units, "loss", history.losses)
    _store_mapping(arrays, units, "metric", history.metrics)
    _store_mapping(arrays, units, "gradient", history.gradients)
    arrays["derived/learning_rate"] = np.asarray(history.learning_rate)
    arrays["derived/gradient_norm"] = np.asarray(history.gradient_norm)
    arrays["derived/gradient_cosine"] = np.asarray(history.gradient_cosine)
    arrays["derived/optimizer_step_norm"] = np.asarray(history.optimizer_step_norm)
    if history.physical_step_norm is not None:
        arrays["derived/physical_step_norm"] = np.asarray(history.physical_step_norm)
    archives = extract_best_archives(history) if archives is None else archives
    _store_archive(arrays, units, "archive/continuous", archives.continuous)
    _store_archive(arrays, units, "archive/spike_feasible", archives.spike_feasible)
    np.savez_compressed(directory / "history.npz", **arrays)
    _write_json(directory / "metadata.json", {**dict(metadata), "units": units})
    _write_json(directory / "summary.json", summary)
    return directory


def plot_diagnostics(history: TrainingHistory):
    """Plot loss, gradient, movement, and spike-count distance histories."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    state_epoch = np.arange(history.num_states)
    update_epoch = np.arange(history.num_updates)
    loss = np.asarray(history.losses["total"])
    axes[0, 0].plot(state_epoch, loss, alpha=0.35)
    axes[0, 0].plot(state_epoch, loss.min(axis=1), color="black", label="best")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(title="Per-start loss", xlabel="State epoch", ylabel="MSE")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(update_epoch, np.asarray(history.gradient_norm), alpha=0.35)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Gradient norm", xlabel="Update", ylabel="L2 norm")
    movement = history.physical_step_norm if history.physical_step_norm is not None else history.optimizer_step_norm
    axes[1, 0].plot(update_epoch, np.asarray(movement), alpha=0.35)
    axes[1, 0].set(title="Parameter movement", xlabel="Update", ylabel="Normalized L2 step")
    signed = [value for name, value in history.metrics.items() if name.startswith("signed_count_error/")]
    if signed:
        distance = np.sum(np.abs(np.stack([np.asarray(value) for value in signed], axis=-1)), axis=-1)
        image = axes[1, 1].imshow(distance.T, aspect="auto", interpolation="nearest", origin="lower")
        figure.colorbar(image, ax=axes[1, 1], label="Count distance")
    axes[1, 1].set(title="Spike-count region", xlabel="State epoch", ylabel="Start")
    return figure


def _require_same_keys(left: Mapping, right: Mapping, *, label: str) -> None:
    if set(left) != set(right):
        raise KeyError(f"{label.title()} keys differ: {tuple(left)!r} != {tuple(right)!r}.")


def _require_voltage_shapes(prediction, target, *, name: str) -> None:
    if prediction.ndim != 3 or target.ndim != 2:
        raise ValueError(
            f"Protocol {name!r} requires prediction [time, start, probe] and target [time, probe]; "
            f"got {prediction.shape!r} and {target.shape!r}."
        )
    if prediction.shape[0] != target.shape[0] or prediction.shape[2] != target.shape[1]:
        raise ValueError(f"Protocol {name!r} prediction and target time/probe shapes differ.")


def _stop_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {name: jax.tree.map(jax.lax.stop_gradient, value) for name, value in values.items()}


def _append_mapping(stacked: Mapping[str, object], endpoint: Mapping[str, object]) -> dict[str, object]:
    _require_same_keys(stacked, endpoint, label="history")
    return {name: _append_value(stacked[name], endpoint[name]) for name in stacked}


def _append_value(stacked, endpoint):
    if isinstance(stacked, u.Quantity):
        unit = stacked.unit
        values = jnp.concatenate((stacked.to_decimal(unit), endpoint.to_decimal(unit)[None]), axis=0)
        return u.Quantity(values, unit)
    return jnp.concatenate((jnp.asarray(stacked), jnp.asarray(endpoint)[None]), axis=0)


def _bound_positions(values: Mapping[str, object], bounds: Mapping[str, tuple[object, object]]):
    positions = {}
    for name, value in values.items():
        if name not in bounds:
            continue
        lower, upper = bounds[name]
        if isinstance(value, u.Quantity):
            unit = value.unit
            raw = value.to_decimal(unit)
            low = lower.to_decimal(unit)
            high = upper.to_decimal(unit)
        else:
            raw, low, high = value, lower, upper
        positions[name] = (jnp.asarray(raw) - low) / (high - low)
    return positions


def _tree_matrices(values: Mapping[str, object], *, num_starts: int) -> tuple[object, ...]:
    matrices = []
    for name, value in values.items():
        raw = value.mantissa if isinstance(value, u.Quantity) else value
        array = jnp.asarray(raw)
        if array.ndim < 2 or array.shape[1] != num_starts:
            raise ValueError(
                f"Diagnostic root {name!r} must have history shape [step, start, ...]; got {array.shape!r}."
            )
        matrices.append(array.reshape((array.shape[0], num_starts, -1)))
    if not matrices:
        raise ValueError("Diagnostics require at least one parameter root.")
    return tuple(matrices)


def _tree_norm_and_cosine(values: Mapping[str, object], *, num_starts: int):
    matrices = _tree_matrices(values, num_starts=num_starts)
    squared = sum(jnp.sum(value**2, axis=-1) for value in matrices)
    norm = jnp.sqrt(squared)
    if norm.shape[0] == 0:
        return norm, norm
    dot = sum(jnp.sum(value[1:] * value[:-1], axis=-1) for value in matrices)
    denominator = jnp.maximum(norm[1:] * norm[:-1], 1e-12)
    cosine = jnp.concatenate((jnp.full((1, num_starts), jnp.nan), dot / denominator), axis=0)
    return norm, cosine


def _tree_step_norm(values: Mapping[str, object], *, num_starts: int):
    matrices = _tree_matrices(values, num_starts=num_starts)
    squared = sum(jnp.sum((value[1:] - value[:-1]) ** 2, axis=-1) for value in matrices)
    return jnp.sqrt(squared)


def _finite_state_mask(history: TrainingHistory):
    finite = jnp.ones((history.num_states, history.num_starts), dtype=bool)
    for values in (
        history.optimizer_values,
        history.physical_values,
        history.losses,
    ):
        for value in values.values():
            raw = value.mantissa if isinstance(value, u.Quantity) else value
            finite = finite & _reduce_state_start(jnp.isfinite(jnp.asarray(raw)))
    for name, value in history.metrics.items():
        if name.startswith("finite/"):
            finite = finite & _reduce_state_start(jnp.asarray(value, dtype=bool))
    return finite


def _reduce_state_start(value):
    value = jnp.asarray(value)
    if value.ndim < 2:
        raise ValueError(f"Archive values must have history shape [state, start, ...]; got {value.shape!r}.")
    return jnp.all(value, axis=tuple(range(2, value.ndim))) if value.ndim > 2 else value


def _extract_archive(history: TrainingHistory, eligible) -> BestCheckpointArchive:
    eligible = jnp.asarray(eligible, dtype=bool)
    expected_shape = (history.num_states, history.num_starts)
    if eligible.shape != expected_shape:
        raise ValueError(f"Archive eligibility must have shape {expected_shape!r}; got {eligible.shape!r}.")
    score = jnp.asarray(history.losses["total"])
    valid = jnp.any(eligible, axis=0)
    selected_epoch = jnp.argmin(jnp.where(eligible, score, jnp.inf), axis=0)
    safe_epoch = jnp.where(valid, selected_epoch, 0)
    return BestCheckpointArchive(
        valid=valid,
        epoch=jnp.where(valid, selected_epoch, -1),
        optimizer_values=_select_archive_mapping(history.optimizer_values, safe_epoch, valid),
        physical_values=_select_archive_mapping(history.physical_values, safe_epoch, valid),
        losses=_select_archive_mapping(history.losses, safe_epoch, valid),
        metrics=_select_archive_mapping(history.metrics, safe_epoch, valid),
    )


def _select_archive_mapping(values: Mapping[str, object], epochs, valid) -> dict[str, object]:
    return {name: _select_archive_value(value, epochs, valid) for name, value in values.items()}


def _select_archive_value(value, epochs, valid):
    if isinstance(value, u.Quantity):
        unit = value.unit
        selected = _select_archive_array(value.to_decimal(unit), epochs, valid)
        return u.Quantity(selected, unit)
    return _select_archive_array(value, epochs, valid)


def _select_archive_array(value, epochs, valid):
    value = jnp.asarray(value)
    if value.ndim < 2 or value.shape[1] != valid.shape[0]:
        raise ValueError(f"Archive value must have history shape [state, start, ...]; got {value.shape!r}.")
    start = jnp.arange(valid.shape[0])
    selected = value[epochs, start]
    mask = valid.reshape(valid.shape + (1,) * (selected.ndim - 1))
    if jnp.issubdtype(selected.dtype, jnp.inexact):
        placeholder = jnp.full_like(selected, jnp.nan)
    else:
        placeholder = jnp.zeros_like(selected)
    return jnp.where(mask, selected, placeholder)


def _metrics_finite(history: TrainingHistory, start: int) -> bool:
    finite = [value for name, value in history.metrics.items() if name.startswith("finite/")]
    return all(bool(np.asarray(value)[-1, start]) for value in finite)


def _region_state(history: TrainingHistory, start: int) -> str:
    errors = [
        int(np.asarray(value)[-1, start])
        for name, value in history.metrics.items()
        if name.startswith("signed_count_error/")
    ]
    if not errors:
        return "unknown"
    if all(value == 0 for value in errors):
        return "feasible"
    if all(value <= 0 for value in errors):
        return "missing"
    if all(value >= 0 for value in errors):
        return "extra"
    return "mixed"


def _optimization_state(loss, gradient_norm, gradient_cosine, movement, *, finite, config):
    if not finite:
        return "non-finite"
    updates = len(gradient_norm)
    window = max(1, min(config.window, updates // 2 if updates > 1 else 1))
    split = max(1, len(loss) - window)
    old_best = np.min(loss[:split])
    recent_best = np.min(loss[split:])
    improvement = (old_best - recent_best) / max(abs(old_best), 1e-12)
    if improvement >= config.relative_improvement:
        return "improving"
    early_gradient = np.nanmedian(gradient_norm[:window])
    recent_gradient = np.nanmedian(gradient_norm[-window:])
    recent_movement = float(np.nansum(movement[-window:]))
    if (
        recent_gradient <= config.flat_gradient_ratio * max(early_gradient, 1e-12)
        and recent_movement <= config.movement_tolerance
    ):
        return "flat"
    if np.nanmedian(gradient_cosine[-window:]) < 0.0:
        return "oscillatory"
    return "slow-progress"


def _near_bound(history: TrainingHistory, start: int, *, margin: float) -> bool | None:
    if not history.bound_positions:
        return None
    return any(
        bool(np.any((np.asarray(value)[-1, start] <= margin) | (np.asarray(value)[-1, start] >= 1.0 - margin)))
        for value in history.bound_positions.values()
    )


def _parameter_error(history: TrainingHistory, targets, *, start: int) -> float | None:
    if targets is None:
        return None
    _require_same_keys(history.physical_values, targets, label="parameter")
    errors = []
    for name, values in history.physical_values.items():
        final = values[-1, start]
        target = targets[name]
        if isinstance(final, u.Quantity):
            unit = final.unit
            final = np.asarray(final.to_decimal(unit), dtype=float)
            target = np.asarray(target.to_decimal(unit), dtype=float)
        else:
            final = np.asarray(final, dtype=float)
            target = np.asarray(target, dtype=float)
        errors.extend(np.ravel(np.abs(final - target) / np.maximum(np.abs(target), 1e-12)).tolist())
    return float(np.mean(errors))


def _store_mapping(arrays, units, prefix: str, values: Mapping[str, object]) -> None:
    for name, value in values.items():
        key = f"{prefix}/{name}"
        if isinstance(value, u.Quantity):
            arrays[key] = np.asarray(value.to_decimal(value.unit))
            units[key] = str(value.unit)
        else:
            arrays[key] = np.asarray(value)


def _store_archive(arrays, units, prefix: str, archive: BestCheckpointArchive) -> None:
    arrays[f"{prefix}/valid"] = np.asarray(archive.valid)
    arrays[f"{prefix}/epoch"] = np.asarray(archive.epoch)
    _store_mapping(arrays, units, f"{prefix}/optimizer", archive.optimizer_values)
    _store_mapping(arrays, units, f"{prefix}/physical", archive.physical_values)
    _store_mapping(arrays, units, f"{prefix}/loss", archive.losses)
    _store_mapping(arrays, units, f"{prefix}/metric", archive.metrics)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(value), file, indent=2, sort_keys=True, allow_nan=False)


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
