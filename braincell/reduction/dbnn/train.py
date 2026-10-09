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

"""DBNN training integration boundary."""

from collections.abc import Iterator, Mapping
import importlib.util
from itertools import product
from pathlib import Path
import time
from typing import Any

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dbnn.dataset import load_dataset, rasterize_events, validate_dataset
from braincell.reduction.dbnn.functional import (
    _apply_gif_closed_loop,
    compute_metrics,
    dbnn_forward,
    masked_mse,
)
from braincell.reduction.dbnn.model import DBNN, DBNNGIF, align_spike_times


_INITIALIZATION_DEFAULTS = {
    "backend": "auto",
    "method": "DE",
    "samples": 32,
    "time_stride": 5,
    "maxiter": 8,
    "popsize": 8,
    "max_pairs": 512,
    "ridge": 1e-4,
    "bounds": ((1.0, 30.0), (5.0, 100.0), (0.05, 5.0)),
}


def build_fit_mask(
    targets: Any,
    dt_ms: float,
    *,
    spike_threshold_mv: float,
    spike_window_pre_ms: float,
    spike_window_post_ms: float,
) -> np.ndarray:
    """Build the source trainer's subthreshold voltage-fit mask.

    Parameters
    ----------
    targets : array-like
        Target voltages with shape ``(trace, time)``.
    dt_ms : float
        Sampling interval in milliseconds.
    spike_threshold_mv : float
        Threshold used to detect up-crossings and mask suprathreshold samples.
    spike_window_pre_ms : float
        Duration excluded before each crossing.
    spike_window_post_ms : float
        Duration excluded after each crossing.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape as ``targets``.
    """
    targets = np.asarray(targets)
    if targets.ndim != 2:
        raise ValueError(f"targets must be two-dimensional, got shape {targets.shape}.")
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}.")
    if spike_window_pre_ms < 0 or spike_window_post_ms < 0:
        raise ValueError("Spike-window durations must be non-negative.")
    mask = targets < spike_threshold_mv
    pre_steps = int(np.ceil(spike_window_pre_ms / dt_ms))
    post_steps = int(np.ceil(spike_window_post_ms / dt_ms))
    for trace_index, voltage in enumerate(targets):
        crossings = np.flatnonzero((voltage[1:] >= spike_threshold_mv) & (voltage[:-1] < spike_threshold_mv)) + 1
        for crossing in crossings:
            start = max(0, int(crossing) - pre_steps)
            stop = min(voltage.size, int(crossing) + post_steps + 1)
            mask[trace_index, start:stop] = False
    return mask


def batches(
    inputs: Any,
    targets: Any,
    fit_masks: Any,
    batch_size: int,
    indices: Any | None = None,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Yield JAX training batches using the source trainer's ordering."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    order = np.arange(len(inputs)) if indices is None else np.asarray(indices)
    for start in range(0, len(order), batch_size):
        selected = order[start : start + batch_size]
        yield jnp.asarray(inputs[selected]), jnp.asarray(targets[selected]), jnp.asarray(fit_masks[selected])


def match_spike_times(
    predicted_ms: Any,
    target_ms: Any,
    *,
    window_ms: float = 10.0,
) -> tuple[tuple[float, float], ...]:
    """Match sorted predicted and target spikes one-to-one within a time window."""
    if window_ms < 0 or not np.isfinite(window_ms):
        raise ValueError(f"window_ms must be finite and non-negative, got {window_ms}.")
    predicted = np.sort(np.asarray(predicted_ms, dtype=float).reshape(-1))
    target = np.sort(np.asarray(target_ms, dtype=float).reshape(-1))
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Spike times must be finite.")
    matches = []
    predicted_index = target_index = 0
    while predicted_index < len(predicted) and target_index < len(target):
        difference = predicted[predicted_index] - target[target_index]
        tolerance = 8.0 * np.finfo(np.float32).eps * max(
            1.0,
            abs(predicted[predicted_index]),
            abs(target[target_index]),
            abs(window_ms),
        )
        boundary = window_ms + tolerance
        if abs(difference) <= boundary:
            matches.append((float(predicted[predicted_index]), float(target[target_index])))
            predicted_index += 1
            target_index += 1
        elif difference < -boundary:
            predicted_index += 1
        else:
            target_index += 1
    return tuple(matches)


def compute_spike_metrics(
    predicted_ms: Any,
    target_ms: Any,
    *,
    window_ms: float = 10.0,
) -> dict[str, Any]:
    """Compute one-to-one spike Precision, Recall, and F1 metrics."""
    predicted_rows = _spike_rows(predicted_ms)
    target_rows = _spike_rows(target_ms)
    if len(predicted_rows) != len(target_rows):
        raise ValueError("Predicted and target spike collections must contain the same number of traces.")
    matches = tuple(
        match_spike_times(predicted, target, window_ms=window_ms)
        for predicted, target in zip(predicted_rows, target_rows)
    )
    true_positive = sum(len(row) for row in matches)
    predicted_count = sum(len(row) for row in predicted_rows)
    target_count = sum(len(row) for row in target_rows)
    false_positive = predicted_count - true_positive
    false_negative = target_count - true_positive
    precision = true_positive / predicted_count if predicted_count else float("nan")
    recall = true_positive / target_count if target_count else float("nan")
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "match_window_ms": float(window_ms),
        "matches": matches,
    }


def _spike_rows(value: Any) -> tuple[np.ndarray, ...]:
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return (value,)
    if isinstance(value, (list, tuple)):
        if not value:
            return (np.asarray([], dtype=float),)
        if all(np.asarray(item).ndim == 0 for item in value):
            return (np.asarray(value),)
        return tuple(np.asarray(row).reshape(-1) for row in value)
    array = np.asarray(value)
    if array.ndim == 1:
        return (array,)
    if array.ndim == 2:
        return tuple(row[np.isfinite(row)] for row in array)
    raise ValueError("Spike times must be a one-dimensional trace or a sequence of traces.")


def evaluate_predictions(
    predictions: Any,
    targets: Any,
    *,
    mask: Any | None = None,
    predicted_spikes: Any | None = None,
    target_spikes: Any | None = None,
    match_window_ms: float = 10.0,
) -> dict[str, Any]:
    """Combine subthreshold voltage and optional spike timing metrics."""
    result = {name: float(np.asarray(value)) for name, value in compute_metrics(predictions, targets, mask).items()}
    if (predicted_spikes is None) != (target_spikes is None):
        raise ValueError("predicted_spikes and target_spikes must be provided together.")
    if predicted_spikes is not None:
        result.update(compute_spike_metrics(predicted_spikes, target_spikes, window_ms=match_window_ms))
    return result


@jax.jit
def _gif_spikes_for_candidates(drive: Any, candidates: Any, dt_ms: Any) -> jax.Array:
    """Evaluate one batch of ``(threshold, threshold increment/tau)`` candidates."""
    drive = jnp.asarray(drive)
    candidates = jnp.asarray(candidates, dtype=drive.dtype)

    def evaluate(candidate):
        params = {
            "v_th": candidate[0],
            "threshold_increment_mv": candidate[1],
            "threshold_tau_ms": candidate[2],
            "dt_ms": dt_ms,
        }
        return _apply_gif_closed_loop(drive, params)[2]

    return jax.vmap(evaluate)(candidates)


def _candidate_values(values: Any, *, name: str, non_negative: bool = False, positive: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite collection.")
    if non_negative and np.any(result < 0):
        raise ValueError(f"{name} must be non-negative.")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive.")
    return result


def _spike_score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    def finite_or_negative_infinity(value: Any) -> float:
        value = float(value)
        return value if np.isfinite(value) else -np.inf

    errors = [predicted - target for row in metrics["matches"] for predicted, target in row]
    timing_mae = float(np.mean(np.abs(errors))) if errors else np.inf
    return (
        finite_or_negative_infinity(metrics["f1"]),
        finite_or_negative_infinity(metrics["recall"]),
        finite_or_negative_infinity(metrics["precision"]),
        -timing_mae,
    )


def spike_time_alignment(metrics: Mapping[str, Any]) -> tuple[float, int, float | None]:
    """Return median signed latency, match count, and raw timing MAE."""
    if not isinstance(metrics, Mapping) or "matches" not in metrics:
        raise TypeError("metrics must contain one-to-one spike matches.")
    errors = np.asarray(
        [predicted - target for row in metrics["matches"] for predicted, target in row],
        dtype=float,
    )
    if not np.isfinite(errors).all():
        raise ValueError("Matched spike timing errors must be finite.")
    if errors.size == 0:
        return 0.0, 0, None
    return float(np.median(errors)), int(errors.size), float(np.mean(np.abs(errors)))


def _search_gif_candidates(
    drive: Any,
    target_spikes: Any,
    candidates: np.ndarray,
    *,
    dt_ms: float,
    match_window_ms: float,
    candidate_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    best = None
    for start in range(0, len(candidates), candidate_batch_size):
        batch = candidates[start : start + candidate_batch_size]
        spike_batch = np.asarray(_gif_spikes_for_candidates(drive, batch, dt_ms))
        for candidate, spike_array in zip(batch, spike_batch):
            predicted = tuple(np.flatnonzero(row) * dt_ms for row in spike_array)
            metrics = compute_spike_metrics(predicted, target_spikes, window_ms=match_window_ms)
            score = _spike_score(metrics)
            if best is None or score > best[0]:
                best = (score, candidate.copy(), metrics)
    return best[1], best[2]


class DBNNTrainer:
    """Train, validate, and evaluate a :class:`DBNN` model."""

    def __init__(
        self,
        model: DBNN,
        *,
        learning_rate: float = 1e-3,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        optimizer: Any | None = None,
        loss: str = "masked_mse",
        gradient_clip: float | None = 1.0,
        seed: int = 42,
        initialization_search: bool = True,
        initialization_options: Mapping[str, Any] | None = None,
    ):
        if not isinstance(model, DBNN) or isinstance(model, DBNNGIF):
            raise TypeError(f"model must be a DBNN, got {type(model).__name__}.")
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(lr_step_size, int) or isinstance(lr_step_size, bool) or lr_step_size < 0:
            raise ValueError("lr_step_size must be a non-negative integer.")
        if lr_gamma <= 0 or not np.isfinite(lr_gamma):
            raise ValueError("lr_gamma must be finite and positive.")
        if loss not in {"mse", "masked_mse"}:
            raise ValueError(f"loss must be 'mse' or 'masked_mse', got {loss!r}.")
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None.")
        self.model = model
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.lr_step_size = lr_step_size
        self.lr_gamma = float(lr_gamma)
        self.gradient_clip = gradient_clip
        self.seed = int(seed)
        if not isinstance(initialization_search, bool):
            raise TypeError("initialization_search must be a boolean.")
        options = dict(_INITIALIZATION_DEFAULTS)
        supplied_options = dict(initialization_options or {})
        unknown_options = set(supplied_options).difference(options)
        if unknown_options:
            raise KeyError(f"Unknown initialization options: {sorted(unknown_options)}.")
        options.update(supplied_options)
        self.initialization_search = initialization_search
        self.initialization_options = options
        self.initialization_completed = False
        self.initialization_report = None
        self.source_fingerprint = getattr(model, "source_fingerprint", None)
        self.input_alignment = getattr(model, "input_alignment", None)
        self.epoch = 0
        self.best_validation_loss = float("inf")
        self._best_params = None
        self._best_optimizer_state = None
        self.param_states = {
            path: state
            for path, state in model.states(brainstate.ParamState).items()
            if path[-1] in {"tau_rise", "tau_decay", "omega", "quadratic_weight_upper"}
        }
        if optimizer is not None and lr_step_size:
            raise ValueError("Set lr_step_size=0 when supplying a custom optimizer.")
        self.lr_scheduler = (
            braintools.optim.StepLR(base_lr=learning_rate, step_size=lr_step_size, gamma=lr_gamma)
            if optimizer is None and lr_step_size
            else None
        )
        self.optimizer = optimizer or braintools.optim.Adam(
            lr=self.lr_scheduler or learning_rate,
            grad_clip_norm=gradient_clip,
        )
        self.optimizer.register_trainable_weights(self.param_states)
        self._compiled_train_step = brainstate.transform.jit(self._train_step_impl)

    def load_data(self, path: str | Path, *, input_alignment: str | None = None) -> dict[str, Any]:
        """Load and validate a sparse dataset bound to the model layout."""
        batch = load_dataset(path)
        alignment = batch.metadata.get("input_alignment")
        if input_alignment is not None and alignment != input_alignment:
            raise ValueError(f"Dataset alignment {alignment!r} does not match requested {input_alignment!r}.")
        source_fingerprint = batch.metadata.get("source_fingerprint")
        if self.source_fingerprint is None:
            self.source_fingerprint = source_fingerprint
        validate_dataset(
            batch,
            self.model.layout,
            dt_ms=self.model.dt_ms,
            source_fingerprint=self.source_fingerprint,
        )
        if self.input_alignment is None:
            self.input_alignment = alignment
        elif alignment != self.input_alignment:
            raise ValueError(
                f"Dataset alignment {alignment!r} is incompatible with trainer alignment {self.input_alignment!r}."
            )
        self.model.source_fingerprint = self.source_fingerprint
        self.model.input_alignment = self.input_alignment
        inputs = rasterize_events(batch.plan, dt_ms=self.model.dt_ms, input_alignment=alignment)
        inputs = self.model.validate_inputs(inputs)
        return {
            "inputs": inputs,
            "targets": jnp.asarray(batch.voltage_mv),
            "spike_times_ms": batch.spike_times_ms,
            "split": batch.plan.split,
            "metadata": batch.metadata,
        }

    def split_data(
        self,
        data: dict[str, Any],
        *,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
    ) -> dict[str, dict[str, Any]]:
        """Reproducibly split imported legacy traces without leakage."""
        if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
            raise ValueError("Split fractions must be positive and leave a non-empty test fraction.")
        n_traces = len(data["inputs"])
        order = np.asarray(brainstate.random.RandomState(self.seed).permutation(n_traces))
        train_end = int(np.floor(n_traces * train_fraction))
        validation_end = train_end + int(np.floor(n_traces * validation_fraction))
        if train_end == 0 or validation_end == train_end or validation_end == n_traces:
            raise ValueError("Dataset is too small for the requested non-empty splits.")
        result = {}
        for name, selected in zip(
            ("train", "validation", "test"),
            (order[:train_end], order[train_end:validation_end], order[validation_end:]),
        ):
            split = {}
            for key, value in data.items():
                if isinstance(value, tuple) and len(value) == n_traces:
                    split[key] = tuple(value[int(index)] for index in selected)
                elif isinstance(value, list) and len(value) == n_traces:
                    split[key] = [value[int(index)] for index in selected]
                elif hasattr(value, "shape") and np.ndim(value) > 0 and value.shape[0] == n_traces:
                    split[key] = value[selected]
                else:
                    split[key] = value
            result[name] = split
            result[name]["trace_indices"] = selected
        return result

    def build_fit_mask(self, targets: Any, **kwargs) -> np.ndarray:
        """Build a spike-neighborhood fit mask using the model sampling interval."""
        return build_fit_mask(targets, self.model.dt_ms, **kwargs)

    def compute_loss(
        self,
        params: dict[str, Any],
        inputs: Any,
        targets: Any,
        *,
        mask: Any | None = None,
    ) -> jax.Array:
        """Compute differentiable open-loop voltage loss from explicit parameters."""
        predictions = dbnn_forward(params, inputs, channel_signs=self.model.channel_signs)
        if self.loss == "mse":
            return jnp.mean(jnp.square(predictions - targets))
        return masked_mse(predictions, targets, mask)

    def train_step(self, inputs: Any, targets: Any, *, mask: Any | None = None) -> dict[str, jax.Array]:
        """Apply one Braintools optimizer step and clip constrained parameters."""
        inputs = self.model.validate_inputs(inputs)
        if mask is not None and not bool(np.any(np.asarray(mask))):
            raise ValueError("masked_mse requires at least one valid sample.")
        if mask is None:
            mask = jnp.ones_like(targets, dtype=bool)
        return self._compiled_train_step(inputs, targets, mask)

    def _train_step_impl(self, inputs: Any, targets: Any, mask: Any) -> dict[str, jax.Array]:
        def loss_fn():
            return self.compute_loss(self.model.get_params(), inputs, targets, mask=mask)

        gradients, loss = brainstate.transform.grad(
            loss_fn,
            grad_states=self.param_states,
            return_value=True,
        )()
        self.optimizer.step(gradients)
        self.model.tau_rise.value = jnp.clip(self.model.tau_rise.value, 0.1, 100.0)
        self.model.tau_decay.value = jnp.clip(self.model.tau_decay.value, 0.1, 200.0)
        self.model.omega.value = jnp.clip(self.model.omega.value, 0.0, 10.0)
        leaves = [jnp.asarray(value) for value in jax.tree_util.tree_leaves(gradients)]
        gradient_norm = jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves))
        return {"loss": loss, "gradient_norm": gradient_norm}

    def search_initialization(
        self,
        data: dict[str, Any],
        **overrides: Any,
    ) -> dict[str, Any]:
        """Search shared kernel parameters and fit packed bilinear weights with ridge regression."""
        start_time = time.perf_counter()
        if self.epoch != 0:
            raise RuntimeError("Initialization search must run before the first training epoch.")
        options = dict(self.initialization_options)
        unknown = set(overrides).difference(options)
        if unknown:
            raise KeyError(f"Unknown initialization options: {sorted(unknown)}.")
        options.update(overrides)
        backend = options["backend"]
        if backend not in {"auto", "nevergrad", "scipy"}:
            raise ValueError("Initialization backend must be 'auto', 'nevergrad', or 'scipy'.")
        for name in ("samples", "time_stride", "maxiter", "popsize"):
            if not isinstance(options[name], int) or isinstance(options[name], bool) or options[name] < 1:
                raise ValueError(f"Initialization {name} must be a positive integer.")
        if (
            not isinstance(options["max_pairs"], int)
            or isinstance(options["max_pairs"], bool)
            or options["max_pairs"] < 0
        ):
            raise ValueError("Initialization max_pairs must be a non-negative integer.")
        if not np.isfinite(options["ridge"]) or options["ridge"] <= 0:
            raise ValueError("Initialization ridge must be finite and positive.")
        bounds = np.asarray(options["bounds"], dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Initialization bounds must contain three finite increasing (low, high) pairs.")

        inputs = self.model.validate_inputs(data["inputs"])
        targets = np.asarray(data["targets"], dtype=np.float32)
        if targets.shape != (inputs.shape[0], inputs.shape[2]):
            raise ValueError(f"Initialization targets must have shape {(inputs.shape[0], inputs.shape[2])}.")
        fit_mask = np.asarray(data.get("mask", np.ones_like(targets, dtype=bool)), dtype=bool)
        if fit_mask.shape != targets.shape or not np.any(fit_mask):
            raise ValueError("Initialization mask must match targets and contain at least one valid sample.")
        sample_count = min(options["samples"], inputs.shape[0])
        sample_indices = np.linspace(0, inputs.shape[0] - 1, sample_count).round().astype(np.int64)
        calibration_inputs = inputs[sample_indices]
        calibration_targets = targets[sample_indices, :: options["time_stride"]]
        calibration_mask = fit_mask[sample_indices, :: options["time_stride"]]
        n_channels = self.model.n_channels
        pair_i, pair_j = np.triu_indices(n_channels, k=1)
        pair_count = len(pair_i)
        if options["max_pairs"] and pair_count > options["max_pairs"]:
            selected_pairs = np.unique(np.linspace(0, pair_count - 1, options["max_pairs"]).round().astype(np.int64))
        else:
            selected_pairs = np.arange(pair_count, dtype=np.int64)
        selected_i = pair_i[selected_pairs]
        selected_j = pair_j[selected_pairs]
        steps = inputs.shape[2]
        fft_size = 1 << (2 * steps - 2).bit_length()
        calibration_fft = jnp.fft.rfft(calibration_inputs, n=fft_size, axis=2)
        time_ms = jnp.arange(steps, dtype=inputs.dtype) * self.model.dt_ms
        channel_signs = self.model.channel_signs

        @jax.jit
        def filtered_for_candidate(candidate):
            tau_rise, tau_decay, omega = candidate
            kernel = omega * (1.0 - jnp.exp(-time_ms / tau_rise)) * jnp.exp(-time_ms / tau_decay)
            kernel_fft = jnp.fft.rfft(kernel, n=fft_size)
            filtered = jnp.fft.irfft(
                calibration_fft * kernel_fft[None, None, :],
                n=fft_size,
                axis=2,
            )
            filtered = filtered[:, :, : steps : options["time_stride"]]
            return jnp.transpose(filtered, (0, 2, 1)) * channel_signs[None, None, :]

        def fit_candidate(candidate):
            filtered = np.asarray(filtered_for_candidate(jnp.asarray(candidate, dtype=inputs.dtype)))
            linear = np.sum(filtered, axis=2)
            residual = (calibration_targets - linear - float(self.model.bias.value))[calibration_mask]
            if len(selected_pairs):
                features = (filtered[:, :, selected_i] * filtered[:, :, selected_j])[calibration_mask]
                if features.shape[1] <= features.shape[0]:
                    gram = features.T @ features
                    gram.flat[:: gram.shape[0] + 1] += options["ridge"]
                    coefficients = np.linalg.solve(gram, features.T @ residual)
                else:
                    dual = features @ features.T
                    dual.flat[:: dual.shape[0] + 1] += options["ridge"]
                    coefficients = features.T @ np.linalg.solve(dual, residual)
                error = residual - features @ coefficients
            else:
                coefficients = np.empty((0,), dtype=np.float32)
                error = residual
            loss = float(np.mean(np.square(error)))
            if not np.isfinite(loss):
                loss = float(np.finfo(np.float32).max)
            return loss, np.asarray(coefficients, dtype=np.float32)

        default_candidate = np.asarray([5.0, 20.0, 2.0], dtype=np.float64)
        default_loss, default_coefficients = fit_candidate(default_candidate)
        nevergrad_available = importlib.util.find_spec("nevergrad") is not None
        if backend == "auto":
            backend = "nevergrad" if nevergrad_available else "scipy"
        if backend == "nevergrad":
            if not nevergrad_available:
                raise ImportError("Nevergrad initialization requires the optional 'nevergrad' package.")
            from braintools.optim import NevergradOptimizer

            def batched_objective(tau_rise, tau_decay, omega):
                candidates = zip(np.asarray(tau_rise), np.asarray(tau_decay), np.asarray(omega))
                return jnp.asarray([fit_candidate(candidate)[0] for candidate in candidates])

            optimizer = NevergradOptimizer(
                batched_loss_fun=batched_objective,
                bounds=tuple(map(tuple, bounds)),
                n_sample=options["popsize"],
                method=options["method"],
                budget=options["maxiter"] * options["popsize"],
            )
            optimizer.parametrization.random_state.seed(self.seed)
            searched_candidate = np.asarray(optimizer.minimize(n_iter=options["maxiter"]), dtype=np.float64)
            evaluations = len(optimizer.errors)
            method = f"braintools_nevergrad_{options['method']}_with_ridge"
        else:
            from scipy.optimize import differential_evolution

            result = differential_evolution(
                lambda candidate: fit_candidate(candidate)[0],
                bounds=tuple(map(tuple, bounds)),
                maxiter=options["maxiter"],
                popsize=options["popsize"],
                seed=self.seed,
                polish=True,
                workers=1,
                updating="immediate",
            )
            searched_candidate = np.asarray(result.x, dtype=np.float64)
            evaluations = int(result.nfev)
            method = "scipy_differential_evolution_with_ridge"
        searched_loss, searched_coefficients = fit_candidate(searched_candidate)
        if searched_loss < default_loss:
            selected_candidate = searched_candidate
            selected_loss = searched_loss
            selected_coefficients = searched_coefficients
            selected_source = "searched"
        else:
            selected_candidate = default_candidate
            selected_loss = default_loss
            selected_coefficients = default_coefficients
            selected_source = "default"

        upper_weights = np.zeros((pair_count,), dtype=np.float32)
        upper_weights[selected_pairs] = selected_coefficients
        params = self.model.get_params()
        params.update(
            tau_rise=jnp.full((n_channels,), selected_candidate[0], dtype=jnp.float32),
            tau_decay=jnp.full((n_channels,), selected_candidate[1], dtype=jnp.float32),
            omega=jnp.full((n_channels,), selected_candidate[2], dtype=jnp.float32),
            quadratic_weight_upper=jnp.asarray(upper_weights),
        )
        self.model.set_params(params)
        report = {
            "method": method,
            "backend": backend,
            "selected_source": selected_source,
            "calibration_samples": sample_count,
            "calibration_time_stride": options["time_stride"],
            "fitted_pairs": int(len(selected_pairs)),
            "total_pairs": int(pair_count),
            "ridge": float(options["ridge"]),
            "tau_rise": float(selected_candidate[0]),
            "tau_decay": float(selected_candidate[1]),
            "omega": float(selected_candidate[2]),
            "calibration_mse": selected_loss,
            "default_calibration_mse": default_loss,
            "searched_calibration_mse": searched_loss,
            "evaluations": evaluations,
            "search_seconds": time.perf_counter() - start_time,
        }
        self.initialization_completed = True
        self.initialization_report = report
        return report

    def fit(
        self,
        train_data: dict[str, Any],
        *,
        validation_data: dict[str, Any] | None = None,
        epochs: int,
        batch_size: int,
        shuffle: bool = True,
        checkpoint_path: str | Path | None = None,
        patience: int | None = None,
    ) -> dict[str, list[float]]:
        """Train by trace batches with validation model selection and early stopping."""
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.initialization_search and not self.initialization_completed:
            self.search_initialization(train_data)
        history = {"train_loss": [], "validation_loss": [], "learning_rates": []}
        stale_epochs = 0
        for _ in range(epochs):
            n_traces = len(train_data["inputs"])
            order = np.arange(n_traces)
            if shuffle:
                order = np.asarray(brainstate.random.RandomState(self.seed + self.epoch).permutation(order))
            learning_rate = (
                float(self.lr_scheduler.current_lrs.value[0]) if self.lr_scheduler is not None else self.learning_rate
            )
            history["learning_rates"].append(learning_rate)
            weighted_loss = 0.0
            seen = 0
            masks = train_data.get("mask", np.ones_like(train_data["targets"], dtype=bool))
            for batch_inputs, batch_targets, batch_mask in batches(
                train_data["inputs"], train_data["targets"], masks, batch_size, order
            ):
                loss = float(np.asarray(self.train_step(batch_inputs, batch_targets, mask=batch_mask)["loss"]))
                weighted_loss += loss * len(batch_inputs)
                seen += len(batch_inputs)
            history["train_loss"].append(weighted_loss / seen)
            self.epoch += 1
            improved = False
            should_stop = False
            if validation_data is not None:
                validation = self.evaluate(validation_data)
                validation_loss = validation["mse"]
                history["validation_loss"].append(validation_loss)
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    self._best_params = {name: np.array(value) for name, value in self.model.get_params().items()}
                    stale_epochs = 0
                    improved = True
                else:
                    stale_epochs += 1
                    should_stop = patience is not None and stale_epochs >= patience
            if self.lr_scheduler is not None:
                self.lr_scheduler.step_epoch()
            if improved:
                self._best_optimizer_state = self._snapshot_optimizer_state()
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)
            if should_stop:
                break
        if validation_data is not None:
            self._restore_best_snapshot()
        return history

    def _snapshot_optimizer_state(self) -> dict[Any, Any]:
        return {
            path: jax.tree_util.tree_map(lambda value: jnp.array(value), state.value)
            for path, state in brainstate.graph.states(self.optimizer).items()
            if not isinstance(state, brainstate.ParamState)
        }

    def _restore_best_snapshot(self) -> None:
        if self._best_params is None:
            return
        self.model.set_params(self._best_params)
        current_states = brainstate.graph.states(self.optimizer)
        for path, value in self._best_optimizer_state.items():
            current_states[path].value = jax.tree_util.tree_map(lambda item: jnp.array(item), value)

    def calibrate_gif(
        self,
        validation_data: dict[str, Any],
        *,
        thresholds_mv: Any | None = None,
        threshold_increments_mv: Any = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
        threshold_taus_ms: Any = (10.0, 30.0, 50.0, 100.0),
        match_window_ms: float = 10.0,
        candidate_batch_size: int = 128,
    ) -> tuple[DBNNGIF, dict[str, Any]]:
        """Freeze the trained DBNN and calibrate a new GIF readout on validation spikes."""
        target_spikes = validation_data.get("spike_times_ms")
        if target_spikes is None:
            raise ValueError("DBNN-GIF calibration requires validation spike_times_ms.")
        target_rows = _spike_rows(target_spikes)
        if not any(len(row) for row in target_rows):
            raise ValueError("DBNN-GIF calibration requires at least one validation spike.")
        if candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive.")
        if match_window_ms < 0 or not np.isfinite(match_window_ms):
            raise ValueError("match_window_ms must be finite and non-negative.")
        inputs = self.model.validate_inputs(validation_data["inputs"])
        drive = self.model.predict(inputs, mode=self.model.mode)["voltage"]
        host_drive = np.asarray(drive)
        if thresholds_mv is None:
            thresholds_mv = np.linspace(np.percentile(host_drive, 1), np.percentile(host_drive, 99), 64)
        thresholds = _candidate_values(thresholds_mv, name="thresholds_mv")
        threshold_increments = _candidate_values(
            threshold_increments_mv, name="threshold_increments_mv", non_negative=True
        )
        threshold_taus = _candidate_values(threshold_taus_ms, name="threshold_taus_ms", positive=True)

        gif_candidates = np.asarray(
            [
                (threshold, increment, tau)
                for threshold, increment, tau in product(thresholds, threshold_increments, threshold_taus)
            ],
            dtype=np.float32,
        )
        best_gif, gif_metrics = _search_gif_candidates(
            drive,
            target_rows,
            gif_candidates,
            dt_ms=self.model.dt_ms,
            match_window_ms=match_window_ms,
            candidate_batch_size=candidate_batch_size,
        )

        model = DBNNGIF.from_dbnn(self.model)
        params = model.get_params()
        params.update(
            v_th=best_gif[0],
            threshold_increment_mv=best_gif[1],
            threshold_tau_ms=best_gif[2],
        )
        model.set_params(params)
        spike_time_offset_ms, matched_count, raw_timing_mae_ms = spike_time_alignment(gif_metrics)
        model.set_spike_time_offset(spike_time_offset_ms)
        raw_spikes = tuple(
            np.flatnonzero(row) * self.model.dt_ms for row in np.asarray(model.predict(inputs)["spike"])
        )
        duration_ms = (inputs.shape[-1] - 1) * self.model.dt_ms
        aligned_spikes = align_spike_times(
            raw_spikes,
            offset_ms=spike_time_offset_ms,
            duration_ms=duration_ms,
        )
        aligned_metrics = compute_spike_metrics(
            aligned_spikes,
            target_rows,
            window_ms=match_window_ms,
        )
        _, aligned_matched_count, aligned_timing_mae_ms = spike_time_alignment(aligned_metrics)
        report = {
            "gif": {
                "candidate_count": len(gif_candidates),
                "threshold_mv": float(best_gif[0]),
                "threshold_increment_mv": float(best_gif[1]),
                "threshold_tau_ms": float(best_gif[2]),
                "spike_time_offset_ms": spike_time_offset_ms,
                "matched_count": matched_count,
                "aligned_matched_count": aligned_matched_count,
                "raw_timing_mae_ms": raw_timing_mae_ms,
                "aligned_timing_mae_ms": aligned_timing_mae_ms,
                "raw_validation_metrics": gif_metrics,
                "validation_metrics": aligned_metrics,
            },
        }
        return model, report

    def calibrate_threshold(
        self,
        validation_data: dict[str, Any],
        *,
        candidates_mv: Any | None = None,
        match_window_ms: float = 10.0,
    ) -> dict[str, float]:
        """Select the validation threshold that maximizes one-to-one spike F1."""
        target_spikes = validation_data.get("spike_times_ms")
        if target_spikes is None:
            raise ValueError("Threshold calibration requires validation spike_times_ms.")
        voltage = np.asarray(self.model.predict(validation_data["inputs"])["voltage"])
        if candidates_mv is None:
            candidates = np.linspace(np.percentile(voltage, 1), np.percentile(voltage, 99), 64)
        else:
            candidates = np.asarray(candidates_mv, dtype=float).reshape(-1)
        if not len(candidates) or not np.isfinite(candidates).all():
            raise ValueError("Threshold candidates must be a non-empty finite collection.")
        best = None
        for threshold in candidates:
            above = voltage >= threshold
            predicted = tuple((np.flatnonzero(row[1:] & ~row[:-1]) + 1) * self.model.dt_ms for row in above)
            metrics = compute_spike_metrics(predicted, target_spikes, window_ms=match_window_ms)
            score = metrics["f1"]
            ranking = -np.inf if np.isnan(score) else score
            if best is None or ranking > best[0]:
                best = (ranking, float(threshold), metrics)
        self.model.v_th.value = jnp.asarray(best[1], dtype=jnp.float32)
        return {"threshold_mv": best[1], "f1": best[2]["f1"]}

    def evaluate(self, data: dict[str, Any], *, mask: Any | None = None) -> dict[str, Any]:
        """Evaluate voltage and optional one-to-one spike timing metrics."""
        selected_mask = data.get("mask") if mask is None else mask
        prediction = self.model.predict(data["inputs"])
        predicted_spikes = None
        target_spikes = data.get("spike_times_ms")
        if target_spikes is not None:
            predicted_spikes = tuple(
                np.flatnonzero(np.asarray(row)) * self.model.dt_ms for row in np.asarray(prediction["spike"])
            )
        return evaluate_predictions(
            prediction["voltage"],
            data["targets"],
            mask=selected_mask,
            predicted_spikes=predicted_spikes,
            target_spikes=target_spikes,
        )

    def predict(self, inputs: Any) -> dict[str, jax.Array]:
        """Delegate inference to the owned DBNN model."""
        return self.model.predict(inputs)

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        """Save model, optimizer, and training progress."""
        from braincell.reduction.dbnn.asset import save_training_checkpoint

        save_training_checkpoint(path, self, metadata=metadata)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, and training progress."""
        from braincell.reduction.dbnn.asset import load_training_checkpoint

        load_training_checkpoint(path, self)


__all__ = [
    "batches",
    "build_fit_mask",
    "compute_spike_metrics",
    "compute_metrics",
    "dbnn_forward",
    "match_spike_times",
    "spike_time_alignment",
    "masked_mse",
    "DBNNTrainer",
    "evaluate_predictions",
]
