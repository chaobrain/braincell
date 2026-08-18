#!/usr/bin/env python3
"""Run three Composite-loss ablations for heterogeneous nine-parameter fitting.

The experiment contract is documented in
``docs/specs/2026-08-18-nine-parameter-composite-loss-ablation.md``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

import conductance_learning_core as core
import heterogeneous_nine_parameter_training as baseline
import heterogeneous_protocol_dataset as dataset


COMPONENT_NAMES = core.COMPONENT_NAMES
COMPONENT_FLOORS = np.asarray([0.1, 1e-3, 0.1, 1e-4, 1e-3, 1.0], dtype=float)
PEAK_WINDOW_MS = (20.0, 100.0)
PEAK_START_INDEX = int(round(PEAK_WINDOW_MS[0] / dataset.DT_MS))
DEFAULT_OUTPUT_ROOT = dataset.ARTIFACT_ROOT


@dataclass(frozen=True)
class LossConfiguration:
    """Describe one fixed Composite component-weight vector."""

    name: str
    weights: np.ndarray
    output_name: str


def validate_component_weights(weights) -> np.ndarray:
    """Return a validated six-component nonnegative weight vector."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (len(COMPONENT_NAMES),):
        raise ValueError(f"Expected {len(COMPONENT_NAMES)} component weights, got {values.shape!r}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Component weights must be finite and nonnegative.")
    if not np.any(values > 0.0):
        raise ValueError("At least one component weight must be positive.")
    return values


CONFIGURATIONS = {
    "voltage_count": LossConfiguration(
        name="voltage_count",
        weights=validate_component_weights([1.0, 0.0, 0.0, 0.0, 0.4, 0.0]),
        output_name="heterogeneous_nine_parameter_voltage_count",
    ),
    "without_count_composite": LossConfiguration(
        name="without_count_composite",
        weights=validate_component_weights([1.0, 0.1, 0.25, 0.75, 0.0, 2.0]),
        output_name="heterogeneous_nine_parameter_without_count_composite",
    ),
    "full_composite": LossConfiguration(
        name="full_composite",
        weights=validate_component_weights([1.0, 0.1, 0.25, 0.75, 0.4, 2.0]),
        output_name="heterogeneous_nine_parameter_full_composite",
    ),
}


def loss_configuration(name: str) -> LossConfiguration:
    """Return one approved loss configuration by name."""
    try:
        return CONFIGURATIONS[name]
    except KeyError as error:
        raise ValueError(f"Unknown loss configuration {name!r}; expected one of {tuple(CONFIGURATIONS)!r}.") from error


def output_directory(name: str) -> Path:
    """Return the dedicated default artifact directory for one configuration."""
    return DEFAULT_OUTPUT_ROOT / loss_configuration(name).output_name


def _filtered_smooth_events(soma_mv):
    time_major = jnp.moveaxis(soma_mv, 1, 0)
    filtered = core.exponential_event_filter(core.smooth_crossings(time_major))
    return jnp.moveaxis(filtered, 0, 1)


def _smooth_peak(soma_mv):
    beta = 0.2
    return jax.scipy.special.logsumexp(beta * soma_mv[:, PEAK_START_INDEX:], axis=1) / beta


def raw_loss_components(prediction_mv, target_mv, target_mask):
    """Return six unnormalized differentiable components for every protocol."""
    prediction = jnp.asarray(prediction_mv)
    target = jnp.asarray(target_mv)
    mask = jnp.asarray(target_mask)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"Expected matching (protocol, time, 3) traces, got {prediction.shape} and {target.shape}.")
    if mask.shape != prediction.shape[:2]:
        raise ValueError(f"Expected target mask {prediction.shape[:2]}, got {mask.shape}.")

    error = prediction - target
    weights = mask[..., None]
    voltage = jnp.sum(weights * core._huber(error), axis=(1, 2))
    voltage /= jnp.sum(mask, axis=1) * prediction.shape[2]

    predicted_dv = jnp.diff(prediction, axis=1)
    target_dv = jnp.diff(target, axis=1)
    derivative_mask = jnp.minimum(mask[:, :-1], mask[:, 1:])
    derivative = jnp.sum(derivative_mask[..., None] * core._huber(predicted_dv - target_dv, delta=0.5), axis=(1, 2))
    derivative /= jnp.sum(derivative_mask, axis=1) * prediction.shape[2]

    block = 20
    n_blocks = prediction.shape[1] // block
    predicted_coarse = prediction[:, : n_blocks * block].reshape(prediction.shape[0], n_blocks, block, 3).mean(axis=2)
    target_coarse = target[:, : n_blocks * block].reshape(target.shape[0], n_blocks, block, 3).mean(axis=2)
    multiscale = jnp.mean(core._huber(predicted_coarse - target_coarse), axis=(1, 2))

    predicted_soma = prediction[:, :, 0]
    target_soma = target[:, :, 0]
    predicted_crossings = core.smooth_crossings(jnp.moveaxis(predicted_soma, 1, 0))
    target_crossings = core.smooth_crossings(jnp.moveaxis(target_soma, 1, 0))
    predicted_events = jnp.moveaxis(core.exponential_event_filter(predicted_crossings), 0, 1)
    target_events = jnp.moveaxis(core.exponential_event_filter(target_crossings), 0, 1)
    event = jnp.mean((predicted_events - target_events) ** 2, axis=1)
    count = (jnp.sum(predicted_crossings, axis=0) - jnp.sum(target_crossings, axis=0)) ** 2
    peak = (_smooth_peak(predicted_soma) - _smooth_peak(target_soma)) ** 2
    return jnp.stack((voltage, derivative, multiscale, event, count, peak), axis=1)


def normalized_component_objective(raw_components, normalizers, component_weights):
    """Return one normalized weighted objective value per protocol."""
    raw = jnp.asarray(raw_components)
    scales = jnp.asarray(normalizers)
    weights = jnp.asarray(component_weights)
    return jnp.sum(weights * raw / scales, axis=-1) / jnp.sum(weights)


class CompositePlaybackProblem(baseline.PlaybackProblem):
    """Extend the batched playback problem with six differentiable losses."""

    def __init__(self, component_weights) -> None:
        super().__init__()
        self.component_weights = jnp.asarray(validate_component_weights(component_weights))
        self.normalizer = brainstate.State(jnp.ones((baseline.BATCH_SIZE, len(COMPONENT_NAMES))))

    def raw_loss_components(self, prediction_mv):
        """Return all raw loss components for the currently bound batch."""
        return raw_loss_components(prediction_mv, self.target.value, self.weights.value)

    def normalized_protocol_losses(self, prediction_mv):
        """Return the configured normalized objective for each protocol."""
        return normalized_component_objective(
            self.raw_loss_components(prediction_mv),
            self.normalizer.value,
            self.component_weights,
        )

    def loss_with_aux(self):
        """Return the configured batch objective and raw component diagnostics."""
        raw = self.raw_loss_components(self.simulate())
        protocol_losses = normalized_component_objective(raw, self.normalizer.value, self.component_weights)
        return jnp.mean(protocol_losses), raw


def compute_component_normalizers(
    data: baseline.LoadedDataset,
    masks: np.ndarray,
    component_weights,
) -> np.ndarray:
    """Evaluate fixed canonical scales for every protocol and component."""
    problem = CompositePlaybackProblem(component_weights)
    canonical_factors = np.tile(core.CANONICAL_INITIAL_PARAMETERS / core.TARGET_PARAMETERS, len(dataset.SITES))
    problem.set_physical_parameters(baseline.TARGET_PARAMETERS * canonical_factors)
    all_indices = jnp.arange(data.currents_na.shape[0], dtype=jnp.int32).reshape(-1, baseline.BATCH_SIZE)
    placeholder_scales = np.ones((data.currents_na.shape[0], len(COMPONENT_NAMES)))

    def evaluate(indices):
        problem.set_batch(data, indices, placeholder_scales, masks)
        return problem.raw_loss_components(problem.simulate())

    raw = brainstate.transform.for_loop(evaluate, all_indices)
    jax.block_until_ready(raw)
    values = np.asarray(raw).reshape(-1, len(COMPONENT_NAMES))
    return np.maximum(values, COMPONENT_FLOORS[None, :])


@contextmanager
def configured_objective(configuration: LossConfiguration):
    """Temporarily install one Composite objective in the baseline experiment."""
    weights = validate_component_weights(configuration.weights)

    class ConfiguredCompositePlaybackProblem(CompositePlaybackProblem):
        def __init__(self) -> None:
            super().__init__(weights)

    original_problem = baseline.PlaybackProblem
    original_normalizers = baseline.compute_normalizers
    baseline.PlaybackProblem = ConfiguredCompositePlaybackProblem
    baseline.compute_normalizers = lambda data, masks: compute_component_normalizers(data, masks, weights)
    try:
        yield
    finally:
        baseline.PlaybackProblem = original_problem
        baseline.compute_normalizers = original_normalizers


def _append_objective_metadata(
    paths: dict[str, Path],
    configuration: LossConfiguration,
) -> None:
    arrays_path = paths["arrays"]
    with np.load(arrays_path) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    arrays["component_weights"] = configuration.weights
    arrays["component_normalizer_floors"] = COMPONENT_FLOORS
    np.savez_compressed(arrays_path, **arrays)

    summary_path = paths["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["loss"] = f"normalized {configuration.name} Composite objective"
    summary["objective"] = {
        "configuration": configuration.name,
        "component_names": list(COMPONENT_NAMES),
        "component_weights": configuration.weights.tolist(),
        "component_normalizer_floors": COMPONENT_FLOORS.tolist(),
        "normalization": "per protocol and component at canonical initialization",
        "peak_window_ms": list(PEAK_WINDOW_MS),
        "smooth_crossing_temperature_mV": 2.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_configuration(
    data: baseline.LoadedDataset,
    configuration: LossConfiguration,
    *,
    n_epochs: int = baseline.DEFAULT_EPOCHS,
    learning_rate: float = 0.01,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Train, diagnose, and save one complete Composite configuration."""
    destination = output_directory(configuration.name) if output_dir is None else output_dir
    with configured_objective(configuration):
        result = baseline.run_training(data, n_epochs=n_epochs, learning_rate=learning_rate)
        landscape = baseline.compute_landscape(data, result)
        quality = baseline.compute_quality_diagnostics(data, result)
        paths = baseline.save_results(data, result, landscape, quality, destination)
    _append_objective_metadata(paths, configuration)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=tuple(CONFIGURATIONS), required=True)
    parser.add_argument("--dataset", type=Path, default=baseline.DATASET_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=baseline.DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run one requested CPU Composite-loss experiment."""
    args = parse_args(argv)
    if jax.default_backend() != "cpu":
        raise RuntimeError("Run this exploratory experiment with JAX_PLATFORMS=cpu.")
    data = baseline.load_dataset(args.dataset)
    configuration = loss_configuration(args.configuration)
    destination = output_directory(configuration.name) if args.output_dir is None else args.output_dir
    with brainstate.environ.context(dt=dataset.DT):
        paths = run_configuration(
            data,
            configuration,
            n_epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=destination,
        )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
