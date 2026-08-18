#!/usr/bin/env python3
"""Fit nine regional conductances to the heterogeneous protocol dataset.

The training, evaluation, and figure contracts are documented in
``docs/specs/2026-08-17-heterogeneous-nine-parameter-training.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

import braincell
from braincell import mech
from braincell.filter import AllRegion, BranchSlice, at

import conductance_learning_core as core
import heterogeneous_protocol_dataset as dataset


DATASET_PATH = dataset.DEFAULT_OUTPUT_DIR / "dataset.npz"
DEFAULT_OUTPUT_DIR = dataset.ARTIFACT_ROOT / "heterogeneous_nine_parameter_training"
PARAMETER_NAMES = tuple(f"{site}_{channel}" for site in dataset.SITES for channel in ("leak", "na", "k"))
LOWER_BOUNDS = np.tile(core.LOWER_BOUNDS, len(dataset.SITES))
UPPER_BOUNDS = np.tile(core.UPPER_BOUNDS, len(dataset.SITES))
TARGET_PARAMETERS = dataset.TARGET_PARAMETERS.reshape(-1).astype(float)
BATCH_SIZE = 18
BATCHES_PER_EPOCH = 6
START_CHUNK_SIZE = 2
DEFAULT_EPOCHS = 30
RANDOM_SEED = 20260817
HUBER_DELTA_MV = 2.0
SPIKE_WINDOW_WEIGHT = 0.1
SPIKE_TIMING_TOLERANCE_MS = 0.5


@dataclass(frozen=True)
class LoadedDataset:
    """Hold numeric dataset arrays and metadata."""

    time_ms: np.ndarray
    currents_na: np.ndarray
    voltages_mv: np.ndarray
    families: np.ndarray
    sites: np.ndarray
    splits: np.ndarray
    protocol_ids: np.ndarray

    def indices(self, split: str) -> np.ndarray:
        """Return row indices belonging to one split."""
        return np.flatnonzero(self.splits == split)


@dataclass(frozen=True)
class TrainingResult:
    """Store all eight fitted starts and their diagnostics."""

    starts: np.ndarray
    target_parameters: np.ndarray
    train_losses: np.ndarray
    validation_losses: np.ndarray
    gradients: np.ndarray
    parameter_trajectories: np.ndarray
    best_parameters: np.ndarray
    best_epochs: np.ndarray
    test_losses: np.ndarray
    initial_test_traces: np.ndarray
    fitted_test_traces: np.ndarray
    target_test_traces: np.ndarray
    target_test_spike_counts: np.ndarray
    fitted_test_spike_counts: np.ndarray
    schedule: np.ndarray
    normalizers: np.ndarray
    learning_rate: float
    random_seed: int
    wall_seconds: float


@dataclass(frozen=True)
class LandscapeResult:
    """Store the trajectory-informed plane and one-dimensional profiles."""

    axis_1: np.ndarray
    axis_2: np.ndarray
    x_values: np.ndarray
    y_values: np.ndarray
    plane_losses: np.ndarray
    plane_spike_mismatch: np.ndarray
    projected_trajectories: np.ndarray
    best_projection: np.ndarray
    profile_parameters: np.ndarray
    profile_losses: np.ndarray
    profile_spike_mismatch: np.ndarray


@dataclass(frozen=True)
class QualityDiagnostics:
    """Store parameter, voltage, and hard-spike quality metrics."""

    parameter_relative_rms: np.ndarray
    relative_parameter_step_rms: np.ndarray
    best_parameter_relative_rms: np.ndarray
    best_parameter_signed_relative_error: np.ndarray
    validation_voltage_rmse_mv: np.ndarray
    validation_spike_count_exact: np.ndarray
    validation_spike_timing_exact: np.ndarray
    test_voltage_rmse_mv: np.ndarray
    test_protocol_voltage_rmse_mv: np.ndarray
    test_spike_count_error: np.ndarray
    test_spike_count_exact: np.ndarray
    test_spike_timing_exact: np.ndarray
    test_spike_max_timing_error_ms: np.ndarray


def load_dataset(path: Path = DATASET_PATH) -> LoadedDataset:
    """Load and validate the saved heterogeneous protocol dataset."""
    with np.load(path) as values:
        loaded = LoadedDataset(
            time_ms=np.asarray(values["time_ms"]),
            currents_na=np.asarray(values["current_nA"]),
            voltages_mv=np.asarray(values["voltage_mV"]),
            families=np.asarray(values["family"]),
            sites=np.asarray(values["injection_site"]),
            splits=np.asarray(values["split"]),
            protocol_ids=np.asarray(values["protocol_id"]),
        )
    if loaded.currents_na.shape != (144, dataset.N_STEPS, 3):
        raise ValueError(f"Unexpected current array shape {loaded.currents_na.shape!r}.")
    if loaded.voltages_mv.shape != loaded.currents_na.shape:
        raise ValueError("Voltage and current arrays must have the same shape.")
    if tuple(loaded.time_ms.shape) != (dataset.N_STEPS,):
        raise ValueError(f"Unexpected time array shape {loaded.time_ms.shape!r}.")
    return loaded


def load_saved_training_result(output_dir: Path = DEFAULT_OUTPUT_DIR) -> TrainingResult:
    """Load a completed run for diagnostics-only artifact generation."""
    arrays_path = output_dir / "training_results.npz"
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as values:
        target_test_traces = np.asarray(values["target_test_traces"])
        fitted_test_traces = np.asarray(values["fitted_test_traces"])
        return TrainingResult(
            starts=np.asarray(values["starts"]),
            target_parameters=np.asarray(values["target_parameters"]),
            train_losses=np.asarray(values["train_losses"]),
            validation_losses=np.asarray(values["validation_losses"]),
            gradients=np.asarray(values["gradients"]),
            parameter_trajectories=np.asarray(values["parameter_trajectories"]),
            best_parameters=np.asarray(values["best_parameters"]),
            best_epochs=np.asarray(values["best_epochs"]),
            test_losses=np.asarray(values["test_losses"]),
            initial_test_traces=np.asarray(values["initial_test_traces"]),
            fitted_test_traces=fitted_test_traces,
            target_test_traces=target_test_traces,
            target_test_spike_counts=hard_spike_counts(target_test_traces),
            fitted_test_spike_counts=hard_spike_counts(fitted_test_traces),
            schedule=np.asarray(values["schedule"]),
            normalizers=np.asarray(values["normalizers"]),
            learning_rate=float(summary["learning_rate"]),
            random_seed=int(summary["random_seed"]),
            wall_seconds=float(summary["training_wall_seconds"]),
        )


def make_initial_grid() -> np.ndarray:
    """Return eight target-scaled starts in the nine-dimensional space."""
    starts = []
    for leak_factor in (0.5, 1.5):
        for sodium_factor in (0.5, 1.5):
            for potassium_factor in (0.5, 1.5):
                factors = np.tile((leak_factor, sodium_factor, potassium_factor), len(dataset.SITES))
                starts.append(TARGET_PARAMETERS * factors)
    return np.asarray(starts, dtype=float)


def make_batch_schedule(data: LoadedDataset, n_epochs: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Build deterministic, balanced, without-replacement minibatches."""
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs!r}.")
    rng = brainstate.random.RandomState(seed)
    schedules = []
    family_counts = {"dc": 3, "paired": 2, "sine": 1}
    for _ in range(n_epochs):
        batches = [[] for _ in range(BATCHES_PER_EPOCH)]
        for site in dataset.SITES:
            for family in dataset.FAMILIES:
                stratum = np.flatnonzero((data.splits == "train") & (data.sites == site) & (data.families == family))
                expected = BATCHES_PER_EPOCH * family_counts[family]
                if stratum.size != expected:
                    raise ValueError(f"Expected {expected} train rows for {site}/{family}, got {stratum.size}.")
                shuffled = np.asarray(rng.permutation(jnp.asarray(stratum)))
                grouped = shuffled.reshape(BATCHES_PER_EPOCH, family_counts[family])
                for batch_index in range(BATCHES_PER_EPOCH):
                    batches[batch_index].extend(grouped[batch_index].tolist())
        rows = []
        for batch in batches:
            permutation = np.asarray(rng.permutation(BATCH_SIZE))
            rows.append(np.asarray(batch, dtype=np.int32)[permutation])
        schedules.append(np.stack(rows))
    return np.stack(schedules).astype(np.int32, copy=False)


def target_weight_masks(voltages_mv: np.ndarray) -> np.ndarray:
    """Down-weight fixed target-spike windows for a protocol collection."""
    if voltages_mv.ndim != 3 or voltages_mv.shape[-1] != 3:
        raise ValueError(f"voltages_mv must have shape (protocol, time, 3), got {voltages_mv.shape!r}.")
    weights = np.ones(voltages_mv.shape[:2], dtype=np.float32)
    crossings = (voltages_mv[:, :-1, 0] < 0.0) & (voltages_mv[:, 1:, 0] >= 0.0)
    before = int(round(1.0 / dataset.DT_MS))
    after = int(round(3.0 / dataset.DT_MS))
    for protocol_index, local_crossings in enumerate(crossings):
        for crossing in np.flatnonzero(local_crossings) + 1:
            start = max(0, int(crossing) - before)
            stop = min(weights.shape[1], int(crossing) + after + 1)
            weights[protocol_index, start:stop] = SPIKE_WINDOW_WEIGHT
    return weights


def hard_spike_counts(voltages_mv: np.ndarray) -> np.ndarray:
    """Count soma upward zero crossings along the penultimate axis."""
    soma = np.asarray(voltages_mv)[..., 0]
    return np.sum((soma[..., :-1] < 0.0) & (soma[..., 1:] >= 0.0), axis=-1)


def relative_parameter_rms(parameters: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a scale-free RMS L2 distance to the target parameters."""
    values = np.asarray(parameters, dtype=float)
    target = np.asarray(target, dtype=float)
    if values.shape[-1:] != target.shape or target.ndim != 1:
        raise ValueError(f"Expected parameters (..., {target.size}) and target ({target.size},), got {values.shape}.")
    if np.any(target == 0.0):
        raise ValueError("Relative parameter distance requires nonzero target values.")
    return np.sqrt(np.mean(((values - target) / target) ** 2, axis=-1))


def relative_parameter_step_rms(parameters: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return target-scaled RMS movement between adjacent checkpoints."""
    values = np.asarray(parameters, dtype=float)
    target = np.asarray(target, dtype=float)
    if values.ndim < 2 or values.shape[-1:] != target.shape or target.ndim != 1:
        raise ValueError(
            f"Expected checkpoint parameters (..., checkpoint, {target.size}) and target ({target.size},), "
            f"got {values.shape}."
        )
    if np.any(target == 0.0):
        raise ValueError("Relative parameter movement requires nonzero target values.")
    return np.sqrt(np.mean((np.diff(values, axis=-2) / target) ** 2, axis=-1))


def signed_relative_parameter_error(parameters: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return signed elementwise parameter error relative to the target."""
    values = np.asarray(parameters, dtype=float)
    target = np.asarray(target, dtype=float)
    if values.shape[-1:] != target.shape or target.ndim != 1:
        raise ValueError(f"Expected parameters (..., {target.size}) and target ({target.size},), got {values.shape}.")
    if np.any(target == 0.0):
        raise ValueError("Relative parameter error requires nonzero target values.")
    return (values - target) / target


def protocol_voltage_rmse(prediction_mv: np.ndarray, target_mv: np.ndarray) -> np.ndarray:
    """Return all-probe voltage RMSE for each candidate and protocol."""
    prediction = np.asarray(prediction_mv, dtype=float)
    target = np.asarray(target_mv, dtype=float)
    if target.ndim != 3 or prediction.ndim < 3 or prediction.shape[-3:] != target.shape:
        raise ValueError(
            "prediction_mv must end in the target (protocol, time, probe) shape; "
            f"got prediction {prediction.shape!r} and target {target.shape!r}."
        )
    return np.sqrt(np.mean((prediction - target) ** 2, axis=(-2, -1)))


def hard_spike_protocol_quality(
    prediction_mv: np.ndarray,
    target_mv: np.ndarray,
    *,
    tolerance_ms: float = SPIKE_TIMING_TOLERANCE_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed count error, count success, and count-plus-timing success."""
    prediction = np.asarray(prediction_mv)
    target = np.asarray(target_mv)
    if target.ndim != 3 or prediction.ndim < 3 or prediction.shape[-3:] != target.shape:
        raise ValueError(
            "prediction_mv must end in the target (protocol, time, probe) shape; "
            f"got prediction {prediction.shape!r} and target {target.shape!r}."
        )
    if tolerance_ms < 0.0:
        raise ValueError(f"tolerance_ms must be nonnegative, got {tolerance_ms!r}.")
    target_counts = hard_spike_counts(target)
    prediction_counts = hard_spike_counts(prediction)
    count_error = prediction_counts - target_counts
    count_exact = count_error == 0
    timing_exact = np.zeros(count_exact.shape, dtype=bool)
    leading_shape = prediction.shape[: -target.ndim]
    flattened = prediction.reshape((-1,) + target.shape)
    flattened_timing = timing_exact.reshape((-1, target.shape[0]))
    tolerance_steps = tolerance_ms / dataset.DT_MS
    for candidate_index, candidate in enumerate(flattened):
        for protocol_index in range(target.shape[0]):
            target_soma = target[protocol_index, :, 0]
            predicted_soma = candidate[protocol_index, :, 0]
            target_indices = np.flatnonzero((target_soma[:-1] < 0.0) & (target_soma[1:] >= 0.0)) + 1
            predicted_indices = np.flatnonzero((predicted_soma[:-1] < 0.0) & (predicted_soma[1:] >= 0.0)) + 1
            if target_indices.size == predicted_indices.size:
                maximum_error = 0.0 if target_indices.size == 0 else np.max(np.abs(predicted_indices - target_indices))
                flattened_timing[candidate_index, protocol_index] = maximum_error <= tolerance_steps
    timing_exact = flattened_timing.reshape(leading_shape + (target.shape[0],))
    return count_error, count_exact, timing_exact


def max_ordered_spike_timing_error_ms(prediction_mv: np.ndarray, target_mv: np.ndarray) -> np.ndarray:
    """Return maximum ordered soma-spike timing error, or NaN for count mismatch."""
    prediction = np.asarray(prediction_mv)
    target = np.asarray(target_mv)
    if target.ndim != 3 or prediction.ndim < 3 or prediction.shape[-3:] != target.shape:
        raise ValueError(
            "prediction_mv must end in the target (protocol, time, probe) shape; "
            f"got prediction {prediction.shape!r} and target {target.shape!r}."
        )
    leading_shape = prediction.shape[: -target.ndim]
    flattened = prediction.reshape((-1,) + target.shape)
    timing_error = np.full((flattened.shape[0], target.shape[0]), np.nan, dtype=float)
    for candidate_index, candidate in enumerate(flattened):
        for protocol_index in range(target.shape[0]):
            target_soma = target[protocol_index, :, 0]
            predicted_soma = candidate[protocol_index, :, 0]
            target_indices = np.flatnonzero((target_soma[:-1] < 0.0) & (target_soma[1:] >= 0.0)) + 1
            predicted_indices = np.flatnonzero((predicted_soma[:-1] < 0.0) & (predicted_soma[1:] >= 0.0)) + 1
            if target_indices.size != predicted_indices.size:
                continue
            maximum_steps = 0.0 if target_indices.size == 0 else np.max(np.abs(predicted_indices - target_indices))
            timing_error[candidate_index, protocol_index] = maximum_steps * dataset.DT_MS
    return timing_error.reshape(leading_shape + (target.shape[0],))


def _build_playback_cell() -> braincell.Cell:
    """Build a trainable population without population-shaped clamp values."""
    cell = braincell.Cell(
        dataset.build_morphology(),
        cv_policy=braincell.CVPerBranch(),
        V_init=-65.0 * u.mV,
        solver="staggered",
        pop_size=(BATCH_SIZE,),
    )
    cell.paint(
        AllRegion(),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
    )
    for index, site in enumerate(dataset.SITES):
        leak, sodium, potassium = dataset.TARGET_PARAMETERS[index]
        region = BranchSlice(branch_index=index, prox=0.0, dist=1.0)
        cell.paint(
            region,
            mech.Channel(
                "ExplorationTrainableNa",
                name=f"{site}_na",
                g_max=sodium * dataset.CONDUCTANCE_UNIT,
            ),
            mech.Channel(
                "ExplorationTrainableK",
                name=f"{site}_k",
                g_max=potassium * dataset.CONDUCTANCE_UNIT,
            ),
            mech.Channel(
                "ExplorationTrainableLeak",
                name=f"{site}_leak",
                g_max=leak * dataset.CONDUCTANCE_UNIT,
                E=-54.387 * u.mV,
            ),
        )
        cell.place(
            at(site, 0.5),
            mech.CurrentClamp(
                durations=np.asarray([100.0]) * u.ms,
                amplitudes=np.asarray([0.0]) * u.nA,
            ),
        )
        cell.place(at(site, 0.5), mech.StateProbe(name=f"{site}_v", field="v"))
    cell.init_state()
    return cell


class PlaybackProblem:
    """Own one fixed-shape population model and state-backed training data."""

    def __init__(self) -> None:
        self.cell = _build_playback_cell()
        found = dataset.find_conductance_parameters(self.cell)
        self.parameters = tuple(found[name] for name in PARAMETER_NAMES)
        self.current = brainstate.State(jnp.zeros((BATCH_SIZE, dataset.N_STEPS, 3)))
        self.target = brainstate.State(jnp.zeros((BATCH_SIZE, dataset.N_STEPS, 3)))
        self.weights = brainstate.State(jnp.ones((BATCH_SIZE, dataset.N_STEPS)))
        self.normalizer = brainstate.State(jnp.ones((BATCH_SIZE,)))
        midpoint_ids = jnp.asarray(self.cell.node_tree.cv_to_mid_node_id)
        midpoint_area = self.cell.runtime.point_area.to_decimal(u.cm**2)[midpoint_ids]

        def playback_current(_point_voltage):
            time_ms = self.cell._resolve_t().to_decimal(u.ms)
            index = jnp.clip(jnp.rint(time_ms / dataset.DT_MS).astype(jnp.int32), 0, dataset.N_STEPS - 1)
            density = jnp.zeros((BATCH_SIZE, self.cell.n_point))
            density = density.at[:, midpoint_ids].set(self.current.value[:, index, :] / midpoint_area)
            return density * u.nA / u.cm**2

        self.cell.add_current_input("heterogeneous_dataset_playback", playback_current)

    @property
    def parameter_states(self) -> dict[str, brainstate.State]:
        """Return optimizer states in the declared physical order."""
        return {name: parameter.val for name, parameter in zip(PARAMETER_NAMES, self.parameters)}

    def physical_parameters(self):
        """Return all nine conductances in mS/cm2."""
        return jnp.stack([parameter.value().to_decimal(dataset.CONDUCTANCE_UNIT) for parameter in self.parameters])

    def set_physical_parameters(self, values) -> None:
        """Set all nine conductances from mS/cm2 values."""
        for parameter, value in zip(self.parameters, jnp.asarray(values)):
            parameter.set_value(value * dataset.CONDUCTANCE_UNIT)

    def set_batch(self, data: LoadedDataset, indices, normalizers, masks) -> None:
        """Bind one batch of saved currents, targets, masks, and scales."""
        self.current.value = jnp.asarray(data.currents_na)[indices]
        self.target.value = jnp.asarray(data.voltages_mv)[indices]
        self.weights.value = jnp.asarray(masks)[indices]
        self.normalizer.value = jnp.asarray(normalizers)[indices]

    def simulate(self):
        """Reset and simulate the current playback batch."""
        self.cell.reset_state()
        result = self.cell.run(dt=dataset.DT, duration=dataset.DURATION)
        time_leading = u.math.stack(tuple(result.traces[f"{site}_v"] for site in dataset.SITES), axis=-1)
        return jnp.moveaxis(time_leading.to_decimal(u.mV), 0, 1)

    def raw_protocol_losses(self, prediction_mv):
        """Return spike-masked Huber loss independently for every protocol."""
        error = prediction_mv - self.target.value
        absolute = jnp.abs(error)
        huber = jnp.where(
            absolute <= HUBER_DELTA_MV,
            0.5 * error**2,
            HUBER_DELTA_MV * (absolute - 0.5 * HUBER_DELTA_MV),
        )
        weights = self.weights.value[..., None]
        return jnp.sum(weights * huber, axis=(1, 2)) / (jnp.sum(weights, axis=(1, 2)) * 3.0)

    def normalized_protocol_losses(self, prediction_mv):
        """Return normalized objective values independently for every protocol."""
        return self.raw_protocol_losses(prediction_mv) / self.normalizer.value

    def loss_with_aux(self):
        """Return normalized batch loss and unnormalized per-protocol losses."""
        raw = self.raw_protocol_losses(self.simulate())
        return jnp.mean(raw / self.normalizer.value), raw


def _state_leaves(node) -> list[brainstate.State]:
    return jax.tree.leaves(brainstate.graph.states(node), is_leaf=lambda value: isinstance(value, brainstate.State))


def _unique_states(*nodes) -> list[brainstate.State]:
    unique = []
    seen = set()
    for node in nodes:
        for state in _state_leaves(node):
            if id(state) not in seen:
                seen.add(id(state))
                unique.append(state)
    return unique


def _broadcast_state(state: brainstate.State, size: int) -> None:
    state.value = jax.tree.map(
        lambda value: jnp.broadcast_to(jnp.asarray(value), (size,) + jnp.asarray(value).shape),
        state.value,
    )


def _state_mapping(states: list[brainstate.State]) -> dict[str, brainstate.State]:
    return {f"state_{index}": state for index, state in enumerate(states)}


def compute_normalizers(data: LoadedDataset, masks: np.ndarray) -> np.ndarray:
    """Evaluate the fixed canonical prediction scale for all protocols."""
    problem = PlaybackProblem()
    canonical_factors = np.tile(core.CANONICAL_INITIAL_PARAMETERS / core.TARGET_PARAMETERS, len(dataset.SITES))
    problem.set_physical_parameters(TARGET_PARAMETERS * canonical_factors)
    all_indices = jnp.arange(data.currents_na.shape[0], dtype=jnp.int32).reshape(-1, BATCH_SIZE)

    def evaluate(indices):
        problem.set_batch(data, indices, np.ones(data.currents_na.shape[0]), masks)
        prediction = problem.simulate()
        return problem.raw_protocol_losses(prediction)

    raw = brainstate.transform.for_loop(evaluate, all_indices)
    return np.maximum(np.asarray(raw).reshape(-1), 0.1)


def _run_start_chunk(
    data: LoadedDataset,
    starts: np.ndarray,
    schedule: np.ndarray,
    masks: np.ndarray,
    normalizers: np.ndarray,
    *,
    learning_rate: float,
) -> dict[str, np.ndarray]:
    """Train two starts with shared protocol batches and independent states."""
    if starts.shape != (START_CHUNK_SIZE, 9):
        raise ValueError(f"A start chunk must have shape (2, 9), got {starts.shape!r}.")
    problem = PlaybackProblem()
    optimizer = braintools.optim.Adam(lr=learning_rate, grad_clip_norm=1.0)
    optimizer.register_trainable_weights(problem.parameter_states)
    cell_states = _unique_states(problem.cell)
    mapped_states = _unique_states(problem.cell, optimizer)
    for state in mapped_states:
        _broadcast_state(state, START_CHUNK_SIZE)
    problem.set_physical_parameters(jnp.asarray(starts).T)
    mapped_cell_states = _state_mapping(cell_states)
    mapped_states_dict = _state_mapping(mapped_states)
    grad_fn = brainstate.transform.grad(
        problem.loss_with_aux,
        grad_states=problem.parameter_states,
        has_aux=True,
        return_value=True,
    )

    def simulate_one(_):
        return problem.simulate()

    batched_simulate = brainstate.transform.vmap(
        simulate_one,
        in_axes=None,
        out_axes=0,
        in_states=mapped_cell_states,
        out_states=mapped_cell_states,
        axis_size=START_CHUNK_SIZE,
    )

    def train_one(active):
        old_values = tuple(state.value for state in problem.parameter_states.values())
        gradients, loss, raw = grad_fn()
        gradient_vector = jnp.stack(tuple(gradients[name] for name in PARAMETER_NAMES))
        optimizer.update({name: gradients[name] * active for name in PARAMETER_NAMES})
        for state, old_value in zip(problem.parameter_states.values(), old_values):
            state.value = jnp.where(active, state.value, old_value)
        return loss, jnp.mean(raw), gradient_vector, problem.physical_parameters()

    batched_train = brainstate.transform.vmap(
        train_one,
        in_axes=0,
        out_axes=0,
        in_states=mapped_states_dict,
        out_states=mapped_states_dict,
    )

    def evaluate_one(_):
        loss, _ = problem.loss_with_aux()
        return loss

    batched_evaluate = brainstate.transform.vmap(
        evaluate_one,
        in_axes=None,
        out_axes=0,
        in_states=mapped_cell_states,
        out_states=mapped_cell_states,
        axis_size=START_CHUNK_SIZE,
    )
    validation_indices = jnp.asarray(data.indices("validation"), dtype=jnp.int32)
    test_indices = jnp.asarray(data.indices("test"), dtype=jnp.int32)
    problem.set_batch(data, test_indices, normalizers, masks)
    initial_test_traces = batched_simulate(None)
    initial_parameters = jnp.moveaxis(problem.physical_parameters(), 0, 1)

    initial_best_loss = jnp.full((START_CHUNK_SIZE,), jnp.inf)
    initial_best_parameters = initial_parameters
    initial_patience = jnp.zeros((START_CHUNK_SIZE,), dtype=jnp.int32)
    initial_active = jnp.ones((START_CHUNK_SIZE,), dtype=bool)

    def epoch_step(carry, epoch_data):
        best_loss, best_parameters, patience, active = carry
        epoch_index, batch_indices = epoch_data

        def batch_step(indices):
            problem.set_batch(data, indices, normalizers, masks)
            return batched_train(active.astype(float))

        batch_loss, raw_loss, gradients, parameters = brainstate.transform.for_loop(batch_step, batch_indices)
        problem.set_batch(data, validation_indices, normalizers, masks)
        validation_loss = batched_evaluate(None)
        relative_threshold = best_loss * (1.0 - 1e-3)
        improved = validation_loss < relative_threshold
        current_parameters = jnp.moveaxis(problem.physical_parameters(), 0, 1)
        next_best_loss = jnp.where(improved, validation_loss, best_loss)
        next_best_parameters = jnp.where(improved[:, None], current_parameters, best_parameters)
        next_patience = jnp.where(improved, 0, patience + 1)
        can_stop = epoch_index >= 9
        next_active = active & ~(can_stop & (next_patience >= 6))
        outputs = (
            jnp.mean(batch_loss, axis=0),
            validation_loss,
            jnp.mean(gradients, axis=0),
            parameters[-1],
            next_active,
        )
        return (next_best_loss, next_best_parameters, next_patience, next_active), outputs

    epoch_numbers = jnp.arange(schedule.shape[0], dtype=jnp.int32)
    final_carry, histories = brainstate.transform.scan(
        epoch_step,
        (initial_best_loss, initial_best_parameters, initial_patience, initial_active),
        (epoch_numbers, jnp.asarray(schedule)),
    )
    best_loss, best_parameters, _, _ = final_carry
    problem.set_physical_parameters(jnp.asarray(best_parameters).T)
    problem.set_batch(data, test_indices, normalizers, masks)
    test_losses = batched_evaluate(None)
    fitted_test_traces = batched_simulate(None)
    jax.block_until_ready((test_losses, fitted_test_traces))
    train_losses, validation_losses, gradients, epoch_parameters, active_history = histories
    trajectory = jnp.concatenate((initial_parameters[None, ...], epoch_parameters), axis=0)
    trajectory = jnp.moveaxis(trajectory, 0, 1)
    validation_array = np.asarray(validation_losses).T
    best_epochs = np.argmin(validation_array, axis=1)
    return {
        "train_losses": np.asarray(train_losses).T,
        "validation_losses": validation_array,
        "gradients": np.moveaxis(np.asarray(gradients), 0, 1),
        "parameter_trajectories": np.asarray(trajectory),
        "best_parameters": np.asarray(best_parameters),
        "best_losses": np.asarray(best_loss),
        "best_epochs": best_epochs,
        "test_losses": np.asarray(test_losses),
        "initial_test_traces": np.asarray(initial_test_traces),
        "fitted_test_traces": np.asarray(fitted_test_traces),
        "active_history": np.asarray(active_history).T,
    }


def _concatenate_chunks(chunks: tuple[dict[str, np.ndarray], ...], key: str) -> np.ndarray:
    return np.concatenate(tuple(chunk[key] for chunk in chunks), axis=0)


def run_training(
    data: LoadedDataset,
    *,
    n_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = 0.01,
    seed: int = RANDOM_SEED,
) -> TrainingResult:
    """Run the eight-start, nine-parameter minibatch experiment."""
    starts = make_initial_grid()
    schedule = make_batch_schedule(data, n_epochs, seed)
    masks = target_weight_masks(data.voltages_mv)
    started = time.perf_counter()
    normalizers = compute_normalizers(data, masks)
    chunk_0 = _run_start_chunk(data, starts[0:2], schedule, masks, normalizers, learning_rate=learning_rate)
    chunk_1 = _run_start_chunk(data, starts[2:4], schedule, masks, normalizers, learning_rate=learning_rate)
    chunk_2 = _run_start_chunk(data, starts[4:6], schedule, masks, normalizers, learning_rate=learning_rate)
    chunk_3 = _run_start_chunk(data, starts[6:8], schedule, masks, normalizers, learning_rate=learning_rate)
    chunks = (chunk_0, chunk_1, chunk_2, chunk_3)
    test_indices = data.indices("test")
    fitted = _concatenate_chunks(chunks, "fitted_test_traces")
    return TrainingResult(
        starts=starts,
        target_parameters=TARGET_PARAMETERS.copy(),
        train_losses=_concatenate_chunks(chunks, "train_losses"),
        validation_losses=_concatenate_chunks(chunks, "validation_losses"),
        gradients=_concatenate_chunks(chunks, "gradients"),
        parameter_trajectories=_concatenate_chunks(chunks, "parameter_trajectories"),
        best_parameters=_concatenate_chunks(chunks, "best_parameters"),
        best_epochs=_concatenate_chunks(chunks, "best_epochs"),
        test_losses=_concatenate_chunks(chunks, "test_losses"),
        initial_test_traces=_concatenate_chunks(chunks, "initial_test_traces"),
        fitted_test_traces=fitted,
        target_test_traces=data.voltages_mv[test_indices],
        target_test_spike_counts=hard_spike_counts(data.voltages_mv[test_indices]),
        fitted_test_spike_counts=hard_spike_counts(fitted),
        schedule=schedule,
        normalizers=normalizers,
        learning_rate=learning_rate,
        random_seed=seed,
        wall_seconds=time.perf_counter() - started,
    )


def _landscape_axes(result: TrainingResult, best_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = result.target_parameters
    paths = np.log(np.clip(result.parameter_trajectories, 1e-12, None) / target)
    best = np.log(result.best_parameters[best_index] / target)
    axis_1 = -best
    if np.linalg.norm(axis_1) < 1e-8:
        axis_1 = np.eye(target.size)[0]
    axis_1 /= np.linalg.norm(axis_1)
    _, _, vh = np.linalg.svd(paths.reshape(-1, target.size), full_matrices=False)
    axis_2 = vh[0] - np.dot(vh[0], axis_1) * axis_1
    if np.linalg.norm(axis_2) < 1e-8:
        axis_2 = vh[1] - np.dot(vh[1], axis_1) * axis_1
    axis_2 /= np.linalg.norm(axis_2)
    projected = np.stack((paths @ axis_1, paths @ axis_2), axis=-1)
    return axis_1, axis_2, projected


def _evaluate_parameter_batches(
    data: LoadedDataset,
    parameters: np.ndarray,
    masks: np.ndarray,
    normalizers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate candidate batches against the fixed validation set."""
    batch_width = parameters.shape[1]
    problem = PlaybackProblem()
    cell_states = _unique_states(problem.cell)
    for state in cell_states:
        _broadcast_state(state, batch_width)
    mapped = _state_mapping(cell_states)
    validation_indices = jnp.asarray(data.indices("validation"), dtype=jnp.int32)
    target_counts = jnp.asarray(hard_spike_counts(data.voltages_mv[data.indices("validation")]))
    problem.set_batch(data, validation_indices, normalizers, masks)

    def evaluate_one(_):
        prediction = problem.simulate()
        normalized = problem.normalized_protocol_losses(prediction)
        counts = jnp.sum(
            (prediction[:, :-1, 0] < 0.0) & (prediction[:, 1:, 0] >= 0.0),
            axis=1,
        )
        return jnp.mean(normalized), jnp.sum(jnp.abs(counts - target_counts))

    batched_evaluate = brainstate.transform.vmap(
        evaluate_one,
        in_axes=None,
        out_axes=0,
        in_states=mapped,
        out_states=mapped,
        axis_size=batch_width,
    )

    def evaluate_batch(values):
        problem.set_physical_parameters(values.T)
        return batched_evaluate(None)

    losses, mismatch = brainstate.transform.for_loop(evaluate_batch, jnp.asarray(parameters))
    jax.block_until_ready(losses)
    return np.asarray(losses), np.asarray(mismatch)


def _hard_spike_quality_jax(prediction_mv, target_mv, *, max_spikes: int):
    predicted_crossings = (prediction_mv[:, :-1, 0] < 0.0) & (prediction_mv[:, 1:, 0] >= 0.0)
    target_crossings = (target_mv[:, :-1, 0] < 0.0) & (target_mv[:, 1:, 0] >= 0.0)
    predicted_counts = jnp.sum(predicted_crossings, axis=1)
    target_counts = jnp.sum(target_crossings, axis=1)
    count_exact = predicted_counts == target_counts
    if max_spikes == 0:
        return count_exact, count_exact
    time_indices = jnp.arange(1, prediction_mv.shape[1], dtype=jnp.int32)[None, :]

    def ordered_indices(crossings):
        crossing_ranks = jnp.cumsum(crossings, axis=1)
        return jnp.stack(
            tuple(
                jnp.max(jnp.where(crossings & (crossing_ranks == rank), time_indices, 0), axis=1)
                for rank in range(1, max_spikes + 1)
            ),
            axis=1,
        )

    predicted_indices = ordered_indices(predicted_crossings)
    target_indices = ordered_indices(target_crossings)
    active_spikes = jnp.arange(max_spikes)[None, :] < target_counts[:, None]
    timing_error_steps = jnp.max(
        jnp.where(active_spikes, jnp.abs(predicted_indices - target_indices), 0),
        axis=1,
    )
    timing_exact = count_exact & (timing_error_steps * dataset.DT_MS <= SPIKE_TIMING_TOLERANCE_MS)
    return count_exact, timing_exact


def compute_quality_diagnostics(data: LoadedDataset, result: TrainingResult) -> QualityDiagnostics:
    """Replay all saved checkpoints and compute interpretable quality metrics."""
    trajectories = np.asarray(result.parameter_trajectories)
    if trajectories.ndim != 3 or trajectories.shape[0] != 8 or trajectories.shape[2] != 9:
        raise ValueError(f"Expected parameter trajectories with shape (8, checkpoint, 9), got {trajectories.shape!r}.")
    checkpoint_count = trajectories.shape[1]
    problem = PlaybackProblem()
    cell_states = _unique_states(problem.cell)
    for state in cell_states:
        _broadcast_state(state, checkpoint_count)
    mapped = _state_mapping(cell_states)
    validation_indices = jnp.asarray(data.indices("validation"), dtype=jnp.int32)
    masks = target_weight_masks(data.voltages_mv)
    problem.set_batch(data, validation_indices, result.normalizers, masks)
    max_target_spikes = int(np.max(hard_spike_counts(data.voltages_mv[data.indices("validation")])))

    def evaluate_one(_):
        prediction = problem.simulate()
        voltage_rmse = jnp.sqrt(jnp.mean((prediction - problem.target.value) ** 2))
        count_exact, timing_exact = _hard_spike_quality_jax(
            prediction,
            problem.target.value,
            max_spikes=max_target_spikes,
        )
        return voltage_rmse, count_exact, timing_exact

    batched_evaluate = brainstate.transform.vmap(
        evaluate_one,
        in_axes=None,
        out_axes=0,
        in_states=mapped,
        out_states=mapped,
        axis_size=checkpoint_count,
    )

    def evaluate_seed(parameter_checkpoints):
        problem.set_physical_parameters(parameter_checkpoints.T)
        return batched_evaluate(None)

    voltage_rmse, validation_count_exact, validation_timing_exact = brainstate.transform.for_loop(
        evaluate_seed,
        jnp.asarray(trajectories),
    )
    jax.block_until_ready((voltage_rmse, validation_timing_exact))
    test_count_error, test_count_exact, test_timing_exact = hard_spike_protocol_quality(
        result.fitted_test_traces,
        result.target_test_traces,
    )
    return _assemble_quality_diagnostics(
        result,
        np.asarray(voltage_rmse),
        np.asarray(validation_count_exact),
        np.asarray(validation_timing_exact),
        test_count_error=np.asarray(test_count_error),
        test_count_exact=np.asarray(test_count_exact),
        test_timing_exact=np.asarray(test_timing_exact),
    )


def _assemble_quality_diagnostics(
    result: TrainingResult,
    validation_voltage_rmse_mv: np.ndarray,
    validation_spike_count_exact: np.ndarray,
    validation_spike_timing_exact: np.ndarray,
    *,
    test_count_error: np.ndarray | None = None,
    test_count_exact: np.ndarray | None = None,
    test_timing_exact: np.ndarray | None = None,
) -> QualityDiagnostics:
    """Combine saved validation replay values with endpoint diagnostics."""
    parameter_distance = relative_parameter_rms(result.parameter_trajectories, result.target_parameters)
    if test_count_error is None or test_count_exact is None or test_timing_exact is None:
        test_count_error, test_count_exact, test_timing_exact = hard_spike_protocol_quality(
            result.fitted_test_traces,
            result.target_test_traces,
        )
    test_protocol_rmse = protocol_voltage_rmse(result.fitted_test_traces, result.target_test_traces)
    return QualityDiagnostics(
        parameter_relative_rms=parameter_distance,
        relative_parameter_step_rms=relative_parameter_step_rms(
            result.parameter_trajectories,
            result.target_parameters,
        ),
        best_parameter_relative_rms=relative_parameter_rms(result.best_parameters, result.target_parameters),
        best_parameter_signed_relative_error=signed_relative_parameter_error(
            result.best_parameters,
            result.target_parameters,
        ),
        validation_voltage_rmse_mv=np.asarray(validation_voltage_rmse_mv),
        validation_spike_count_exact=np.asarray(validation_spike_count_exact),
        validation_spike_timing_exact=np.asarray(validation_spike_timing_exact),
        test_voltage_rmse_mv=np.sqrt(np.mean(test_protocol_rmse**2, axis=1)),
        test_protocol_voltage_rmse_mv=test_protocol_rmse,
        test_spike_count_error=np.asarray(test_count_error),
        test_spike_count_exact=np.asarray(test_count_exact),
        test_spike_timing_exact=np.asarray(test_timing_exact),
        test_spike_max_timing_error_ms=max_ordered_spike_timing_error_ms(
            result.fitted_test_traces,
            result.target_test_traces,
        ),
    )


def load_saved_quality_diagnostics(
    result: TrainingResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> QualityDiagnostics:
    """Build endpoint diagnostics while reusing saved validation replay arrays."""
    arrays_path = output_dir / "training_results.npz"
    required = (
        "validation_voltage_rmse_mv",
        "validation_spike_count_exact",
        "validation_spike_timing_exact",
    )
    with np.load(arrays_path) as values:
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"Saved diagnostics are missing validation arrays: {missing!r}.")
        validation_values = tuple(np.asarray(values[name]) for name in required)
    return _assemble_quality_diagnostics(result, *validation_values)


def compute_landscape(data: LoadedDataset, result: TrainingResult) -> LandscapeResult:
    """Compute one trajectory-informed plane and nine local profiles."""
    best_index = int(np.argmin(np.min(result.validation_losses, axis=1)))
    axis_1, axis_2, projected = _landscape_axes(result, best_index)
    all_x = np.concatenate((projected[..., 0].ravel(), np.asarray([0.0])))
    all_y = np.concatenate((projected[..., 1].ravel(), np.asarray([0.0])))
    x_values = np.linspace(all_x.min() - 0.15, all_x.max() + 0.15, 15)
    y_values = np.linspace(all_y.min() - 0.15, all_y.max() + 0.15, 15)
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    log_values = x_grid[..., None] * axis_1 + y_grid[..., None] * axis_2
    plane_parameters = TARGET_PARAMETERS * np.exp(log_values)
    plane_parameters = np.clip(plane_parameters, LOWER_BOUNDS + 1e-4, UPPER_BOUNDS - 1e-4)
    masks = target_weight_masks(data.voltages_mv)
    plane_loss, plane_mismatch = _evaluate_parameter_batches(
        data,
        plane_parameters,
        masks,
        result.normalizers,
    )

    anchors = (TARGET_PARAMETERS, result.best_parameters[best_index])
    profile_parameters = np.empty((2, 9, 31, 9), dtype=float)
    for anchor_index, anchor in enumerate(anchors):
        for parameter_index in range(9):
            lower = max(LOWER_BOUNDS[parameter_index] + 1e-4, TARGET_PARAMETERS[parameter_index] * 0.45)
            upper = min(UPPER_BOUNDS[parameter_index] - 1e-4, TARGET_PARAMETERS[parameter_index] * 1.65)
            values = np.geomspace(lower, upper, 31)
            target_index = int(np.argmin(np.abs(values - TARGET_PARAMETERS[parameter_index])))
            values[target_index] = TARGET_PARAMETERS[parameter_index]
            anchor_value = anchor[parameter_index]
            anchor_index_in_values = int(np.argmin(np.abs(values - anchor_value)))
            values[anchor_index_in_values] = anchor_value
            values.sort()
            profile_parameters[anchor_index, parameter_index] = anchor
            profile_parameters[anchor_index, parameter_index, :, parameter_index] = values
    profile_loss, profile_mismatch = _evaluate_parameter_batches(
        data,
        profile_parameters.reshape(18, 31, 9),
        masks,
        result.normalizers,
    )
    best_log = np.log(result.best_parameters[best_index] / TARGET_PARAMETERS)
    return LandscapeResult(
        axis_1=axis_1,
        axis_2=axis_2,
        x_values=x_values,
        y_values=y_values,
        plane_losses=plane_loss,
        plane_spike_mismatch=plane_mismatch,
        projected_trajectories=projected,
        best_projection=np.asarray((best_log @ axis_1, best_log @ axis_2)),
        profile_parameters=profile_parameters,
        profile_losses=profile_loss.reshape(2, 9, 31),
        profile_spike_mismatch=profile_mismatch.reshape(2, 9, 31),
    )


def _plot_losses(result: TrainingResult):
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True, constrained_layout=True)
    for start, axis in enumerate(axes.flat):
        epochs = np.arange(1, result.train_losses.shape[1] + 1)
        axis.semilogy(epochs, result.train_losses[start], label="Train", color="#3478a4")
        axis.semilogy(epochs, result.validation_losses[start], label="Validation", color="#d1495b")
        axis.axvline(result.best_epochs[start] + 1, color="#222222", linestyle=":", linewidth=0.8)
        axis.set(title=f"Start {start}", xlabel="Epoch", ylabel="Normalized objective")
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Nine-parameter minibatch training")
    return figure


def _plot_parameter_distance(result: TrainingResult, quality: QualityDiagnostics):
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
        constrained_layout=True,
    )
    distance_axis, step_axis = axes
    colors = plt.cm.tab10(np.arange(8))
    epochs = np.arange(quality.parameter_relative_rms.shape[1])
    for start, color in enumerate(colors):
        distance = quality.parameter_relative_rms[start]
        best_checkpoint = int(result.best_epochs[start]) + 1
        distance_axis.plot(epochs, distance, color=color, linewidth=1.8, label=f"Start {start}")
        distance_axis.scatter(
            best_checkpoint,
            distance[best_checkpoint],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=45,
            zorder=3,
        )
        step_axis.plot(
            epochs[1:],
            quality.relative_parameter_step_rms[start],
            color=color,
            linewidth=1.35,
        )
        step_axis.scatter(
            best_checkpoint,
            quality.relative_parameter_step_rms[start, best_checkpoint - 1],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=36,
            zorder=3,
        )
    distance_axis.axhline(0.0, color="#111111", linestyle=":", linewidth=1.0)
    distance_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    distance_axis.set(
        title="Distance to the nine-parameter ground truth",
        ylabel="Relative RMS parameter error",
        ylim=(0.0, None),
    )
    distance_axis.grid(alpha=0.18)
    distance_axis.legend(frameon=False, ncol=4)
    step_axis.set_yscale("symlog", linthresh=1e-5, linscale=0.7)
    step_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    step_axis.set(
        title="Target-scaled movement between adjacent checkpoints",
        xlabel="Epoch (0 is initialization)",
        ylabel="Relative RMS parameter step",
        xlim=(0, epochs[-1]),
        ylim=(0.0, None),
    )
    step_axis.grid(alpha=0.18)
    return figure


def _plot_validation_quality(result: TrainingResult, quality: QualityDiagnostics):
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    colors = plt.cm.tab10(np.arange(8))
    epochs = np.arange(quality.parameter_relative_rms.shape[1])
    count_fraction = np.mean(quality.validation_spike_count_exact, axis=2)
    timing_fraction = np.mean(quality.validation_spike_timing_exact, axis=2)
    series = (
        (quality.validation_voltage_rmse_mv, "Voltage RMSE (mV)"),
        (count_fraction, "Protocols with exact spike count"),
        (timing_fraction, f"Exact count and all spike times within {SPIKE_TIMING_TOLERANCE_MS:g} ms"),
    )
    for axis, (values, ylabel) in zip(axes, series):
        for start, color in enumerate(colors):
            best_checkpoint = int(result.best_epochs[start]) + 1
            axis.plot(epochs, values[start], color=color, linewidth=1.6, label=f"Start {start}")
            axis.scatter(
                best_checkpoint,
                values[start, best_checkpoint],
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=28,
                zorder=3,
            )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18)
    axes[0].set_yscale("log")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False, ncol=4)
    axes[-1].set(xlabel="Epoch (0 is initialization)", xlim=(0, epochs[-1]))
    figure.suptitle("Validation quality over saved checkpoints")
    return figure


def _short_protocol_label(data: LoadedDataset, source_index: int) -> str:
    family = str(data.families[source_index])
    site = str(data.sites[source_index])
    protocol_id = str(data.protocol_ids[source_index])
    suffix = protocol_id.removeprefix(f"{family}_{site}_")
    return f"{family}/{site}\n{suffix}"


def _decorate_protocol_axis(axis, data: LoadedDataset) -> None:
    """Label test protocols and separate site/family blocks."""
    test_indices = data.indices("test")
    groups = [(str(data.sites[index]), str(data.families[index])) for index in test_indices]
    for position in range(1, len(groups)):
        if groups[position] != groups[position - 1]:
            axis.axvline(position - 0.5, color="white", linewidth=1.8)
            axis.axvline(position - 0.5, color="#333333", linewidth=0.45)
    axis.set_xticks(np.arange(test_indices.size))
    axis.set_xticklabels([_short_protocol_label(data, int(index)) for index in test_indices])
    axis.tick_params(axis="x", labelrotation=45, labelsize=7)


def _plot_test_spike_metrics(data: LoadedDataset, quality: QualityDiagnostics):
    figure = plt.figure(figsize=(18, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(3.1, 1.0))
    heatmap_axis = figure.add_subplot(grid[0])
    bar_axis = figure.add_subplot(grid[1])
    errors = quality.test_spike_count_error
    maximum = max(1, int(np.max(np.abs(errors))))
    image = heatmap_axis.imshow(errors, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
    for start in range(errors.shape[0]):
        for protocol in range(errors.shape[1]):
            value = int(errors[start, protocol])
            heatmap_axis.text(
                protocol,
                start,
                f"{value:+d}" if value else "0",
                ha="center",
                va="center",
                color="white" if abs(value) > maximum / 2 else "#222222",
                fontsize=8,
            )
    test_indices = data.indices("test")
    heatmap_axis.set(
        title="Test soma spike-count error (fitted minus target)",
        xlabel="Protocol",
        ylabel="Start",
        yticks=np.arange(8),
        yticklabels=[f"Start {index}" for index in range(8)],
        xticks=np.arange(test_indices.size),
        xticklabels=[_short_protocol_label(data, int(index)) for index in test_indices],
    )
    heatmap_axis.tick_params(axis="x", labelrotation=45, labelsize=7)
    figure.colorbar(image, ax=heatmap_axis, label="Spike-count error", shrink=0.85)

    x_values = np.arange(8)
    width = 0.34
    count_fraction = np.mean(quality.test_spike_count_exact, axis=1)
    timing_fraction = np.mean(quality.test_spike_timing_exact, axis=1)
    count_bars = bar_axis.bar(
        x_values - width / 2,
        count_fraction,
        width,
        color="#3478a4",
        label="Exact count",
    )
    timing_bars = bar_axis.bar(
        x_values + width / 2,
        timing_fraction,
        width,
        color="#d1495b",
        label=f"Count + timing <= {SPIKE_TIMING_TOLERANCE_MS:g} ms",
    )
    bar_axis.bar_label(
        count_bars, labels=[f"{int(value)}/18" for value in np.sum(quality.test_spike_count_exact, axis=1)]
    )
    bar_axis.bar_label(
        timing_bars,
        labels=[f"{int(value)}/18" for value in np.sum(quality.test_spike_timing_exact, axis=1)],
    )
    bar_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    bar_axis.set(
        xlabel="Start",
        ylabel="Successful test protocols",
        xticks=x_values,
        xticklabels=[f"Start {index}" for index in range(8)],
        ylim=(0.0, 1.12),
    )
    bar_axis.grid(axis="y", alpha=0.18)
    bar_axis.legend(frameon=False, ncol=2)
    return figure


def _plot_test_protocol_voltage_rmse(data: LoadedDataset, quality: QualityDiagnostics):
    values = np.asarray(quality.test_protocol_voltage_rmse_mv)
    positive = values[values > 0.0]
    vmin = max(float(np.min(positive)) if positive.size else 1e-3, 1e-3)
    vmax = max(float(np.max(values)), vmin * 1.001)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    figure, axis = plt.subplots(figsize=(18, 6.2), constrained_layout=True)
    image = axis.imshow(values, cmap="magma_r", norm=norm, aspect="auto")
    contrast = np.sqrt(vmin * vmax)
    for start in range(values.shape[0]):
        for protocol in range(values.shape[1]):
            value = float(values[start, protocol])
            axis.text(
                protocol,
                start,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= contrast else "#202020",
                fontsize=7.5,
            )
    axis.set(
        title="Test voltage RMSE by protocol (all three compartments)",
        xlabel="Protocol",
        ylabel="Start",
        yticks=np.arange(values.shape[0]),
        yticklabels=[f"Start {index}" for index in range(values.shape[0])],
    )
    _decorate_protocol_axis(axis, data)
    figure.colorbar(image, ax=axis, label="Voltage RMSE (mV)", shrink=0.86)
    return figure


def _plot_best_parameter_signed_error(quality: QualityDiagnostics):
    values = np.asarray(quality.best_parameter_signed_relative_error)
    maximum = max(0.01, float(np.max(np.abs(values))))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    figure, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    for start in range(values.shape[0]):
        for parameter in range(values.shape[1]):
            value = float(values[start, parameter])
            axis.text(
                parameter,
                start,
                f"{value:+.1%}",
                ha="center",
                va="center",
                color="white" if abs(value) > maximum * 0.52 else "#202020",
                fontsize=8,
            )
    channel_labels = {"leak": "Leak", "na": "Na", "k": "K"}
    labels = []
    for name in PARAMETER_NAMES:
        site, channel = name.rsplit("_", 1)
        labels.append(f"{site}\n{channel_labels[channel]}")
    axis.set(
        title="Signed parameter error at each validation-best checkpoint",
        xlabel="Compartment and conductance",
        ylabel="Start",
        xticks=np.arange(values.shape[1]),
        xticklabels=labels,
        yticks=np.arange(values.shape[0]),
        yticklabels=[f"Start {index}" for index in range(values.shape[0])],
    )
    for position in (2.5, 5.5):
        axis.axvline(position, color="white", linewidth=2.2)
        axis.axvline(position, color="#333333", linewidth=0.5)
    colorbar = figure.colorbar(image, ax=axis, label="(best - target) / target", shrink=0.86)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    return figure


def _plot_test_spike_timing_error(data: LoadedDataset, quality: QualityDiagnostics):
    values = np.asarray(quality.test_spike_max_timing_error_ms)
    finite = values[np.isfinite(values)]
    observed_maximum = float(np.max(finite)) if finite.size else SPIKE_TIMING_TOLERANCE_MS
    upper = max(50.0, float(np.ceil(observed_maximum)) + 1.0)
    boundaries = np.asarray([0.0, 0.500001, 1.0, 2.0, 5.0, 10.0, 20.0, upper])
    colors = ("#2a9d8f", "#8fcf8a", "#f1d77a", "#f4a261", "#e76f51", "#b23a48", "#6d213c")
    cmap = ListedColormap(colors).with_extremes(bad="#bdbdbd")
    norm = BoundaryNorm(boundaries, cmap.N)
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(figsize=(18, 6.2), constrained_layout=True)
    image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    for start in range(values.shape[0]):
        for protocol in range(values.shape[1]):
            value = values[start, protocol]
            label = "count" if not np.isfinite(value) else f"{value:.2f}"
            axis.text(
                protocol,
                start,
                label,
                ha="center",
                va="center",
                color="#202020" if not np.isfinite(value) or value <= 2.0 else "white",
                fontsize=7.5,
            )
    axis.set(
        title="Maximum ordered soma-spike timing error; gray cells have count mismatch",
        xlabel="Protocol",
        ylabel="Start",
        yticks=np.arange(values.shape[0]),
        yticklabels=[f"Start {index}" for index in range(values.shape[0])],
    )
    _decorate_protocol_axis(axis, data)
    colorbar = figure.colorbar(image, ax=axis, label="Maximum timing error (ms)", shrink=0.86)
    colorbar.set_ticks((0.25, 0.75, 1.5, 3.5, 7.5, 15.0, (20.0 + upper) / 2.0))
    colorbar.set_ticklabels(("<=0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", ">20"))
    return figure


def _plot_endpoint_pareto(quality: QualityDiagnostics):
    parameter_error = np.asarray(quality.best_parameter_relative_rms)
    voltage_rmse = np.asarray(quality.test_voltage_rmse_mv)
    count_fraction = np.mean(quality.test_spike_count_exact, axis=1)
    timing_fraction = np.mean(quality.test_spike_timing_exact, axis=1)
    marker_sizes = 90.0 + 360.0 * count_fraction
    figure, axis = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
    points = axis.scatter(
        parameter_error,
        voltage_rmse,
        c=timing_fraction,
        s=marker_sizes,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    annotations = []
    for start, (x_value, y_value) in enumerate(zip(parameter_error, voltage_rmse)):
        annotations.append(
            axis.annotate(
                f"Start {start}",
                (x_value, y_value),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
            )
        )
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set(
        title="Validation-best endpoint trade-off",
        xlabel="Relative RMS parameter error",
        ylabel="Test voltage RMSE (mV; all three compartments)",
        xlim=(0.0, max(0.05, float(np.max(parameter_error)) * 1.18)),
        ylim=(0.0, max(1.0, float(np.max(voltage_rmse)) * 1.12)),
    )
    axis.grid(alpha=0.18)
    colorbar = figure.colorbar(points, ax=axis, label=f"Count + timing <= {SPIKE_TIMING_TOLERANCE_MS:g} ms")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    size_handles = tuple(
        axis.scatter([], [], s=90.0 + 360.0 * fraction, color="#777777", edgecolor="white", label=label)
        for fraction, label in ((0.25, "25% count match"), (0.75, "75% count match"), (1.0, "100% count match"))
    )
    axis.legend(handles=size_handles, frameon=False, title="Marker area", loc="best")
    _place_nonoverlapping_annotations(figure, axis, annotations)
    return figure


def _place_nonoverlapping_annotations(figure, axis, annotations) -> None:
    """Place point labels using rendered bounding boxes to avoid collisions."""
    vertical_offsets = (5, 20, -10, 35, -25, 50, -40, 65, -55)
    right_candidates = tuple((6, vertical) for vertical in vertical_offsets)
    left_candidates = tuple((-52, vertical) for vertical in vertical_offsets)
    figure.canvas.draw()
    axis_box = axis.get_window_extent()
    placed = []
    x_midpoint = np.mean(axis.get_xlim())
    for annotation in annotations:
        candidates = (
            left_candidates + right_candidates if annotation.xy[0] > x_midpoint else right_candidates + left_candidates
        )
        selected_box = None
        for candidate in candidates:
            annotation.set_position(candidate)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            box = annotation.get_window_extent(renderer).expanded(1.04, 1.12)
            inside = (
                box.x0 >= axis_box.x0 + 2.0
                and box.x1 <= axis_box.x1 - 2.0
                and box.y0 >= axis_box.y0 + 2.0
                and box.y1 <= axis_box.y1 - 2.0
            )
            if inside and not any(box.overlaps(previous) for previous in placed):
                selected_box = box
                break
        if selected_box is None:
            selected_box = annotation.get_window_extent(figure.canvas.get_renderer()).expanded(1.04, 1.12)
        placed.append(selected_box)


def _plot_trace_atlas(data: LoadedDataset, result: TrainingResult, start: int):
    figure, axes = plt.subplots(3, 6, figsize=(19, 9), sharex=True, sharey=True, constrained_layout=True)
    test_indices = data.indices("test")
    colors = ("#1b1b1b", "#367ba8", "#c44e52")
    for local_index, (axis, source_index) in enumerate(zip(axes.flat, test_indices)):
        for probe_index, color in enumerate(colors):
            axis.plot(
                data.time_ms,
                result.target_test_traces[local_index, :, probe_index],
                color=color,
                linewidth=1.6,
            )
            axis.plot(
                data.time_ms,
                result.fitted_test_traces[start, local_index, :, probe_index],
                color=color,
                linewidth=1.35,
                linestyle="--",
            )
        family = str(data.families[source_index])
        site = str(data.sites[source_index])
        prefix = f"{family}_{site}_"
        protocol_id = str(data.protocol_ids[source_index])
        suffix = protocol_id.removeprefix(prefix)
        axis.set(
            title=f"{family} / {site}\n{suffix}",
            xlim=(0.0, 100.0),
            xlabel="ms",
            ylabel="mV",
        )
        axis.title.set_fontsize(8)
        axis.grid(alpha=0.12)
    compartment_handles = tuple(
        Line2D((0,), (0,), color=color, linewidth=2.0, label=probe) for probe, color in zip(dataset.SITES, colors)
    )
    trace_handles = (
        Line2D((0,), (0,), color="#444444", linewidth=2.0, linestyle="-", label="Target"),
        Line2D((0,), (0,), color="#444444", linewidth=2.0, linestyle="--", label="Best checkpoint"),
    )
    figure.legend(handles=compartment_handles, loc="upper left", bbox_to_anchor=(0.04, 1.01), frameon=False, ncol=3)
    figure.legend(handles=trace_handles, loc="upper right", bbox_to_anchor=(0.96, 1.01), frameon=False, ncol=2)
    figure.suptitle(f"Start {start}: target and validation-best test traces")
    return figure


def _plot_landscape(result: TrainingResult, landscape: LandscapeResult):
    figure, axis = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    image = axis.contourf(
        landscape.x_values,
        landscape.y_values,
        np.log10(np.maximum(landscape.plane_losses, 1e-8)),
        levels=18,
        cmap="viridis",
    )
    for start, path in enumerate(landscape.projected_trajectories):
        axis.plot(path[:, 0], path[:, 1], linewidth=0.9, alpha=0.8, label=f"Start {start}")
        axis.scatter(path[0, 0], path[0, 1], marker="o", s=18)
    axis.scatter(0.0, 0.0, marker="*", s=180, color="black", edgecolor="white", label="Target")
    axis.scatter(*landscape.best_projection, marker="X", s=90, color="#d1495b", edgecolor="white", label="Best")
    axis.set(
        title="Validation loss in a trajectory-informed 9D plane",
        xlabel="Best-to-target log-ratio direction",
        ylabel="Orthogonal trajectory PCA direction",
    )
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.colorbar(image, ax=axis, label="log10 normalized objective")
    return figure


def _plot_profiles(landscape: LandscapeResult):
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for parameter_index, axis in enumerate(axes.flat):
        for anchor_index, (label, color) in enumerate((("Target anchor", "#3478a4"), ("Best anchor", "#d1495b"))):
            x = landscape.profile_parameters[anchor_index, parameter_index, :, parameter_index]
            axis.semilogy(x, landscape.profile_losses[anchor_index, parameter_index], color=color, label=label)
            mismatch = landscape.profile_spike_mismatch[anchor_index, parameter_index]
            matched = mismatch == 0
            axis.scatter(
                x[matched],
                landscape.profile_losses[anchor_index, parameter_index, matched],
                s=10,
                color="white",
                edgecolor=color,
            )
        axis.axvline(TARGET_PARAMETERS[parameter_index], color="#111111", linestyle=":", linewidth=0.9)
        axis.set(title=PARAMETER_NAMES[parameter_index], xlabel="mS/cm2", ylabel="Validation loss")
        axis.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("One-dimensional profiles; hollow points match all validation spike counts")
    return figure


def _quality_array_values(quality: QualityDiagnostics) -> dict[str, np.ndarray]:
    return {
        "parameter_relative_rms": quality.parameter_relative_rms,
        "relative_parameter_step_rms": quality.relative_parameter_step_rms,
        "best_parameter_relative_rms": quality.best_parameter_relative_rms,
        "best_parameter_signed_relative_error": quality.best_parameter_signed_relative_error,
        "validation_voltage_rmse_mv": quality.validation_voltage_rmse_mv,
        "validation_spike_count_exact": quality.validation_spike_count_exact,
        "validation_spike_timing_exact": quality.validation_spike_timing_exact,
        "test_voltage_rmse_mv": quality.test_voltage_rmse_mv,
        "test_protocol_voltage_rmse_mv": quality.test_protocol_voltage_rmse_mv,
        "test_spike_count_error": quality.test_spike_count_error,
        "test_spike_count_exact": quality.test_spike_count_exact,
        "test_spike_timing_exact": quality.test_spike_timing_exact,
        "test_spike_max_timing_error_ms": quality.test_spike_max_timing_error_ms,
    }


def _finite_values_or_none(values: np.ndarray) -> list:
    """Convert numeric arrays to JSON lists while representing NaN as null."""
    converted = np.asarray(values).astype(object)
    converted[~np.isfinite(np.asarray(values, dtype=float))] = None
    return converted.tolist()


def _quality_summary_values(quality: QualityDiagnostics) -> dict:
    return {
        "best_parameter_relative_rms": quality.best_parameter_relative_rms.tolist(),
        "best_parameter_signed_relative_error": quality.best_parameter_signed_relative_error.tolist(),
        "test_voltage_rmse_mV": quality.test_voltage_rmse_mv.tolist(),
        "test_protocol_voltage_rmse_mV": quality.test_protocol_voltage_rmse_mv.tolist(),
        "test_spike_count_exact_fraction": np.mean(quality.test_spike_count_exact, axis=1).tolist(),
        "test_spike_timing_exact_fraction": np.mean(quality.test_spike_timing_exact, axis=1).tolist(),
        "test_spike_max_timing_error_ms": _finite_values_or_none(quality.test_spike_max_timing_error_ms),
        "spike_timing_tolerance_ms": SPIKE_TIMING_TOLERANCE_MS,
    }


def _save_quality_figures(
    data: LoadedDataset,
    result: TrainingResult,
    quality: QualityDiagnostics,
    output_dir: Path,
) -> dict[str, Path]:
    figures = {
        "parameter_distance_to_target": _plot_parameter_distance(result, quality),
        "validation_quality_trajectories": _plot_validation_quality(result, quality),
        "test_spike_hard_metrics": _plot_test_spike_metrics(data, quality),
        "test_protocol_voltage_rmse": _plot_test_protocol_voltage_rmse(data, quality),
        "best_parameter_signed_error": _plot_best_parameter_signed_error(quality),
        "test_spike_timing_error": _plot_test_spike_timing_error(data, quality),
        "endpoint_pareto_summary": _plot_endpoint_pareto(quality),
    }
    paths = {}
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths[name] = path
    for start in range(8):
        figure = _plot_trace_atlas(data, result, start)
        path = output_dir / f"test_trace_fit_start_{start}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths[f"trace_start_{start}"] = path
    stale_parameter_paths = output_dir / "parameter_paths.png"
    stale_parameter_paths.unlink(missing_ok=True)
    (output_dir / "parameter_distance.png").unlink(missing_ok=True)
    (output_dir / "validation_quality.png").unlink(missing_ok=True)
    return paths


def save_quality_diagnostics(
    data: LoadedDataset,
    result: TrainingResult,
    quality: QualityDiagnostics,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Merge diagnostics into an existing run and redraw quality figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "training_results.npz"
    summary_path = output_dir / "summary.json"
    with np.load(arrays_path) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    arrays.update(_quality_array_values(quality))
    np.savez_compressed(arrays_path, **arrays)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("parameter_relative_rms", None)
    summary.update(_quality_summary_values(quality))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = {"arrays": arrays_path, "summary": summary_path}
    paths.update(_save_quality_figures(data, result, quality, output_dir))
    return paths


def save_results(
    data: LoadedDataset,
    result: TrainingResult,
    landscape: LandscapeResult,
    quality: QualityDiagnostics,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write numeric results, summary, and all requested figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "training_results.npz"
    arrays = dict(
        starts=result.starts,
        target_parameters=result.target_parameters,
        train_losses=result.train_losses,
        validation_losses=result.validation_losses,
        gradients=result.gradients,
        parameter_trajectories=result.parameter_trajectories,
        best_parameters=result.best_parameters,
        best_epochs=result.best_epochs,
        test_losses=result.test_losses,
        initial_test_traces=result.initial_test_traces,
        fitted_test_traces=result.fitted_test_traces,
        target_test_traces=result.target_test_traces,
        schedule=result.schedule,
        normalizers=result.normalizers,
        plane_losses=landscape.plane_losses,
        plane_spike_mismatch=landscape.plane_spike_mismatch,
        profile_parameters=landscape.profile_parameters,
        profile_losses=landscape.profile_losses,
        profile_spike_mismatch=landscape.profile_spike_mismatch,
    )
    arrays.update(_quality_array_values(quality))
    np.savez_compressed(arrays_path, **arrays)
    relative_error = np.abs(result.best_parameters - TARGET_PARAMETERS) / TARGET_PARAMETERS
    trace_rmse = np.sqrt(np.mean((result.fitted_test_traces - result.target_test_traces[None]) ** 2, axis=(1, 2, 3)))
    summary = {
        "backend": jax.default_backend(),
        "epochs": int(result.train_losses.shape[1]),
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": BATCHES_PER_EPOCH,
        "updates_per_start": int(result.train_losses.shape[1] * BATCHES_PER_EPOCH),
        "learning_rate": result.learning_rate,
        "random_seed": result.random_seed,
        "loss": "target-spike-window-masked voltage Huber",
        "huber_delta_mV": HUBER_DELTA_MV,
        "spike_window_ms": [-1.0, 3.0],
        "spike_window_weight": SPIKE_WINDOW_WEIGHT,
        "parameter_names": list(PARAMETER_NAMES),
        "best_epochs_zero_based": result.best_epochs.tolist(),
        "test_losses": result.test_losses.tolist(),
        "test_trace_rmse_mV": trace_rmse.tolist(),
        "mean_parameter_relative_error": np.mean(relative_error, axis=1).tolist(),
        "target_test_spike_counts": result.target_test_spike_counts.tolist(),
        "fitted_test_spike_counts": result.fitted_test_spike_counts.tolist(),
        "training_wall_seconds": result.wall_seconds,
    }
    summary.update(_quality_summary_values(quality))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figures = {
        "losses": _plot_losses(result),
        "landscape": _plot_landscape(result, landscape),
        "profiles": _plot_profiles(landscape),
    }
    paths = {"arrays": arrays_path, "summary": summary_path}
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths[name] = path
    paths.update(_save_quality_figures(data, result, quality, output_dir))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Reuse an existing training_results.npz and redraw quality diagnostics without training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the complete CPU experiment and save its artifacts."""
    args = parse_args(argv)
    if jax.default_backend() != "cpu":
        raise RuntimeError("Run this exploratory experiment with JAX_PLATFORMS=cpu.")
    data = load_dataset(args.dataset)
    with brainstate.environ.context(dt=dataset.DT):
        if args.diagnostics_only:
            result = load_saved_training_result(args.output_dir)
            quality = load_saved_quality_diagnostics(result, args.output_dir)
            paths = save_quality_diagnostics(data, result, quality, args.output_dir)
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
            return
        result = run_training(data, n_epochs=args.epochs, learning_rate=args.learning_rate)
        landscape = compute_landscape(data, result)
        quality = compute_quality_diagnostics(data, result)
    paths = save_results(data, result, landscape, quality, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
