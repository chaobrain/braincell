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

"""Train a two-dendrite HH cell on Jaxley's nonlinear pattern task."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import braincell
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.experimental.optim import build_trajectory_value_and_grad
from braincell.filter import AllRegion


DT_MS = 0.025
T_MAX_MS = 5.95
NUM_STEPS = int(round((T_MAX_MS + 2 * DT_MS) / DT_MS))
STIMULUS_START_MS = 1.0
STIMULUS_DURATION_MS = 0.9
STIMULUS_STOP_MS = STIMULUS_START_MS + STIMULUS_DURATION_MS
READOUT_TIME_MS = 3.0
READOUT_INDEX = 120
CURRENT_SCALE_NA = 0.05
SOFT_SPIKE_THRESHOLD_MV = 0.0
SOFT_PEAK_TEMPERATURE_MV = 2.0
SOFT_PROBABILITY_TEMPERATURE_MV = 5.0
PEAK_MARGIN_MV = 10.0
PEAK_MARGIN_TEMPERATURE_MV = 5.0
LOSS_KINDS = ("peak_margin", "soft_spike", "voltage_at_3ms")
DEFAULT_LOSS_KIND = "peak_margin"

N_CV_PER_BRANCH = 4
N_CV = 12
DEFAULT_TRAIN_SIZE = 32
DEFAULT_TEST_SIZE = 16
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_SWEEP_MIN = 0.0
DEFAULT_SWEEP_MAX = 5.0
DEFAULT_SWEEP_STEP = 0.1
DEFAULT_SWEEP_CHUNK_SIZE = 256
DEFAULT_BATCH_SIZE = 1
CLUSTER_CENTERS = ((1.5, 3.5), (2.5, 2.5), (3.5, 1.5))

CONDUCTANCE_UNIT = u.mS / u.cm**2
PARAMETER_BOUNDS = {
    "na.g_max": (50.0, 1100.0),
    "k.g_max": (10.0, 300.0),
    "leak.g_max": (0.1, 1.0),
}
TARGET_VOLTAGES_MV = (-70.0, 35.0)


@dataclass(frozen=True)
class PatternDataset:
    """Store train/test inputs and scalar voltage labels."""

    train_inputs: object
    train_targets_mv: object
    train_classes: object
    test_inputs: object
    test_targets_mv: object
    test_classes: object


@dataclass(frozen=True)
class ExperimentResult:
    """Store one seed's training history and endpoint predictions."""

    seed: int
    epochs: int
    loss_kind: str
    batch_size: int
    losses: object
    sample_losses: object
    initial_physical_parameters: dict[str, object]
    initial_train_traces_mv: object
    final_train_traces_mv: object
    final_test_traces_mv: object
    train_targets_mv: object
    test_targets_mv: object
    train_inputs: object
    test_inputs: object
    train_classes: object
    test_classes: object
    representative_inputs: object
    initial_representative_traces_mv: object
    final_representative_traces_mv: object
    sweep: "SweepResult"
    physical_parameters: dict[str, object]
    train_accuracy: object
    test_accuracy: object
    compile_and_train_seconds: float


@dataclass(frozen=True)
class SweepResult:
    """Store initial and final response metrics over a regular input grid."""

    x_values: object
    y_values: object
    initial_voltage_mv: object
    final_voltage_mv: object
    initial_spike_count: object
    final_spike_count: object
    initial_first_spike_ms: object
    final_first_spike_ms: object


def generate_dataset(
    *,
    seed: int,
    n_train: int = DEFAULT_TRAIN_SIZE,
    n_test: int = DEFAULT_TEST_SIZE,
    standard_deviation: float = 0.1,
) -> PatternDataset:
    """Generate the three collinear Gaussian input clusters used by Jaxley."""
    if n_train < 2 or n_train % 2 or n_test < 2 or n_test % 2:
        raise ValueError("n_train and n_test must be positive even integers of at least two.")
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    random = brainstate.random.RandomState(seed)

    def split(size: int):
        centers = jnp.asarray(CLUSTER_CENTERS, dtype=jnp.float64)
        outer_count = size // 2
        first_count = outer_count // 2
        third_count = outer_count - first_count
        first = random.normal(size=(first_count, 2), dtype=jnp.float64) * standard_deviation + centers[0]
        middle = random.normal(size=(size // 2, 2), dtype=jnp.float64) * standard_deviation + centers[1]
        third = random.normal(size=(third_count, 2), dtype=jnp.float64) * standard_deviation + centers[2]
        inputs = jnp.concatenate((first, third, middle), axis=0)
        classes = jnp.concatenate(
            (jnp.ones((outer_count,), dtype=jnp.int32), jnp.zeros((size // 2,), dtype=jnp.int32))
        )
        order = random.permutation(size)
        classes = classes[order]
        targets = jnp.where(classes == 1, TARGET_VOLTAGES_MV[1], TARGET_VOLTAGES_MV[0])
        return inputs[order], targets, classes

    train_inputs, train_targets, train_classes = split(n_train)
    test_inputs, test_targets, test_classes = split(n_test)
    return PatternDataset(
        train_inputs,
        train_targets,
        train_classes,
        test_inputs,
        test_targets,
        test_classes,
    )


def build_morphology() -> braincell.Morphology:
    """Build one root branch with two child dendrites."""
    branch_length = 10.5 * N_CV_PER_BRANCH
    soma = braincell.Branch.from_lengths(
        lengths=[branch_length] * u.um,
        radii=[2.55, 2.55] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    for name in ("dend_a", "dend_b"):
        morphology.attach(
            parent="soma",
            child_branch=braincell.Branch.from_lengths(
                lengths=[branch_length] * u.um,
                radii=[2.55, 2.55] * u.um,
                type="dendrite",
            ),
            child_name=name,
            parent_x=1.0,
        )
    return morphology


class NonlinearPatternExperiment:
    """Own the differentiable cell, dynamic stimulus, and exact-RTRL engine."""

    def __init__(
        self,
        *,
        seed: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        loss_kind: str = DEFAULT_LOSS_KIND,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if loss_kind not in LOSS_KINDS:
            raise ValueError(f"loss_kind must be one of {LOSS_KINDS!r}.")
        self.seed = int(seed)
        self.loss_kind = loss_kind
        self.cell = self._build_cell(seed=self.seed)
        self.parameters = self.cell.trainables.parameters()
        self.parameter_states = self.parameters.states()
        self.optimizer = braintools.optim.Adam(lr=learning_rate)
        self.optimizer.register_trainable_weights(self.parameter_states)
        self._attach_dynamic_stimulus()
        self.engine = build_trajectory_value_and_grad(
            self.cell,
            step=self._step,
            loss=self._loss,
            method="rtrl",
        )

    @staticmethod
    def _build_cell(*, seed: int) -> braincell.Cell:
        random = brainstate.random.RandomState(seed + 1_000)
        cell = braincell.Cell(
            build_morphology(),
            cv_policy=braincell.CVPerBranch(N_CV_PER_BRANCH),
            pop_size=(1,),
            V_init=-70.0 * u.mV,
            solver="staggered",
        )
        cell.paint(
            AllRegion(),
            braincell.mech.CableProperty(
                resting_potential=-70.0 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=3000.0 * u.ohm * u.cm,
            ),
            braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
            braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            braincell.mech.Channel("Na_HH1952", name="na"),
            braincell.mech.Channel("K_HH1952", name="k"),
            braincell.mech.Channel("IL", name="leak", E=-54.387 * u.mV),
        )
        cell.init_state()
        for channel_name in ("na", "k", "leak"):
            lower, upper = PARAMETER_BOUNDS[f"{channel_name}.g_max"]
            # Jaxley initializes every compartment from one sampled value per field.
            sampled = random.uniform(lower, upper, dtype=jnp.float64)
            initial = jnp.full((N_CV,), sampled, dtype=jnp.float64) * CONDUCTANCE_UNIT
            cell.channels[channel_name].trainable(
                g_max=braincell.trainable.parameter(
                    initial,
                    group_by="cv",
                    transform=brainstate.nn.SigmoidT(
                        lower * CONDUCTANCE_UNIT,
                        upper * CONDUCTANCE_UNIT,
                    ),
                    name=f"{channel_name}.g_max",
                )
            )
        cell.reset_state()
        if cell.n_cv != N_CV:
            raise RuntimeError(f"Expected {N_CV} CVs, got {cell.n_cv}.")
        return cell

    def _attach_dynamic_stimulus(self) -> None:
        branch_to_cvs = self.cell.cv_tree.branch_to_cv_ids
        stimulus_cvs = np.asarray((branch_to_cvs[1][-1], branch_to_cvs[2][-1]), dtype=np.int32)
        self.stimulus_point_ids = jnp.asarray(self.cell.node_tree.cv_to_mid_node_id[stimulus_cvs])
        self.soma_cv_id = int(branch_to_cvs[0][0])
        self._stimulus_density = brainstate.ShortTermState(
            jnp.zeros((1, self.cell.n_point), dtype=jnp.float64) * u.nA / u.cm**2
        )
        self._stimulus_area_cm2 = self.cell.runtime.point_area.to_decimal(u.cm**2)[self.stimulus_point_ids]
        self.cell.add_current_input("nonlinear_pattern", lambda _point_voltage: self._stimulus_density.value)

    def step_data(self, inputs, target_mv):
        """Return fixed-shape time, two-site current, and repeated target arrays."""
        times_ms = jnp.arange(NUM_STEPS, dtype=jnp.float64) * DT_MS
        active = (times_ms >= STIMULUS_START_MS) & (times_ms < STIMULUS_STOP_MS)
        currents_na = active[:, None] * (CURRENT_SCALE_NA * jnp.asarray(inputs)[None, :])
        targets_mv = jnp.full((NUM_STEPS,), target_mv, dtype=jnp.float64)
        return times_ms, currents_na, targets_mv

    def _step(self, data):
        time_ms, currents_na, _target_mv = data
        density = jnp.zeros((1, self.cell.n_point), dtype=jnp.float64)
        density = density.at[0, self.stimulus_point_ids].set(currents_na / self._stimulus_area_cm2)
        self._stimulus_density.value = density * u.nA / u.cm**2
        with brainstate.environ.context(t=time_ms * u.ms):
            self.cell.update()
        return self.cell.V.value.to_decimal(u.mV)[0, self.soma_cv_id]

    def _loss(self, observations_mv, data):
        target_mv = data[2][0]
        if self.loss_kind == "voltage_at_3ms":
            return jnp.abs(observations_mv[READOUT_INDEX] - target_mv)
        target_spike = (target_mv == TARGET_VOLTAGES_MV[1]).astype(observations_mv.dtype)
        if self.loss_kind == "peak_margin":
            signed_target = 2.0 * target_spike - 1.0
            signed_margin = signed_target * soft_peak_voltage(observations_mv)
            return jax.nn.softplus(
                (PEAK_MARGIN_MV - signed_margin) / PEAK_MARGIN_TEMPERATURE_MV
            )
        logit = soft_spike_logit(observations_mv)
        return jax.nn.softplus(logit) - target_spike * logit

    def simulate(self, inputs, target_mv=0.0):
        """Return the soma trace for one input without computing gradients."""
        self.cell.trainables.materialize()
        self.cell.reset_state()
        data = self.step_data(inputs, target_mv)
        return brainstate.transform.for_loop(self._step, data)

    def simulate_dataset(self, inputs, targets_mv):
        """Return one independently reset soma trace per input sample."""
        def one_sample(sample):
            values, target = sample
            return self.simulate(values, target)

        return brainstate.transform.for_loop(one_sample, (inputs, targets_mv))

    def simulate_sweep(self, inputs, *, chunk_size: int = DEFAULT_SWEEP_CHUNK_SIZE):
        """Evaluate voltage and spike metrics in fixed-size compiled chunks."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        inputs = jnp.asarray(inputs, dtype=jnp.float64)
        if inputs.ndim != 2 or inputs.shape[1] != 2 or inputs.shape[0] < 1:
            raise ValueError("inputs must have shape (n, 2) with n >= 1.")
        n_inputs = inputs.shape[0]
        n_chunks = (n_inputs + chunk_size - 1) // chunk_size
        n_padding = n_chunks * chunk_size - n_inputs
        padded = jnp.pad(inputs, ((0, n_padding), (0, 0)), mode="edge")
        chunks = padded.reshape((n_chunks, chunk_size, 2))

        def one_chunk(chunk):
            traces = self.simulate_dataset(chunk, jnp.zeros((chunk_size,), dtype=jnp.float64))
            crossings = (traces[:, :-1] < 0.0) & (traces[:, 1:] >= 0.0)
            has_spike = jnp.any(crossings, axis=1)
            first_spike = (jnp.argmax(crossings, axis=1) + 1) * DT_MS
            first_spike = jnp.where(has_spike, first_spike, jnp.nan)
            return traces[:, READOUT_INDEX], jnp.sum(crossings, axis=1), first_spike

        voltage, count, first_spike = brainstate.transform.for_loop(one_chunk, chunks)
        return (
            voltage.reshape((-1,))[:n_inputs],
            count.reshape((-1,))[:n_inputs],
            first_spike.reshape((-1,))[:n_inputs],
        )

    def train(
        self,
        dataset: PatternDataset,
        *,
        epochs: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[object, object]:
        """Average independent exact-RTRL gradients before each Adam update."""
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        random = brainstate.random.RandomState(self.seed + 2_000)
        batches = build_training_batches(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            random=random,
        )
        example = self.step_data(dataset.train_inputs[0], dataset.train_targets_mv[0])
        self.engine.prepare(jax.tree.map(lambda value: value[0], example))

        def sample_gradient(index):
            data = self.step_data(dataset.train_inputs[index], dataset.train_targets_mv[index])
            result = self.engine(data)
            return result.loss, result.gradients

        def train_step(indices):
            sample_losses, gradients = brainstate.transform.for_loop(sample_gradient, indices)
            mean_gradients = jax.tree.map(lambda value: jnp.mean(value, axis=0), gradients)
            self.optimizer.update(mean_gradients)
            return jnp.mean(sample_losses), sample_losses

        batch_losses, sample_losses = brainstate.transform.for_loop(train_step, batches)
        return batch_losses, sample_losses.reshape((-1,))


def build_training_batches(
    dataset: PatternDataset,
    *,
    epochs: int,
    batch_size: int,
    random: brainstate.random.RandomState,
):
    """Build shuffled batches, using a 1:2:1 cluster ratio for batches above one."""
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    n_train = int(dataset.train_inputs.shape[0])
    if batch_size < 1 or n_train % batch_size:
        raise ValueError("batch_size must be positive and divide the training-set size.")
    if batch_size == 1:
        orders = jnp.stack([random.permutation(n_train) for _ in range(epochs)])
        return orders.reshape((-1, 1))
    if batch_size % 4:
        raise ValueError("Stratified batch_size must be a multiple of four.")

    inputs = np.asarray(dataset.train_inputs)
    classes = np.asarray(dataset.train_classes)
    groups = (
        np.flatnonzero((classes == 1) & (inputs[:, 0] < inputs[:, 1])),
        np.flatnonzero(classes == 0),
        np.flatnonzero((classes == 1) & (inputs[:, 0] > inputs[:, 1])),
    )
    expected_sizes = (n_train // 4, n_train // 2, n_train // 4)
    if tuple(group.size for group in groups) != expected_sizes:
        raise ValueError("Dataset must contain left/middle/right clusters in a 1:2:1 ratio.")

    n_batches = n_train // batch_size
    group_batch_sizes = (batch_size // 4, batch_size // 2, batch_size // 4)
    epoch_batches = []
    for _ in range(epochs):
        shuffled_groups = [
            jnp.asarray(group)[random.permutation(group.size)].reshape((n_batches, group_batch_size))
            for group, group_batch_size in zip(groups, group_batch_sizes)
        ]
        batches = jnp.concatenate(shuffled_groups, axis=1)
        within_batch_order = jax.vmap(random.permutation)(batches)
        epoch_batches.append(within_batch_order)
    return jnp.concatenate(epoch_batches, axis=0)


def run_experiment(
    *,
    seed: int = 0,
    epochs: int = DEFAULT_EPOCHS,
    n_train: int = DEFAULT_TRAIN_SIZE,
    n_test: int = DEFAULT_TEST_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    loss_kind: str = DEFAULT_LOSS_KIND,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sweep_min: float = DEFAULT_SWEEP_MIN,
    sweep_max: float = DEFAULT_SWEEP_MAX,
    sweep_step: float = DEFAULT_SWEEP_STEP,
    sweep_chunk_size: int = DEFAULT_SWEEP_CHUNK_SIZE,
) -> ExperimentResult:
    """Run one x64 exact-RTRL nonlinear-pattern experiment."""
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        dataset = generate_dataset(seed=seed, n_train=n_train, n_test=n_test)
        experiment = NonlinearPatternExperiment(seed=seed, learning_rate=learning_rate, loss_kind=loss_kind)
        initial_physical = {
            name: value.to_decimal(CONDUCTANCE_UNIT)
            for name, value in experiment.parameters.physical_values().items()
        }
        initial_train = experiment.simulate_dataset(dataset.train_inputs, dataset.train_targets_mv)
        representative_inputs = jnp.asarray(CLUSTER_CENTERS, dtype=jnp.float64)
        initial_representative = experiment.simulate_dataset(representative_inputs, jnp.zeros((3,)))
        x_values, y_values, sweep_inputs = generate_sweep_grid(
            minimum=sweep_min,
            maximum=sweep_max,
            step=sweep_step,
        )
        initial_sweep = experiment.simulate_sweep(sweep_inputs, chunk_size=sweep_chunk_size)
        started = time.perf_counter()
        losses, sample_losses = experiment.train(dataset, epochs=epochs, batch_size=batch_size)
        jax.block_until_ready((losses, sample_losses))
        elapsed = time.perf_counter() - started
        final_train = experiment.simulate_dataset(dataset.train_inputs, dataset.train_targets_mv)
        final_test = experiment.simulate_dataset(dataset.test_inputs, dataset.test_targets_mv)
        final_representative = experiment.simulate_dataset(representative_inputs, jnp.zeros((3,)))
        final_sweep = experiment.simulate_sweep(sweep_inputs, chunk_size=sweep_chunk_size)
        train_accuracy = classification_accuracy(final_train[:, READOUT_INDEX], dataset.train_classes)
        test_accuracy = classification_accuracy(final_test[:, READOUT_INDEX], dataset.test_classes)
        physical = {
            name: value.to_decimal(CONDUCTANCE_UNIT)
            for name, value in experiment.parameters.physical_values().items()
        }
        return ExperimentResult(
            seed=seed,
            epochs=epochs,
            loss_kind=loss_kind,
            batch_size=batch_size,
            losses=losses,
            sample_losses=sample_losses,
            initial_physical_parameters=initial_physical,
            initial_train_traces_mv=initial_train,
            final_train_traces_mv=final_train,
            final_test_traces_mv=final_test,
            train_targets_mv=dataset.train_targets_mv,
            test_targets_mv=dataset.test_targets_mv,
            train_inputs=dataset.train_inputs,
            test_inputs=dataset.test_inputs,
            train_classes=dataset.train_classes,
            test_classes=dataset.test_classes,
            representative_inputs=representative_inputs,
            initial_representative_traces_mv=initial_representative,
            final_representative_traces_mv=final_representative,
            sweep=SweepResult(
                x_values=x_values,
                y_values=y_values,
                initial_voltage_mv=initial_sweep[0].reshape((y_values.size, x_values.size)),
                final_voltage_mv=final_sweep[0].reshape((y_values.size, x_values.size)),
                initial_spike_count=initial_sweep[1].reshape((y_values.size, x_values.size)),
                final_spike_count=final_sweep[1].reshape((y_values.size, x_values.size)),
                initial_first_spike_ms=initial_sweep[2].reshape((y_values.size, x_values.size)),
                final_first_spike_ms=final_sweep[2].reshape((y_values.size, x_values.size)),
            ),
            physical_parameters=physical,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            compile_and_train_seconds=elapsed,
        )


def generate_sweep_grid(*, minimum: float, maximum: float, step: float):
    """Return inclusive regular axes and flattened ``(x1, x2)`` inputs."""
    if step <= 0.0:
        raise ValueError("step must be positive.")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum.")
    intervals = (maximum - minimum) / step
    rounded_intervals = int(round(intervals))
    if not np.isclose(intervals, rounded_intervals, rtol=0.0, atol=1e-10):
        raise ValueError("The sweep range must be an integer multiple of step.")
    axis = jnp.linspace(minimum, maximum, rounded_intervals + 1, dtype=jnp.float64)
    x_grid, y_grid = jnp.meshgrid(axis, axis, indexing="xy")
    inputs = jnp.stack((x_grid.reshape((-1,)), y_grid.reshape((-1,))), axis=1)
    return axis, axis, inputs


def soft_spike_logit(voltage_mv):
    """Return a smooth full-trace spike-presence logit."""
    voltage_mv = jnp.asarray(voltage_mv)
    if voltage_mv.ndim != 1 or voltage_mv.size < 1:
        raise ValueError("voltage_mv must be a non-empty one-dimensional trace.")
    scaled = voltage_mv / SOFT_PEAK_TEMPERATURE_MV
    smooth_peak_mv = SOFT_PEAK_TEMPERATURE_MV * (
        jax.scipy.special.logsumexp(scaled) - jnp.log(voltage_mv.size)
    )
    return (smooth_peak_mv - SOFT_SPIKE_THRESHOLD_MV) / SOFT_PROBABILITY_TEMPERATURE_MV


def soft_peak_voltage(voltage_mv):
    """Return a differentiable peak estimate without fixing its time."""
    voltage_mv = jnp.asarray(voltage_mv)
    if voltage_mv.ndim != 1 or voltage_mv.size < 1:
        raise ValueError("voltage_mv must be a non-empty one-dimensional trace.")
    weights = jax.nn.softmax(voltage_mv / SOFT_PEAK_TEMPERATURE_MV)
    return jnp.sum(weights * voltage_mv)


def soft_spike_probability(voltage_mv):
    """Return the differentiable probability that a trace contains a spike."""
    return jax.nn.sigmoid(soft_spike_logit(voltage_mv))


def classification_accuracy(readout_voltage_mv, classes):
    """Classify by the midpoint between the two voltage targets."""
    threshold = 0.5 * sum(TARGET_VOLTAGES_MV)
    predictions = jnp.asarray(readout_voltage_mv) > threshold
    return jnp.mean(predictions == jnp.asarray(classes))


def spike_counts(traces_mv):
    """Count upward zero-mV crossings in every voltage trace."""
    traces = jnp.asarray(traces_mv)
    return jnp.sum((traces[:, :-1] < 0.0) & (traces[:, 1:] >= 0.0), axis=1)


def spike_presence_accuracy(traces_mv, target_voltage_mv):
    """Compare observed spike presence with the high/low target class."""
    expected = jnp.asarray(target_voltage_mv) == TARGET_VOLTAGES_MV[1]
    return jnp.mean((spike_counts(traces_mv) > 0) == expected)


def save_result(result: ExperimentResult, output_dir: Path) -> None:
    """Write array history and a compact JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "loss": np.asarray(result.losses),
        "sample_loss": np.asarray(result.sample_losses),
        "epoch_loss": np.asarray(result.sample_losses).reshape((result.epochs, -1)).mean(axis=1),
        "initial_train_traces_mv": np.asarray(result.initial_train_traces_mv),
        "final_train_traces_mv": np.asarray(result.final_train_traces_mv),
        "final_test_traces_mv": np.asarray(result.final_test_traces_mv),
        "train_targets_mv": np.asarray(result.train_targets_mv),
        "test_targets_mv": np.asarray(result.test_targets_mv),
        "train_inputs": np.asarray(result.train_inputs),
        "test_inputs": np.asarray(result.test_inputs),
        "train_classes": np.asarray(result.train_classes),
        "test_classes": np.asarray(result.test_classes),
        "representative_inputs": np.asarray(result.representative_inputs),
        "initial_representative_traces_mv": np.asarray(result.initial_representative_traces_mv),
        "final_representative_traces_mv": np.asarray(result.final_representative_traces_mv),
        **{
            f"initial_parameter/{name}": np.asarray(value)
            for name, value in result.initial_physical_parameters.items()
        },
        **{f"parameter/{name}": np.asarray(value) for name, value in result.physical_parameters.items()},
    }
    np.savez_compressed(output_dir / "history.npz", **arrays)
    np.savez_compressed(
        output_dir / "sweep.npz",
        x_values=np.asarray(result.sweep.x_values),
        y_values=np.asarray(result.sweep.y_values),
        initial_voltage_mv=np.asarray(result.sweep.initial_voltage_mv),
        final_voltage_mv=np.asarray(result.sweep.final_voltage_mv),
        initial_spike_count=np.asarray(result.sweep.initial_spike_count),
        final_spike_count=np.asarray(result.sweep.final_spike_count),
        initial_first_spike_ms=np.asarray(result.sweep.initial_first_spike_ms),
        final_first_spike_ms=np.asarray(result.sweep.final_first_spike_ms),
    )
    initial_readout = np.asarray(result.initial_train_traces_mv)[:, READOUT_INDEX]
    final_readout = np.asarray(result.final_train_traces_mv)[:, READOUT_INDEX]
    train_targets = np.asarray(result.train_targets_mv)
    initial_train_traces = np.asarray(result.initial_train_traces_mv)
    final_train_traces = np.asarray(result.final_train_traces_mv)
    final_test_traces = np.asarray(result.final_test_traces_mv)
    initial_train_spikes = _numpy_spike_counts(initial_train_traces)
    final_train_spikes = _numpy_spike_counts(final_train_traces)
    final_test_spikes = _numpy_spike_counts(final_test_traces)
    expected_train_spikes = train_targets == TARGET_VOLTAGES_MV[1]
    expected_test_spikes = np.asarray(result.test_targets_mv) == TARGET_VOLTAGES_MV[1]
    summary = {
        "seed": result.seed,
        "gradient_method": "exact_full_rtrl",
        "loss_kind": result.loss_kind,
        "loss_unit": "mV" if result.loss_kind == "voltage_at_3ms" else "dimensionless",
        "soft_spike_threshold_mv": SOFT_SPIKE_THRESHOLD_MV,
        "soft_peak_temperature_mv": SOFT_PEAK_TEMPERATURE_MV,
        "soft_probability_temperature_mv": SOFT_PROBABILITY_TEMPERATURE_MV,
        "peak_margin_mv": PEAK_MARGIN_MV,
        "peak_margin_temperature_mv": PEAK_MARGIN_TEMPERATURE_MV,
        "precision": 64,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "batch_size": result.batch_size,
        "epochs": result.epochs,
        "train_size": int(np.asarray(result.train_inputs).shape[0]),
        "test_size": int(np.asarray(result.test_inputs).shape[0]),
        "dt_ms": DT_MS,
        "stimulus_ms": [STIMULUS_START_MS, STIMULUS_STOP_MS],
        "readout_time_ms": READOUT_TIME_MS,
        "parameter_count": N_CV * len(PARAMETER_BOUNDS),
        "updates": int(np.asarray(result.losses).size),
        "trajectory_evaluations": int(np.asarray(result.sample_losses).size),
        "final_epoch_loss": float(
            np.asarray(result.sample_losses).reshape((result.epochs, -1))[-1].mean()
        ),
        "initial_train_mae_mv": float(np.mean(np.abs(initial_readout - train_targets))),
        "final_train_mae_mv": float(np.mean(np.abs(final_readout - train_targets))),
        "train_accuracy": float(result.train_accuracy),
        "test_accuracy": float(result.test_accuracy),
        "train_voltage_accuracy": float(result.train_accuracy),
        "test_voltage_accuracy": float(result.test_accuracy),
        "initial_train_spike_presence_accuracy": float(
            np.mean((initial_train_spikes > 0) == expected_train_spikes)
        ),
        "final_train_spike_presence_accuracy": float(
            np.mean((final_train_spikes > 0) == expected_train_spikes)
        ),
        "final_test_spike_presence_accuracy": float(
            np.mean((final_test_spikes > 0) == expected_test_spikes)
        ),
        "initial_train_spike_counts": initial_train_spikes.tolist(),
        "final_train_spike_counts": final_train_spikes.tolist(),
        "final_test_spike_counts": final_test_spikes.tolist(),
        "compile_and_train_seconds": result.compile_and_train_seconds,
        "array_file": "history.npz",
        "sweep_file": "sweep.npz",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _numpy_spike_counts(traces_mv: np.ndarray) -> np.ndarray:
    return np.sum((traces_mv[:, :-1] < 0.0) & (traces_mv[:, 1:] >= 0.0), axis=1)
