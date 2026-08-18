#!/usr/bin/env python3
"""Provide example-local conductance-fitting infrastructure.

The active experiment contracts live in
``docs/specs/2026-08-17-heterogeneous-protocol-dataset.md`` and
``docs/specs/2026-08-17-heterogeneous-nine-parameter-training.md``. The
channel adapters in this module are deliberately not part of BrainCell's
public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import mech
from braincell.channel import IL, K_HH1952, Na_HH1952
from braincell.filter import AllRegion, at


DT = 0.025 * u.ms
DURATION = 100.0 * u.ms
CONDUCTANCE_UNIT = u.mS / u.cm**2
PARAMETER_NAMES = ("leak_g_max", "na_g_max", "k_g_max")
COMPONENT_NAMES = ("voltage", "derivative", "multiscale", "event", "count", "peak")

TARGET_PARAMETERS = np.asarray([0.6, 120.0, 36.0], dtype=float)
CANONICAL_INITIAL_PARAMETERS = np.asarray([0.2, 70.0, 65.0], dtype=float)
LOWER_BOUNDS = np.asarray([0.05, 20.0, 5.0], dtype=float)
UPPER_BOUNDS = np.asarray([1.2, 220.0, 100.0], dtype=float)
COMPOSITE_WEIGHTS = jnp.asarray([1.0, 0.10, 0.25, 0.75, 0.40, 2.0])

STIMULUS_DELAY = 10.0 * u.ms
STIMULUS_DURATIONS = np.asarray([10, 10, 10, 10, 10, 10, 15, 15], dtype=float) * u.ms
STIMULUS_AMPLITUDES = np.asarray([0.02, 0.0, 0.05, 0.0, 0.075, 0.0, 0.20, 0.0]) * u.nA
SPIKE_THRESHOLD_MV = 0.0
SPIKE_SEGMENT = (70.0, 85.0)


def _shared_scalar_parameter(owner, g_max, lower, upper, fit: bool) -> None:
    dense = np.asarray(g_max.to_decimal(CONDUCTANCE_UNIT), dtype=float)
    active_mask = dense != 0.0
    if not np.any(active_mask):
        raise ValueError("A trainable conductance requires at least one active point.")
    active_values = dense[active_mask]
    if not np.allclose(active_values, active_values[0]):
        raise ValueError("A shared conductance requires one active g_max value.")
    owner._g_max_mask = jnp.asarray(active_mask)
    owner.g_max = brainstate.nn.Param(
        float(active_values[0]) * CONDUCTANCE_UNIT,
        t=brainstate.nn.SigmoidT(lower * CONDUCTANCE_UNIT, upper * CONDUCTANCE_UNIT),
        fit=fit,
    )


@mech.register_channel("ExplorationTrainableLeak")
class ExplorationTrainableLeak(IL):
    """Represent one all-compartment shared leak conductance."""

    def __init__(self, size, g_max=CANONICAL_INITIAL_PARAMETERS[0] * CONDUCTANCE_UNIT, E=-54.387 * u.mV, name=None):
        super().__init__(size=size, g_max=g_max, E=E, name=name)
        _shared_scalar_parameter(self, self.g_max, LOWER_BOUNDS[0], UPPER_BOUNDS[0], True)

    def current(self, V):
        """Return masked leak current density."""
        return self.g_max.value() * self._g_max_mask * (self.E - V)


@mech.register_channel("ExplorationFrozenLeak")
class ExplorationFrozenLeak(ExplorationTrainableLeak):
    """Represent the frozen target leak conductance."""

    def __init__(self, size, g_max=TARGET_PARAMETERS[0] * CONDUCTANCE_UNIT, E=-54.387 * u.mV, name=None):
        IL.__init__(self, size=size, g_max=g_max, E=E, name=name)
        _shared_scalar_parameter(self, self.g_max, LOWER_BOUNDS[0], UPPER_BOUNDS[0], False)


@mech.register_channel("ExplorationTrainableNa")
class ExplorationTrainableNa(Na_HH1952):
    """Represent one all-compartment shared sodium conductance."""

    def __init__(self, size, g_max=CANONICAL_INITIAL_PARAMETERS[1] * CONDUCTANCE_UNIT, name=None):
        super().__init__(size=size, g_max=g_max, name=name)
        _shared_scalar_parameter(self, self.g_max, LOWER_BOUNDS[1], UPPER_BOUNDS[1], True)

    def current(self, V, Na):
        """Return masked sodium current density."""
        return self.g_max.value() * self._g_max_mask * self.conductance_factor(V, Na) * (Na.E - V)


@mech.register_channel("ExplorationFrozenNa")
class ExplorationFrozenNa(ExplorationTrainableNa):
    """Represent the frozen target sodium conductance."""

    def __init__(self, size, g_max=TARGET_PARAMETERS[1] * CONDUCTANCE_UNIT, name=None):
        Na_HH1952.__init__(self, size=size, g_max=g_max, name=name)
        _shared_scalar_parameter(self, self.g_max, LOWER_BOUNDS[1], UPPER_BOUNDS[1], False)


@mech.register_channel("ExplorationTrainableK")
class ExplorationTrainableK(K_HH1952):
    """Represent one all-compartment shared potassium conductance."""

    def __init__(self, size, g_max=CANONICAL_INITIAL_PARAMETERS[2] * CONDUCTANCE_UNIT, name=None):
        super().__init__(size=size, g_max=g_max, name=name)
        _shared_scalar_parameter(self, self.g_max, LOWER_BOUNDS[2], UPPER_BOUNDS[2], True)

    def current(self, V, K):
        """Return masked potassium current density."""
        return self.g_max.value() * self._g_max_mask * self.conductance_factor(V, K) * (K.E - V)


@mech.register_channel("ExplorationFrozenK")
class ExplorationFrozenK(ExplorationTrainableK):
    """Represent the frozen target potassium conductance."""

    def __init__(self, size, g_max=TARGET_PARAMETERS[2] * CONDUCTANCE_UNIT, name=None):
        K_HH1952.__init__(self, size=size, g_max=g_max, name=name)
        _shared_scalar_parameter(self, self.g_max, LOWER_BOUNDS[2], UPPER_BOUNDS[2], False)


@dataclass(frozen=True)
class SimulationOutput:
    """Store voltage and smooth event traces from one rollout."""

    voltages: object
    smooth_events: object


@dataclass(frozen=True)
class TrainingProblem:
    """Own target data, fitted cell, selected parameters, and fixed loss data."""

    target_cell: braincell.Cell
    fitted_cell: braincell.Cell
    target: SimulationOutput
    parameters: tuple
    target_mask: object
    normalizers: object
    voltage_mse_normalizer: object

    @property
    def param_states(self) -> dict[str, brainstate.ParamState]:
        """Return the three explicitly selected optimizer states."""
        return {name: parameter.val for name, parameter in zip(PARAMETER_NAMES, self.parameters)}

    def physical_parameters(self):
        """Return the three conductances in canonical numeric units."""
        return jnp.stack([parameter.value().to_decimal(CONDUCTANCE_UNIT) for parameter in self.parameters])

    def set_physical_parameters(self, values) -> None:
        """Assign three values expressed in ``mS/cm^2``."""
        values = jnp.asarray(values)
        for parameter, value in zip(self.parameters, values):
            parameter.set_value(value * CONDUCTANCE_UNIT)

    def loss_with_aux(self, mode: Literal["voltage", "composite"] = "composite"):
        """Return one scalar loss and normalized component vector."""
        prediction = simulate(self.fitted_cell)
        raw = raw_loss_components(prediction, self.target, self.target_mask)
        normalized = raw / self.normalizers
        if mode == "voltage":
            error = prediction.voltages.to_decimal(u.mV) - self.target.voltages.to_decimal(u.mV)
            total = jnp.mean(error**2) / self.voltage_mse_normalizer
        elif mode == "composite":
            total = jnp.sum(COMPOSITE_WEIGHTS * normalized) / jnp.sum(COMPOSITE_WEIGHTS)
        else:
            raise ValueError(f"Unknown loss mode {mode!r}.")
        return total, normalized


@dataclass(frozen=True)
class TrainingConfig:
    """Describe one optimizer and parameter-release experiment."""

    name: str
    optimizer: Literal["adam", "radam"] = "adam"
    loss_mode: Literal["voltage", "composite"] = "composite"
    n_epochs: int = 180
    learning_rate: float = 0.02
    staged: bool = False


@dataclass(frozen=True)
class TrainingResult:
    """Store complete diagnostics for one conductance-fitting run."""

    config: TrainingConfig
    target_traces: object
    initial_traces: object
    fitted_traces: object
    losses: object
    component_losses: object
    gradients: object
    gradient_norms: object
    parameter_trajectory: object
    stage_masks: object
    initial_parameters: object
    fitted_parameters: object
    target_parameters: object


def build_morphology() -> braincell.Morphology:
    """Build the symmetric three-branch morphology used by the shared fit."""
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend_a = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    dend_b = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dend_a = dend_a
    morphology.soma.dend_b = dend_b
    return morphology


def build_cell(parameters, *, trainable: bool) -> braincell.Cell:
    """Build the shared-conductance three-compartment experiment cell."""
    values = np.asarray(parameters, dtype=float)
    if values.shape != (3,):
        raise ValueError(f"parameters must have shape (3,), got {values.shape!r}.")
    cell = braincell.Cell(
        build_morphology(),
        cv_policy=braincell.CVPerBranch(),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    cell.paint(
        AllRegion(),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel(
            "ExplorationTrainableNa" if trainable else "ExplorationFrozenNa",
            name="shared_na",
            g_max=values[1] * CONDUCTANCE_UNIT,
        ),
        mech.Channel(
            "ExplorationTrainableK" if trainable else "ExplorationFrozenK",
            name="shared_k",
            g_max=values[2] * CONDUCTANCE_UNIT,
        ),
        mech.Channel(
            "ExplorationTrainableLeak" if trainable else "ExplorationFrozenLeak",
            name="shared_leak",
            g_max=values[0] * CONDUCTANCE_UNIT,
            E=-54.387 * u.mV,
        ),
    )
    cell.place(
        at("soma", 0.5),
        mech.CurrentClamp(
            delay=STIMULUS_DELAY,
            durations=STIMULUS_DURATIONS,
            amplitudes=STIMULUS_AMPLITUDES,
        ),
    )
    cell.place(at("soma", 0.5), mech.StateProbe(name="soma_v", field="v"))
    cell.place(at("dend_a", 0.5), mech.StateProbe(name="dend_a_v", field="v"))
    cell.init_state()
    return cell


def _find_parameter(cell: braincell.Cell, instance_name: str):
    for layout in cell.layouts:
        declaration = cell.runtime.get_layout_mechanism(layout.id)
        if getattr(declaration, "instance_name", None) == instance_name:
            node = cell.get_runtime_node(layout.id)
            if hasattr(node, "g_max") and isinstance(node.g_max, brainstate.nn.Param):
                return node.g_max
    raise LookupError(f"Could not find runtime parameter for {instance_name!r}.")


def find_parameters(cell: braincell.Cell) -> tuple:
    """Return leak, sodium, and potassium parameters in canonical order."""
    return (
        _find_parameter(cell, "shared_leak"),
        _find_parameter(cell, "shared_na"),
        _find_parameter(cell, "shared_k"),
    )


def smooth_crossings(voltage_mv, temperature_mv: float = 2.0):
    """Return differentiable upward zero-threshold crossing strengths."""
    voltage_mv = jnp.asarray(voltage_mv)
    previous = jnp.concatenate((voltage_mv[:1], voltage_mv[:-1]), axis=0)
    rising_from_below = jax.nn.sigmoid((SPIKE_THRESHOLD_MV - previous) / temperature_mv)
    ending_above = jax.nn.sigmoid((voltage_mv - SPIKE_THRESHOLD_MV) / temperature_mv)
    return rising_from_below * ending_above


def exponential_event_filter(events, tau_ms: float = 2.0):
    """Filter a smooth event trace with a causal exponential kernel."""
    decay = jnp.exp(-DT.to_decimal(u.ms) / tau_ms)

    def step(carry, event):
        carry = decay * carry + event
        return carry, carry

    _, filtered = brainstate.transform.scan(step, jnp.zeros(events.shape[1:]), events)
    return filtered


def simulate(cell: braincell.Cell) -> SimulationOutput:
    """Reset all cell state and run one deterministic stimulation protocol."""
    cell.reset_state()
    result = cell.run(dt=DT, duration=DURATION)
    voltages = u.math.stack((result.traces["soma_v"], result.traces["dend_a_v"]), axis=-1)
    soma_mv = voltages[:, 0].to_decimal(u.mV)
    events = exponential_event_filter(smooth_crossings(soma_mv)[:, None])[:, 0]
    return SimulationOutput(voltages=voltages, smooth_events=events)


def hard_spike_indices(voltages) -> np.ndarray:
    """Return rising zero-millivolt crossings from a soma voltage trace."""
    soma = np.asarray(voltages.to_decimal(u.mV))[:, 0]
    return np.flatnonzero((soma[:-1] < SPIKE_THRESHOLD_MV) & (soma[1:] >= SPIKE_THRESHOLD_MV)) + 1


def target_weight_mask(target_voltages) -> jax.Array:
    """Build a fixed low-weight window around target spikes."""
    weights = np.ones(target_voltages.shape[0], dtype=float)
    before = int(round(1.0 / DT.to_decimal(u.ms)))
    after = int(round(3.0 / DT.to_decimal(u.ms)))
    for index in hard_spike_indices(target_voltages):
        weights[max(0, index - before) : min(weights.size, index + after + 1)] = 0.1
    return jnp.asarray(weights)


def _huber(error, delta: float = 2.0):
    absolute = jnp.abs(error)
    return jnp.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta))


def _smooth_peak(voltage_mv):
    start = int(round(SPIKE_SEGMENT[0] / DT.to_decimal(u.ms)))
    stop = int(round(SPIKE_SEGMENT[1] / DT.to_decimal(u.ms)))
    beta = 0.2
    return jax.scipy.special.logsumexp(beta * voltage_mv[start:stop]) / beta


def raw_loss_components(prediction: SimulationOutput, target: SimulationOutput, target_mask):
    """Return the six unnormalized, differentiable loss components."""
    predicted_mv = prediction.voltages.to_decimal(u.mV)
    target_mv = target.voltages.to_decimal(u.mV)
    weights = target_mask[:, None]
    voltage = jnp.sum(weights * _huber(predicted_mv - target_mv)) / (jnp.sum(weights) * predicted_mv.shape[1])

    predicted_dv = jnp.diff(predicted_mv, axis=0)
    target_dv = jnp.diff(target_mv, axis=0)
    derivative_weights = jnp.minimum(target_mask[:-1], target_mask[1:])[:, None]
    derivative = jnp.sum(derivative_weights * _huber(predicted_dv - target_dv, delta=0.5))
    derivative /= jnp.sum(derivative_weights) * predicted_mv.shape[1]

    block = 20
    n_blocks = predicted_mv.shape[0] // block
    predicted_coarse = predicted_mv[: n_blocks * block].reshape(n_blocks, block, 2).mean(axis=1)
    target_coarse = target_mv[: n_blocks * block].reshape(n_blocks, block, 2).mean(axis=1)
    multiscale = jnp.mean(_huber(predicted_coarse - target_coarse))

    event = jnp.mean((prediction.smooth_events - target.smooth_events) ** 2)
    predicted_crossings = smooth_crossings(predicted_mv[:, 0])
    target_crossings = smooth_crossings(target_mv[:, 0])
    count = (jnp.sum(predicted_crossings) - jnp.sum(target_crossings)) ** 2
    peak = (_smooth_peak(predicted_mv[:, 0]) - _smooth_peak(target_mv[:, 0])) ** 2
    return jnp.stack((voltage, derivative, multiscale, event, count, peak))


def make_training_problem(initial_parameters=CANONICAL_INITIAL_PARAMETERS) -> TrainingProblem:
    """Construct target data and a fitted cell with fixed loss normalization."""
    target_cell = build_cell(TARGET_PARAMETERS, trainable=False)
    target = simulate(target_cell)
    fitted_cell = build_cell(CANONICAL_INITIAL_PARAMETERS, trainable=True)
    parameters = find_parameters(fitted_cell)
    mask = target_weight_mask(target.voltages)
    canonical_prediction = simulate(fitted_cell)
    raw_normalizers = raw_loss_components(canonical_prediction, target, mask)
    canonical_error = canonical_prediction.voltages.to_decimal(u.mV) - target.voltages.to_decimal(u.mV)
    voltage_mse_normalizer = jax.lax.stop_gradient(jnp.maximum(jnp.mean(canonical_error**2), 0.1))
    floors = jnp.asarray([0.1, 1e-3, 0.1, 1e-4, 1e-3, 1.0])
    normalizers = jax.lax.stop_gradient(jnp.maximum(raw_normalizers, floors))
    problem = TrainingProblem(
        target_cell=target_cell,
        fitted_cell=fitted_cell,
        target=target,
        parameters=parameters,
        target_mask=mask,
        normalizers=normalizers,
        voltage_mse_normalizer=voltage_mse_normalizer,
    )
    problem.set_physical_parameters(initial_parameters)
    return problem


def make_stage_masks(n_epochs: int, staged: bool) -> jax.Array:
    """Return per-epoch masks in leak, sodium, potassium order."""
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs!r}.")
    masks = np.ones((n_epochs, 3), dtype=float)
    if staged:
        first = min(n_epochs, max(1, n_epochs // 5))
        second = min(n_epochs, max(first + 1, 2 * n_epochs // 5))
        masks[:first] = np.asarray([1.0, 0.0, 0.0])
        masks[first:second] = np.asarray([1.0, 0.0, 1.0])
    return jnp.asarray(masks)


def _stack_parameter_tree(tree) -> jax.Array:
    return jnp.stack([jnp.asarray(tree[name]) for name in PARAMETER_NAMES])


def run_training(config: TrainingConfig, initial_parameters=CANONICAL_INITIAL_PARAMETERS) -> TrainingResult:
    """Run one compiled joint or staged conductance-fitting experiment."""
    problem = make_training_problem(initial_parameters)
    states = problem.param_states

    def objective():
        return problem.loss_with_aux(config.loss_mode)

    grad_fn = brainstate.transform.grad(
        objective,
        grad_states=states,
        has_aux=True,
        return_value=True,
    )
    if config.optimizer == "adam":
        optimizer_type = braintools.optim.Adam
    elif config.optimizer == "radam":
        optimizer_type = braintools.optim.RAdam
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}.")
    optimizer = optimizer_type(lr=config.learning_rate, grad_clip_norm=1.0)
    optimizer.register_trainable_weights(states)
    stage_masks = make_stage_masks(config.n_epochs, config.staged)
    initial_output = simulate(problem.fitted_cell)
    initial_physical = problem.physical_parameters()

    def train_step(mask):
        gradients, loss, components = grad_fn()
        gradient_vector = _stack_parameter_tree(gradients)
        masked_gradients = {name: gradients[name] * mask[index] for index, name in enumerate(PARAMETER_NAMES)}
        optimizer.update(masked_gradients)
        physical = problem.physical_parameters()
        return loss, components, gradient_vector, jnp.linalg.norm(gradient_vector), physical

    losses, components, gradients, gradient_norms, updated_parameters = brainstate.transform.for_loop(
        train_step,
        stage_masks,
    )
    trajectory = jnp.concatenate((initial_physical[None, :], updated_parameters), axis=0)
    fitted_output = simulate(problem.fitted_cell)
    return TrainingResult(
        config=config,
        target_traces=problem.target.voltages,
        initial_traces=initial_output.voltages,
        fitted_traces=fitted_output.voltages,
        losses=losses,
        component_losses=components,
        gradients=gradients,
        gradient_norms=gradient_norms,
        parameter_trajectory=trajectory,
        stage_masks=stage_masks,
        initial_parameters=initial_physical,
        fitted_parameters=problem.physical_parameters(),
        target_parameters=jnp.asarray(TARGET_PARAMETERS),
    )


def evaluate_metrics(target_traces, fitted_traces) -> dict[str, float]:
    """Return voltage and hard-spike metrics as host scalars."""
    target_mv = np.asarray(target_traces.to_decimal(u.mV))
    fitted_mv = np.asarray(fitted_traces.to_decimal(u.mV))
    error = fitted_mv - target_mv
    target_spikes = hard_spike_indices(target_traces)
    fitted_spikes = hard_spike_indices(fitted_traces)
    latency_error = np.inf
    if target_spikes.size and fitted_spikes.size:
        latency_error = abs(float(fitted_spikes[0] - target_spikes[0])) * float(DT.to_decimal(u.ms))
    return {
        "rmse_mv": float(np.sqrt(np.mean(error**2))),
        "mae_mv": float(np.mean(np.abs(error))),
        "target_spike_count": int(target_spikes.size),
        "fitted_spike_count": int(fitted_spikes.size),
        "first_spike_latency_error_ms": float(latency_error),
    }
