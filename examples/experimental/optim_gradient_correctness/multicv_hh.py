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

"""Model-specific compact/full/BPTT regression for configurable HH cells.

This module remains the compact-projection correctness reference. Normal full
RTRL training experiments should use ``gradients`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.filter import AllRegion, at
from examples.experimental.optim._forward_sensitivity import (
    ActiveStateProjection,
    ActiveStateSelection,
    FunctionalStep,
    ParameterCoordinates,
    build_active_state_projection,
    build_parameter_coordinates,
    build_stateful_step,
    compact_forward_sensitivity_step,
    forward_sensitivity_step,
    initialize_forward_sensitivity,
)


DT = 0.025 * u.ms
BENCHMARK_DURATION = 50.0 * u.ms
CHANNEL_NAMES = ("leak", "na", "k")
THREE_CV_TARGET_ROW_SCALES = {
    "leak.scale": jnp.asarray([1.15, 0.90, 1.30]),
    "na.scale": jnp.asarray([0.85, 1.20, 0.95]),
    "k.scale": jnp.asarray([1.25, 0.80, 1.10]),
}
FIVE_CV_TARGET_ROW_SCALES = {
    "leak.scale": jnp.asarray([1.15, 0.90, 1.30, 1.05, 0.80]),
    "na.scale": jnp.asarray([0.85, 1.20, 0.95, 1.10, 0.75]),
    "k.scale": jnp.asarray([1.25, 0.80, 1.10, 0.90, 1.20]),
}
_BASE_G_MAX = {
    "leak": 0.1 * u.mS / u.cm**2,
    "na": 120.0 * u.mS / u.cm**2,
    "k": 10.0 * u.mS / u.cm**2,
}


@dataclass(frozen=True)
class MultiCVProblem:
    """All static metadata and initial values for one gradient comparison."""

    cell: braincell.Cell
    functional_step: FunctionalStep
    projection: ActiveStateProjection
    parameter_coordinates: ParameterCoordinates
    initial_state_values: object
    initial_full_state_tangents: object
    initial_active_state_tangents: object
    parameter_values: tuple[object, ...]
    step_data: object
    target_voltage_mv: object


@dataclass(frozen=True)
class GradientComparison:
    """Terminal losses and gradients from the three differentiation orders."""

    compact_loss: object
    compact_gradient: object
    compact_sensitivity: object
    full_loss: object
    full_gradient: object
    bptt_loss: object
    bptt_gradient: object


@dataclass(frozen=True)
class MethodBenchmark:
    """Compile, execution and static-memory measurements for one method."""

    method: str
    compile_seconds: float
    first_seconds: float
    steady_seconds: tuple[float, ...]
    steady_median_seconds: float
    argument_bytes: int
    output_bytes: int
    temporary_bytes: int
    carry_bytes: int | None


def build_bifurcating_morphology(dendrite_segments: tuple[int, int]) -> braincell.Morphology:
    """Return a soma whose distal endpoint bifurcates into two segmented arms."""
    if (
        not isinstance(dendrite_segments, tuple)
        or len(dendrite_segments) != 2
        or any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in dendrite_segments)
    ):
        raise ValueError("dendrite_segments must be a pair of positive integers.")
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    arm_specs = (
        ("dend_a", "basal_dendrite", 80.0, 2.0, 1.2),
        ("dend_b", "apical_dendrite", 120.0, 2.5, 1.0),
    )
    for segment_count, (prefix, branch_type, base_length, proximal_radius, terminal_radius) in zip(
        dendrite_segments,
        arm_specs,
    ):
        parent = "soma"
        for segment in range(segment_count):
            fraction = segment / segment_count
            next_fraction = (segment + 1) / segment_count
            radius_proximal = proximal_radius + fraction * (terminal_radius - proximal_radius)
            radius_distal = proximal_radius + next_fraction * (terminal_radius - proximal_radius)
            branch = braincell.Branch.from_lengths(
                lengths=[base_length / segment_count] * u.um,
                radii=[radius_proximal, radius_distal] * u.um,
                type=branch_type,
            )
            child_name = f"{prefix}_{segment}"
            morphology.attach(
                parent=parent,
                child_branch=branch,
                child_name=child_name,
                parent_x=1.0,
            )
            parent = child_name
    return morphology


def build_multicv_hh_cell(
    *,
    dendrite_segments: tuple[int, int],
    trainable: bool,
    row_scales=None,
) -> braincell.Cell:
    """Build a configurable multi-CV HH cell with optional row-wise roots."""
    cell = braincell.Cell(
        build_bifurcating_morphology(dendrite_segments),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(1,),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("IL", name="leak", g_max=_BASE_G_MAX["leak"]),
        braincell.mech.Channel("Na_HH1952", name="na", g_max=_BASE_G_MAX["na"]),
        braincell.mech.Channel("K_HH1952", name="k", g_max=_BASE_G_MAX["k"]),
    )
    cell.place(
        at("soma", 0.5),
        braincell.mech.CurrentClamp(
            delay=0.0 * u.ms,
            durations=40.0 * u.ms,
            amplitudes=0.05 * u.nA,
        ),
    )
    if trainable:
        for name in CHANNEL_NAMES:
            cell.channels[name].trainable(
                g_max=braincell.trainable.scale(
                    group_by="row",
                    name=f"{name}.scale",
                )
            )
    else:
        if row_scales is None:
            raise ValueError("A non-trainable target Cell requires row_scales.")
        expected_shape = (cell.n_compartment,)
        for name in CHANNEL_NAMES:
            scales = jnp.asarray(row_scales[f"{name}.scale"])
            if scales.shape != expected_shape:
                raise ValueError(f"Target scale {name!r} must have shape {expected_shape!r}, got {scales.shape!r}.")
            cell.channels[name].set(g_max=scales * _BASE_G_MAX[name])
    return cell


def simulate_voltage(cell: braincell.Cell, times_ms, *, dt=DT):
    """Return pre-step voltage samples for a fixed time vector."""
    cell.reset_state()

    def step(time_ms):
        voltage_mv = cell.V.value.to_decimal(u.mV)[0]
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return voltage_mv

    with brainstate.environ.context(dt=dt):
        return brainstate.transform.for_loop(step, times_ms)


def build_multicv_problem(
    *,
    dendrite_segments: tuple[int, int],
    target_row_scales,
    num_steps: int,
    dt=DT,
) -> MultiCVProblem:
    """Build target/candidate cells and exact full/compact sensitivity seeds."""
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    dt_ms = float(np.asarray(dt.to_decimal(u.ms)).reshape(()))
    times_ms = jnp.arange(num_steps) * dt_ms

    target_cell = build_multicv_hh_cell(
        dendrite_segments=dendrite_segments,
        trainable=False,
        row_scales=target_row_scales,
    )
    target_cell.init_state()
    target_voltage_mv = simulate_voltage(target_cell, times_ms, dt=dt)

    cell = build_multicv_hh_cell(
        dendrite_segments=dendrite_segments,
        trainable=True,
    )
    cell.init_state()
    parameter_states = cell.trainables.parameters().states()

    def reset_and_zero(_):
        cell.reset_state()
        return jnp.asarray(0.0)

    def local_loss(step_item):
        time_ms, target_mv = step_item
        cell.trainables.materialize()
        voltage_mv = cell.V.value.to_decimal(u.mV)[0]
        loss = jnp.mean((voltage_mv - target_mv) ** 2) / num_steps
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return loss

    with brainstate.environ.context(dt=dt):
        initializer_step = build_stateful_step(reset_and_zero, None, parameter_states)
        functional_step = build_stateful_step(
            local_loss,
            (times_ms[0], target_voltage_mv[0]),
            parameter_states,
        )

    projection = _build_hh_active_projection(cell, functional_step)
    parameter_coordinates = build_parameter_coordinates(functional_step)
    initial_state_values, initial_full_state_tangents = initialize_forward_sensitivity(
        initializer_step,
        functional_step,
        None,
    )
    initial_active_state_tangents = projection.extract_tangents(initial_full_state_tangents)
    parameter_values = tuple(initial_state_values[index] for index in functional_step.parameter_indices)
    return MultiCVProblem(
        cell=cell,
        functional_step=functional_step,
        projection=projection,
        parameter_coordinates=parameter_coordinates,
        initial_state_values=initial_state_values,
        initial_full_state_tangents=initial_full_state_tangents,
        initial_active_state_tangents=initial_active_state_tangents,
        parameter_values=parameter_values,
        step_data=(times_ms, target_voltage_mv),
        target_voltage_mv=target_voltage_mv,
    )


def compare_gradients(problem: MultiCVProblem) -> GradientComparison:
    """Run compact RTRL, full-coordinate RTRL and reverse BPTT."""
    compact_loss, compact_gradient, compact_sensitivity = compact_terminal_gradient(
        problem,
        problem.parameter_values,
        problem.step_data,
    )
    full_loss, full_gradient = full_terminal_gradient(
        problem,
        problem.parameter_values,
        problem.step_data,
    )
    bptt_value, bptt_roots = jax.value_and_grad(lambda roots: bptt_loss(problem, roots, problem.step_data))(
        problem.parameter_values
    )
    bptt_gradient = problem.parameter_coordinates.flatten(dict(zip(problem.parameter_coordinates.names, bptt_roots)))
    return GradientComparison(
        compact_loss,
        compact_gradient,
        compact_sensitivity,
        full_loss,
        full_gradient,
        bptt_value,
        bptt_gradient,
    )


def compact_terminal_gradient(problem: MultiCVProblem, parameter_values, step_data):
    """Return terminal loss, gradient and compact sensitivity without time history."""
    state_values = _replace_parameter_values(problem, parameter_values)
    sensitivity = problem.initial_active_state_tangents
    gradient = jnp.zeros((problem.parameter_coordinates.size,), dtype=sensitivity.dtype)
    total_loss = jnp.asarray(0.0, dtype=sensitivity.dtype)

    def scan_step(carry, data):
        values, current_sensitivity, current_gradient, current_loss = carry
        next_values, next_sensitivity, local_loss, local_gradient = compact_forward_sensitivity_step(
            problem.functional_step,
            problem.projection,
            problem.parameter_coordinates,
            values,
            current_sensitivity,
            data,
        )
        return (
            next_values,
            next_sensitivity,
            current_gradient + local_gradient,
            current_loss + local_loss,
        ), None

    (_, sensitivity, gradient, total_loss), _ = jax.lax.scan(
        scan_step,
        (state_values, sensitivity, gradient, total_loss),
        step_data,
    )
    return total_loss, gradient, sensitivity


def full_terminal_gradient(problem: MultiCVProblem, parameter_values, step_data):
    """Return terminal loss and gradient while carrying the full tangent tree."""
    state_values = _replace_parameter_values(problem, parameter_values)
    state_tangents = problem.initial_full_state_tangents
    gradient = jnp.zeros((problem.parameter_coordinates.size,), dtype=parameter_values[0].dtype)
    total_loss = jnp.asarray(0.0, dtype=parameter_values[0].dtype)

    def scan_step(carry, data):
        values, tangents, current_gradient, current_loss = carry
        next_values, next_tangents, local_loss, local_gradient = forward_sensitivity_step(
            problem.functional_step,
            values,
            tangents,
            data,
        )
        return (
            next_values,
            next_tangents,
            current_gradient + local_gradient,
            current_loss + local_loss,
        ), None

    (_, _, gradient, total_loss), _ = jax.lax.scan(
        scan_step,
        (state_values, state_tangents, gradient, total_loss),
        step_data,
    )
    return total_loss, gradient


def bptt_loss(problem: MultiCVProblem, parameter_values, step_data):
    """Return the same scalar objective through a reverse-differentiable scan."""
    state_values = _replace_parameter_values(problem, parameter_values)

    def scan_step(values, data):
        next_values, local_loss = problem.functional_step.call(values, data)
        return next_values, local_loss

    _, losses = jax.lax.scan(scan_step, state_values, step_data)
    return jnp.sum(losses)


def benchmark_problem(problem: MultiCVProblem, *, repeats: int = 3) -> tuple[MethodBenchmark, ...]:
    """Benchmark full BPTT and the two exact forward sensitivity carries."""
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    methods = (
        (
            "bptt",
            jax.value_and_grad(lambda roots, data: bptt_loss(problem, roots, data)),
            None,
        ),
        (
            "rtrl_full",
            lambda roots, data: full_terminal_gradient(problem, roots, data),
            _tree_nbytes(problem.initial_full_state_tangents),
        ),
        (
            f"rtrl_compact_{problem.projection.size}x{problem.parameter_coordinates.size}",
            lambda roots, data: compact_terminal_gradient(problem, roots, data)[:2],
            _tree_nbytes(problem.initial_active_state_tangents),
        ),
    )
    results = []
    arguments = (problem.parameter_values, problem.step_data)
    for name, function, carry_bytes in methods:
        started = time.perf_counter()
        compiled = jax.jit(function).lower(*arguments).compile()
        compile_seconds = time.perf_counter() - started

        started = time.perf_counter()
        _block_until_ready(compiled(*arguments))
        first_seconds = time.perf_counter() - started
        steady = []
        for _ in range(repeats):
            started = time.perf_counter()
            _block_until_ready(compiled(*arguments))
            steady.append(time.perf_counter() - started)
        memory = compiled.memory_analysis()
        results.append(
            MethodBenchmark(
                method=name,
                compile_seconds=compile_seconds,
                first_seconds=first_seconds,
                steady_seconds=tuple(steady),
                steady_median_seconds=float(np.median(steady)),
                argument_bytes=int(memory.argument_size_in_bytes),
                output_bytes=int(memory.output_size_in_bytes),
                temporary_bytes=int(memory.temp_size_in_bytes),
                carry_bytes=carry_bytes,
            )
        )
    return tuple(results)


def _build_hh_active_projection(cell: braincell.Cell, functional_step: FunctionalStep):
    cv_ids = tuple(range(cell.n_compartment))
    na_node = _runtime_channel(cell, "channel:Na_HH1952")
    k_node = _runtime_channel(cell, "channel:K_HH1952")
    return build_active_state_projection(
        functional_step,
        (
            ActiveStateSelection("V", cell.V, tuple(range(cell.n_compartment))),
            ActiveStateSelection("Na.m", na_node.p, cv_ids),
            ActiveStateSelection("Na.h", na_node.q, cv_ids),
            ActiveStateSelection("K.n", k_node.p, cv_ids),
        ),
    )


def _runtime_channel(cell: braincell.Cell, kind: str):
    layouts = tuple(layout for layout in cell.layouts if layout.kind == kind)
    if len(layouts) != 1:
        raise RuntimeError(f"Expected one runtime layout for {kind!r}, got {len(layouts)}.")
    return cell.runtime.get_runtime_node(layouts[0].id)


def _replace_parameter_values(problem: MultiCVProblem, parameter_values):
    if len(parameter_values) != len(problem.functional_step.parameter_indices):
        raise ValueError("Parameter value tuple does not match the functional step roots.")
    values = list(problem.initial_state_values)
    for state_index, value in zip(problem.functional_step.parameter_indices, parameter_values):
        values[state_index] = value
    return tuple(values)


def _tree_nbytes(tree) -> int:
    return sum(
        int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(tree)
    )


def _block_until_ready(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def main() -> None:
    with jax.enable_x64(True), brainstate.environ.context(precision=64):
        num_steps = int(round(float(BENCHMARK_DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms))))
        problem = build_multicv_problem(
            dendrite_segments=(2, 2),
            target_row_scales=FIVE_CV_TARGET_ROW_SCALES,
            num_steps=num_steps,
        )
        comparison = compare_gradients(problem)
        print(f"active state count: {problem.projection.size}")
        print(f"parameter DOF: {problem.parameter_coordinates.size}")
        print(f"compact sensitivity shape: {comparison.compact_sensitivity.shape}")
        print(
            "compact/full/BPTT max abs gradient error: "
            f"{float(jnp.max(jnp.abs(comparison.compact_gradient - comparison.full_gradient))):.3e} / "
            f"{float(jnp.max(jnp.abs(comparison.compact_gradient - comparison.bptt_gradient))):.3e}"
        )
        for result in benchmark_problem(problem):
            print(
                f"{result.method}: compile={result.compile_seconds:.3f}s, "
                f"steady={result.steady_median_seconds:.6f}s, temp={result.temporary_bytes} bytes, "
                f"carry={result.carry_bytes}"
            )


if __name__ == "__main__":
    main()
