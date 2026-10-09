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

"""Prepare detailed states and measure REST, POST and rebase response banks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import partial

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dif.calibration_utils import CalibrationExecution, _root_cv
from braincell.reduction.dif.parameters_utils import _normalize_voltage_grid, _pack_analysis_metadata
from braincell.reduction.dif.responses_utils import (
    CLAMP_SERIES_RESISTANCE_MOHM,
    SINGLE_RESPONSE_HORIZON_BOUND_MV,
    SINGLE_RESPONSE_TAIL_MS,
    SINGLE_RESPONSE_TOLERANCE_MV,
    AlignedConditionResult,
    PostSpikeLibrary,
    PreparedResponseStates,
    _capture_population_state,
    _condition_shards,
    _device_put_value,
    _host_aligned_result,
    _interpolate_rebase_single_curve,
    _make_calibration_cell,
    _rebase_pair_conditions,
    _rebase_pair_curve_layout,
    _rebase_pair_node_mask,
    _restore_population_rows,
    _seed_prepared_states,
    _single_curve_layout,
    _soma_clamp_current,
    allocate_single_voltage,
    build_response_parameters,
    lower_compact_calibration_synapses,
    padded_conditions,
    pair_conditions,
    pair_curve_layout,
    require_no_condition_spikes,
    truncate_voltage_after_first_spike,
    voltage_response_support_steps,
)
from braincell.reduction.dif.tables import ResponseMeasurements
from braincell.reduction.dif.tables_utils import confirmation_voltage_mV


def measure_responses(
    network,
    runtime_plan,
    calibration_plan,
    *,
    analysis,
    execution=None,
):
    """Return response measurements and their conditional backgrounds in memory."""
    # 1. Resolve the representative's REST voltage grid and response axes.
    preparation = {}
    voltage_grid = _normalize_voltage_grid(
        analysis.rest_voltage_grid_mV,
        threshold_mV=analysis.teacher_threshold_mV,
    )
    runtime_timing = runtime_plan.timing
    location_ptr = np.asarray(runtime_plan.population_location_ptr, dtype=np.int32)
    pair_ptr = np.asarray(runtime_plan.population_pair_ptr, dtype=np.int32)
    probe_steps = int(calibration_plan.timing.probe_steps)

    plan = calibration_plan.populations[0]

    # 2. Measure voltage-clamped REST backgrounds, singles and pair residuals.
    rest = build_rest_library(
        network,
        runtime_plan,
        calibration_plan.timing,
        plan,
        analysis,
        voltage_grid,
        preparation=preparation,
        execution=execution,
    )
    # 3. Measure canonical POST states and the fixed-state history rebase.
    post_library = build_post_spike_library(
        network,
        runtime_plan,
        plan,
        analysis,
        tau_steps=rest["tau_steps"],
        event_age_steps=rest["event_age_steps"],
        release_age_steps=rest["release_age_steps"],
        probe_steps=probe_steps,
        preparation=preparation,
        execution=execution,
    )
    # 4. Join the banks and retain their backgrounds for conductance fitting.
    history_support = np.maximum.reduce(
        (
            rest["rest_single_support_steps"],
            post_library.single_support_steps,
            post_library.rebase_single_support_steps,
        )
    ).astype(np.int32, copy=False)
    clamp_duration_ms = int(calibration_plan.timing.clamp_steps) * float(runtime_timing.dt_ms)
    table = ResponseMeasurements(
        calibration_date=date.today().isoformat(),
        **rest,
        post_single_support_steps=post_library.single_support_steps,
        rebase_single_support_steps=post_library.rebase_single_support_steps,
        history_support_steps=history_support,
        rest_voltage_grid_mV=voltage_grid,
        population_location_ptr=location_ptr,
        population_pair_ptr=pair_ptr,
        pair_slots=runtime_plan.pair_slots,
        slots=runtime_plan.slots,
        dt_ms=np.asarray(runtime_timing.dt_ms, dtype=np.float64),
        voltage_interval_mV=np.diff(voltage_grid).astype(np.float64, copy=False),
        single_response_tolerance_mV=np.asarray(SINGLE_RESPONSE_TOLERANCE_MV, dtype=np.float64),
        single_response_tail_ms=np.asarray(SINGLE_RESPONSE_TAIL_MS, dtype=np.float64),
        single_response_horizon_bound_mV=np.asarray(SINGLE_RESPONSE_HORIZON_BOUND_MV, dtype=np.float64),
        single_probe_ms=np.asarray(probe_steps * runtime_timing.dt_ms, dtype=np.float64),
        clamp_duration_ms=np.asarray(clamp_duration_ms, dtype=np.float64),
        clamp_series_resistance_MOhm=np.asarray(CLAMP_SERIES_RESISTANCE_MOHM, dtype=np.float64),
        **_pack_analysis_metadata(analysis),
        post_state_ptr=post_library.state_ptr,
        post_state_voltage_mV=post_library.state_voltage_mV,
        post_rebase_single_voltage_mV=post_library.rebase_single_voltage_mV,
        post_rebase_single_voltage_ptr=post_library.rebase_single_voltage_ptr,
        post_single_voltage_mV=post_library.single_voltage_mV,
        post_single_voltage_ptr=post_library.single_voltage_ptr,
        post_pair_voltage_mV=post_library.pair_voltage_mV,
        post_pair_voltage_ptr=post_library.pair_voltage_ptr,
        post_rebase_pair_voltage_mV=post_library.rebase_pair_voltage_mV,
        post_rebase_pair_voltage_ptr=post_library.rebase_pair_voltage_ptr,
    )
    return table, preparation


def build_rest_library(
    network,
    runtime_plan,
    calibration_timing,
    plan,
    analysis,
    voltage_grid,
    *,
    preparation=None,
    execution=None,
):
    """Calibrate voltage-clamped REST singles and pair residuals into final banks."""
    # 1. Prepare a complete detailed state at every REST voltage node.
    execution = CalibrationExecution() if execution is None else execution
    run_conditions = partial(run_aligned_conditions, execution=execution)
    batch_size = execution.batch_size
    population = 0
    name = runtime_plan.population_names[0]
    voltage_count = int(voltage_grid.size)
    runtime_timing = runtime_plan.timing
    fixed_time_steps = np.asarray(runtime_timing.time_grid_steps, dtype=np.int32)
    population_count = 1
    location_ptr = np.asarray(runtime_plan.population_location_ptr, dtype=np.int32)
    pair_ptr = np.asarray(runtime_plan.population_pair_ptr, dtype=np.int32)
    location_count = np.diff(location_ptr).astype(np.int32, copy=False)

    V_init_mV = np.empty((population_count,), dtype=np.float64)
    V_rest_mV = np.empty((population_count,), dtype=np.float64)
    V_th_mV = np.empty((population_count,), dtype=np.float64)

    source_cell = network.populations[name].cell
    parameters = build_response_parameters(source_cell)
    V_init_mV[population] = parameters.V_init_mV
    V_rest_mV[population] = parameters.V_rest_mV
    V_th_mV[population] = parameters.V_th_mV
    prepared = prepare_voltage_states(
        source_cell,
        voltage_grid_mV=voltage_grid,
        calibration_timing=calibration_timing,
        dt=runtime_timing.dt,
        start_time_ms=runtime_plan.start_time_ms,
        device=execution.devices[0],
    )

    # 2. Measure fresh singles against the no-input baseline and find support.
    rest_support = np.zeros((int(location_ptr[-1]),), dtype=np.int32)
    n_location = int(location_count[population])
    rows_per_voltage = n_location + 1
    condition_count = voltage_count * rows_per_voltage
    voltage_index = np.repeat(np.arange(voltage_count, dtype=np.int32), rows_per_voltage)
    local_input = np.tile(
        np.concatenate((np.asarray([-1], dtype=np.int32), np.arange(n_location, dtype=np.int32))),
        voltage_count,
    )
    condition_input = local_input[:, None]
    condition_arrival = np.where(condition_input >= 0, 0, -1).astype(np.int32)
    result = run_conditions(
        network.populations[name].cell,
        plan,
        prepared,
        voltage_index=voltage_index,
        release_step=np.zeros((condition_count,), dtype=np.int32),
        condition_input=condition_input,
        condition_arrival=condition_arrival,
        probe_steps=int(calibration_timing.probe_steps),
        dt=runtime_timing.dt,
        spike_threshold_mV=float(analysis.teacher_threshold_mV),
    )
    voltage = result.voltage_age_mV
    spike = result.spike_age
    voltage = voltage.reshape(voltage_count, rows_per_voltage, voltage.shape[-1])
    spike = spike.reshape(voltage_count, rows_per_voltage, spike.shape[-1])
    baseline = voltage[:, 0].copy()
    require_no_condition_spikes(
        spike[:, 0, :],
        description=f"population {name!r} tau=0 no-input baseline",
    )
    input_voltage = voltage[:, 1:, :].reshape(-1, voltage.shape[-1])
    input_spike = spike[:, 1:, :].reshape(-1, spike.shape[-1])
    input_baseline = np.broadcast_to(baseline[:, None, :], voltage[:, 1:, :].shape).reshape(-1, voltage.shape[-1])
    input_voltage = truncate_voltage_after_first_spike(
        input_voltage,
        input_spike,
        input_baseline,
    ).reshape(voltage[:, 1:, :].shape)
    single0 = np.transpose(input_voltage - baseline[:, None, :], (1, 0, 2))
    begin = int(location_ptr[population])
    end = int(location_ptr[population + 1])
    rest_support[begin:end] = voltage_response_support_steps(
        single0,
        dt_ms=float(runtime_timing.dt_ms),
        description=f"population {name!r} REST",
    )
    del result, voltage, spike, input_voltage, input_spike, input_baseline

    # 3. Allocate the shared age axes and pack the fresh-single response bank.
    tau_steps = fixed_time_steps.copy()
    event_age_steps = fixed_time_steps.copy()
    release_age_steps = fixed_time_steps.copy()
    if np.any(rest_support >= int(fixed_time_steps[-1])):
        raise RuntimeError(
            "DIF REST voltage support reaches the final time-grid node; extend the probe and explicit time axis."
        )
    tau_count = int(tau_steps.size)
    probe_steps = int(calibration_timing.probe_steps)
    baseline_voltage = np.zeros((population_count, voltage_count, probe_steps + 1), dtype=np.float64)
    single_voltage = np.zeros(
        (
            int(location_ptr[-1]),
            voltage_count,
            tau_count,
            probe_steps + 1,
        ),
        dtype=np.float64,
    )
    # Each curve has at most probe_steps samples. Reserve its zero guard
    # in the final bank on allocation.
    runtime_single_voltage, runtime_single_ptr = allocate_single_voltage(rest_support, voltage_count, probe_steps)
    begin = int(location_ptr[population])
    end = int(location_ptr[population + 1])
    baseline_voltage[population] = baseline[:, : probe_steps + 1]
    for local, global_location in enumerate(range(begin, end)):
        single_voltage[global_location, :, 0, :] = single0[local, :, : probe_steps + 1]
        length = int(rest_support[global_location]) + 1
        destination = runtime_single_voltage[
            runtime_single_ptr[global_location] : runtime_single_ptr[global_location + 1]
        ].reshape(voltage_count, length)
        destination[:] = single0[local, :, :length]
    del single0, baseline

    # 4. Measure aged singles needed to subtract pair backgrounds.
    # Complete the Dai--Li singlet library for old inputs that have evolved
    # under the voltage clamp for tau before release.
    begin = int(location_ptr[population])
    end = int(location_ptr[population + 1])
    local_support = rest_support[begin:end]
    local_parts = []
    tau_parts = []
    for local, location_support in enumerate(local_support.tolist()):
        valid_tau = np.flatnonzero((tau_steps > 0) & (tau_steps < int(location_support))).astype(np.int32, copy=False)
        local_parts.append(np.full(valid_tau.size, local, dtype=np.int32))
        tau_parts.append(valid_tau)
    local_base = np.concatenate(local_parts)
    tau_base = np.concatenate(tau_parts)
    local = np.tile(local_base, voltage_count)
    tau_index = np.tile(tau_base, voltage_count)
    voltage_index = np.repeat(np.arange(voltage_count, dtype=np.int32), local_base.size)
    condition_count = int(local.size)
    single_batch = min(batch_size, max(1, condition_count))
    batch_count = (condition_count + single_batch - 1) // single_batch
    for batch in range(batch_count):
        row_begin = batch * single_batch
        row_end = min(condition_count, row_begin + single_batch)
        actual = row_end - row_begin
        batch_local = padded_conditions(local, row_begin, row_end, single_batch)
        batch_tau = padded_conditions(tau_index, row_begin, row_end, single_batch)
        batch_voltage = padded_conditions(voltage_index, row_begin, row_end, single_batch)
        release = tau_steps[batch_tau]
        result = run_conditions(
            network.populations[name].cell,
            plan,
            prepared,
            voltage_index=batch_voltage,
            release_step=release,
            condition_input=batch_local[:, None],
            condition_arrival=np.zeros((single_batch, 1), dtype=np.int32),
            probe_steps=probe_steps,
            dt=runtime_timing.dt,
            spike_threshold_mV=float(analysis.teacher_threshold_mV),
        )
        aligned = result.voltage_age_mV[:actual]
        spike = result.spike_age[:actual]
        row_baseline = baseline_voltage[population, batch_voltage[:actual], :]
        aligned = truncate_voltage_after_first_spike(
            aligned,
            spike,
            row_baseline,
        )
        for row in range(actual):
            loc = int(batch_local[row])
            v = int(batch_voltage[row])
            tau = int(batch_tau[row])
            length = probe_steps - int(tau_steps[tau]) + 1
            single_voltage[begin + loc, v, tau, :length] = (
                aligned[row, :length] - baseline_voltage[population, v, :length]
            )
        del result, aligned, spike

    # 5. Measure pairs and subtract the baseline plus both single responses.
    pair_voltage_ptr = pair_curve_layout(
        runtime_plan.pair_slots,
        rest_support,
        tau_steps,
    )
    pair_state_stride = int(pair_voltage_ptr[-1])
    pair_voltage = np.zeros(voltage_count * pair_state_stride + probe_steps, dtype=np.float64)
    n_location = int(location_count[population])
    location_begin = int(location_ptr[population])
    location_end = int(location_ptr[population + 1])
    local_support = rest_support[location_begin:location_end]
    pair_begin = int(pair_ptr[population])
    local_pairs = runtime_plan.pair_slots[pair_begin : int(pair_ptr[population + 1])] - location_begin
    pair_rows = {tuple(pair): pair_begin + row for row, pair in enumerate(local_pairs)}
    old, new, tau_index = pair_conditions(local_pairs, local_support, tau_steps)
    condition_voltage = np.repeat(np.arange(voltage_count, dtype=np.int32), old.size)
    condition_old = np.tile(old, voltage_count)
    condition_new = np.tile(new, voltage_count)
    condition_tau = np.tile(tau_index, voltage_count)
    condition_count = int(condition_old.size)
    pair_batch = min(batch_size, max(1, condition_count))
    batch_count = (condition_count + pair_batch - 1) // pair_batch
    for batch in range(batch_count):
        row_begin = batch * pair_batch
        row_end = min(condition_count, row_begin + pair_batch)
        actual = row_end - row_begin
        batch_voltage = padded_conditions(condition_voltage, row_begin, row_end, pair_batch)
        batch_old = padded_conditions(condition_old, row_begin, row_end, pair_batch)
        batch_new = padded_conditions(condition_new, row_begin, row_end, pair_batch)
        batch_tau = padded_conditions(condition_tau, row_begin, row_end, pair_batch)
        release = tau_steps[batch_tau]
        result = run_conditions(
            network.populations[name].cell,
            plan,
            prepared,
            voltage_index=batch_voltage,
            release_step=release,
            condition_input=np.stack((batch_old, batch_new), axis=1),
            condition_arrival=np.stack((np.zeros_like(release), release), axis=1),
            probe_steps=probe_steps,
            dt=runtime_timing.dt,
            spike_threshold_mV=float(analysis.teacher_threshold_mV),
        )
        aligned = result.voltage_age_mV[:actual]
        spike = result.spike_age[:actual]
        row_baseline = baseline_voltage[population, batch_voltage[:actual], :]
        aligned = truncate_voltage_after_first_spike(
            aligned,
            spike,
            row_baseline,
        )
        for local_row in range(actual):
            voltage_index = int(batch_voltage[local_row])
            local_old = int(batch_old[local_row])
            local_new = int(batch_new[local_row])
            tau = int(batch_tau[local_row])
            pair_row = pair_rows[local_old, local_new]
            slot = pair_row * tau_count + tau
            packed_begin = int(pair_voltage_ptr[slot])
            packed_end = int(pair_voltage_ptr[slot + 1])
            length = packed_end - packed_begin
            residual = aligned[local_row, :length]
            residual = residual - baseline_voltage[population, voltage_index, :length]
            residual = (
                residual
                - single_voltage[
                    location_begin + local_old,
                    voltage_index,
                    tau,
                    :length,
                ]
            )
            residual = (
                residual
                - single_voltage[
                    location_begin + local_new,
                    voltage_index,
                    0,
                    :length,
                ]
            )
            destination = voltage_index * pair_state_stride + packed_begin
            pair_voltage[destination : destination + length] = residual
            if tau == 0 and local_old != local_new:
                mirror_row = pair_rows[local_new, local_old]
                mirror_slot = mirror_row * tau_count
                mirror_begin = int(pair_voltage_ptr[mirror_slot])
                mirror_end = int(pair_voltage_ptr[mirror_slot + 1])
                mirror_length = mirror_end - mirror_begin
                mirror_destination = voltage_index * pair_state_stride + mirror_begin
                pair_voltage[mirror_destination : mirror_destination + mirror_length] = residual[:mirror_length]
        del result, aligned, spike
    del (
        old,
        new,
        tau_index,
        condition_voltage,
        condition_old,
        condition_new,
        condition_tau,
    )

    # 6. Retain fitting backgrounds and return the final REST banks.
    # Aged singles are scratch for pair residuals; the final singles were
    # written directly after measuring support.
    if preparation is not None:
        preparation['rest'] = dict(
            aged_single_voltage_mV=single_voltage,
            baseline_voltage_mV=baseline_voltage,
            tau_steps=np.asarray(tau_steps),
        )
    del single_voltage
    return {
        "baseline_voltage_mV": baseline_voltage,
        "single_voltage_mV": runtime_single_voltage,
        "single_voltage_ptr": runtime_single_ptr,
        "pair_voltage_mV": pair_voltage,
        "pair_voltage_ptr": pair_voltage_ptr,
        "rest_single_support_steps": rest_support,
        "tau_steps": tau_steps,
        "event_age_steps": event_age_steps,
        "release_age_steps": release_age_steps,
        "V_init_mV": V_init_mV,
        "V_rest_mV": V_rest_mV,
        "V_th_mV": V_th_mV,
    }


def build_post_spike_library(
    network,
    runtime_plan,
    plan,
    analysis,
    *,
    tau_steps,
    event_age_steps,
    release_age_steps,
    probe_steps,
    preparation=None,
    execution=None,
):
    """Measure POST responses and rebase histories at the captured R+1 state."""
    # 1. Restore complete canonical states for the falling branch and rebase.
    execution = CalibrationExecution() if execution is None else execution
    run_conditions = partial(run_aligned_conditions, execution=execution)
    batch_size = execution.batch_size
    population = 0
    name = runtime_plan.population_names[0]
    location_ptr = np.asarray(runtime_plan.population_location_ptr, dtype=np.int32)
    pair_ptr = np.asarray(runtime_plan.population_pair_ptr, dtype=np.int32)
    tau_steps = np.asarray(tau_steps, dtype=np.int32)
    event_age_steps = np.asarray(event_age_steps, dtype=np.int32)
    release_age_steps = np.asarray(release_age_steps, dtype=np.int32)
    tau_count = int(tau_steps.size)
    event_age_count = int(event_age_steps.size)
    release_age_count = int(release_age_steps.size)
    state_voltage = np.asarray(analysis.falling_voltage_grid_mV, dtype=np.float64)
    state_ptr = np.asarray([0, state_voltage.size], dtype=np.int32)
    state_count = int(state_voltage.size)
    population_count = 1
    baseline = np.zeros((state_count, probe_steps + 1), dtype=np.float64)
    branch_aged_single = np.zeros(
        (
            state_count,
            int(location_ptr[-1]),
            tau_count,
            probe_steps + 1,
        ),
        dtype=np.float64,
    )
    rebase_baseline = np.zeros((population_count, probe_steps + 1), dtype=np.float64)
    rebase_fresh_single = np.zeros((int(location_ptr[-1]), probe_steps + 1), dtype=np.float64)
    post_support = np.zeros((int(location_ptr[-1]),), dtype=np.int32)
    rebase_support = np.zeros_like(post_support)

    state_begin = int(state_ptr[population])
    state_end = int(state_ptr[population + 1])
    local_state_count = state_end - state_begin
    rebase_state = analysis.rebase_state
    if rebase_state is None:
        raise RuntimeError(f"Population {name!r} has no detailed state captured before R+1.")
    captured = analysis.falling_states + (rebase_state,)
    local_voltage = np.concatenate(
        (
            analysis.falling_voltage_grid_mV,
            np.asarray(
                [analysis.canonical_voltage_mV[analysis.canonical_spike_step + analysis.refractory_steps]],
                dtype=np.float64,
            ),
        )
    )
    prepared_state_count = local_state_count + 1
    rebase_local_state = local_state_count
    prepared = prepare_post_voltage_states(
        network.populations[name].cell,
        captured_states=captured,
        voltage_grid_mV=local_voltage,
        start_time_ms=runtime_plan.start_time_ms,
        device=execution.devices[0],
    )
    location_begin = int(location_ptr[population])
    location_end = int(location_ptr[population + 1])
    count = location_end - location_begin
    pair_begin = int(pair_ptr[population])
    local_pairs = runtime_plan.pair_slots[pair_begin : int(pair_ptr[population + 1])] - location_begin
    pair_rows = {tuple(pair): pair_begin + row for row, pair in enumerate(local_pairs)}
    # 2. Measure fresh singles and baselines at each restored state.
    rows_per_state = count + 1
    fresh_state = np.repeat(np.arange(prepared_state_count, dtype=np.int32), rows_per_state)
    fresh_input = np.tile(
        np.concatenate((np.asarray([-1], dtype=np.int32), np.arange(count, dtype=np.int32))),
        prepared_state_count,
    )
    fresh_batch = min(batch_size, max(1, int(fresh_state.size)))
    for begin in range(0, fresh_state.size, fresh_batch):
        end = min(fresh_state.size, begin + fresh_batch)
        actual = end - begin
        batch_state = padded_conditions(fresh_state, begin, end, fresh_batch)
        batch_input = padded_conditions(fresh_input, begin, end, fresh_batch)
        result = run_conditions(
            network.populations[name].cell,
            plan,
            prepared,
            voltage_index=batch_state,
            release_step=np.zeros((fresh_batch,), dtype=np.int32),
            condition_input=batch_input[:, None],
            condition_arrival=np.where(batch_input[:, None] >= 0, 0, -1).astype(np.int32),
            probe_steps=probe_steps,
            dt=runtime_plan.timing.dt,
            spike_threshold_mV=float(analysis.teacher_threshold_mV),
        )
        voltage = result.voltage_age_mV[:actual]
        spike = result.spike_age[:actual]
        no_input_rows = np.flatnonzero(batch_input[:actual] < 0)
        require_no_condition_spikes(
            spike[no_input_rows],
            description=f"population {name!r} post no-input baseline",
        )
        for row in no_input_rows.tolist():
            local_state = int(batch_state[row])
            if local_state == rebase_local_state:
                rebase_baseline[population] = voltage[row]
            else:
                baseline[state_begin + local_state] = voltage[row]
        input_rows = np.flatnonzero(batch_input[:actual] >= 0)
        input_baseline = np.empty((input_rows.size, voltage.shape[1]), dtype=voltage.dtype)
        for destination, row in enumerate(input_rows.tolist()):
            local_state = int(batch_state[row])
            if local_state == rebase_local_state:
                input_baseline[destination] = rebase_baseline[population]
            else:
                input_baseline[destination] = baseline[state_begin + local_state]
        input_voltage = truncate_voltage_after_first_spike(
            voltage[input_rows],
            spike[input_rows],
            input_baseline,
        )
        for input_row, row in enumerate(input_rows):
            local_state = int(batch_state[row])
            local = int(batch_input[row])
            if local_state == rebase_local_state:
                rebase_fresh_single[location_begin + local] = input_voltage[input_row] - rebase_baseline[population]
                continue
            state = state_begin + local_state
            branch_aged_single[state, location_begin + local, 0, :] = input_voltage[input_row] - baseline[state]

    # 3. Determine supports and allocate POST and rebase curve layouts.
    post_response = np.transpose(
        branch_aged_single[state_begin:state_end, location_begin:location_end, 0, :],
        (1, 0, 2),
    )
    post_support[location_begin:location_end] = voltage_response_support_steps(
        post_response,
        dt_ms=float(runtime_plan.timing.dt_ms),
        description=f"population {name!r} POST",
    )
    rebase_support[location_begin:location_end] = voltage_response_support_steps(
        rebase_fresh_single[location_begin:location_end, None, :],
        dt_ms=float(runtime_plan.timing.dt_ms),
        description=f"population {name!r} fixed-R+1",
    )
    if np.any(post_support[location_begin:location_end] >= tau_steps[-1]) or np.any(
        rebase_support[location_begin:location_end] >= tau_steps[-1]
    ):
        raise RuntimeError(
            "DIF POST or fixed-R+1 voltage support reaches the final "
            "time-grid node; extend the probe and explicit axis."
        )

    single_ptr = _single_curve_layout(rebase_support, event_age_steps)
    pair_curve_ptr = pair_curve_layout(runtime_plan.pair_slots, post_support, tau_steps)
    rebase_pair_node_mask = _rebase_pair_node_mask(
        runtime_plan.pair_slots,
        rebase_support,
        tau_steps,
        release_age_steps,
    )
    rebase_pair_curve_ptr = _rebase_pair_curve_layout(
        rebase_pair_node_mask,
        runtime_plan.pair_slots,
        rebase_support,
        tau_steps,
        release_age_steps,
    )
    fresh_single, fresh_single_ptr = allocate_single_voltage(post_support, state_count, probe_steps)
    pair_state_stride = int(pair_curve_ptr[-1])
    pair = np.zeros(state_count * pair_state_stride + probe_steps, dtype=np.float64)
    rebase_single = np.zeros(int(single_ptr[-1]) + probe_steps, dtype=np.float64)
    rebase_pair = np.zeros(int(rebase_pair_curve_ptr[-1]) + probe_steps, dtype=np.float64)
    for location in range(location_begin, location_end):
        fresh_length = int(post_support[location]) + 1
        fresh = fresh_single[fresh_single_ptr[location] : fresh_single_ptr[location + 1]].reshape(
            state_count, fresh_length
        )
        fresh[:] = branch_aged_single[:, location, 0, :fresh_length]
        slot = location * event_age_count
        destination = int(single_ptr[slot])
        length = int(single_ptr[slot + 1] - destination)
        rebase_single[destination : destination + length] = rebase_fresh_single[location, :length]

    # 4. Measure aged singles for the ordinary POST pair backgrounds.
    local_support = post_support[location_begin:location_end]
    local_parts = []
    tau_parts = []
    for local in range(count):
        valid = np.flatnonzero((tau_steps > 0) & (tau_steps < int(local_support[local]))).astype(np.int32, copy=False)
        local_parts.append(np.full(valid.size, local, dtype=np.int32))
        tau_parts.append(valid)
    local_base = np.concatenate(local_parts)
    tau_base = np.concatenate(tau_parts)
    aged_state = np.repeat(np.arange(local_state_count, dtype=np.int32), local_base.size)
    aged_local = np.tile(local_base, local_state_count)
    aged_tau = np.tile(tau_base, local_state_count)
    branch_batch = min(batch_size, max(1, int(aged_state.size)))
    for begin in range(0, aged_state.size, branch_batch):
        end = min(aged_state.size, begin + branch_batch)
        actual = end - begin
        batch_state = padded_conditions(aged_state, begin, end, branch_batch)
        batch_local = padded_conditions(aged_local, begin, end, branch_batch)
        batch_tau = padded_conditions(aged_tau, begin, end, branch_batch)
        release = tau_steps[batch_tau]
        result = run_conditions(
            network.populations[name].cell,
            plan,
            prepared,
            voltage_index=batch_state,
            release_step=release,
            condition_input=batch_local[:, None],
            condition_arrival=np.zeros((branch_batch, 1), dtype=np.int32),
            probe_steps=probe_steps,
            dt=runtime_plan.timing.dt,
            spike_threshold_mV=float(analysis.teacher_threshold_mV),
        )
        aligned = result.voltage_age_mV[:actual]
        spike = result.spike_age[:actual]
        row_baseline = baseline[state_begin + batch_state[:actual]]
        aligned = truncate_voltage_after_first_spike(
            aligned,
            spike,
            row_baseline,
        )
        for row in range(actual):
            state = state_begin + int(batch_state[row])
            location = location_begin + int(batch_local[row])
            tau_index = int(batch_tau[row])
            length = probe_steps - int(tau_steps[tau_index]) + 1
            branch_aged_single[state, location, tau_index, :length] = aligned[row, :length] - baseline[state, :length]

    # 5. Measure historical singles released from the fixed rebase state.
    rebase_local_parts = []
    rebase_age_parts = []
    for local in range(count):
        valid_age = np.flatnonzero(
            (event_age_steps > 0) & (event_age_steps < int(rebase_support[location_begin + local]))
        ).astype(np.int32, copy=False)
        rebase_local_parts.append(np.full(valid_age.size, local, dtype=np.int32))
        rebase_age_parts.append(valid_age)
    rebase_local = np.concatenate(rebase_local_parts)
    rebase_age = np.concatenate(rebase_age_parts)
    rebase_batch = min(batch_size, max(1, int(rebase_local.size)))
    for begin in range(0, rebase_local.size, rebase_batch):
        end = min(rebase_local.size, begin + rebase_batch)
        actual = end - begin
        batch_local = padded_conditions(rebase_local, begin, end, rebase_batch)
        batch_age = padded_conditions(rebase_age, begin, end, rebase_batch)
        release = event_age_steps[batch_age]
        result = run_conditions(
            network.populations[name].cell,
            plan,
            prepared,
            voltage_index=np.full((rebase_batch,), rebase_local_state, dtype=np.int32),
            release_step=release,
            condition_input=batch_local[:, None],
            condition_arrival=np.zeros((rebase_batch, 1), dtype=np.int32),
            probe_steps=probe_steps,
            dt=runtime_plan.timing.dt,
            spike_threshold_mV=float(analysis.teacher_threshold_mV),
        )
        aligned = result.voltage_age_mV[:actual]
        spike = result.spike_age[:actual]
        row_baseline = np.broadcast_to(rebase_baseline[population], aligned.shape)
        aligned = truncate_voltage_after_first_spike(
            aligned,
            spike,
            row_baseline,
        )
        for row in range(actual):
            location = location_begin + int(batch_local[row])
            age_index = int(batch_age[row])
            slot = location * event_age_count + age_index
            destination = int(single_ptr[slot])
            length = int(single_ptr[slot + 1] - destination)
            rebase_single[destination : destination + length] = (
                aligned[row, :length] - rebase_baseline[population, :length]
            )

    # 6. Subtract single backgrounds from the ordinary POST pair measurements.
    old, new, ordinary_tau = pair_conditions(local_pairs, local_support, tau_steps)
    ordinary_state = np.repeat(np.arange(local_state_count, dtype=np.int32), old.size)
    ordinary_old = np.tile(old, local_state_count)
    ordinary_new = np.tile(new, local_state_count)
    ordinary_tau_all = np.tile(ordinary_tau, local_state_count)
    ordinary_batch = min(batch_size, max(1, int(ordinary_state.size)))
    for begin in range(0, ordinary_state.size, ordinary_batch):
        end = min(ordinary_state.size, begin + ordinary_batch)
        actual = end - begin
        batch_state = padded_conditions(ordinary_state, begin, end, ordinary_batch)
        batch_old = padded_conditions(ordinary_old, begin, end, ordinary_batch)
        batch_new = padded_conditions(ordinary_new, begin, end, ordinary_batch)
        batch_tau = padded_conditions(ordinary_tau_all, begin, end, ordinary_batch)
        release = tau_steps[batch_tau]
        result = run_conditions(
            network.populations[name].cell,
            plan,
            prepared,
            voltage_index=batch_state,
            release_step=release,
            condition_input=np.stack((batch_old, batch_new), axis=1),
            condition_arrival=np.stack((np.zeros_like(release), release), axis=1),
            probe_steps=probe_steps,
            dt=runtime_plan.timing.dt,
            spike_threshold_mV=float(analysis.teacher_threshold_mV),
        )
        aligned = result.voltage_age_mV[:actual]
        spike = result.spike_age[:actual]
        row_baseline = baseline[state_begin + batch_state[:actual]]
        aligned = truncate_voltage_after_first_spike(
            aligned,
            spike,
            row_baseline,
        )
        for row in range(actual):
            state = state_begin + int(batch_state[row])
            old_local = int(batch_old[row])
            new_local = int(batch_new[row])
            tau_index = int(batch_tau[row])
            pair_row = pair_rows[old_local, new_local]
            pair_slot = pair_row * tau_count + tau_index
            destination = int(pair_curve_ptr[pair_slot])
            length = int(pair_curve_ptr[pair_slot + 1] - destination)
            residual = aligned[row, :length] - baseline[state, :length]
            residual -= branch_aged_single[state, location_begin + old_local, tau_index, :length]
            residual -= branch_aged_single[state, location_begin + new_local, 0, :length]
            destination += state * pair_state_stride
            pair[destination : destination + length] = residual
            if tau_index == 0 and old_local != new_local:
                mirror_row = pair_rows[new_local, old_local]
                mirror_slot = mirror_row * tau_count
                mirror_destination = state * pair_state_stride + int(pair_curve_ptr[mirror_slot])
                pair[mirror_destination : mirror_destination + length] = residual

    # 7. Measure rebase pairs and subtract singles at their own physical ages.
    rebase_old, rebase_new, rebase_tau, rebase_release = _rebase_pair_conditions(
        local_pairs,
        pair_begin,
        rebase_pair_node_mask,
        rebase_pair_curve_ptr,
    )
    rebase_voltage_state = np.full(rebase_old.size, rebase_local_state, dtype=np.int32)
    rebase_batch = min(batch_size, max(1, int(rebase_old.size)))
    for begin in range(0, rebase_old.size, rebase_batch):
        end = min(rebase_old.size, begin + rebase_batch)
        actual = end - begin
        batch_state = padded_conditions(rebase_voltage_state, begin, end, rebase_batch)
        batch_old = padded_conditions(rebase_old, begin, end, rebase_batch)
        batch_new = padded_conditions(rebase_new, begin, end, rebase_batch)
        batch_tau = padded_conditions(rebase_tau, begin, end, rebase_batch)
        batch_release = padded_conditions(rebase_release, begin, end, rebase_batch)
        release = release_age_steps[batch_release]
        arrival_new = tau_steps[batch_tau]
        result = run_conditions(
            network.populations[name].cell,
            plan,
            prepared,
            voltage_index=batch_state,
            release_step=release,
            condition_input=np.stack((batch_old, batch_new), axis=1),
            condition_arrival=np.stack((np.zeros_like(release), arrival_new), axis=1),
            probe_steps=probe_steps,
            dt=runtime_plan.timing.dt,
            spike_threshold_mV=float(analysis.teacher_threshold_mV),
        )
        aligned = result.voltage_age_mV[:actual]
        spike = result.spike_age[:actual]
        row_baseline = np.broadcast_to(rebase_baseline[population], aligned.shape)
        aligned = truncate_voltage_after_first_spike(
            aligned,
            spike,
            row_baseline,
        )
        for row in range(actual):
            old_local = int(batch_old[row])
            new_local = int(batch_new[row])
            tau_index = int(batch_tau[row])
            release_index = int(batch_release[row])
            pair_row = pair_rows[old_local, new_local]
            rebase_slot = (pair_row * tau_count + tau_index) * release_age_count + release_index
            destination = int(rebase_pair_curve_ptr[rebase_slot])
            length = int(rebase_pair_curve_ptr[rebase_slot + 1] - destination)
            release_age = int(release_age_steps[release_index])
            tau = int(tau_steps[tau_index])
            new_age = release_age - tau
            residual = aligned[row, :length] - rebase_baseline[population, :length]
            old_curve = _interpolate_rebase_single_curve(
                rebase_single,
                single_ptr,
                location=location_begin + old_local,
                event_age=release_age,
                event_age_steps=event_age_steps,
                event_age_count=event_age_count,
                requested_length=length,
            )
            new_curve = _interpolate_rebase_single_curve(
                rebase_single,
                single_ptr,
                location=location_begin + new_local,
                event_age=new_age,
                event_age_steps=event_age_steps,
                event_age_count=event_age_count,
                requested_length=length,
            )
            residual -= old_curve
            residual -= new_curve
            rebase_pair[destination : destination + length] = residual

    # 8. Retain fitting backgrounds and return the POST and rebase banks.
    if preparation is not None:
        preparation['post'] = dict(
            post_baseline_voltage_mV=baseline,
            post_aged_single_voltage_mV=branch_aged_single,
            rebase_baseline_voltage_mV=rebase_baseline,
        )
    return PostSpikeLibrary(
        state_ptr=state_ptr,
        state_voltage_mV=state_voltage,
        single_support_steps=post_support,
        rebase_single_support_steps=rebase_support,
        single_voltage_mV=fresh_single,
        single_voltage_ptr=fresh_single_ptr,
        pair_voltage_mV=pair,
        pair_voltage_ptr=pair_curve_ptr,
        rebase_single_voltage_mV=rebase_single,
        rebase_single_voltage_ptr=single_ptr,
        rebase_pair_voltage_mV=rebase_pair,
        rebase_pair_voltage_ptr=rebase_pair_curve_ptr,
    )


def prepare_voltage_states(
    source_cell,
    *,
    voltage_grid_mV: np.ndarray,
    calibration_timing,
    dt,
    start_time_ms: float,
    device=None,
) -> PreparedResponseStates:
    """Equilibrate one detailed state at each requested somatic voltage."""
    # 1. Replicate the representative detailed cell at every REST voltage node.
    voltage_grid = np.asarray(voltage_grid_mV, dtype=np.float64)
    if device is None:
        device = jax.devices("gpu")[0]
    compact_ref = [None]
    cell = _make_calibration_cell(
        source_cell,
        int(voltage_grid.size),
        compact_ref,
        device=device,
    )

    # 2. Equilibrate all internal states under the somatic voltage clamp.
    def clamp_current(point_voltage):
        return _soma_clamp_current(
            cell,
            point_voltage,
            voltage_grid,
            jnp.ones(voltage_grid.shape, dtype=jnp.bool_),
        )

    cell.add_current_input("dai_li_soma_voltage_clamp", clamp_current)
    steps = int(calibration_timing.clamp_steps)
    times = float(start_time_ms) * u.ms + u.math.arange(steps) * dt
    with brainstate.environ.context(dt=dt):

        def step(t):
            with brainstate.environ.context(t=t):
                cell._update_dynamics()
            return jnp.asarray(0, dtype=jnp.int32)

        jax.block_until_ready(brainstate.transform.for_loop(step, times))

    # 3. Retain the complete states and their common response start time.
    del step, times
    return PreparedResponseStates(
        cell=cell,
        voltage_grid_mV=voltage_grid,
        response_start_ms=(
            float(start_time_ms) + steps * float(np.asarray(dt.to_decimal(u.ms), dtype=np.float64).reshape(()))
        ),
    )


def prepare_post_voltage_states(
    source_cell,
    *,
    captured_states,
    voltage_grid_mV: np.ndarray,
    start_time_ms: float,
    device=None,
) -> PreparedResponseStates:
    """Build a reusable bank from directed post-spike voltage crossings."""
    # 1. Allocate one detailed cell for each captured canonical state.
    states = tuple(captured_states)
    voltage_grid = np.asarray(voltage_grid_mV, dtype=np.float64)
    if device is None:
        device = jax.devices("gpu")[0]
    if not states or voltage_grid.shape != (len(states),):
        raise ValueError("DIF post states and their voltage grid must be non-empty and aligned.")
    compact_ref = [None]
    cell = _make_calibration_cell(source_cell, len(states), compact_ref, device=device)
    # 2. Restore all captured internal states, including the soma voltage.
    captured_by_state = [dict(state.population_state) for state in states]
    assigned = 0
    for path, target_state in cell.states().items():
        key = str(path)
        if not all(key in captured for captured in captured_by_state):
            continue
        values = [captured[key] for captured in captured_by_state]
        try:
            stacked = u.math.stack(values, axis=0)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"DIF cannot stack captured post-spike state {key!r}.") from error
        if tuple(stacked.shape) != tuple(target_state.value.shape):
            continue
        target_state.value = _device_put_value(stacked, device)
        assigned += 1
    if assigned == 0:
        raise RuntimeError("DIF captured post-spike states do not match the calibration cell.")
    cell.spike.value = jnp.zeros_like(cell.spike.value)
    cell.clear_ion_total_current_cache()
    # 3. Check the restored voltage coordinates and retain the state bank.
    root_cv = _root_cv(cell)
    captured_voltage = np.asarray(
        jax.device_get(cell.V.value[..., root_cv].to_decimal(u.mV)),
        dtype=np.float64,
    )
    maximum_error = float(np.max(np.abs(captured_voltage - voltage_grid)))
    if maximum_error > 2.0:
        raise RuntimeError(
            "A directed post-spike crossing missed its 2 mV grid coordinate "
            f"by {maximum_error:.3f} mV. Reduce dt or choose a later Tref."
        )
    return PreparedResponseStates(
        cell=cell,
        voltage_grid_mV=voltage_grid,
        response_start_ms=float(start_time_ms),
    )


def run_aligned_conditions(
    source_cell,
    plan,
    prepared: PreparedResponseStates,
    *,
    voltage_index: np.ndarray,
    release_step: np.ndarray,
    condition_input: np.ndarray,
    condition_arrival: np.ndarray,
    condition_amplitude: np.ndarray | None = None,
    probe_steps: int,
    dt,
    spike_threshold_mV: float,
    execution=None,
) -> AlignedConditionResult:
    """Run one calibration batch concurrently across the selected GPUs."""
    # 1. Normalize condition rows and their arrival/release coordinates.
    voltage_index = np.asarray(voltage_index, dtype=np.int32)
    release_step = np.asarray(release_step, dtype=np.int32)
    condition_input = np.asarray(condition_input, dtype=np.int32)
    condition_arrival = np.asarray(condition_arrival, dtype=np.int32)
    if condition_amplitude is None:
        condition_amplitude = np.ones(condition_input.shape, dtype=np.float64)
    else:
        condition_amplitude = np.asarray(condition_amplitude, dtype=np.float64)
    condition_count = int(voltage_index.size)
    if condition_count <= 0:
        raise ValueError("DIF calibration requires at least one condition.")
    if condition_input.shape != condition_arrival.shape:
        raise ValueError("DIF condition inputs and arrivals must align.")
    if condition_amplitude.shape != condition_input.shape:
        raise ValueError("DIF condition amplitudes and inputs must align.")
    if condition_input.shape[0] != condition_count:
        raise ValueError("DIF condition endpoints must have one row per condition.")
    if release_step.shape != (condition_count,):
        raise ValueError("DIF release_step must have one value per condition.")
    if np.any(release_step < 0) or np.any(release_step > int(probe_steps)):
        raise ValueError("DIF clamp-release steps must lie within the calibration probe.")
    execution = CalibrationExecution() if execution is None else execution

    # 2. Bound memory by splitting oversized requests into condition batches.
    if condition_count > execution.batch_size:
        results = []
        for begin in range(0, condition_count, execution.batch_size):
            part = slice(begin, min(begin + execution.batch_size, condition_count))
            results.append(
                run_aligned_conditions(
                    source_cell,
                    plan,
                    prepared,
                    voltage_index=voltage_index[part],
                    release_step=release_step[part],
                    condition_input=condition_input[part],
                    condition_arrival=condition_arrival[part],
                    condition_amplitude=condition_amplitude[part],
                    probe_steps=probe_steps,
                    dt=dt,
                    spike_threshold_mV=spike_threshold_mV,
                    execution=execution,
                )
            )
        return AlignedConditionResult(
            voltage_age_mV=np.concatenate([result.voltage_age_mV for result in results], axis=0),
            spike_age=np.concatenate([result.spike_age for result in results], axis=0),
        )
    # 3. Restore and execute each device's share of the batch.
    devices = execution.devices
    shards = _condition_shards(condition_count, len(devices))

    def run_shard(worker: int, part: slice) -> AlignedConditionResult:
        device = devices[worker]
        # BrainState environments are thread-local; worker threads otherwise
        # start at the library's FP32 default even when the parent selected 64.
        brainstate.environ.set(precision=64)
        with jax.default_device(device):
            result = _run_aligned_conditions_on_device(
                source_cell,
                plan,
                prepared,
                voltage_index=voltage_index[part],
                release_step=release_step[part],
                condition_input=condition_input[part],
                condition_arrival=condition_arrival[part],
                condition_amplitude=condition_amplitude[part],
                probe_steps=probe_steps,
                dt=dt,
                spike_threshold_mV=spike_threshold_mV,
                device=device,
            )
            return _host_aligned_result(result)

    if len(shards) == 1:
        return run_shard(0, shards[0])
    with ThreadPoolExecutor(max_workers=len(shards), thread_name_prefix="reduction-calibration-gpu") as executor:
        futures = tuple(executor.submit(run_shard, worker, part) for worker, part in enumerate(shards))
        results = tuple(future.result() for future in futures)

    # 4. Join host traces in their original condition order.
    return AlignedConditionResult(
        voltage_age_mV=np.concatenate(tuple(result.voltage_age_mV for result in results), axis=0),
        spike_age=np.concatenate(tuple(result.spike_age for result in results), axis=0),
    )


def _run_aligned_conditions_on_device(
    source_cell,
    plan,
    prepared: PreparedResponseStates,
    *,
    voltage_index: np.ndarray,
    release_step: np.ndarray,
    condition_input: np.ndarray,
    condition_arrival: np.ndarray,
    condition_amplitude: np.ndarray,
    probe_steps: int,
    dt,
    spike_threshold_mV: float,
    device,
) -> AlignedConditionResult:
    """Run detailed conditions and align every trace to clamp release.

    ``condition_arrival`` is measured from the start of this condition.  The
    old input is normally at zero and the new input at ``release_step``.
    The soma is clamped while ``step < release_step`` and is free on the
    release step itself.
    """
    # 1. Restore the selected detailed states and pack their synaptic arrivals.
    condition_count = int(voltage_index.size)
    # Keep the author's fixed total horizon measured from the old input.
    # Every batch therefore has one scan shape; a larger tau simply leaves a
    # shorter valid trace after clamp release.
    n_steps = int(probe_steps) + 1
    active_endpoint = condition_input >= 0
    condition_index, endpoint = np.nonzero(active_endpoint)
    compact_ref = [None]
    cell = _make_calibration_cell(source_cell, condition_count, compact_ref, device=device)
    _seed_prepared_states(prepared, cell, voltage_index, device=device)
    frozen_cell_state = _capture_population_state(cell, condition_count)
    compact_synapses = lower_compact_calibration_synapses(
        cell,
        plan,
        condition_index=condition_index.astype(np.int32, copy=False),
        arrival_input=condition_input[condition_index, endpoint],
        arrival_step=condition_arrival[condition_index, endpoint],
        arrival_amplitude=condition_amplitude[condition_index, endpoint],
        n_steps=n_steps,
        device=device,
    )
    compact_ref[0] = compact_synapses
    root_cv = _root_cv(cell)
    initial_voltage = cell.V.value[..., root_cv].to_decimal(u.mV)
    cursor = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))
    release_device = jnp.asarray(release_step, dtype=jnp.int32)
    maximum_release = int(np.max(release_step, initial=0))
    target_voltage = prepared.voltage_grid_mV[voltage_index]

    # 2. Bind synaptic current and hold the soma until each condition releases.
    def synapse_current(point_voltage):
        return compact_synapses.current(
            point_voltage,
            condition_active=cursor.value >= release_device,
        )

    cell.add_current_input("reduce_compact_synapses", synapse_current)

    def clamp_current(point_voltage):
        return _soma_clamp_current(
            cell,
            point_voltage,
            target_voltage,
            cursor.value < release_device,
        )

    cell.add_current_input("dai_li_soma_voltage_clamp", clamp_current)
    zero = jnp.asarray(0, dtype=jnp.int32)

    def idle_drive():
        return zero

    drive_branches = [idle_drive]
    for bucket in compact_synapses.buckets:

        def apply_bucket(bucket=bucket):
            point_voltage = cell._dhs_point_voltage(cell.V.value)
            compact_synapses.apply_bucket(bucket, point_voltage)
            return zero

        drive_branches.append(apply_bucket)
    drive_branches = tuple(drive_branches)
    # 3. Advance the compiled detailed loop, aging only synapses before release.
    times = prepared.response_start_ms * u.ms + u.math.arange(n_steps) * dt
    with brainstate.environ.context(dt=dt):

        def step(t):
            with brainstate.environ.context(t=t):
                index = cursor.value
                brainstate.transform.switch(compact_synapses.step_bucket[index], drive_branches)
                cell._update_dynamics()
                inactive = index < release_device
                if maximum_release:

                    def restore_cell_state():
                        _restore_population_rows(frozen_cell_state, inactive)
                        return zero

                    brainstate.transform.cond(
                        index < maximum_release,
                        restore_cell_state,
                        idle_drive,
                    )
                cursor.value = index + 1
            return cell.V.value[..., root_cv].to_decimal(u.mV)

        voltage_mV = jax.block_until_ready(brainstate.transform.for_loop(step, times))

    # 4. Align voltage and confirmation crossings to each release boundary.
    age = jnp.arange(int(probe_steps) + 1, dtype=jnp.int32)[None, :]
    aligned_step = jnp.asarray(release_step, dtype=jnp.int32)[:, None] + age
    valid_age = aligned_step < n_steps
    aligned_step = jnp.minimum(aligned_step, n_steps - 1)
    voltage_by_condition = jnp.swapaxes(voltage_mV, 0, 1)
    aligned_voltage = jnp.take_along_axis(voltage_by_condition, aligned_step, axis=1)
    # REST preparation ends at onset + 1 mV. Retain the first crossing of
    # the same onset + 1.5 mV confirmation level used by the CUDA monitor.
    confirmation = confirmation_voltage_mV(spike_threshold_mV)
    above_confirmation = aligned_voltage >= jnp.asarray(confirmation, dtype=aligned_voltage.dtype)
    previously_above = jnp.concatenate(
        (
            (initial_voltage >= confirmation)[:, None],
            above_confirmation[:, :-1],
        ),
        axis=1,
    )
    aligned_spike = above_confirmation & ~previously_above & valid_age
    aligned_voltage, aligned_spike = jax.block_until_ready((aligned_voltage, aligned_spike))
    # Break current-input closures before the next calibration batch.
    cell = compact_synapses = cursor = frozen_cell_state = drive_branches = None
    return AlignedConditionResult(aligned_voltage, aligned_spike)
