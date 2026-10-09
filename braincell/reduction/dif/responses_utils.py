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

"""Lay out response banks and restore detailed states and synapses for their measurement."""

from __future__ import annotations

from dataclasses import dataclass, replace

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute.layouts import mechanism_signature
from braincell._multi_compartment.cell import Cell
from braincell.mech import ScalarEventInput, TriggerEventInput, get_registry
from braincell.mech import Synapse as SynapsePlacement
from braincell.quad import IndependentIntegration, ind_exp_euler_step
from braincell.quad._staggered import dhs_voltage_step
from braincell.reduction.dif.calibration_utils import _root_cv, _root_cv_and_point, explicit_time_coordinates

CLAMP_SERIES_RESISTANCE_MOHM = 1.0e-9
_CLAMP_CONDUCTANCE_US = 1.0 / CLAMP_SERIES_RESISTANCE_MOHM


@dataclass(frozen=True)
class PreparedResponseStates:
    """Detailed cell state bank after Dai--Li somatic voltage clamps."""

    cell: Cell
    voltage_grid_mV: np.ndarray
    response_start_ms: float


def _make_calibration_cell(
    source_cell,
    condition_count: int,
    compact_ref,
    *,
    device,
):
    calibration_voltage_step = getattr(
        source_cell.solver,
        "braincell_calibration_voltage_step",
        None,
    )

    def calibration_solver(target):
        compact_calibration_staggered_step(
            target,
            compact_ref[0],
            voltage_step=calibration_voltage_step,
        )

    cell = Cell(
        source_cell._declaration_morpho,
        pop_size=(condition_count,),
        cv_policy=source_cell.cv_policy,
        V_th=source_cell._V_th_declaration,
        V_init=source_cell.V.value.reshape((-1, source_cell.runtime.n_cv))[0, 0],
        spk_fun=source_cell.spk_fun,
        solver=calibration_solver,
        cache_ion_total_current=source_cell.cache_ion_total_current,
        ion_channel_update_order=source_cell.ion_channel_update_order,
        name=None if source_cell.name is None else f"{source_cell.name}_reduce_calibration",
    )
    cell._paint_rules = source_cell.paint_rules
    place_rules = []
    for rule in source_cell.place_rules:
        mechanisms = tuple(mechanism for mechanism in rule.mechanisms if not isinstance(mechanism, SynapsePlacement))
        if mechanisms:
            place_rules.append(replace(rule, mechanisms=mechanisms))
    cell._place_rules = tuple(place_rules)
    cell._invalidate_discretization_cache()
    cell.init_state()
    _copy_homogeneous_density_state(source_cell, cell)
    cell.reset_state()
    _put_cell_states_on_device(cell, device)
    return cell


def _soma_clamp_current(cell, point_voltage, target_mV, active):
    """Return bNEURON-equivalent SEClamp current in point-density space."""
    _, root_point = _root_cv_and_point(cell)
    target = jnp.asarray(target_mV, dtype=jnp.asarray(0.0).dtype) * u.mV
    voltage_error = target - point_voltage[..., root_point]
    total_current = (voltage_error * (_CLAMP_CONDUCTANCE_US * u.uS)).in_unit(u.nA)
    density = (total_current / cell.runtime.point_area[root_point]).in_unit(u.nA / u.cm**2)
    active_density = u.math.where(active, density, u.math.zeros_like(density))
    output = u.Quantity(
        jnp.zeros(point_voltage.shape, dtype=point_voltage.dtype),
        u.nA / u.cm**2,
    )
    return output.at[..., root_point].set(active_density)


def _density_layout_key(cell, layout) -> tuple[object, ...]:
    declaration = cell.runtime.get_layout_mechanism(layout.id)
    return (layout.target,) + mechanism_signature(declaration)


def _broadcast_homogeneous_density_value(
    value,
    *,
    source_pop_size: tuple[int, ...],
    target_pop_size: tuple[int, ...],
    target_value,
    description: str,
):
    source_is_quantity = isinstance(value, u.Quantity)
    target_is_quantity = isinstance(target_value, u.Quantity)
    if source_is_quantity != target_is_quantity:
        raise RuntimeError(f"DIF calibration density state kind differs for {description}.")
    payload = value.mantissa if source_is_quantity else value
    target_payload = target_value.mantissa if target_is_quantity else target_value
    array = np.asarray(payload)
    target_array = np.asarray(target_payload)
    source_rank = len(source_pop_size)
    target_rank = len(target_pop_size)
    trailing_shape = tuple(array.shape[source_rank:])
    if tuple(target_array.shape[target_rank:]) != trailing_shape:
        raise RuntimeError(f"DIF calibration density layout differs for {description}.")
    representative = array.reshape((-1,) + trailing_shape)[0]
    broadcast = np.broadcast_to(representative, tuple(target_pop_size) + trailing_shape).copy()
    if source_is_quantity:
        return u.Quantity(broadcast, value.unit)
    return broadcast


def _copy_homogeneous_density_state(source_cell, calibration_cell) -> None:
    target_layout_by_key = {
        _density_layout_key(calibration_cell, layout): layout
        for layout in calibration_cell.runtime.layouts
        if layout.target == "density"
    }
    for source_layout in source_cell.runtime.layouts:
        if source_layout.target != "density":
            continue
        target_layout = target_layout_by_key[_density_layout_key(source_cell, source_layout)]
        for (layout_id, name), value in source_cell.runtime.state_buffers.items():
            if int(layout_id) != int(source_layout.id):
                continue
            target_key = (int(target_layout.id), str(name))
            value = source_cell.runtime.get_state(layout_id, name)
            target_value = calibration_cell.runtime.get_state(*target_key)
            calibration_cell.runtime.set_state(
                target_layout.id,
                name,
                _broadcast_homogeneous_density_value(
                    value,
                    source_pop_size=tuple(source_cell.runtime.pop_size),
                    target_pop_size=tuple(calibration_cell.runtime.pop_size),
                    target_value=target_value,
                    description=f"layout {source_layout.kind!r}, field {name!r}",
                ),
            )


def _put_cell_states_on_device(cell, device) -> None:
    for state in cell.states().values():
        state.value = _device_put_value(state.value, device)


@dataclass(frozen=True)
class AlignedConditionResult:
    """Soma traces aligned to each condition's clamp-release step."""

    voltage_age_mV: object
    spike_age: object


def _condition_shards(condition_count: int, device_count: int) -> tuple[slice, ...]:
    worker_count = min(int(condition_count), int(device_count))
    quotient, remainder = divmod(int(condition_count), worker_count)
    shards = []
    begin = 0
    for worker in range(worker_count):
        size = quotient + (1 if worker < remainder else 0)
        shards.append(slice(begin, begin + size))
        begin += size
    return tuple(shards)


def _host_aligned_result(result: AlignedConditionResult) -> AlignedConditionResult:
    return AlignedConditionResult(
        voltage_age_mV=np.asarray(jax.device_get(result.voltage_age_mV)),
        spike_age=np.asarray(jax.device_get(result.spike_age)),
    )


def _select_prepared_value(value, voltage_index, *, device):
    if isinstance(value, u.Quantity):
        payload = jax.device_put(value.mantissa, device)
        selected = jnp.take(payload, voltage_index, axis=0)
        return u.Quantity(selected, value.unit)
    payload = jax.device_put(value, device)
    return jnp.take(payload, voltage_index, axis=0)


def _seed_prepared_states(
    prepared: PreparedResponseStates,
    target_cell,
    voltage_index,
    *,
    device,
) -> None:
    """Copy every population-shaped detailed state from the clamp bank."""
    source_cell = prepared.cell
    index = jnp.asarray(voltage_index, dtype=jnp.int32)
    source_states = source_cell.states()
    target_states = target_cell.states()
    source_rows = int(prepared.voltage_grid_mV.size)
    target_rows = int(np.asarray(voltage_index).size)
    for path, target_state in target_states.items():
        source_state = source_states.get(path)
        if source_state is None:
            continue
        source_value = source_state.value
        target_value = target_state.value
        source_shape = getattr(source_value, "shape", ())
        target_shape = getattr(target_value, "shape", ())
        if not source_shape or not target_shape:
            continue
        if int(source_shape[0]) != source_rows or int(target_shape[0]) != target_rows:
            continue
        if tuple(source_shape[1:]) != tuple(target_shape[1:]):
            raise RuntimeError(
                f"DIF prepared state shape differs at {path!r}: {source_shape!r} versus {target_shape!r}."
            )
        target_state.value = _select_prepared_value(source_value, index, device=device)
    target_cell.spike.value = jnp.zeros_like(target_cell.spike.value)
    target_cell.clear_ion_total_current_cache()


def _capture_population_state(cell, condition_count: int):
    """Capture cell-owned population rows, excluding compact synapses."""
    frozen = []
    for state in cell.states().values():
        value = state.value
        shape = getattr(value, "shape", ())
        if shape and int(shape[0]) == int(condition_count):
            frozen.append((state, value))
    return tuple(frozen)


def _restore_population_rows(frozen_state, restore_row) -> None:
    """Restore pre-aged rows so only their compact synapses can evolve."""
    for state, initial_value in frozen_state:
        value = state.value
        row_mask = restore_row.reshape((restore_row.shape[0],) + (1,) * (len(value.shape) - 1))
        state.value = u.math.where(row_mask, initial_value, value)


@dataclass(frozen=True)
class PostSpikeLibrary:
    state_ptr: np.ndarray
    state_voltage_mV: np.ndarray
    single_support_steps: np.ndarray
    rebase_single_support_steps: np.ndarray
    single_voltage_mV: np.ndarray
    single_voltage_ptr: np.ndarray
    pair_voltage_mV: np.ndarray
    pair_voltage_ptr: np.ndarray
    rebase_single_voltage_mV: np.ndarray
    rebase_single_voltage_ptr: np.ndarray
    rebase_pair_voltage_mV: np.ndarray
    rebase_pair_voltage_ptr: np.ndarray


def _single_curve_layout(support_steps, event_age_steps):
    age_count = int(event_age_steps.size)
    lengths = np.zeros((support_steps.size, age_count), dtype=np.int64)
    for location, support in enumerate(support_steps.tolist()):
        # Dp is the final above-tolerance sample.  A node at or beyond Dp is
        # an implicit zero guard for interpolation, not another live curve.
        valid = event_age_steps < int(support)
        lengths[location, valid] = support - event_age_steps[valid] + 1
    pointer = np.empty((lengths.size + 1,), dtype=np.int64)
    pointer[0] = 0
    np.cumsum(lengths.reshape(-1), out=pointer[1:])
    return pointer


def _axis_coordinates(grid, maximum):
    """Return exact lower/upper grid indices for every integer target."""
    targets = np.arange(int(maximum) + 1, dtype=np.int32)
    insertion = np.searchsorted(grid, targets, side="left")
    if np.any(insertion >= grid.size):
        raise RuntimeError("DIF rebase support exceeds its explicit time axis.")
    exact = grid[insertion] == targets
    lower = np.maximum(insertion - 1, 0).astype(np.int32, copy=False)
    upper = insertion.astype(np.int32, copy=False)
    lower[exact] = insertion[exact]
    return lower, upper


def _rebase_pair_node_mask(
    pair_slots,
    support_steps,
    tau_steps,
    release_age_steps,
):
    """Mark every Dai--Li grid node required by a valid runtime query.

    The valid ``(delta_t, delta_T)`` domain is triangular and differs by the
    two synapse supports.  Instead of padding it heuristically, enumerate the
    integer runtime domain once per distinct support pair and retain exactly
    the three nodes used by the aligned Dai--Li interpolation rule.
    """
    tau_count = int(tau_steps.size)
    release_count = int(release_age_steps.size)
    required = np.zeros((len(pair_slots), tau_count, release_count), dtype=np.bool_)
    cached = {}
    for row, (old, new) in enumerate(pair_slots):
        support_key = (int(support_steps[old]), int(support_steps[new]))
        mask = cached.get(support_key)
        if mask is None:
            old_support, new_support = support_key
            tau_target = np.arange(old_support, dtype=np.int32)
            release_target = np.arange(old_support, dtype=np.int32)
            valid = (release_target[None, :] >= tau_target[:, None]) & (
                release_target[None, :] < tau_target[:, None] + new_support
            )
            target_tau, target_release = np.nonzero(valid)
            tau_lower, tau_upper = _axis_coordinates(tau_steps, old_support)
            release_lower, release_upper = _axis_coordinates(release_age_steps, old_support)
            tl = tau_lower[target_tau]
            tu = tau_upper[target_tau]
            rl = release_lower[target_release]
            ru = release_upper[target_release]
            triangular = (tl != tu) & (rl != ru) & (release_age_steps[rl] < tau_steps[tu])
            mask = np.zeros((tau_count, release_count), dtype=np.bool_)
            mask[tl, rl] = True
            mask[tl, ru] = True
            mask[tu[~triangular], rl[~triangular]] = True
            mask[tu[triangular], ru[triangular]] = True
            cached[support_key] = mask
        required[row] = mask
    return required


def _rebase_pair_curve_layout(
    node_mask,
    pair_slots,
    support_steps,
    tau_steps,
    release_age_steps,
):
    lengths = np.zeros(node_mask.shape, dtype=np.int64)
    for row, (old, new) in enumerate(pair_slots):
        tau_index, release_index = np.nonzero(node_mask[row])
        old_remaining = int(support_steps[old]) - release_age_steps[release_index]
        new_age = release_age_steps[release_index] - tau_steps[tau_index]
        new_remaining = int(support_steps[new]) - new_age
        physical = np.minimum(old_remaining, new_remaining) + 1
        lengths[row, tau_index, release_index] = np.maximum(physical, 0)
    pointer = np.empty((lengths.size + 1,), dtype=np.int64)
    pointer[0] = 0
    np.cumsum(lengths.reshape(-1), out=pointer[1:])
    return pointer


def _rebase_pair_conditions(
    pair_slots,
    pair_begin,
    node_mask,
    curve_pointer,
):
    old_parts = []
    new_parts = []
    tau_parts = []
    release_parts = []
    tau_index = np.arange(node_mask.shape[1], dtype=np.int64)[:, None]
    release_index = np.arange(node_mask.shape[2], dtype=np.int64)[None, :]
    for local_row, (old, new) in enumerate(pair_slots):
        row = pair_begin + local_row
        slots = (row * node_mask.shape[1] + tau_index) * node_mask.shape[2] + release_index
        physical = (curve_pointer[slots + 1] - curve_pointer[slots]) > 0
        tau, release = np.nonzero(node_mask[row] & physical)
        tau = tau.astype(np.int32, copy=False)
        release = release.astype(np.int32, copy=False)
        if release.size:
            old_parts.append(np.full(release.size, old, dtype=np.int32))
            new_parts.append(np.full(release.size, new, dtype=np.int32))
            tau_parts.append(tau)
            release_parts.append(release)
    if not old_parts:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty, empty, empty
    return tuple(
        np.concatenate(parts).astype(np.int32, copy=False) for parts in (old_parts, new_parts, tau_parts, release_parts)
    )


def _interpolate_rebase_single_curve(
    values,
    pointer,
    *,
    location,
    event_age,
    event_age_steps,
    event_age_count,
    requested_length,
):
    """Interpolate an aged Tref single using physical event age."""
    lower, upper, ratio = explicit_time_coordinates(event_age_steps, event_age)
    slot = location * event_age_count
    lower_begin = int(pointer[slot + lower])
    lower_end = int(pointer[slot + lower + 1])
    lower_curve = np.zeros((requested_length,), dtype=values.dtype)
    lower_payload = values[lower_begin:lower_end]
    lower_curve[: min(requested_length, lower_payload.size)] = lower_payload[:requested_length]
    if lower == upper:
        return lower_curve
    upper_begin = int(pointer[slot + upper])
    upper_end = int(pointer[slot + upper + 1])
    upper_curve = np.zeros((requested_length,), dtype=values.dtype)
    upper_payload = values[upper_begin:upper_end]
    upper_curve[: min(requested_length, upper_payload.size)] = upper_payload[:requested_length]
    return lower_curve + ratio * (upper_curve - lower_curve)


SINGLE_RESPONSE_TOLERANCE_MV = 1.0e-3
SINGLE_RESPONSE_TAIL_MS = 10.0
SINGLE_RESPONSE_HORIZON_BOUND_MV = 2.5e-2


@dataclass(frozen=True)
class ResponseParameters:
    V_init_mV: float
    V_rest_mV: float
    V_th_mV: float


def build_response_parameters(cell) -> ResponseParameters:
    root_voltage = cell.V.value[..., _root_cv(cell)]
    representative = root_voltage.reshape((-1,))[0]
    initial_mV = float(np.asarray(representative.to_decimal(u.mV), dtype=np.float64).reshape(()))
    threshold = np.asarray(cell._V_th_declaration.to_decimal(u.mV), dtype=np.float64)
    return ResponseParameters(
        V_init_mV=initial_mV,
        V_rest_mV=initial_mV,
        V_th_mV=float(threshold.reshape(-1)[0]),
    )


def voltage_response_support_steps(
    response_mV: np.ndarray,
    *,
    dt_ms: float,
    description: str,
) -> np.ndarray:
    """Measure per-location Dai--Li Dp from a bank of soma responses."""
    response = np.asarray(response_mV, dtype=np.float64)
    if response.ndim != 3 or response.shape[-1] < 2:
        raise ValueError("DIF single responses must have shape (location,state,age).")
    envelope = np.max(np.abs(response), axis=1)
    tail_steps = max(1, int(np.ceil(SINGLE_RESPONSE_TAIL_MS / float(dt_ms))))
    if envelope.shape[-1] <= tail_steps:
        raise RuntimeError("DIF single probe is shorter than its required quiet tail.")
    tail_peak = np.max(envelope[:, -tail_steps:], axis=1)
    bad = np.flatnonzero(tail_peak > SINGLE_RESPONSE_TOLERANCE_MV)
    if bad.size:
        unsafe = bad[tail_peak[bad] > SINGLE_RESPONSE_HORIZON_BOUND_MV]
        if unsafe.size:
            raise RuntimeError(
                f"{description} 240 ms single-response truncation exceeds "
                f"{SINGLE_RESPONSE_HORIZON_BOUND_MV:g} mV; "
                f"locations={unsafe.tolist()}, "
                f"tail_peak_mV={tail_peak[unsafe].tolist()}."
            )
    age = np.arange(envelope.shape[-1], dtype=np.int32)
    active = envelope > SINGLE_RESPONSE_TOLERANCE_MV
    last = np.max(np.where(active, age[None, :], -1), axis=1)
    # Keep one timestep before the probe endpoint so the nonuniform time-grid
    # lookup always has a valid upper neighbor.  The discarded tail is
    # explicitly bounded above rather than silently treated as zero.
    return np.minimum(np.maximum(last, 0), envelope.shape[-1] - 2).astype(np.int32, copy=False)


def allocate_single_voltage(
    support_steps: np.ndarray,
    state_count: int,
    guard_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate final single-curve storage, including its zero guard."""
    support = np.asarray(support_steps, dtype=np.int32)
    pointer = np.empty((support.size + 1,), dtype=np.int64)
    pointer[0] = 0
    np.cumsum((support.astype(np.int64) + 1) * state_count, out=pointer[1:])
    return np.zeros(int(pointer[-1]) + guard_steps, dtype=np.float64), pointer


def pair_conditions(pair_slots, support_steps, tau_steps):
    """Return unordered simultaneous and ordered delayed pair conditions."""
    old_parts = []
    new_parts = []
    tau_index_parts = []

    simultaneous = pair_slots[pair_slots[:, 0] <= pair_slots[:, 1]]
    old_parts.append(simultaneous[:, 0])
    new_parts.append(simultaneous[:, 1])
    tau_index_parts.append(np.zeros(simultaneous.shape[0], dtype=np.int32))

    for old in np.unique(pair_slots[:, 0]):
        partners = pair_slots[pair_slots[:, 0] == old, 1]
        valid_tau = np.flatnonzero((tau_steps > 0) & (tau_steps < int(support_steps[old]))).astype(np.int32, copy=False)
        if valid_tau.size == 0:
            continue
        old_parts.append(np.full(valid_tau.size * partners.size, old, dtype=np.int32))
        new_parts.append(np.tile(partners, valid_tau.size))
        tau_index_parts.append(np.repeat(valid_tau, partners.size))
    return (
        np.concatenate(old_parts),
        np.concatenate(new_parts),
        np.concatenate(tau_index_parts),
    )


def pair_curve_layout(
    pair_slots: np.ndarray,
    single_support: np.ndarray,
    tau_steps: np.ndarray,
) -> np.ndarray:
    """Index pair/tau curves within one state-major voltage bank."""
    n_tau = int(tau_steps.size)
    n_pair = len(pair_slots)
    lengths = np.zeros((n_pair, n_tau), dtype=np.int64)
    for row, (old, new) in enumerate(pair_slots):
        valid_tau = tau_steps < int(single_support[old])
        lengths[row, valid_tau] = (
            np.minimum(int(single_support[old]) - tau_steps[valid_tau], int(single_support[new])) + 1
        )
    pointer = np.empty((n_pair * n_tau + 1,), dtype=np.int64)
    pointer[0] = 0
    np.cumsum(lengths.reshape(-1), out=pointer[1:])
    return pointer


def padded_conditions(values: np.ndarray, begin: int, end: int, size: int) -> np.ndarray:
    """Fill a fixed-size calibration batch by repeating its first condition."""
    output = np.empty((size,), dtype=values.dtype)
    actual = end - begin
    output[:actual] = values[begin:end]
    output[actual:] = output[0]
    return output


def require_no_condition_spikes(spike_age, *, description: str) -> None:
    """Require baseline calibration rows to remain non-spiking."""
    spike = np.asarray(spike_age, dtype=np.bool_)
    if spike.ndim != 2:
        raise ValueError("DIF calibration spike mask must be two-dimensional.")
    rows = np.flatnonzero(np.any(spike, axis=1))
    if rows.size:
        first_age = np.argmax(spike[rows], axis=1)
        raise RuntimeError(
            f"DIF calibration rejected: {description} emitted detailed "
            f"spikes in rows={rows[:16].tolist()} first_spike_age_steps="
            f"{first_age[:16].tolist()}. A no-input baseline may not spike."
        )


def truncate_voltage_after_first_spike(
    voltage_mV,
    spike_age,
    baseline_voltage_mV,
) -> np.ndarray:
    """End each input response immediately after its first confirmation crossing.

    The first confirmation sample is retained so the reduced runtime can
    detect the same onset + 1.5 mV crossing. Every later absolute-voltage
    sample is replaced by that row's no-input baseline.  Consequently a
    single response becomes zero after the cutoff, while a pair residual
    cancels its constituent singles and makes the complete pair condition
    disappear.  No detailed AP/AHP tail is packed into an ordinary table.
    """
    voltage = np.asarray(voltage_mV, dtype=np.float64)
    spike = np.asarray(spike_age, dtype=np.bool_)
    baseline = np.asarray(baseline_voltage_mV, dtype=np.float64)
    if voltage.ndim != 2 or spike.shape != voltage.shape:
        raise ValueError("DIF calibration voltage and spike arrays must have the same two-dimensional shape.")
    baseline = np.broadcast_to(baseline, voltage.shape)
    rows = np.flatnonzero(np.any(spike, axis=1))
    if rows.size:
        voltage = voltage.copy()
        first_age = np.argmax(spike[rows], axis=1)
        for row, cutoff in zip(rows.tolist(), first_age.tolist(), strict=True):
            voltage[row, int(cutoff) + 1 :] = baseline[row, int(cutoff) + 1 :]
    return voltage


@dataclass(frozen=True)
class CompactSynapseGroup:
    """One synapse type stored over its actual calibration instances."""

    synapse: object
    flat_point_index: object
    condition_index: object
    point_area: object
    event_template: object


@dataclass(frozen=True)
class CompactArrivalInjection:
    """Static unit arrivals for one compact mechanism group."""

    group_index: int
    instance_index: object
    amplitude: object


@dataclass(frozen=True)
class CompactArrivalBucket:
    """All direct synapse injections at one statically known arrival step."""

    injections: tuple[CompactArrivalInjection, ...]


@dataclass(frozen=True)
class CompactCalibrationSynapses:
    """Executable compact synapse groups and their static event schedule."""

    groups: tuple[CompactSynapseGroup, ...]
    step_bucket: object
    buckets: tuple[CompactArrivalBucket, ...]

    def apply_bucket(self, bucket: CompactArrivalBucket, point_voltage) -> None:
        """Inject this step's precompiled arrivals without persistent state."""
        flat_voltage = point_voltage.reshape((-1,))
        for injection in bucket.injections:
            group = self.groups[injection.group_index]
            synapse = group.synapse
            zero = group.event_template
            if isinstance(synapse.event_input, ScalarEventInput):
                amplitude = u.Quantity(injection.amplitude, synapse.event_input.unit)
            else:
                amplitude = injection.amplitude.astype(jnp.int32)
            drive = zero.at[injection.instance_index].add(amplitude)
            synapse_voltage = flat_voltage[group.flat_point_index]
            synapse.apply_events(drive, synapse_voltage)

    def current(self, point_voltage, condition_active=None):
        """Gather instance voltage and scatter mechanism current to point space."""
        flat_voltage = point_voltage.reshape((-1,))
        density = u.Quantity(
            jnp.zeros(
                flat_voltage.shape,
                dtype=jnp.asarray(0.0).dtype,
            ),
            u.nA / u.cm**2,
        )
        for group in self.groups:
            local_voltage = flat_voltage[group.flat_point_index]
            contribution = group.synapse.current(local_voltage)
            if condition_active is not None:
                contribution = contribution * condition_active[group.condition_index]
            contribution = contribution / group.point_area
            density = density.at[group.flat_point_index].add(contribution)
        return density.reshape(point_voltage.shape)

    def integrate(self, point_voltage) -> None:
        """Advance every actual compact mechanism instance by one timestep."""
        flat_voltage = point_voltage.reshape((-1,))
        for group in self.groups:
            local_voltage = flat_voltage[group.flat_point_index]
            if isinstance(group.synapse, IndependentIntegration):
                group.synapse.ind_update(local_voltage)
            else:
                ind_exp_euler_step(group.synapse, local_voltage)


def lower_compact_calibration_synapses(
    calibration_cell,
    plan,
    *,
    condition_index,
    arrival_input,
    arrival_step,
    arrival_amplitude=None,
    n_steps: int,
    device,
) -> CompactCalibrationSynapses:
    """Lower condition endpoints into actual-instance synapse populations."""
    condition = np.asarray(condition_index, dtype=np.int32)
    arrival_input = np.asarray(arrival_input, dtype=np.int32)
    arrival_step = np.asarray(arrival_step, dtype=np.int32)
    if arrival_amplitude is None:
        arrival_amplitude = np.ones(arrival_step.shape, dtype=np.float64)
    else:
        arrival_amplitude = np.array(arrival_amplitude, dtype=np.float64, copy=True)
    if arrival_amplitude.shape != arrival_step.shape:
        raise ValueError("Calibration arrival amplitudes must align with arrivals.")
    if not np.all(np.isfinite(arrival_amplitude)) or np.any(arrival_amplitude <= 0.0):
        raise ValueError("Calibration arrival amplitudes must be finite and positive.")
    input_layout = np.asarray(plan.input_layout_id, dtype=np.int32)
    input_synapse = np.asarray(plan.input_synapse_index, dtype=np.int32)
    source_layouts = {
        id(cell): {int(layout.id): layout for layout, _ in cell.runtime.iter_synapse_layouts()}
        for cell in {id(cell): cell for cell in plan.input_cells}.values()
    }

    builders: dict[str, dict[str, object]] = {}
    arrival_group = np.empty(arrival_input.shape, dtype=np.int32)
    arrival_instance = np.empty(arrival_input.shape, dtype=np.int32)
    n_point = int(calibration_cell.runtime.n_point)

    for arrival_ordinal, (condition_id, input_id) in enumerate(zip(condition.tolist(), arrival_input.tolist())):
        layout_id = int(input_layout[input_id])
        synapse_index = int(input_synapse[input_id])
        input_cell = plan.input_cells[input_id]
        layout = source_layouts[id(input_cell)][layout_id]
        declaration = input_cell.runtime.get_layout_mechanism(layout_id)
        synapse_type = declaration.synapse_type
        builder = builders.get(synapse_type)
        if builder is None:
            builder = {
                "group_index": len(builders),
                "runtime_cls": get_registry().get("synapse", synapse_type),
                "instances": [],
                "instance_by_key": {},
            }
            builders[synapse_type] = builder

        instance_by_key = builder["instance_by_key"]
        instance_key = (condition_id, int(plan.input_instance_index[input_id]))
        instance_id = instance_by_key.get(instance_key)
        if instance_id is None:
            instance_id = len(builder["instances"])
            instance_by_key[instance_key] = instance_id
            point_index = int(layout.point_index[synapse_index])
            builder["instances"].append(
                (
                    condition_id,
                    layout,
                    input_cell.runtime.get_runtime_node(layout_id),
                    synapse_index,
                    point_index,
                )
            )
        event_input = builder["runtime_cls"].event_input
        if isinstance(event_input, ScalarEventInput):
            arrival_amplitude[arrival_ordinal] *= float(plan.input_event_weights[input_id].to_decimal(event_input.unit))
        arrival_group[arrival_ordinal] = int(builder["group_index"])
        arrival_instance[arrival_ordinal] = int(instance_id)

    groups = []
    initial_point_voltage = calibration_cell._dhs_point_voltage(calibration_cell.V.value)
    initial_flat_voltage = initial_point_voltage.reshape((-1,))
    point_area = calibration_cell.runtime.point_area
    for synapse_type, builder in builders.items():
        instances = builder["instances"]
        runtime_cls = builder["runtime_cls"]
        parameters = _compact_constructor_parameters(runtime_cls, instances, device=device)
        synapse = runtime_cls(
            size=(len(instances),),
            name=f"reduce_calibration_{synapse_type}",
            **parameters,
        )
        flat_point_index = jnp.asarray(
            [condition_id * n_point + point_index for condition_id, _, _, _, point_index in instances],
            dtype=jnp.int32,
        )
        condition_index = jnp.asarray(
            [condition_id for condition_id, _, _, _, _ in instances],
            dtype=jnp.int32,
        )
        physical_point_index = flat_point_index % n_point
        synapse.init_state(initial_flat_voltage[flat_point_index])
        if isinstance(synapse.event_input, ScalarEventInput):
            event_template = u.Quantity(jnp.zeros((len(instances),)), synapse.event_input.unit)
        elif isinstance(synapse.event_input, TriggerEventInput):
            event_template = jnp.zeros((len(instances),), dtype=jnp.int32)
        else:
            raise TypeError("Calibration requires a discrete synapse event-input contract.")
        groups.append(
            CompactSynapseGroup(
                synapse=synapse,
                flat_point_index=flat_point_index,
                condition_index=condition_index,
                point_area=point_area[physical_point_index],
                event_template=event_template,
            )
        )

    step_bucket, buckets = _compile_arrival_schedule(
        arrival_step,
        arrival_group,
        arrival_instance,
        arrival_amplitude,
        n_steps=int(n_steps),
    )
    return CompactCalibrationSynapses(
        groups=tuple(groups),
        step_bucket=step_bucket,
        buckets=buckets,
    )


def compact_calibration_staggered_step(
    target,
    compact_synapses: CompactCalibrationSynapses | None,
    *,
    voltage_step=None,
) -> None:
    """Run one staggered Cell step with optional compact synapses.

    ``voltage_step`` is supplied by the detailed source solver when it uses a
    numerically specialised DHS implementation. It accepts ``target`` and
    keyword arguments ``t`` and ``dt``, like BrainCell's ordinary DHS step.
    The rest of the staggered ordering stays local because calibration uses
    compact synapses rather than the source Cell's declared synapse runtime.
    """
    t = brainstate.environ.get("t", 0.0)
    dt = brainstate.environ.get("dt")
    target.cache_ion_total_currents(target.V.value)
    if voltage_step is None:
        voltage_step = dhs_voltage_step
    voltage_step(target, t=t, dt=dt)
    point_voltage = target._dhs_point_voltage(target.V.value)
    if compact_synapses is not None:
        compact_synapses.integrate(point_voltage)
    if target.ion_channel_update_order == "family":
        target._update_ion_channel_families(target.V.value)
    elif target.ion_channel_update_order == "integration":
        target._update_ion_channels_by_integration(target.V.value)
    else:
        raise ValueError(
            f"ion_channel_update_order must be 'family' or 'integration', got {target.ion_channel_update_order!r}."
        )


def _compact_constructor_parameters(runtime_cls, instances, *, device) -> dict[str, object]:
    parameters = {}
    for parameter in runtime_cls.parameters:
        values = []
        for _, layout, source_synapse, synapse_index, _ in instances:
            values.append(
                _source_instance_value(
                    getattr(source_synapse, parameter),
                    n_active=int(layout.n_active),
                    synapse_index=int(synapse_index),
                )
            )
        parameters[parameter] = _device_put_value(u.math.stack(tuple(values)), device)
    return parameters


def _source_instance_value(value, *, n_active: int, synapse_index: int):
    shape = getattr(value, "shape", ())
    if not shape:
        return value
    return value.reshape((-1, n_active))[0, synapse_index]


def _compile_arrival_schedule(
    arrival_step: np.ndarray,
    arrival_group: np.ndarray,
    arrival_instance: np.ndarray,
    arrival_weight: np.ndarray,
    *,
    n_steps: int,
) -> tuple[object, tuple[CompactArrivalBucket, ...]]:
    """Compile fixed endpoints into one direct injection per step/group."""
    if arrival_step.size == 0:
        return (
            jnp.zeros((n_steps,), dtype=jnp.int32),
            (),
        )
    if np.any(arrival_step < 0) or np.any(arrival_step >= n_steps):
        raise ValueError("DIF calibration arrival step is outside its horizon.")

    order = np.lexsort((arrival_instance, arrival_group, arrival_step))
    arrival_step = arrival_step[order]
    arrival_group = arrival_group[order]
    arrival_instance = arrival_instance[order]
    arrival_weight = arrival_weight[order]
    first = np.flatnonzero(
        np.concatenate(
            (
                np.ones((1,), dtype=np.bool_),
                (arrival_step[1:] != arrival_step[:-1])
                | (arrival_group[1:] != arrival_group[:-1])
                | (arrival_instance[1:] != arrival_instance[:-1]),
            )
        )
    )
    arrival_amplitude = np.add.reduceat(arrival_weight, first).astype(np.float64, copy=False)
    arrival_step = arrival_step[first]
    arrival_group = arrival_group[first]
    arrival_instance = arrival_instance[first]

    unique_step, step_begin = np.unique(arrival_step, return_index=True)
    step_end = np.append(step_begin[1:], arrival_step.size)
    step_bucket = np.zeros((n_steps,), dtype=np.int32)
    buckets = []
    for bucket_id, (step, begin, end) in enumerate(zip(unique_step, step_begin, step_end), start=1):
        injections = []
        group_slice = arrival_group[begin:end]
        unique_group, group_begin = np.unique(group_slice, return_index=True)
        group_end = np.append(group_begin[1:], group_slice.size)
        for group_index, local_begin, local_end in zip(
            unique_group.tolist(),
            group_begin.tolist(),
            group_end.tolist(),
        ):
            selected = slice(begin + local_begin, begin + local_end)
            injections.append(
                CompactArrivalInjection(
                    group_index=int(group_index),
                    instance_index=jnp.asarray(arrival_instance[selected], dtype=jnp.int32),
                    amplitude=jnp.asarray(
                        arrival_amplitude[selected],
                        dtype=jnp.asarray(0.0).dtype,
                    ),
                )
            )
        step_bucket[step] = bucket_id
        buckets.append(CompactArrivalBucket(injections=tuple(injections)))
    return (
        jnp.asarray(step_bucket, dtype=jnp.int32),
        tuple(buckets),
    )


def _device_put_value(value, device):
    if isinstance(value, u.Quantity):
        return u.Quantity(jax.device_put(value.mantissa, device), value.unit)
    return jax.device_put(value, device)
