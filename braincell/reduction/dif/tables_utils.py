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

"""Match physical inputs and pack calibrated arrays for DIF execution."""

from __future__ import annotations

import hashlib
import json

import brainunit as u
import numpy as np

from braincell.reduction.dif.tables import _DIMENSIONS, DIFTable, ResponseSlot


def _coordinate_lut(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(grid, dtype=np.int32)
    age = np.arange(int(grid[-1]) + 1, dtype=np.int32)
    insertion = np.searchsorted(grid, age, side="left").astype(np.int32, copy=False)
    exact = age == grid[insertion]
    lower = np.where(exact, insertion, insertion - 1).astype(np.int32, copy=False)
    upper = insertion
    ratio = np.zeros(age.shape, dtype=np.float64)
    interpolate = ~exact
    ratio[interpolate] = (age[interpolate] - grid[lower[interpolate]]) / (
        grid[upper[interpolate]] - grid[lower[interpolate]]
    )
    return lower, upper, ratio


def build_table(archive, *, path=None):
    """Build the same runtime table from measured arrays or a saved archive."""
    from braincell.reduction.dif.parameters_utils import candidate_refractory_age_steps

    # Storage names remain stable; runtime names describe DIF quantities.
    names = {
        'rest_single': 'single_voltage_mV',
        'rest_single_ptr': 'single_voltage_ptr',
        'rest_pair': 'pair_voltage_mV',
        'rest_pair_ptr': 'pair_voltage_ptr',
        'post_single': 'post_single_voltage_mV',
        'post_single_ptr': 'post_single_voltage_ptr',
        'post_pair': 'post_pair_voltage_mV',
        'post_pair_ptr': 'post_pair_voltage_ptr',
        'rebase_single': 'post_rebase_single_voltage_mV',
        'rebase_single_ptr': 'post_rebase_single_voltage_ptr',
        'rebase_pair': 'post_rebase_pair_voltage_mV',
        'rebase_pair_ptr': 'post_rebase_pair_voltage_ptr',
        'rest_support': 'rest_single_support_steps',
        'post_support': 'post_single_support_steps',
        'rebase_support': 'rebase_single_support_steps',
        'history_support': 'history_support_steps',
        'rest_grid': 'rest_voltage_grid_mV',
        'rebase_single_initial': 'dif_rebase_single_initial_mV',
        'rebase_single_initial_ptr': 'dif_rebase_single_initial_ptr',
        'rebase_pair_initial': 'dif_rebase_pair_initial_mV',
        'rebase_pair_initial_ptr': 'dif_rebase_pair_initial_ptr',
        'reversal': 'dif_reversal_mV',
        'query_support': 'dif_query_support_steps',
    }
    values = {name: np.asarray(archive[key]) for name, key in names.items()}
    slots = tuple(
        ResponseSlot(str(signature), int(instance), float(magnitude), () if trigger else tuple(map(float, dims)))
        for signature, instance, magnitude, trigger, dims in zip(
            archive['slot_signature'],
            archive['slot_instance'],
            archive['slot_magnitude'],
            archive['slot_is_trigger'],
            archive['slot_dimensions'],
        )
    )
    location_count = len(slots)
    pairs = np.asarray(archive['pair_slots'], dtype=np.int64)
    pair_keys = pairs[:, 0] * location_count + pairs[:, 1]
    reverse = np.searchsorted(pair_keys, pairs[:, 1] * location_count + pairs[:, 0])
    axes = np.asarray(archive['dif_axes_steps'], dtype=np.int64)
    lower, upper, ratio = zip(*(_coordinate_lut(axis) for axis in axes))
    state_begin, state_end = map(int, archive['post_state_ptr'][:2])
    eta_begin, eta_end = map(int, archive['eta_ptr'][:2])
    eta = np.asarray(archive['eta_voltage_mV'][eta_begin:eta_end])
    rebase_age = int(archive['rebase_age_steps'].item())
    rest_voltage = float(archive['V_rest_mV'].item())
    threshold = float(archive['teacher_spike_threshold_mV'].item())
    response_steps = max(
        int(np.max(np.diff(values['rest_single_ptr']))) // values['rest_grid'].size,
        int(np.max(np.diff(values['post_single_ptr']))) // (state_end - state_begin),
        *(
            int(np.max(np.diff(values[name])))
            for name in (
                'rest_pair_ptr',
                'post_pair_ptr',
                'rebase_single_ptr',
                'rebase_pair_ptr',
            )
        ),
        int(np.max(values['history_support'])) + 1,
        int(np.max(values['rebase_support'])) + 1,
    )
    return DIFTable(
        path=path,
        slots=slots,
        calibration_date=str(archive['calibration_date'].item()),
        dt_ms=float(archive['dt_ms'].item()),
        leak_per_ms=float(archive['dif_leak_per_ms'].item()),
        abort_voltage_mV=float(archive['abort_voltage_mV'].item()),
        pair_source_ptr=np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=location_count))],
        pair_sources=np.ascontiguousarray(pairs[:, 0]),
        pair_targets=np.ascontiguousarray(pairs[:, 1]),
        pair_reverse=reverse,
        post_state_grid=np.asarray(archive['post_state_voltage_mV'][state_begin:state_end]),
        time_steps=axes,
        time_lower=np.asarray(lower),
        time_upper=np.asarray(upper),
        time_ratio=np.asarray(ratio),
        time_inverse_width=1.0 / np.diff(axes).astype(np.float64),
        response_steps=response_steps,
        eta=eta,
        rest_voltage=rest_voltage,
        initial_voltage=float(archive['V_init_mV'].item()),
        spike_threshold=threshold,
        confirmation_voltage=confirmation_voltage_mV(threshold),
        rebase_age=rebase_age,
        candidate_refractory_age=candidate_refractory_age_steps(
            eta,
            threshold_mV=threshold,
            rest_mV=rest_voltage,
            minimum_rebase_age_steps=rebase_age,
        ),
        trough_age=int(archive['trough_age_steps'].item()),
        **values,
    )


def _parameter_key(value):
    if isinstance(value, u.Quantity):
        dimensions = tuple(float(value.unit.dim.get_dimension(name)) for name in _DIMENSIONS)
        return (np.asarray(value.mantissa) * value.unit.magnitude).tolist(), dimensions
    return np.asarray(value).tolist()


def synapse_signature(synapse):
    """Identify the calibrated placement, mechanism, and parameter values."""
    description = (
        synapse.branch_id,
        synapse.branch_x,
        synapse.synapse_type,
        tuple((name, _parameter_key(value)) for name, value in sorted(synapse.parameters.items())),
    )
    return hashlib.sha256(json.dumps(description, separators=(",", ":")).encode()).hexdigest()


def synapse_identities(synapses):
    """Match physical inputs across members without depending on other sites.

    Colocated independent mechanisms retain distinct occurrence numbers.
    A different site's presence, order, or user-facing name cannot shift them.
    """
    occurrences, identities = {}, {}
    for synapse in synapses:
        signature = synapse_signature(synapse)
        key = (synapse.population_index, signature)
        instance = occurrences.get(key, 0)
        occurrences[key] = instance + 1
        identities[synapse.id] = (signature, instance)
    return identities


def confirmation_voltage_mV(onset_mV):
    """Use the same early-confirmation level in calibration and execution."""
    return onset_mV + 1.5
