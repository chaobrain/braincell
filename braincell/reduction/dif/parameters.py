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

"""Select the teacher threshold, derive AP/AHP coordinates, and assemble eta."""

from __future__ import annotations

import brainunit as u
import numpy as np

from braincell.reduction.dif.calibration_utils import _root_cv
from braincell.reduction.dif.parameters_utils import (
    PopulationSpikeAnalysis,
    SpikeCalibrationSpec,
    _canonical_recovery_sample,
    _capture_canonical_states,
    _derive_refractory_steps,
    _directed_voltage_grid,
    _rest_voltage_grid,
    _run_canonical_voltage,
    _scalar_quantity,
    _upward_crossings,
    extract_spike_template,
    fit_teacher,
)


def extract_parameters(
    cell,
    voltage,
    arrivals,
    plan,
    *,
    device,
    canonical_current_delay=20.0 * u.ms,
    canonical_current_duration=0.3 * u.ms,
    canonical_current_amplitude=6.125 * u.nA,
):
    """Extract one representative's complete parameters from its real workload."""
    # 1. Select the candidate threshold from complete workload excursions.
    dt_ms = float(plan.timing.dt_ms)
    fit = fit_teacher(voltage, dt_ms)
    spec = SpikeCalibrationSpec(
        threshold=fit.threshold_mV * u.mV,
        canonical_current_delay=canonical_current_delay,
        canonical_current_duration=canonical_current_duration,
        canonical_current_amplitude=canonical_current_amplitude,
    )
    # 2. Trigger the canonical AP and capture its complete post-spike states.
    voltage = np.asarray(voltage, dtype=np.float64)
    if voltage.ndim == 1:
        voltage = voltage[:, None]
    if voltage.ndim != 2 or voltage.shape[1] != 1:
        raise ValueError("Voltage-response calibration requires one neuron per population.")
    canonical = _canonical_analysis(
        cell,
        spec,
        dt=plan.timing.dt,
        dt_ms=float(plan.timing.dt_ms),
        time_grid_steps=plan.timing.time_grid_steps,
        device=device,
    )

    # 3. Match workload APs to their onsets and locate REST input arrivals.
    # True APs define the teacher. Failed threshold excursions remain part of
    # REST activity and are scored by parameter extraction, not rejected here.
    onset, _ = _upward_crossings(voltage[:, 0], canonical["threshold_mV"])
    actual, _ = _upward_crossings(voltage[:, 0], 0.0)
    indices = np.searchsorted(onset, actual, side="right") - 1
    spike_step = np.unique(onset[indices[indices >= 0]])
    if spike_step.size == 0:
        raise RuntimeError("The detailed noisy workload produced no teacher spike.")
    refractory_steps = int(canonical["refractory_steps"])
    if np.any(np.diff(spike_step) <= refractory_steps):
        raise RuntimeError("Teacher spikes overlap the canonical-AP-derived refractory time.")
    root = _root_cv(cell)
    rest_mV = float(np.asarray(cell.V.value[..., root].to_decimal(u.mV)).reshape(-1)[0])
    rest_grid = _rest_voltage_grid(
        voltage[:, 0],
        spike_step,
        arrivals,
        trough_age_steps=int(canonical["trough_age_steps"]),
        rest_mV=rest_mV,
        threshold_mV=float(canonical["threshold_mV"]),
        dt_ms=float(plan.timing.dt_ms),
    )
    analysis = PopulationSpikeAnalysis(
        teacher_threshold_mV=float(canonical["threshold_mV"]),
        refractory_steps=refractory_steps,
        rest_voltage_grid_mV=np.asarray(rest_grid, dtype=np.float64),
        canonical_voltage_mV=canonical["voltage_mV"],
        canonical_spike_step=int(canonical["spike_step"]),
        canonical_eta_voltage_mV=np.asarray(canonical["eta_voltage_mV"], dtype=np.float64),
        rebase_age_steps=int(canonical["rebase_age_steps"]),
        trough_age_steps=int(canonical["trough_age_steps"]),
        eta_recovery_age_steps=int(canonical["eta_recovery_age_steps"]),
        falling_voltage_grid_mV=canonical["falling_voltage_grid_mV"],
        rebase_state=canonical["rebase_state"],
        falling_states=canonical["falling_states"],
    )

    # 4. Join the measured AP onset with canonical AHP recovery into eta.
    template = extract_spike_template(
        fit,
        dt_ms=dt_ms,
        rest_mV=rest_mV,
        canonical_eta_voltage_mV=analysis.canonical_eta_voltage_mV,
        canonical_rebase_age_steps=analysis.rebase_age_steps,
        canonical_trough_age_steps=analysis.trough_age_steps,
        canonical_recovery_age_steps=analysis.eta_recovery_age_steps,
    )
    return analysis, template


def _canonical_analysis(
    source_cell,
    spec,
    *,
    dt,
    dt_ms,
    time_grid_steps,
    device,
):
    """Locate one autonomous AP, then capture its POST and rebase states."""
    # 1. Trigger one AP with the somatic pulse and locate its true onset.
    threshold = _scalar_quantity(spec.threshold, unit=u.mV, name="threshold")
    voltage, delay_ms, duration_ms = _run_canonical_voltage(
        source_cell,
        spec,
        dt=dt,
        dt_ms=dt_ms,
        device=device,
    )
    delivery_step, _ = _upward_crossings(voltage, 0.0)
    if delivery_step.size != 1:
        raise RuntimeError(
            f"Canonical IClamp must generate exactly one regenerative 0 mV crossing; got {delivery_step.size}."
        )
    delivery = int(delivery_step[0])
    spike_step, _ = _upward_crossings(voltage, threshold)
    # A short initiating pulse may cross the candidate threshold and recede
    # before the autonomous AP. Use the onset belonging to the real spike.
    spike = int(spike_step[spike_step <= delivery][-1])
    last_injected_step = int(np.ceil((delay_ms + duration_ms) / dt_ms)) - 1
    # The short pulse initiates the AP, but must be absent before the full
    # regenerative waveform crosses 0 mV.  The AP peak, repolarization, and
    # AHP used by eta are therefore autonomous rather than current-clamp tail.
    if last_injected_step >= delivery:
        raise RuntimeError(
            "Canonical IClamp remains active into the regenerative AP: "
            f"last_injected_step={last_injected_step}, "
            f"zero_crossing_step={delivery}."
        )

    # 2. Derive refractory time, the rebase boundary and complete AHP recovery.
    refractory_steps = _derive_refractory_steps(
        voltage,
        spike_step=spike,
        threshold_mV=threshold,
        dt_ms=dt_ms,
    )
    rebase_age = refractory_steps + 1
    rebase_sample = spike + refractory_steps
    if rebase_sample + 2 >= voltage.size:
        raise RuntimeError("Canonical trace ends before R+1.")
    rest_mV = float(np.asarray(source_cell.V.value[..., _root_cv(source_cell)].to_decimal(u.mV)).reshape(-1)[0])
    search = voltage[rebase_sample:]
    trough_sample = rebase_sample + int(np.argmin(search))
    eta_recovery_sample = _canonical_recovery_sample(
        voltage,
        trough_step=trough_sample,
        rest_mV=rest_mV,
    )

    # 3. Select falling-branch coordinates and capture complete detailed states.
    falling_grid, falling_capture = _directed_voltage_grid(
        voltage,
        begin_step=rebase_sample,
        end_step=trough_sample,
        spike_step=spike,
        time_grid_steps=time_grid_steps,
    )
    rebase_capture = spike + rebase_age
    capture = np.concatenate(
        (
            np.asarray([rebase_capture], dtype=np.int64),
            falling_capture,
        )
    )
    capture_steps, order = np.unique(capture, return_inverse=True)
    sorted_states = _capture_canonical_states(
        source_cell,
        spec,
        dt=dt,
        capture_steps=capture_steps,
        device=device,
    )
    states = tuple(sorted_states[index] for index in order)

    # 4. Retain the full recovery waveform independently of branch hand-back.
    # Hand-back only changes which input-response bank receives new events.
    # The already-created AP/AHP history continues along the complete isolated
    # IClamp waveform instead of being truncated at the branch boundary.
    eta = voltage[spike:] - rest_mV
    return {
        "threshold_mV": threshold,
        "refractory_steps": refractory_steps,
        "voltage_mV": voltage,
        "spike_step": spike,
        "eta_voltage_mV": eta,
        "rebase_age_steps": rebase_age,
        "trough_age_steps": trough_sample - spike,
        "eta_recovery_age_steps": eta_recovery_sample - spike,
        "falling_voltage_grid_mV": falling_grid,
        "rebase_state": states[0],
        "falling_states": tuple(states[1:]),
    }
