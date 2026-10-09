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

"""Fit spike waveforms and capture the detailed states used by parameter extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dif.calibration_utils import _root_cv, _root_cv_and_point
from braincell.reduction.dif.responses_utils import _make_calibration_cell

FALSE_EMISSION_LIMIT = 0.05
THRESHOLD_RESOLUTION_MV = 0.05


@dataclass(frozen=True)
class SpikeCalibrationSpec:
    """Teacher threshold and clean-IClamp coordinates for one population."""

    threshold: object
    canonical_current_delay: object
    canonical_current_duration: object
    canonical_current_amplitude: object


@dataclass(frozen=True)
class TeacherFit:
    threshold_mV: float
    abort_voltage_mV: float
    mean_time_ms: np.ndarray
    mean_voltage_mV: np.ndarray


def zero_crossings(voltage, dt_ms):
    """Return true AP left samples and their interpolated 0 mV times."""
    left = np.flatnonzero((voltage[:-1] < 0.0) & (voltage[1:] >= 0.0))
    fraction = -voltage[left] / (voltage[left + 1] - voltage[left])
    return left, (left + fraction) * dt_ms


def mean_spike_waveform(voltage, dt_ms, crossings, *, pre_ms=20.0, post_ms=30.0, previous_ms=20.0, next_ms=25.0):
    """Align isolated real APs at 0 mV and average their voltage samples."""
    relative = np.arange(-pre_ms, post_ms + 0.5 * dt_ms, dt_ms)
    total = np.zeros(relative.size, dtype=np.float64)
    count = 0
    for index, crossing in enumerate(crossings):
        previous = crossings[index - 1] if index else -np.inf
        following = crossings[index + 1] if index + 1 < crossings.size else np.inf
        if crossing - previous < previous_ms or following - crossing < next_ms:
            continue
        samples = (crossing + relative) / dt_ms
        if samples[0] < 0.0 or samples[-1] > voltage.size - 1:
            continue
        left = np.floor(samples).astype(np.int64)
        right = np.minimum(left + 1, voltage.size - 1)
        total += voltage[left] + (samples - left) * (voltage[right] - voltage[left])
        count += 1
    if count < 2:
        raise ValueError(
            "DIF parameter extraction needs at least two isolated real spikes in the "
            "declared workload. Increase the calibration workload duration or revise the network inputs."
        )
    return relative, total / count


def crossing_time(time_ms, voltage, left, level):
    fraction = (level - voltage[left]) / (voltage[left + 1] - voltage[left])
    return float(time_ms[left] + fraction * (time_ms[left + 1] - time_ms[left]))


def _scalar(value, *, name: str, dtype=float):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value.")
    result = dtype(array.reshape(-1)[0])
    if dtype is float and not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _falling_threshold_crossing_steps(
    eta_voltage_mV,
    *,
    threshold_mV: float,
    rest_mV: float,
) -> tuple[int, float]:
    """Return the right sample and interpolated age of AP repolarization."""
    eta = np.asarray(eta_voltage_mV, dtype=np.float64).reshape(-1)
    if eta.size < 2 or not np.all(np.isfinite(eta)):
        raise ValueError("DIF eta must be one finite vector with two samples.")
    if not np.isfinite(threshold_mV) or not np.isfinite(rest_mV):
        raise ValueError("DIF eta voltage coordinates must be finite.")
    voltage = float(rest_mV) + eta
    peak = int(np.argmax(voltage))
    falling = np.flatnonzero((voltage[peak:-1] > float(threshold_mV)) & (voltage[peak + 1 :] <= float(threshold_mV)))
    if falling.size == 0:
        raise ValueError("DIF eta has no descending candidate-threshold crossing after its AP peak.")
    left = peak + int(falling[0])
    delta = float(voltage[left + 1] - voltage[left])
    fraction = (float(threshold_mV) - float(voltage[left])) / delta
    return left + 1, float(left + fraction)


def candidate_refractory_age_steps(
    eta_voltage_mV,
    *,
    threshold_mV: float,
    rest_mV: float,
    minimum_rebase_age_steps: int = 0,
) -> int:
    """First spike age at which a new upward candidate may be accepted.

    The AP is refractory through the right sample of its descending threshold
    crossing.  Response-table handoff is an independent lower bound: a new
    spike cannot be accepted while pre-handoff arrivals are still hidden.
    """
    minimum_rebase = int(minimum_rebase_age_steps)
    if minimum_rebase < 0:
        raise ValueError("minimum_rebase_age_steps must be non-negative.")
    falling_right, _ = _falling_threshold_crossing_steps(
        eta_voltage_mV,
        threshold_mV=float(threshold_mV),
        rest_mV=float(rest_mV),
    )
    return max(minimum_rebase, falling_right + 1)


@dataclass(frozen=True)
class SpikeTemplate:
    """Small spike-only overlay for an immutable voltage-response table."""

    threshold_mV: float
    abort_voltage_mV: float
    dt_ms: float
    rebase_age_steps: int
    trough_age_steps: int
    eta_recovery_age_steps: int
    eta_voltage_mV: np.ndarray
    continuous_tref_ms: float
    response_table_rebase_age_steps: int

    def __post_init__(self):
        eta = np.asarray(self.eta_voltage_mV, dtype=np.float64)
        if eta.ndim != 1 or eta.size < 2 or not np.all(np.isfinite(eta)):
            raise ValueError("DIF spike-template eta must be one finite vector.")
        if not np.isfinite(self.threshold_mV) or not np.isfinite(self.dt_ms):
            raise ValueError("DIF spike-template coordinates must be finite.")
        if self.dt_ms <= 0.0:
            raise ValueError("DIF spike-template dt must be positive.")
        if not 0 < self.rebase_age_steps < self.trough_age_steps:
            raise ValueError("DIF spike-template rebase/trough ordering is invalid.")
        if not self.trough_age_steps < self.eta_recovery_age_steps < eta.size:
            raise ValueError("DIF spike-template trough/recovery ordering is invalid.")
        if not np.isfinite(self.continuous_tref_ms) or self.continuous_tref_ms <= 0:
            raise ValueError("DIF spike-template refractory time must be positive.")
        join_jump_mV = float(eta[self.rebase_age_steps] - eta[self.rebase_age_steps - 1])
        if join_jump_mV > 1.0e-9:
            raise ValueError(
                "DIF spike-template response handoff rises by "
                f"{join_jump_mV:.12g} mV; the noisy head and canonical tail "
                "must meet on the same descending phase."
            )
        # eta[0] is the candidate-threshold voltage relative to rest.  A
        # second upward crossing would make the one-spike overlay trigger a
        # new spike by itself, which is always a construction error.
        repeated_crossing = np.flatnonzero((eta[:-1] < eta[0]) & (eta[1:] >= eta[0]))
        if repeated_crossing.size:
            raise ValueError("DIF spike-template eta contains a post-spike upward candidate-threshold crossing.")
        candidate_refractory_age_steps(
            eta,
            threshold_mV=float(self.threshold_mV),
            rest_mV=float(self.threshold_mV - eta[0]),
            minimum_rebase_age_steps=int(self.rebase_age_steps),
        )
        object.__setattr__(self, "eta_voltage_mV", eta)

    def apply(self, table):
        """Apply the candidate threshold and waveform, sharing response arrays."""
        table_dt = _scalar(table.dt_ms, name="table.dt_ms")
        table_rebase = _scalar(table.rebase_age_steps, name="table.rebase_age_steps", dtype=int)
        if not np.isclose(self.dt_ms, table_dt, rtol=0.0, atol=1.0e-12):
            raise ValueError("DIF spike template and response table dt differ.")
        if self.response_table_rebase_age_steps != table_rebase:
            raise ValueError("DIF spike template was extracted for another response-table R+1 coordinate.")
        return replace(
            table,
            teacher_spike_threshold_mV=np.asarray([self.threshold_mV], dtype=np.float64),
            abort_voltage_mV=np.asarray([self.abort_voltage_mV], dtype=np.float64),
            rebase_age_steps=np.asarray([self.rebase_age_steps], dtype=np.int32),
            trough_age_steps=np.asarray([self.trough_age_steps], dtype=np.int32),
            eta_recovery_age_steps=np.asarray([self.eta_recovery_age_steps], dtype=np.int32),
            eta_ptr=np.asarray([0, self.eta_voltage_mV.size], dtype=np.int64),
            eta_voltage_mV=self.eta_voltage_mV,
        )


def extract_spike_template(
    fit,
    *,
    dt_ms,
    rest_mV,
    canonical_eta_voltage_mV,
    canonical_rebase_age_steps,
    canonical_trough_age_steps,
    canonical_recovery_age_steps,
):
    """Join the selected teacher mean AP to the captured canonical AHP."""
    threshold_mV = fit.threshold_mV
    relative_ms, averaged = fit.mean_time_ms, fit.mean_voltage_mV
    search_steps = int(np.ceil(30.0 / dt_ms))
    old_eta = np.asarray(canonical_eta_voltage_mV, dtype=np.float64).reshape(-1)
    old_rebase, old_trough, old_recovery = (
        canonical_rebase_age_steps,
        canonical_trough_age_steps,
        canonical_recovery_age_steps,
    )
    if not 0 < old_rebase < old_trough < old_recovery < old_eta.size:
        raise ValueError("Response table has invalid canonical eta coordinates.")

    zero_index = int(np.argmin(np.abs(relative_ms)))
    rising = np.flatnonzero((averaged[:zero_index] < threshold_mV) & (averaged[1 : zero_index + 1] >= threshold_mV))
    if rising.size == 0:
        raise RuntimeError("Noisy mean waveform has no teacher-threshold onset.")
    rising_left = int(rising[-1])
    rising_ms = crossing_time(relative_ms, averaged, rising_left, threshold_mV)
    peak_index = zero_index + int(np.argmax(averaged[zero_index : zero_index + search_steps]))
    tail_entry_mV = float(rest_mV + old_eta[old_rebase])
    tail_match = np.flatnonzero(
        (averaged[peak_index:-1] > tail_entry_mV) & (averaged[peak_index + 1 :] <= tail_entry_mV)
    )
    if tail_match.size == 0:
        raise RuntimeError("Noisy mean waveform never reaches the response-table R+1 voltage on its descending phase.")
    tail_match_left = peak_index + int(tail_match[0])
    tail_match_ms = crossing_time(relative_ms, averaged, tail_match_left, tail_entry_mV)
    tail_match_age_steps = float((tail_match_ms - rising_ms) / dt_ms)
    new_rebase = int(np.ceil(tail_match_age_steps - 1.0e-12))
    if new_rebase <= 0:
        raise RuntimeError("Noisy response handoff does not follow spike onset.")

    # Ages before the response handoff come from the zero-aligned noisy mean.
    # The first canonical-tail sample is exactly old R+1.  Taking the ceiling
    # of the continuous match age leaves the preceding noisy sample on the
    # more-depolarized side, so the discrete join cannot jump upward.
    head_age_ms = np.arange(new_rebase, dtype=np.float64) * dt_ms
    head_voltage = np.interp(rising_ms + head_age_ms, relative_ms, averaged)
    head_voltage[0] = threshold_mV
    join_jump_mV = float(tail_entry_mV - head_voltage[-1])
    if join_jump_mV > 1.0e-9:
        raise RuntimeError(f"Descending phase match produced an upward response handoff: {join_jump_mV:.12g} mV.")
    eta = np.concatenate((head_voltage - rest_mV, old_eta[old_rebase:]), axis=0)
    _, falling_age_steps = _falling_threshold_crossing_steps(
        eta,
        threshold_mV=float(threshold_mV),
        rest_mV=float(rest_mV),
    )
    continuous_tref_ms = float(falling_age_steps * dt_ms)
    shift = new_rebase - old_rebase
    return SpikeTemplate(
        threshold_mV=float(threshold_mV),
        abort_voltage_mV=float(fit.abort_voltage_mV),
        dt_ms=float(dt_ms),
        rebase_age_steps=int(new_rebase),
        trough_age_steps=int(old_trough + shift),
        eta_recovery_age_steps=int(old_recovery + shift),
        eta_voltage_mV=eta,
        continuous_tref_ms=continuous_tref_ms,
        response_table_rebase_age_steps=int(old_rebase),
    )


def _pack_analysis_metadata(analysis):
    """Preserve the archive's single-class axis for spike coordinates."""
    return {
        "teacher_spike_threshold_mV": np.asarray([analysis.teacher_threshold_mV], dtype=np.float64),
        "abort_voltage_mV": np.asarray([analysis.teacher_threshold_mV], dtype=np.float64),
        "rebase_age_steps": np.asarray([analysis.rebase_age_steps], dtype=np.int32),
        "trough_age_steps": np.asarray([analysis.trough_age_steps], dtype=np.int32),
        "eta_recovery_age_steps": np.asarray([analysis.eta_recovery_age_steps], dtype=np.int32),
        "eta_ptr": np.asarray([0, analysis.canonical_eta_voltage_mV.size], dtype=np.int32),
        "eta_voltage_mV": np.asarray(analysis.canonical_eta_voltage_mV, dtype=np.float64),
    }


def fit_teacher(voltage_mV, dt_ms):
    """Select the lowest onset in the first accepted band below 0 mV.

    Scan downward from 0 mV in 0.05 mV steps toward the trace's minimum.
    The mean zero-aligned AP supplies each threshold's delay. Once an
    accepted band starts, stop at the first rejected threshold or missing
    mean AP onset; retain the last accepted threshold without searching
    for lower, disconnected accepted bands.

    A complete excursion above the threshold is a false emission if it has
    no real AP and lasts at least the corresponding mean AP delay. Its
    fraction is false emissions / (true APs + false emissions). Candidate
    intervals must represent every real AP separately and end in the trace.
    Runtime cancellation uses a 1 mV margin below the candidate threshold.
    """
    voltage = np.asarray(voltage_mV, dtype=np.float64).reshape(-1)
    if not np.isfinite(voltage).all():
        raise ValueError("DIF teacher voltage contains nonfinite samples.")
    _, reference = zero_crossings(voltage, dt_ms)
    relative, average = mean_spike_waveform(voltage, dt_ms, reference)
    zero = int(np.argmin(np.abs(relative)))
    lowest = int(np.ceil(np.min(voltage) / THRESHOLD_RESOLUTION_MV))
    previous, following = voltage[:-1], voltage[1:]
    best = None
    for tick in range(0, lowest - 1, -1):
        threshold = round(tick * THRESHOLD_RESOLUTION_MV, 10)
        rising = np.flatnonzero((average[:zero] < threshold) & (average[1 : zero + 1] >= threshold))
        if not rising.size:
            if best is not None:
                break
            continue
        delay = -crossing_time(relative, average, int(rising[-1]), threshold)
        up = np.flatnonzero((previous < threshold) & (following >= threshold))
        down = np.flatnonzero((previous >= threshold) & (following < threshold))
        starts = (up + (threshold - previous[up]) / (following[up] - previous[up])) * dt_ms
        ends = (down + (threshold - previous[down]) / (following[down] - previous[down])) * dt_ms
        next_down = np.searchsorted(ends, starts, side="right")
        complete = next_down < ends.size
        unfinished = bool(np.any(~complete))
        starts, ends = starts[complete], ends[next_down[complete]]
        spike_counts = np.searchsorted(reference, ends, side="right") - np.searchsorted(reference, starts)
        true_count = int(np.count_nonzero(spike_counts))
        false_widths = (ends - starts)[spike_counts == 0]
        false_count = int(np.count_nonzero(false_widths >= delay))
        total = true_count + false_count
        fraction = false_count / total if total else None
        if true_count == reference.size and not unfinished and fraction < FALSE_EMISSION_LIMIT:
            best = threshold
        elif best is not None:
            break
    if best is None:
        raise ValueError(
            "The detailed teacher has no complete onset with fewer than 5% false emissions. "
            "Increase the calibration workload duration or revise the network inputs."
        )
    return TeacherFit(
        threshold_mV=best,
        abort_voltage_mV=best - 1.0,
        mean_time_ms=relative,
        mean_voltage_mV=average,
    )


VOLTAGE_GRID_FINE_INTERVAL_MV = 1.0
VOLTAGE_GRID_FINE_NODE_COUNT = 5
VOLTAGE_GRID_ABOVE_ONSET_NODE_COUNT = 1
VOLTAGE_GRID_COARSE_INTERVAL_MV = 2.0
VOLTAGE_GRID_INTERVAL_MV = 2.0
VOLTAGE_GRID_WARMUP_MS = 20.0
CANONICAL_PROBE_MS = 200.0
CANONICAL_RECOVERY_FRACTION = 0.05


@dataclass(frozen=True)
class CapturedCellState:
    """One complete detailed state captured before a runtime transition."""

    population_state: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class PopulationSpikeAnalysis:
    """Workload teacher coordinates plus one clean canonical AP/AHP."""

    teacher_threshold_mV: float
    refractory_steps: int
    rest_voltage_grid_mV: np.ndarray
    canonical_voltage_mV: np.ndarray
    canonical_spike_step: int
    canonical_eta_voltage_mV: np.ndarray
    rebase_age_steps: int
    trough_age_steps: int
    eta_recovery_age_steps: int
    falling_voltage_grid_mV: np.ndarray
    rebase_state: CapturedCellState | None = None
    falling_states: tuple[CapturedCellState, ...] = ()


def _scalar_quantity(value, *, unit, name: str) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"SpikeCalibrationSpec.{name} must carry units.")
    array = np.asarray(value.to_decimal(unit), dtype=np.float64)
    if array.shape not in ((), (1,)) or not np.isfinite(array.reshape(())):
        raise ValueError(f"SpikeCalibrationSpec.{name} must be one finite scalar.")
    return float(array.reshape(()))


def _capture_cell_state(cell, neuron: int = 0) -> CapturedCellState:
    values = []
    for path, state in cell.states().items():
        value = state.value
        if getattr(value, "shape", ()) and int(value.shape[0]) > neuron:
            values.append((str(path), value[neuron]))
    return CapturedCellState(population_state=tuple(values))


def _upward_crossings(voltage_mV: np.ndarray, threshold_mV: float):
    voltage = np.asarray(voltage_mV, dtype=np.float64)
    if voltage.ndim == 1:
        voltage = voltage[:, None]
    step, neuron = np.nonzero((voltage[:-1] < threshold_mV) & (voltage[1:] >= threshold_mV))
    return step.astype(np.int64) + 1, neuron.astype(np.int32)


def _canonical_current(cell, *, amplitude_nA, delay_ms, duration_ms):
    _, root_point = _root_cv_and_point(cell)

    def current(point_voltage):
        t = brainstate.environ.get("t", 0.0 * u.ms)
        t_ms = t.to_decimal(u.ms) if hasattr(t, "to_decimal") else t
        active = (t_ms >= delay_ms) & (t_ms < delay_ms + duration_ms)
        total = u.math.where(active, amplitude_nA, 0.0) * u.nA
        density = (total / cell.runtime.point_area[root_point]).in_unit(u.nA / u.cm**2)
        output = u.Quantity(
            jnp.zeros(point_voltage.shape, dtype=point_voltage.dtype),
            u.nA / u.cm**2,
        )
        return output.at[..., root_point].set(density)

    return current


def _new_canonical_cell(source_cell, spec: SpikeCalibrationSpec, *, device):
    compact_ref = [None]
    cell = _make_calibration_cell(source_cell, 1, compact_ref, device=device)
    delay_ms = _scalar_quantity(spec.canonical_current_delay, unit=u.ms, name="canonical_current_delay")
    duration_ms = _scalar_quantity(
        spec.canonical_current_duration,
        unit=u.ms,
        name="canonical_current_duration",
    )
    amplitude_nA = _scalar_quantity(
        spec.canonical_current_amplitude,
        unit=u.nA,
        name="canonical_current_amplitude",
    )
    if delay_ms < 0.0 or duration_ms <= 0.0 or amplitude_nA <= 0.0:
        raise ValueError("Canonical IClamp delay/amplitude/duration are invalid.")
    cell.add_current_input(
        "reduce_canonical_soma_iclamp",
        _canonical_current(
            cell,
            amplitude_nA=amplitude_nA,
            delay_ms=delay_ms,
            duration_ms=duration_ms,
        ),
    )
    return cell, delay_ms, duration_ms


def _run_canonical_voltage(
    source_cell,
    spec,
    *,
    dt,
    dt_ms,
    device,
):
    cell, delay_ms, duration_ms = _new_canonical_cell(source_cell, spec, device=device)
    steps = int(np.rint(CANONICAL_PROBE_MS / dt_ms))
    times = u.math.arange(steps) * dt
    root = _root_cv(cell)
    with brainstate.environ.context(dt=dt):

        def step(t):
            with brainstate.environ.context(t=t):
                cell._update_dynamics()
                return cell.V.value[..., root].to_decimal(u.mV)

        result = jax.block_until_ready(brainstate.transform.for_loop(step, times))
    voltage = np.asarray(jax.device_get(result), dtype=np.float64).reshape(steps)
    return voltage, delay_ms, duration_ms


def _capture_canonical_states(
    source_cell,
    spec,
    *,
    dt,
    capture_steps: np.ndarray,
    device,
):
    cell, _, _ = _new_canonical_cell(source_cell, spec, device=device)
    targets = np.asarray(capture_steps, dtype=np.int64)
    if np.any(targets <= 0) or np.any(np.diff(targets) < 0):
        raise ValueError("Canonical capture steps must be positive and sorted.")
    times = u.math.arange(int(targets[-1])) * dt

    def step(t):
        with brainstate.environ.context(t=t):
            cell._update_dynamics()
        return jnp.asarray(0, dtype=jnp.int32)

    captured = []
    cursor = 0
    with brainstate.environ.context(dt=dt):
        for target in targets.tolist():
            if target > cursor:
                jax.block_until_ready(brainstate.transform.for_loop(step, times[cursor:target]))
            captured.append(_capture_cell_state(cell))
            cursor = target
    return tuple(captured)


def _directed_voltage_grid(
    voltage_mV: np.ndarray,
    *,
    begin_step: int,
    end_step: int,
    spike_step: int,
    time_grid_steps: np.ndarray,
):
    segment = np.asarray(voltage_mV[begin_step : end_step + 1], dtype=np.float64)
    if segment.size < 2 or not np.all(np.isfinite(segment)):
        raise RuntimeError("Canonical post-spike branch is incomplete.")
    lower = VOLTAGE_GRID_INTERVAL_MV * np.ceil(float(np.min(segment)) / VOLTAGE_GRID_INTERVAL_MV)
    upper = VOLTAGE_GRID_INTERVAL_MV * np.floor(float(np.max(segment)) / VOLTAGE_GRID_INTERVAL_MV)
    levels = np.arange(
        lower,
        upper + VOLTAGE_GRID_INTERVAL_MV,
        VOLTAGE_GRID_INTERVAL_MV,
        dtype=np.float64,
    )
    kept = []
    capture = []
    for level in levels.tolist():
        crossing = np.flatnonzero((segment[:-1] > level) & (segment[1:] <= level))
        if crossing.size:
            sample = begin_step + int(crossing[0]) + 1
            kept.append(float(voltage_mV[sample]))
            capture.append(sample + 1)

    # Add the response library's physical-time nodes within this descending
    # segment, storing the actual waveform voltage and full captured state.
    # The later AHP recovery segment is not part of this POST voltage grid.
    for age_steps in np.asarray(time_grid_steps, dtype=np.int64).tolist():
        sample = spike_step + age_steps
        if begin_step < sample < end_step:
            kept.append(float(voltage_mV[sample]))
            capture.append(sample + 1)

    # Include the actual rebase and trough endpoints even when neither lies
    # on a 2 mV level.
    kept.extend((float(segment[0]), float(segment[-1])))
    capture.extend((begin_step + 1, end_step + 1))
    order = np.argsort(np.asarray(kept), kind="stable")
    ordered_voltage = np.asarray(kept, dtype=np.float64)[order]
    ordered_capture = np.asarray(capture, dtype=np.int64)[order]
    unique = np.concatenate(
        (
            np.asarray([True]),
            np.diff(ordered_voltage) > 1.0e-4,
        )
    )
    ordered_voltage = ordered_voltage[unique]
    ordered_capture = ordered_capture[unique]
    if ordered_voltage.size < 2:
        raise RuntimeError("Canonical FALLING has no measurable voltage range.")
    return ordered_voltage, ordered_capture


def _canonical_recovery_sample(
    voltage_mV: np.ndarray,
    *,
    trough_step: int,
    rest_mV: float,
) -> int:
    """Find the IClamp AHP's first stable return to its resting neighbourhood."""
    voltage = np.asarray(voltage_mV, dtype=np.float64)
    trough_mV = float(voltage[trough_step])
    excursion_mV = abs(trough_mV - float(rest_mV))
    if excursion_mV <= 0.0:
        raise RuntimeError("Canonical IClamp trace has no measurable AHP.")
    tolerance_mV = CANONICAL_RECOVERY_FRACTION * excursion_mV
    deviation = np.abs(voltage - float(rest_mV))
    maximum_remaining_deviation = np.maximum.accumulate(deviation[::-1])[::-1]
    candidate = np.flatnonzero((np.arange(voltage.size) >= trough_step) & (maximum_remaining_deviation <= tolerance_mV))
    if candidate.size == 0:
        raise RuntimeError(
            "Canonical IClamp AHP does not settle inside its 5% recovery band before the canonical trace ends."
        )
    sample = int(candidate[0])
    if sample <= trough_step:
        raise RuntimeError("Canonical IClamp recovery precedes its AHP trough.")
    return sample


def _crossing_age_ms(
    voltage_mV: np.ndarray,
    *,
    left_step: int,
    right_step: int,
    level_mV: float,
    spike_step: int,
    dt_ms: float,
) -> float:
    left = float(voltage_mV[left_step])
    right = float(voltage_mV[right_step])
    delta = right - left
    if delta == 0.0:
        raise RuntimeError("Canonical crossing has zero voltage slope.")
    fraction = (float(level_mV) - left) / delta
    if not 0.0 <= fraction <= 1.0:
        raise RuntimeError("Canonical crossing interpolation left its timestep.")
    return float((left_step + fraction - spike_step) * dt_ms)


def _derive_refractory_steps(
    voltage_mV: np.ndarray,
    *,
    spike_step: int,
    threshold_mV: float,
    dt_ms: float,
) -> int:
    """Derive Tref from the same isolated AP used for canonical eta."""
    voltage = np.asarray(voltage_mV, dtype=np.float64)
    peak_step = spike_step + int(np.argmax(voltage[spike_step:]))
    peak_mV = float(voltage[peak_step])
    if peak_step <= 0 or peak_mV <= threshold_mV:
        raise RuntimeError("Canonical AP has no peak above the teacher threshold.")

    repolarization = np.flatnonzero((voltage[peak_step:-1] > threshold_mV) & (voltage[peak_step + 1 :] <= threshold_mV))
    if repolarization.size == 0:
        raise RuntimeError("Canonical AP does not contain a complete repolarization crossing.")

    repolarization_left = peak_step + int(repolarization[0])
    upward_threshold_age_ms = _crossing_age_ms(
        voltage,
        left_step=spike_step - 1,
        right_step=spike_step,
        level_mV=threshold_mV,
        spike_step=spike_step,
        dt_ms=dt_ms,
    )
    falling_threshold_age_ms = _crossing_age_ms(
        voltage,
        left_step=repolarization_left,
        right_step=repolarization_left + 1,
        level_mV=threshold_mV,
        spike_step=spike_step,
        dt_ms=dt_ms,
    )
    # Tref is the physical interval between equal-voltage crossings: the
    # upward threshold crossing and the first post-peak downward crossing of
    # exactly the same threshold.  The runtime event is quantized to the first
    # sample after the upward crossing, hence ceil maps the continuous end to
    # the first safe discrete rebase step.
    continuous_tref_ms = float(falling_threshold_age_ms - upward_threshold_age_ms)
    refractory_steps = int(np.ceil(continuous_tref_ms / dt_ms - 1.0e-12))
    if refractory_steps <= 0:
        raise RuntimeError("Canonical AP produced a non-positive refractory period.")
    return refractory_steps


def _workload_rest_pre_step_mask(
    voltage_mV,
    spike_step,
    *,
    trough_age_steps,
):
    voltage = np.asarray(voltage_mV, dtype=np.float64)
    if voltage.ndim != 1 or not np.all(np.isfinite(voltage)):
        raise ValueError("DIF teacher phase mask requires one finite voltage trace.")
    count = int(voltage.size)
    teacher = np.zeros((count,), dtype=np.bool_)
    teacher[np.asarray(spike_step, dtype=np.int64)] = True
    rest = np.empty((count,), dtype=np.bool_)
    post_active = False
    last_spike = -count
    for step in range(count):
        # This is the phase seen by arrivals at the beginning of ``step``.
        # Hand-back is decided only after the detailed teacher has completed
        # this step, so it can affect arrivals no earlier than ``step + 1``.
        rest[step] = not post_active
        if teacher[step]:
            # A newly emitted spike is the only boundary that re-expresses
            # old histories.  It also wins over any voltage-rise hand-back on
            # the same completed timestep.
            post_active = True
            last_spike = step
        elif (
            post_active and step - last_spike > int(trough_age_steps) and step > 0 and voltage[step] > voltage[step - 1]
        ):
            post_active = False
    return rest


def _rest_voltage_grid(
    voltage,
    spike_step,
    arrival_step,
    *,
    trough_age_steps,
    rest_mV,
    threshold_mV,
    dt_ms,
):
    arrival = np.unique(np.asarray(arrival_step, dtype=np.int64))
    warmup = int(np.ceil(VOLTAGE_GRID_WARMUP_MS / dt_ms - 1.0e-12))
    arrival = arrival[(arrival >= warmup) & (arrival < voltage.size)]
    rest = _workload_rest_pre_step_mask(
        voltage,
        spike_step,
        trough_age_steps=trough_age_steps,
    )
    arrival = arrival[rest[arrival]]
    if arrival.size == 0:
        raise RuntimeError("No accepted REST arrival remains for the voltage grid.")
    pre = np.where(arrival == 0, rest_mV, voltage[np.maximum(arrival - 1, 0)])
    # Keep the onset region at 1 mV spacing through onset + 1 mV, below
    # the onset + 1.5 mV confirmation level.
    highest = float(threshold_mV)
    descending = [highest - VOLTAGE_GRID_FINE_INTERVAL_MV * index for index in range(VOLTAGE_GRID_FINE_NODE_COUNT)]
    lower_target = float(np.min(pre)) - VOLTAGE_GRID_FINE_INTERVAL_MV
    while descending[-1] > lower_target:
        descending.append(descending[-1] - VOLTAGE_GRID_COARSE_INTERVAL_MV)
    above = [
        highest + VOLTAGE_GRID_FINE_INTERVAL_MV * index for index in range(1, VOLTAGE_GRID_ABOVE_ONSET_NODE_COUNT + 1)
    ]
    grid = np.asarray(descending[::-1] + above, dtype=np.float64)
    if grid.size < VOLTAGE_GRID_FINE_NODE_COUNT + VOLTAGE_GRID_ABOVE_ONSET_NODE_COUNT + 1:
        raise RuntimeError("REST voltage grid has too few nonuniform states.")
    return grid


def _normalize_voltage_grid(voltage_grid_mV, *, threshold_mV) -> np.ndarray:
    grid = np.asarray(voltage_grid_mV, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("DIF calibration voltage grid must contain at least two values.")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("DIF calibration voltage grid must be finite and increasing.")
    upper = float(threshold_mV) + VOLTAGE_GRID_ABOVE_ONSET_NODE_COUNT * VOLTAGE_GRID_FINE_INTERVAL_MV
    # Cached teacher analyses may still include the old onset + 2 mV node.
    grid = grid[grid <= upper + 1.0e-12]
    if not grid.size or not np.isclose(grid[-1], upper, rtol=0.0, atol=1.0e-12):
        raise ValueError("The REST preparation grid must end at onset + 1 mV.")
    fine_node_count = VOLTAGE_GRID_FINE_NODE_COUNT + VOLTAGE_GRID_ABOVE_ONSET_NODE_COUNT
    if grid.size < fine_node_count + 1:
        raise ValueError(
            f"DIF calibration voltage grid must contain {fine_node_count} fine nodes and at least one coarse node."
        )
    spacing = np.diff(grid)
    # Four intervals below onset, followed by one above it.
    fine_intervals = fine_node_count - 1
    if not np.allclose(spacing[-fine_intervals:], VOLTAGE_GRID_FINE_INTERVAL_MV, rtol=0.0, atol=1.0e-12):
        raise ValueError("The highest DIF voltage nodes must use 1 mV spacing.")
    if not np.allclose(spacing[:-fine_intervals], VOLTAGE_GRID_COARSE_INTERVAL_MV, rtol=0.0, atol=1.0e-12):
        raise ValueError("DIF voltage nodes below the fine region must use 2 mV spacing.")
    return grid
