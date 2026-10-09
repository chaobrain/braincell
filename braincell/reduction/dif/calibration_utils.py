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

"""Prepare calibration inputs, time coordinates and conductance fits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import brainunit as u
import jax
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from braincell.network.event import round_half_up_steps_host
from braincell.network.lowering import _expand_delay_steps
from braincell.reduction.dif.tables import ResponseSlot


@dataclass(frozen=True)
class CalibrationExecution:
    """Configure the GPUs and maximum condition batch per device.

    Parameters
    ----------
    device_ids : tuple of int, optional
        Logical GPU indices visible to JAX. Defaults to the first GPU.
    batch_per_device : int, optional
        Maximum number of simultaneous detailed conditions on each GPU.
    """

    device_ids: tuple[int, ...] = (0,)
    batch_per_device: int = 30000

    def __post_init__(self):
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("device_ids must contain distinct GPU indices.")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.device_ids):
            raise ValueError("GPU indices must be nonnegative integers.")
        if (
            isinstance(self.batch_per_device, bool)
            or not isinstance(self.batch_per_device, int)
            or self.batch_per_device < 1
        ):
            raise ValueError("batch_per_device must be a positive integer.")

    @property
    def batch_size(self):
        """Return the total condition capacity across selected devices."""
        return self.batch_per_device * len(self.device_ids)

    @property
    def devices(self):
        """Resolve the selected logical GPU indices on demand."""
        available = jax.devices("gpu")
        if max(self.device_ids) >= len(available):
            raise ValueError(f"GPU indices {self.device_ids} exceed the {len(available)} visible GPUs.")
        return tuple(available[index] for index in self.device_ids)


def connected_synapses(connections):
    """Collect only logical inputs exercised by a nonzero connection."""
    active = {}
    for connection in connections:
        ids = connection.synapse_id
        if connection.weight is not None:
            ids = ids[np.asarray(u.get_mantissa(connection.weight)) != 0]
        active.setdefault(connection.post_population, set()).update(map(int, ids))
    return active


_TIME_GRID_LIMIT_TENTHS = (2, 4, 12, 20, 40, 160, 240, 800)
_TIME_GRID_INCREMENT_TENTHS = (1, 2, 4, 8, 10, 20, 40, 80, 160)
SINGLE_PROBE_MS = 240.0
CLAMP_PREPARATION_MS = 180.0


def next_time_grid_step(step: int, dt_ms: float) -> int:
    """Advance one node using the shared physical-time spacing rule."""
    if not np.isfinite(dt_ms) or dt_ms <= 0 or step < 0:
        raise ValueError("DIF time-grid steps must be nonnegative and dt must be finite and positive.")
    time_tenths = int(np.floor(float(step) * float(dt_ms) * 10.0 + 1.0e-9))
    phase = int(np.searchsorted(_TIME_GRID_LIMIT_TENTHS, time_tenths, side="right"))
    increment_ms = _TIME_GRID_INCREMENT_TENTHS[phase] / 10.0
    increment = max(1, int(np.rint(increment_ms / float(dt_ms))))
    result = int(step) + increment
    if result > np.iinfo(np.int32).max:
        raise OverflowError("DIF response time exceeds the supported integer time axis.")
    return result


@dataclass(frozen=True)
class Timing:
    """Runtime clock and the current explicit library time axis."""

    dt: object
    dt_ms: float
    n_steps: int
    time_grid_steps: np.ndarray


@dataclass(frozen=True)
class CalibrationTiming:
    """Dai--Li voltage-clamp preparation and response clock."""

    clamp_steps: int
    probe_steps: int


@dataclass(frozen=True)
class PopulationCalibrationPlan:
    """One actual target/weight response slot per calibrated input."""

    input_layout_id: np.ndarray
    input_synapse_index: np.ndarray
    input_instance_index: np.ndarray
    input_cells: tuple
    input_event_weights: tuple


@dataclass(frozen=True)
class CalibrationPlan:
    timing: CalibrationTiming
    populations: tuple[PopulationCalibrationPlan, ...]


def time_quantity_to_scalar_ms(value, *, name: str) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"DIF {name} must carry time units.")
    array = np.asarray(value.to_decimal(u.ms), dtype=np.float64)
    if array.shape not in ((), (1,)):
        raise ValueError(f"DIF {name} must be scalar.")
    return float(array.reshape(()))


def _time_to_steps(value_ms: float, dt_ms: float, *, name: str) -> int:
    if not np.isfinite(value_ms) or not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("DIF time coordinates must be finite and dt must be positive.")
    raw = value_ms / dt_ms
    steps = int(np.rint(raw))
    if steps < 0 or not np.isclose(raw, steps, rtol=0.0, atol=1.0e-7):
        raise ValueError(f"DIF dt must place {name} on an integer timestep.")
    if steps > np.iinfo(np.int32).max:
        raise OverflowError("DIF time coordinate exceeds the supported integer time axis.")
    return steps


def build_time_grid_steps(
    *,
    dt_ms: float,
    maximum_ms: float,
    calibration_time_grid=None,
) -> np.ndarray:
    """Generate a nonuniform axis up to a measured cutoff, or validate an explicit axis."""
    maximum = _time_to_steps(maximum_ms, dt_ms, name="the DIF probe horizon")
    if calibration_time_grid is None:
        if maximum < 1:
            raise ValueError("DIF response time axis must include at least one timestep.")
        steps = [0]
        while steps[-1] < maximum:
            steps.append(min(maximum, next_time_grid_step(steps[-1], dt_ms)))
        return np.asarray(steps, dtype=np.int32)
    else:
        if not isinstance(calibration_time_grid, u.Quantity):
            raise TypeError("DIF calibration_time_grid must carry time units.")
        coordinates_ms = np.asarray(calibration_time_grid.to_decimal(u.ms), dtype=np.float64)
        if coordinates_ms.ndim != 1:
            raise ValueError("DIF calibration_time_grid must be one-dimensional.")
    if coordinates_ms.size < 2 or not np.all(np.isfinite(coordinates_ms)):
        raise ValueError("DIF calibration_time_grid must contain at least two finite nodes.")
    grid = np.asarray(
        [_time_to_steps(value, dt_ms, name=f"the {value:g} ms library node") for value in coordinates_ms.tolist()],
        dtype=np.int32,
    )
    if grid[0] != 0 or np.any(np.diff(grid) <= 0):
        raise RuntimeError("DIF library time nodes must start at zero and increase.")
    if int(grid[-1]) != maximum:
        raise ValueError(f"DIF calibration_time_grid must end at the {maximum_ms:g} ms probe horizon.")
    return grid


def explicit_time_coordinates(grid: np.ndarray, target: int) -> tuple[int, int, float]:
    """Locate an integer timestep on an explicit, potentially uneven grid."""
    insertion = int(np.searchsorted(grid, target, side="left"))
    if insertion < grid.size and target == int(grid[insertion]):
        return insertion, insertion, 0.0
    if insertion <= 0 or insertion >= grid.size:
        endpoint = 0 if insertion <= 0 else int(grid.size) - 1
        return endpoint, endpoint, 0.0
    lower = insertion - 1
    upper = insertion
    denominator = int(grid[upper]) - int(grid[lower])
    ratio = (target - int(grid[lower])) / denominator
    return lower, upper, ratio


def normalize_timing(*, dt, duration, calibration_time_grid=None) -> Timing:
    dt_ms = time_quantity_to_scalar_ms(dt, name="dt")
    duration_ms = time_quantity_to_scalar_ms(duration, name="duration")
    if dt_ms <= 0.0:
        raise ValueError("DIF dt must be > 0.")
    if duration_ms <= 0.0:
        raise ValueError("DIF duration must be > 0.")
    n_steps = int(np.ceil(duration_ms / dt_ms - 1.0e-12))
    return Timing(
        dt=dt,
        dt_ms=dt_ms,
        n_steps=n_steps,
        time_grid_steps=build_time_grid_steps(
            dt_ms=dt_ms,
            maximum_ms=SINGLE_PROBE_MS,
            calibration_time_grid=calibration_time_grid,
        ),
    )


def normalize_calibration_timing(*, clamp_duration, timing: Timing):
    clamp_ms = time_quantity_to_scalar_ms(clamp_duration, name="calibration clamp duration")
    if clamp_ms <= 0.0:
        raise ValueError("DIF calibration clamp duration must be > 0 ms.")
    clamp_steps = _time_to_steps(clamp_ms, timing.dt_ms, name="calibration clamp duration")
    probe_steps = _time_to_steps(
        SINGLE_PROBE_MS,
        timing.dt_ms,
        name=f"the {SINGLE_PROBE_MS:g} ms single probe",
    )
    return CalibrationTiming(
        clamp_steps=clamp_steps,
        probe_steps=probe_steps,
    )


@dataclass(frozen=True)
class CalibrationLayout:
    """Packed response identities and the detailed calibration clock."""

    population_names: tuple[str, ...]
    population_location_ptr: np.ndarray
    population_pair_ptr: np.ndarray
    pair_slots: np.ndarray
    slots: tuple[ResponseSlot, ...]
    timing: Timing
    start_time_ms: float


@dataclass(frozen=True)
class ConnectionRows:
    """Logical connections collected before any target-backend lowering."""

    post_population: str
    event_source: object
    source_index: np.ndarray
    synapse_id: np.ndarray
    weight: object
    delay: object


def collect_connections(populations):
    """Snapshot Cell-owned connection rows, including scheduled sources."""
    return tuple(
        ConnectionRows(
            name,
            connection.source,
            np.asarray(connection.source_index, dtype=np.int64),
            np.asarray(connection.synapse_id, dtype=np.int64),
            connection.weight,
            connection.delay,
        )
        for name, population in populations.items()
        if population.kind == "cell"
        for connection in population.cell.connections._call_views()
    )


def _arrival_steps(network, population, result, dt):
    """Recover inputs affecting the representative's voltage update."""
    cell = network.populations[population].cell
    dt_ms = float(dt.to_decimal(u.ms))
    live_events = {
        id(view.owner): result.events[name][port]
        for name, owner in network._cell_populations().items()
        for port, view in owner.event_outputs.items()
    }
    sources, arrivals = {}, []
    for connection in cell.connections._call_views():
        active = np.asarray(connection.synapse.population_index) == 0
        if connection.weight is not None:
            active &= np.asarray(u.get_mantissa(connection.weight)) != 0
        if not np.any(active):
            continue
        source = connection.source
        if id(source) not in sources:
            if source.is_scheduled:
                # Use the full schedule: an event before t=0 may arrive later.
                events = source.events
                ids = np.asarray(events.source_index)
                times = np.asarray(events.time.to_decimal(u.ms))
            else:
                events = live_events[id(source)]
                keep = np.asarray(events.count) > 0
                ids = np.asarray(events.source_id)[keep]
                times = np.asarray(events.time.to_decimal(u.ms))[keep]
            order = np.argsort(ids, kind="stable")
            sources[id(source)] = ids[order], times[order]
        ids, times = sources[id(source)]
        delays = np.broadcast_to(np.asarray(connection.delay.to_decimal(u.ms)), (len(connection),))
        if not source.is_scheduled:
            delay_steps = _expand_delay_steps(
                connection.delay,
                dt=dt,
                n_contact=len(connection),
                quantization="nearest",
            )
        for row in np.flatnonzero(active):
            first = np.searchsorted(ids, connection.source_index[row], side="left")
            last = np.searchsorted(ids, connection.source_index[row], side="right")
            event_times = times[first:last]
            if source.is_scheduled:
                steps = round_half_up_steps_host((event_times + delays[row]) / dt_ms)
            else:
                # Live zero-delay events are applied after dynamics and first
                # affect voltage on the following update, as in Network.run.
                steps = round_half_up_steps_host(event_times / dt_ms) + max(1, int(delay_steps[row]))
            arrivals.append(steps)
    steps = np.unique(np.concatenate(arrivals)) if arrivals else np.empty(0, dtype=np.int64)
    return steps[(steps >= 0) & (steps < len(result.time))].astype(np.int64)


TAIL_FRACTION = 0.03


def waveform(rise, decay):
    if not 0 < rise < decay:
        raise ValueError('Exp2 selection requires 0 < tau_rise < tau_decay.')
    peak_time = rise * decay / (decay - rise) * np.log(decay / rise)
    peak = np.exp(-peak_time / decay) - np.exp(-peak_time / rise)

    def evaluate(t, order=0):
        t = np.asarray(t)
        return ((-1 / decay) ** order * np.exp(-t / decay) - (-1 / rise) ** order * np.exp(-t / rise)) / peak

    return evaluate


def anchors(rise, decay, dt):
    # Keep the arithmetic and rounding of the adopted selection rule.
    peak_time = rise * decay / (decay - rise) * math.log(decay / rise)
    raw = lambda t: np.exp(-np.asarray(t) / decay) - np.exp(-np.asarray(t) / rise)
    peak = float(raw(peak_time))
    exact_end = brentq(
        lambda t: float(raw(t)) / peak - TAIL_FRACTION, peak_time, -decay * math.log(TAIL_FRACTION) + peak_time + decay
    )
    end_tick = math.ceil(exact_end / dt - 1e-12)
    nearby = {math.floor(peak_time / dt), math.ceil(peak_time / dt)}
    peak_tick = max(nearby, key=lambda tick: float(raw(tick * dt)))
    if peak_tick <= 0 or peak_tick >= end_tick:
        raise ValueError('The timestep does not resolve distinct start, peak and tail anchors.')
    return round(peak_tick * dt, 9), round(end_tick * dt, 9), peak_time, exact_end


def segment_nodes(a, b, intervals, evaluate, dt):
    dense = np.linspace(a, b, max(2, int(np.ceil((b - a) / (dt / 20))) + 1))
    mass = cumulative_trapezoid(np.sqrt(abs(evaluate(dense, 2))), dense, initial=0.0)
    points = np.interp(np.linspace(0.0, mass[-1], intervals + 1), mass, dense)
    ticks = np.rint(points / dt).astype(int)
    ticks[0], ticks[-1] = int(round(a / dt)), int(round(b / dt))
    return ticks if np.all(np.diff(ticks) > 0) else None


def phi1(k):
    k = np.asarray(k, dtype=np.float64)
    out = np.ones_like(k)
    small = np.abs(k) < 1e-8
    out[~small] = -np.expm1(-k[~small]) / k[~small]
    out[small] = 1.0 - 0.5 * k[small]
    return out


def replay(g, s, x0, lam, dt):
    """Exponential Euler replay of x' = -(lam + g) x + s, vectorised over rows."""
    N, L = g.shape
    x = np.empty((N, L))
    x[:, 0] = x0
    for n in range(L - 1):
        k = (lam + g[:, n]) * dt
        x[:, n + 1] = x[:, n] * np.exp(-k) + s[:, n] * dt * phi1(k)
    return x


def invert_single(b, beta, E, lam, dt, iterations=30):
    """Conductance g (N,L) such that exponential Euler with source g(E-beta) reproduces b."""
    D = E - beta
    bn, bn1, Dn = b[:, :-1], b[:, 1:], D[:, :-1]
    g = ((bn1 - bn) / dt + lam * bn) / (Dn - bn)
    for _ in range(iterations):

        def f(x):
            k = (lam + x) * dt
            return bn * np.exp(-k) + x * Dn * dt * phi1(k) - bn1

        val = f(g)
        h = 1e-7 * (1.0 + np.abs(g))
        slope = (f(g + h) - val) / h
        step = np.divide(val, slope, out=np.zeros_like(val), where=np.abs(slope) > 0)
        g = g - step
        if np.max(np.abs(step)) < 1e-15:
            break
    out = np.zeros_like(b)
    if not np.isfinite(g).all():
        raise ValueError('Single conductance inversion produced non-finite values.')
    out[:, :-1] = g
    return out


def integration_conductance(target, g, s, beta, lam, dt, E_int):
    u, target_next = target[:, :-1], target[:, 1:]
    g, s = g[:, :-1], s[:, :-1]
    driving = E_int - beta[:, :-1]
    H = ((target_next - u) / dt + (lam + g) * u - s) / (driving - u)
    for _ in range(30):

        def residual(h):
            k = (lam + g + h) * dt
            return u * np.exp(-k) + (s + h * driving) * dt * phi1(k) - target_next

        value = residual(H)
        delta = 1e-7 * (1 + abs(H))
        slope = (residual(H + delta) - value) / delta
        update = np.divide(value, slope, out=np.zeros_like(value), where=abs(slope) > 0)
        H -= update
        if np.max(abs(update)) < 1e-15:
            break
    if not np.isfinite(H).all():
        raise ValueError('Pair conductance inversion produced non-finite values.')
    out = np.zeros_like(target)
    out[:, :-1] = H
    k = (lam + g + H) * dt
    err = float(np.max(abs(u * np.exp(-k) + (s + H * driving) * dt * phi1(k) - target_next)))
    if not np.isfinite(err):
        raise ValueError('Non-finite pair replay residual.')
    return out, err


def pack(rows, padding):
    lengths = np.asarray([len(x) for x in rows], dtype=np.int64)
    ptr = np.r_[0, np.cumsum(lengths)]
    return np.r_[np.concatenate(rows), np.zeros(padding)], ptr


def interpolate_single(raw, loc, age, length):
    axis = raw['event_age_steps']
    lo = int(np.clip(np.searchsorted(axis, age, side='right') - 1, 0, len(axis) - 1))
    hi = min(lo + 1, len(axis) - 1)
    ratio = 0.0 if hi == lo else float(np.clip((age - axis[lo]) / (axis[hi] - axis[lo]), 0.0, 1.0))
    ptr = raw['post_rebase_single_voltage_ptr']
    values = raw['post_rebase_single_voltage_mV']

    def get(node):
        a, b = ptr[loc * len(axis) + node : loc * len(axis) + node + 2]
        out = np.zeros(length)
        n = min(length, b - a)
        out[:n] = values[a : a + n]
        return out

    a = get(lo)
    return a if ratio == 0 else a + ratio * (get(hi) - a)


def _root_cv(cell) -> int:
    roots = [int(cv.id) for cv in cell.cvs if cv.parent_cv is None]
    if len(roots) != 1:
        raise ValueError("DIF calibration requires exactly one root CV.")
    return roots[0]


def _root_cv_and_point(cell) -> tuple[int, int]:
    root_cv = _root_cv(cell)
    root_point = int(cell.node_tree.cv_to_mid_node_id[root_cv])
    return root_cv, root_point
