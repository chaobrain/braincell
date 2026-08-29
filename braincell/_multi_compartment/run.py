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

"""Simulation time loop and trace helpers for :meth:`Cell.run`.

Propagates ``t`` through :mod:`brainstate.environ` inside the
``for_loop`` scan instead of mutating ``Cell._current_time``
per step; the final post-loop time is pinned once after the scan.
"""

import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import is_traced_value
from braincell.recording import SampleBlock

if TYPE_CHECKING:
    from .cell import Cell

__all__ = ["RunResult", "run"]


@dataclass(frozen=True)
class RunResult:
    """Return value of :meth:`Cell.run`.

    Attributes
    ----------
    time : brainunit.Quantity
        ``(n_steps,)`` time array spanning ``[start_t, start_t + duration)``.
    traces : dict[str, Any]
        Probe-name → trace array mapping; one entry per placed probe.
    """

    time: object
    traces: dict
    samples: dict = field(default_factory=dict)
    start_time: object | None = None
    stop_time: object | None = None
    dt: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", MappingProxyType(dict(self.traces)))
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))

    @classmethod
    def concat(cls, parts) -> "RunResult":
        """Concatenate contiguous result segments with identical schemas.

        Parameters
        ----------
        parts : iterable of RunResult
            Ordered contiguous segments.

        Returns
        -------
        RunResult
            Concatenated immutable result.
        """
        parts = tuple(parts)
        if not parts:
            raise ValueError("RunResult.concat(...) requires at least one part.")
        first = parts[0]
        if any(part.dt is None for part in parts):
            raise ValueError("RunResult.concat(...) requires results carrying dt metadata.")
        for previous, current in zip(parts, parts[1:]):
            if not _same_time_quantity(previous.dt, current.dt):
                raise ValueError("RunResult.concat(...) requires identical dt values.")
            if not _same_time_quantity(previous.stop_time, current.start_time):
                raise ValueError("RunResult.concat(...) requires contiguous segments.")
            if tuple(previous.samples) != tuple(current.samples):
                raise ValueError("RunResult.concat(...) requires identical recording names.")
            for name in previous.samples:
                if previous.samples[name].schema != current.samples[name].schema:
                    raise ValueError(f"Recording schema changed for {name!r}.")
        samples = {name: _concat_sample_blocks(tuple(part.samples[name] for part in parts)) for name in first.samples}
        common_traces = set(first.traces)
        for part in parts[1:]:
            common_traces.intersection_update(part.traces)
        traces = {
            name: _concat_values(tuple(part.traces[name] for part in parts))
            for name in first.traces
            if name in common_traces
        }
        return cls(
            time=_concat_values(tuple(part.time for part in parts)),
            traces=traces,
            samples=samples,
            start_time=first.start_time,
            stop_time=parts[-1].stop_time,
            dt=first.dt,
        )


def run(rcell: "Cell", *, dt, duration) -> RunResult:
    """Advance ``rcell`` for ``duration`` at ``dt`` and collect recordings."""
    _validate_time_quantity(dt, name="dt")
    _validate_time_quantity(duration, name="duration")
    n_steps = _duration_steps(duration, dt)
    rcell.connections.prepare_runtime(dt)

    compiled_recordings = rcell._compiled_recordings(dt)
    probe_samples = rcell.sample_probes()
    ordered_probe_names = tuple(sorted(probe_samples))
    schedule_issues = _differentiable_schedule_issues(compiled_recordings, n_steps=n_steps)

    with brainstate.environ.context(dt=dt):
        relative_times = u.math.arange(n_steps) * brainstate.environ.get_dt()
        cached_run = _cached_run_loop(
            rcell,
            dt=dt,
            n_steps=n_steps,
            compiled_recordings=compiled_recordings,
            ordered_probe_names=ordered_probe_names,
        )
        times, traces_over_time = cached_run(relative_times)

    traced = is_traced_value(times)
    if traced and schedule_issues:
        details = "; ".join(schedule_issues)
        raise ValueError(
            "Cell.run(...) recording under jax.jit/grad requires start=0 and duration "
            "to be an integer multiple of every recording period; "
            f"incompatible recordings: {details}. Adjust start, period, or duration."
        )
    if not traced:
        _warn_for_variable_length_schedules(compiled_recordings, n_steps=n_steps, duration=duration)

    n_output = len(compiled_recordings) + len(ordered_probe_names)
    values_tuple = _normalize_run_traces(traces_over_time, n_traces=n_output)
    recording_values = values_tuple[: len(compiled_recordings)]
    probe_values = values_tuple[len(compiled_recordings) :]
    start_time = rcell.current_time - duration
    stop_time = rcell.current_time

    samples = {}
    for compiled, values in zip(compiled_recordings, recording_values):
        if _has_fixed_shape_schedule(compiled, n_steps=n_steps):
            indices = _fixed_recording_indices(
                start_time,
                dt=dt,
                period_steps=compiled.period_steps,
                n_steps=n_steps,
            )
            selected = values[indices]
            first_time = times[indices[0]]
        else:
            mask = _recording_time_mask(times, compiled.schema)
            selected = values[mask]
            first_time = None if not np.any(mask) else times[int(np.flatnonzero(mask)[0])]
        samples[compiled.spec.name] = SampleBlock(
            values=selected,
            schema=compiled.schema,
            segment_start=start_time,
            segment_stop=stop_time,
            first_time=first_time,
        )
    traces = {name: trace for name, trace in zip(ordered_probe_names, probe_values)}
    traces.update({name: block.values for name, block in samples.items()})
    return RunResult(
        time=times,
        traces=traces,
        samples=samples,
        start_time=start_time,
        stop_time=stop_time,
        dt=dt,
    )


def _cached_run_loop(
    rcell: "Cell",
    *,
    dt,
    n_steps: int,
    compiled_recordings: tuple,
    ordered_probe_names: tuple[str, ...],
):
    """Return a persistent jitted loop for one run shape.

    Notes
    -----
    ``brainstate.transform.for_loop`` builds a fresh ``scan`` when called
    from a new Python closure. Caching the enclosing jitted function avoids
    paying XLA compile time on repeated ``Cell.run`` calls with the same
    timestep, step count, and probe layout.
    """
    cache = getattr(rcell, "_run_loop_cache", None)
    if cache is None:
        return _make_run_loop(
            rcell,
            dt=dt,
            compiled_recordings=compiled_recordings,
            ordered_probe_names=ordered_probe_names,
        )

    key = (
        _time_quantity_cache_value(dt),
        int(n_steps),
        tuple((item.spec.name, item.schema.size) for item in compiled_recordings),
        tuple(ordered_probe_names),
        int(brainstate.environ.get_precision()),
    )
    cached = cache.get(key)
    if cached is None:
        cached = _make_run_loop(
            rcell,
            dt=dt,
            compiled_recordings=compiled_recordings,
            ordered_probe_names=ordered_probe_names,
        )
        cache[key] = cached
    return cached


def _make_run_loop(rcell: "Cell", *, dt, compiled_recordings: tuple, ordered_probe_names: tuple[str, ...]):
    """Create the jitted stateful run loop for a fixed observer layout."""

    def _run_loop(relative_times):
        with brainstate.environ.context(dt=dt):
            start_t = rcell.current_time
            times = start_t + relative_times
            with brainstate.environ.context(t=start_t):
                rcell._prepare_next_synapse_inputs()

            def _step(t):
                with brainstate.environ.context(t=t):
                    with jax.named_scope("braincell:cell_run:sample_recordings"):
                        recording_snapshot = tuple(item.sample() for item in compiled_recordings)
                    with jax.named_scope("braincell:cell_run:begin_step"):
                        rcell._begin_step()
                    with jax.named_scope("braincell:cell_run:update_dynamics"):
                        rcell._update_dynamics()
                    with jax.named_scope("braincell:cell_run:route_live_connections"):
                        rcell._apply_direct_live_connection_events()
                    with jax.named_scope("braincell:cell_run:prepare_next_synapse_inputs"):
                        rcell._prepare_next_synapse_inputs(t=t + brainstate.environ.get_dt())
                    # Placed Probe objects keep their legacy post-step sampling
                    # contract during the transition to layout-free recordings.
                    with jax.named_scope("braincell:cell_run:sample_legacy_probes"):
                        probe_snapshot = rcell.sample_probes()
                return recording_snapshot + tuple(probe_snapshot[name] for name in ordered_probe_names)

            traces_over_time = brainstate.transform.for_loop(_step, times)
            rcell._set_current_time(start_t + int(times.shape[0]) * brainstate.environ.get_dt())
        return times, traces_over_time

    return brainstate.transform.jit(_run_loop)


def _validate_time_quantity(value, *, name: str) -> None:
    """Require ``value`` to be a positive scalar time :class:`Quantity`."""
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"Cell.run(...) {name} must be a time quantity, got {value!r}.")
    decimal = np.asarray(value.to_decimal(u.ms), dtype=float)
    if decimal.shape not in ((), (1,)):
        raise ValueError(f"Cell.run(...) {name} must be scalar, got shape {decimal.shape!r}.")
    if float(decimal.reshape(())) <= 0.0:
        raise ValueError(f"Cell.run(...) {name} must be > 0, got {value!r}.")


def _normalize_run_traces(values, *, n_traces: int) -> tuple:
    """Wrap scalar ``for_loop`` output when a single trace is collected."""
    if n_traces == 0:
        return ()
    if n_traces == 1:
        return values if isinstance(values, tuple) else (values,)
    if not isinstance(values, tuple):
        raise TypeError(f"Cell.run(...) expected {n_traces} trace arrays, got {type(values).__name__!s}.")
    if len(values) != n_traces:
        raise ValueError(f"Cell.run(...) expected {n_traces} trace arrays, got {len(values)!r}.")
    return values


def _time_quantity_cache_value(value) -> tuple[float, str]:
    """Return a stable cache token for a scalar time quantity."""
    return (float(np.asarray(value.to_decimal(u.ms), dtype=float).reshape(())), "ms")


def _duration_steps(duration, dt) -> int:
    duration_ms = float(np.asarray(duration.to_decimal(u.ms), dtype=float).reshape(()))
    dt_ms = float(np.asarray(dt.to_decimal(u.ms), dtype=float).reshape(()))
    ratio = duration_ms / dt_ms
    steps = int(round(ratio))
    if steps <= 0 or not np.isclose(ratio, steps, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Cell.run(...) duration must be an integer multiple of dt; got {duration!r} and {dt!r}.")
    return steps


def _recording_time_mask(times, schema) -> np.ndarray:
    time_ms = np.asarray(times.to_decimal(u.ms), dtype=float)
    start_ms = float(np.asarray(schema.schedule_start.to_decimal(u.ms)).reshape(()))
    period_ms = float(np.asarray(schema.period.to_decimal(u.ms)).reshape(()))
    relative = (time_ms - start_ms) / period_ms
    return (time_ms >= start_ms - 1e-9) & np.isclose(relative, np.rint(relative), rtol=1e-6, atol=1e-6)


def _has_fixed_shape_schedule(compiled, *, n_steps: int) -> bool:
    return compiled.start_steps == 0 and n_steps % compiled.period_steps == 0


def _differentiable_schedule_issues(compiled_recordings, *, n_steps: int) -> tuple[str, ...]:
    issues = []
    for compiled in compiled_recordings:
        reasons = []
        if compiled.start_steps != 0:
            reasons.append(f"start={compiled.schema.schedule_start!r}")
        if n_steps % compiled.period_steps != 0:
            reasons.append(f"period={compiled.schema.period!r} does not divide duration")
        if reasons:
            issues.append(f"{compiled.spec.name!r} ({', '.join(reasons)})")
    return tuple(issues)


def _warn_for_variable_length_schedules(compiled_recordings, *, n_steps: int, duration) -> None:
    unaligned = tuple(compiled for compiled in compiled_recordings if n_steps % compiled.period_steps != 0)
    if not unaligned:
        return
    names = ", ".join(repr(compiled.spec.name) for compiled in unaligned)
    warnings.warn(
        "Cell.run(...) duration "
        f"{duration!r} is not an integer multiple of the recording period for {names}; "
        "eager mode returns variable-length sample blocks, but this schedule is unsupported "
        "under jax.jit/grad. Adjust period or duration for differentiable recording.",
        RuntimeWarning,
        stacklevel=3,
    )


def _fixed_recording_indices(start_time, *, dt, period_steps: int, n_steps: int):
    dt_ms = float(np.asarray(dt.to_decimal(u.ms), dtype=float).reshape(()))
    start_step = jnp.rint(start_time.to_decimal(u.ms) / dt_ms).astype(jnp.int32)
    offset = jnp.mod(-start_step, period_steps)
    return offset + jnp.arange(n_steps // period_steps, dtype=jnp.int32) * period_steps


def _same_time_quantity(left, right) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        np.allclose(
            np.asarray(left.to_decimal(u.ms), dtype=float),
            np.asarray(right.to_decimal(u.ms), dtype=float),
            rtol=1e-7,
            atol=1e-9,
        )
    )


def _concat_values(values):
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(u.math.concatenate(tuple(value.to_decimal(unit) for value in values), axis=0), unit)
    return u.math.concatenate(values, axis=0)


def _concat_sample_blocks(blocks):
    first = blocks[0]
    first_time = next((block.first_time for block in blocks if block.first_time is not None), None)
    return SampleBlock(
        values=_concat_values(tuple(block.values for block in blocks)),
        schema=first.schema,
        segment_start=first.segment_start,
        segment_stop=blocks[-1].segment_stop,
        first_time=first_time,
    )
