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

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import brainstate
import brainunit as u
import jax
import numpy as np

from braincell._misc import (
    concat_values as _concat_values,
    normalize_loop_outputs as _normalize_loop_outputs,
    scalar_decimal as _scalar_decimal,
    same_time_quantity as _same_time_quantity,
    validate_time_quantity,
)
from braincell._multi_compartment import probes
from braincell.network.recording import SampleBlock, concat_sample_blocks

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
        samples = {name: concat_sample_blocks(tuple(part.samples[name] for part in parts)) for name in first.samples}
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
    ordered_probe_names = tuple(sorted(probes.probe_names(rcell)))

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

    n_output = len(compiled_recordings) + len(ordered_probe_names)
    values_tuple = _normalize_loop_outputs(
        traces_over_time,
        count=n_output,
        prefix="Cell.run(...)",
        noun="trace arrays",
    )
    recording_values = values_tuple[: len(compiled_recordings)]
    probe_values = values_tuple[len(compiled_recordings) :]
    start_time = rcell.current_time - duration
    stop_time = rcell.current_time

    samples = {}
    for compiled, values in zip(compiled_recordings, recording_values):
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
    cache = rcell._run_loop_cache
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
    validate_time_quantity(value, name=name, prefix="Cell.run(...)")


def _time_quantity_cache_value(value) -> tuple[float, str]:
    """Return a stable cache token for a scalar time quantity."""
    return (_scalar_decimal(value, u.ms), "ms")


def _duration_steps(duration, dt) -> int:
    duration_ms = _scalar_decimal(duration, u.ms)
    dt_ms = _scalar_decimal(dt, u.ms)
    ratio = duration_ms / dt_ms
    steps = int(round(ratio))
    if steps <= 0 or not np.isclose(ratio, steps, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Cell.run(...) duration must be an integer multiple of dt; got {duration!r} and {dt!r}.")
    return steps


def _recording_time_mask(times, schema) -> np.ndarray:
    time_ms = np.asarray(times.to_decimal(u.ms), dtype=float)
    start_ms = _scalar_decimal(schema.schedule_start, u.ms)
    period_ms = _scalar_decimal(schema.period, u.ms)
    relative = (time_ms - start_ms) / period_ms
    return (time_ms >= start_ms - 1e-9) & np.isclose(relative, np.rint(relative), rtol=1e-6, atol=1e-6)
