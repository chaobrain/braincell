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
``for_loop`` scan instead of mutating the Cell's clock per step. The
step times are ``origin + (step + k)·dt`` from the clock's integer step
count, so a run split into segments steps at the same times as one run;
the clock advances once after the scan.
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

from braincell._misc import (
    concat_values as _concat_values,
    is_traced_value,
    normalize_loop_outputs as _normalize_loop_outputs,
    register_result_pytree,
    scalar_decimal as _scalar_decimal,
    same_time_quantity as _same_time_quantity,
    validate_time_quantity,
)
from braincell.network.recording import EventSeries, SampleBlock, concat_sample_blocks

if TYPE_CHECKING:
    from braincell._multi_compartment.cell import Cell

__all__ = ["RunResult", "run"]

#: The most steps a clock counts: its step count is an int32 (CELL-45).
MAX_STEPS = 2**31 - 1


def host_clock_start(clock, dt) -> tuple[float, int, bool]:
    """``(origin in ms, step, counts dt)`` a run at ``dt`` starts from, on the host, from ``Cell._clock_host()``.

    The host side of ``Cell._clock_start``: a ``dt`` other than the one the
    clock counts (``counts dt`` false) moves the origin to the time reached
    and the count to 0. The two ``dt`` are compared in the run precision,
    as the compiled loop compares them.
    """
    origin_ms, step, clock_dt_ms = clock
    dt_ms = _scalar_decimal(dt, u.ms)
    float_type = brainstate.environ.dftype()
    if np.asarray(clock_dt_ms, dtype=float_type) == np.asarray(dt_ms, dtype=float_type):
        return origin_ms, step, True
    return origin_ms + step * clock_dt_ms, 0, False


def check_step_limit(step: int, n_steps: int, *, dt, prefix: str) -> None:
    """Raise ``ValueError`` if ``n_steps`` more steps would take the step count past :data:`MAX_STEPS`."""
    if int(step) + int(n_steps) <= MAX_STEPS:
        return
    hours = MAX_STEPS * _scalar_decimal(dt, u.ms) / 3.6e6
    raise ValueError(
        f"{prefix} would take the step count to {int(step) + int(n_steps)}, past its limit of "
        f"2**31 - 1 = {MAX_STEPS} steps ({hours:.3g} h at dt={dt!r}); call reset_state() to start "
        "again from t = 0."
    )


@dataclass(frozen=True)
class RunResult:
    """Return value of :meth:`Cell.run`.

    Attributes
    ----------
    time : brainunit.Quantity
        ``(n_steps,)`` time array spanning ``[start_t, start_t + duration)``.
    traces : Mapping[str, Any]
        Recording-name → values of every :meth:`Cell.record` recording, the
        ``values`` of the matching ``samples`` block.
    samples : Mapping[str, SampleBlock]
        Recording-name → regular samples of each :meth:`Cell.record`
        recording.
    start_time, stop_time, dt : brainunit.Quantity or None
        Segment bounds ``[start_time, stop_time)`` and the fixed step.
    spikes : EventSeries or None
        Spikes of the Cell's ``"spike"`` port in the segment, in the format
        of ``NetworkResult.events[population]["spike"]`` for the same Cell
        in a Network (whose metadata also holds ``"population"``). ``None``
        when the run is traced (``jax.jit``, ``jax.grad``): the number of
        events depends on the data.

    Notes
    -----
    The ``"spike"`` port is ``cell.spike`` at the CV holding the root
    branch midpoint (the one CV of a lumped Cell, or a reduction model's
    event). A step from ``t_k`` to ``t_k + dt`` gives an event at ``t_k``
    when ``V(t_k) < V_th <= V(t_k + dt)`` for the default ``spk_fun``;
    voltage staying above ``V_th`` gives no further event. ``count`` is 1
    per crossing (a reduction model's event value). Rows are sorted by
    time, then ``source_id``.

    ``source_id`` indexes the rows of ``(batch,) + pop_size`` flattened in C
    order: without a batch axis and with a one-dimensional ``pop_size`` it is
    the population member. ``metadata["cv_id"]`` gives each row's CV; with a
    batch axis, ``metadata["batch_index"]`` and
    ``metadata["population_index"]`` give its batch element and member.

    A result is a JAX pytree, so ``jax.block_until_ready(result)`` waits for
    its arrays and a jitted function can return it.

    Examples
    --------
    >>> import brainunit as u
    >>> import braincell
    >>> from braincell.filter import RootLocation
    >>> mech = braincell.mech
    >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
    >>> cell = cell.paint(mech.Ion("SodiumFixed", E=50.0 * u.mV), mech.Ion("PotassiumFixed", E=-77.0 * u.mV))
    >>> cell = cell.paint(mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2))
    >>> cell = cell.paint(mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2))
    >>> cell = cell.paint(mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV))
    >>> clamp = mech.CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA)
    >>> cell = cell.place(RootLocation(0.5), clamp)
    >>> spikes = cell.run(dt=0.025 * u.ms, duration=10.0 * u.ms).spikes
    >>> spikes.source_id.tolist(), spikes.metadata["port"]
    ([0], 'spike')
    """

    time: object
    traces: dict
    samples: dict = field(default_factory=dict)
    start_time: object | None = None
    stop_time: object | None = None
    dt: object | None = None
    spikes: EventSeries | None = None

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
            Concatenated immutable result. Its ``spikes`` joins those of the
            parts, or is ``None`` if any part has none.
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
            spikes=_concat_spikes(tuple(part.spikes for part in parts)),
        )


register_result_pytree(RunResult, ("time", "traces", "samples", "start_time", "stop_time", "dt", "spikes"))


def _concat_spikes(series) -> EventSeries | None:
    """Join the spikes of contiguous segments; ``None`` if any segment has none."""
    if any(item is None for item in series):
        return None
    return EventSeries(
        time=_concat_values(tuple(item.time for item in series)),
        source_id=np.concatenate(tuple(item.source_id for item in series)),
        count=np.concatenate(tuple(item.count for item in series)),
        metadata=series[0].metadata,
    )


def run(rcell: "Cell", *, dt, duration) -> RunResult:
    """Advance ``rcell`` for ``duration`` at ``dt`` and collect recordings.

    Notes
    -----
    Each call dispatches one cached, jitted program: the step loop, the
    segment start / stop times and the selection of fixed-schedule samples
    all run inside it. Python work per call is limited to argument checks,
    the cache lookup, turning the step-by-step spike bits into sparse rows
    (one host pass over ``n_steps * n_rows / 8`` bytes) and wrapping the
    outputs, so short repeated runs are not dominated by eager array
    operations. A traced run (inside ``jax.jit`` or ``jax.grad``) cannot
    build sparse rows and uses a loop without the spike output.
    """
    from braincell.network.event import _prepare_netstim_schedules, _schedule_horizon

    _validate_time_quantity(dt, name="dt")
    _validate_time_quantity(duration, name="duration")
    n_steps = _duration_steps(duration, dt)
    rcell.connections.prepare_runtime(dt)

    # A scheduled source needs its events up to the last step's next-input
    # boundary, ``stop``; one more step bounds every rounding. A traced start
    # leaves the end unknown, so such a run generates every event.
    clock = rcell._clock_host()
    until_ms = None
    with_spikes = clock is not None
    if with_spikes:
        origin_ms, step, _ = host_clock_start(clock, dt)
        check_step_limit(step, n_steps, dt=dt, prefix="Cell.run(...)")
        until_ms = origin_ms + (step + n_steps + 1) * _scalar_decimal(dt, u.ms)
    schedule_key = _prepare_netstim_schedules(
        (view.source for view in rcell.connections._call_views(scheduled=True)),
        until_ms=until_ms,
    )

    compiled_recordings = rcell._compiled_recordings(dt)
    plan = _cached_run_plan(
        rcell,
        dt=dt,
        n_steps=n_steps,
        compiled_recordings=compiled_recordings,
        schedule_key=schedule_key,
        with_spikes=with_spikes,
    )

    # Value inputs are arguments of the compiled loop, which applies them to
    # its first step; as constants of its trace they would act on every step
    # of this run and of every later run that reuses the loop.
    values = rcell._take_value_inputs()
    try:
        with brainstate.environ.context(dt=dt), _schedule_horizon(until_ms):
            times, start_time, stop_time, recorded, spike_rows = plan.loop(values)
    except BaseException:
        rcell._restore_value_inputs(values)
        raise

    traced = is_traced_value(times)
    if traced and plan.schedule_issues:
        details = "; ".join(plan.schedule_issues)
        raise ValueError(
            "Cell.run(...) recording under jax.jit/grad requires start=0 and duration "
            "to be an integer multiple of every recording period; "
            f"incompatible recordings: {details}. Adjust start, period, or duration."
        )
    if not traced and plan.unaligned_names:
        warnings.warn(
            "Cell.run(...) duration "
            f"{duration!r} is not an integer multiple of the recording period for {plan.unaligned_names}; "
            "eager mode returns variable-length sample blocks, but this schedule is unsupported "
            "under jax.jit/grad. Adjust period or duration for differentiable recording.",
            RuntimeWarning,
            stacklevel=3,
        )

    samples = {}
    for compiled, fixed, (values, first_time) in zip(compiled_recordings, plan.fixed_schedules, recorded):
        if not fixed:
            mask = _recording_time_mask(times, compiled.schema)
            values = values[mask]
            first_time = None if not np.any(mask) else times[int(np.flatnonzero(mask)[0])]
        samples[compiled.spec.name] = SampleBlock(
            values=values,
            schema=compiled.schema,
            segment_start=start_time,
            segment_stop=stop_time,
            first_time=first_time,
        )
    traces = {name: block.values for name, block in samples.items()}
    return RunResult(
        time=times,
        traces=traces,
        samples=samples,
        start_time=start_time,
        stop_time=stop_time,
        dt=dt,
        spikes=None if traced or plan.spike_port is None else plan.spike_port.series(rcell, times, spike_rows),
    )


@dataclass(frozen=True)
class _SpikePort:
    """How one run loop reads the Cell's ``"spike"`` port, and its sparse rows.

    The port is the one ``cell.event_outputs["spike"]`` publishes to a
    Network: ``cell.spike`` at the CV holding the root branch midpoint, the
    one CV of a lumped Cell, or a reduction model's event. Detailed and
    lumped spikes are 0 or 1 (the Heaviside forward of ``spk_fun``), so each
    step packs them eight rows to a byte; a reduction model's event is kept
    as it is, since it may count several events in one step.
    """

    cv_id: int
    packed: bool

    @classmethod
    def of(cls, rcell: "Cell") -> "_SpikePort":
        return cls(
            cv_id=int(rcell._get_spike_event_source().cv_id),
            packed=not rcell._uses_reduction,
        )

    def read(self, rcell: "Cell"):
        """This step's port values, inside the loop."""
        spike = rcell.spike.value
        if not self.packed:
            return spike
        if not rcell._is_lumped:
            spike = spike[..., self.cv_id]
        return jnp.packbits(spike.reshape(-1) != 0)

    def series(self, rcell: "Cell", times, rows) -> EventSeries:
        """Sparse rows in the format of ``NetworkResult.events[population]["spike"]``."""
        rows = np.asarray(rows)
        if rcell._uses_reduction:
            values = rows.reshape(rows.shape[0], -1)
            step, source_id = np.nonzero(values)
            count = values[step, source_id]
            n_rows = values.shape[1]
        else:
            shape = rcell.spike.value.shape
            n_rows = int(np.prod(shape if rcell._is_lumped else shape[:-1], dtype=np.int64))
            # Only the nonzero bytes are expanded to their eight rows.
            step, byte = np.nonzero(rows)
            hit, bit = np.nonzero(np.unpackbits(rows[step, byte][:, None], axis=1))
            step, source_id = step[hit], byte[hit] * 8 + bit
            count = np.ones(source_id.shape, dtype=np.int64)
        metadata = {"port": "spike", "cv_id": np.full(n_rows, self.cv_id, dtype=np.int64)}
        batch_size = rcell._runtime_batch_size
        if batch_size is not None:
            metadata["batch_index"], metadata["population_index"] = np.divmod(
                np.arange(n_rows, dtype=np.int64), n_rows // batch_size
            )
        return EventSeries(
            time=u.Quantity(np.asarray(u.get_mantissa(times))[step], u.get_unit(times)),
            source_id=source_id,
            count=count,
            metadata=metadata,
        )


@dataclass(frozen=True)
class _RunPlan:
    """Static schedule facts and the jitted loop for one run shape."""

    loop: object
    fixed_schedules: tuple[bool, ...]
    schedule_issues: tuple[str, ...]
    unaligned_names: str
    spike_port: _SpikePort | None = None


def _cached_run_plan(
    rcell: "Cell",
    *,
    dt,
    n_steps: int,
    compiled_recordings: tuple,
    schedule_key: tuple = (),
    with_spikes: bool = True,
) -> _RunPlan:
    """Return the persistent plan for one run shape.

    Notes
    -----
    ``brainstate.transform.for_loop`` builds a fresh ``scan`` when called
    from a new Python closure. Caching the enclosing jitted function avoids
    paying XLA compile time on repeated ``Cell.run`` calls with the same
    timestep, step count, and recording layout. The schedule checks depend on
    the same key, so they are computed once here as well. ``schedule_key``
    names the NetStim events the loop bakes in, so a run that needed more
    of them compiles a new loop. ``with_spikes`` selects the loop that
    also outputs the spike port; a traced run uses the one without.
    """
    cache = rcell._run_loop_cache
    key = (
        _time_quantity_cache_value(dt),
        int(n_steps),
        tuple((item.spec.name, item.schema.size) for item in compiled_recordings),
        int(brainstate.environ.get_precision()),
        schedule_key,
        bool(with_spikes),
    )
    cached = cache.get(key)
    if cached is None:
        fixed_schedules = tuple(
            _has_fixed_shape_schedule(compiled, dt=dt, n_steps=n_steps) for compiled in compiled_recordings
        )
        unaligned = tuple(
            compiled for compiled in compiled_recordings if n_steps % _recording_steps(compiled.schema.period, dt) != 0
        )
        spike_port = _SpikePort.of(rcell) if with_spikes else None
        cached = _RunPlan(
            loop=_make_run_loop(
                rcell,
                dt=dt,
                n_steps=n_steps,
                compiled_recordings=compiled_recordings,
                fixed_schedules=fixed_schedules,
                spike_port=spike_port,
            ),
            fixed_schedules=fixed_schedules,
            schedule_issues=_differentiable_schedule_issues(compiled_recordings, dt=dt, n_steps=n_steps),
            unaligned_names=", ".join(repr(compiled.spec.name) for compiled in unaligned),
            spike_port=spike_port,
        )
        cache[key] = cached
    return cached


def _make_run_loop(
    rcell: "Cell",
    *,
    dt,
    n_steps: int,
    compiled_recordings: tuple,
    fixed_schedules: tuple[bool, ...],
    spike_port: _SpikePort | None = None,
):
    """Create the jitted stateful run loop for a fixed observer layout.

    The loop takes the value inputs (``Cell._take_value_inputs``), which act
    on its first step, and returns
    ``(times, start_time, stop_time, recorded, spike_rows)``;
    ``spike_rows`` stacks ``spike_port.read`` over the steps, or is ``None``
    without a port.
    ``recorded`` holds one ``(values, first_time)`` pair per recording: the
    selected samples for a fixed-shape schedule, or every step's values and
    ``None`` for a variable-length one (selected eagerly by the caller).
    """
    n_recordings = len(compiled_recordings)
    period_steps = tuple(_recording_steps(compiled.schema.period, dt) for compiled in compiled_recordings)

    def _run_loop(values):
        with brainstate.environ.context(dt=dt):
            origin, step = rcell._clock_start(dt)
            start_t = origin + step * dt
            times = origin + (step + jnp.arange(n_steps, dtype=jnp.int32)) * dt
            rcell._prepare_first_synapse_inputs(t=start_t)

            def _step(t):
                with brainstate.environ.context(t=t):
                    with jax.named_scope("braincell:cell_run:prepare_clamps"):
                        rcell._prepare_step_clamps(t=t, dt=brainstate.environ.get_dt())
                    with jax.named_scope("braincell:cell_run:sample_recordings"):
                        pre_recording_snapshot = {
                            index: item.sample()
                            for index, item in enumerate(compiled_recordings)
                            if item.phase == "pre"
                        }
                    with jax.named_scope("braincell:cell_run:begin_step"):
                        rcell._begin_step()
                    with jax.named_scope("braincell:cell_run:update_dynamics"):
                        with brainstate.environ.context(_braincell_step_clamps_prepared=True):
                            rcell._update_dynamics()
                    with jax.named_scope("braincell:cell_run:route_live_connections"):
                        rcell._apply_direct_live_connection_events()
                    with jax.named_scope("braincell:cell_run:prepare_next_synapse_inputs"):
                        rcell._prepare_next_synapse_inputs(t=t + brainstate.environ.get_dt())
                    with jax.named_scope("braincell:cell_run:sample_outputs"):
                        recording_snapshot = tuple(
                            pre_recording_snapshot[index] if item.phase == "pre" else item.sample()
                            for index, item in enumerate(compiled_recordings)
                        )
                    spikes = ()
                    if spike_port is not None:
                        with jax.named_scope("braincell:cell_run:sample_spikes"):
                            spikes = (spike_port.read(rcell),)
                return recording_snapshot + spikes

            if any(values):

                def _step_with_values(index, t):
                    # The value inputs act on the first step only.
                    with rcell._value_input_step(values, first=index == 0):
                        return _step(t)

                traces_over_time = brainstate.transform.for_loop(_step_with_values, jnp.arange(n_steps), times)
            else:
                traces_over_time = brainstate.transform.for_loop(_step, times)
            stop_t = origin + (step + n_steps) * dt
            rcell._set_clock(origin, step + n_steps, dt)

        values = _normalize_loop_outputs(
            traces_over_time,
            count=n_recordings + (spike_port is not None),
            prefix="Cell.run(...)",
            noun="trace arrays",
        )
        spike_rows = None
        if spike_port is not None:
            spike_rows = values[-1]
            values = values[:-1]
        recorded = []
        for index, fixed in enumerate(fixed_schedules):
            if fixed:
                indices = _fixed_recording_indices(
                    origin, step, dt=dt, period_steps=period_steps[index], n_steps=n_steps
                )
                recorded.append((values[index][indices], times[indices[0]]))
            else:
                recorded.append((values[index], None))
        return times, start_t, stop_t, tuple(recorded), spike_rows

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


def _recording_steps(value, dt) -> int:
    return int(round(_scalar_decimal(value, u.ms) / _scalar_decimal(dt, u.ms)))


def _has_fixed_shape_schedule(compiled, *, dt, n_steps: int) -> bool:
    start_steps = _recording_steps(compiled.schema.schedule_start, dt)
    period_steps = _recording_steps(compiled.schema.period, dt)
    return start_steps == 0 and n_steps % period_steps == 0


def _differentiable_schedule_issues(compiled_recordings, *, dt, n_steps: int) -> tuple[str, ...]:
    issues = []
    for compiled in compiled_recordings:
        reasons = []
        period_steps = _recording_steps(compiled.schema.period, dt)
        if _recording_steps(compiled.schema.schedule_start, dt) != 0:
            reasons.append(f"start={compiled.schema.schedule_start!r}")
        if n_steps % period_steps != 0:
            reasons.append(f"period={compiled.schema.period!r} does not divide duration")
        if reasons:
            issues.append(f"{compiled.spec.name!r} ({', '.join(reasons)})")
    return tuple(issues)


def _fixed_recording_indices(origin, step, *, dt, period_steps: int, n_steps: int):
    """Indices of the steps on the recording grid ``0, period, 2·period, …`` (in steps from ``t = 0``).

    The run starts ``step`` steps after the clock's ``origin``, which is 0
    unless ``dt`` changed: the step count places the run on the grid
    exactly, however long the run has been (CELL-45).
    """
    dt_ms = float(np.asarray(dt.to_decimal(u.ms), dtype=float).reshape(()))
    start_step = step + jnp.rint(origin.to_decimal(u.ms) / dt_ms).astype(jnp.int32)
    offset = jnp.mod(-start_step, period_steps)
    return offset + jnp.arange(n_steps // period_steps, dtype=jnp.int32) * period_steps
