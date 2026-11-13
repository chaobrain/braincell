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

import contextlib
import functools
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
from braincell.quad._staggered import axial_operator_per_run

if TYPE_CHECKING:
    from braincell._multi_compartment.cell import Cell

__all__ = ["RunResult", "run"]

#: The most steps a clock counts: its step count is an int32 (CELL-45).
MAX_STEPS = 2**31 - 1

# The spike output paths of a run (CELL-49). ``None`` chooses by the
# platform of the Cell's spike state: on GPU a run packs the bits every K
# steps and copies a large grid back compacted. A bool forces the choice
# on any backend; the tests force both paths on CPU.
_BLOCKED_SPIKE_OUTPUT: bool | None = None
_COMPACT_SPIKE_COPY: bool | None = None
#: The block sizes a blocked run takes the largest divisor of its step count from.
_BLOCK_STEPS = range(32, 7, -1)
#: A compacted copy has room for one in this many of the grid's 4-byte groups.
_COMPACT_CAPACITY_DIVISOR = 64
#: Smaller grids are copied back whole, which is faster below about 1 MB on GPU.
_COMPACT_MIN_BYTES = 1 << 20


def host_clock_start(clock, dt) -> tuple[float, int, bool]:
    """``(origin in ms, step, counts dt)`` a run at ``dt`` starts from, on the host, from ``Cell._clock_host()``.

    The host side of ``Cell._clock_start``: a ``dt`` other than the one the
    clock counts (``counts dt`` false) moves the origin to the time reached
    and the count to 0. The two ``dt`` are compared in the run precision,
    as the compiled loop compares them. ``Cell.run`` passes the clock's
    float64 values (:func:`_float64_clock`), so the origin is at float64.
    """
    origin_ms, step, clock_dt_ms = clock
    dt_ms = _scalar_decimal(dt, u.ms)
    float_type = brainstate.environ.dftype()
    if np.asarray(clock_dt_ms, dtype=float_type) == np.asarray(dt_ms, dtype=float_type):
        return origin_ms, step, True
    return origin_ms + step * clock_dt_ms, 0, False


def _float64_clock(rcell: "Cell", clock) -> tuple[float, int, float]:
    """``clock`` (``Cell._clock_host()``) with its origin and ``dt`` at float64, as far as the host knows them.

    The clock's states hold the origin and ``dt`` in the run precision: in
    float32, 3 steps of 0.025 ms reach 0.07500000298 ms. A float64 run
    reaches 0.075 ms, and the arrival steps count from it (CELL-48), so a
    float32 run delivers each event at the step a float64 run does, as from
    time 0 (CELL-46). A run that changed ``dt`` keeps the float64 values of
    the clock it leaves (:func:`_keep_float64_clock`); they hold while the
    states still have the values kept beside them. A clock that something
    else changed (``reset_state``, a traced run at another ``dt``) is taken
    as its states are.
    """
    origin_ms, step, dt_ms = clock
    kept = getattr(rcell, "_run_float64_clock", None)
    if kept is not None and kept[:2] == (origin_ms, dt_ms):
        return kept[2], step, kept[3]
    return clock


def _keep_float64_clock(rcell: "Cell", clock, origin_ms: float, dt, handover=None) -> None:
    """After a run at ``dt`` that changed the clock's ``dt``, keep its float64 origin and ``dt``, and its hand-over.

    ``clock`` is the clock before the run and ``origin_ms`` the float64
    origin the run counted from (:func:`host_clock_start`). The origin
    state moved only if the clock had counted steps, so only then is it read
    back, after the loop. ``handover`` is the run's
    :class:`~braincell.network.event.ArrivalHandover`, which the later runs
    at that ``dt`` keep (:func:`_kept_handover`).
    """
    dt_ms = _scalar_decimal(dt, u.ms)
    origin_image = clock[0] if clock[1] == 0 else _scalar_decimal(rcell._clock_origin.value, u.ms)
    dt_image = float(np.asarray(dt_ms, dtype=brainstate.environ.dftype()))
    rcell._run_float64_clock = (origin_image, dt_image, origin_ms, dt_ms)
    rcell._run_handover = handover


def _keep_started_clock(rcell: "Cell", time_ms: float, dt_ms: float, handover) -> None:
    """Keep the float64 origin ``time_ms`` and ``dt_ms``, and ``handover``, of a clock just started at another ``dt``.

    A Network starting its Cells at its ``dt`` (``Network._start_clocks``)
    keeps them as :func:`_keep_float64_clock` does, beside the clock's
    states as they are now, so that a Cell running on its own after it
    skips the events the Network took, counted at float64 (CELL-57).
    """
    origin_image, _, dt_image = rcell._clock_host()
    rcell._run_float64_clock = (origin_image, dt_image, time_ms, dt_ms)
    rcell._run_handover = handover


def _kept_handover(rcell: "Cell", clock):
    """The :class:`~braincell.network.event.ArrivalHandover` of the clock's ``dt``, or ``None``.

    It is the one of the run that changed the clock to that ``dt``, kept
    while the clock's states hold what that run left
    (:func:`_float64_clock`).
    """
    kept = getattr(rcell, "_run_float64_clock", None)
    if kept is None or kept[:2] != (clock[0], clock[2]):
        return None
    return getattr(rcell, "_run_handover", None)


def _dt_change_handover(rcell: "Cell", clock, dt, prepared: bool | None = None):
    """The :class:`~braincell.network.event.ArrivalHandover` of a run at ``dt`` from ``clock``, at another ``dt``.

    ``clock`` is ``Cell._clock_host()`` and ``prepared`` the Cell's
    prepared-boundary flag, read here when ``None``. The run before took
    the events of its boundaries before the time reached, and of the
    boundary there too when it prepared it; a ``Cell.run`` or Network step
    prepares and consumes the boundary it starts at (CELL-55), so neither
    leaves it prepared. They count
    on that run's grid (:func:`_arrival_origin`), at its float64 ``dt``.
    Of what it took and what the runs before it took (the hand-over it
    kept), the one with the later edge holds. The new run's first boundary
    is the time reached, or the next one when that is prepared (CELL-57).
    ``None`` when nothing ran since ``init_state`` or ``reset_state``.
    """
    from braincell.network.event import ArrivalHandover, round_half_up_steps_host

    origin_ms, step, old_dt_ms = _float64_clock(rcell, clock)
    if old_dt_ms == 0.0:
        return None
    if prepared is None:
        prepared = bool(np.asarray(rcell._next_synapse_inputs_prepared.value))
    numbering = _arrival_origin_ms(origin_ms, old_dt_ms)
    if numbering is None:
        taken_origin, taken_origin_step = 0.0, 0
        reached = int(round_half_up_steps_host(np.asarray(origin_ms / old_dt_ms))) + step
    else:
        taken_origin, taken_origin_step = numbering
        reached = taken_origin_step + step
    dt_ms = _scalar_decimal(dt, u.ms)
    time_ms = origin_ms + step * old_dt_ms
    numbering = _arrival_origin_ms(time_ms, dt_ms)
    first = int(round_half_up_steps_host(np.asarray(time_ms / dt_ms))) if numbering is None else numbering[1]
    handover = ArrivalHandover(
        taken_origin_ms=float(taken_origin),
        taken_dt_ms=float(old_dt_ms),
        taken_origin_step=int(taken_origin_step),
        taken_last_step=reached if prepared else reached - 1,
        first_step=first + 1 if prepared else first,
    )
    chain = _kept_handover(rcell, clock)
    if chain is not None and chain.edge_ms > handover.edge_ms:
        handover = handover._replace(**{name: getattr(chain, name) for name in ArrivalHandover._fields[:4]})
    elif handover.taken_last_step < 0:
        # At t = 0 with nothing prepared: nothing was taken.
        return None
    return handover


def _arrival_origin(origin_ms: float, dt) -> tuple[float, int] | None:
    """``(origin in ms, its step)`` the arrival steps of a run from ``origin_ms`` count from, or ``None``.

    ``None`` when the origin is on the ``dt`` grid from time 0, to within
    the run precision: the loop then numbers its steps on that grid
    (CELL-46), as on every run that has not changed ``dt``. Otherwise the
    arrivals count from the origin, which the loop numbers by its nearest
    step on the grid (``event.arrival_origin``, CELL-48).
    """
    return _arrival_origin_ms(origin_ms, _scalar_decimal(dt, u.ms))


def _arrival_origin_ms(origin_ms: float, dt_ms: float) -> tuple[float, int] | None:
    """:func:`_arrival_origin` with ``dt`` in ms."""
    from braincell.network.event import round_half_up_steps_host

    ratio = origin_ms / dt_ms
    if abs(ratio - np.rint(ratio)) <= 4.0 * np.finfo(brainstate.environ.dftype()).eps * max(1.0, abs(ratio)):
        return None
    return float(origin_ms), int(round_half_up_steps_host(np.asarray(ratio)))


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
    the cache lookup, turning the packed spike bits into sparse rows (one
    host pass over ``n_steps * n_rows / 8`` bytes, or over the nonzero part
    of them) and wrapping the outputs, so short repeated runs are not
    dominated by eager array operations. A traced run (inside ``jax.jit``
    or ``jax.grad``) cannot build sparse rows and uses a loop without the
    spike output.

    On GPU, where each step of the compiled loop costs a launch of its
    kernels, a run whose step count has a divisor in [8, 32] steps in
    blocks of the largest one, K: each step ORs its spike bits into one
    word per row, and each block stores its words once (``_SpikePort``).
    Other step counts pack the bits at every step, as on CPU. A grid of
    1 MiB or more is compacted on the device, so only its nonzero 4-byte
    groups are copied back (CELL-49).

    The geometry is fixed for the run, so the program builds the axial
    operator once, before the steps; each step reads it.

    A run at a ``dt`` the time reached is not a multiple of (3 steps of
    0.025 ms, then 0.05 ms) steps from that time. The arrival steps of its
    scheduled events count from it too, at float64, so each event acts on
    the step nearest it on the run's own grid instead of the grid from time
    0 (CELL-48); the program for such a run is compiled for its origin. A
    run at another ``dt`` prepares the boundary at the time reached, as each
    step prepares the one it starts at (CELL-55), and skips the events the
    runs before took (:func:`_dt_change_handover`, CELL-57); its program is
    compiled for that hand-over, which the segments that follow at that
    ``dt`` keep. A traced run has no host clock and keeps the grid from
    time 0.
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
    arrival_origin = None
    handover = None
    with_spikes = clock is not None
    if with_spikes:
        origin_ms, step, counts_dt = host_clock_start(_float64_clock(rcell, clock), dt)
        check_step_limit(step, n_steps, dt=dt, prefix="Cell.run(...)")
        until_ms = origin_ms + (step + n_steps + 1) * _scalar_decimal(dt, u.ms)
        arrival_origin = _arrival_origin(origin_ms, dt)
        handover = _kept_handover(rcell, clock) if counts_dt else _dt_change_handover(rcell, clock, dt)
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
        arrival_origin=arrival_origin,
        handover=handover,
        spike_path=_spike_output_path(rcell) if with_spikes else None,
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
    if with_spikes and not counts_dt:
        _keep_float64_clock(rcell, clock, origin_ms, dt, handover)

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

    ``block_steps`` is K when the loop packs every K steps instead
    (:meth:`read_bits`); ``capacity`` is the number of nonzero 4-byte
    groups of the packed grid the loop copies back compacted
    (:func:`_compact_spike_rows`), or ``None`` to copy it whole.
    """

    cv_id: int
    packed: bool
    block_steps: int | None = None
    capacity: int | None = None

    @classmethod
    def of(cls, rcell: "Cell", *, n_steps: int, path: tuple[bool, bool]) -> "_SpikePort":
        """The port of ``rcell`` for a run of ``n_steps`` on ``path`` (:func:`_spike_output_path`)."""
        packed = not rcell._uses_reduction
        block_steps = capacity = None
        if packed:
            blocked, compact = path
            if blocked:
                block_steps = next((k for k in _BLOCK_STEPS if n_steps % k == 0), None)
            n_bytes = n_steps * -(-_spike_rows(rcell) // 8)
            if compact and n_bytes >= _COMPACT_MIN_BYTES:
                capacity = max(1, -(-n_bytes // 4) // _COMPACT_CAPACITY_DIVISOR)
        return cls(
            cv_id=int(rcell._get_spike_event_source().cv_id),
            packed=packed,
            block_steps=block_steps,
            capacity=capacity,
        )

    def read(self, rcell: "Cell"):
        """This step's port values, inside the loop."""
        spike = rcell.spike.value
        if not self.packed:
            return spike
        if not rcell._is_lumped:
            spike = spike[..., self.cv_id]
        # Pack the stored spike. Left to itself, XLA recomputes it from the
        # voltages before and after the step inside the packing kernel, which
        # keeps a copy of the old voltage alive on every step (CELL-39).
        spike = jax.lax.optimization_barrier(spike)
        return jnp.packbits(spike.reshape(-1) != 0)

    def read_bits(self, rcell: "Cell"):
        """This step's spike bit of each row, as a ``uint32`` 0 or 1, inside a blocked loop.

        The loop ORs it into the row's word, which fuses into the kernel
        that steps the voltage, so no kernel is left to keep the old voltage
        alive and :meth:`read` needs no barrier here (CELL-49).
        """
        spike = rcell.spike.value
        if not rcell._is_lumped:
            spike = spike[..., self.cv_id]
        return (spike.reshape(-1) != 0).astype(jnp.uint32)

    def series(self, rcell: "Cell", times, rows) -> EventSeries:
        """Sparse rows in the format of ``NetworkResult.events[population]["spike"]``.

        ``rows`` is the loop's spike output: the steps' port values, or
        their packed grid, or ``(grid, count, groups, bytes)`` when
        ``capacity`` is set (:func:`_compact_spike_rows`).
        """
        if rcell._uses_reduction:
            values = np.asarray(rows)
            values = values.reshape(values.shape[0], -1)
            step, source_id = np.nonzero(values)
            count = values[step, source_id]
            n_rows = values.shape[1]
        else:
            n_rows = _spike_rows(rcell)
            step, source_id = self._packed_events(rows)
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

    def _packed_events(self, rows):
        """``(step, source_id)`` of the set bits of the loop's packed grid, in the grid's C order."""
        if self.capacity is not None:
            rows, count, groups, values = rows
            count = int(count)
            if count <= self.capacity:
                return _compacted_grid_events(np.asarray(groups)[:count], np.asarray(values)[:count], rows.shape[1])
        return _grid_events(np.asarray(rows))


def _spike_rows(rcell: "Cell") -> int:
    """The number of rows of a detailed or lumped Cell's spike port: its members, times its batch."""
    shape = rcell.spike.value.shape
    return int(np.prod(shape if rcell._is_lumped else shape[:-1], dtype=np.int64))


def _grid_events(rows):
    """``(step, row)`` of the set bits of the packed grid ``rows``, a ``(steps, bytes)`` array, in C order."""
    # Only the nonzero bytes are expanded to their eight rows. NumPy finds
    # the nonzero entries of a flat boolean array several times faster than
    # those of a 2-D byte array (CELL-39).
    flat = np.flatnonzero(rows != 0)
    return _set_bits(flat, rows.reshape(-1)[flat], rows.shape[1])


def _compacted_grid_events(groups, values, n_bytes: int):
    """:func:`_grid_events` of a grid of ``n_bytes`` columns from its nonzero 4-byte ``groups`` and their ``values``."""
    group, offset = np.nonzero(values)
    flat = groups[group].astype(np.int64) * 4 + offset
    return _set_bits(flat, values[group, offset], n_bytes)


def _set_bits(flat, values, n_bytes: int):
    """``(step, row)`` of the set bits of the bytes ``values`` at C-order positions ``flat`` of a packed grid."""
    step, byte = np.divmod(flat, n_bytes)
    hit, bit = np.nonzero(np.unpackbits(values[:, None], axis=1))
    return step[hit], byte[hit] * 8 + bit


def _on_gpu(value) -> bool:
    """Whether the array (or Quantity) ``value`` lives on a GPU."""
    devices = getattr(u.get_mantissa(value), "devices", None)
    return devices is not None and any(device.platform == "gpu" for device in devices())


def _spike_output_path(rcell: "Cell") -> tuple[bool, bool]:
    """``(blocked, compact)``: whether a run of ``rcell`` may pack its spikes in blocks and copy them compacted.

    Both on GPU and neither elsewhere, unless the module switches force
    them; a reduction model's port is not packed and takes neither.
    """
    on_gpu = False
    if None in (_BLOCKED_SPIKE_OUTPUT, _COMPACT_SPIKE_COPY) and not rcell._uses_reduction:
        on_gpu = _on_gpu(rcell.spike.value)
    blocked = on_gpu if _BLOCKED_SPIKE_OUTPUT is None else bool(_BLOCKED_SPIKE_OUTPUT)
    compact = on_gpu if _COMPACT_SPIKE_COPY is None else bool(_COMPACT_SPIKE_COPY)
    return blocked, compact


def _words_to_rows(words, block_steps: int):
    """The ``(blocks, rows)`` words of a blocked loop as its ``(steps, ceil(rows / 8))`` packed grid.

    Bit ``j`` of a row's word is the row's spike at step ``j`` of the
    block; each step's bytes are ``jnp.packbits`` of its rows, as a loop
    that packs at every step stacks them.
    """
    bits = (words[:, None, :] >> jnp.arange(block_steps, dtype=jnp.uint32)[:, None]) & jnp.uint32(1)
    packed = jnp.packbits(bits, axis=-1)
    return packed.reshape((-1, packed.shape[-1]))


def _compact_spike_rows(rows, capacity: int):
    """``(count, groups, values)``: the nonzero 4-byte groups of the packed grid ``rows``, on the device.

    The grid's bytes in C order make groups of four, the last padded with
    zeros. ``groups`` holds the indices of the first ``capacity`` nonzero
    groups in order and ``values`` their bytes; ``count`` is how many there
    are, so a count above ``capacity`` means the grid is to be copied whole.
    """
    flat = rows.reshape(-1)
    n_groups = -(-flat.shape[0] // 4)
    groups = jnp.pad(flat, (0, 4 * n_groups - flat.shape[0])).reshape((n_groups, 4))
    nonzero = jnp.any(groups != 0, axis=1)
    position = jnp.cumsum(nonzero, dtype=jnp.int32)
    slot = jnp.where(nonzero, position - 1, capacity)
    index = jnp.zeros(capacity, jnp.int32).at[slot].set(jnp.arange(n_groups, dtype=jnp.int32), mode="drop")
    return position[-1], index, groups[index]


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
    arrival_origin: tuple[float, int] | None = None,
    handover=None,
    spike_path: tuple[bool, bool] | None = None,
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
    ``arrival_origin`` (:func:`_arrival_origin`) is baked into the arrival
    steps; it is ``None`` unless the run changed ``dt`` off the new grid,
    and stays the same over the segments that follow at that ``dt``. So is
    ``handover`` (:func:`_dt_change_handover`), ``None`` unless a run at
    another ``dt`` came before. ``spike_path`` (:func:`_spike_output_path`)
    selects how the loop packs the spikes and copies them back; ``None`` is
    neither way.
    """
    cache = rcell._run_loop_cache
    key = (
        _time_quantity_cache_value(dt),
        int(n_steps),
        tuple((item.spec.name, item.schema.size) for item in compiled_recordings),
        int(brainstate.environ.get_precision()),
        schedule_key,
        bool(with_spikes),
        arrival_origin,
        handover,
        spike_path,
    )
    cached = cache.get(key)
    if cached is None:
        fixed_schedules = tuple(
            _has_fixed_shape_schedule(compiled, dt=dt, n_steps=n_steps) for compiled in compiled_recordings
        )
        unaligned = tuple(
            compiled for compiled in compiled_recordings if n_steps % _recording_steps(compiled.schema.period, dt) != 0
        )
        spike_port = None
        if with_spikes:
            spike_port = _SpikePort.of(rcell, n_steps=n_steps, path=spike_path or (False, False))
        cached = _RunPlan(
            loop=_make_run_loop(
                rcell,
                dt=dt,
                n_steps=n_steps,
                compiled_recordings=compiled_recordings,
                fixed_schedules=fixed_schedules,
                spike_port=spike_port,
                arrival_origin=arrival_origin,
                handover=handover,
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
    arrival_origin: tuple[float, int] | None = None,
    handover=None,
):
    """Create the jitted stateful run loop for a fixed observer layout.

    The loop takes the value inputs (``Cell._take_value_inputs``), which act
    on its first step, and returns
    ``(times, start_time, stop_time, recorded, spike_rows)``;
    ``spike_rows`` stacks ``spike_port.read`` over the steps, or is ``None``
    without a port. With ``spike_port.block_steps``, K, the loop runs blocks
    of K steps: the recordings and each block's words of spike bits are
    buffers carried through both loops and written in place, and the words
    become the same packed grid after the loop. With ``spike_port.capacity``
    the grid comes with its compacted copy (:func:`_compact_spike_rows`).
    ``recorded`` holds one ``(values, first_time)`` pair per recording: the
    selected samples for a fixed-shape schedule, or every step's values and
    ``None`` for a variable-length one (selected eagerly by the caller).
    ``arrival_origin`` is ``(origin in ms, its step)`` when the clock's
    origin is off the ``dt`` grid from time 0 (:func:`_arrival_origin`).
    ``handover`` holds what the runs at another ``dt`` before took
    (:func:`_dt_change_handover`).
    """
    from braincell.network.event import arrival_origin as arrivals_from, boundary_step, grid_step, handed_over
    from braincell.network.plasticity import reads_voltage_before_step

    n_recordings = len(compiled_recordings)
    period_steps = tuple(_recording_steps(compiled.schema.period, dt) for compiled in compiled_recordings)
    origin_ms, origin_step = (None, 0) if arrival_origin is None else arrival_origin
    block_steps = None if spike_port is None else spike_port.block_steps

    def _run_loop(values):
        with brainstate.environ.context(dt=dt), arrivals_from(origin_ms, origin_step), handed_over(handover):
            origin, step = rcell._clock_start(dt)
            start_t = origin + step * dt
            times = origin + (step + jnp.arange(n_steps, dtype=jnp.int32)) * dt
            if origin_ms is None:
                # The steps on the dt grid from time 0, which the event
                # arrivals are compared with (CELL-46).
                first = grid_step(origin, step, dt)
            else:
                # The clock counts from an origin off that grid; so do the
                # arrivals, from its step (CELL-48).
                first = (step + origin_step).astype(jnp.int32)
            grid = first + jnp.arange(n_steps, dtype=jnp.int32)

            def _step(t, s):
                with brainstate.environ.context(t=t), boundary_step(t, s):
                    with jax.named_scope("braincell:cell_run:prepare_clamps"):
                        rcell._prepare_step_clamps(t=t, dt=brainstate.environ.get_dt())
                    with jax.named_scope("braincell:cell_run:sample_recordings"):
                        pre_recording_snapshot = {
                            index: item.sample()
                            for index, item in enumerate(compiled_recordings)
                            if item.phase == "pre"
                        }
                    # The step prepares the boundary it starts at, as a
                    # Network step does, so an edit between runs (a weight
                    # set, a new dt) acts from the next run's first boundary
                    # (CELL-55).
                    with jax.named_scope("braincell:cell_run:prepare_synapse_inputs"):
                        rcell._prepare_next_synapse_inputs(t=t, step=s)
                    with jax.named_scope("braincell:cell_run:begin_step"):
                        rcell._begin_step()
                    with jax.named_scope("braincell:cell_run:update_dynamics"):
                        with brainstate.environ.context(_braincell_step_clamps_prepared=True):
                            rcell._update_dynamics()
                    with jax.named_scope("braincell:cell_run:route_live_connections"):
                        rcell._apply_direct_live_connection_events()
                    with jax.named_scope("braincell:cell_run:sample_outputs"):
                        recording_snapshot = tuple(
                            pre_recording_snapshot[index] if item.phase == "pre" else item.sample()
                            for index, item in enumerate(compiled_recordings)
                        )
                    spikes = ()
                    if spike_port is not None:
                        with jax.named_scope("braincell:cell_run:sample_spikes"):
                            read = spike_port.read if block_steps is None else spike_port.read_bits
                            spikes = (read(rcell),)
                return recording_snapshot + spikes

            prepared = rcell._next_synapse_inputs_prepared
            previous_V = None if reads_voltage_before_step(rcell) else getattr(rcell, "_event_previous_V", None)

            def _carried_step(index, t, s):
                # Two states are rewritten by every step before anything reads
                # them, so the loop need not carry them from step to step; on
                # GPU each carried rewrite costs a device copy per step
                # (CELL-39). The prepared-boundary flag ends every step as it
                # began, false: each step prepares its boundary and consumes
                # it. The voltage before the step, which a VoltageCrossingSource
                # reads within the step, leaves the loop from its last step
                # only; a rule's post detector reads it at the next step's
                # boundary, so the loop then carries it (CELL-55).
                flag = prepared.value
                previous = None if previous_V is None else previous_V.value
                outputs = _step(t, s)
                prepared.value = flag
                if previous_V is not None:
                    previous_V.value = u.math.where(index == n_steps - 1, previous_V.value, previous)
                return outputs

            step_fn = _carried_step
            if any(values):

                def step_fn(index, t, s):
                    # The value inputs act on the first step only.
                    with rcell._value_input_step(values, first=index == 0):
                        return _carried_step(index, t, s)

            if block_steps is None:
                traces_over_time = brainstate.transform.for_loop(step_fn, jnp.arange(n_steps), times, grid)
            else:
                # The buffers take the shapes of the recordings' samples,
                # read once here; the reads are dead code after tracing.
                with brainstate.environ.context(t=start_t), boundary_step(start_t, first):
                    samples = tuple(item.sample() for item in compiled_recordings)
                traces_over_time = _blocked_loop(
                    step_fn, samples, times, grid, block_steps=block_steps, n_rows=_spike_rows(rcell)
                )
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
            if spike_port.capacity is not None:
                spike_rows = (spike_rows,) + _compact_spike_rows(spike_rows, spike_port.capacity)
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

    return brainstate.transform.jit(axial_operator_before_steps(_run_loop, (rcell,)))


def _blocked_loop(step_fn, samples, times, grid, *, block_steps: int, n_rows: int):
    """``step_fn(index, t, s)``'s recordings and packed spike grid over the steps, in blocks of ``block_steps``.

    ``step_fn`` returns a step's samples followed by its spike bits, one
    for each of ``n_rows`` (``_SpikePort.read_bits``); ``samples`` are
    samples of the same shapes.
    An outer loop over the blocks runs an inner loop over a block's steps.
    Both carry the recordings, ``(steps, ...)`` buffers each step writes its
    samples into at its step, and the ``(blocks, rows)`` words each block
    writes its word of each row into once; each step ORs its bit into bit
    ``j`` of the row's word, ``j`` its step in the block. On GPU each step
    then launches the kernels of a loop without spike output: in place,
    ``dynamic_update_index_in_dim`` adds none (``.at[].set`` would check
    its index with one more), and nothing is stacked or copied per block.
    Returns the recordings and the grid of :func:`_words_to_rows`.
    """
    n_steps = times.shape[0]
    n_blocks = n_steps // block_steps

    def write(buffers, values, at):
        return jax.tree.map(lambda b, v: jax.lax.dynamic_update_index_in_dim(b, v, at, 0), buffers, values)

    def in_block(carry, x):
        buffers, word = carry
        index, t, s = x
        *values, bit = step_fn(index, t, s)
        word = word | (bit << (index % block_steps).astype(jnp.uint32))
        return (write(buffers, tuple(values), index), word), None

    def over_blocks(carry, x):
        buffers, words = carry
        (buffers, word), _ = brainstate.transform.scan(in_block, (buffers, jnp.zeros_like(words[0])), x)
        return (buffers, write(words, word, x[0][0] // block_steps)), None

    buffers = jax.tree.map(lambda a: jnp.zeros((n_steps,) + jnp.shape(a), jnp.result_type(a)), samples)
    words = jnp.zeros((n_blocks, n_rows), jnp.uint32)
    blocks = tuple(a.reshape((n_blocks, block_steps)) for a in (jnp.arange(n_steps), times, grid))
    (buffers, words), _ = brainstate.transform.scan(over_blocks, (buffers, words), blocks)
    return buffers + (_words_to_rows(words, block_steps),)


def axial_operator_before_steps(loop, cells):
    """``loop``, building the axial operator of each of ``cells`` once, before the steps it traces.

    The geometry is the same at every step of a run; built inside the step,
    the operator would be built again at every step, and at every stage of
    the generic solvers (CELL-50). See ``axial_operator_per_run``.
    """

    @functools.wraps(loop)
    def built_once(*args):
        with contextlib.ExitStack() as stack:
            for cell in cells:
                stack.enter_context(axial_operator_per_run(cell))
            return loop(*args)

    return built_once


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
