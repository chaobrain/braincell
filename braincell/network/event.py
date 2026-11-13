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

"""Logical event-source owners, views, and scheduled event declarations.

The event-input *contracts* a target mechanism declares
(:class:`~braincell.mech.EventInput` and friends) live in
:mod:`braincell.mech._event_contract`, so that the channel and synapse
base classes can depend on them without reaching into this package.
"""

from abc import ABC
import contextlib
from dataclasses import dataclass, field
import hashlib
from typing import Any, NamedTuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from braincell._misc import is_traced_value
from braincell._parameter_schema import RuntimeParameterState

__all__ = [
    "EventSource",
    "EventSourceView",
    "EventTable",
    "EventSequence",
    "NetStim",
    "VoltageCrossingSource",
]

#: A NetStim schedule with at most this many entries (``size * max(number)``)
#: is generated in full at construction; a larger one on demand.
_EAGER_SCHEDULE_ENTRIES = 1 << 20

#: ``brainstate.environ`` key carrying the last event time a run can deliver.
_SCHEDULE_HORIZON_KEY = "_braincell_netstim_horizon_ms"

#: ``brainstate.environ`` key carrying ``(t, step)``: the integer step of the
#: boundary ``t`` a run loop set (:func:`boundary_step`).
_BOUNDARY_STEP_KEY = "_braincell_boundary_step"

#: ``brainstate.environ`` key carrying ``(origin in ms, step)``: the arrival
#: steps count from that time, the boundary numbered ``step`` (:func:`arrival_origin`).
_ARRIVAL_ORIGIN_KEY = "_braincell_arrival_origin"

#: ``brainstate.environ`` key carrying the :class:`ArrivalHandover` of a run
#: that takes over from a run at another ``dt`` (:func:`handed_over`).
_HANDOVER_KEY = "_braincell_arrival_handover"

#: Arrival steps past the int32 step count, which no run reaches, are clipped to it.
_MAX_STEP = 2**31 - 1

#: The arrival step of an event a run never delivers: an earlier run took it (CELL-57).
TAKEN_STEP = -1


class EventSource(ABC):
    """Own a one-dimensional collection of logical event-output endpoints.

    Concrete sources may store schedules, threshold-detector declarations, or
    bindings to another model's live output. Indexing always returns the same
    lightweight :class:`EventSourceView` type.
    """

    @property
    def ids(self) -> np.ndarray:
        """Return stable source-local identifiers."""
        return np.arange(self.size, dtype=np.int64)

    @property
    def source_type(self) -> str:
        """Return the stable public type used by connection queries."""
        return type(self).__name__

    @property
    def source_name(self) -> str | None:
        """Return the explicit semantic source name, when one exists."""
        return getattr(self, "name", None)

    @property
    def is_scheduled(self) -> bool:
        """Whether all event times are known before runtime."""
        return False

    @property
    def execution_owner(self):
        """Return the runtime object that must be advanced with this source."""
        return None

    @property
    def view(self) -> "EventSourceView":
        """Return a full view over this source."""
        return EventSourceView(self, self.ids)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, selector) -> "EventSourceView":
        return self.view[selector]

    def trainable(self, **fields):
        """Bind threshold parameter sources to all detector endpoints.

        Parameters
        ----------
        **fields
            ``threshold`` mapped to a trainable source.

        Returns
        -------
        EventSource
            This detector.
        """
        self.view.trainable(**fields)
        return self

    def event_count(self, source_index, *, t, delay, dt):
        """Read live event counts; scheduled sources override this method."""
        _ = t, dt
        if not isinstance(delay, u.Quantity):
            raise TypeError("Connection delay must be a time quantity.")
        delay_ms = np.asarray(delay.to_decimal(u.ms), dtype=float)
        if np.any(delay_ms != 0.0):
            raise NotImplementedError(
                f"Delayed delivery from live {type(self).__name__} sources is not implemented yet."
            )
        current = getattr(self, "current_event_count", None)
        if current is None:
            raise NotImplementedError(f"{type(self).__name__} does not implement live event counts.")
        return current(np.asarray(source_index, dtype=np.int64))


class EventSourceView:
    """Duplicate-preserving ordered selection from one EventSource owner."""

    __slots__ = ("_owner", "_source_ids")

    def __init__(self, owner: EventSource, source_ids) -> None:
        if not isinstance(owner, EventSource):
            raise TypeError(f"EventSourceView.owner must be EventSource, got {type(owner).__name__!s}.")
        values = np.asarray(source_ids)
        if values.dtype.kind not in "iu" or values.dtype.kind == "b":
            raise TypeError("EventSourceView.source_ids must contain integers.")
        values = values.astype(np.int64, copy=False).reshape(-1)
        if np.any(values < 0) or np.any(values >= owner.size):
            raise IndexError("EventSourceView source index is out of range.")
        self._owner = owner
        self._source_ids = np.array(values, copy=True)

    @property
    def owner(self) -> EventSource:
        """Return the source object that owns the selected endpoints."""
        return self._owner

    @property
    def root(self) -> "EventSourceView":
        """Return a full view over the source owner."""
        return self._owner.view

    @property
    def source_id(self) -> np.ndarray:
        """Return selected stable source-local IDs."""
        return np.array(self._source_ids, copy=True)

    def __len__(self) -> int:
        return int(self._source_ids.size)

    def trainable(self, **fields):
        """Bind threshold parameter sources to selected detector endpoints.

        Parameters
        ----------
        **fields
            ``threshold`` mapped to a trainable source.

        Returns
        -------
        EventSourceView
            This selection.
        """
        from braincell.trainable._targets import register_detector

        register_detector(self, fields)
        return self

    def __getitem__(self, selector) -> "EventSourceView":
        selected = self._source_ids[selector]
        return EventSourceView(self._owner, np.asarray(selected).reshape(-1))

    def __repr__(self) -> str:
        return (
            f"EventSourceView(source={type(self._owner).__name__}, size={len(self)}, ids={self._source_ids.tolist()!r})"
        )


@dataclass(frozen=True)
class EventTable:
    """Canonical flat scheduled-event rows."""

    source_index: Any
    time: Any
    event_id: Any | None = None

    def __post_init__(self) -> None:
        source_index = np.asarray(self.source_index)
        if source_index.ndim != 1 or source_index.dtype.kind not in "iu" or source_index.dtype.kind == "b":
            raise TypeError("EventTable.source_index must be a one-dimensional integer array.")
        if not isinstance(self.time, u.Quantity):
            raise TypeError("EventTable.time must be a time quantity.")
        try:
            time_ms = np.asarray(self.time.to_decimal(u.ms), dtype=np.float64)
        except Exception as exc:
            raise ValueError("EventTable.time must have time dimensions.") from exc
        if time_ms.shape != source_index.shape:
            raise ValueError(
                "EventTable source_index and time must have the same shape; "
                f"got {source_index.shape!r} and {time_ms.shape!r}."
            )
        if np.any(source_index < 0):
            raise ValueError("EventTable.source_index entries must be >= 0.")
        if np.any(~np.isfinite(time_ms)) or np.any(time_ms < 0.0):
            raise ValueError("EventTable.time entries must be finite and >= 0 ms.")
        if self.event_id is None:
            event_id = np.arange(source_index.size, dtype=np.int64)
        else:
            event_id = np.asarray(self.event_id)
            if event_id.shape != source_index.shape or event_id.dtype.kind not in "iu" or event_id.dtype.kind == "b":
                raise TypeError("EventTable.event_id must be an integer array aligned with source_index.")
            event_id = event_id.astype(np.int64, copy=False)
            if len(np.unique(event_id)) != len(event_id):
                raise ValueError("EventTable.event_id entries must be unique.")
        object.__setattr__(self, "source_index", np.array(source_index, dtype=np.int64, copy=True))
        object.__setattr__(self, "time", u.Quantity(np.array(time_ms, copy=True), u.ms))
        object.__setattr__(self, "event_id", np.array(event_id, copy=True))

    def __len__(self) -> int:
        return int(self.source_index.size)


@dataclass(frozen=True)
class EventSequence(EventSource):
    """A population of sources backed by an explicit flat event table."""

    size: int
    events: EventTable
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, (int, np.integer)):
            raise TypeError(f"EventSequence.size must be an integer, got {self.size!r}.")
        size = int(self.size)
        if size < 1:
            raise ValueError(f"EventSequence.size must be >= 1, got {size!r}.")
        if not isinstance(self.events, EventTable):
            raise TypeError(f"EventSequence.events must be EventTable, got {type(self.events).__name__!s}.")
        if len(self.events) and int(np.max(self.events.source_index)) >= size:
            raise IndexError("EventSequence event source index is out of range.")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("EventSequence.name must be a non-empty string or None.")
        object.__setattr__(self, "size", size)

    @classmethod
    def from_times(cls, times, *, name: str | None = None) -> "EventSequence":
        """Build from one possibly ragged sequence of event times per source."""
        rows = tuple(times)
        source_index: list[int] = []
        time_ms: list[float] = []
        for source_id, row in enumerate(rows):
            if not isinstance(row, u.Quantity):
                raise TypeError("EventSequence.from_times rows must be time quantities.")
            values = np.asarray(row.to_decimal(u.ms), dtype=np.float64).reshape(-1)
            source_index.extend([source_id] * len(values))
            time_ms.extend(values.tolist())
        return cls(
            size=len(rows),
            events=EventTable(source_index=source_index, time=np.asarray(time_ms) * u.ms),
            name=name,
        )

    @property
    def instance_name(self) -> str:
        return self.name if self.name is not None else "EventSequence"

    @property
    def is_scheduled(self) -> bool:
        return True

    def event_count(self, source_index, *, t, delay, dt):
        """Count delayed events assigned to the nearest fixed-step boundary.

        An event at ``time`` arrives at ``time + delay`` and is delivered at
        step ``t`` when the arrival lies in ``[t - dt / 2, t + dt / 2)``, so
        an exact half-step tie goes to the later boundary. Ties and
        NEURON's fixed-step delivery are as in :meth:`NetStim.event_count`.

        The arrival step is ``(time + delay) / dt`` rounded at float64 on
        the host, from the float64 event times and delays, so a float32 run
        delivers each event at the step a float64 run does (CELL-46). Inside
        ``Cell.run``, ``Network.run`` and ``Network.update()`` it is compared
        with the loop's integer step; after ``Cell.run`` changes ``dt``, both
        count from the time reached (CELL-48). A run after one at another
        ``dt`` skips the events that run took, and delivers the others no
        earlier than its first boundary that run did not prepare (CELL-57).
        For a ``t`` the caller chose, the step is ``t / dt`` rounded in the
        run precision. A ``delay`` or ``dt`` that is traced (an argument of
        a ``jax.jit`` function) has no host value; its arrivals are then
        rounded in the run precision too.
        """
        source_index = np.asarray(source_index, dtype=np.int64).reshape(-1)
        plan = _ArrivalPlan.build(
            event_source=self.events.source_index,
            event_time_ms=np.asarray(self.events.time.to_decimal(u.ms), dtype=np.float64),
            source_index=source_index,
            t=t,
            delay=delay,
            dt=dt,
            one_dimensional=True,
        )
        if plan is not None:
            return plan.count(t=t, dt=dt)
        return _flat_event_count(self.events, source_index=source_index, t=t, delay=delay, dt=dt)


@dataclass(frozen=True)
class NetStim(EventSource):
    """A population of deterministic or noisy artificial spike sources.

    Parameters
    ----------
    size : int, optional
        Number of independent sources. Defaults to one.
    start : Quantity[ms], optional
        Most likely first event time, scalar or one value per source. With
        nonzero ``noise``, the realized first event occurs after an
        exponential waiting time with mean ``noise * interval``.
    number : int or array-like of int, optional
        Event count, scalar or one value per source. It may be very large,
        as in NEURON's ``number = 1e9`` for an unlimited train; see Notes.
    interval : Quantity[ms], optional
        Mean inter-event interval, scalar or one value per source.
    noise : float or array-like, optional
        Fraction of each interval drawn from an exponential distribution.
        Zero is periodic and one is Poisson-like.
    seed : int or None, optional
        Seed of the noise. ``None`` draws one integer from the BrainState
        default random stream when the source is built, so an enclosing
        ``brainstate.random.seed_context(s)`` (or an earlier
        ``brainstate.random.seed(s)``) controls the noise. Only a source
        with nonzero ``noise`` draws; a periodic one leaves the default
        stream alone. Registered in a ``Network``, a source without a seed
        uses a stream derived from the Network's root and its population
        name instead, but the draw made when it was built has still
        advanced the default stream. Inside ``brainstate.transform.jit``,
        ``for_loop`` or another transformation the default stream is
        traced, and a noisy source without a seed raises ``TypeError``:
        build it outside, or pass an integer.
    name : str or None, optional
        Optional display name.

    Notes
    -----
    Every source has its own noise stream: draw ``j`` of source ``i``
    depends only on the seed, ``i`` and ``j``. A source's events therefore
    do not depend on how many other sources there are or on their
    parameters, a larger ``number`` keeps the earlier events, and the
    events are the same at every float precision. The schedule is a
    constant: every batch member of a ``Cell`` receives the same events.

    A schedule with at most ``2**20`` entries (``size * max(number)``) is
    generated in full at construction. A larger one is generated on demand:
    ``Cell.run`` and ``Network.run`` generate events only as far as the end
    of the run, so their memory does not grow with ``number``; a later run
    extends the schedule, at least doubling its width so that a compiled
    run is rarely rebuilt. A shorter schedule is a prefix of the longer
    one, so the events are those of the full schedule. Entries that do not
    know when the run ends generate all ``number`` events, as for a small
    schedule: :attr:`event_times`, :attr:`events`, :meth:`event_count`
    outside a run, ``Cell.run`` traced under ``jax.jit`` or ``jax.grad``,
    and ``Network.prepare_run`` / ``Network.update``.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> import braincell
    >>> with brainstate.random.seed_context(42):
    ...     first = braincell.NetStim(size=2, number=3, interval=5.0 * u.ms, noise=1.0)
    >>> with brainstate.random.seed_context(42):
    ...     again = braincell.NetStim(size=2, number=3, interval=5.0 * u.ms, noise=1.0)
    >>> bool(u.math.allclose(first.event_times, again.event_times))
    True
    >>> longer = braincell.NetStim(size=3, number=5, interval=5.0 * u.ms, noise=1.0, seed=7)
    >>> shorter = braincell.NetStim(size=2, number=3, interval=5.0 * u.ms, noise=1.0, seed=7)
    >>> bool(u.math.allclose(longer.event_times[:2, :3], shorter.event_times))
    True
    """

    size: int = 1
    start: Any = field(default_factory=lambda: 0.0 * u.ms)
    number: Any = 1
    interval: Any = field(default_factory=lambda: 10.0 * u.ms)
    noise: Any = 0.0
    seed: int | None = None
    name: str | None = None
    _event_times_ms: np.ndarray = field(init=False, repr=False, compare=False)
    _event_mask: np.ndarray = field(init=False, repr=False, compare=False)
    _network_binding: tuple[int, str] | None = field(init=False, default=None, repr=False, compare=False)
    # Generation state: the effective seed and whether ``_event_times_ms``
    # holds every event.
    _schedule_seed: int = field(init=False, default=0, repr=False, compare=False)
    _schedule_complete: bool = field(init=False, default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, (int, np.integer)):
            raise TypeError(f"NetStim.size must be an integer, got {self.size!r}.")
        size = int(self.size)
        if size < 1:
            raise ValueError(f"NetStim.size must be >= 1, got {size!r}.")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer))):
            raise TypeError(f"NetStim.seed must be an integer or None, got {self.seed!r}.")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"NetStim.name must be a non-empty string or None, got {self.name!r}.")

        start = _quantity_vector(self.start, unit=u.ms, size=size, name="NetStim.start")
        interval = _quantity_vector(self.interval, unit=u.ms, size=size, name="NetStim.interval")
        number = _integer_vector(self.number, size=size, name="NetStim.number")
        noise = _real_vector(self.noise, size=size, name="NetStim.noise")
        start_ms = np.asarray(start.to_decimal(u.ms), dtype=np.float64)
        interval_ms = np.asarray(interval.to_decimal(u.ms), dtype=np.float64)
        if np.any(start_ms < 0.0):
            raise ValueError("NetStim.start must be >= 0 ms.")
        if np.any(interval_ms <= 0.0):
            raise ValueError("NetStim.interval must be > 0 ms.")
        if np.any(number < 0):
            raise ValueError("NetStim.number must be >= 0.")
        if np.any((noise < 0.0) | (noise > 1.0)):
            raise ValueError("NetStim.noise must be within [0, 1].")

        object.__setattr__(self, "size", size)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "noise", noise)
        if self.seed is not None:
            seed = int(self.seed)
        elif np.any(noise > 0.0):
            from braincell._random import draw_seed

            seed = draw_seed("NetStim(seed=None) with noise")
        else:
            # A periodic schedule draws nothing; leave the default stream alone.
            seed = 0
        self._reset_schedule(seed)

    @property
    def instance_name(self) -> str:
        """Display label for this source population."""
        return self.name if self.name is not None else "NetStim"

    @property
    def is_scheduled(self) -> bool:
        return True

    @property
    def event_times(self) -> u.Quantity:
        """Padded continuous event-time matrix with shape ``(size, max(number))``.

        Reading it generates every event; see the class Notes.
        """
        self._ensure_schedule()
        return u.Quantity(self._event_times_ms, u.ms)

    def event_count(self, source_index, *, t, delay, dt):
        """Count delayed events assigned to the nearest fixed-step boundary.

        An arrival is delivered at step ``t`` when it lies in the half-open
        interval ``[t - dt / 2, t + dt / 2)``, so an exact half-step tie goes
        to the later boundary; a step ratio within four ulp of a half counts
        as a tie. Away from ties this is the step NEURON's fixed-step
        delivery picks. At an exact tie NEURON's choice follows its
        accumulated ``t``: arrivals at 0.0125, 5.0375 and 8.0125 ms with
        ``dt = 0.025 ms`` land one step earlier in NEURON 9.0 than here.

        Inside ``Cell.run`` and ``Network.run`` it reads the events up to
        the end of the run; elsewhere it generates every event (see the
        class Notes). With a fixed ``delay`` and ``dt`` the arrival steps
        are computed once, when a compiled loop is traced, and each step
        looks its arrivals up in them.

        The arrival step is ``(time + delay) / dt`` rounded at float64 on
        the host, from the float64 event times and delays, so a float32 run
        delivers each event at the step a float64 run does (CELL-46). Inside
        ``Cell.run``, ``Network.run`` and ``Network.update()`` it is compared
        with the loop's integer step; after ``Cell.run`` changes ``dt``, both
        count from the time reached (CELL-48). A run after one at another
        ``dt`` skips the events that run took, and delivers the others no
        earlier than its first boundary that run did not prepare (CELL-57).
        For a ``t`` the caller chose, the step is ``t / dt`` rounded in the
        run precision. A ``delay`` or ``dt`` that is traced (an argument of
        a ``jax.jit`` function) has no host value; its arrivals are then
        rounded in the run precision too.
        """
        self._ensure_schedule(until_ms=brainstate.environ.get(_SCHEDULE_HORIZON_KEY, None))
        source_index = np.asarray(source_index, dtype=np.int32)
        event_source, event_column = np.nonzero(self._event_mask)
        plan = _ArrivalPlan.build(
            event_source=event_source,
            event_time_ms=self._event_times_ms[event_source, event_column],
            # Index as the matrix below does: negative entries wrap, others raise.
            source_index=np.arange(self.size)[source_index],
            t=t,
            delay=delay,
            dt=dt,
        )
        if plan is not None:
            return plan.count(t=t, dt=dt)
        mask = self._event_mask[source_index]
        times = np.where(mask, self._event_times_ms[source_index], 0.0)
        arrival_steps = _matrix_arrival_steps(times, delay, dt, rows=source_index.shape)
        selected = jnp.asarray(mask) & (arrival_steps == _boundary_step(t, dt))
        return u.math.sum(selected, axis=-1)

    @property
    def events(self) -> EventTable:
        """Return the generated schedule as canonical flat event rows.

        Reading it generates every event; see the class Notes.
        """
        return self._event_table()

    def _event_table(self, until_ms: float | None = None) -> EventTable:
        """Flat rows of the generated events, at least those up to ``until_ms``."""
        self._ensure_schedule(until_ms=until_ms)
        source_index, event_index = np.nonzero(self._event_mask)
        return EventTable(
            source_index=source_index,
            time=self._event_times_ms[source_index, event_index] * u.ms,
        )

    @property
    def _schedule_key(self) -> tuple[int, int]:
        """Cache token of the generated events: effective seed and width."""
        return self._schedule_seed, int(self._event_times_ms.shape[1])

    def _reset_schedule(self, seed: int) -> None:
        """Set the effective seed; generate a small schedule in full now."""
        max_events = int(np.max(self.number, initial=0))
        object.__setattr__(self, "_schedule_seed", int(seed))
        if self.size * max_events <= _EAGER_SCHEDULE_ENTRIES:
            event_times, event_mask = _build_event_schedule(
                start_ms=np.asarray(self.start.to_decimal(u.ms), dtype=np.float64),
                number=self.number,
                interval_ms=np.asarray(self.interval.to_decimal(u.ms), dtype=np.float64),
                noise=self.noise,
                seed=int(seed),
            )
            complete = True
        else:
            event_times = np.empty((self.size, 0), dtype=np.float64)
            event_mask = np.zeros((self.size, 0), dtype=bool)
            complete = False
        object.__setattr__(self, "_event_times_ms", event_times)
        object.__setattr__(self, "_event_mask", event_mask)
        object.__setattr__(self, "_schedule_complete", complete)

    def _ensure_schedule(self, until_ms: float | None = None) -> None:
        """Generate at least every event up to ``until_ms`` (all when ``None``).

        A schedule only grows, by at least doubling its width, so a compiled
        run that baked in an earlier width is reused when it still covers.
        """
        if self._schedule_complete:
            return
        max_events = int(np.max(self.number, initial=0))
        width = int(self._event_times_ms.shape[1])
        if until_ms is None:
            width = max_events
        elif self._covers(until_ms):
            return
        else:
            width = max(_estimated_width(self, until_ms), 2 * width)
        while True:
            width = min(width, max_events)
            event_times, event_mask = _build_event_schedule(
                start_ms=np.asarray(self.start.to_decimal(u.ms), dtype=np.float64),
                number=self.number,
                interval_ms=np.asarray(self.interval.to_decimal(u.ms), dtype=np.float64),
                noise=self.noise,
                seed=self._schedule_seed,
                width=width,
            )
            object.__setattr__(self, "_event_times_ms", event_times)
            object.__setattr__(self, "_event_mask", event_mask)
            object.__setattr__(self, "_schedule_complete", width == max_events)
            if self._schedule_complete or self._covers(until_ms):
                return
            width *= 2

    def _covers(self, until_ms: float) -> bool:
        """Whether every source has all its events, or one after ``until_ms``."""
        width = int(self._event_times_ms.shape[1])
        if width == 0:
            return False
        pending = self.number > width
        return bool(np.all(self._event_times_ms[pending, width - 1] > until_ms))

    def _bind_network_seed(self, network_seed: int, population_name: str) -> None:
        """Bind an implicit seed to a canonical, order-independent Network path."""
        binding = (int(network_seed), str(population_name))
        if self._network_binding is not None and self._network_binding != binding:
            raise RuntimeError("One NetStim cannot be bound to multiple Network population paths.")
        object.__setattr__(self, "_network_binding", binding)
        if self.seed is not None:
            return
        digest = hashlib.blake2b(
            f"braincell.netstim\0{binding[0]}\0{binding[1]}".encode("utf-8"),
            digest_size=8,
        ).digest()
        effective_seed = int.from_bytes(digest, "little") & 0x7FFF_FFFF
        self._reset_schedule(effective_seed)


class _CellVoltageThresholdDefault:
    def __repr__(self) -> str:
        return "<Cell.V_th>"


_CELL_VOLTAGE_THRESHOLD = _CellVoltageThresholdDefault()


class VoltageCrossingSource(EventSource):
    """Detect voltage threshold crossings at one or more locations per cell.

    Omitting ``threshold`` uses the Cell's possibly heterogeneous ``V_th``.
    Omitting ``location`` selects the root branch midpoint. Endpoint rows are
    ordered population-major, preserving the evaluated location order and any
    duplicate locations.

    Parameters
    ----------
    cells : Cell or CellView
        Cell endpoints whose voltages are observed.
    location : location expression, optional
        Morphology locations; defaults to the root midpoint. An expression
        such as ``at("axon", 0.5)`` follows edits of the Cell's morphology.
        A resolved ``LocsetMask`` holds default branch indices: they stay
        valid when branches are appended, but once a branch is removed or
        moved, :attr:`cv_id` raises ``RuntimeError`` rather than watch
        another branch. An expression that names its branch by index, such
        as ``at(1, 0.5)``, does not follow such an edit either: :attr:`cv_id`
        raises ``ValueError``. A Cell with a connection from such a detector
        raises the same error at ``init_state``, before initializing.
    threshold : brainunit.Quantity, optional
        Explicit voltage threshold, or the owning Cell's V_th when omitted.
    direction : str, optional
        ``rising`` uses last < threshold <= next; ``falling`` reverses both
        comparisons. Staying at equality never repeats an event.
    spk_fun : callable or None, optional
        Hard Heaviside forward function (one at zero) with surrogate derivative.
        Defaults to the Cell's spk_fun. Direction only changes its input sign.
    name : str or None, optional
        Source name.
    """

    __slots__ = (
        "cells",
        "location",
        "direction",
        "name",
        "_threshold",
        "_uses_cell_threshold",
        "_population_indices",
        "_location_indices",
        "_cv_ids",
        "_grid_generation",
        "_index_basis",
        "spk_fun",
        "_training_id",
    )

    def __init__(
        self,
        cells,
        *,
        location=None,
        threshold=_CELL_VOLTAGE_THRESHOLD,
        direction: str = "rising",
        spk_fun=None,
        name: str | None = None,
    ) -> None:
        from braincell.filter import RootLocation
        from braincell.filter._index_refs import held_branch_indices

        root = getattr(cells, "root", cells)
        population_indices = getattr(cells, "population_indices", None)
        if population_indices is None:
            population_size = 1 if len(getattr(root, "pop_size", ())) == 0 else int(root.pop_size[0])
            population_indices = tuple(range(population_size))
        if location is None:
            location = RootLocation(0.5)
        if not hasattr(location, "evaluate") and not (hasattr(location, "branch_id") and hasattr(location, "branch_x")):
            raise TypeError("VoltageCrossingSource.location must be a location expression or resolved locset mask.")
        location_cv_ids = _resolve_detector_location_cvs(root, location)
        if location_cv_ids.size == 0:
            raise ValueError("VoltageCrossingSource.location must resolve to at least one morphology location.")
        if direction not in {"rising", "falling"}:
            raise ValueError("VoltageCrossingSource.direction must be 'rising' or 'falling'.")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("VoltageCrossingSource.name must be a non-empty string or None.")

        population_indices = np.asarray(population_indices, dtype=np.int64).reshape(-1)
        n_population = int(population_indices.size)
        n_location = int(location_cv_ids.size)
        uses_cell_threshold = threshold is _CELL_VOLTAGE_THRESHOLD
        if uses_cell_threshold:
            normalized_threshold = None
        else:
            if not isinstance(threshold, u.Quantity):
                raise TypeError("VoltageCrossingSource.threshold must be a voltage quantity when provided.")
            try:
                threshold_mv = np.asarray(threshold.to_decimal(u.mV), dtype=np.float64)
            except Exception as exc:
                raise ValueError("VoltageCrossingSource.threshold must have voltage dimensions.") from exc
            if threshold_mv.shape == (n_population,):
                threshold_mv = threshold_mv[:, None]
            elif threshold_mv.shape == (n_location,):
                threshold_mv = threshold_mv[None, :]
            elif threshold_mv.shape == (n_population * n_location,):
                threshold_mv = threshold_mv.reshape(n_population, n_location)
            try:
                threshold_mv = np.broadcast_to(threshold_mv, (n_population, n_location))
            except ValueError as exc:
                raise ValueError(
                    "VoltageCrossingSource.threshold must be scalar or broadcast to "
                    f"(population, location) shape {(n_population, n_location)!r}; got {threshold_mv.shape!r}."
                ) from exc
            normalized_threshold = u.Quantity(np.array(threshold_mv, copy=True).reshape(-1), u.mV)

        self.cells = root
        self.location = location
        self.direction = direction
        if spk_fun is not None and not callable(spk_fun):
            raise TypeError("VoltageCrossingSource.spk_fun must be callable or None.")
        self.spk_fun = spk_fun
        self.name = name
        self._threshold = None if uses_cell_threshold else RuntimeParameterState(normalized_threshold)
        self._training_id = root._next_detector_id
        root._next_detector_id += 1
        self._uses_cell_threshold = uses_cell_threshold
        self._population_indices = np.repeat(population_indices, n_location)
        self._location_indices = np.tile(np.arange(n_location, dtype=np.int64), n_population)
        self._cv_ids = np.tile(location_cv_ids, n_population)
        self._grid_generation = root._view_generation
        # A resolved mask holds default branch indices of the Cell's tree at
        # this revision, and so does an expression that names branches by
        # index (MORPH-17); any other expression is evaluated again on every
        # refresh and follows edits.
        holds_indices = not hasattr(location, "evaluate") or held_branch_indices(location)
        self._index_basis = root._index_basis() if holds_indices else None

    @property
    def size(self) -> int:
        return int(self._population_indices.size)

    @property
    def population_index(self) -> np.ndarray:
        return np.array(self._population_indices, copy=True)

    @property
    def location_index(self) -> np.ndarray:
        """Return the evaluated-location row for each source endpoint."""
        return np.array(self._location_indices, copy=True)

    @property
    def threshold(self):
        """Return the explicit threshold, or the Cell threshold when omitted."""
        return self.cells.V_th if self._uses_cell_threshold else self._threshold.value

    @property
    def instance_name(self) -> str:
        return self.name if self.name is not None else "VoltageCrossingSource"

    @property
    def execution_owner(self):
        """Return the Cell whose voltage drives this detector."""
        return self.cells

    @property
    def _reads_previous_voltage(self) -> bool:
        """Whether :meth:`current_event_count` compares the voltage before a step with the voltage after it.

        A rising detector at the Cell's threshold, with the Cell's spike
        function, reads the Cell's spike instead.
        """
        return not (self._uses_cell_threshold and self.direction == "rising" and self.spk_fun is None)

    @property
    def cv_id(self) -> np.ndarray:
        """Return the containing CV for each source endpoint."""
        # A detector is a continuous location declaration, not a saved CV View.
        _ = self.cells._discretization
        if self._grid_generation != self.cells._view_generation:
            error = self._moved_location_error()
            if error is not None:
                raise error
            cv_ids = _resolve_detector_location_cvs(self.cells, self.location)
            n_location = int(np.max(self._location_indices)) + 1
            if len(cv_ids) != n_location:
                raise RuntimeError("Detector location count changed; recreate the detector and its connections.")
            self._cv_ids = cv_ids[self._location_indices]
            self._grid_generation = self.cells._view_generation
        return np.array(self._cv_ids, copy=True)

    def _moved_location_error(self) -> Exception | None:
        """The error to raise if an edit has shifted the branch indices :attr:`location` holds, else ``None``.

        ``RuntimeError`` for a resolved mask (MORPH-16), ``ValueError`` for
        an expression that names branches by index (MORPH-17). Read by
        :attr:`cv_id`, and by ``Cell.init_state`` for a detector connected
        to the Cell, before it initializes.
        """
        basis = self._index_basis
        if not self.cells._held_indices_moved(basis):
            return None
        if hasattr(self.location, "evaluate"):
            from braincell._discretization.mechanism import moved_index_message

            return ValueError(
                moved_index_message(
                    "VoltageCrossingSource location",
                    self.location,
                    basis[1],
                    "Recreate the detector, and connect the new one in place of this one",
                )
            )
        return RuntimeError(
            f"VoltageCrossingSource location is a resolved {type(self.location).__name__}, whose branch "
            f"indices are those of morphology revision {basis[1]}; a branch has since been removed or "
            "moved, so they no longer name the same branches. Recreate the detector with a location "
            'expression such as at("axon", 0.5).'
        )

    def current_event_count(self, source_index):
        """Return current-boundary crossing values for selected detector rows."""
        self.cells._raise_if_not_initialized("read VoltageCrossingSource")
        if self.cells._uses_reduction:
            raise RuntimeError("VoltageCrossingSource is unavailable while its Cell uses a reduction model.")
        if not hasattr(self.cells, "_event_previous_V"):
            raise RuntimeError("Cell runtime does not expose previous voltage for threshold detection.")
        source_index = np.asarray(source_index, dtype=np.int64)
        population_index = self._population_indices[source_index]
        cv_id = self.cv_id[source_index]
        if self.cells._is_lumped:
            # A lumped Cell has no CV axis: each row reads its member.
            index = (population_index,)
        elif len(self.cells.pop_size) == 0:
            index = (cv_id,)
        else:
            index = (population_index, cv_id)
        if not self._reads_previous_voltage:
            return self.cells.spike.value[index]

        threshold = self.cells.V_th[index] if self._uses_cell_threshold else self._threshold.value[source_index]
        from braincell._base_neuron import _threshold_crossing

        return _threshold_crossing(
            self.cells._event_previous_V.value[index],
            self.cells.V.value[index],
            threshold,
            self.cells.spk_fun if self.spk_fun is None else self.spk_fun,
            direction=self.direction,
        )


class _CellSpikeSource(EventSource):
    """Live owner behind ``cell.event_outputs['spike']``."""

    __slots__ = ("cell", "location")

    def __init__(self, cell) -> None:
        from braincell.filter import RootLocation

        self.cell = cell
        self.location = RootLocation(0.5)

    @property
    def size(self) -> int:
        return 1 if len(self.cell.pop_size) == 0 else int(self.cell.pop_size[0])

    @property
    def cv_id(self) -> int:
        return _resolve_cell_location_cv(self.cell, self.location)

    @property
    def instance_name(self) -> str:
        return f"{self.cell.name or type(self.cell).__name__}.spike"

    @property
    def source_type(self) -> str:
        return "CellSpikeSource"

    @property
    def source_name(self) -> str:
        return self.instance_name

    @property
    def execution_owner(self):
        """Return the Cell whose canonical spike output this source exposes."""
        return self.cell

    def current_event_count(self, source_index):
        self.cell._raise_if_not_initialized("read cell.event_outputs['spike']")
        source_index = np.asarray(source_index, dtype=np.int64)
        spike = self.cell.spike.value
        if self.cell._uses_reduction or self.cell._is_lumped:
            return spike[..., source_index]
        return spike[..., source_index, self.cv_id]


class EventOutputCollection:
    """Named live event-output ports exposed by one Cell or CellView."""

    __slots__ = ("_cell", "_population_indices")

    def __init__(self, cell, population_indices=None) -> None:
        self._cell = cell
        if population_indices is None:
            size = 1 if len(cell.pop_size) == 0 else int(cell.pop_size[0])
            population_indices = np.arange(size, dtype=np.int64)
        self._population_indices = np.asarray(population_indices, dtype=np.int64).reshape(-1)

    def __getitem__(self, name: str) -> EventSourceView:
        if name != "spike":
            raise KeyError(f"Unknown Cell event output {name!r}; available outputs: ('spike',).")
        source = self._cell._get_spike_event_source()
        return EventSourceView(source, self._population_indices)

    def for_population(self, population_indices) -> "EventOutputCollection":
        return EventOutputCollection(self._cell, population_indices)

    def __iter__(self):
        return iter(("spike",))

    def __len__(self) -> int:
        return 1


def _resolve_cell_location_cv(cell, location) -> int:
    cv_ids = _resolve_cell_location_cvs(cell, location)
    if cv_ids.size != 1:
        raise ValueError(f"Event source location must resolve to one point, got {cv_ids.size!r}.")
    return int(cv_ids[0])


def _resolve_detector_location_cvs(cell, location) -> np.ndarray:
    """:func:`_resolve_cell_location_cvs` for a detector, naming it when its branch is gone."""
    from braincell._discretization.mechanism import missing_branch_name

    try:
        return _resolve_cell_location_cvs(cell, location)
    except KeyError as error:
        name = missing_branch_name(cell.morpho, error)
        if name is None:
            raise
        raise KeyError(
            f"VoltageCrossingSource location {location!r} names branch {name!r}, which the morphology "
            f"(revision {cell.morpho.revision}) does not have: an edit removed it, or it never existed. "
            f"Attach a branch named {name!r} again, or recreate the detector on a branch the tree has."
        ) from error


def _resolve_cell_location_cvs(cell, location) -> np.ndarray:
    from braincell._discretization.base import locate_cv_on_branch

    mask = location.evaluate(cell.morpho) if hasattr(location, "evaluate") else location
    return np.asarray(
        [
            locate_cv_on_branch(
                cell.cv_tree.branch_to_cv_ids[int(branch_id)],
                cell.cvs,
                x=float(branch_x),
            )
            for branch_id, branch_x in zip(mask.branch_id, mask.branch_x)
        ],
        dtype=np.int64,
    )


def round_half_up_steps_host(values: np.ndarray) -> np.ndarray:
    """Round non-negative step ratios, snapping numerical half ties upward.

    Host-side counterpart of :func:`_round_half_up_steps`, kept in numpy so
    that quantization runs at float64. Shared by
    :mod:`braincell.network.connection` and
    :mod:`braincell.network.lowering`, which quantize delays during
    lowering, outside any trace, and by the arrival steps of scheduled
    events (CELL-46).

    Notes
    -----
    The two are **not** interchangeable, which is why both exist. Routing
    the host callers through the traced version would push float64 step
    ratios through ``u.math.asarray`` and truncate them to float32 under
    the default JAX precision, changing the result for ratios that sit
    within a few ulp of a half tie — exactly the inputs this snapping
    exists to handle.
    """
    half = np.floor(values) + 0.5
    magnitude = np.abs(values)
    ulp = np.nextafter(magnitude, np.inf) - magnitude
    snapped = np.where(np.abs(values - half) <= 4.0 * ulp, half, values)
    return np.floor(snapped + 0.5)


def _round_half_up_steps(values):
    """Round non-negative step ratios, snapping numerical half ties upward.

    Trace-safe variant, for the step of a ``t`` the caller chose and for
    arrivals whose ``delay`` or ``dt`` is traced. See
    :func:`round_half_up_steps_host` for why the host callers keep a
    separate numpy implementation.
    """
    values = u.math.asarray(values)
    half = u.math.floor(values) + 0.5
    magnitude = u.math.abs(values)
    toward_positive = u.math.asarray(np.inf, dtype=values.dtype)
    ulp = u.math.nextafter(magnitude, toward_positive) - magnitude
    snapped = u.math.where(u.math.abs(values - half) <= 4.0 * ulp, half, values)
    return u.math.floor(snapped + 0.5)


def boundary_step(t, step):
    """Context in which the event counts of the boundary ``t`` use the integer ``step``.

    The run loops set it next to ``t``. ``step`` counts boundaries from time
    0 on the ``dt`` grid, so it is compared with the arrival steps as it is,
    without rounding ``t / dt`` in the run precision (CELL-46). A ``step``
    of ``None`` clears an outer loop's step, for a ``t`` the caller chose.
    """
    return brainstate.environ.context(**{_BOUNDARY_STEP_KEY: None if step is None else (t, step)})


def grid_step(origin, step, dt):
    """The int32 step on the ``dt`` grid from time 0 of step ``step`` counted from ``origin``.

    A Cell's clock counts its steps from an origin, which is 0 until a run
    changes ``dt`` (CELL-45); the event arrival steps count from time 0.
    """
    offset = _round_half_up_steps(u.math.asarray(origin.to_decimal(u.ms)) / u.math.asarray(dt.to_decimal(u.ms)))
    return (step + offset.astype(jnp.int32)).astype(jnp.int32)


def arrival_origin(origin_ms, step):
    """Context in which the arrival steps count ``dt`` steps from ``origin_ms``, the boundary numbered ``step``.

    The arrival steps count from time 0 on the ``dt`` grid. After
    ``Cell.run`` changes ``dt``, its steps count from the time reached
    (CELL-45), which need not be on the new grid: after 3 steps of 0.025 ms,
    steps of 0.05 ms start at 0.075 ms. That run numbers its boundaries from
    ``step`` and sets this context while its loop is traced, so an arrival at
    ``time + delay`` is due at the boundary
    ``step + round((time + delay - origin_ms) / dt)``, the one of its steps
    nearest the arrival, rounded as on the grid from time 0 (CELL-48).
    ``origin_ms`` of ``None`` keeps the grid from time 0.
    """
    if origin_ms is None:
        return contextlib.nullcontext()
    return brainstate.environ.context(**{_ARRIVAL_ORIGIN_KEY: (float(origin_ms), int(step))})


def _boundary_step(t, dt):
    """The int32 step of the boundary ``t``.

    It is the run loop's count when the loop set this ``t``
    (:func:`boundary_step`). For a ``t`` the caller chose, it is ``t / dt``
    rounded in the run precision, counted from the :func:`arrival_origin`.
    """
    pair = brainstate.environ.get(_BOUNDARY_STEP_KEY, None)
    if pair is not None and pair[0] is t:
        return jnp.asarray(pair[1], dtype=jnp.int32)
    t_ms = u.math.asarray(t.to_decimal(u.ms))
    dt_ms = u.math.asarray(dt.to_decimal(u.ms))
    origin = brainstate.environ.get(_ARRIVAL_ORIGIN_KEY, None)
    if origin is None:
        return _round_half_up_steps(t_ms / dt_ms).astype(jnp.int32)
    origin_ms, origin_step = origin
    return (_round_half_up_steps((t_ms - origin_ms) / dt_ms) + origin_step).astype(jnp.int32)


def _host_ms(value):
    """The ms decimals of a time quantity as float64 numpy, or ``None`` when traced."""
    decimal = value.to_decimal(u.ms)
    if is_traced_value(decimal):
        return None
    return np.asarray(decimal, dtype=np.float64)


class ArrivalHandover(NamedTuple):
    """What a run at a new ``dt`` takes over from the runs before it (CELL-57).

    A step of a Cell's run, like a Network step, prepares and consumes the
    boundary it starts at, with the events of that boundary's window at its
    own ``dt`` (CELL-55). The run that
    follows at another ``dt`` keeps what they took: an event whose arrival
    step, counted in steps of ``taken_dt_ms`` from ``taken_origin_ms`` (the
    boundary numbered ``taken_origin_step`` there, CELL-48), is at most
    ``taken_last_step`` was taken, and is not delivered again
    (:data:`TAKEN_STEP`). Every other event arrives at its step on the new
    run's grid, but no earlier than ``first_step``, the new run's first
    boundary that no earlier run prepared. Without it, a ``dt`` that grows
    loses the events between the two half steps after the change, and a
    ``dt`` that shrinks delivers them twice.

    The steps are those the runs computed, at float64 on the host, so each
    event is taken by exactly one run.
    """

    taken_origin_ms: float
    taken_dt_ms: float
    taken_origin_step: int
    taken_last_step: int
    first_step: int

    @property
    def edge_ms(self) -> float:
        """The end of the last window taken: an arrival before it was taken."""
        return self.taken_origin_ms + (self.taken_last_step - self.taken_origin_step + 0.5) * self.taken_dt_ms

    def taken(self, arrival_ms) -> np.ndarray:
        """Whether the runs before took each of ``arrival_ms`` (float64 ms)."""
        ratio = (np.asarray(arrival_ms, dtype=np.float64) - self.taken_origin_ms) / self.taken_dt_ms
        return round_half_up_steps_host(ratio) + self.taken_origin_step <= self.taken_last_step

    def steps(self, steps, arrival_ms) -> np.ndarray:
        """``steps``, the arrival steps of ``arrival_ms`` on the new grid, with the hand-over applied."""
        return np.where(self.taken(arrival_ms), TAKEN_STEP, np.maximum(steps, self.first_step))


def handed_over(handover: "ArrivalHandover | None"):
    """Context in which the arrival steps apply ``handover`` (:class:`ArrivalHandover`); ``None`` changes nothing.

    The run loops set it while they are traced, next to
    :func:`arrival_origin`, so the arrival plans built inside it keep it.
    """
    if handover is None:
        return contextlib.nullcontext()
    return brainstate.environ.context(**{_HANDOVER_KEY: handover})


def _host_arrival_steps(time_ms, delay_ms, dt_ms, handover: "ArrivalHandover | None" = None) -> np.ndarray:
    """int32 arrival steps ``(time + delay) / dt``, rounded at float64 on the host.

    The event times and delays are float64 declarations, so the step does
    not depend on the run precision: a float32 run delivers an event at the
    step a float64 run does (CELL-46). Inside :func:`arrival_origin` the
    steps count from its origin (CELL-48). ``handover``, or else the one of
    :func:`handed_over`, gives the events earlier runs took
    :data:`TAKEN_STEP` and moves the others to its first step or later
    (CELL-57).
    """
    arrival_ms = np.asarray(time_ms, dtype=np.float64) + delay_ms
    origin = brainstate.environ.get(_ARRIVAL_ORIGIN_KEY, None)
    if origin is None:
        steps = round_half_up_steps_host(arrival_ms / dt_ms)
    else:
        origin_ms, origin_step = origin
        steps = round_half_up_steps_host((arrival_ms - origin_ms) / dt_ms) + origin_step
    if handover is None:
        handover = brainstate.environ.get(_HANDOVER_KEY, None)
    if handover is not None:
        steps = handover.steps(steps, arrival_ms)
    return np.minimum(steps, _MAX_STEP).astype(np.int32)


def _matrix_arrival_steps(time_ms, delay, dt, *, rows):
    """Arrival steps of ``time_ms + delay[..., None]``, one row of events per connection row.

    At float64 on the host, unless ``delay`` or ``dt`` is traced; then in
    the run precision. Both count from the :func:`arrival_origin`.
    """
    delay_ms, dt_ms = _host_ms(delay), _host_ms(dt)
    if delay_ms is None or dt_ms is None:
        delay_ms = u.math.asarray(delay.to_decimal(u.ms))
        if getattr(delay_ms, "ndim", 0) == 0:
            delay_ms = u.math.broadcast_to(delay_ms, rows)
        arrivals = u.math.asarray(time_ms) + delay_ms[..., None]
        dt_ms = u.math.asarray(dt.to_decimal(u.ms))
        origin = brainstate.environ.get(_ARRIVAL_ORIGIN_KEY, None)
        if origin is None:
            return _round_half_up_steps(arrivals / dt_ms).astype(jnp.int32)
        origin_ms, origin_step = origin
        return (_round_half_up_steps((arrivals - origin_ms) / dt_ms) + origin_step).astype(jnp.int32)
    if delay_ms.ndim == 0:
        delay_ms = np.broadcast_to(delay_ms, rows)
    return jnp.asarray(_host_arrival_steps(time_ms, delay_ms[..., None], dt_ms))


@dataclass(frozen=True)
class _ArrivalPlan:
    """Arrival steps of a fixed schedule, sorted for a per-step lookup.

    ``event_count`` used to rebuild the full ``(rows, events)`` arrival
    matrix and compare it with the current step on every step (CELL-18a).
    The arrival steps do not depend on ``t``, so they are computed once,
    when the step is traced, at float64 on the host (CELL-46), and sorted;
    ``count`` compares them with the loop's integer step. Rows that share a
    source and a delay share their arrivals (a group). A step then costs
    one binary search for the first arrival at that step, a read of at
    most ``window`` arrivals (the most that share any step), and one gather
    from groups to rows.
    """

    steps: np.ndarray
    group: np.ndarray
    window: int
    row_group: np.ndarray
    n_groups: int
    shape: tuple[int, ...]

    @classmethod
    def build(cls, *, event_source, event_time_ms, source_index, t, delay, dt, one_dimensional=False):
        """Return a plan, or ``None`` when the matrix path must run instead.

        That is when ``delay`` or ``dt`` is traced, ``t`` or ``dt`` is not
        a scalar, or the delays do not broadcast as the matrix path does.
        """
        if not all(isinstance(value, u.Quantity) for value in (t, delay, dt)):
            return None
        delay_ms, dt_ms = _host_ms(delay), _host_ms(dt)
        if delay_ms is None or dt_ms is None:
            return None
        if np.ndim(dt_ms) != 0 or np.ndim(t.mantissa) != 0:
            return None
        if delay_ms.ndim == 0:
            shape = source_index.shape
        elif one_dimensional and delay_ms.shape != source_index.shape:
            return None
        else:
            try:
                shape = np.broadcast_shapes(source_index.shape, delay_ms.shape)
            except ValueError:
                return None
        rows = np.broadcast_to(source_index, shape).reshape(-1).astype(np.int64)
        row_delay = np.broadcast_to(delay_ms, shape).reshape(-1)

        # Groups: distinct (source, delay bit pattern) pairs.
        delay_bits = row_delay.view(np.dtype(f"u{row_delay.dtype.itemsize}")).astype(np.int64)
        keys, row_group = np.unique(np.stack([rows, delay_bits], axis=1), axis=0, return_inverse=True)
        row_group = row_group.reshape(-1).astype(np.int32)
        _, first_row = np.unique(row_group, return_index=True)
        group_source = keys[:, 0]
        group_delay = row_delay[first_row]

        # Pairs: every event of each group's source.
        order = np.argsort(event_source, kind="stable")
        sorted_source = np.asarray(event_source, dtype=np.int64)[order]
        sorted_time = np.asarray(event_time_ms, dtype=np.float64)[order]
        lo = np.searchsorted(sorted_source, group_source, side="left")
        per_group = np.searchsorted(sorted_source, group_source, side="right") - lo
        n_pairs = int(per_group.sum())
        if n_pairs == 0:
            return cls(np.zeros(0, np.int32), np.zeros(0, np.int32), 0, row_group, len(keys), shape)
        pair_group = np.repeat(np.arange(len(keys), dtype=np.int32), per_group)
        pair_event = np.arange(n_pairs) + np.repeat(lo - (np.cumsum(per_group) - per_group), per_group)
        steps = _host_arrival_steps(sorted_time[pair_event], group_delay[pair_group], dt_ms)
        delivered = steps != TAKEN_STEP
        if not delivered.all():
            # Events an earlier run took (CELL-57) never arrive.
            steps, pair_group = steps[delivered], pair_group[delivered]
            n_pairs = int(steps.size)
            if n_pairs == 0:
                return cls(np.zeros(0, np.int32), np.zeros(0, np.int32), 0, row_group, len(keys), shape)
        by_step = np.argsort(steps, kind="stable")
        steps, pair_group = steps[by_step], pair_group[by_step]
        boundaries = np.flatnonzero(np.diff(steps) != 0)
        window = int(np.max(np.diff(np.concatenate([[-1], boundaries, [n_pairs - 1]]))))
        return cls(steps, pair_group, window, row_group, len(keys), shape)

    def count(self, *, t, dt):
        """Arrivals at the step of ``t``, one count per row."""
        count_dtype = jax.dtypes.canonicalize_dtype(np.int64)
        current_step = _boundary_step(t, dt)
        if self.window == 0:
            return jnp.zeros(self.shape, dtype=count_dtype)
        steps = jnp.asarray(self.steps)
        # Unrolled: a compiled ``while`` costs a device round trip per
        # iteration on GPU, where this search then took ~80 us per step.
        first = jnp.searchsorted(steps, current_step, side="left", method="scan_unrolled")
        position = first + jnp.arange(self.window)
        within = jnp.minimum(position, self.steps.size - 1)
        due = (position < self.steps.size) & (steps[within] == current_step)
        group_count = jnp.zeros(self.n_groups, dtype=count_dtype)
        group_count = group_count.at[jnp.asarray(self.group)[within]].add(due.astype(count_dtype))
        return group_count[jnp.asarray(self.row_group)].reshape(self.shape)


def _flat_event_count(table: EventTable, *, source_index, t, delay, dt):
    """Count flat scheduled arrivals for one source index per connection row."""
    source_index = np.asarray(source_index, dtype=np.int64).reshape(-1)
    event_times = np.asarray(table.time.to_decimal(u.ms), dtype=np.float64)
    arrival_steps = _matrix_arrival_steps(event_times[None, :], delay, dt, rows=source_index.shape)
    selected_source = np.asarray(table.source_index)[None, :] == source_index[:, None]
    return u.math.sum(jnp.asarray(selected_source) & (arrival_steps == _boundary_step(t, dt)), axis=-1)


def _quantity_vector(value, *, unit, size: int, name: str) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a quantity.")
    try:
        decimal = np.asarray(value.to_decimal(unit), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"{name} has an incompatible unit.") from exc
    decimal = _broadcast_vector(decimal, size=size, name=name)
    if np.any(~np.isfinite(decimal)):
        raise ValueError(f"{name} must contain finite values.")
    return u.Quantity(decimal, unit)


def _integer_vector(value, *, size: int, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise TypeError(f"{name} must contain integers.")
    return _broadcast_vector(values, size=size, name=name).astype(np.int64, copy=False)


def _real_vector(value, *, size: int, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.dtype.kind not in "iuf" or values.dtype.kind == "b":
        raise TypeError(f"{name} must contain real numbers.")
    result = _broadcast_vector(values, size=size, name=name).astype(np.float64, copy=False)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result


def _broadcast_vector(value: np.ndarray, *, size: int, name: str) -> np.ndarray:
    if value.ndim == 0:
        return np.array(np.broadcast_to(value, (size,)), copy=True)
    if value.shape == (size,):
        return np.array(value, copy=True)
    raise ValueError(f"{name} must be scalar or have shape {(size,)!r}, got {value.shape!r}.")


def _build_event_schedule(*, start_ms, number, interval_ms, noise, seed, width=None):
    """Return the first ``width`` events of every source (all when ``None``).

    Each noisy source draws its own exponentials (:func:`_source_exponentials`),
    and the times are a running sum along each row, so a narrower schedule is
    a prefix of a wider one and a source's events do not depend on the others.
    """
    max_events = int(np.max(number, initial=0))
    columns = max_events if width is None else int(width)
    mask = np.arange(columns)[None, :] < number[:, None]
    if columns == 0:
        return np.empty((len(number), 0), dtype=np.float64), mask

    exponential = np.zeros((len(number), columns), dtype=np.float64)
    noisy = np.flatnonzero(noise > 0.0)
    if noisy.size:
        exponential[noisy] = _source_exponentials(seed, noisy, columns)
    first_wait = noise * interval_ms * exponential[:, 0]
    if columns == 1:
        offsets = first_wait[:, None]
    else:
        deterministic = (1.0 - noise[:, None]) * interval_ms[:, None]
        stochastic = noise[:, None] * interval_ms[:, None] * exponential[:, 1:]
        gaps = deterministic + stochastic
        offsets = first_wait[:, None] + np.concatenate(
            [
                np.zeros((len(number), 1), dtype=np.float64),
                np.cumsum(gaps, axis=1),
            ],
            axis=1,
        )
    return start_ms[:, None] + offsets, mask


def _estimated_width(source: NetStim, until_ms: float) -> int:
    """Columns likely to hold every event up to ``until_ms`` plus one more.

    Exact for a periodic source; for a noisy one 1.25 times the mean count
    plus 16, which the caller doubles when a source runs short.
    """
    start_ms = np.asarray(source.start.to_decimal(u.ms), dtype=np.float64)
    interval_ms = np.asarray(source.interval.to_decimal(u.ms), dtype=np.float64)
    span = np.maximum(until_ms - start_ms, 0.0) / interval_ms
    min_gap = 1.0 - source.noise
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(min_gap > 0.0, np.floor(span / min_gap) + 2.0, np.inf)
    estimate = np.minimum(bound, np.ceil(1.25 * span) + 16.0)
    return int(np.max(np.minimum(estimate, source.number), initial=0))


def _source_exponentials(seed: int, sources: np.ndarray, columns: int) -> np.ndarray:
    """Standard exponential draws ``0 .. columns - 1`` of each source in ``sources``.

    Draw ``j`` of source ``i`` is a function of ``(seed, i, j)`` only: its 64
    random bits are the key data of ``fold_in(fold_in(key(seed), i), j)``
    with the ``threefry2x32`` generator, the top 53 of them give a uniform
    ``u`` in ``[0, 1)``, and the draw is ``-log1p(-u)`` in float64. It does
    not depend on the other sources, on ``number``, on how many columns are
    drawn, on the float precision or on JAX's generator settings.
    """
    with jax.ensure_compile_time_eval():
        root = jax.random.key_data(jax.random.key(seed, impl="threefry2x32"))
        data = _source_bits(root, np.asarray(sources, dtype=np.uint32), np.arange(columns, dtype=np.uint32))
    data = np.asarray(data)
    bits = (data[..., 0].astype(np.uint64) << np.uint64(32)) | data[..., 1]
    draws = (bits >> np.uint64(11)).astype(np.float64)
    del bits
    draws *= -(2.0**-53)
    np.log1p(draws, out=draws)
    return np.negative(draws, out=draws)


@jax.jit
def _source_bits(root, sources, columns):
    """Key data of ``fold_in(fold_in(root, source), column)`` for every source and column."""
    key = jax.random.wrap_key_data(root, impl="threefry2x32")
    per_source = jax.vmap(lambda source: jax.random.fold_in(key, source))(sources)
    return jax.random.key_data(
        jax.vmap(lambda row: jax.vmap(lambda column: jax.random.fold_in(row, column))(columns))(per_source)
    )


def _prepare_netstim_schedules(sources, *, until_ms: float | None) -> tuple:
    """Generate each NetStim's events up to ``until_ms`` and return a cache token.

    Run entries call this before looking up their compiled loop: the loop
    bakes in the generated schedule, so the token keys it.
    """
    seen = set()
    keys = []
    for source in sources:
        if not isinstance(source, NetStim) or id(source) in seen:
            continue
        seen.add(id(source))
        source._ensure_schedule(until_ms=until_ms)
        keys.append(source._schedule_key)
    return tuple(keys)


def _schedule_horizon(until_ms: float | None):
    """Context in which :meth:`NetStim.event_count` reads events up to ``until_ms``."""
    if until_ms is None:
        return contextlib.nullcontext()
    return brainstate.environ.context(**{_SCHEDULE_HORIZON_KEY: float(until_ms)})
