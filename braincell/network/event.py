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
from dataclasses import dataclass, field
import hashlib
from typing import Any

import brainstate
import brainunit as u
import numpy as np

__all__ = [
    "EventSource",
    "EventSourceView",
    "EventTable",
    "EventSequence",
    "NetStim",
    "VoltageCrossingSource",
]


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

    @property
    def ids(self) -> np.ndarray:
        """Alias for :attr:`source_id`."""
        return self.source_id

    def __len__(self) -> int:
        return int(self._source_ids.size)

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
        return _flat_event_count(
            self.events,
            source_index=np.asarray(source_index, dtype=np.int64),
            t=t,
            delay=delay,
            dt=dt,
        )


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
        Event count, scalar or one value per source.
    interval : Quantity[ms], optional
        Mean inter-event interval, scalar or one value per source.
    noise : float or array-like, optional
        Fraction of each interval drawn from an exponential distribution.
        Zero is periodic and one is Poisson-like.
    seed : int or None, optional
        Source-local seed. ``None`` uses the reproducible standalone root 0.
    name : str or None, optional
        Optional display name.
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

        event_times, event_mask = _build_event_schedule(
            start_ms=start_ms,
            number=number,
            interval_ms=interval_ms,
            noise=noise,
            seed=0 if self.seed is None else int(self.seed),
        )
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "_event_times_ms", event_times)
        object.__setattr__(self, "_event_mask", event_mask)

    @property
    def instance_name(self) -> str:
        """Display label for this source population."""
        return self.name if self.name is not None else "NetStim"

    @property
    def is_scheduled(self) -> bool:
        return True

    @property
    def event_times(self) -> u.Quantity:
        """Padded continuous event-time matrix with shape ``(size, max(number))``."""
        return u.Quantity(self._event_times_ms, u.ms)

    def event_count(self, source_index, *, t, delay, dt):
        """Count delayed events assigned to the nearest fixed-step boundary.

        An arrival is delivered at step ``t`` when it lies in the half-open
        interval ``[t - dt / 2, t + dt / 2)``. This matches NEURON's
        fixed-step event delivery rule and assigns an exact half-step tie to
        the later boundary.
        """
        source_index = np.asarray(source_index, dtype=np.int32)
        times = u.math.asarray(self._event_times_ms[source_index])
        mask = u.math.asarray(self._event_mask[source_index])
        delay_ms = u.math.asarray(delay.to_decimal(u.ms))
        if getattr(delay_ms, "ndim", 0) == 0:
            delay_ms = u.math.broadcast_to(delay_ms, source_index.shape)
        arrivals = times + delay_ms[..., None]
        t_ms = u.math.asarray(t.to_decimal(u.ms))
        dt_ms = u.math.asarray(dt.to_decimal(u.ms))
        arrival_steps = _round_half_up_steps(arrivals / dt_ms)
        current_step = _round_half_up_steps(t_ms / dt_ms)
        selected = mask & (arrival_steps == current_step)
        return u.math.sum(selected, axis=-1)

    @property
    def events(self) -> EventTable:
        """Return the generated schedule as canonical flat event rows."""
        source_index, event_index = np.nonzero(self._event_mask)
        return EventTable(
            source_index=source_index,
            time=self._event_times_ms[source_index, event_index] * u.ms,
        )

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
        event_times, event_mask = _build_event_schedule(
            start_ms=np.asarray(self.start.to_decimal(u.ms), dtype=np.float64),
            number=np.asarray(self.number, dtype=np.int64),
            interval_ms=np.asarray(self.interval.to_decimal(u.ms), dtype=np.float64),
            noise=np.asarray(self.noise, dtype=np.float64),
            seed=effective_seed,
        )
        object.__setattr__(self, "_event_times_ms", event_times)
        object.__setattr__(self, "_event_mask", event_mask)


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
    )

    def __init__(
        self,
        cells,
        *,
        location=None,
        threshold=_CELL_VOLTAGE_THRESHOLD,
        direction: str = "rising",
        name: str | None = None,
    ) -> None:
        from braincell.filter import RootLocation

        root = getattr(cells, "root", cells)
        population_indices = getattr(cells, "population_indices", None)
        if population_indices is None:
            population_size = 1 if len(getattr(root, "pop_size", ())) == 0 else int(root.pop_size[0])
            population_indices = tuple(range(population_size))
        if location is None:
            location = RootLocation(0.5)
        if not hasattr(location, "evaluate") and not (hasattr(location, "branch_id") and hasattr(location, "branch_x")):
            raise TypeError("VoltageCrossingSource.location must be a location expression or resolved locset mask.")
        location_cv_ids = _resolve_cell_location_cvs(root, location)
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
        self.name = name
        self._threshold = normalized_threshold
        self._uses_cell_threshold = uses_cell_threshold
        self._population_indices = np.repeat(population_indices, n_location)
        self._location_indices = np.tile(np.arange(n_location, dtype=np.int64), n_population)
        self._cv_ids = np.tile(location_cv_ids, n_population)

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
        return self.cells.V_th if self._uses_cell_threshold else self._threshold

    @property
    def instance_name(self) -> str:
        return self.name if self.name is not None else "VoltageCrossingSource"

    @property
    def execution_owner(self):
        """Return the Cell whose voltage drives this detector."""
        return self.cells

    @property
    def cv_id(self) -> np.ndarray:
        """Return the containing CV for each source endpoint."""
        return np.array(self._cv_ids, copy=True)

    def current_event_count(self, source_index):
        """Return current-boundary crossing values for selected detector rows."""
        self.cells._raise_if_not_initialized("read VoltageCrossingSource")
        if not hasattr(self.cells, "_event_previous_V"):
            raise RuntimeError("Cell runtime does not expose previous voltage for threshold detection.")
        source_index = np.asarray(source_index, dtype=np.int64)
        population_index = self._population_indices[source_index]
        cv_id = self._cv_ids[source_index]
        if self._uses_cell_threshold and self.direction == "rising":
            spike = self.cells.spike.value
            if len(self.cells.pop_size) == 0:
                return spike[cv_id]
            return spike[population_index, cv_id]

        last_v = self.cells._event_previous_V.value
        next_v = self.cells.V.value
        if len(self.cells.pop_size) == 0:
            last = last_v[cv_id]
            next_value = next_v[cv_id]
            threshold = self.cells.V_th[cv_id] if self._uses_cell_threshold else self._threshold[source_index]
        else:
            last = last_v[population_index, cv_id]
            next_value = next_v[population_index, cv_id]
            threshold = (
                self.cells.V_th[population_index, cv_id] if self._uses_cell_threshold else self._threshold[source_index]
            )
        if self.direction == "rising":
            return (last < threshold) & (next_value >= threshold)
        return (last > threshold) & (next_value <= threshold)


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
        if len(self.cell.pop_size) == 0:
            return u.math.broadcast_to(spike[self.cv_id], source_index.shape)
        return spike[source_index, self.cv_id]


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


def _round_half_up_steps(values):
    """Round non-negative step ratios, snapping numerical half ties upward."""
    values = u.math.asarray(values)
    half = u.math.floor(values) + 0.5
    magnitude = u.math.abs(values)
    toward_positive = u.math.asarray(np.inf, dtype=values.dtype)
    ulp = u.math.nextafter(magnitude, toward_positive) - magnitude
    snapped = u.math.where(u.math.abs(values - half) <= 4.0 * ulp, half, values)
    return u.math.floor(snapped + 0.5)


def _flat_event_count(table: EventTable, *, source_index, t, delay, dt):
    """Count flat scheduled arrivals for one source index per connection row."""
    source_index = np.asarray(source_index, dtype=np.int64).reshape(-1)
    delay_ms = u.math.asarray(delay.to_decimal(u.ms))
    if getattr(delay_ms, "ndim", 0) == 0:
        delay_ms = u.math.broadcast_to(delay_ms, source_index.shape)
    event_sources = u.math.asarray(table.source_index)
    event_times = u.math.asarray(table.time.to_decimal(u.ms))
    arrivals = event_times[None, :] + delay_ms[:, None]
    selected_source = event_sources[None, :] == u.math.asarray(source_index)[:, None]
    arrival_steps = _round_half_up_steps(arrivals / u.math.asarray(dt.to_decimal(u.ms)))
    current_step = _round_half_up_steps(u.math.asarray(t.to_decimal(u.ms)) / u.math.asarray(dt.to_decimal(u.ms)))
    return u.math.sum(selected_source & (arrival_steps == current_step), axis=-1)


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


def _build_event_schedule(*, start_ms, number, interval_ms, noise, seed):
    max_events = int(np.max(number, initial=0))
    mask = np.arange(max_events)[None, :] < number[:, None]
    if max_events == 0:
        return np.empty((len(number), 0), dtype=np.float64), mask

    rng = brainstate.random.RandomState(seed)
    exponential = np.asarray(
        rng.exponential(scale=1.0, size=(len(number), max_events)),
        dtype=np.float64,
    )
    first_wait = noise * interval_ms * exponential[:, 0]
    if max_events == 1:
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
