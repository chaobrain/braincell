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

"""Direct event-source to logical-synapse connections."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._misc import scalar_decimal
from braincell._multi_compartment.synapses import SynapseView, _cell_label
from .pairing import (
    PairingContext,
    PairingSpec,
    by_source,
    by_synapse,
    degree,
    independent,
    match_degrees,
    materialize_pairing,
    source_first,
    synapse_first,
)
from braincell.mech import (
    NoEventInput,
    ScalarEventInput,
    TriggerEventInput,
    get_registry,
)
from .core import Population
from .event import (
    EventSource,
    EventSourceView,
    _quantity_vector,
    round_half_up_steps_host as _round_half_up_steps,
)
from .recording import _require_name

__all__ = [
    "ConnectionView",
    "NetworkConnections",
    "PairingContext",
    "PairingSpec",
    "by_source",
    "by_synapse",
    "connect",
    "degree",
    "independent",
    "match_degrees",
    "source_first",
    "synapse_first",
]


_UNSET = object()
_CONNECT_CALL_WARNING_THRESHOLD = 256


@dataclass
class _ConnectionCall:
    """Metadata and per-call payload storage for one batched ``connect()``."""

    id: int
    name: str
    source: EventSource
    synapse_type: str
    synapse_name: str
    row_ids: np.ndarray
    weight: object
    live_history: object = None
    live_delay_steps: np.ndarray | None = None
    live_dt_ms: float | None = None


class _ConnectionStore:
    """Cell-owned SoA columns for all direct event-routing rows."""

    __slots__ = (
        "cell",
        "id",
        "connect_id",
        "source_index",
        "synapse_id",
        "target_index",
        "delay",
        "active",
        "_row_of_id",
        "_calls",
        "_call_by_id",
        "_call_by_name",
        "_next_row_id",
        "_next_connect_id",
        "_warning_emitted",
    )

    def __init__(self, cell) -> None:
        self.cell = cell
        self.id = np.asarray([], dtype=np.int64)
        self.connect_id = np.asarray([], dtype=np.int64)
        self.source_index = np.asarray([], dtype=np.int64)
        self.synapse_id = np.asarray([], dtype=np.int64)
        self.target_index = np.asarray([], dtype=np.int64)
        self.delay = np.asarray([], dtype=np.float64) * u.ms
        self.active = np.asarray([], dtype=bool)
        self._row_of_id = np.asarray([], dtype=np.int64)
        self._calls: list[_ConnectionCall] = []
        self._call_by_id: dict[int, _ConnectionCall] = {}
        self._call_by_name: dict[str, _ConnectionCall] = {}
        self._next_row_id = 0
        self._next_connect_id = 0
        self._warning_emitted = False

    def add(
        self,
        *,
        name: str,
        source: EventSource,
        source_index,
        synapse: SynapseView,
        synapse_id,
        target_index,
        weight,
        delay,
    ) -> np.ndarray:
        if name in self._call_by_name:
            raise ValueError(f"Connection name {name!r} is already used on Cell {self.cell.name!r}.")
        count = len(source_index)
        connect_id = self._next_connect_id
        self._next_connect_id += 1
        row_ids = np.arange(self._next_row_id, self._next_row_id + count, dtype=np.int64)
        self._next_row_id += count

        start = len(self.id)
        self.id = np.concatenate((self.id, row_ids))
        self.connect_id = np.concatenate((self.connect_id, np.full(count, connect_id, dtype=np.int64)))
        self.source_index = np.concatenate((self.source_index, np.asarray(source_index, dtype=np.int64)))
        self.synapse_id = np.concatenate((self.synapse_id, np.asarray(synapse_id, dtype=np.int64)))
        self.target_index = np.concatenate((self.target_index, np.asarray(target_index, dtype=np.int64)))
        self.delay = _concat_quantities(self.delay, delay, unit=u.ms)
        self.active = np.concatenate((self.active, np.ones(count, dtype=bool)))
        self._row_of_id = np.concatenate((self._row_of_id, np.arange(start, start + count, dtype=np.int64)))

        call = _ConnectionCall(
            id=connect_id,
            name=name,
            source=source,
            synapse_type=str(synapse.synapse_type[0]),
            synapse_name=str(synapse.name[0]),
            row_ids=row_ids,
            weight=weight,
        )
        self._calls.append(call)
        self._call_by_id[connect_id] = call
        self._call_by_name[name] = call
        if len(self._calls) > _CONNECT_CALL_WARNING_THRESHOLD and not self._warning_emitted:
            warnings.warn(
                f"Cell {self.cell.name!r} has more than {_CONNECT_CALL_WARNING_THRESHOLD} connect() calls; "
                "prefer one batched connect() call for many routing rows.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warning_emitted = True
        return row_ids

    def rows(self, ids) -> np.ndarray:
        return self._row_of_id[np.asarray(ids, dtype=np.int64)]

    def active_ids(self, ids=None) -> np.ndarray:
        ids = self.id if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.asarray([], dtype=np.int64)
        rows = self.rows(ids)
        return ids[self.active[rows]]

    def call(self, connect_id: int) -> _ConnectionCall:
        return self._call_by_id[int(connect_id)]

    def active_call_ids(self, ids) -> tuple[int, ...]:
        rows = self.rows(self.active_ids(ids))
        return tuple(dict.fromkeys(int(item) for item in self.connect_id[rows].tolist()))

    def weight_for(self, ids):
        ids = self.active_ids(ids)
        if ids.size == 0:
            raise ValueError("Cannot access weight on an empty ConnectionView.")
        calls = [self.call(int(value)) for value in self.connect_id[self.rows(ids)].tolist()]
        if calls[0].weight is None:
            if any(call.weight is not None for call in calls):
                raise TypeError("ConnectionView mixes trigger-only and scalar-weight rows.")
            return None
        if any(call.weight is None for call in calls):
            raise TypeError("ConnectionView mixes trigger-only and scalar-weight rows.")
        # ``add`` builds ``row_ids`` as one ``np.arange``, so a row's slot
        # in its call's weight vector is its offset from that call's first
        # row. Searching for it cost O(rows) per row.
        return _stack_values(
            [call.weight[int(row_id) - int(call.row_ids[0])] for row_id, call in zip(ids.tolist(), calls)]
        )

    def set_weight(self, ids, values) -> None:
        ids = self.active_ids(ids)
        split = _split_values(values, len(ids))
        connect_id = self.connect_id[self.rows(ids)]
        # ``_set_quantity_or_array`` rebuilds the call's whole weight
        # vector, so doing it once per row was quadratic in the rows of a
        # single call. Group first, then assign each call's slots at once.
        for call_id in np.unique(connect_id).tolist():
            mask = connect_id == call_id
            call = self.call(int(call_id))
            selected = [value for value, keep in zip(split, mask.tolist()) if keep]
            call.weight = _set_quantity_or_array(
                call.weight,
                ids[mask] - int(call.row_ids[0]),
                _stack_values(selected),
            )


class ConnectionView:
    """View an ordered selection of concrete event-routing rows."""

    __slots__ = ("_store", "_ids")

    def __init__(self, store: _ConnectionStore, ids=None) -> None:
        self._store = store
        self._ids = None if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)

    @property
    def cell(self):
        """Return the Cell that owns the destination synapses."""
        return self._store.cell

    @property
    def root(self) -> "ConnectionView":
        return ConnectionView(self._store)

    @property
    def id(self) -> np.ndarray:
        return np.array(self._active_ids, copy=True)

    @property
    def connect_id(self) -> np.ndarray:
        return self._column("connect_id")

    @property
    def connect_name(self) -> np.ndarray:
        return np.asarray([self._store.call(value).name for value in self.connect_id.tolist()], dtype=object)

    @property
    def connect_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(item) for item in self.connect_name.tolist()))

    @property
    def source_index(self) -> np.ndarray:
        return self._column("source_index")

    @property
    def source_type(self) -> np.ndarray:
        return np.asarray(
            [self._store.call(value).source.source_type for value in self.connect_id.tolist()],
            dtype=object,
        )

    @property
    def source_name(self) -> np.ndarray:
        return np.asarray(
            [self._store.call(value).source.source_name for value in self.connect_id.tolist()],
            dtype=object,
        )

    @property
    def source_views(self) -> tuple[EventSourceView, ...]:
        grouped: dict[int, tuple[EventSource, list[int]]] = {}
        for connect_id, source_index in zip(self.connect_id.tolist(), self.source_index.tolist()):
            source = self._store.call(connect_id).source
            key = id(source)
            if key not in grouped:
                grouped[key] = (source, [])
            grouped[key][1].append(int(source_index))
        return tuple(EventSourceView(source, indices) for source, indices in grouped.values())

    @property
    def source_view(self) -> EventSourceView:
        views = self.source_views
        if len(views) != 1:
            raise TypeError(
                f"ConnectionView contains {len(views)} source owners; use source_views or filter by_source()."
            )
        return views[0]

    @property
    def source(self) -> EventSource:
        return self.source_view.owner

    @property
    def synapse(self) -> SynapseView:
        return SynapseView(self.cell, self.synapse_id)

    @property
    def synapse_id(self) -> np.ndarray:
        return self._column("synapse_id")

    @property
    def target_index(self) -> np.ndarray:
        """Return compatibility indices into the SynapseView passed to connect()."""
        return self._column("target_index")

    @property
    def synapse_type(self) -> np.ndarray:
        return self.synapse.synapse_type

    @property
    def synapse_name(self) -> np.ndarray:
        return self.synapse.name

    @property
    def weight(self):
        self._require_homogeneous_synapse_type("access weight")
        return self._store.weight_for(self._active_ids)

    @property
    def delay(self):
        return self._store.delay[self._rows]

    @property
    def removed(self) -> bool:
        return len(self) == 0

    @property
    def _candidate_ids(self) -> np.ndarray:
        return self._store.id if self._ids is None else self._ids

    @property
    def _active_ids(self) -> np.ndarray:
        return self._store.active_ids(self._candidate_ids)

    @property
    def _rows(self) -> np.ndarray:
        return self._store.rows(self._active_ids)

    def _column(self, name: str) -> np.ndarray:
        return np.array(np.asarray(getattr(self._store, name))[self._rows], copy=True)

    def __len__(self) -> int:
        return int(self._active_ids.size)

    def __getitem__(self, selector) -> "ConnectionView":
        if isinstance(selector, str):
            return self.by_connect_name(selector)
        selected = self._active_ids[selector]
        return ConnectionView(self._store, np.asarray(selected, dtype=np.int64).reshape(-1))

    def by_connect_name(self, name: str) -> "ConnectionView":
        _require_name(name, "connection name")
        return ConnectionView(self._store, self._active_ids[self.connect_name == name])

    def by_source(self, source: EventSource | EventSourceView) -> "ConnectionView":
        owner = source.owner if isinstance(source, EventSourceView) else source
        if not isinstance(owner, EventSource):
            raise TypeError("source must be an EventSource or EventSourceView.")
        mask = np.asarray(
            [self._store.call(value).source is owner for value in self.connect_id.tolist()],
            dtype=bool,
        )
        return ConnectionView(self._store, self._active_ids[mask])

    def by_source_type(self, source_type: str) -> "ConnectionView":
        _require_name(source_type, "source_type")
        return ConnectionView(self._store, self._active_ids[self.source_type == source_type])

    def by_source_name(self, source_name: str) -> "ConnectionView":
        _require_name(source_name, "source_name")
        return ConnectionView(self._store, self._active_ids[self.source_name == source_name])

    def by_synapse_type(self, synapse_type: str) -> "ConnectionView":
        _require_name(synapse_type, "synapse_type")
        return ConnectionView(self._store, self._active_ids[self.synapse_type == synapse_type])

    def by_synapse_name(self, synapse_name: str) -> "ConnectionView":
        _require_name(synapse_name, "synapse_name")
        return ConnectionView(self._store, self._active_ids[self.synapse_name == synapse_name])

    def by_synapse_ids(self, synapse_ids) -> "ConnectionView":
        """Return rows targeting selected stable logical synapse IDs."""
        selected = np.asarray(synapse_ids, dtype=np.int64).reshape(-1)
        return ConnectionView(self._store, self._active_ids[np.isin(self.synapse_id, selected)])

    def for_population(self, population_indices) -> "ConnectionView":
        selected = np.asarray(tuple(int(index) for index in population_indices), dtype=np.int64)
        return ConnectionView(self._store, self._active_ids[np.isin(self.synapse.population_index, selected)])

    def set(self, *, weight=_UNSET, delay=_UNSET) -> "ConnectionView":
        """Update selected routing rows before Cell initialization."""
        self.cell._raise_if_initialized("modify Connection")
        if weight is not _UNSET:
            self._require_homogeneous_synapse_type("modify weight")
            normalized = _normalize_weight(self.synapse, weight, count=len(self), omitted=False)
            current = self._store.weight_for(self._active_ids)
            if (current is None) != (normalized is None):
                raise TypeError("Connection weight cannot change the target event-input payload kind.")
            if normalized is not None:
                self._store.set_weight(self._active_ids, normalized)
        if delay is not _UNSET:
            normalized = _normalize_delay(delay, count=len(self))
            self._store.delay = _set_quantity_or_array(self._store.delay, self._rows, normalized)
        return self

    def remove(self) -> None:
        """Remove selected rows while leaving stable row IDs and names unused."""
        self.cell._raise_if_initialized("remove Connection")
        self._store.active[self._store.rows(self._candidate_ids)] = False

    def _call_views(self, *, scheduled=None) -> tuple["ConnectionView", ...]:
        """Split this view into one sub-view per originating ``connect()`` call.

        Both ``_active_ids`` and ``connect_id`` are recomputed properties that
        walk every row, so asking for them once per connect call made this
        O(calls x rows) -- 20.7 s for 256 calls over 1600 rows against 9 ms
        for the single grouping pass below. ``Network.run`` reaches here on
        every call, as does ``Cell`` once per synapse layout.
        """
        ids = self._active_ids
        if ids.size == 0:
            return ()
        connect_id = self._store.connect_id[self._store.rows(ids)]
        call_ids, first_seen, inverse = np.unique(connect_id, return_index=True, return_inverse=True)
        sorter = np.argsort(inverse, kind="stable")
        starts = np.searchsorted(inverse[sorter], np.arange(call_ids.size))
        grouped = np.split(ids[sorter], starts[1:])
        result = []
        # ``np.unique`` sorts; the previous ``dict.fromkeys`` shape yielded
        # calls in first-appearance order, so restore that ordering.
        for position in np.argsort(first_seen).tolist():
            call = self._store.call(int(call_ids[position]))
            if scheduled is not None and bool(call.source.is_scheduled) != bool(scheduled):
                continue
            result.append(ConnectionView(self._store, grouped[position]))
        return tuple(result)

    def event_count(self, *, t, dt):
        """Return one arrival count per row for a single connect call."""
        call = self._require_single_call("evaluate events")
        if not call.source.is_scheduled:
            return self._live_event_count(call=call, dt=dt)
        return call.source.event_count(self.source_index, t=t, delay=self.delay, dt=dt)

    def prepare_runtime(self, dt) -> None:
        for view in self._call_views(scheduled=False):
            view._prepare_live_runtime(dt)

    def _prepare_live_runtime(self, dt) -> None:
        call = self._require_single_call("prepare runtime")
        dt_ms = scalar_decimal(dt, u.ms)
        if dt_ms <= 0.0:
            raise ValueError("Connection runtime dt must be > 0 ms.")
        if call.live_dt_ms is not None:
            if not np.isclose(call.live_dt_ms, dt_ms):
                raise ValueError(
                    "A live Connection cannot change dt while retaining delayed events; "
                    "reset the target cell state first."
                )
            return
        delay_ms = np.asarray(self.delay.to_decimal(u.ms), dtype=float)
        steps = _round_half_up_steps(delay_ms / dt_ms).astype(np.int32)
        call.live_dt_ms = dt_ms
        call.live_delay_steps = steps
        max_steps = int(np.max(steps, initial=0))
        if max_steps > 0:
            call.live_history = brainstate.ShortTermState(jnp.zeros((max_steps, len(self))))

    def reset_runtime(self) -> None:
        for call in (self._store.call(item) for item in self._store.active_call_ids(self._candidate_ids)):
            if call.live_history is not None:
                call.live_history.value = jnp.zeros_like(call.live_history.value)

    def clear_runtime(self) -> None:
        for call in (self._store.call(item) for item in self._store.active_call_ids(self._candidate_ids)):
            call.live_history = None
            call.live_delay_steps = None
            call.live_dt_ms = None

    def _live_event_count(self, *, call: _ConnectionCall, dt):
        self._prepare_live_runtime(dt)
        current = jnp.asarray(call.source.current_event_count(self.source_index))
        steps = call.live_delay_steps
        if steps is None or int(np.max(steps, initial=0)) == 0:
            return current
        rows = np.arange(len(self), dtype=np.int32)
        delayed_row = np.maximum(steps - 1, 0)
        delayed = call.live_history.value[delayed_row, rows]
        output = jnp.where(jnp.asarray(steps) == 0, current, delayed)
        call.live_history.value = jnp.concatenate((current[None, :], call.live_history.value[:-1]), axis=0)
        return output

    def _require_single_call(self, action: str) -> _ConnectionCall:
        # ``active_call_ids`` is this exact expression, one layer down.
        connect_ids = self._store.active_call_ids(self._candidate_ids)
        if len(connect_ids) != 1:
            raise TypeError(f"Cannot {action}: ConnectionView contains {len(connect_ids)} connect calls.")
        return self._store.call(connect_ids[0])

    def _require_homogeneous_synapse_type(self, action: str) -> str:
        types = tuple(dict.fromkeys(str(item) for item in self.synapse_type.tolist()))
        if len(types) == 0:
            raise ValueError(f"Cannot {action} on an empty ConnectionView.")
        if len(types) != 1:
            raise TypeError(
                f"Cannot {action}: ConnectionView contains multiple synapse types {types!r}; use by_synapse_type()."
            )
        return types[0]

    def __repr__(self) -> str:
        names = self.connect_names
        header = (
            f"ConnectionView(cell={_cell_label(self.cell, self.synapse.population_index)}, "
            f"rows={len(self)}, connects={len(names)})"
        )
        lines = [header]
        for name in names[:_CONNECT_CALL_WARNING_THRESHOLD]:
            view = self.by_connect_name(name)
            call = view._require_single_call("format connection")
            lines.append(
                f"  {name}: {call.source.source_type}({call.source.source_name}) -> "
                f"{call.synapse_type}({call.synapse_name}), rows={len(view)}"
            )
        if len(names) > _CONNECT_CALL_WARNING_THRESHOLD:
            lines.append(f"  ... +{len(names) - _CONNECT_CALL_WARNING_THRESHOLD} more")
        return "\n".join(lines)


class NetworkConnections:
    """Aggregate target-owned connections without copying routing rows.

    Parameters
    ----------
    network : Network
        Network whose Cell populations own the concrete connection stores.

    Notes
    -----
    Network-level selection adds only a target-population dimension. Selecting
    a target returns that population's original :class:`ConnectionView`.
    """

    __slots__ = ("_network",)

    def __init__(self, network) -> None:
        self._network = network

    @property
    def target_names(self) -> tuple[str, ...]:
        """Return target population names with active connection rows."""
        return tuple(
            name
            for name, population in self._network.populations.items()
            if population.kind == "cell" and len(population.connections) > 0
        )

    @property
    def n_connections(self) -> int:
        """Return the number of active named connect calls."""
        return sum(len(self[name].connect_names) for name in self.target_names)

    @property
    def n_rows(self) -> int:
        """Return the number of active concrete routing rows."""
        return sum(len(self[name]) for name in self.target_names)

    def __len__(self) -> int:
        return self.n_connections

    def __getitem__(self, selector):
        if isinstance(selector, tuple):
            if len(selector) != 2:
                raise KeyError("Network connections require (target_population, connection_name).")
            target, name = selector
            return self[target][name]
        if not isinstance(selector, str) or not selector:
            raise TypeError("Network connection target must be a non-empty population name.")
        try:
            population = self._network.populations[selector]
        except KeyError as exc:
            raise KeyError(f"Network has no population named {selector!r}.") from exc
        if population.kind != "cell":
            raise TypeError(f"Population {selector!r} owns {population.kind}, not a Cell connection target.")
        return population.connections

    def __repr__(self) -> str:
        lines = [
            f"NetworkConnections(targets={len(self.target_names)}, "
            f"connections={self.n_connections}, rows={self.n_rows})"
        ]
        for target in self.target_names:
            view = self[target]
            lines.append(f"  {target}: connections={len(view.connect_names)}, rows={len(view)}")
            for name in view.connect_names:
                lines.append(f"    {self.describe(target, name)}")
        return "\n".join(lines)

    def describe(self, target: str, name: str) -> str:
        """Return one target-qualified connect-call summary."""
        view = self[target, name]
        call = view._require_single_call("format connection")
        source_name = call.source.source_name
        if not source_name:
            owner = call.source if call.source.is_scheduled else call.source.execution_owner
            source_name = next(
                (population.name for population in self._network.populations.values() if population.model is owner),
                call.source.source_type,
            )
        return (
            f"{name}: {call.source.source_type}({source_name}) -> "
            f"{call.synapse_type}({call.synapse_name}), rows={len(view)}"
        )


def connect(
    name: str,
    *,
    source: EventSource | EventSourceView,
    synapse: SynapseView,
    pairing: PairingSpec | None = None,
    weight=_UNSET,
    delay=0.0 * u.ms,
) -> ConnectionView:
    """Connect aligned event-source and logical-synapse endpoints.

    Parameters
    ----------
    name : str
        Semantic name unique within the destination Cell.
    source : EventSource or EventSourceView
        Ordered source endpoints.
    synapse : SynapseView
        Ordered concrete destination synapses. The view must contain exactly
        one synapse type and one synapse name.
    pairing : PairingSpec, optional
        Endpoint sampling declaration returned by a helper in
        ``braincell.network.connection``. When omitted, endpoints use the existing
        equal-size or singleton broadcasting rule.
    weight : Quantity or array-like, optional
        Scalar event payload, scalar or one value per resulting routing row.
    delay : Quantity, optional
        Non-negative delay, scalar or one value per resulting routing row.

    Returns
    -------
    ConnectionView
        The concrete routing rows created by this call.
    """
    return _connect_with_pairing_seed(
        name,
        source=source,
        synapse=synapse,
        pairing=pairing,
        weight=weight,
        delay=delay,
        pairing_seed_root=0,
        pairing_seed_path=(),
    )


def _connect_with_pairing_seed(
    name,
    *,
    source,
    synapse,
    pairing,
    weight,
    delay,
    pairing_seed_root,
    pairing_seed_path,
):
    _require_name(name, "connection name")
    source_view = _as_source_view(source)
    if not isinstance(synapse, SynapseView):
        raise TypeError(f"connect synapse must be SynapseView, got {type(synapse).__name__!s}.")
    if len(source_view) == 0 or len(synapse) == 0:
        raise ValueError("connect source and synapse cannot be empty.")
    _require_single_synapse_group(synapse)
    if pairing is None:
        source_index, synapse_id, target_index = _align_endpoints(source_view, synapse)
    else:
        pairs = materialize_pairing(
            pairing,
            source_view,
            synapse,
            seed_root=pairing_seed_root,
            seed_path=pairing_seed_path,
        )
        source_index = source_view.source_id[pairs.source_position]
        synapse_id = synapse.id[pairs.synapse_position]
        target_index = pairs.synapse_position
    cell = synapse.cell
    cell._raise_if_initialized("add Connection")
    selected_synapse = SynapseView(cell, synapse_id)
    normalized_weight = _normalize_weight(
        selected_synapse,
        weight,
        count=len(source_index),
        omitted=weight is _UNSET,
    )
    normalized_delay = _normalize_delay(delay, count=len(source_index))
    ids = cell._get_connection_store().add(
        name=name,
        source=source_view.owner,
        source_index=source_index,
        synapse=selected_synapse,
        synapse_id=synapse_id,
        target_index=target_index,
        weight=normalized_weight,
        delay=normalized_delay,
    )
    return ConnectionView(cell._get_connection_store(), ids)


def _as_source_view(source) -> EventSourceView:
    if isinstance(source, Population):
        event_outputs = source.event_outputs
        if len(event_outputs) != 1:
            raise ValueError(
                "A Population can be passed to connect() only when it exposes exactly one event output; "
                "select population.event_outputs[name] explicitly otherwise."
            )
        source = next(iter(event_outputs.values()))
    if isinstance(source, EventSourceView):
        return source
    if isinstance(source, EventSource):
        return source.view
    raise TypeError(f"connect source must be EventSource or EventSourceView, got {type(source).__name__!s}.")


def _align_endpoints(source: EventSourceView, synapse: SynapseView):
    source_ids = source.source_id
    synapse_ids = synapse.id
    source_size = len(source_ids)
    synapse_size = len(synapse_ids)
    if source_size == synapse_size:
        source_index = source_ids
        aligned_synapse_id = synapse_ids
        target_index = np.arange(synapse_size, dtype=np.int64)
    elif source_size == 1:
        source_index = np.full(synapse_size, int(source_ids[0]), dtype=np.int64)
        aligned_synapse_id = synapse_ids
        target_index = np.arange(synapse_size, dtype=np.int64)
    elif synapse_size == 1:
        source_index = source_ids
        aligned_synapse_id = np.full(source_size, int(synapse_ids[0]), dtype=np.int64)
        target_index = np.zeros(source_size, dtype=np.int64)
    else:
        raise ValueError(
            "connect endpoints must have equal lengths or one side must have length 1; "
            "use duplicate-preserving source/synapse indexing to align arbitrary rows; "
            f"got source={source_size!r}, synapse={synapse_size!r}."
        )
    return source_index, aligned_synapse_id, target_index


def _require_single_synapse_group(synapse: SynapseView) -> None:
    types = tuple(dict.fromkeys(str(item) for item in synapse.synapse_type.tolist()))
    names = tuple(dict.fromkeys(str(item) for item in synapse.name.tolist()))
    if len(types) != 1 or len(names) != 1:
        raise ValueError(
            "One connect() call must select exactly one synapse type and one synapse name; "
            f"got types={types!r}, names={names!r}."
        )


def _normalize_delay(value, *, count: int):
    result = _quantity_vector(value, unit=u.ms, size=count, name="Connection.delay")
    if np.any(np.asarray(result.to_decimal(u.ms)) < 0.0):
        raise ValueError("Connection.delay must be >= 0 ms.")
    return result


def _normalize_weight(synapse: SynapseView, value, *, count: int, omitted: bool = False):
    contracts = []
    for synapse_type in synapse.synapse_type.tolist():
        runtime_cls = get_registry().get("synapse", str(synapse_type))
        contract = getattr(runtime_cls, "event_input", None)
        if contract is None:
            raise TypeError(f"Synapse type {synapse_type!r} does not declare event_input.")
        contracts.append(contract)
    first = contracts[0]
    if any(contract != first for contract in contracts[1:]):
        raise ValueError("Connection synapse mixes incompatible event-input contracts.")
    if isinstance(first, NoEventInput):
        raise TypeError("Connection synapse model does not declare a discrete event input.")
    if isinstance(first, TriggerEventInput):
        if omitted or value is None:
            return None
        raise TypeError("Trigger-only Connection synapses require weight=None.")
    if not isinstance(first, ScalarEventInput):
        raise TypeError(f"Unsupported synapse event-input contract {type(first).__name__!s}.")
    if omitted:
        value = 1.0 * first.unit
    if value is None:
        raise TypeError("Scalar-event Connection weight cannot be None.")
    return _quantity_vector(value, unit=first.unit, size=count, name="Connection.weight")


def _concat_quantities(left, right, *, unit):
    return u.Quantity(
        np.concatenate((np.asarray(left.to_decimal(unit)), np.asarray(right.to_decimal(unit)))),
        unit,
    )


def _set_quantity_or_array(destination, rows, values):
    if isinstance(destination, u.Quantity):
        unit = destination.unit
        decimal = np.array(destination.to_decimal(unit), copy=True)
        decimal[rows] = np.asarray(values.to_decimal(unit))
        return u.Quantity(decimal, unit)
    result = np.array(destination, copy=True)
    result[rows] = np.asarray(values)
    return result


def _stack_values(values):
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(np.asarray([value.to_decimal(unit) for value in values]), unit)
    return np.asarray(values)


def _split_values(value, count: int):
    if isinstance(value, u.Quantity):
        unit = value.unit
        decimal = np.asarray(value.to_decimal(unit))
        return tuple(u.Quantity(decimal[index], unit) for index in range(count))
    array = np.asarray(value)
    return tuple(array[index] for index in range(count))
