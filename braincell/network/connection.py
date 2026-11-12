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

import dataclasses
from dataclasses import dataclass
import warnings

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._misc import require_name as _require_name, scalar_decimal
from braincell._parameter_schema import RuntimeParameterState
from braincell._multi_compartment.synapses import SynapseView, _cell_label
from braincell._multi_compartment.lifecycle import DiscreteView
from braincell.network.pairing import (
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
from braincell.network.core import Population
from braincell.network.event import (
    EventSource,
    EventSourceView,
    _quantity_vector,
    round_half_up_steps_host as _round_half_up_steps,
)
from braincell.network.plasticity import bind_rule

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
    #: Identity of each row's synapse (``_SynapseStore.identity``) and the
    #: sites of each place rule it was connected on, as of ``connect`` (MORPH-15).
    synapse_identity: tuple | None = None
    synapse_sites: dict | None = None
    #: ``_SynapseStore.serial`` of the store the rows' synapse ids refer to.
    synapse_store_serial: int = -1
    #: The call's plasticity rule (``network.plasticity.RuleBinding``), or ``None``.
    plasticity: object = None

    def __post_init__(self):
        if self.weight is not None:
            object.__setattr__(self, "weight", RuntimeParameterState(self.weight))

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return value.value if name == "weight" and isinstance(value, RuntimeParameterState) else value

    def __setattr__(self, name, value):
        state = vars(self).get(name)
        if name == "weight" and isinstance(state, RuntimeParameterState):
            state.value = value
        else:
            object.__setattr__(self, name, value)


def _store_column(name: str, doc: str) -> property:
    def read(store: "_ConnectionStore"):
        if store._pending:
            store._concatenate_pending()
        return store._columns[name]

    return property(read, doc=doc)


class _ConnectionStore:
    """Cell-owned SoA columns for all direct event-routing rows.

    ``add`` keeps each call's columns aside and the first read concatenates
    everything kept since the last one (CELL-18e): re-concatenating every
    column on every call made ``n`` connects cost O(n x rows). Reads return
    the concatenated arrays themselves, so writes into them (``remove``,
    ``follow_synapses``) change the store, as before.
    """

    __slots__ = (
        "cell",
        "_columns",
        "_pending",
        "_n_rows",
        "_calls",
        "_call_by_id",
        "_call_by_name",
        "_next_row_id",
        "_next_connect_id",
        "_warning_emitted",
    )

    id = _store_column("id", "Stable row id of each row.")
    connect_id = _store_column("connect_id", "Id of the connect call each row came from.")
    source_index = _store_column("source_index", "Index of each row's event source member.")
    synapse_id = _store_column("synapse_id", "Logical id of each row's synapse.")
    target_index = _store_column("target_index", "Index of each row in the SynapseView passed to connect().")
    delay = _store_column("delay", "Delay of each row, in ms.")
    active = _store_column("active", "Whether each row is still connected.")
    _row_of_id = _store_column("_row_of_id", "Row of each stable row id.")

    def __init__(self, cell) -> None:
        self.cell = cell
        self._columns = {
            "id": np.asarray([], dtype=np.int64),
            "connect_id": np.asarray([], dtype=np.int64),
            "source_index": np.asarray([], dtype=np.int64),
            "synapse_id": np.asarray([], dtype=np.int64),
            "target_index": np.asarray([], dtype=np.int64),
            "delay": np.asarray([], dtype=np.float64) * u.ms,
            "active": np.asarray([], dtype=bool),
            "_row_of_id": np.asarray([], dtype=np.int64),
        }
        self._pending: list[dict[str, np.ndarray]] = []
        self._n_rows = 0
        self._calls: list[_ConnectionCall] = []
        self._call_by_id: dict[int, _ConnectionCall] = {}
        self._call_by_name: dict[str, _ConnectionCall] = {}
        self._next_row_id = 0
        self._next_connect_id = 0
        self._warning_emitted = False

    def _concatenate_pending(self) -> None:
        pending, self._pending = self._pending, []
        columns = self._columns
        for name, column in columns.items():
            if name == "delay":
                # Joined as ms decimals, as each call's delay was converted in ``add``.
                decimals = [np.asarray(column.to_decimal(u.ms)), *(chunk[name] for chunk in pending)]
                columns[name] = u.Quantity(np.concatenate(decimals), u.ms)
            else:
                columns[name] = np.concatenate([column, *(chunk[name] for chunk in pending)])

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
        plasticity=None,
    ) -> np.ndarray:
        if name in self._call_by_name:
            raise ValueError(f"Connection name {name!r} is already used on Cell {self.cell.name!r}.")
        count = len(source_index)
        connect_id = self._next_connect_id
        self._next_connect_id += 1
        row_ids = np.arange(self._next_row_id, self._next_row_id + count, dtype=np.int64)
        self._next_row_id += count

        start = self._n_rows
        self._pending.append(
            {
                "id": row_ids,
                "connect_id": np.full(count, connect_id, dtype=np.int64),
                "source_index": np.asarray(source_index, dtype=np.int64),
                "synapse_id": np.asarray(synapse_id, dtype=np.int64),
                "target_index": np.asarray(target_index, dtype=np.int64),
                "delay": np.asarray(delay.to_decimal(u.ms)),
                "active": np.ones(count, dtype=bool),
                "_row_of_id": np.arange(start, start + count, dtype=np.int64),
            }
        )
        self._n_rows += count

        synapses = synapse._store
        identity = synapses.identity(synapses.rows_of(synapse_id)[0])
        call = _ConnectionCall(
            id=connect_id,
            name=name,
            source=source,
            synapse_type=str(synapse.synapse_type[0]),
            synapse_name=str(synapse.name[0]),
            row_ids=row_ids,
            weight=weight,
            synapse_identity=identity,
            synapse_sites={
                key: synapses.sites[key] for key in dict.fromkeys(zip(identity[0].tolist(), identity[1].tolist()))
            },
            synapse_store_serial=synapses.serial,
            plasticity=plasticity,
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

    def follow_synapses(self, synapses) -> None:
        """Point every active row at the synapse it was connected to, on the current grid.

        Logical synapse ids number placements across all place rules, so a
        rebuild that gives an earlier rule another site shifts every later
        id (MORPH-15). Each row keeps its synapse's identity (place rule,
        mechanism, member, branch name, ``branch_x``, repeat) and is moved
        to the id that synapse has now. Nothing changes when no id moved.

        Raises
        ------
        RuntimeError
            Before changing any row, naming the first connection whose
            place rule now resolves to other sites than when it was
            connected: its rows could not all reach the same synapses.
        """
        updates = []
        for call in self._calls:
            if call.synapse_store_serial == synapses.serial:
                continue
            rows = self._row_of_id[call.row_ids]
            active = self.active[rows]
            if not active.any():
                continue
            for key, sites in call.synapse_sites.items():
                now = synapses.sites.get(key, ())
                if now != sites:
                    raise RuntimeError(
                        f"Connection {call.name!r} targets synapse group {call.synapse_name!r} "
                        f"({call.synapse_type}), whose Cell.place(...) rule now resolves to {len(now)} sites "
                        f"({_describe_sites(now)}); when connected it resolved to {len(sites)} "
                        f"({_describe_sites(sites)}). Its rows would leave sites unconnected or reach none. "
                        f"Undo the edit, or remove the connection (cell.connections[{call.name!r}].remove()) and "
                        "connect the group again under a new name."
                    )
            identity = tuple(column[active] for column in call.synapse_identity)
            rows = rows[active]
            found_rows, found = synapses.rows_of(self.synapse_id[rows])
            if found.all() and _same_identity(synapses.identity(found_rows), identity):
                updates.append((call, rows, None))
                continue
            ids = synapses.ids_with_identity(identity)
            # The sites match, so every identity is still there.
            assert (ids >= 0).all(), call.name
            updates.append((call, rows, ids))
        for call, rows, ids in updates:
            if ids is not None:
                self.synapse_id[rows] = ids
            call.synapse_store_serial = synapses.serial

    def active_ids(self, ids=None) -> np.ndarray:
        ids = self.id if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.asarray([], dtype=np.int64)
        rows = self.rows(ids)
        return ids[self.active[rows]]

    def call(self, connect_id: int) -> _ConnectionCall:
        return self._call_by_id[int(connect_id)]

    def call_names(self) -> np.ndarray:
        """Each call's name at the position of its connect id."""
        names = np.empty(self._next_connect_id, dtype=object)
        for call in self._calls:
            names[call.id] = call.name
        return names

    def active_call_ids(self, ids) -> tuple[int, ...]:
        rows = self.rows(self.active_ids(ids))
        return tuple(dict.fromkeys(int(item) for item in self.connect_id[rows].tolist()))

    def weight_for(self, ids):
        ids = self.active_ids(ids)
        if ids.size == 0:
            raise ValueError("Cannot access weight on an empty ConnectionView.")
        connect_id = self.connect_id[self.rows(ids)]
        call_ids = list(dict.fromkeys(connect_id.tolist()))
        calls = [self.call(int(call_id)) for call_id in call_ids]
        trigger = [call.weight is None for call in calls]
        if any(trigger):
            if not all(trigger):
                raise TypeError("ConnectionView mixes trigger-only and scalar-weight rows.")
            return None
        # ``add`` builds ``row_ids`` as one ``np.arange``, so a row's slot
        # in its call's weight vector is its offset from that call's first
        # row. Searching for it cost O(rows) per row. One gather per call:
        # stacking one scalar per row built an XLA op with an operand per
        # row, whose compile took 33 s at 1e4 rows and 245 s at 3e4 (CELL-12a).
        first = calls[0].weight
        unit = first.unit if isinstance(first, u.Quantity) else None
        positions, pieces = [], []
        for call_id, call in zip(call_ids, calls):
            position = np.flatnonzero(connect_id == call_id)
            value = call.weight[ids[position] - int(call.row_ids[0])]
            positions.append(position)
            pieces.append(jnp.asarray(value.to_decimal(unit)) if unit is not None else jnp.asarray(value))
        if len(pieces) == 1:
            mantissa = pieces[0]
        else:
            mantissa = jnp.concatenate(pieces)[np.argsort(np.concatenate(positions), kind="stable")]
        return mantissa if unit is None else u.Quantity(mantissa, unit)

    def set_weight(self, ids, values) -> None:
        ids = self.active_ids(ids)
        split = _split_values(values, len(ids))
        connect_id = self.connect_id[self.rows(ids)]
        # ``_set_quantity_or_array`` rebuilds the call's whole weight
        # vector, so doing it once per row was quadratic in the rows of a
        # single call. Group first, then assign each call's slots at once.
        # Every call's new weight is checked before any is assigned, so a
        # weight a plasticity rule rejects leaves all calls as they were.
        updates = []
        for call_id in np.unique(connect_id).tolist():
            mask = connect_id == call_id
            call = self.call(int(call_id))
            selected = [value for value, keep in zip(split, mask.tolist()) if keep]
            offsets = ids[mask] - int(call.row_ids[0])
            weight = _set_quantity_or_array(call.weight, offsets, _stack_values(selected))
            if call.plasticity is not None:
                call.plasticity.rule.check_weight(weight)
            updates.append((call, offsets, weight))
        for call, offsets, weight in updates:
            call.weight = weight
            if call.plasticity is not None:
                call.plasticity.set_weight(offsets, weight)

    def rule_calls(self) -> tuple[_ConnectionCall, ...]:
        """The calls with a plasticity rule and at least one active row."""
        return tuple(
            call
            for call in self._calls
            if call.plasticity is not None and bool(self.active[self.rows(call.row_ids)].any())
        )


def _same_identity(first, second) -> bool:
    return all(np.array_equal(a, b) for a, b in zip(first, second))


def _describe_sites(sites, limit: int = 4) -> str:
    """``dend(1), axon(1)`` for a place rule's sites; ``[member]`` prefixes a per-member site."""
    parts = [f"{'' if member < 0 else f'[{member}] '}{branch}({x:g})" for member, branch, x in sites[:limit]]
    return ", ".join(parts) + (", ..." if len(sites) > limit else "")


class ConnectionView(DiscreteView):
    """View an ordered selection of concrete event-routing rows."""

    __slots__ = ("_store", "_ids")

    def __init__(self, store: _ConnectionStore, ids=None) -> None:
        self._store = store
        self._ids = None if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)
        self._bind_view(store.cell)

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
        return self._store.call_names()[self.connect_id]

    @property
    def connect_names(self) -> tuple[str, ...]:
        return tuple(call.name for call, _ in self._call_groups())

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
        self._check_view()
        return int(self._active_ids.size)

    def __getitem__(self, selector) -> "ConnectionView":
        self._check_view()
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

    def trainable(self, **fields):
        """Bind weight sources to selected contacts; delay remains static.

        Parameters
        ----------
        **fields
            ``weight`` mapped to a trainable parameter source.

        Returns
        -------
        ConnectionView
            This selection.
        """
        from braincell.trainable._targets import register_connection

        register_connection(self, fields)
        return self

    def set(self, *, weight=_UNSET, delay=_UNSET) -> "ConnectionView":
        """Update existing runtime weights without modifying routing or delay.

        On rows with a plasticity rule, the new weight is both the weight
        ``reset_state`` returns to and the current dynamic weight
        (``rule_state("w")``).

        Raises
        ------
        ValueError
            If a plasticity rule rejects the new weight (outside its
            ``w_min``/``w_max``, for example); no row changes.
        """
        self.cell._raise_if_not_initialized(
            "ConnectionView.set()", hint="Before init_state(), give the values to connect()."
        )
        if delay is not _UNSET:
            raise RuntimeError("Connection delay is structural; configure it in connect() before init_state().")
        if weight is not _UNSET:
            from braincell.trainable._targets import require_unbound

            require_unbound(self.cell, "connection", "weight", self.id, "weight")
            self._require_homogeneous_synapse_type("modify weight")
            normalized = _normalize_weight(self.synapse, weight, count=len(self), omitted=False)
            current = self._store.weight_for(self._active_ids)
            if (current is None) != (normalized is None):
                raise TypeError("Connection weight cannot change the target event-input payload kind.")
            if normalized is not None:
                self._store.set_weight(self._active_ids, normalized)
        return self

    def remove(self) -> None:
        """Remove selected rows while leaving stable row IDs and names unused."""
        self.cell._raise_if_initialized("remove Connection")
        self._store.active[self._store.rows(self._candidate_ids)] = False

    def _call_groups(self) -> tuple[tuple[_ConnectionCall, np.ndarray], ...]:
        """Each connect call of this view, in first-row order, with its row ids.

        One grouping pass over the rows: asking ``_active_ids`` and
        ``connect_id``, which walk every row, once per call cost
        O(calls x rows).
        """
        ids = self._active_ids
        if ids.size == 0:
            return ()
        connect_id = self._store.connect_id[self._store.rows(ids)]
        call_ids, first_seen, inverse = np.unique(connect_id, return_index=True, return_inverse=True)
        sorter = np.argsort(inverse, kind="stable")
        starts = np.searchsorted(inverse[sorter], np.arange(call_ids.size))
        grouped = np.split(ids[sorter], starts[1:])
        # ``np.unique`` sorts; calls come in first-appearance order.
        return tuple(
            (self._store.call(int(call_ids[position])), grouped[position])
            for position in np.argsort(first_seen).tolist()
        )

    def _call_views(self, *, scheduled=None) -> tuple["ConnectionView", ...]:
        """Split this view into one sub-view per originating ``connect()`` call.

        ``Network.run`` reaches here on every call, as does ``Cell`` once
        per synapse layout.
        """
        return tuple(view for _, view in self._call_items(scheduled=scheduled))

    def _call_items(self, *, scheduled=None) -> tuple[tuple[_ConnectionCall, "ConnectionView"], ...]:
        """``(call, sub-view)`` for each originating ``connect()`` call, as :meth:`_call_views`."""
        return tuple(
            (call, ConnectionView(self._store, ids))
            for call, ids in self._call_groups()
            if scheduled is None or bool(call.source.is_scheduled) == bool(scheduled)
        )

    def rule_state(self, name: str):
        """Return one state of the plasticity rule of these rows.

        Parameters
        ----------
        name : str
            A state of the rule: ``"w"``, the dynamic weight, for every rule;
            ``PairSTDP`` also has the traces ``"x_pre"`` and ``"y_post"``.

        Returns
        -------
        brainunit.Quantity or jax.Array
            One value per row, in the order of this view; ``"w"`` carries the
            weight's unit. Inside a compiled loop the value is traced.

        Raises
        ------
        RuntimeError
            Before the Cell's ``init_state()``: the state exists from then on.
        TypeError
            If a row comes from a ``connect`` call without a rule.
        KeyError
            If the rule has no state ``name``.
        ValueError
            On an empty view.

        Examples
        --------
        >>> import brainunit as u
        >>> import braincell
        >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
        >>> cell = cell.place(braincell.mech.Synapse("ExpSyn", name="s"))
        >>> rule = braincell.mech.ConnectionRule(
        ...     "PairSTDP",
        ...     A_plus=0.01 * u.uS,
        ...     A_minus=0.0105 * u.uS,
        ...     tau_plus=20.0 * u.ms,
        ...     tau_minus=20.0 * u.ms,
        ... )
        >>> source = braincell.EventSequence.from_times(([1.0] * u.ms,))
        >>> view = braincell.connect("c", source=source, synapse=cell.synapses["s"], weight=0.05 * u.uS, plasticity=rule)
        >>> cell.init_state()
        >>> print(cell.connections["c"].rule_state("w"))
        [0.05] uS
        """
        self.cell._raise_if_not_initialized("ConnectionView.rule_state()")
        if self._active_ids.size == 0:
            raise ValueError("Cannot read rule state on an empty ConnectionView.")
        groups = self._rule_groups()
        return _join_rule_pieces(
            [(positions, call.plasticity.value(name), offsets) for call, positions, offsets in groups],
            len(self),
        )

    def _rule_groups(self):
        """``(call, positions, offsets)`` per rule call: the view positions of its rows and their slots in its state.

        Raises
        ------
        TypeError
            If a row comes from a ``connect`` call without a rule.
        """
        ids = self._active_ids
        connect_id = self._store.connect_id[self._store.rows(ids)]
        groups = []
        for call_id in dict.fromkeys(connect_id.tolist()):
            call = self._store.call(int(call_id))
            if call.plasticity is None:
                raise TypeError(f"Connection {call.name!r} has no plasticity rule, so it has no rule state.")
            positions = np.flatnonzero(connect_id == call_id)
            groups.append((call, positions, ids[positions] - int(call.row_ids[0])))
        return groups

    def _with_ids(self, ids) -> "ConnectionView":
        """The rows ``ids`` that are still connected, in the given order."""
        return ConnectionView(self._store, np.asarray(ids, dtype=np.int64))

    def _rule_type_names(self) -> list[str]:
        """The registered name of each row's rule."""
        names = {}
        for call, positions, _ in self._rule_groups():
            rule_type = call.plasticity.declaration.rule_type
            name = rule_type if isinstance(rule_type, str) else getattr(rule_type, "__name__", repr(rule_type))
            names.update(dict.fromkeys(positions.tolist(), name))
        return [names[index] for index in range(len(names))]

    def _rule_state_sampler(self, name: str):
        """A recording sampler of the rule state ``name`` of these rows, before each boundary's update.

        A Cell prepares a boundary's inputs, and so runs its rules, either
        before a step samples its recordings (``Cell.run`` prepares the next
        boundary at the end of each step) or after (``Network`` prepares it
        within the step). The sampler reads the state the rule's update got
        at the boundary of the step: the copy a Cell outside a Network keeps
        for a recorded state while that boundary is prepared, and the state
        itself otherwise.

        Raises
        ------
        RuntimeError
            When sampled on a Cell outside a Network that a Network
            initialized: it has no copy.
        """
        groups = self._rule_groups() if len(self) else []
        cell = self.cell
        size = len(self)

        def sample():
            if not groups:
                return jnp.zeros((0,), dtype=brainstate.environ.dftype())
            if cell.network_owner is None and any(name not in call.plasticity.before for call, _, _ in groups):
                raise RuntimeError(
                    f"Cell.run cannot record rule state {name!r}: a Network initialized this Cell. Call reset() "
                    "and init_state() first."
                )
            prepared = cell._next_synapse_inputs_prepared.value
            return _join_rule_pieces(
                [
                    (positions, call.plasticity.value_before_boundary(name, prepared), offsets)
                    for call, positions, offsets in groups
                ],
                size,
            )

        return sample

    def record(self, name: str, state: str, *, period=None, frequency=None, start=0.0 * u.ms):
        """Record one plasticity-rule state of these rows, as ``Cell.record`` records a state.

        Parameters
        ----------
        name : str
            Recording name, unique in the target Cell.
        state : str
            A state of the rows' rule: ``"w"``, the dynamic weight, for every
            rule; ``PairSTDP`` also has ``"x_pre"`` and ``"y_post"``.
        period, frequency : Quantity, optional
            Mutually exclusive sampling interval; every ``dt`` when both are
            omitted.
        start : Quantity, optional
            Start of the sampling schedule.

        Returns
        -------
        RecordingSpec
            The declaration, owned by the target Cell.

        Notes
        -----
        The samples have one column per row of this view still connected
        at the first run, in view order: column ``j`` is the row whose
        value ``rule_state(state)[j]`` reads. Sample ``k`` is the state
        just before the update at the boundary ``time[k]``, the ``state``
        the rule's ``update`` receives there, and sample 0 of the first run
        is the initial state. For ``"w"`` it is the weight that the events
        arriving at ``time[k]`` are delivered with. A trace is the value the
        rule keeps: ``PairSTDP`` decays its traces at the start of each
        update, so ``"x_pre"`` at ``time[k]`` is the trace just after the
        boundary before, and ``exp(-dt / tau_plus)`` times it is the trace
        just before ``time[k]``.

        A ``Network`` and a standalone ``Cell.run`` record the same samples,
        but ``rule_state`` reads the state itself, which the two update at
        different points of a step. In a ``Network``, ``rule_state`` read
        before the ``k``-th ``update()`` (from 0) is sample ``k``. A
        standalone Cell updates the next boundary at the end of each step:
        ``rule_state`` read after its ``j``-th ``update()`` is sample
        ``j + 2``, and after a ``run`` of ``n`` steps it is the sample a
        longer run takes at step ``n + 1``. For this, a Cell outside a
        Network keeps a copy of each recorded state from before the latest
        update, one more state in its state tree.

        Each row of the schema (``samples[name].schema.rows``) gives its
        synapse's site and ``synapse_id``, ``mechanism_category``
        ``"connection_rule"``, the rule as ``mechanism_type``, the connect
        name as ``mechanism_name``, and the row's ``connection_id`` (its
        ``id``) and ``source_index``.

        Raises
        ------
        RuntimeError
            After the target Cell's ``init_state()``; and at ``Cell.run``, on
            a Cell that a Network initialized, once that Network is gone:
            call ``reset()`` and ``init_state()`` first.
        TypeError
            If a row comes from a ``connect`` call without a rule, or the
            rows' ``state`` differs in dimension between their calls.
        KeyError
            If the rule has no state ``state``.
        ValueError
            On an empty view, a ``state`` that is not a non-empty string, a
            ``name`` the Cell already records, or an invalid schedule.

        Examples
        --------
        >>> import brainunit as u
        >>> import braincell
        >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
        >>> cell = cell.place(braincell.mech.Synapse("ExpSyn", name="s"))
        >>> rule = braincell.mech.ConnectionRule(
        ...     "PairSTDP",
        ...     A_plus=0.01 * u.uS,
        ...     A_minus=0.0105 * u.uS,
        ...     tau_plus=20.0 * u.ms,
        ...     tau_minus=20.0 * u.ms,
        ...     signals={"post": braincell.EventSequence.from_times(([0.3] * u.ms,))},
        ... )
        >>> source = braincell.EventSequence.from_times(([0.1] * u.ms,))
        >>> view = braincell.connect("c", source=source, synapse=cell.synapses["s"], weight=0.05 * u.uS, plasticity=rule)
        >>> _ = view.record("w", "w")
        >>> result = cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        >>> print(result.samples["w"].values[:, 0])
        [0.05      0.05      0.05      0.05      0.0599005] uS

        The pre event arrives at 0.1 ms and the post event at 0.3 ms; the
        sample at 0.4 ms is the first after the boundary at 0.3 ms, so it
        holds ``0.05 + 0.01 * exp(-0.2 / 20)`` uS.
        """
        from braincell.network.recording import RecordingSpec, _RuleStateObservable

        cell = self.cell
        cell._raise_if_initialized("add a recording")
        _require_name(state, "rule state")
        if len(self) == 0:
            raise ValueError("Cannot record rule state on an empty ConnectionView.")
        first = None
        for call, _, _ in self._rule_groups():
            initial = call.plasticity.initial_state(call.weight)
            if state not in initial:
                raise KeyError(
                    f"Connection rule {call.plasticity.declaration.rule_type!r} of connection {call.name!r} has no "
                    f"state {state!r}; its states are {tuple(initial)!r}."
                )
            unit = u.get_unit(initial[state])
            if first is None:
                first = (call.name, unit)
            elif unit.dim != first[1].dim:
                raise TypeError(
                    f"Rule state {state!r} is in {unit} on connection {call.name!r} and in {first[1]} on "
                    f"{first[0]!r}; one recording takes rows of one dimension: record them separately."
                )
        return cell._add_recording(
            RecordingSpec(
                name=name,
                scope=cell._root_scope(),
                observable=_RuleStateObservable(tuple(int(item) for item in self._active_ids.tolist()), state),
                period=period,
                frequency=frequency,
                start=start,
            )
        )

    def event_count(self, *, t, dt):
        """Return one arrival count per row for a single connect call."""
        call = self._require_single_call("evaluate events")
        if not call.source.is_scheduled:
            return self._live_event_count(call=call, dt=dt)
        return call.source.event_count(self.source_index, t=t, delay=self.delay, dt=dt)

    def prepare_runtime(self, dt) -> None:
        for call, _ in self._call_items(scheduled=False):
            if call.plasticity is not None:
                # Raised before any live history is allocated.
                raise NotImplementedError(
                    f"Connection {call.name!r} has a plasticity rule on a live source, which runs only in a "
                    "braincell.Network: a standalone Cell.run delivers live events on another schedule. Add the "
                    "Cells to a Network and run that, or connect a scheduled source (EventSequence, NetStim)."
                )
        for view in self._call_views(scheduled=False):
            view._prepare_live_runtime(dt)

    def _init_rules(self) -> dict | None:
        """Allocate the state of every plasticity rule (``Cell.init_state``).

        Returns
        -------
        dict or None
            ``{connect name: states}`` for the Cell to keep, so that its state
            tree holds them; ``None`` without a rule.
        """
        # Only Cell.run samples a step after running the rules at its boundary
        # (it prepares the next boundary at the end of a step), so only a Cell
        # outside a Network keeps the state before each update.
        recorded = self._recorded_rule_states() if self.cell.network_owner is None else {}
        rules = {}
        for call in self._store.rule_calls():
            rules[call.name] = call.plasticity.init(self.cell, call, self._store, recorded=recorded.get(call.id, ()))
        return rules or None

    def _recorded_rule_states(self) -> dict[int, tuple[str, ...]]:
        """``{connect id: state names}`` that the Cell's recordings record (``ConnectionView.record``)."""
        from braincell.network.recording import _RuleStateObservable

        recorded: dict[int, dict[str, None]] = {}
        for spec in self.cell._recording_specs.values():
            observable = spec.observable
            if not isinstance(observable, _RuleStateObservable):
                continue
            ids = self._store.active_ids(observable.row_ids)
            for call_id in np.unique(self._store.connect_id[self._store.rows(ids)]).tolist():
                recorded.setdefault(int(call_id), {})[observable.field] = None
        return {call_id: tuple(names) for call_id, names in recorded.items()}

    def _prepare_rules(self, dt, *, delay_quantization: str) -> None:
        """Prepare the live plasticity rows for a Network run setup."""
        for call in self._store.rule_calls():
            call.plasticity.prepare_network(call, dt=dt, delay_quantization=delay_quantization)

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
            if call.plasticity is not None:
                call.plasticity.reset(call.weight)

    def clear_runtime(self) -> None:
        for call in (self._store.call(item) for item in self._store.active_call_ids(self._candidate_ids)):
            call.live_history = None
            call.live_delay_steps = None
            call.live_dt_ms = None
            if call.plasticity is not None:
                call.plasticity.clear()

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
        calls = self._call_groups()
        header = (
            f"ConnectionView(cell={_cell_label(self.cell, self.synapse.population_index)}, "
            f"rows={len(self)}, connects={len(calls)})"
        )
        lines = [header]
        for call, ids in calls[:_CONNECT_CALL_WARNING_THRESHOLD]:
            lines.append(
                f"  {call.name}: {call.source.source_type}({call.source.source_name}) -> "
                f"{call.synapse_type}({call.synapse_name}), rows={ids.size}"
            )
        if len(calls) > _CONNECT_CALL_WARNING_THRESHOLD:
            lines.append(f"  ... +{len(calls) - _CONNECT_CALL_WARNING_THRESHOLD} more")
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
        return sum(len(self[name]._call_groups()) for name in self.target_names)

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
        summaries = self._summaries()
        lines = [
            f"NetworkConnections(targets={len(summaries)}, "
            f"connections={sum(len(calls) for _, _, calls in summaries)}, "
            f"rows={sum(rows for _, rows, _ in summaries)})"
        ]
        for target, rows, calls in summaries:
            lines.append(f"  {target}: connections={len(calls)}, rows={rows}")
            lines.extend(f"    {line}" for line in calls)
        return "\n".join(lines)

    def _summaries(self) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
        """``(target, rows, one describe() line per connect call)`` per target, reading each row once."""
        summaries = []
        for target in self.target_names:
            view = self[target]
            calls = view._call_groups()
            summaries.append(
                (
                    target,
                    sum(ids.size for _, ids in calls),
                    tuple(self._describe(call, ids.size) for call, ids in calls),
                )
            )
        return tuple(summaries)

    def describe(self, target: str, name: str) -> str:
        """Return one target-qualified connect-call summary."""
        view = self[target, name]
        return self._describe(view._require_single_call("format connection"), len(view))

    def _describe(self, call: _ConnectionCall, rows: int) -> str:
        source_name = call.source.source_name
        if not source_name:
            owner = call.source if call.source.is_scheduled else call.source.execution_owner
            source_name = next(
                (population.name for population in self._network.populations.values() if population.model is owner),
                call.source.source_type,
            )
        return f"{call.name}: {call.source.source_type}({source_name}) -> {call.synapse_type}({call.synapse_name}), rows={rows}"


def connect(
    name: str,
    *,
    source: EventSource | EventSourceView,
    synapse: SynapseView,
    pairing: PairingSpec | None = None,
    weight=_UNSET,
    delay=0.0 * u.ms,
    plasticity=None,
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
        equal-size or singleton broadcasting rule. A rule without its own
        ``seed`` draws one integer from the BrainState default random
        stream here and samples exactly as if it had been given that seed,
        so an enclosing ``brainstate.random.seed_context(s)`` (or an earlier
        ``brainstate.random.seed(s)``) controls the rows. ``Network.connect``
        derives the seed from the Network's root instead.
    weight : Quantity or array-like, optional
        Scalar event payload, scalar or one value per resulting routing row.
    delay : Quantity, optional
        Non-negative delay, scalar or one value per resulting routing row.
    plasticity : braincell.mech.ConnectionRule, optional
        A plasticity rule for every row of this call: the rule changes each
        row's weight as the model runs, and ``rule_state("w")`` reads it
        (see Notes). ``None`` (default) keeps the weight fixed.

    Returns
    -------
    ConnectionView
        The concrete routing rows created by this call.

    Raises
    ------
    TypeError
        If ``pairing`` has no seed and the call runs inside
        ``brainstate.transform.jit``, ``for_loop`` or another
        transformation, where the default stream is traced. Call
        ``connect`` outside the transformation. If ``plasticity`` is not a
        ``ConnectionRule``, the synapse is trigger-only (``weight=None``),
        or the rule binds a signal it does not declare.
    KeyError
        If no rule is registered under ``plasticity``'s name.
    ValueError
        If the rule rejects its parameters or the initial weight (a unit
        incompatible with the weight, a non-positive time constant, a weight
        outside the bounds), or its ``post`` source has another number of
        endpoints than the target Cell has members or is a live source of
        another model.
    RuntimeError
        At the destination Cell's ``init_state`` (or ``Network.init_state``,
        before any Cell initializes), if the grid was rebuilt after this call
        and the ``Cell.place`` rule of the connected synapses now resolves to
        other sites (an edit gave it a site more or less). The rows could not
        all reach the synapses they were made for.

    Notes
    -----
    Each row remembers its synapse by place rule, mechanism, population
    member, branch name and ``branch_x``. When the grid is rebuilt before
    initialization -- an edit of the morphology, or a later ``place`` -- the
    logical synapse ids can shift, and ``init_state`` moves each row to the
    id its synapse has now. A branch removed and attached again under the
    same name keeps its synapses' rows.

    With ``plasticity``, the target Cell runs the rule once per event
    boundary, before its synapses consume that boundary's events. A row's
    pre time is its arrival (after the delay), and the event delivered
    there carries the weight before that boundary's update.
    ``init_state`` and ``reset_state`` set the dynamic weight to the rows'
    weight (``ConnectionView.weight``, or the value a bound trainable gives
    it); ``set(weight=...)`` changes both. A rule on a live source runs in a
    Network only.

    Examples
    --------
    >>> import brainunit as u
    >>> import braincell
    >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
    >>> cell = cell.place(braincell.mech.Synapse("ExpSyn", name="s"))
    >>> source = braincell.EventSequence.from_times(([1.0, 3.0] * u.ms,))
    >>> view = braincell.connect("drive", source=source, synapse=cell.synapses["s"], weight=0.01 * u.uS)
    >>> len(view)
    1
    """
    return _connect_with_pairing_seed(
        name,
        source=source,
        synapse=synapse,
        pairing=pairing,
        weight=weight,
        delay=delay,
        plasticity=plasticity,
        pairing_seed_root=None,
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
    plasticity=None,
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
        if pairing_seed_root is None:
            # A direct ``connect``: a rule without its own seed draws one from the default stream.
            if pairing.seed is None:
                from braincell._random import draw_seed

                seed = draw_seed(
                    "braincell.connect with a pairing rule without a seed",
                    remedy="Call connect outside the transformation.",
                )
                pairing = dataclasses.replace(pairing, seed=seed)
            pairing_seed_root = 0
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
    rule = None
    if plasticity is not None:
        rule = bind_rule(plasticity, cell=cell, weight=normalized_weight, n_rows=len(source_index))
    ids = cell._get_connection_store().add(
        name=name,
        source=source_view.owner,
        source_index=source_index,
        synapse=selected_synapse,
        synapse_id=synapse_id,
        target_index=target_index,
        weight=normalized_weight,
        delay=normalized_delay,
        plasticity=rule,
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


def _set_quantity_or_array(destination, rows, values):
    if isinstance(destination, u.Quantity):
        unit = destination.unit
        decimal = np.array(destination.to_decimal(unit), copy=True)
        decimal[rows] = np.asarray(values.to_decimal(unit))
        return u.Quantity(decimal, unit)
    result = np.array(destination, copy=True)
    result[rows] = np.asarray(values)
    return result


def _join_rule_pieces(pieces, size: int):
    """One value per view row from ``(positions, call state, offsets)`` per rule call, in the first call's unit."""
    selected = []
    for positions, value, offsets in pieces:
        if not (offsets.size == value.shape[0] and np.array_equal(offsets, np.arange(offsets.size))):
            value = value[offsets]
        selected.append((positions, value))
    if len(selected) == 1:
        return selected[0][1]
    unit = u.get_unit(selected[0][1])
    mantissas = [jnp.asarray(u.Quantity(value).to_decimal(unit)) for _, value in selected]
    out = jnp.zeros((size,), dtype=jnp.result_type(*mantissas))
    for (positions, _), mantissa in zip(selected, mantissas):
        out = out.at[positions].set(mantissa)
    return out if unit.is_unitless else u.Quantity(out, unit)


def _stack_values(values):
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(jnp.stack([jnp.asarray(value.to_decimal(unit)) for value in values]), unit)
    return jnp.stack([jnp.asarray(value) for value in values])


def _split_values(value, count: int):
    if isinstance(value, u.Quantity):
        unit = value.unit
        decimal = np.asarray(value.to_decimal(unit))
        return tuple(u.Quantity(decimal[index], unit) for index in range(count))
    array = np.asarray(value)
    return tuple(array[index] for index in range(count))
