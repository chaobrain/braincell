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

"""Static observables, recording declarations, and immutable result blocks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._misc import (
    concat_values as _concat_values,
    freeze_array as _freeze_array,
    register_result_pytree,
    require_name as _require_name,
    scalar_decimal as _scalar_decimal,
)

__all__ = [
    "EventSeries",
    "RecordingRow",
    "RecordingSchema",
    "RecordingSpec",
    "SampleBlock",
    "concat_sample_blocks",
    "observe",
]


@dataclass(frozen=True)
class _CellStateObservable:
    field: str


@dataclass(frozen=True)
class _OutputObservable:
    name: str


@dataclass(frozen=True)
class _MechanismStateObservable:
    category: str
    selector: tuple[str, object] | None
    field: str


@dataclass(frozen=True)
class _CurrentObservable:
    category: str
    selector: tuple[str, object] | None
    reduce: str


@dataclass(frozen=True)
class _MembraneCurrentObservable:
    pass


@dataclass(frozen=True)
class _ClampCurrentObservable:
    ids: tuple[int, ...] | None
    reduce: str


@dataclass(frozen=True)
class _RuleStateObservable:
    """One plasticity-rule state of the connection rows ``row_ids`` (``ConnectionView.record``)."""

    row_ids: tuple[int, ...]
    field: str


class _MechanismObservableBuilder:
    __slots__ = ("_category", "_selector")

    def __init__(self, category: str, selector: tuple[str, object] | None) -> None:
        self._category = category
        self._selector = selector

    def state(self, field: str):
        """Select one state field from the chosen mechanism owners."""
        _require_name(field, "state field")
        return _MechanismStateObservable(self._category, self._selector, field)

    def current(self, *, reduce: str = "sum"):
        """Select current contributions, optionally reduced per Cell/CV."""
        if reduce not in {"sum", "none"}:
            raise ValueError("current reduce must be 'sum' or 'none'.")
        return _CurrentObservable(self._category, self._selector, reduce)


class _ObserveNamespace:
    """Construct typed observable declarations without resolving model data."""

    __slots__ = ()

    def state(self, field: str):
        """Observe a Cell state field: the membrane voltage.

        Parameters
        ----------
        field : str
            ``"V"``, the membrane voltage, as in ``cell.V``. The rows carry
            the field ``"V"``.

        Returns
        -------
        object
            Observable declaration for :meth:`braincell.Cell.record`: one row
            per selected ``(population member, CV)``.

        Raises
        ------
        KeyError
            If ``field`` is not ``"V"``, including ``"v"``, which was
            removed in 0.3.

        Examples
        --------
        >>> import braincell
        >>> braincell.observe.state("V").field
        'V'
        """
        _require_name(field, "state field")
        return _CellStateObservable(_cell_state_field(field))

    def output(self, name: str):
        """Observe one selected model output after each model update."""
        _require_name(name, "output name")
        return _OutputObservable(name)

    def channel(self, *, type: str | None = None, name: str | None = None):
        """Select density channels by one explicit type or name."""
        return _MechanismObservableBuilder("channel", _one_selector(type=type, name=name))

    def ion(
        self,
        *,
        species: str | None = None,
        type: str | None = None,
        name: str | None = None,
    ):
        """Select ions by one explicit species, type, or name."""
        return _MechanismObservableBuilder(
            "ion",
            _one_selector(species=species, type=type, name=name),
        )

    def synapse(
        self,
        *,
        type: str | None = None,
        name: str | None = None,
        ids=None,
    ):
        """Select synapses by one explicit type, name, or stable-ID list."""
        selector = _one_selector(type=type, name=name, ids=ids)
        if selector is not None and selector[0] == "ids":
            values = np.asarray(selector[1])
            if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
                raise TypeError("synapse ids must be a one-dimensional integer array.")
            selector = ("ids", tuple(int(item) for item in values.tolist()))
        return _MechanismObservableBuilder("synapse", selector)

    def membrane_current(self):
        """Observe total membrane current density at selected CVs."""
        return _MembraneCurrentObservable()

    def clamp_current(self, *, reduce: str = "sum"):
        """Observe cached external clamp current within selected CVs."""
        if reduce not in {"sum", "none"}:
            raise ValueError("clamp current reduce must be 'sum' or 'none'.")
        return _ClampCurrentObservable(None, reduce)


observe = _ObserveNamespace()


def _cell_state_field(field: str) -> str:
    """The one recordable Cell state is the membrane voltage ``"V"``, named as ``cell.V``."""
    if field == "V":
        return field
    if field == "v":
        raise KeyError("Cell state 'v' was removed in 0.3; record the membrane voltage as 'V', as in cell.V.")
    raise KeyError(f"Cell state {field!r} is not recordable; record the membrane voltage as 'V'.")


def as_observable(observable):
    """Return the observable a :meth:`braincell.Cell.record` argument declares.

    ``"V"`` stands for ``observe.state("V")``; an ``observe.*`` declaration
    is returned unchanged.

    Raises
    ------
    KeyError
        If ``observable`` is a string other than ``"V"``.
    TypeError
        If ``observable`` is neither a string nor an ``observe.*`` declaration.
    """
    if isinstance(observable, str):
        return observe.state(observable)
    if not isinstance(observable, _OBSERVABLE_TYPES):
        raise TypeError(
            f"Cell.record observable must be 'V' or a braincell.observe declaration, got {type(observable).__name__}."
        )
    return observable


_OBSERVABLE_TYPES = (
    _CellStateObservable,
    _OutputObservable,
    _MechanismStateObservable,
    _CurrentObservable,
    _MembraneCurrentObservable,
    _ClampCurrentObservable,
)


@dataclass(frozen=True)
class RecordingSpec:
    """One immutable observer declaration owned by a Cell."""

    name: str
    scope: object = field(repr=False, compare=False)
    observable: object
    period: object | None = None
    frequency: object | None = None
    start: object = field(default_factory=lambda: 0.0 * u.ms)

    def __post_init__(self) -> None:
        _require_name(self.name, "recording name")
        if self.period is not None and self.frequency is not None:
            raise ValueError("Recording period and frequency are mutually exclusive.")
        if self.period is not None:
            _positive_quantity(self.period, u.ms, "recording period")
        if self.frequency is not None:
            _positive_quantity(self.frequency, u.Hz, "recording frequency")
        _nonnegative_quantity(self.start, u.ms, "recording start")

    def period_for_dt(self, dt):
        return (
            dt
            if self.period is None and self.frequency is None
            else (self.period if self.period is not None else 1.0 / self.frequency)
        )


@dataclass(frozen=True)
class RecordingRow:
    """Static metadata for one column in a SampleBlock.

    ``population_index`` is the flat index of the row's population member:
    on a multi-dimensional ``pop_size``, its position in C order over the
    population axes, the index a spike's ``source_id`` uses;
    ``np.unravel_index(population_index, pop_size)`` gives the member's
    index on each axis. A whole-Cell recording lists members in that order,
    then CVs (or points, clamps, mechanisms) within each member.

    A row of ``ConnectionView.record`` is one connection row: its site
    fields and ``synapse_id`` are its synapse's, ``mechanism_category`` is
    ``"connection_rule"``, ``mechanism_type`` its rule, ``mechanism_name``
    its connect name, ``connection_id`` its stable ID
    (``ConnectionView.id``) and ``source_index`` its pre endpoint
    (``ConnectionView.source_index``). Other rows leave these two ``None``.

    Examples
    --------
    >>> import brainunit as u
    >>> import braincell
    >>> cell = braincell.Cell.point((2, 3), V_init=-65.0 * u.mV)
    >>> _ = cell[1, 2].record("V", "V")
    >>> result = cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
    >>> [row.population_index for row in result.samples["V"].schema.rows]
    [5]
    """

    population_index: int
    cv_id: int | None
    point_id: int | None
    branch_id: int | None
    field: str
    unit: object | None
    mechanism_category: str | None = None
    mechanism_type: str | None = None
    mechanism_name: str | None = None
    synapse_id: int | None = None
    clamp_id: int | None = None
    placement_id: int | None = None
    clamp_ids: tuple[int, ...] = ()
    contributor_ids: tuple[int, ...] = ()
    output_name: str | None = None
    output_index: tuple[int, ...] = ()
    connection_id: int | None = None
    source_index: int | None = None


@dataclass(frozen=True)
class RecordingSchema:
    """Static row schema shared by immutable sample segments."""

    name: str
    rows: tuple[RecordingRow, ...]
    period: object
    schedule_start: object
    time_offset: object = field(default_factory=lambda: 0.0 * u.ms)

    @property
    def size(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class SampleBlock:
    """Immutable regular samples plus their static logical-row schema.

    A JAX pytree: ``values`` and the time fields are its leaves, ``schema``
    is static.
    """

    values: object
    schema: RecordingSchema
    segment_start: object
    segment_stop: object
    first_time: object | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_array(self.values))

    @property
    def time(self):
        count = int(self.values.shape[0])
        if count == 0 or self.first_time is None:
            return np.asarray([], dtype=float) * u.ms
        return self.first_time + self.schema.time_offset + np.arange(count) * self.schema.period


def concat_sample_blocks(blocks):
    """Join contiguous sample blocks of one recording into a single block.

    Both run drivers need this: :meth:`braincell.RunResult.concat` for one
    cell and :meth:`braincell.network.NetworkResult.concat` for a network.
    It lives beside :class:`SampleBlock` so neither has to reach into the
    other's module for it.

    The caller is responsible for checking that the blocks are contiguous
    and share a schema; this only joins them.

    Parameters
    ----------
    blocks : sequence of SampleBlock
        Ordered contiguous segments of one recording. Must be non-empty.

    Returns
    -------
    SampleBlock
        One block spanning ``blocks[0].segment_start`` to
        ``blocks[-1].segment_stop``. ``first_time`` is taken from the first
        segment that actually sampled, since a leading segment may span an
        interval in which the recording never fired.
    """
    first = blocks[0]
    return SampleBlock(
        values=_concat_values(tuple(block.values for block in blocks)),
        schema=first.schema,
        segment_start=first.segment_start,
        segment_stop=blocks[-1].segment_stop,
        first_time=next((block.first_time for block in blocks if block.first_time is not None), None),
    )


@dataclass(frozen=True)
class EventSeries:
    """Immutable sparse event rows.

    Attributes
    ----------
    time : brainunit.Quantity
        ``(n_events,)`` event times.
    source_id : numpy.ndarray
        ``(n_events,)`` int64 source rows; for a Cell output they index the
        row-aligned arrays in ``metadata``.
    count : numpy.ndarray
        ``(n_events,)`` int64 event counts.
    metadata : Mapping[str, object]
        Read-only port facts, such as ``"port"`` and the row-aligned
        ``"cv_id"``.

    Notes
    -----
    A JAX pytree: ``time``, ``source_id`` and ``count`` are its leaves,
    ``metadata`` is static.
    """

    time: object
    source_id: np.ndarray
    count: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _freeze_array(self.time))
        source_id = np.asarray(self.source_id, dtype=np.int64).reshape(-1)
        count = np.asarray(self.count, dtype=np.int64).reshape(-1)
        if source_id.shape != count.shape or tuple(getattr(self.time, "shape", ())) != source_id.shape:
            raise ValueError("EventSeries time, source_id, and count must have the same shape.")
        source_id.flags.writeable = False
        count.flags.writeable = False
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "count", count)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType({key: _freeze_array(value) for key, value in self.metadata.items()}),
        )


register_result_pytree(SampleBlock, ("values", "segment_start", "segment_stop", "first_time"), static=("schema",))
register_result_pytree(EventSeries, ("time", "source_id", "count"), static=("metadata",))


def compile_recording(cell, spec: RecordingSpec, *, dt):
    """Resolve one RecordingSpec to a static schema and sampler."""
    period = spec.period_for_dt(dt)
    _aligned_steps(period, dt, "recording period")
    _aligned_steps(spec.start, dt, "recording start", allow_zero=True)
    rows, sampler = _observable_rows_and_sampler(cell, spec)
    with brainstate.environ.context(t=cell.current_time, dt=dt):
        values = sampler()
    if int(values.shape[-1]) != len(rows):
        raise RuntimeError(
            f"Recording {spec.name!r} sampler returned {values.shape[-1]!r} rows for schema size {len(rows)!r}."
        )
    unit = values.unit if isinstance(values, u.Quantity) else None
    if unit is not None:
        rows = tuple(replace(row, unit=unit) if row.unit is None else row for row in rows)
    schema = RecordingSchema(
        name=spec.name,
        rows=rows,
        period=period,
        schedule_start=spec.start,
        time_offset=(0.5 * dt if isinstance(spec.observable, _ClampCurrentObservable) else 0.0 * u.ms),
    )
    phase = "post" if isinstance(spec.observable, _OutputObservable) else "pre"
    return _CompiledRecording(spec=spec, schema=schema, sample=sampler, phase=phase)


@dataclass(frozen=True)
class _CompiledRecording:
    spec: RecordingSpec
    schema: RecordingSchema
    sample: object = field(compare=False, repr=False)
    phase: str = "pre"


def recording_is_active(cell, spec: RecordingSpec) -> bool:
    """Return whether a declaration belongs to the currently selected model."""
    if isinstance(spec.observable, _OutputObservable):
        return spec.observable.name in cell.outputs
    return not cell._uses_reduction


def _observable_rows_and_sampler(cell, spec: RecordingSpec):
    observable = spec.observable
    if isinstance(observable, _OutputObservable):
        if spec.scope.spatially_restricted:
            raise ValueError("observe.output(...) only supports Cell or population-only CellView scopes.")
        value = cell.outputs[observable.name]
        runtime_prefix_rank = len(cell.pop_size) + (1 if cell._runtime_batch_size is not None else 0)
        feature_shape = tuple(value.shape[runtime_prefix_rank:])
        rows = tuple(
            RecordingRow(
                population_index=int(population_index),
                cv_id=None,
                point_id=None,
                branch_id=None,
                field=observable.name,
                unit=None,
                output_name=observable.name,
                output_index=tuple(int(item) for item in feature_index),
            )
            for population_index in spec.scope.population_indices
            for feature_index in np.ndindex(feature_shape)
        )

        def sample():
            selected = cell._selected_outputs(spec.scope.population_indices)[observable.name]
            batch_prefix = (selected.shape[0],) if cell._runtime_batch_size is not None else ()
            return u.math.reshape(selected, batch_prefix + (len(rows),))

        return rows, sample

    if isinstance(observable, _CellStateObservable):
        if observable.field != "V":
            raise KeyError(f"Cell state {observable.field!r} is not recordable in v1.")
        rows = tuple(_spatial_row(cell, pair, field="V") for pair in spec.scope.pairs)

        if getattr(cell, "_is_lumped", False):
            # A lumped Cell's V has no CV axis: every pair is (member, the one CV).
            def sample():
                pop = np.asarray([pair[0] for pair in spec.scope.pairs], dtype=np.int32)
                return _flat_members(cell, cell.V.value, cv_axis=False)[..., pop]

            return rows, sample

        def sample():
            cv = np.asarray([pair[1] for pair in spec.scope.pairs], dtype=np.int32)
            pop = np.asarray([pair[0] for pair in spec.scope.pairs], dtype=np.int32)
            return _flat_members(cell, cell.V.value, cv_axis=True)[..., pop, cv]

        return rows, sample

    if isinstance(observable, _MechanismStateObservable):
        if observable.category in {"channel", "ion"}:
            view = _selected_density_view(cell, spec.scope, observable.category, observable.selector)
            rows = tuple(_density_recording_row(cell, row, field=observable.field) for row in view.rows)

            groups = _density_owner_groups(view)

            def sample():
                return _scatter_group_values(
                    [(positions, _rows_last(cell, selected.get(observable.field))) for positions, selected in groups],
                    len(view),
                )

            return rows, sample
        view = _selected_synapse_view(cell, spec.scope, observable.selector)
        rows = tuple(_synapse_recording_row(cell, view, index, observable.field) for index in range(len(view)))

        def sample():
            return _sample_synapse_field(view, observable.field)

        return rows, sample

    if isinstance(observable, _CurrentObservable):
        return _current_rows_and_sampler(cell, spec.scope, observable)

    if isinstance(observable, _MembraneCurrentObservable):
        from braincell._multi_compartment import currents

        rows = tuple(_spatial_row(cell, pair, field="membrane_current") for pair in spec.scope.pairs)

        def sample():
            pop = np.asarray([pair[0] for pair in spec.scope.pairs], dtype=np.int32)
            if getattr(cell, "_is_lumped", False):
                return _flat_members(cell, cell._lumped_membrane_current(), cv_axis=False)[..., pop]
            total = currents.total_membrane_current(
                cell,
                V_cv=cell.V.value,
                t=brainstate.environ.get("t"),
                pop=False,
            )
            cv = np.asarray([pair[1] for pair in spec.scope.pairs], dtype=np.int32)
            return _flat_members(cell, total, cv_axis=True)[..., pop, cv]

        return rows, sample

    if isinstance(observable, _ClampCurrentObservable):
        return _clamp_current_rows_and_sampler(cell, spec.scope, observable)

    if isinstance(observable, _RuleStateObservable):
        return _rule_state_rows_and_sampler(cell, observable)

    raise TypeError(f"Unsupported observable {type(observable).__name__!s}.")


def _rule_state_rows_and_sampler(cell, observable: _RuleStateObservable):
    """Rows and sampler of a ``ConnectionView.record`` declaration.

    The rows are the declared connection rows still connected, in the order
    of the view they were declared on; each row's site is its synapse's.
    """
    view = cell.connections._with_ids(observable.row_ids)
    synapse = view.synapse
    # Each view property gathers every row: read each once.
    population_index = synapse.population_index.tolist()
    cv_id = synapse.cv_id.tolist()
    point_id = synapse.point_id.tolist()
    branch_of_cv = [int(cv.branch_id) for cv in cell.cvs]
    # The sampler joins the rows in the first call's unit; set it here, as
    # filling it in later replaces every row (``compile_recording``).
    groups = view._rule_groups() if len(view) else []
    first = groups[0][0].plasticity.value(observable.field) if groups else None
    unit = first.unit if isinstance(first, u.Quantity) and not first.unit.is_unitless else None
    columns = zip(
        population_index,
        cv_id,
        point_id,
        view._rule_type_names(),
        view.connect_name.tolist(),
        view.synapse_id.tolist(),
        view.id.tolist(),
        view.source_index.tolist(),
    )
    rows = tuple(
        RecordingRow(
            population_index=int(member),
            cv_id=int(cv),
            point_id=int(point),
            branch_id=branch_of_cv[int(cv)],
            field=observable.field,
            unit=unit,
            mechanism_category="connection_rule",
            mechanism_type=rule_type,
            mechanism_name=str(connect_name),
            synapse_id=int(synapse_id),
            connection_id=int(row_id),
            source_index=int(source_index),
        )
        for member, cv, point, rule_type, connect_name, synapse_id, row_id, source_index in columns
    )
    return rows, view._rule_state_sampler(observable.field)


def _clamp_current_rows_and_sampler(cell, scope, observable: _ClampCurrentObservable):
    view = cell.clamps.for_scope_pairs(scope.pairs)
    if observable.ids is not None:
        selected = set(int(item) for item in observable.ids)
        view = type(view)(cell, [logical_id for logical_id in view.id.tolist() if logical_id in selected])
    store_rows = view._store.row_indices(view.id)

    if observable.reduce == "none":
        rows = tuple(_clamp_recording_row(cell, record) for record in view.instances)

        def sample():
            values = cell._step_clamp_components.value
            return values[..., store_rows]

        return rows, sample

    groups: dict[tuple[int, int], list[int]] = {}
    for store_row, population_index, cv_id in zip(
        store_rows.tolist(), view.population_index.tolist(), view.cv_id.tolist()
    ):
        groups.setdefault((int(population_index), int(cv_id)), []).append(int(store_row))
    rows = tuple(
        replace(
            _spatial_row(cell, pair, field="clamp_current"),
            clamp_ids=tuple(int(cell.clamps._store.id[index]) for index in indices),
        )
        for pair, indices in groups.items()
    )

    def sample_sum():
        values = cell._step_clamp_components.value
        return _stack_values([u.math.sum(values[..., indices], axis=-1) for indices in groups.values()])

    return rows, sample_sum


def _current_rows_and_sampler(cell, scope, observable: _CurrentObservable):
    if observable.category in {"channel", "ion"}:
        view = _selected_density_view(cell, scope, observable.category, observable.selector)
        raw_rows = tuple(view.rows)
        owner_groups = _density_owner_groups(view)

        def raw_sample():
            return _scatter_group_values(
                [(positions, _density_group_current(cell, selected.rows)) for positions, selected in owner_groups],
                len(view),
            )

        record_rows = tuple(_density_recording_row(cell, row, field="current") for row in raw_rows)
    else:
        view = _selected_synapse_view(cell, scope, observable.selector)
        raw_rows = tuple(range(len(view)))

        def raw_sample():
            return _sample_synapse_current(cell, view)

        record_rows = tuple(_synapse_recording_row(cell, view, index, "current") for index in raw_rows)

    if observable.reduce == "none":
        return record_rows, raw_sample

    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(record_rows):
        groups.setdefault((row.population_index, row.cv_id), []).append(index)
    reduced_rows = tuple(
        _spatial_row(cell, pair, field=f"{observable.category}_current", contributor_ids=tuple(indices))
        for pair, indices in groups.items()
    )

    def sample_sum():
        values = raw_sample()
        return _stack_values([u.math.sum(values[..., indices], axis=-1) for indices in groups.values()])

    return reduced_rows, sample_sum


def _selected_density_view(cell, scope, category: str, selector):
    from braincell._multi_compartment.density_views import ChannelView, IonView

    view = ChannelView(cell, scope) if category == "channel" else IonView(cell, scope)
    if selector is None:
        return view
    kind, value = selector
    if kind == "name":
        return view.by_name(value)
    if kind == "type":
        return view.by_type(value)
    if kind == "species" and category == "ion":
        return view.by_species(value)
    raise ValueError(f"Unsupported {category} selector {kind!r}.")


def _selected_synapse_view(cell, scope, selector):
    view = cell.synapses.for_scope_pairs(scope.pairs)
    if selector is None:
        return view
    kind, value = selector
    if kind == "name":
        return view.by_name(value)
    if kind == "type":
        return view.by_type(value)
    if kind == "ids":
        selected = set(value)
        return view[np.asarray([logical_id in selected for logical_id in view.id], dtype=bool)]
    raise ValueError(f"Unsupported synapse selector {kind!r}.")


def _sample_synapse_field(view, field: str):
    return _scatter_group_values(
        [(positions, selected.get(field)) for positions, selected in _synapse_type_groups(view)],
        len(view),
    )


def _sample_synapse_current(cell, view):
    from braincell._multi_compartment.currents import synapse_current

    point_v = cell._point_voltage_for_mechanisms(cell.V.value)
    groups = []
    for positions, selected in _synapse_type_groups(view):
        synapse_type = selected._require_homogeneous_type()
        layout_id = selected._store.layout_id(synapse_type)
        layout = cell.runtime.layouts[layout_id]
        node = cell.runtime.get_runtime_node(layout_id)
        local_v = (
            point_v[..., layout.point_index]
            if layout.population_index is None
            else point_v[..., layout.population_index, layout.point_index]
        )
        current = synapse_current(cell.runtime, layout, node, local_v)
        groups.append((positions, current[..., selected.runtime_index]))
    return _scatter_group_values(groups, len(view))


def _density_group_current(cell, rows):
    from braincell._multi_compartment.density_views import _runtime_layout

    row = rows[0]
    cv_v = cell.V.value
    if row.category == "ion":
        current = cell.runtime.get_ion(row.name).current(cv_v, include_external=False)
    else:
        layout = _runtime_layout(cell, row)
        node = cell.runtime.get_runtime_node(layout.id)
        bound = cell.runtime.bound_ion_keys.get(layout.id, ())
        if len(bound) == 0:
            current = node.current(cv_v)
        else:
            current = node.current(cv_v, *tuple(cell.runtime.get_ion(key).pack_info() for key in bound))
            if isinstance(current, dict):
                current = sum(current.values())
    if getattr(cell, "_is_lumped", False):
        # A lumped node's current has no CV axis; its one CV is index 0.
        current = current[..., None]
    cv_ids = np.asarray([item.cv_id for item in rows], dtype=np.int32)
    population_indices = np.asarray([item.population_index for item in rows], dtype=np.int32)
    return _flat_members(cell, current, cv_axis=True)[..., population_indices, cv_ids]


def _density_owner_groups(view):
    grouped = {}
    for position, row in enumerate(view.rows):
        grouped.setdefault((row.category, row.mechanism_type, row.name), []).append(position)
    return tuple(
        (
            np.asarray(positions, dtype=np.int32),
            type(view)(view.cell, None, rows=(view.rows[position] for position in positions)),
        )
        for positions in grouped.values()
    )


def _synapse_type_groups(view):
    grouped = {}
    for position, synapse_type in enumerate(view.synapse_type.tolist()):
        grouped.setdefault(str(synapse_type), []).append(position)
    return tuple(
        (
            np.asarray(positions, dtype=np.int32),
            view[np.asarray(positions, dtype=np.int32)],
        )
        for positions in grouped.values()
    )


def _scatter_group_values(groups, size: int):
    if not groups:
        return np.asarray([], dtype=float)
    first = groups[0][1]
    unit = first.unit if isinstance(first, u.Quantity) else None
    first_mantissa = first.to_decimal(unit) if unit is not None else first
    output = jnp.zeros(tuple(first_mantissa.shape[:-1]) + (size,), dtype=jnp.asarray(first_mantissa).dtype)
    for positions, values in groups:
        if (values.unit if isinstance(values, u.Quantity) else None) != unit:
            raise TypeError("Recorded mechanism rows have incompatible units.")
        mantissa = values.to_decimal(unit) if unit is not None else values
        output = output.at[..., positions].set(mantissa)
    return u.Quantity(output, unit) if unit is not None else output


def _flat_members(cell, value, *, cv_axis: bool):
    """``value`` with the population axes of a multi-dimensional ``pop_size`` merged into one, in C order.

    ``value`` is shaped ``(batch,) + pop_size`` (a lumped Cell) or
    ``(batch,) + pop_size + (n_cv,)``, without the batch axis when there is
    none. The merged axis is indexed by a recording row's flat
    ``population_index``.
    """
    rank = len(cell.pop_size)
    if rank == 1:
        return value
    shape = tuple(value.shape)
    stop = len(shape) - (1 if cv_axis else 0)
    members = int(np.prod(cell.pop_size, dtype=np.int64))
    return u.math.reshape(value, shape[: stop - rank] + (members,) + shape[stop:])


def _rows_last(cell, value):
    """Move the row axis that a density view's ``get`` stacks first behind the batch axis."""
    if cell._runtime_batch_size is None:
        return value
    return u.math.moveaxis(value, 0, -1)


def _spatial_row(cell, pair, *, field: str, contributor_ids=()) -> RecordingRow:
    population_index, cv_id = (int(pair[0]), int(pair[1]))
    cv = cell.cvs[cv_id]
    return RecordingRow(
        population_index=population_index,
        cv_id=cv_id,
        point_id=int(cell.node_tree.cv_to_mid_node_id[cv_id]),
        branch_id=int(cv.branch_id),
        field=field,
        unit=None,
        contributor_ids=tuple(int(item) for item in contributor_ids),
    )


def _density_recording_row(cell, row, *, field: str) -> RecordingRow:
    spatial = _spatial_row(cell, (row.population_index, row.cv_id), field=field)
    return replace(
        spatial,
        mechanism_category=row.category,
        mechanism_type=row.mechanism_type,
        mechanism_name=row.name,
    )


def _synapse_recording_row(cell, view, index: int, field: str) -> RecordingRow:
    logical_id = int(view.id[index])
    cv_id = int(view.cv_id[index])
    spatial = _spatial_row(cell, (int(view.population_index[index]), cv_id), field=field)
    return replace(
        spatial,
        point_id=int(view.point_id[index]),
        mechanism_category="synapse",
        mechanism_type=str(view.synapse_type[index]),
        mechanism_name=str(view.name[index]),
        synapse_id=logical_id,
    )


def _clamp_recording_row(cell, record) -> RecordingRow:
    spatial = _spatial_row(cell, (record.population_index, record.cv_id), field="clamp_current")
    return replace(
        spatial,
        point_id=int(record.point_id),
        mechanism_category="clamp",
        mechanism_type=str(record.clamp_type),
        clamp_id=int(record.id),
        placement_id=int(record.placement_id),
        clamp_ids=(int(record.id),),
    )


def _stack_values(values):
    if not values:
        return np.asarray([], dtype=float)
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(u.math.stack([value.to_decimal(unit) for value in values], axis=-1), unit)
    return u.math.stack(values, axis=-1)


def _one_selector(**values):
    supplied = [(name, value) for name, value in values.items() if value is not None]
    if len(supplied) > 1:
        raise TypeError(f"Observable selectors are mutually exclusive; got {[name for name, _ in supplied]!r}.")
    if not supplied:
        return None
    name, value = supplied[0]
    if name != "ids":
        _require_name(value, f"{name} selector")
    return (name, value)


def _aligned_steps(value, dt, name: str, *, allow_zero: bool = False) -> int:
    ratio = _scalar_decimal(value, u.ms) / _scalar_decimal(dt, u.ms)
    rounded = int(round(ratio))
    if (not allow_zero and rounded <= 0) or (allow_zero and rounded < 0) or not np.isclose(ratio, rounded):
        raise ValueError(f"{name} must be an integer multiple of dt; got {value!r} for dt={dt!r}.")
    return rounded


def _positive_quantity(value, unit, name: str) -> None:
    _nonnegative_quantity(value, unit, name)
    if _scalar_decimal(value, unit) <= 0.0:
        raise ValueError(f"{name} must be > 0.")


def _nonnegative_quantity(value, unit, name: str) -> None:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a physical Quantity.")
    try:
        decimal = np.asarray(value.to_decimal(unit), dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} has incompatible units.") from exc
    if decimal.shape not in ((), (1,)) or not np.isfinite(decimal).all() or float(decimal.reshape(())) < 0.0:
        raise ValueError(f"{name} must be one finite non-negative scalar.")
