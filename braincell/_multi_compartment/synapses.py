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

"""Logical synapse storage and user-facing views.

``SynapseView`` is the public logical-instance facade. The owning ``Cell``
keeps the actual column store private, and initialization binds stable logical
ids to rows in one runtime SoA node per registered synapse type.
"""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._compute.layouts import _stack_synapse_values
from braincell.mech import Synapse, get_registry

__all__ = ["SynapseView"]


@dataclass(frozen=True)
class _SynapseRecord:
    """Compatibility row assembled on demand from the private SoA store."""

    id: int
    placement_id: int
    population_index: int
    point_id: int
    cv_id: int
    branch_id: int
    branch_x: float
    mechanism: Synapse


class _SynapseStore:
    """Cell-owned SoA columns for declaration-time logical synapses."""

    __slots__ = (
        "cell",
        "id",
        "placement_id",
        "population_index",
        "point_id",
        "cv_id",
        "branch_id",
        "branch_x",
        "name",
        "synapse_type",
        "mechanism",
        "declaration",
        "parameter_columns",
        "_type_rows",
        "_type_local_by_id",
        "_row_by_id",
        "_layout_by_type",
        "_runtime_row_by_id",
        "_mechanism_layouts",
    )

    def __init__(self, cell) -> None:
        self.cell = cell
        rows = []
        synapse_placements = tuple(
            placement for placement in cell.point_placements if isinstance(placement.mechanism, Synapse)
        )
        if len(cell.pop_size) > 1 and synapse_placements:
            raise ValueError(
                f"Synapse views currently require scalar or one-dimensional pop_size; got {cell.pop_size!r}."
            )
        population_size = 1 if len(cell.pop_size) == 0 else int(cell.pop_size[0])
        for placement in synapse_placements:
            mechanism = placement.mechanism
            owners = (
                range(population_size) if placement.population_index is None else (int(placement.population_index),)
            )
            for owner in owners:
                rows.append((int(owner), int(placement.id), placement, mechanism))

        # Public logical order is stable and cell-major, independently of how
        # broadcast placements happen to be represented by discretization.
        rows.sort(key=lambda item: (item[0], item[1]))
        self.population_index = np.asarray([row[0] for row in rows], dtype=np.int64)
        self.placement_id = np.asarray([row[1] for row in rows], dtype=np.int64)
        self.id = self.placement_id * int(population_size) + self.population_index
        self.point_id = np.asarray([row[2].point_id for row in rows], dtype=np.int64)
        self.cv_id = np.asarray([row[2].cv_id for row in rows], dtype=np.int64)
        self.branch_id = np.asarray([row[2].branch_id for row in rows], dtype=np.int64)
        self.branch_x = np.asarray([row[2].branch_x for row in rows], dtype=float)
        self.mechanism = tuple(row[3] for row in rows)
        origins = getattr(cell, "_synapse_origins", {})
        self.declaration = tuple(origins.get(id(item), item) for item in self.mechanism)
        self.name = np.asarray([item.instance_name for item in self.mechanism], dtype=object)
        self.synapse_type = np.asarray([item.synapse_type for item in self.mechanism], dtype=object)
        self._row_by_id = {int(logical_id): row for row, logical_id in enumerate(self.id.tolist())}
        self.parameter_columns: dict[str, dict[str, object]] = {}
        self._type_rows: dict[str, np.ndarray] = {}
        self._type_local_by_id: dict[int, int] = {}
        self._layout_by_type: dict[str, int] = {}
        self._runtime_row_by_id: dict[int, int] = {}
        self._mechanism_layouts: dict[int, _MechanismRowLayout] = {}
        self._validate_name_types()
        self._build_parameter_columns()

    def _validate_name_types(self) -> None:
        type_by_name: dict[str, str] = {}
        for name, synapse_type in zip(self.name.tolist(), self.synapse_type.tolist()):
            previous = type_by_name.setdefault(str(name), str(synapse_type))
            if previous != synapse_type:
                raise ValueError(
                    f"Synapses with the same name {name!r} cannot use different synapse types "
                    f"({previous!r} and {synapse_type!r})."
                )

    def _build_parameter_columns(self) -> None:
        for raw_type in dict.fromkeys(self.synapse_type.tolist()):
            synapse_type = str(raw_type)
            runtime_cls = get_registry().get("synapse", synapse_type)
            schema = dict(runtime_cls.parameters)
            rows = np.flatnonzero(self.synapse_type == synapse_type).astype(np.int64)
            self._type_rows[synapse_type] = rows
            for local_row, store_row in enumerate(rows.tolist()):
                self._type_local_by_id[int(self.id[store_row])] = int(local_row)
                unknown = set(self.mechanism[store_row].params).difference(schema)
                if unknown:
                    raise TypeError(f"Synapse type {synapse_type!r} has no parameters {tuple(sorted(unknown))!r}.")
            columns = {}
            for parameter, spec in schema.items():
                values = []
                for store_row in rows.tolist():
                    logical_id = int(self.id[store_row])
                    mechanism = self.mechanism[store_row]
                    key = (
                        int(self.placement_id[store_row]),
                        int(self.population_index[store_row]),
                        str(parameter),
                    )
                    if key in self.cell._synapse_parameter_overrides:
                        value = self.cell._synapse_parameter_overrides[key]
                    elif parameter in mechanism.params:
                        value = _select_declared_parameter_value(
                            mechanism.params[parameter],
                            logical_id=logical_id,
                            store=self,
                            mechanism=mechanism,
                        )
                    else:
                        value = spec.default
                    spec.validate(value, parameter)
                    values.append(value)
                columns[parameter] = _stack_synapse_values(values, parameter=parameter)
            runtime_cls.validate_parameter_values(columns)
            self.parameter_columns[synapse_type] = columns

    def bind_runtime(self, synapse_type: str, layout_id: int, logical_ids: np.ndarray) -> None:
        """Bind logical ids to the rows of one materialized runtime node."""
        key = str(synapse_type)
        self._layout_by_type[key] = int(layout_id)
        ids = np.asarray(logical_ids, dtype=np.int64)
        self._runtime_row_by_id.update(
            (int(logical_id), int(runtime_row)) for runtime_row, logical_id in enumerate(ids.tolist())
        )

    def mechanism_row_layout(self, mechanism: Synapse) -> "_MechanismRowLayout":
        """Return the cached row layout shared by one declaration's synapses.

        The layout depends only on the declaring ``mechanism``, so it is built
        once per declaration rather than re-derived for every logical synapse
        that declaration produced.
        """
        key = id(mechanism)
        cached = self._mechanism_layouts.get(key)
        if cached is None:
            cached = _MechanismRowLayout.build(self, mechanism)
            self._mechanism_layouts[key] = cached
        return cached

    def layout_id(self, synapse_type: str) -> int:
        try:
            return self._layout_by_type[str(synapse_type)]
        except KeyError as exc:
            raise RuntimeError(f"Synapse type {synapse_type!r} has not been materialized.") from exc

    def runtime_rows(self, logical_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(logical_ids, dtype=np.int64)
        try:
            return np.asarray([self._runtime_row_by_id[int(item)] for item in ids.tolist()], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError("Selected logical synapses have not been materialized.") from exc

    def parameter_value(self, logical_id: int, parameter: str):
        index = self.row_index(int(logical_id))
        synapse_type = str(self.synapse_type[index])
        if parameter not in self.parameter_columns[synapse_type]:
            raise KeyError(f"Synapse type {synapse_type!r} does not declare parameter {parameter!r}.")
        local = self._type_local_by_id[int(logical_id)]
        return _take_vector_items(self.parameter_columns[synapse_type][parameter], local)

    def parameter_column(self, logical_ids, parameter: str):
        """Gather one declared parameter across several logical synapses.

        Parameters
        ----------
        logical_ids : array-like of int
            Stable logical ids, all belonging to the same synapse type.
        parameter : str
            Name of a parameter declared by that type.

        Returns
        -------
        array-like
            One vector holding ``parameter`` for each id, in the order given. An
            empty selection yields an empty dimensionless float array, matching
            what stacking zero values produced.

        Raises
        ------
        TypeError
            If the ids span more than one synapse type.
        KeyError
            If the type does not declare ``parameter``.

        See Also
        --------
        parameter_value : Read the same parameter for a single logical id.

        Notes
        -----
        This is the vector form of :meth:`parameter_value`, and the form every
        caller holding a whole selection should use. Reading a selection one id
        at a time is quadratic: each :meth:`parameter_value` call re-materializes
        the entire per-type column through ``to_decimal`` only to take a single
        element, and the caller then restacks the scalars it just tore apart.
        Slicing the column once is constant in the number of rows read.
        """
        ids = np.asarray(logical_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.asarray([], dtype=float)
        types = {str(self.synapse_type[self.row_index(int(item))]) for item in ids.tolist()}
        if len(types) != 1:
            raise TypeError("Cannot read one parameter across multiple synapse types.")
        synapse_type = types.pop()
        columns = self.parameter_columns[synapse_type]
        if parameter not in columns:
            raise KeyError(f"Synapse type {synapse_type!r} does not declare parameter {parameter!r}.")
        local_rows = np.asarray([self._type_local_by_id[int(item)] for item in ids.tolist()], dtype=np.int64)
        return _take_vector_items(columns[parameter], local_rows)

    def set_parameter(self, logical_ids: np.ndarray, parameter: str, values) -> None:
        self.set_parameters(logical_ids, {parameter: values})

    def set_parameters(self, logical_ids: np.ndarray, updates: dict[str, object]) -> None:
        ids = np.asarray(logical_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        types = {str(self.synapse_type[self.row_index(int(item))]) for item in ids.tolist()}
        if len(types) != 1:
            raise TypeError("Cannot set one parameter across multiple synapse types.")
        synapse_type = types.pop()
        runtime_cls = get_registry().get("synapse", synapse_type)
        local_rows = np.asarray([self._type_local_by_id[int(item)] for item in ids.tolist()], dtype=np.int64)
        columns = dict(self.parameter_columns[synapse_type])
        for parameter, values in updates.items():
            spec = runtime_cls.parameters[parameter]
            spec.validate(values, parameter)
            columns[parameter] = _set_vector_items(columns[parameter], local_rows, values)
        runtime_cls.validate_parameter_values(columns)
        self.parameter_columns[synapse_type] = columns

    def records(self, logical_ids: np.ndarray) -> tuple[_SynapseRecord, ...]:
        result = []
        for raw in np.asarray(logical_ids, dtype=np.int64).tolist():
            logical_id = int(raw)
            index = self.row_index(logical_id)
            result.append(
                _SynapseRecord(
                    id=logical_id,
                    placement_id=int(self.placement_id[index]),
                    population_index=int(self.population_index[index]),
                    point_id=int(self.point_id[index]),
                    cv_id=int(self.cv_id[index]),
                    branch_id=int(self.branch_id[index]),
                    branch_x=float(self.branch_x[index]),
                    mechanism=self.mechanism[index],
                )
            )
        return tuple(result)

    def row_index(self, logical_id: int) -> int:
        """Return the private physical row for one stable logical id."""
        try:
            return self._row_by_id[int(logical_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown logical synapse id {logical_id!r}.") from exc

    def row_indices(self, logical_ids) -> np.ndarray:
        """Return private physical rows for stable logical ids."""
        return np.asarray([self.row_index(int(item)) for item in np.asarray(logical_ids).tolist()], dtype=np.int64)


class SynapseView:
    """View a stable ordered selection of logical synapse instances.

    A view owns no parameter or state arrays. Before initialization it gathers
    declaration values from the owning cell's private store; afterwards it
    follows the logical-id mapping into the materialized runtime node.

    Parameters
    ----------
    cell : Cell
        Root cell that owns the logical synapses.
    logical_ids : array-like of int, optional
        Stable ids selected from the root store. ``None`` selects all rows.
    """

    __slots__ = ("_cell", "_logical_ids")

    def __init__(self, cell, logical_ids=None) -> None:
        self._cell = cell
        if logical_ids is None:
            logical_ids = cell._get_synapse_store().id
        self._logical_ids = np.asarray(logical_ids, dtype=np.int64).reshape(-1)

    @property
    def cell(self):
        """Return the root Cell that owns the selected synapses."""
        return self._cell

    @property
    def root(self) -> "SynapseView":
        """Return a full view over all logical synapses on the cell."""
        return SynapseView(self._cell)

    @property
    def instances(self) -> tuple[_SynapseRecord, ...]:
        """Return compatibility row records assembled from the SoA store."""
        return self._store.records(self._logical_ids)

    @property
    def _store(self) -> _SynapseStore:
        return self._cell._get_synapse_store()

    @property
    def id(self) -> np.ndarray:
        """Return stable logical synapse identifiers."""
        return np.array(self._logical_ids, copy=True)

    @property
    def placement_id(self) -> np.ndarray:
        """Return source point-placement identifiers."""
        return self._column("placement_id")

    @property
    def population_index(self) -> np.ndarray:
        """Return owning cell-population indices."""
        return self._column("population_index")

    @property
    def point_id(self) -> np.ndarray:
        """Return resolved electrical point identifiers."""
        return self._column("point_id")

    @property
    def cv_id(self) -> np.ndarray:
        """Return owning control-volume identifiers."""
        return self._column("cv_id")

    @property
    def branch_id(self) -> np.ndarray:
        """Return original morphology branch identifiers."""
        return self._column("branch_id")

    @property
    def branch_x(self) -> np.ndarray:
        """Return original normalized branch coordinates."""
        return self._column("branch_x")

    @property
    def name(self) -> np.ndarray:
        """Return user-facing logical group names."""
        return self._column("name")

    @property
    def synapse_type(self) -> np.ndarray:
        """Return registered runtime model keys."""
        return self._column("synapse_type")

    @property
    def mechanism(self) -> tuple[Synapse, ...]:
        """Return immutable declarations associated with selected rows."""
        return tuple(self._store.mechanism[index] for index in self._store.row_indices(self._logical_ids).tolist())

    @property
    def runtime_index(self) -> np.ndarray:
        """Return row indices in the homogeneous materialized runtime node."""
        self._cell._raise_if_not_initialized("inspect SynapseView.runtime_index")
        self._require_homogeneous_type()
        return np.array(self._store.runtime_rows(self._logical_ids), copy=True)

    def _column(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self._store, name))[self._store.row_indices(self._logical_ids)]

    def __len__(self) -> int:
        return int(self._logical_ids.size)

    def __getitem__(self, selector: object) -> "SynapseView":
        if isinstance(selector, Synapse):
            selected = [
                logical_id
                for logical_id in self._logical_ids.tolist()
                if self._store.declaration[self._store.row_index(int(logical_id))] is selector
            ]
            return SynapseView(self._cell, selected)
        if isinstance(selector, str):
            return self.by_name(selector)
        selected = self._logical_ids[selector]
        return SynapseView(self._cell, np.asarray(selected, dtype=np.int64).reshape(-1))

    def by_name(self, name: str) -> "SynapseView":
        """Return selected rows with one semantic synapse name."""
        if not isinstance(name, str) or not name:
            raise ValueError("synapse name must be a non-empty string.")
        return SynapseView(self._cell, self._logical_ids[self.name == name])

    def by_type(self, synapse_type: str) -> "SynapseView":
        """Return selected rows using one registered synapse type.

        Parameters
        ----------
        synapse_type : str
            Registry key such as ``"ExpSyn"``.

        Returns
        -------
        SynapseView
            Type-filtered view in the existing logical order.
        """
        if not isinstance(synapse_type, str) or not synapse_type:
            raise ValueError("synapse_type must be a non-empty string.")
        return SynapseView(self._cell, self._logical_ids[self.synapse_type == synapse_type])

    def for_population(self, population_indices) -> "SynapseView":
        """Return rows owned by selected population members.

        Parameters
        ----------
        population_indices : iterable of int
            Root Cell population indices.

        Returns
        -------
        SynapseView
            Population-filtered logical rows in existing view order.
        """
        selected = np.asarray(tuple(int(index) for index in population_indices), dtype=np.int64)
        mask = np.isin(self.population_index, selected)
        return SynapseView(self._cell, self._logical_ids[mask])

    def for_scope_pairs(self, pairs) -> "SynapseView":
        """Return rows whose owning ``(population, CV)`` pair is selected.

        Parameters
        ----------
        pairs : iterable of tuple of int
            Root population and CV identifiers.

        Returns
        -------
        SynapseView
            Scope-filtered rows in the existing logical order.
        """
        selected = {(int(population), int(cv_id)) for population, cv_id in pairs}
        mask = np.asarray(
            [
                (int(population), int(cv_id)) in selected
                for population, cv_id in zip(self.population_index.tolist(), self.cv_id.tolist())
            ],
            dtype=bool,
        )
        return SynapseView(self._cell, self._logical_ids[mask])

    def get(self, field: str):
        """Gather one model parameter or dynamic state in view order.

        Parameters
        ----------
        field : str
            Registered model parameter or initialized state name.

        Returns
        -------
        array-like
            Selected values with logical synapse rows on the last axis.

        Raises
        ------
        TypeError
            If the view contains more than one synapse type.
        KeyError
            If the field is unknown or is a state requested before initialization.
        """
        synapse_type = self._require_homogeneous_type()
        parameter_names = self._parameter_names(synapse_type)
        if not self._cell._initialized:
            if field not in parameter_names:
                raise KeyError(f"Synapse field {field!r} is unavailable before init_state().")
            return self._store.parameter_column(self._logical_ids, field)

        layout_id = self._store.layout_id(synapse_type)
        node = self._cell.runtime.get_runtime_node(layout_id)
        if not hasattr(node, field):
            raise KeyError(f"Synapse type {synapse_type!r} has no parameter or state {field!r}.")
        value = getattr(node, field)
        if isinstance(value, brainstate.nn.Param):
            value = value.value()
        if callable(value):
            raise KeyError(f"Synapse field {field!r} is callable and cannot be viewed as data.")
        if isinstance(value, brainstate.State):
            value = value.value
        return _take_last_axis(value, self._store.runtime_rows(self._logical_ids))

    def set(self, **parameters: object) -> "SynapseView":
        """Set model parameters before or after runtime materialization.

        Parameters
        ----------
        **parameters
            Parameter values, each scalar or aligned with the selected rows.

        Returns
        -------
        SynapseView
            This view.

        Raises
        ------
        KeyError
            If a field is unknown or is a dynamic state.
        ValueError
            If a value has an incompatible shape or unit.
        """
        synapse_type = self._require_homogeneous_type()
        valid = self._parameter_names(synapse_type)
        normalized_updates = {}
        for parameter, value in parameters.items():
            if parameter not in valid:
                if self._cell._initialized:
                    layout_id = self._store.layout_id(synapse_type)
                    candidate = getattr(self._cell.runtime.get_runtime_node(layout_id), parameter, None)
                    if isinstance(candidate, brainstate.State):
                        raise KeyError(f"Synapse field {parameter!r} is a dynamic state; use set_state().")
                raise KeyError(f"Synapse type {synapse_type!r} has no parameter {parameter!r}.")
            normalized_updates[parameter] = _normalize_selected_value(
                value, template=self.get(parameter), count=len(self)
            )

        if not self._cell._initialized:
            self._store.set_parameters(self._logical_ids, normalized_updates)
            for parameter, normalized in normalized_updates.items():
                for logical_id, item in zip(self._logical_ids.tolist(), _split_values(normalized)):
                    index = self._store.row_index(int(logical_id))
                    self._cell._synapse_parameter_overrides[
                        (
                            int(self._store.placement_id[index]),
                            int(self._store.population_index[index]),
                            str(parameter),
                        )
                    ] = item
            return self

        layout_id = self._store.layout_id(synapse_type)
        node = self._cell.runtime.get_runtime_node(layout_id)
        rows = self._store.runtime_rows(self._logical_ids)
        proposed = {}
        for field in valid:
            field_value = getattr(node, field)
            proposed[field] = field_value.value() if isinstance(field_value, brainstate.nn.Param) else field_value
        for parameter, normalized in normalized_updates.items():
            proposed[parameter] = _set_last_axis(proposed[parameter], rows, normalized)
        for parameter in normalized_updates:
            type(node).parameters[parameter].validate(proposed[parameter], parameter)
        type(node).validate_parameter_values(proposed)

        for parameter, updated in proposed.items():
            if parameter not in normalized_updates:
                continue
            runtime_parameter = getattr(node, parameter)
            if isinstance(runtime_parameter, brainstate.nn.Param):
                runtime_parameter.set_value(updated)
                self._cell.runtime.state_buffers[(layout_id, parameter)] = updated
                continue
            self._cell.runtime.set_state(layout_id, parameter, updated)
        return self

    def set_state(self, **states: object) -> "SynapseView":
        """Set initialized dynamic states while keeping parameters separate.

        Parameters
        ----------
        **states
            Dynamic state values, each scalar or aligned with selected rows.

        Returns
        -------
        SynapseView
            This view.

        Raises
        ------
        RuntimeError
            If the Cell has not been initialized.
        KeyError
            If a field is unknown or is a model parameter.
        """
        self._cell._raise_if_not_initialized("set SynapseView state")
        synapse_type = self._require_homogeneous_type()
        layout_id = self._store.layout_id(synapse_type)
        node = self._cell.runtime.get_runtime_node(layout_id)
        parameters = self._parameter_names(synapse_type)
        rows = self._store.runtime_rows(self._logical_ids)
        for state, value in states.items():
            target = getattr(node, state, None)
            if not isinstance(target, brainstate.State):
                if state in parameters:
                    raise KeyError(f"Synapse field {state!r} is a parameter; use set().")
                raise KeyError(f"Synapse type {synapse_type!r} has no dynamic state {state!r}.")
            current = target.value
            selected = _take_last_axis(current, rows)
            normalized = _normalize_selected_value(value, template=selected, count=len(self))
            target.value = _set_last_axis(current, rows, normalized)
        return self

    def _parameter_names(self, synapse_type: str) -> set[str]:
        runtime_cls = get_registry().get("synapse", str(synapse_type))
        return set(runtime_cls.parameters)

    def _require_homogeneous_type(self) -> str:
        types = tuple(dict.fromkeys(str(item) for item in self.synapse_type.tolist()))
        if len(types) == 0:
            raise ValueError("Cannot access model fields on an empty SynapseView.")
        if len(types) != 1:
            raise TypeError(f"SynapseView contains multiple synapse types {types!r}; select a name or use by_type().")
        return types[0]

    def __getattr__(self, field: str):
        if field.startswith("_"):
            raise AttributeError(field)
        try:
            return self.get(field)
        except KeyError as exc:
            raise AttributeError(str(exc)) from exc

    def __repr__(self) -> str:
        target = _cell_label(self._cell, self.population_index)
        if len(self) == 0:
            return f"SynapseView(target={target}, size=0)"
        types = tuple(dict.fromkeys(str(item) for item in self.synapse_type.tolist()))
        if len(types) == 1:
            synapse_type = types[0]
            names = _count_labels(self.name)
            parameters = tuple(sorted(self._parameter_names(synapse_type)))
            if len(self) == 1:
                values = {field: self.get(field)[0] for field in parameters}
                return (
                    f"SynapseView(target={target}, size=1, synapse_type={synapse_type}, "
                    f"names={names!r}, parameters={values!r})"
                )
            return (
                f"SynapseView(target={target}, size={len(self)}, synapse_type={synapse_type}, "
                f"names={names!r}, parameters={parameters!r})"
            )

        lines = [f"SynapseView(target={target}, size={len(self)})"]
        for synapse_type in types:
            selected = self.by_type(synapse_type)
            lines.append(f"  {synapse_type}  instances={len(selected)}  names={_count_labels(selected.name)!r}")
        return "\n".join(lines)


def _count_labels(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in np.asarray(values, dtype=object).tolist():
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _cell_label(cell, population_indices) -> str:
    owners = tuple(dict.fromkeys(int(item) for item in np.asarray(population_indices).tolist()))
    size = 1 if len(cell.pop_size) == 0 else int(cell.pop_size[0])
    if owners == tuple(range(size)):
        return f"Cell(name={cell.name!r})"
    return f"CellView(name={cell.name!r}, cells={list(owners)!r})"


@dataclass(frozen=True)
class _MechanismRowLayout:
    """Row bookkeeping shared by every logical synapse of one declaration.

    Declared parameters broadcast over the ``(owner, local position)`` grid this
    describes. It depends only on the declaration, so it is built once and
    reused for each of that declaration's synapses and parameters.
    """

    size: int
    owner_count: int
    common_length: int | None
    #: logical id -> (flat position, owner position, position within that owner)
    positions: dict[int, tuple[int, int, int]]

    @classmethod
    def build(cls, store: "_SynapseStore", mechanism: Synapse) -> "_MechanismRowLayout":
        matching_rows = np.asarray(
            [row for row, candidate in enumerate(store.mechanism) if candidate is mechanism],
            dtype=np.int64,
        )
        matching = store.id[matching_rows]
        owners = store.population_index[matching_rows]
        owner_order = tuple(dict.fromkeys(int(owner) for owner in owners.tolist()))
        owner_position_by_owner = {owner: index for index, owner in enumerate(owner_order)}
        counts = tuple(int(np.sum(owners == item)) for item in owner_order)
        common_length = counts[0] if counts and all(count == counts[0] for count in counts) else None

        positions: dict[int, tuple[int, int, int]] = {}
        seen_per_owner: dict[int, int] = {}
        for position, (logical_id, owner) in enumerate(zip(matching.tolist(), owners.tolist())):
            local_position = seen_per_owner.get(int(owner), 0)
            seen_per_owner[int(owner)] = local_position + 1
            positions[int(logical_id)] = (position, owner_position_by_owner[int(owner)], local_position)
        return cls(
            size=int(matching.size),
            owner_count=len(owner_order),
            common_length=common_length,
            positions=positions,
        )


def _select_declared_parameter_value(value, *, logical_id: int, store: _SynapseStore, mechanism: Synapse):
    shape = getattr(value, "shape", ())
    if shape in ((), None):
        return value
    layout = store.mechanism_row_layout(mechanism)
    position, owner_position, local_position = layout.positions[int(logical_id)]
    common_length = layout.common_length
    array = np.asarray(value.to_decimal(value.unit) if isinstance(value, u.Quantity) else value)

    if array.size == 1:
        selected = array.reshape(-1)[0]
    elif common_length is not None and array.shape == (common_length,):
        selected = array[local_position]
    elif array.shape == (layout.owner_count, 1):
        selected = array[owner_position, 0]
    elif common_length is not None and array.shape == (layout.owner_count, common_length):
        selected = array[owner_position, local_position]
    elif array.shape == (layout.size,):
        selected = array[position]
    else:
        raise ValueError(f"Synapse parameter shape {shape!r} cannot broadcast to {layout.size!r} logical instances.")
    return u.Quantity(selected, value.unit) if isinstance(value, u.Quantity) else selected


def _take_vector_items(value, indices):
    """Index into one stored parameter column, keeping its unit.

    ``indices`` may be a scalar row or an array of rows; the result follows
    numpy indexing, so a scalar index yields a scalar and a row vector yields a
    vector.
    """
    if isinstance(value, u.Quantity):
        return u.Quantity(np.asarray(value.to_decimal(value.unit))[indices], value.unit)
    return np.asarray(value)[indices]


def _set_vector_items(value, indices: np.ndarray, selected):
    if isinstance(value, u.Quantity):
        if not isinstance(selected, u.Quantity):
            raise TypeError("Synapse parameter requires a quantity.")
        array = np.array(value.to_decimal(value.unit), copy=True)
        array[indices] = np.asarray(selected.to_decimal(value.unit))
        return u.Quantity(array, value.unit)
    if isinstance(selected, u.Quantity):
        raise TypeError("Synapse parameter is dimensionless.")
    array = np.array(value, copy=True)
    array[indices] = np.asarray(selected)
    return array


def _normalize_selected_value(value, *, template, count: int):
    if isinstance(template, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError("Synapse field requires a quantity with compatible units.")
        try:
            array = np.asarray(value.to_decimal(template.unit))
        except Exception as exc:
            raise ValueError("Synapse field has incompatible units.") from exc
        if array.shape == ():
            array = np.broadcast_to(array, (count,)).copy()
        elif array.shape != (count,):
            raise ValueError(f"Synapse field must be scalar or shape {(count,)!r}, got {array.shape!r}.")
        return u.Quantity(array, template.unit)
    if isinstance(value, u.Quantity):
        raise TypeError("Synapse field is dimensionless.")
    array = np.asarray(value)
    if array.shape == ():
        array = np.broadcast_to(array, (count,)).copy()
    elif array.shape != (count,):
        raise ValueError(f"Synapse field must be scalar or shape {(count,)!r}, got {array.shape!r}.")
    return array


def _split_values(value) -> tuple[object, ...]:
    if isinstance(value, u.Quantity):
        return tuple(u.Quantity(item, value.unit) for item in np.asarray(value.to_decimal(value.unit)).tolist())
    return tuple(np.asarray(value).tolist())


def _take_last_axis(value, indices: np.ndarray):
    indices = np.asarray(indices, dtype=np.int64)
    return value[..., indices]


def _set_last_axis(value, indices: np.ndarray, selected):
    indices = np.asarray(indices, dtype=np.int64)
    if isinstance(value, u.Quantity):
        if not isinstance(selected, u.Quantity):
            raise TypeError("Synapse field requires a quantity.")
        mantissa = jnp.asarray(value.to_decimal(value.unit))
        replacement = jnp.asarray(selected.to_decimal(value.unit))
        return u.Quantity(mantissa.at[..., indices].set(replacement), value.unit)
    if isinstance(selected, u.Quantity):
        raise TypeError("Synapse field is dimensionless.")
    return jnp.asarray(value).at[..., indices].set(jnp.asarray(selected))
