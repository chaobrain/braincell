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

import itertools
from collections import Counter
from dataclasses import dataclass

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._compute.layouts import _stack_synapse_values
from braincell._multi_compartment.lifecycle import DiscreteView
from braincell.mech import Synapse, get_registry
from braincell.mech._synapse_schema import describe_mechanism, unit_error_message

__all__ = ["SynapseView"]

#: Numbers every synapse store, so a holder of logical ids can tell whether
#: they still refer to the store it took them from.
_STORE_SERIALS = itertools.count()

#: Branch name -> process-wide code, so synapse identities compare as integers
#: across stores (MORPH-15).
_BRANCH_NAME_CODES: dict[str, int] = {}


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
        "branch_name_code",
        "rule_index",
        "mechanism_index",
        "repeat_index",
        "sites",
        "serial",
        "_id_order",
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
        self.serial = next(_STORE_SERIALS)
        self._id_order = None
        synapse_placements = tuple(
            placement for placement in cell.point_placements if isinstance(placement.mechanism, Synapse)
        )
        if len(cell.pop_size) > 1 and synapse_placements:
            raise ValueError(
                f"Synapse views currently require scalar or one-dimensional pop_size; got {cell.pop_size!r}."
            )
        population_size = 1 if len(cell.pop_size) == 0 else int(cell.pop_size[0])
        # A placement's identity across rebuilds (MORPH-15): its place rule
        # and mechanism there, branch name, x, and which repeat of that
        # location it is. ``sites`` lists each rule-mechanism's locations.
        repeats: dict[tuple, int] = {}
        placement_repeat = []
        placement_owner = []
        sites: dict[tuple[int, int], list] = {}
        for placement in synapse_placements:
            rule_key = (placement._rule_index, placement._mechanism_index)
            member = -1 if placement.population_index is None else int(placement.population_index)
            site = (member, placement.branch_name, float(placement.branch_x))
            placement_repeat.append(repeats.get(rule_key + site, 0))
            repeats[rule_key + site] = placement_repeat[-1] + 1
            sites.setdefault(rule_key, []).append(site)
            placement_owner.append(member)
        self.sites = {key: tuple(sorted(value)) for key, value in sites.items()}

        # One row per (owner, placement): a placement without a population
        # index has a row in every member. Rows are built and ordered as
        # arrays over placements (CELL-38): Python work per row made this
        # build the main cost of ``init_state`` for large populations.
        owner = np.asarray(placement_owner, dtype=np.int64)
        broadcast = np.flatnonzero(owner < 0)
        single = np.flatnonzero(owner >= 0)
        placement_of_row = np.concatenate([np.tile(broadcast, population_size), single])
        owner_of_row = np.concatenate(
            [np.repeat(np.arange(population_size, dtype=np.int64), broadcast.size), owner[single]]
        )
        placement_ids = np.asarray([placement.id for placement in synapse_placements], dtype=np.int64)
        # Public logical order is stable and cell-major, independently of how
        # broadcast placements happen to be represented by discretization.
        order = np.lexsort((placement_ids[placement_of_row], owner_of_row))
        # Each placement's first row, so per-placement facts can be visited
        # in the order of the rows they stand for.
        first_row = np.full(len(synapse_placements), order.size, dtype=np.int64)
        np.minimum.at(first_row, placement_of_row[order], np.arange(order.size, dtype=np.int64))
        placement_of_row = placement_of_row[order]
        self.population_index = owner_of_row[order]
        self.placement_id = placement_ids[placement_of_row]
        self.id = self.placement_id * int(population_size) + self.population_index

        def per_row(values, dtype):
            return np.asarray(values, dtype=dtype)[placement_of_row]

        self.point_id = per_row([p.point_id for p in synapse_placements], np.int64)
        self.cv_id = per_row([p.cv_id for p in synapse_placements], np.int64)
        self.branch_id = per_row([p.branch_id for p in synapse_placements], np.int64)
        self.branch_x = per_row([p.branch_x for p in synapse_placements], float)
        self.branch_name_code = per_row(
            [_BRANCH_NAME_CODES.setdefault(p.branch_name, len(_BRANCH_NAME_CODES)) for p in synapse_placements],
            np.int64,
        )
        self.rule_index = per_row([p._rule_index for p in synapse_placements], np.int64)
        self.mechanism_index = per_row([p._mechanism_index for p in synapse_placements], np.int64)
        self.repeat_index = per_row(placement_repeat, np.int64)
        mechanisms = [p.mechanism for p in synapse_placements]
        origins = getattr(cell, "_synapse_origins", {})
        self.mechanism = _take_objects(mechanisms, placement_of_row)
        self.declaration = _take_objects([origins.get(id(item), item) for item in mechanisms], placement_of_row)
        self.name = per_row([item.instance_name for item in mechanisms], object)
        self.synapse_type = per_row([item.synapse_type for item in mechanisms], object)
        self._row_by_id = dict(zip(self.id.tolist(), range(self.id.size)))
        self.parameter_columns: dict[str, dict[str, object]] = {}
        self._type_rows: dict[str, np.ndarray] = {}
        self._type_local_by_id: dict[int, int] = {}
        self._layout_by_type: dict[str, int] = {}
        self._runtime_row_by_id: dict[int, int] = {}
        self._mechanism_layouts: dict[int, _MechanismRowLayout] = {}
        # Placements with rows, in the order of their first rows. A check
        # that stops at the first failing row stops at the first row of a
        # placement or declaration, since its other rows repeat that row.
        by_first_row = np.argsort(first_row, kind="stable")
        in_row_order = by_first_row[first_row[by_first_row] < order.size].tolist()
        raise_on_name_type_conflict(
            (mechanisms[item].instance_name, mechanisms[item].synapse_type) for item in in_row_order
        )
        # The distinct declarations (by identity) in that order, and the
        # number of each row's.
        declarations: dict[int, Synapse] = {}
        for item in in_row_order:
            declarations.setdefault(id(mechanisms[item]), mechanisms[item])
        code_by_id = {key: code for code, key in enumerate(declarations)}
        declaration_of_placement = np.asarray([code_by_id.get(id(item), -1) for item in mechanisms], dtype=np.int64)
        self._build_parameter_columns(tuple(declarations.values()), declaration_of_placement[placement_of_row])

    def _error_contexts(self) -> dict[int, tuple[str, str | None]]:
        """Map each declaration (by identity) to its error owner and ``place`` source."""
        contexts = {}
        for rule in self.cell.place_rules:
            at = f" at {rule.origin}" if rule.origin else ""
            for mechanism in rule.mechanisms:
                if isinstance(mechanism, Synapse) and id(mechanism) not in contexts:
                    owner = describe_mechanism("synapse", mechanism.synapse_type, mechanism.instance_name)
                    contexts[id(mechanism)] = (owner, f"placed on {rule.locset!r}{at}.")
        return contexts

    def _build_parameter_columns(self, declarations: tuple[Synapse, ...], declaration_of_row: np.ndarray) -> None:
        """Stack every parameter of each synapse type into one column.

        ``declarations`` are the distinct declarations behind the rows, in
        the order of their first rows, and ``declaration_of_row`` numbers
        each row's. Each declaration's value is resolved and checked once
        for all of its rows (CELL-38). A column whose fast build fails is
        rebuilt one row at a time, which raises the error of the first
        failing row with the declaration's name and ``place`` source.
        """
        for synapse_type in dict.fromkeys(str(item.synapse_type) for item in declarations):
            runtime_cls = get_registry().get("synapse", synapse_type)
            schema = runtime_cls.parameter_info()
            codes = np.asarray(
                [code for code, item in enumerate(declarations) if str(item.synapse_type) == synapse_type],
                dtype=np.int64,
            )
            for code in codes.tolist():
                unknown = set(declarations[code].params).difference(schema)
                if unknown:
                    raise TypeError(f"Synapse type {synapse_type!r} has no parameters {tuple(sorted(unknown))!r}.")
            in_type = np.zeros(len(declarations), dtype=bool)
            in_type[codes] = True
            rows = np.flatnonzero(in_type[declaration_of_row]).astype(np.int64)
            self._type_rows[synapse_type] = rows
            self._type_local_by_id.update(zip(self.id[rows].tolist(), range(rows.size)))
            # The type's rows grouped by declaration, in row order within each
            # group. Rows are cell-major, so a group's members come in runs:
            # number each run within its group (``owner``) and each row
            # within its run (``local``), as ``_MechanismRowLayout`` does.
            local_code = declaration_of_row[rows]
            by_code = np.argsort(local_code, kind="stable")
            sorted_code = local_code[by_code]
            members = self.population_index[rows[by_code]]
            index = np.arange(rows.size)
            group_start = np.r_[True, sorted_code[1:] != sorted_code[:-1]]
            run_start = group_start | np.r_[True, members[1:] != members[:-1]]
            run = np.cumsum(run_start) - 1
            owner = run - run[np.maximum.accumulate(np.where(group_start, index, 0))]
            local = index - np.maximum.accumulate(np.where(run_start, index, 0))
            bounds = zip(
                np.searchsorted(sorted_code, codes, side="left").tolist(),
                np.searchsorted(sorted_code, codes, side="right").tolist(),
            )
            groups = [
                (declarations[code], by_code[low:high], owner[low:high], local[low:high])
                for code, (low, high) in zip(codes.tolist(), bounds)
            ]
            columns = {}
            for parameter, spec in schema.items():
                try:
                    columns[parameter] = _column_by_declaration(parameter, spec, groups, rows.size)
                except Exception:
                    columns[parameter] = self._column_by_row(parameter, spec, rows)
            try:
                runtime_cls.validate_parameter_values(columns)
            except ValueError as exc:
                raise self._name_the_invalid_declaration(
                    runtime_cls, columns, rows, self._error_contexts(), exc
                ) from None
            self.parameter_columns[synapse_type] = columns

    def _column_by_row(self, parameter: str, spec, rows: np.ndarray):
        """Build one column a row at a time, raising at the first failing row.

        This is the reference for :func:`_column_by_declaration`, used only
        when that fails: its errors name the synapse and where it was placed.
        """
        contexts = self._error_contexts()
        values = []
        for store_row in rows.tolist():
            mechanism = self.mechanism[store_row]
            if parameter in mechanism.params:
                value = _select_declared_parameter_value(
                    mechanism.params[parameter],
                    logical_id=int(self.id[store_row]),
                    store=self,
                    mechanism=mechanism,
                )
            else:
                value = spec.default
            owner, source = contexts.get(id(mechanism), (None, None))
            spec.validate(value, parameter, owner=owner, source=source)
            values.append(value)
        return _stack_synapse_values(values, parameter=parameter)

    def _name_the_invalid_declaration(self, runtime_cls, columns, rows, contexts, error: ValueError) -> ValueError:
        """Return ``error`` naming a declaration whose own rows fail the type's check.

        ``validate_parameter_values`` checks the columns of every synapse of a
        type at once (positive time constants, ``tau1 < tau2``), so its message
        cannot say which ``place`` declared the bad value. This runs only after
        that check failed, so a valid store pays nothing for it. It halves the
        failing range of declarations until one is left, about ``2 log2(n)``
        checks for ``n`` declarations (one per cell with per-cell locations),
        and finds the first failing declaration when the check is elementwise.
        When neither half fails alone (a check relating rows of different
        declarations), ``error`` is returned as is.
        """
        mechanisms = [self.mechanism[store_row] for store_row in rows.tolist()]
        code_by_id: dict[int, int] = {}
        codes = np.asarray([code_by_id.setdefault(id(item), len(code_by_id)) for item in mechanisms], dtype=np.int64)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(code_by_id) + 1))
        values = {
            name: (np.asarray(column.to_decimal(column.unit)), column.unit)
            if isinstance(column, u.Quantity)
            else (np.asarray(column), None)
            for name, column in columns.items()
        }

        def failure(first: int, last: int) -> ValueError | None:
            selected = order[bounds[first] : bounds[last]]
            subset = {
                name: array[selected] if unit is None else u.Quantity(array[selected], unit)
                for name, (array, unit) in values.items()
            }
            try:
                runtime_cls.validate_parameter_values(subset)
            except ValueError as exc:
                return exc
            return None

        first, last, found = 0, len(code_by_id), error
        while last - first > 1:
            middle = (first + last) // 2
            for low, high in ((first, middle), (middle, last)):
                narrower = failure(low, high)
                if narrower is not None:
                    first, last, found = low, high, narrower
                    break
            else:
                return error
        mechanism = mechanisms[int(order[bounds[first]])]
        fallback = describe_mechanism("synapse", mechanism.synapse_type, mechanism.instance_name)
        owner, source = contexts.get(id(mechanism), (fallback, None))
        return _name_the_synapse(found, owner, source)

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

        Notes
        -----
        Every caller holds a whole selection, so this is the only read
        accessor. Reading a selection one id at a time would be quadratic:
        each call re-materializes the entire per-type column through
        ``to_decimal`` only to take a single element, and the caller then
        restacks the scalars it just tore apart. Slicing the column once is
        constant in the number of rows read.
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
            spec = runtime_cls.parameter_info()[parameter]
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

    def rows_of(self, logical_ids) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rows, found)`` for ``logical_ids``: vectorized :meth:`row_indices` that does not raise."""
        ids = np.asarray(logical_ids, dtype=np.int64).reshape(-1)
        if self._id_order is None:
            self._id_order = np.argsort(self.id, kind="stable")
        order = self._id_order
        sorted_ids = self.id[order]
        position = np.minimum(np.searchsorted(sorted_ids, ids), max(len(sorted_ids) - 1, 0))
        found = sorted_ids[position] == ids if len(sorted_ids) else np.zeros(ids.shape, dtype=bool)
        return (order[position] if len(sorted_ids) else position), found

    def identity(self, rows) -> tuple[np.ndarray, ...]:
        """``(rule, mechanism, population, branch name code, x, repeat)`` of ``rows``.

        This names a logical synapse across rebuilds of the grid: its place
        rule and the mechanism within that rule, its population member, its
        branch by name and ``branch_x``, and which repeat of that location
        it is. Its logical id is not stable: it numbers placements across
        all rules, so an earlier rule gaining a site shifts it (MORPH-15).
        """
        rows = np.asarray(rows, dtype=np.int64)
        return (
            self.rule_index[rows],
            self.mechanism_index[rows],
            self.population_index[rows],
            self.branch_name_code[rows],
            self.branch_x[rows],
            self.repeat_index[rows],
        )

    def ids_with_identity(self, identity) -> np.ndarray:
        """Logical ids of the synapses with ``identity`` (:meth:`identity`), ``-1`` where none has it."""
        rule, mechanism = identity[0], identity[1]
        candidates = np.zeros(len(self.id), dtype=bool)
        for key in dict.fromkeys(zip(rule.tolist(), mechanism.tolist())):
            candidates |= (self.rule_index == key[0]) & (self.mechanism_index == key[1])
        table_rows = np.flatnonzero(candidates)
        n_table = len(table_rows)
        group = _identity_groups([np.concatenate(pair) for pair in zip(self.identity(table_rows), identity)])
        lookup = np.full(int(group.max(initial=-1)) + 1, -1, dtype=np.int64)
        lookup[group[:n_table]] = self.id[table_rows]
        return lookup[group[n_table:]]


def _identity_groups(columns) -> np.ndarray:
    """Number equal rows of ``columns`` alike: one group number per row."""
    order = np.lexsort(columns[::-1])
    starts = np.zeros(len(order), dtype=bool)
    starts[:1] = True
    for column in columns:
        ordered = column[order]
        starts[1:] |= ordered[1:] != ordered[:-1]
    group = np.empty(len(order), dtype=np.int64)
    group[order] = np.cumsum(starts) - 1
    return group


class SynapseView(DiscreteView):
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
        self._bind_view(cell)

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
        self._check_view()
        return int(self._logical_ids.size)

    def __getitem__(self, selector: object) -> "SynapseView":
        self._check_view()
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

    def trainable(self, **fields):
        """Bind parameter sources to selected logical synapses.

        Parameters
        ----------
        **fields
            Constructor parameter names mapped to trainable sources.

        Returns
        -------
        SynapseView
            This selection.
        """
        from braincell.trainable._targets import register_synapse

        register_synapse(self, fields)
        return self

    def parameter_info(self):
        """Return signature-derived parameter metadata for this synapse type.

        Returns
        -------
        dict
            Parameter metadata keyed by constructor field.
        """
        return get_registry().get("synapse", self._require_homogeneous_type()).parameter_info()

    def set(self, **parameters: object) -> "SynapseView":
        """Set existing model parameters after runtime initialization.

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
            If a value has an incompatible shape or unit, or fails the
            synapse type's check (a non-positive time constant, ``tau1 >=
            tau2``); the message names the synapse, and no value changes.
        """
        self._cell._raise_if_not_initialized(
            "SynapseView.set()", hint="Before init_state(), give the values in the place() declaration."
        )
        synapse_type = self._require_homogeneous_type()
        from braincell.trainable._targets import require_unbound

        for field in parameters:
            require_unbound(self._cell, "synapse", synapse_type, self.id, field)
        valid = self._parameter_names(synapse_type)
        names = set(self.name.tolist())
        owner = describe_mechanism("synapse", synapse_type, names.pop() if len(names) == 1 else None)
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
                value, template=self.get(parameter), count=len(self), name=parameter, owner=owner
            )

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
            type(node).parameter_info()[parameter].validate(proposed[parameter], parameter, owner=owner)
        try:
            type(node).validate_parameter_values(proposed)
        except ValueError as exc:
            raise _name_the_synapse(exc, owner) from None

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
        return set(runtime_cls.parameter_info())

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


def raise_on_name_type_conflict(pairs, *, known: dict[str, str] | None = None) -> dict[str, str]:
    """Reject one synapse name standing for two different synapse types.

    A synapse *name* is the handle a user reads back through
    ``cell.synapses[name]``, so it has to denote one runtime type. The rule
    is checked twice against different inputs -- at ``Cell.place()`` time
    against the declarations, and again when the store materializes its
    columns -- which is why it is stated here rather than in either caller.

    Parameters
    ----------
    pairs : iterable of tuple
        ``(name, synapse_type)`` pairs, in declaration order.
    known : dict, optional
        Names already claimed by an earlier pass. Not mutated.

    Returns
    -------
    dict
        ``name -> synapse_type`` for every pair seen, including ``known``.

    Raises
    ------
    ValueError
        On the first name that claims a second type.
    """
    type_by_name = {} if known is None else dict(known)
    for name, synapse_type in pairs:
        name, synapse_type = str(name), str(synapse_type)
        previous = type_by_name.setdefault(name, synapse_type)
        if previous != synapse_type:
            raise ValueError(
                f"Synapses with the same name {name!r} cannot use different synapse types "
                f"({previous!r} and {synapse_type!r})."
            )
    return type_by_name


def _count_labels(values) -> dict[str, int]:
    """Count each label, in first-seen order.

    Returned as a plain ``dict`` rather than the ``Counter`` itself
    because the callers interpolate it with ``{...!r}`` into a repr.
    """
    return dict(Counter(str(value) for value in np.asarray(values, dtype=object).tolist()))


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


def _select_declared_rows(array: np.ndarray, owner: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Pick the entry of an array-valued declared parameter for each of its rows.

    ``owner`` numbers the member of each of the declaration's rows among its
    members, from 0 and in row order; ``local`` numbers the row among that
    member's rows. This is :func:`_select_declared_parameter_value` for all
    rows of the declaration at once.
    """
    size = owner.size
    owner_count = int(owner[-1]) + 1
    counts = np.bincount(owner, minlength=owner_count)
    common_length = int(counts[0]) if np.all(counts == counts[0]) else None
    if array.size == 1:
        return np.repeat(array.reshape(-1)[:1], size)
    if common_length is not None and array.shape == (common_length,):
        return array[local]
    if array.shape == (owner_count, 1):
        return array[owner, 0]
    if common_length is not None and array.shape == (owner_count, common_length):
        return array[owner, local]
    if array.shape == (size,):
        return array
    raise ValueError(f"Synapse parameter shape {array.shape!r} cannot broadcast to {size!r} logical instances.")


def _column_by_declaration(parameter: str, spec, groups, size: int):
    """Build one parameter column of a synapse type from its declarations.

    Parameters
    ----------
    parameter : str
        Parameter name.
    spec : ParameterSpec
        The type's declaration of ``parameter``.
    groups : list of tuple
        ``(declaration, positions, owner, local)`` for each declaration of the
        type, in the order of their first rows: ``positions`` index the
        type's rows, ``owner`` and ``local`` are as in
        :func:`_select_declared_rows`.
    size : int
        Number of rows of the type.

    Returns
    -------
    array-like
        The column the row-by-row build stacks: equal values, dtype, unit and
        order.

    Raises
    ------
    Exception
        Whatever a value fails on; the caller then rebuilds the column one
        row at a time to raise the first failing row's error with its context.

    Notes
    -----
    A scalar or default value is one object shared by all rows that use it,
    so checking it once checks all of them. Array values are checked as the
    column of their rows, one check per unit and dtype, as ``SynapseView.set``
    and the runtime constructors check columns. The column takes the unit of
    the first row and the dtype ``numpy.asarray`` gives the per-row values.
    """
    first = None
    shared: dict[int, list] = {}
    batches: dict[tuple, list] = {}
    for declaration, positions, owner, local in groups:
        value = declaration.params[parameter] if parameter in declaration.params else spec.default
        if first is None:
            first = value
        if getattr(value, "shape", ()) in ((), None):
            shared.setdefault(id(value), [value, []])[1].append(positions)
            continue
        unit = value.unit if isinstance(value, u.Quantity) else None
        selected = _select_declared_rows(np.asarray(value if unit is None else value.to_decimal(unit)), owner, local)
        batch = batches.setdefault((id(unit), selected.dtype), [unit, [], []])
        batch[1].append(selected)
        batch[2].append(positions)
    pieces = []
    for value, positions in shared.values():
        spec.validate(value, parameter)
        pieces.append((value, np.concatenate(positions), ()))
    for unit, arrays, positions in batches.values():
        array = np.concatenate(arrays)
        value = array if unit is None else u.Quantity(array, unit)
        spec.validate(value, parameter)
        pieces.append((value, np.concatenate(positions), array.shape))
    if isinstance(first, u.Quantity):
        decimals = [np.asarray(value.to_decimal(first.unit)) for value, _, _ in pieces]
    elif any(isinstance(value, u.Quantity) for value, _, _ in pieces):
        raise TypeError(f"Synapse parameter {parameter!r} mixes dimensioned and dimensionless values.")
    else:
        decimals = [np.asarray(value) for value, _, _ in pieces]
    column = np.empty(size, dtype=np.result_type(*(decimal.dtype for decimal in decimals)))
    for decimal, (_, positions, shape) in zip(decimals, pieces):
        if decimal.shape != shape:
            raise ValueError(f"Synapse parameter {parameter!r} must resolve to one scalar per logical instance.")
        column[positions] = decimal
    return u.Quantity(column, first.unit) if isinstance(first, u.Quantity) else column


def _take_objects(items, rows: np.ndarray) -> tuple:
    """Return ``tuple(items[row] for row in rows)`` without a Python loop over ``rows``."""
    objects = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        objects[index] = item
    return tuple(objects[rows].tolist())


#: How a type's ``validate_parameter_values`` names a parameter when it does
#: not know which synapse it checks (``braincell.mech.positive``).
_UNNAMED_PARAMETER = "Mechanism parameter "


def _name_the_synapse(error: ValueError, owner: str, source: str | None = None) -> ValueError:
    """Return a ``ValueError`` saying which synapse ``error`` is about.

    The message takes the form of the per-value checks (``<owner> parameter
    '<name>' ...``, then ``source``) when ``error`` names a parameter the
    way :func:`braincell.mech.positive` does, and ``<owner>: <message>``
    otherwise (``Exp2Syn requires tau1 < tau2.``).
    """
    message = str(error)
    if message.startswith(_UNNAMED_PARAMETER):
        message = f"{owner} parameter {message[len(_UNNAMED_PARAMETER) :]}"
    else:
        message = f"{owner}: {message}"
    if source:
        message = f"{message} {source[:1].upper()}{source[1:]}"
    return ValueError(message)


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


def _normalize_selected_value(value, *, template, count: int, name=None, owner=None):
    def message(problem: str, legacy: str) -> str:
        return legacy if name is None else unit_error_message(name, problem, owner=owner, value=value)

    if isinstance(template, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError(
                message(
                    f"requires a quantity compatible with {template.unit}",
                    "Synapse field requires a quantity with compatible units.",
                )
            )
        try:
            array = np.asarray(value.to_decimal(template.unit))
        except Exception as exc:
            raise ValueError(
                message(f"has units incompatible with {template.unit}", "Synapse field has incompatible units.")
            ) from exc
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
