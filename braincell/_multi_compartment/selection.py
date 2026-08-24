"""Immutable spatial selections shared by Cell mechanism views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from braincell._discretization.node_build import _EPS_PARAM, _locate_branch_cv_by_x
from braincell.filter import LocsetBatch, LocsetExpr, LocsetMask, RegionExpr, RegionMask

__all__ = ["BranchSelector", "CVSelector"]


@dataclass(frozen=True)
class _LocationRow:
    population_index: int
    branch_id: int
    branch_x: float
    cv_id: int
    display_name: str


@dataclass(frozen=True)
class _CellScope:
    """Static population/spatial selection without owning model arrays."""

    population_indices: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    coverage: tuple[tuple[int, float], ...] = ()
    locations: tuple[_LocationRow, ...] = ()
    exact_branch_ids: tuple[int, ...] | None = None
    spatially_restricted: bool = False

    @classmethod
    def root(cls, cell) -> "_CellScope":
        populations = tuple(range(cell._population_size))
        pairs = tuple((population, cv_id) for population in populations for cv_id in range(cell.n_cv))
        return cls(population_indices=populations, pairs=pairs)

    @property
    def cv_ids(self) -> tuple[int, ...]:
        return _ordered_unique(cv_id for _, cv_id in self.pairs)

    @property
    def branch_ids(self) -> tuple[int, ...]:
        return self.exact_branch_ids if self.exact_branch_ids is not None else ()

    @property
    def pair_population_index(self) -> np.ndarray:
        return np.asarray([population for population, _ in self.pairs], dtype=np.int64)

    @property
    def pair_cv_id(self) -> np.ndarray:
        return np.asarray([cv_id for _, cv_id in self.pairs], dtype=np.int64)

    @property
    def coverage_by_cv(self) -> dict[int, float]:
        return dict(self.coverage)

    def select_population(self, local_selection) -> "_CellScope":
        local = _normalize_selection(local_selection, size=len(self.population_indices), name="Cell population")
        selected = tuple(self.population_indices[index] for index in local)
        return _CellScope(
            population_indices=selected,
            pairs=tuple(pair for population in selected for pair in self.pairs if pair[0] == population),
            coverage=self.coverage,
            locations=tuple(
                row for population in selected for row in self.locations if row.population_index == population
            ),
            exact_branch_ids=self.exact_branch_ids,
            spatially_restricted=self.spatially_restricted,
        )

    def select_cv_local(self, selection) -> "_CellScope":
        cv_ids = self.cv_ids
        local = _normalize_selection(selection, size=len(cv_ids), name="CV")
        return self.select_cv_ids(tuple(cv_ids[index] for index in local))

    def select_cv_ids(self, cv_ids) -> "_CellScope":
        requested = _normalize_global_ids(cv_ids, name="CV")
        requested_set = set(requested)
        available = set(self.cv_ids)
        selected = tuple(cv_id for cv_id in requested if cv_id in available)
        selected_set = set(selected)
        return _CellScope(
            population_indices=self.population_indices,
            pairs=tuple(pair for pair in self.pairs if pair[1] in selected_set),
            coverage=tuple((cv_id, fraction) for cv_id, fraction in self.coverage if cv_id in selected_set),
            locations=tuple(row for row in self.locations if row.cv_id in selected_set),
            exact_branch_ids=self.exact_branch_ids,
            spatially_restricted=True,
        )

    def select_branches(self, cell, branch_ids) -> "_CellScope":
        selected = _normalize_global_ids(branch_ids, name="branch")
        if any(branch_id < 0 or branch_id >= len(cell.morpho.branches) for branch_id in selected):
            raise IndexError("Branch index is out of range.")
        cv_ids = tuple(cv.id for cv in cell.cvs if cv.branch_id in set(selected))
        result = self.select_cv_ids(cv_ids)
        return _CellScope(
            population_indices=result.population_indices,
            pairs=result.pairs,
            coverage=result.coverage,
            locations=result.locations,
            exact_branch_ids=selected,
            spatially_restricted=True,
        )

    def select_region(self, cell, region: RegionExpr | RegionMask) -> "_CellScope":
        if not isinstance(region, (RegionExpr, RegionMask)):
            raise TypeError(f"Cell.on(...) expects RegionExpr or RegionMask, got {type(region).__name__!s}.")
        fractions = cell._cv_coverage_fractions(region)
        selected = tuple(cv_id for cv_id, fraction in fractions.items() if float(fraction) > 0.0)
        result = self.select_cv_ids(selected)
        selected_set = set(result.cv_ids)
        return _CellScope(
            population_indices=result.population_indices,
            pairs=result.pairs,
            coverage=tuple((cv_id, float(fractions[cv_id])) for cv_id in selected if cv_id in selected_set),
            locations=(),
            exact_branch_ids=None,
            spatially_restricted=True,
        )

    def select_locations(self, cell, locset) -> "_CellScope":
        rows = _resolve_location_rows(cell, self.population_indices, locset)
        allowed = set(self.pairs)
        rows = tuple(row for row in rows if (row.population_index, row.cv_id) in allowed)
        pairs = _ordered_unique((row.population_index, row.cv_id) for row in rows)
        return _CellScope(
            population_indices=self.population_indices,
            pairs=pairs,
            locations=rows,
            exact_branch_ids=None,
            spatially_restricted=True,
        )


class BranchSelector:
    """Select morphology branches from one Cell scope."""

    __slots__ = ("_owner",)

    def __init__(self, owner) -> None:
        self._owner = owner

    def __getitem__(self, selector):
        if isinstance(selector, str):
            branch = self._owner.root.morpho.branch(name=selector)
            return self._owner._with_scope(self._owner._scope.select_branches(self._owner.root, (branch.index,)))
        branch_ids = np.arange(len(self._owner.root.morpho.branches), dtype=np.int64)
        selected = np.asarray(branch_ids[selector]).reshape(-1)
        return self._owner._with_scope(self._owner._scope.select_branches(self._owner.root, selected))

    def by_type(self, branch_type: str):
        if not isinstance(branch_type, str) or not branch_type:
            raise ValueError("branch type must be a non-empty string.")
        selected = tuple(branch.index for branch in self._owner.root.morpho.branches if branch.type == branch_type)
        return self._owner._with_scope(self._owner._scope.select_branches(self._owner.root, selected))


class CVSelector:
    """Select control volumes from one Cell scope."""

    __slots__ = ("_owner",)

    def __init__(self, owner) -> None:
        self._owner = owner

    @property
    def ids(self) -> np.ndarray:
        return np.asarray(self._owner._scope.cv_ids, dtype=np.int64)

    @property
    def coverage_fraction(self) -> np.ndarray:
        coverage = self._owner._scope.coverage_by_cv
        return np.asarray([coverage.get(cv_id, 1.0) for cv_id in self._owner._scope.cv_ids], dtype=float)

    @property
    def declarations(self):
        return tuple(self._owner.root.cvs[cv_id] for cv_id in self._owner._scope.cv_ids)

    def __len__(self) -> int:
        return len(self._owner._scope.cv_ids)

    def __getitem__(self, selector):
        return self._owner._with_scope(self._owner._scope.select_cv_local(selector))

    def by_id(self, ids):
        return self._owner._with_scope(self._owner._scope.select_cv_ids(ids))

    def __repr__(self) -> str:
        return f"CVSelector(ids={self.ids.tolist()!r})"


def _resolve_location_rows(cell, population_indices: tuple[int, ...], locset) -> tuple[_LocationRow, ...]:
    if isinstance(locset, LocsetExpr):
        locset = locset.evaluate(cell.morpho)
    if isinstance(locset, LocsetMask):
        return tuple(
            row for population_index in population_indices for row in _mask_rows(cell, population_index, locset)
        )
    if isinstance(locset, LocsetBatch):
        if len(locset) != len(population_indices):
            raise ValueError(
                "LocsetBatch rows must match the selected population size; "
                f"got {len(locset)!r} and {len(population_indices)!r}."
            )
        return tuple(
            row
            for population_index, mask in zip(population_indices, (locset[index] for index in range(len(locset))))
            for row in _mask_rows(cell, population_index, mask)
        )
    raise TypeError(f"Cell.loc(...) expects LocsetExpr, LocsetMask, or LocsetBatch, got {type(locset).__name__!s}.")


def _mask_rows(cell, population_index: int, mask: LocsetMask) -> tuple[_LocationRow, ...]:
    names = mask.display_names
    if names is None:
        names = tuple(
            f"{cell.morpho.branch(index=int(branch_id)).name}({float(branch_x):g})"
            for branch_id, branch_x in mask.points
        )
    rows = []
    for branch_id, branch_x, display_name in zip(mask.branch_id, mask.branch_x, names):
        branch_id = int(branch_id)
        if branch_id < 0 or branch_id >= len(cell.cv_tree.branch_to_cv_ids):
            raise IndexError(f"Location branch id {branch_id!r} is out of range.")
        cv_id = _locate_branch_cv_by_x(
            cell.cv_tree.branch_to_cv_ids[branch_id],
            cell.cvs,
            x=float(branch_x),
            epsilon=_EPS_PARAM,
        )
        rows.append(
            _LocationRow(
                population_index=int(population_index),
                branch_id=branch_id,
                branch_x=float(branch_x),
                cv_id=int(cv_id),
                display_name=str(display_name),
            )
        )
    return tuple(rows)


def _normalize_selection(selection, *, size: int, name: str) -> tuple[int, ...]:
    if isinstance(selection, slice):
        selected = tuple(range(size))[selection]
    elif isinstance(selection, (int, np.integer)) and not isinstance(selection, bool):
        index = int(selection)
        if index < 0:
            index += size
        selected = (index,)
    else:
        values = np.asarray(selection)
        if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
            raise TypeError(f"{name} selection must be an integer, slice, or one-dimensional integer sequence.")
        normalized = []
        for raw in values.tolist():
            index = int(raw)
            if index < 0:
                index += size
            normalized.append(index)
        selected = _ordered_unique(normalized)
    if any(index < 0 or index >= size for index in selected):
        raise IndexError(f"{name} selection is outside [0, {size!r}): {selected!r}.")
    return tuple(selected)


def _normalize_global_ids(ids, *, name: str) -> tuple[int, ...]:
    values = np.asarray(ids)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise TypeError(f"{name} ids must be a one-dimensional integer selection.")
    return _ordered_unique(int(item) for item in values.tolist())


def _ordered_unique(values) -> tuple:
    return tuple(dict.fromkeys(values))
