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

"""Logical current-clamp storage and user-facing selection views."""

from __future__ import annotations

from dataclasses import dataclass

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._compute.layouts import CLAMP_KINDS, _evaluate_clamp_layout
from braincell.mech import CurrentClamp, FunctionClamp, SineClamp

__all__ = ["ClampView"]

_CLAMP_TYPES = (CurrentClamp, SineClamp, FunctionClamp)


@dataclass(frozen=True)
class _ClampRecord:
    id: int
    placement_id: int
    population_index: int
    point_id: int
    cv_id: int
    branch_id: int
    branch_x: float
    clamp_type: str
    declaration: object


class _ClampStore:
    """Cell-owned static columns for logical current-clamp instances."""

    def __init__(self, cell) -> None:
        self.cell = cell
        n_population = int(np.prod(cell.pop_size, dtype=int))
        rows = []
        for placement in cell.point_placements:
            if not isinstance(placement.mechanism, _CLAMP_TYPES):
                continue
            owners = range(n_population) if placement.population_index is None else (int(placement.population_index),)
            for population_index in owners:
                logical_id = int(placement.id) * n_population + int(population_index)
                rows.append((logical_id, int(population_index), placement))
        rows.sort(key=lambda item: item[0])

        self.id = np.asarray([row[0] for row in rows], dtype=np.int64)
        self.population_index = np.asarray([row[1] for row in rows], dtype=np.int64)
        self.placement_id = np.asarray([row[2].id for row in rows], dtype=np.int64)
        self.point_id = np.asarray([row[2].point_id for row in rows], dtype=np.int64)
        self.cv_id = np.asarray([row[2].cv_id for row in rows], dtype=np.int64)
        self.branch_id = np.asarray([row[2].branch_id for row in rows], dtype=np.int64)
        self.branch_x = np.asarray([row[2].branch_x for row in rows], dtype=float)
        self.declaration = tuple(row[2].mechanism for row in rows)
        self.clamp_type = np.asarray([type(item).__name__ for item in self.declaration], dtype=object)
        self._row_by_id = {int(logical_id): index for index, logical_id in enumerate(self.id.tolist())}

    def row_indices(self, logical_ids) -> np.ndarray:
        try:
            return np.asarray([self._row_by_id[int(item)] for item in np.asarray(logical_ids).tolist()], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"Unknown logical clamp id {exc.args[0]!r}.") from exc

    def records(self, logical_ids) -> tuple[_ClampRecord, ...]:
        records = []
        for index in self.row_indices(logical_ids).tolist():
            records.append(
                _ClampRecord(
                    id=int(self.id[index]),
                    placement_id=int(self.placement_id[index]),
                    population_index=int(self.population_index[index]),
                    point_id=int(self.point_id[index]),
                    cv_id=int(self.cv_id[index]),
                    branch_id=int(self.branch_id[index]),
                    branch_x=float(self.branch_x[index]),
                    clamp_type=str(self.clamp_type[index]),
                    declaration=self.declaration[index],
                )
            )
        return tuple(records)

    def evaluate(self, runtime, *, t):
        """Evaluate every logical clamp once and return logical-row currents."""
        components = jnp.zeros((len(self.id),), dtype=float)
        n_population = int(np.prod(runtime.pop_size, dtype=int))
        for layout in runtime.layouts:
            if layout.kind not in CLAMP_KINDS or layout.point_index is None:
                continue
            values = _evaluate_clamp_layout(runtime, layout=layout, t=t).to_decimal(u.nA)
            values = jnp.asarray(values)
            if layout.population_index is None:
                logical_ids = (
                    np.asarray(layout.placement_index, dtype=np.int64)[None, :] * n_population
                    + np.arange(n_population, dtype=np.int64)[:, None]
                )
                component_rows = np.asarray(
                    [self._row_by_id[int(logical_id)] for logical_id in logical_ids.reshape(-1).tolist()],
                    dtype=np.int64,
                )
                components = components.at[component_rows].set(values.reshape(-1))
            else:
                logical_ids = np.asarray(layout.placement_index, dtype=np.int64) * n_population + np.asarray(
                    layout.population_index, dtype=np.int64
                )
                component_rows = np.asarray(
                    [self._row_by_id[int(logical_id)] for logical_id in logical_ids.tolist()],
                    dtype=np.int64,
                )
                components = components.at[component_rows].set(values.reshape(-1))
        return u.Quantity(components, u.nA)

    def scatter_to_points(self, components, *, pop_size: tuple[int, ...], n_point: int):
        """Scatter logical component currents to the solver's point space."""
        n_population = int(np.prod(pop_size, dtype=int))
        point_current = jnp.zeros((n_population, int(n_point)), dtype=jnp.asarray(components.mantissa).dtype)
        point_current = point_current.at[self.population_index, self.point_id].add(components.to_decimal(u.nA))
        return u.Quantity(point_current.reshape(pop_size + (int(n_point),)), u.nA)


class ClampView:
    """View an ordered selection of logical current-clamp instances."""

    __slots__ = ("_cell", "_logical_ids")

    def __init__(self, cell, logical_ids=None) -> None:
        self._cell = cell
        if logical_ids is None:
            logical_ids = cell._get_clamp_store().id
        self._logical_ids = np.asarray(logical_ids, dtype=np.int64).reshape(-1)

    @property
    def _store(self) -> _ClampStore:
        return self._cell._get_clamp_store()

    @property
    def id(self) -> np.ndarray:
        return np.array(self._logical_ids, copy=True)

    @property
    def instances(self) -> tuple[_ClampRecord, ...]:
        return self._store.records(self._logical_ids)

    def _column(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self._store, name))[self._store.row_indices(self._logical_ids)]

    @property
    def placement_id(self):
        """Return source point-placement identifiers."""
        return self._column("placement_id")

    @property
    def population_index(self):
        """Return owning flattened population indices."""
        return self._column("population_index")

    @property
    def point_id(self):
        """Return resolved electrical point identifiers."""
        return self._column("point_id")

    @property
    def cv_id(self):
        """Return owning control-volume identifiers."""
        return self._column("cv_id")

    @property
    def branch_id(self):
        """Return source morphology branch identifiers."""
        return self._column("branch_id")

    @property
    def branch_x(self):
        """Return normalized source branch coordinates."""
        return self._column("branch_x")

    @property
    def clamp_type(self):
        """Return current-clamp type names."""
        return self._column("clamp_type")

    @property
    def declaration(self):
        """Return immutable declarations in view order."""
        rows = self._store.row_indices(self._logical_ids)
        return tuple(self._store.declaration[index] for index in rows.tolist())

    def __len__(self) -> int:
        return int(self._logical_ids.size)

    def __getitem__(self, selector) -> "ClampView":
        if isinstance(selector, _CLAMP_TYPES):
            selected = [
                logical_id
                for logical_id, declaration in zip(self._logical_ids.tolist(), self.declaration)
                if declaration is selector
            ]
            return ClampView(self._cell, selected)
        if isinstance(selector, str) or (isinstance(selector, type) and issubclass(selector, _CLAMP_TYPES)):
            return self.by_type(selector)
        selected = self._logical_ids[selector]
        return ClampView(self._cell, np.asarray(selected, dtype=np.int64).reshape(-1))

    def by_id(self, logical_ids) -> "ClampView":
        """Select stable logical clamp identifiers within this view."""
        requested = np.asarray(logical_ids, dtype=np.int64).reshape(-1)
        unknown = np.setdiff1d(requested, self._logical_ids)
        if unknown.size:
            raise KeyError(f"Clamp ids are outside this view: {unknown.tolist()!r}.")
        return ClampView(self._cell, requested)

    def by_type(self, clamp_type) -> "ClampView":
        """Select clamps by declaration class or class name."""
        name = clamp_type if isinstance(clamp_type, str) else getattr(clamp_type, "__name__", None)
        if name not in {item.__name__ for item in _CLAMP_TYPES}:
            raise ValueError(f"Unknown current clamp type {name!r}.")
        return ClampView(self._cell, self._logical_ids[self.clamp_type == name])

    def for_population(self, population_indices) -> "ClampView":
        """Select clamps owned by the given population members."""
        selected = np.asarray(tuple(int(index) for index in population_indices), dtype=np.int64)
        return ClampView(self._cell, self._logical_ids[np.isin(self.population_index, selected)])

    def for_scope_pairs(self, pairs) -> "ClampView":
        """Select clamps whose owning population/CV pair is in ``pairs``."""
        selected = {(int(population), int(cv_id)) for population, cv_id in pairs}
        mask = np.asarray(
            [pair in selected for pair in zip(self.population_index.tolist(), self.cv_id.tolist())],
            dtype=bool,
        )
        return ClampView(self._cell, self._logical_ids[mask])

    def record(self, name: str, *, period=None, frequency=None, start=0.0 * u.ms):
        """Record the solver-consumed current of every selected clamp.

        Parameters
        ----------
        name : str
            Cell-local recording name.
        period, frequency : Quantity, optional
            Mutually exclusive regular sampling interval declarations.
        start : Quantity, optional
            Absolute recording schedule start.

        Returns
        -------
        RecordingSpec
            Frozen recording declaration owned by the root Cell.
        """
        from braincell.network.recording import RecordingSpec, _ClampCurrentObservable

        return self._cell._add_recording(
            RecordingSpec(
                name=name,
                scope=self._cell._root_scope(),
                observable=_ClampCurrentObservable(tuple(self._logical_ids.tolist()), "none"),
                period=period,
                frequency=frequency,
                start=start,
            )
        )
