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

"""Logical Channel and Ion views over population/CV scopes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._compute.ions import _runtime_ion_species_key
from braincell._misc import require_name as _require_name
from braincell.mech import Density, get_registry

__all__ = ["ChannelView", "IonView"]


@dataclass(frozen=True)
class _DensityRow:
    category: str
    mechanism_type: str
    name: str
    species: str | None
    population_index: int
    cv_id: int
    point_id: int
    mechanism: Density


class _DensityView:
    """Base view aligned to logical owner/population/CV rows."""

    category = ""
    __slots__ = ("_cell", "_rows")

    def __init__(self, cell, scope, rows=None) -> None:
        self._cell = cell
        self._rows = tuple(_density_rows(cell, scope, self.category) if rows is None else rows)

    @property
    def cell(self):
        return self._cell

    @property
    def rows(self) -> tuple[_DensityRow, ...]:
        return self._rows

    @property
    def name(self) -> np.ndarray:
        return np.asarray([row.name for row in self._rows], dtype=object)

    @property
    def mechanism_type(self) -> np.ndarray:
        return np.asarray([row.mechanism_type for row in self._rows], dtype=object)

    @property
    def population_index(self) -> np.ndarray:
        return np.asarray([row.population_index for row in self._rows], dtype=np.int64)

    @property
    def cv_id(self) -> np.ndarray:
        return np.asarray([row.cv_id for row in self._rows], dtype=np.int64)

    @property
    def point_id(self) -> np.ndarray:
        return np.asarray([row.point_id for row in self._rows], dtype=np.int64)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.name for row in self._rows))

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.mechanism_type for row in self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, selector):
        if not isinstance(selector, str):
            raise TypeError(
                f"{type(self).__name__} does not support numeric row indexing; "
                "select population/CV space first, then select a mechanism by name or type."
            )
        return self.by_name(selector)

    def by_name(self, name: str):
        _require_name(name, "mechanism name")
        return type(self)(self._cell, None, rows=(row for row in self._rows if row.name == name))

    def by_type(self, mechanism_type: str):
        _require_name(mechanism_type, "mechanism type")
        return type(self)(
            self._cell,
            None,
            rows=(row for row in self._rows if row.mechanism_type == mechanism_type),
        )

    def get(self, field: str):
        """Gather one parameter or initialized state in stable row order."""
        _require_name(field, "field")
        self._require_one_owner("read a field")
        values = [self._row_value(row, field) for row in self._rows]
        return _stack(values)

    def set(self, **fields):
        """Set shape-preserving fields for the selected logical owner rows."""
        self._require_one_owner("set fields")
        if not fields:
            return self
        for field, raw in fields.items():
            _require_name(field, "field")
            values = _broadcast_rows(raw, len(self._rows), name=field)
            for row, value in zip(self._rows, values):
                self._set_row_value(row, field, value)
        return self

    def _row_value(self, row: _DensityRow, field: str):
        override_key = (
            row.category,
            row.name,
            row.population_index,
            row.cv_id,
            field,
        )
        if not self._cell._initialized and override_key in self._cell._density_parameter_overrides:
            return self._cell._density_parameter_overrides[override_key]
        if not self._cell._initialized:
            if field not in row.mechanism.params:
                raise KeyError(f"{row.category.title()} {row.name!r} has no declared parameter {field!r}.")
            value = row.mechanism.params[field]
            return value(self._cell.cv_contexts[row.cv_id]) if callable(value) else value

        layout = _runtime_layout(self._cell, row)
        runtime = self._cell.runtime
        point_value = None
        if runtime.has_layout_value(layout.id, field):
            point_value = _take_point(runtime.get_state(layout.id, field), row.point_id, self._cell.n_point)
        else:
            node = runtime.get_runtime_node(layout.id)
            if not hasattr(node, field):
                raise KeyError(f"{row.category.title()} {row.name!r} has no parameter or state {field!r}.")
            point_value = getattr(node, field)
            if isinstance(point_value, brainstate.State):
                point_value = point_value.value
            point_value = _take_point(point_value, row.point_id, self._cell.n_point)
        return _take_population(point_value, row.population_index, self._cell._population_size)

    def _set_row_value(self, row: _DensityRow, field: str, value) -> None:
        if not self._cell._initialized:
            if field not in row.mechanism.params:
                raise KeyError(f"{row.category.title()} {row.name!r} has no declared parameter {field!r}.")
            self._cell._density_parameter_overrides[
                (row.category, row.name, row.population_index, row.cv_id, field)
            ] = value
            return

        layout = _runtime_layout(self._cell, row)
        runtime = self._cell.runtime
        if not runtime.has_layout_value(layout.id, field):
            raise KeyError(
                f"Runtime {row.category} {row.name!r} field {field!r} is not a shape-preserving parameter buffer."
            )
        buffer = runtime.get_state(layout.id, field)
        updated = _set_population_point(
            buffer,
            population_index=row.population_index,
            point_id=row.point_id,
            population_size=self._cell._population_size,
            point_size=self._cell.n_point,
            value=value,
        )
        runtime.set_state(layout.id, field, updated)

    def _require_one_owner(self, action: str) -> None:
        owners = tuple(dict.fromkeys((row.mechanism_type, row.name) for row in self._rows))
        if len(owners) > 1:
            raise TypeError(
                f"Cannot {action} across {len(owners)} logical {self.category} owners; select one name first."
            )

    def __repr__(self) -> str:
        grouped: dict[str, Counter] = {}
        for row in self._rows:
            grouped.setdefault(row.mechanism_type, Counter())[row.name] += 1
        details = ", ".join(
            f"{mechanism_type}({', '.join(f'{name}: {count}' for name, count in names.items())})"
            for mechanism_type, names in grouped.items()
        )
        return f"{type(self).__name__}(rows={len(self)}, {details or 'empty'})"


class ChannelView(_DensityView):
    """View Channel logical owners selected in population/CV space."""

    category = "channel"


class IonView(_DensityView):
    """View Ion logical owners selected in population/CV space."""

    category = "ion"

    @property
    def species(self) -> np.ndarray:
        return np.asarray([row.species for row in self._rows], dtype=object)

    def by_species(self, species: str):
        _require_name(species, "ion species")
        return type(self)(self._cell, None, rows=(row for row in self._rows if row.species == species))


def _density_rows(cell, scope, category: str) -> tuple[_DensityRow, ...]:
    by_cv: dict[int, dict[tuple[str, str], Density]] = {}
    species_by_type: dict[str, str | None] = {}
    for cv in cell.cvs:
        owners = by_cv.setdefault(int(cv.id), {})
        for mechanism in cv.density_mech:
            if not isinstance(mechanism, Density) or mechanism.category != category:
                continue
            owners.setdefault((mechanism.class_name, mechanism.instance_name), mechanism)
            if category == "ion" and mechanism.class_name not in species_by_type:
                runtime_cls = get_registry().get("ion", mechanism.class_name)
                species_by_type[mechanism.class_name] = _runtime_ion_species_key(runtime_cls)

    rows = []
    for population_index, cv_id in scope.pairs:
        point_id = int(cell.node_tree.cv_to_mid_node_id[int(cv_id)])
        for mechanism in by_cv.get(int(cv_id), {}).values():
            rows.append(
                _DensityRow(
                    category=category,
                    mechanism_type=mechanism.class_name,
                    name=mechanism.instance_name,
                    species=species_by_type.get(mechanism.class_name),
                    population_index=int(population_index),
                    cv_id=int(cv_id),
                    point_id=point_id,
                    mechanism=mechanism,
                )
            )
    return tuple(rows)


def _runtime_layout(cell, row: _DensityRow):
    matches = []
    for layout in cell.runtime.get_cv_layouts(row.cv_id):
        if layout.target != "density":
            continue
        mechanism = cell.runtime.get_layout_mechanism(layout.id)
        if (
            isinstance(mechanism, Density)
            and mechanism.category == row.category
            and mechanism.class_name == row.mechanism_type
            and mechanism.instance_name == row.name
        ):
            matches.append(layout)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one runtime layout for {row.category} {row.name!r} on CV {row.cv_id}, got {len(matches)!r}."
        )
    return matches[0]


def _take_point(value, point_id: int, point_size: int):
    shape = tuple(getattr(value, "shape", ()))
    if shape and shape[-1] == point_size:
        return value[..., point_id]
    return value


def _take_population(value, population_index: int, population_size: int):
    shape = tuple(getattr(value, "shape", ()))
    if shape and shape[0] == population_size:
        return value[population_index]
    return value


def _set_population_point(buffer, *, population_index, point_id, population_size, point_size, value):
    unit = buffer.unit if isinstance(buffer, u.Quantity) else None
    if unit is not None:
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Density parameter override requires a Quantity compatible with {unit}.")
        decimal = value.to_decimal(unit)
        mantissa = jnp.asarray(buffer.to_decimal(unit))
    else:
        if isinstance(value, u.Quantity):
            raise TypeError("Dimensionless density parameter override cannot be a Quantity.")
        decimal = value
        mantissa = jnp.asarray(buffer)
    if mantissa.shape[-1] != point_size:
        raise ValueError("Density parameter buffer does not expose the point axis.")
    if mantissa.ndim >= 2 and mantissa.shape[0] == population_size:
        mantissa = mantissa.at[population_index, point_id].set(decimal)
    else:
        mantissa = mantissa.at[point_id].set(decimal)
    return u.Quantity(mantissa, unit) if unit is not None else mantissa


def _broadcast_rows(value, count: int, *, name: str) -> tuple:
    if count == 0:
        return ()
    shape = tuple(getattr(value, "shape", ()))
    if shape == ():
        return (value,) * count
    if shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},), got {shape!r}.")
    return tuple(value[index] for index in range(count))


def _stack(values):
    if not values:
        return np.asarray([], dtype=float)
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(u.math.stack([value.to_decimal(unit) for value in values]), unit)
    return u.math.stack(values)
