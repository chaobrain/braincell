# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fixed-grid runtime geometry selection and trainable bindings."""

from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell._parameter_schema import ParameterSpec
from braincell._multi_compartment.lifecycle import DiscreteView


@dataclass(frozen=True)
class GeometryRow:
    category: str
    name: str
    mechanism_type: str
    population_index: int
    cv_id: int
    point_id: int
    logical_id: int


class GeometryView(DiscreteView):
    """Runtime geometry values on the fixed, initialized CV grid."""

    __slots__ = ("_cell", "_rows")

    _SCHEMA = {
        "radius_scale": ParameterSpec(1.0),
        "length": ParameterSpec(1.0 * u.um),
        "Ra": ParameterSpec(100.0 * u.ohm * u.cm),
        "cm": ParameterSpec(1.0 * u.uF / u.cm**2),
    }
    _READONLY = {
        "radius_mid", "diam_mid", "diam_arc_mean", "area",
        "resistance_prox", "resistance_dist",
    }

    def __init__(self, cell, scope=None, rows=None):
        self._cell = cell
        if rows is None:
            pairs = cell._root_scope().pairs if scope is None else scope.pairs
            rows = tuple(
                GeometryRow("geometry", "geometry", "geometry", int(pop), int(cv), int(cell.node_tree.cv_to_mid_node_id[cv]), int(pop * cell.n_cv + cv))
                for pop, cv in pairs
            )
        self._rows = tuple(rows)
        self._bind_view(cell)

    @property
    def cell(self):
        return self._cell

    @property
    def rows(self):
        return self._rows

    def __len__(self):
        self._check_view()
        return len(self._rows)

    def __getitem__(self, selector):
        self._check_view()
        if isinstance(selector, str):
            if selector not in self._SCHEMA and selector not in self._READONLY:
                raise KeyError(selector)
            return GeometryFieldView(self, selector)
        raise TypeError("GeometryView selects fields by name; select population/CV space before geometry fields.")

    def __getattr__(self, field):
        if field.startswith("_"):
            raise AttributeError(field)
        if field in self._SCHEMA or field in self._READONLY:
            return GeometryFieldView(self, field)
        raise AttributeError(field)

    def _trainable_schema(self):
        return self._SCHEMA

    def parameter_info(self):
        return dict(self._SCHEMA)

    def get(self, field):
        self._check_view()
        if field not in self._SCHEMA and field not in self._READONLY:
            raise KeyError(field)
        runtime = self._cell.runtime
        if field in self._READONLY:
            if field in {"area", "resistance_prox", "resistance_dist"}:
                cable_field = {
                    "area": "area",
                    "resistance_prox": "resistance_prox",
                    "resistance_dist": "resistance_dist",
                }[field]
                values = getattr(runtime.current_cable(), cable_field)
                values = u.math.broadcast_to(values, runtime.pop_size + (runtime.n_cv,))
            else:
                reference = runtime.reference_diam_arc_mean if field == "diam_arc_mean" else runtime.reference_radius_mid
                values = reference * runtime.geometry["radius_scale"].value
                if field == "diam_mid":
                    values = 2 * values
                values = u.math.broadcast_to(values, runtime.pop_size + (runtime.n_cv,))
        else:
            values = runtime.geometry[field].value
        return u.math.asarray(values).reshape((-1, runtime.n_cv))[
            np.asarray([r.population_index for r in self._rows]),
            np.asarray([r.cv_id for r in self._rows]),
        ]

    def set(self, **fields):
        self._cell._raise_if_not_initialized("GeometryView.set(); use morphology declarations before init_state()")
        self._check_view()
        pending = {}
        for field, value in fields.items():
            if field not in self._SCHEMA:
                raise KeyError(field)
            values = _broadcast(value, len(self._rows))
            state = self._cell.runtime.geometry[field]
            full = state.value
            for row, item in zip(self._rows, values):
                self._SCHEMA[field].validate(item, field)
                full = full.at[row.population_index, row.cv_id].set(item)
            pending[field] = full
        previous = {field: self._cell.runtime.geometry[field].value for field in pending}
        try:
            for field, value in pending.items():
                self._cell.runtime.geometry[field].value = value
            self._cell._refresh_geometry_runtime()
        except Exception:
            for field, value in previous.items():
                self._cell.runtime.geometry[field].value = value
            raise
        return self

    def trainable(self, **fields):
        self._cell.trainables.register(self, fields)
        return self

    def _row_value(self, row, field):
        return self._cell.runtime.geometry[field].value[row.population_index, row.cv_id]

    def _prepare_write(self, field, values):
        state = self._cell.runtime.geometry[field]
        return [
            (state, (row.population_index, row.cv_id), value)
            for row, value in zip(self._rows, values)
        ]


class GeometryFieldView:
    """Field adapter providing ``trainable`` and ``set`` for one geometry field."""

    def __init__(self, owner, field):
        self.owner, self.field = owner, field

    def get(self):
        return self.owner.get(self.field)

    def set(self, value):
        return self.owner.set(**{self.field: value})

    def trainable(self, source):
        return self.owner.trainable(**{self.field: source})


def _broadcast(value, size):
    if isinstance(value, u.Quantity):
        if getattr(value, "shape", ()) == ():
            return tuple(value for _ in range(size))
        values = np.asarray(value.mantissa).reshape(-1)
        if len(values) != size:
            raise ValueError(f"Geometry value must be scalar or have {size} entries, got {len(values)}.")
        return tuple(u.Quantity(values[i], value.unit) for i in range(size))
    else:
        if np.ndim(value) == 0:
            return tuple(value for _ in range(size))
        values = np.asarray(value).reshape(-1)
    if len(values) != size:
        raise ValueError(f"Geometry value must be scalar or have {size} entries, got {len(values)}.")
    return tuple(value[i] for i in range(size))
