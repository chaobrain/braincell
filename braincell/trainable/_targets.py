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

"""Adapters from logical point targets to the common trainable registry."""

from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell._parameter_schema import ParameterSpec, RuntimeParameterState


@dataclass(frozen=True)
class _PointRow:
    category: str
    name: str
    mechanism_type: str
    population_index: int
    cv_id: int
    point_id: int
    logical_id: int


class _PointTarget:
    def __init__(self, rows, schema, read, write, validate=None):
        self.rows = tuple(rows)
        self._schema = schema
        self._read = read
        self._write = write
        self._validate = validate

    def _trainable_schema(self):
        return self._schema

    def _row_value(self, row, field):
        return self._read(field)[self.rows.index(row)]

    def _prepare_write(self, field, values):
        return self._write(field, values)

    def _validate_bindings(self, bindings):
        if self._validate is not None:
            self._validate(bindings)


def register_synapse(view, fields):
    kind = view._require_homogeneous_type()
    cell = view._cell
    records = view._store.records(view.id)
    rows = [_PointRow("synapse", kind, kind, r.population_index, r.cv_id, r.point_id, r.id) for r in records]

    def write(field, values):
        store = cell._get_synapse_store()
        state = cell._runtime.state_buffers[(store.layout_id(kind), field)]
        return [(state, store.runtime_rows(view.id), values)]

    def validate(bindings):
        from braincell.mech import get_registry

        columns = {field: view.get(field) for field in view.parameter_info()}
        positions = {r.id: i for i, r in enumerate(records)}
        for binding in bindings:
            if binding._rows[0].category != "synapse" or binding.target_owner != kind:
                continue
            values = binding._evaluate()
            for index, row in enumerate(binding._rows):
                if row.logical_id in positions:
                    column = columns[binding.target_field]
                    column = column.at[positions[row.logical_id]].set(values[index])
                    columns[binding.target_field] = column
        get_registry().get("synapse", kind).validate_parameter_values(columns)

    target = _PointTarget(rows, view.parameter_info(), view.get, write, validate)
    cell.trainables.register(target, fields)


def register_connection(view, fields):
    if "delay" in fields:
        raise NotImplementedError("Connection delay is static and cannot be trained.")
    if set(fields).difference({"weight"}):
        raise KeyError("Only Connection weight is trainable; bind threshold on its event detector.")
    view._require_homogeneous_synapse_type("train weight")
    current = view.weight
    if current is None:
        raise TypeError("Trigger-only Connection has no physical weight to train.")
    ids = view.id
    synapses = view.synapse
    rows = [
        _PointRow("connection", "weight", "Connection", int(pop), int(cv), int(point), int(i))
        for i, pop, cv, point in zip(ids, synapses.population_index, synapses.cv_id, synapses.point_id)
    ]

    def write(field, values):
        commits = []
        store = view._store
        call_ids = store.connect_id[store.rows(ids)]
        for call_id in np.unique(call_ids):
            selected = np.flatnonzero(call_ids == call_id)
            call = store.call(int(call_id))
            state = vars(call)["weight"]
            commits.append((state, ids[selected] - call.row_ids[0], values[selected]))
        return commits

    target = _PointTarget(rows, {"weight": ParameterSpec(current[0])}, lambda field: view.weight, write)
    view.cell.trainables.register(target, fields)


def register_detector(view, fields):
    from braincell.network.event import VoltageCrossingSource, _CellSpikeSource

    source = view.owner
    if not isinstance(source, (VoltageCrossingSource, _CellSpikeSource)):
        raise TypeError("Only voltage event detectors have a trainable threshold.")
    if set(fields).difference({"threshold"}):
        raise KeyError("Only the detector threshold is trainable.")
    cell = source.execution_owner
    if cell._initialized:
        raise RuntimeError("trainable() must be called before Cell.init_state().")
    ids = view.source_id
    own_threshold = isinstance(source, VoltageCrossingSource) and not source._uses_cell_threshold
    if isinstance(source, VoltageCrossingSource):
        populations = source._population_indices[ids]
        cvs = source._cv_ids[ids]
    else:
        populations = ids
        cvs = np.full(len(ids), source.cv_id, dtype=np.int64)
    if own_threshold:
        owner = f"detector.{source._training_id}"
        logical = ids
        state = source._threshold
        indices = ids
        read = lambda field: state.value[ids]
    else:
        owner = "V_th"
        logical = populations * cell.n_compartment + cvs
        if cell._V_th_parameter is None:
            cell._V_th_parameter = RuntimeParameterState(cell._materialize_population_parameter("V_th"))
        state = cell._V_th_parameter
        indices = (populations, cvs) if len(cell.pop_size) else (cvs,)
        read = lambda field: state.value[indices]
    if len(set(logical.tolist())) != len(logical):
        raise ValueError("A trainable detector selection cannot repeat the same threshold target.")
    rows = [
        _PointRow("threshold", owner, "VoltageThreshold", int(pop), int(cv), int(cv), int(i))
        for pop, cv, i in zip(populations, cvs, logical)
    ]
    target = _PointTarget(
        rows, {"threshold": ParameterSpec(0.0 * u.mV)}, read, lambda field, values: [(state, indices, values)]
    )
    cell.trainables.register(target, fields)


def require_unbound(cell, category, owner, ids, field):
    for binding in cell.trainables.bindings():
        if binding.target_field != field:
            continue
        for row in binding._rows:
            if row.category == category and row.owner == owner and row.logical_id in ids:
                raise RuntimeError(f"{category} {field} is trainable; update its parameter root instead.")
