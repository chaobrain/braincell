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

"""Lightweight packed synapse-input runtime for reduced Cells."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import brainstate
import brainunit as u
import numpy as np

from braincell.mech import NoEventInput, ScalarEventInput, TriggerEventInput, get_registry
from braincell.reduction.core import (
    ReductionContext,
    ReductionInputGroup,
    ReductionInputGroupSchema,
    ReductionInputs,
    ReductionSynapse,
)

__all__ = ["ReductionInputLayout", "ReductionInputRuntime", "build_reduction_input_runtime"]


@dataclass(frozen=True)
class ReductionInputLayout:
    """Describe one packed event-buffer layout needed by connection lowering."""

    id: int
    kind: str
    n_active: int
    placement_index: np.ndarray
    synapse_index: np.ndarray
    schema: ReductionInputGroupSchema


@dataclass
class ReductionInputRuntime:
    """Own only the event buffers and schema required by a reduced Cell."""

    layouts: tuple[ReductionInputLayout, ...]
    event_buffers: dict[int, brainstate.State]
    context: ReductionContext

    def get_event_buffer(self, layout_id: int):
        """Return the current payload for one input layout."""
        return self.event_buffers[int(layout_id)].value

    def clear_event_buffer(self, layout_id: int) -> None:
        """Set one input buffer to zero without changing its unit or dtype."""
        state = self.event_buffers[int(layout_id)]
        state.value = u.math.zeros_like(state.value)

    def clear_event_buffers(self) -> None:
        """Clear every packed input buffer."""
        for layout_id in self.event_buffers:
            self.clear_event_buffer(layout_id)

    def take_inputs(self) -> ReductionInputs:
        """Snapshot all current payloads and clear their backing buffers."""
        groups = tuple(
            ReductionInputGroup(layout.schema, self.get_event_buffer(layout.id))
            for layout in self.layouts
            if layout.id in self.event_buffers
        )
        self.clear_event_buffers()
        return ReductionInputs(groups)


def build_reduction_input_runtime(cell) -> ReductionInputRuntime:
    """Build a packed input-only runtime from the Cell's current synapses."""
    store = cell._get_synapse_store()
    population_local_index = _population_local_indices(store.population_index)
    layouts = []
    event_buffers = {}
    group_schemas = []

    for layout_id, raw_type in enumerate(dict.fromkeys(store.synapse_type.tolist())):
        synapse_type = str(raw_type)
        logical_ids = store.id[store.synapse_type == synapse_type]
        rows = store.row_indices(logical_ids)
        runtime_cls = get_registry().get("synapse", synapse_type)
        event_input = runtime_cls.event_input
        schema = ReductionInputGroupSchema(
            layout_id=layout_id,
            synapse_type=synapse_type,
            event_input=event_input,
            synapse_id=logical_ids,
            synapse_index=population_local_index[rows],
            population_index=store.population_index[rows],
        )
        layout = ReductionInputLayout(
            id=layout_id,
            kind=f"synapse:{synapse_type}",
            n_active=schema.size,
            placement_index=np.asarray(store.placement_id[rows], dtype=np.int64),
            synapse_index=np.asarray(logical_ids, dtype=np.int64),
            schema=schema,
        )
        layouts.append(layout)
        group_schemas.append(schema)
        if isinstance(event_input, ScalarEventInput):
            zero = u.Quantity(np.zeros((schema.size,), dtype=float), event_input.unit)
        elif isinstance(event_input, TriggerEventInput):
            zero = np.zeros((schema.size,), dtype=np.int32)
        elif isinstance(event_input, NoEventInput):
            continue
        else:
            raise TypeError(f"Unsupported event input {type(event_input).__name__!r} for {synapse_type!r}.")
        event_buffers[layout_id] = brainstate.ShortTermState(zero)
        store.bind_runtime(synapse_type, layout_id, logical_ids)

    synapses = tuple(_synapse_record(store, row, int(population_local_index[row])) for row in range(len(store.id)))
    signature = tuple(
        (
            item.population_index,
            item.synapse_index,
            item.point_id,
            item.name,
            item.synapse_type,
            tuple((name, repr(value)) for name, value in item.parameters.items()),
        )
        for item in synapses
    )
    fingerprint = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
    context = ReductionContext.with_cell(
        cell,
        synapses=synapses,
        input_groups=tuple(group_schemas),
        fingerprint=fingerprint,
    )
    return ReductionInputRuntime(tuple(layouts), event_buffers, context)


def _population_local_indices(population_index: np.ndarray) -> np.ndarray:
    counters = {}
    result = np.empty(len(population_index), dtype=np.int64)
    for row, raw_owner in enumerate(np.asarray(population_index, dtype=np.int64).tolist()):
        owner = int(raw_owner)
        result[row] = counters.get(owner, 0)
        counters[owner] = int(result[row]) + 1
    return result


def _synapse_record(store, row: int, synapse_index: int) -> ReductionSynapse:
    synapse_type = str(store.synapse_type[row])
    parameters = {
        name: _take_one(value, store._type_local_by_id[int(store.id[row])])
        for name, value in store.parameter_columns[synapse_type].items()
    }
    return ReductionSynapse(
        id=int(store.id[row]),
        synapse_index=synapse_index,
        population_index=int(store.population_index[row]),
        placement_id=int(store.placement_id[row]),
        point_id=int(store.point_id[row]),
        cv_id=int(store.cv_id[row]),
        branch_id=int(store.branch_id[row]),
        branch_x=float(store.branch_x[row]),
        name=str(store.name[row]),
        synapse_type=synapse_type,
        parameters=parameters,
    )


def _take_one(value, index: int):
    return value[int(index)]
