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

"""Build static synapse metadata shared by reduction runtimes and calibration."""

import hashlib

import numpy as np

from braincell.mech import NoEventInput, get_registry

from braincell.reduction.core import ReductionContext, ReductionInputGroupSchema, ReductionSynapse


def _build_reduction_context(cell) -> ReductionContext:
    """Build static synapse metadata and bind its logical input rows."""
    store = cell._get_synapse_store()
    synapses = synapse_records(cell)
    population_local_index = np.asarray([item.synapse_index for item in synapses], dtype=np.int64)
    group_schemas = []
    for layout_id, raw_type in enumerate(dict.fromkeys(store.synapse_type.tolist())):
        synapse_type = str(raw_type)
        logical_ids = store.id[store.synapse_type == synapse_type]
        rows = store.row_indices(logical_ids)
        event_input = get_registry().get("synapse", synapse_type).event_input
        group_schemas.append(
            ReductionInputGroupSchema(
                layout_id=layout_id,
                synapse_type=synapse_type,
                event_input=event_input,
                synapse_id=logical_ids,
                synapse_index=population_local_index[rows],
                population_index=store.population_index[rows],
            )
        )
        if not isinstance(event_input, NoEventInput):
            store.bind_runtime(synapse_type, layout_id, logical_ids)

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
    return ReductionContext.with_cell(
        cell,
        synapses=synapses,
        input_groups=tuple(group_schemas),
        fingerprint=fingerprint,
    )


def synapse_records(cell):
    """Read logical synapses in declaration order with member-local channel indices."""
    store = cell._get_synapse_store()
    counters = {}
    synapses = []
    for row in range(len(store.id)):
        population = int(store.population_index[row])
        synapse_index = counters.get(population, 0)
        counters[population] = synapse_index + 1
        synapse_type = str(store.synapse_type[row])
        local_index = int(store._type_local_by_id[int(store.id[row])])
        parameters = {name: value[local_index] for name, value in store.parameter_columns[synapse_type].items()}
        synapses.append(
            ReductionSynapse(
                id=int(store.id[row]),
                synapse_index=synapse_index,
                population_index=population,
                placement_id=int(store.placement_id[row]),
                point_id=int(store.point_id[row]),
                cv_id=int(store.cv_id[row]),
                branch_id=int(store.branch_id[row]),
                branch_x=float(store.branch_x[row]),
                name=str(store.name[row]),
                synapse_type=synapse_type,
                parameters=parameters,
            )
        )
    return tuple(synapses)
