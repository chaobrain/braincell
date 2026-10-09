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

"""Bind DIF connections and coordinate finalized voltage recordings."""

from __future__ import annotations

from dataclasses import replace

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.mech import TriggerEventInput
from braincell.network.event import EventSequence, NetStim, round_half_up_steps_host
from braincell.network.recording import SampleBlock
from braincell.reduction.core import ReductionInputGroupSchema, ReductionRecording
from braincell.reduction.dif.tables import ResponseSlot
from braincell.reduction.dif.tables_utils import synapse_identities
from braincell.reduction.runtime import ReductionInputLayout, ReductionInputRuntime


class DIFInputRuntime(ReductionInputRuntime):
    """Keep each physical synapse/weight pair separate until CUDA delivery."""

    def __init__(self, context, table):
        # 1. Match each declared physical synapse/weight to its calibrated slot.
        connections = context.cell.connections._call_views()
        active = set()
        for connection in connections:
            ids = connection.synapse_id
            if connection.weight is not None:
                ids = ids[np.asarray(u.get_mantissa(connection.weight)) != 0]
            active.update(map(int, ids))
        identities = synapse_identities(s for s in context.synapses if s.id in active)
        by_id = {synapse.id: synapse for synapse in context.synapses}
        by_identity = {(s.population_index, identities[s.id]): s for s in context.synapses if s.id in identities}
        width = context.population_size * len(table.slots)
        slots = {slot: i for i, slot in enumerate(table.slots)}
        self._route_rows = {}
        for connection in connections:
            weights = connection.weight
            for i, (connection_id, synapse_id) in enumerate(zip(connection.id, connection.synapse_id)):
                synapse_id = int(synapse_id)
                weight = None if weights is None else weights[i]
                if weight is not None and np.asarray(u.get_mantissa(weight)) == 0:
                    self._route_rows[int(connection_id)] = (0, 0)
                    continue
                slot = ResponseSlot.from_input(identities[synapse_id], weight)
                index = by_id[synapse_id].population_index * len(table.slots) + slots[slot]
                self._route_rows[int(connection_id)] = (index, 1)
        # 2. Expose slot counts through the common reduction input schema.
        logical_ids, placements = [], []
        for population in range(context.population_size):
            for slot in table.slots:
                synapse = by_identity.get((population, (slot.signature, slot.instance)))
                logical_ids.append(-1 if synapse is None else synapse.id)
                placements.append(-1 if synapse is None else synapse.placement_id)
        schema = ReductionInputGroupSchema(
            layout_id=0,
            synapse_type="calibrated_response",
            event_input=TriggerEventInput(),
            synapse_id=np.asarray(logical_ids, dtype=np.int64),
            synapse_index=np.tile(np.arange(len(table.slots)), context.population_size),
            population_index=np.repeat(np.arange(context.population_size), len(table.slots)),
        )
        layout = ReductionInputLayout(
            id=0,
            kind="reduction:response",
            n_active=width,
            placement_index=np.asarray(placements),
            synapse_index=None,
            schema=schema,
        )
        super().__init__(
            (layout,),
            {0: brainstate.ShortTermState(jnp.zeros(width, dtype=jnp.int64))},
            replace(context, input_groups=(schema,)),
        )

        # 3. Prepare immutable schedules; Network setup binds live sources later.
        self.schedules = pack_scheduled_events(self, table.dt_ms)
        self._sources = ()

    def connection_route(self, connection):
        rows = [self._route_rows[int(index)] for index in connection.id]
        indices = np.asarray([row[0] for row in rows], dtype=np.int32)
        weights = np.asarray([row[1] for row in rows], dtype=np.int64)
        return 0, indices, weights

    def prepare_delivery(self, blocks):
        """Own incoming live routes while preserving public event-source ports."""
        self._sources = configure_live_events(self.runtime, blocks)
        return ()

    def scheduled_inputs(self, layout, *, t, template):
        if self.schedules is not None:
            return u.math.zeros_like(template)
        return super().scheduled_inputs(layout, t=t, template=template)

    def enqueue_events(self):
        if self._sources:
            counts = jnp.concatenate(
                [jnp.asarray(source.current_event_count(source.ids), dtype=jnp.int64) for source in self._sources]
            )
            self.runtime.enqueue(counts)

    def reset_state(self):
        super().reset_state()
        self.runtime.reset_delivery()


class VoltageRecorder:
    """Share one finalized CUDA buffer across all declared voltage views."""

    def __init__(self, runtime, dt_ms):
        self.runtime = runtime
        self.dt_ms = dt_ms
        self.neurons = set()
        self.reset()

    def reset(self):
        self._segment = None
        self._result = None

    def prepare(self, schema):
        if not schema.rows or schema.rows[0].output_name != "voltage":
            return None
        neurons = tuple(row.population_index for row in schema.rows)
        self.neurons.update(neurons)
        period = int(round(float(schema.period.to_decimal(u.ms)) / self.dt_ms))
        start = int(round(float(schema.schedule_start.to_decimal(u.ms)) / self.dt_ms))

        def finish():
            if self._result is None:
                self._result = self.runtime.finish_recording()
            first, voltage = self._result
            stop = first + voltage.shape[0]
            indices = np.arange(first, stop, dtype=np.int64)
            mask = (indices >= start) & ((indices - start) % period == 0)
            columns = {neuron: i for i, neuron in enumerate(sorted(self.neurons))}
            values = voltage[:, [columns[neuron] for neuron in neurons]][mask] * u.mV
            return SampleBlock(
                values=values,
                schema=schema,
                segment_start=first * self.dt_ms * u.ms,
                segment_stop=stop * self.dt_ms * u.ms,
                first_time=None if not np.any(mask) else indices[mask][0] * self.dt_ms * u.ms,
            )

        return ReductionRecording(self.begin, finish)

    def begin(self, start, count):
        first = int(round(float(start.to_decimal(u.ms)) / self.dt_ms))
        segment = (first, count)
        if self._segment != segment:
            self.runtime.begin_recording(first, count, sorted(self.neurons))
            self._segment = segment
            self._result = None


def pack_scheduled_events(inputs, dt_ms):
    """Pack immutable public schedules, retaining duplicate events and contacts."""
    connections = inputs.context.cell.connections._call_views(scheduled=True)
    if any(type(connection.source) not in (EventSequence, NetStim) for connection in connections):
        return None
    sources = {}
    steps, groups, targets, target_ptr = [], [], [], [0]
    for connection in connections:
        source = connection.source
        key = id(source)
        if key not in sources:
            events = source.events
            order = np.argsort(events.source_index, kind="stable")
            times = np.asarray(events.time.to_decimal(u.ms))[order]
            ptr = np.r_[0, np.cumsum(np.bincount(events.source_index, minlength=source.size))]
            sources[key] = times, ptr
        times, ptr = sources[key]
        _, slot, weight = inputs.connection_route(connection)
        active = weight != 0
        pre = connection.source_index[active]
        delay = np.asarray(connection.delay.to_decimal(u.ms))[active]
        slot = slot[active]
        order = np.lexsort((delay, pre))
        pre, delay, slot = pre[order], delay[order], slot[order]
        first = np.r_[0, np.flatnonzero((pre[1:] != pre[:-1]) | (delay[1:] != delay[:-1])) + 1, len(pre)]
        for lo, hi in zip(first[:-1], first[1:]):
            if lo == hi:
                continue
            begin, end = ptr[pre[lo] : pre[lo] + 2]
            ticks = round_half_up_steps_host((times[begin:end] + delay[lo]) / dt_ms).astype(np.int64)
            if not ticks.size:
                continue
            steps.append(ticks)
            groups.append(np.full(ticks.size, len(targets), dtype=np.int64))
            targets.append(slot[lo:hi].astype(np.int64))
            target_ptr.append(target_ptr[-1] + hi - lo)
    events = np.concatenate(steps) if steps else np.empty(0, dtype=np.int64)
    order = np.argsort(events, kind="stable")
    ticks, first, counts = np.unique(events[order], return_index=True, return_counts=True)
    group_dtype = np.uint32 if len(targets) <= np.iinfo(np.uint32).max else np.uint64
    group = np.concatenate(groups)[order].astype(group_dtype) if groups else np.empty(0, dtype=group_dtype)
    return (
        ticks,
        np.r_[first, events.size].astype(np.int64),
        group,
        np.concatenate(targets) if targets else np.empty(0, dtype=np.int64),
        np.asarray(target_ptr, dtype=np.int64),
    ), min(128, (int(counts.max(initial=0)) + 255) // 256)


def configure_live_events(runtime, blocks):
    """Bind incoming live ports, including detailed and other reduced Cells."""
    sources, offsets, routes = [], {}, []
    source_count = 0
    for block in blocks:
        source = block.event_source
        key = id(source)
        if key not in offsets:
            offsets[key] = source_count
            sources.append(source)
            source_count += source.size
        active = np.asarray(block.weight) != 0
        routes.append(
            np.column_stack(
                (
                    block.pre_index[active].astype(np.int64) + offsets[key],
                    block.synapse_index[active],
                    np.maximum(1, block.delay_steps[active]),
                )
            )
        )
    rows = np.concatenate(routes).astype(np.int64) if routes else np.empty((0, 3), dtype=np.int64)
    runtime.configure_delivery(rows, source_count)
    return tuple(sources) if rows.size else ()
