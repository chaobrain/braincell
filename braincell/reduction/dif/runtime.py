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

"""Advance DIF populations through compiled CUDA dynamics."""

from __future__ import annotations

from weakref import WeakValueDictionary

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dif.runtime_utils import (
    _arena,
    _DeviceState,
    _shape,
    _StageStreams,
    load_cuda_module,
    pack_device_table,
)

_ARENAS = WeakValueDictionary()
_MEMORY = ("core", "records", "monitors", "delivery")


class CudaPopulationRuntime:
    """Distribute active neuron work within the ordinary compiled Network loop."""

    def __init__(self, table, shape, *, block_threads, gpu_heap_bytes, high_order=True, schedules=None):
        # 1. Compile the selected kernels and resolve their launch resources.
        data = table
        if not 0 < data.rest_support.size < 32:
            raise ValueError(f"DIF CUDA requires 1 to 31 synapse locations; got {data.rest_support.size}.")
        self.count = int(np.prod(shape))
        self.data = data
        self.threads = block_threads
        self.abort_voltage = np.float64(table.abort_voltage_mV)
        # One merged event per slot and timestep bounds every history index.
        index_bits = 16 if max(data.response_steps, data.rest_grid.size, data.post_state_grid.size) <= 32767 else 32
        _, library, self.call = load_cuda_module(block_threads, index_bits, bool(high_order))
        self.ma_shared_bytes = library.reduction_ma_rebase_configure(int(data.time_steps.shape[1]) ** 2)
        self.panel_shared_bytes = library.reduction_panel_configure(int(data.time_steps.shape[1]))
        if not self.ma_shared_bytes:
            raise RuntimeError("DIF rebase workspace exceeds this GPU's shared-memory limit.")
        if not self.panel_shared_bytes:
            raise RuntimeError("DIF ordinary-input workspace exceeds this GPU's shared-memory limit.")
        self.rebase_threads = library.reduction_configure(True, self.ma_shared_bytes)
        self.ordinary_threads = library.reduction_configure(False, self.panel_shared_bytes)
        self.panel_shared_bytes = self.panel_shared_bytes // 4 * (self.ordinary_threads // 32)
        # 2. Upload the immutable table and bind stream/arena ownership.
        self.table = pack_device_table(table)
        self.stage_streams = _StageStreams(library, self.table.value[0].device)
        if library.reduction_stage_metadata(
            self.stage_streams.handle,
            self.table.host_header.ctypes.data,
            self.table.host_header.size,
            self.table.host_scalars.ctypes.data,
        ):
            raise RuntimeError("CUDA table header ABI mismatch")
        self.arena = _arena(self.table.value[0].device, gpu_heap_bytes, library, arenas=_ARENAS)
        warps = self.ordinary_threads // 32
        rebase_warps = self.rebase_threads // 32
        rebase_tasks = 2 * self.count * (data.rest_support.size**2 + data.rest_support.size)
        self.blocks = min(
            (rebase_tasks + rebase_warps - 1) // rebase_warps,
            library.reduction_resident_blocks(True, self.rebase_threads, self.ma_shared_bytes),
        )
        ordinary_tasks = self.count * data.rest_support.size
        self.ordinary_blocks = min(
            (ordinary_tasks + warps - 1) // warps,
            library.reduction_resident_blocks(False, self.ordinary_threads, self.panel_shared_bytes),
        )
        # 3. Initialize recording, scheduled inputs and live-delivery state.
        job_words = library.reduction_jobs_size(self.count)
        self.clock = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int64))
        self.record_start = 0
        self._tail = None
        self.record_stop = 0
        self.recording_inputs = brainstate.LongTermState(
            (
                jnp.full((self.count,), -1, dtype=jnp.int64),
                jnp.zeros((2,), dtype=jnp.int64),
            )
        )
        self.width = self.count * self.data.rest_support.size
        empty = np.empty(0, dtype=np.int64)
        if schedules is None:
            schedules = (
                (empty, np.zeros(1, dtype=np.int64), empty.astype(np.uint32), empty, np.zeros(1, dtype=np.int64)),
                0,
            )
        self.schedules = brainstate.LongTermState(tuple(jax.device_put(value) for value in schedules[0]))
        self.schedule_blocks = schedules[1]
        self.routes = brainstate.LongTermState(
            tuple(
                jax.device_put(value)
                for value in (np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), empty, empty, empty)
            )
        )
        self.source_count = 0
        self.delays = 1
        self.pending_offset = 0
        self.delivery_bytes = 0
        # 4. Lay out and initialize the mutable neuron/history workspace.
        core_sizes = dict(
            neurons=self.count * library.reduction_neuron_size(),
            history=self.width * library.reduction_history_size(),
            inbox=self.width * 8,
            jobs=job_words * 8,
        )
        self.core_bytes = 0
        self.offsets = {}
        for name, size in core_sizes.items():
            offset = (self.core_bytes + 127) // 128 * 128
            self.offsets[name] = np.int64(offset)
            self.core_bytes = offset + size
        self._state = _DeviceState(self._new_memory(jnp.empty(0, dtype=jnp.uint8)), self.call, self.arena)
        self._state.release_layout = dict(
            count=np.int64(self.count),
            history_offset=self.offsets["history"],
            history_count=np.int64(self.width),
        )
        self._state.stage_streams = self.stage_streams

    def _new_memory(self, delivery):
        core = self.call("reduction_initialize", jax.ShapeDtypeStruct((self.core_bytes,), np.uint8))(
            count=np.int64(self.count),
            initial=np.float64(self.data.initial_voltage),
            rest=np.float64(self.data.rest_voltage),
        )
        return dict(
            core=core,
            records=jnp.empty((0, 0), dtype=jnp.float64),
            monitors=jnp.empty((0, 0), dtype=jnp.float64),
            delivery=delivery,
        )

    def configure_delivery(self, rows, source_count):
        """Pack source/delay groups without changing neuron or recording state."""
        order = np.lexsort((rows[:, 1], rows[:, 2], rows[:, 0]))
        rows = rows[order]
        first = (
            np.r_[0, np.flatnonzero((rows[1:, 0] != rows[:-1, 0]) | (rows[1:, 2] != rows[:-1, 2])) + 1]
            if rows.size
            else np.empty(0, dtype=np.int64)
        )
        sources = rows[first, 0]
        self.source_count = source_count
        self.delays = int(rows[:, 2].max(initial=0)) + 1
        self.routes.value = tuple(
            jax.device_put(value)
            for value in (
                np.r_[0, np.cumsum(np.bincount(sources, minlength=source_count))].astype(np.int64),
                np.r_[first, len(rows)].astype(np.int64),
                rows[:, 1].copy(),
                rows[first, 2].copy(),
                sources,
            )
        )
        self.pending_offset = (self.delays * source_count * 8 + 127) // 128 * 128
        self.delivery_bytes = self.pending_offset + self.delays * ((len(first) + 31) // 32) * 4
        self.reset_delivery()

    def reset_delivery(self):
        """Clear private arrivals when the input adapter resets its queues."""
        self._state.value = dict(self._state.value, delivery=jnp.zeros(self.delivery_bytes, dtype=jnp.uint8))

    def enqueue(self, counts):
        """Queue current live events after every population has advanced."""
        memory = self._state.value
        delivery = self.call("reduction_deliver", _shape(memory["delivery"]), {4: 0})(
            self.routes.value[0],
            self.routes.value[3],
            counts,
            self.clock.value,
            memory["delivery"],
            sources=np.int64(self.source_count),
            delays=np.int64(self.delays),
            threads=np.int64(self.threads),
            pending_offset=np.int64(self.pending_offset),
            stage_handle=np.int64(self.stage_streams.handle),
        )
        self._state.value = dict(memory, delivery=delivery)

    def advance(self, counts):
        """Pack one step, run CUDA dynamics, and publish the updated state."""
        # 1. Gather the table, current state and all input/recording buffers.
        memory = self._state.value
        inputs = (
            *self.table.value,
            self.arena.value,
            self.clock.value,
            memory["core"],
            counts.reshape(-1),
            *self.schedules.value,
            self.recording_inputs.value[0],
            memory["records"],
            memory["monitors"],
            self.recording_inputs.value[1],
            memory["delivery"],
            *self.routes.value,
        )
        output_shapes = (
            tuple(_shape(memory[name]) for name in _MEMORY)
            + (_shape(self.arena.value),)
            + (
                jax.ShapeDtypeStruct((self.count,), np.float64),
                jax.ShapeDtypeStruct((self.count,), np.bool_),
                _shape(self.clock.value),
            )
        )
        # 2. Advance arrivals, response histories, voltage and candidate spikes.
        outputs = self.call(
            "reduction_advance",
            output_shapes,
            {5: 0, 13: 1, 14: 2, 16: 3, 3: 4},
        )(
            *inputs,
            count=np.int64(self.count),
            delays=np.int64(self.delays),
            pending_offset=np.int64(self.pending_offset),
            width=np.int64(self.width),
            schedule_blocks=np.int64(self.schedule_blocks),
            abort_voltage=self.abort_voltage,
            threads=np.int64(self.threads),
            rebase_threads=np.int64(self.rebase_threads),
            ordinary_threads=np.int64(self.ordinary_threads),
            blocks=np.int64(self.blocks),
            ordinary_blocks=np.int64(self.ordinary_blocks),
            arena_handle=np.int64(self.arena.handle),
            stage_handle=np.int64(self.stage_streams.handle),
            **self.table.bank_offsets,
            **{f"{name}_offset": offset for name, offset in self.offsets.items() if name != "neurons"},
            ma_shared_bytes=np.int64(self.ma_shared_bytes),
            panel_shared_bytes=np.int64(self.panel_shared_bytes),
        )
        # 3. Carry persistent state forward and return this step's outputs.
        self._state.value = dict(zip(_MEMORY, outputs[:4]))
        self.arena.value = outputs[4]
        self.clock.value = outputs[7]
        return outputs[5:7]

    def reset(self):
        # The input adapter resets delivery separately, before model state.
        delivery = self._state.value["delivery"]
        self._state.close()
        self._state.value = self._new_memory(delivery)
        self._state.closed = False
        self.clock.value = jnp.asarray(0, dtype=jnp.int64)
        self.recording_inputs.value = (jnp.full((self.count,), -1, dtype=jnp.int64), jnp.zeros((2,), dtype=jnp.int64))
        self.record_start = 0
        self._tail = None
        self.record_stop = 0

    def begin_recording(self, start, count, neurons):
        self.record_start = start if self._tail is None else self._tail[0]
        columns = np.full(self.count, -1, dtype=np.int64)
        columns[neurons] = np.arange(len(neurons))
        self.recording_inputs.value = (
            jax.device_put(columns),
            jnp.asarray([self.record_start, len(neurons)], dtype=jnp.int64),
        )
        memory = self._state.value
        records = jnp.empty((start + count - self.record_start, len(neurons)), dtype=jnp.float64)
        monitors = jnp.empty_like(records)
        if self._tail is not None:
            _, old, monitor = self._tail
            records = records.at[: len(old)].set(old)
            monitors = monitors.at[: len(monitor)].set(monitor)
        self._state.value = dict(memory, records=records, monitors=monitors)
        self.record_stop = start + count
        self._tail = None

    def finish_recording(self):
        memory = self._state.value
        pending = np.asarray(
            self.call("reduction_inspect", jax.ShapeDtypeStruct((self.count,), np.int64))(memory["core"])
        )
        active = pending[(pending >= 0) & (np.asarray(self.recording_inputs.value[0]) >= 0)]
        stop = max(self.record_start, int(active.min(initial=self.record_stop)))
        size = stop - self.record_start
        values = np.asarray(memory["records"])
        if stop < self.record_stop:
            self._tail = (stop, values[size:].copy(), np.asarray(memory["monitors"])[size:].copy())
        result = self.record_start, values[:size].copy()
        self._state.value = dict(
            memory, records=jnp.empty((0, 0), dtype=jnp.float64), monitors=jnp.empty((0, 0), dtype=jnp.float64)
        )
        self.recording_inputs.value = (jnp.full((self.count,), -1, dtype=jnp.int64), jnp.zeros((2,), dtype=jnp.int64))
        return result
