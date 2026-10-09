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

"""Compile DIF kernels, pack tables and own CUDA resources."""

from __future__ import annotations

import ctypes
from functools import cache, partial
from pathlib import Path

import brainevent
import brainstate
import jax
import numpy as np

_INDICES = (
    "rest_single_ptr",
    "rest_pair_ptr",
    "post_single_ptr",
    "post_pair_ptr",
    "rebase_single_ptr",
    "rebase_pair_ptr",
    "rest_support",
    "post_support",
    "rebase_support",
    "history_support",
    "time_lower",
    "time_upper",
    "time_lower",
    "time_upper",
    "time_steps",
    "time_steps",
    "time_steps",
    "pair_source_ptr",
    "pair_sources",
    "pair_targets",
    "pair_reverse",
    "rebase_single_initial_ptr",
    "rebase_pair_initial_ptr",
    "query_support",
)
_REALS = (
    "rest_grid",
    "post_state_grid",
    "time_ratio",
    "time_ratio",
    "time_inverse_width",
    "time_inverse_width",
    "time_inverse_width",
    "rebase_single_initial",
    "rebase_pair_initial",
    "reversal",
)


def _shape(array):
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


def _call(name, outputs, aliases=None, *, prefix):
    return jax.ffi.ffi_call(
        f"{prefix}.{name}",
        outputs,
        has_side_effect=True,
        input_output_aliases=aliases,
        vmap_method="sequential",
    )


class _StageStreams:
    """Own one device's auxiliary CUDA stream and its ordering events."""

    def __init__(self, library, device):
        self.library = library
        self.handle = library.reduction_stage_create(int(device.local_hardware_id))
        if not self.handle:
            raise RuntimeError("CUDA could not create reduction concurrent-stage resources.")

    def close(self):
        if self.handle:
            # DeviceState retains this owner until queued use of its core ends.
            self.library.reduction_stage_free(self.handle)
            self.handle = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _Arena(brainstate.LongTermState):
    """Own an explicit CUDA arena, with physical pages committed on demand."""

    def __init__(self, library, capacity):
        self.library = library
        self.capacity = capacity
        self.handle = library.reduction_arena_create(capacity)
        if not self.handle:
            raise RuntimeError("CUDA could not create a reduction event arena.")
        control = np.zeros(library.reduction_arena_control_size(), dtype=np.int64)
        control[1 : 1 + (control.size - 2) // 4] = -1
        super().__init__(jax.device_put(control))

    def __del__(self):
        try:
            if self.handle:
                jax.block_until_ready(self.value)
                jax.effects_barrier()
                self.library.reduction_arena_free(self.handle)
                self.handle = 0
        except Exception:
            pass


def _arena(device, capacity, library, *, arenas):
    """Share the event arena through explicit State ownership, including runners."""
    arena = arenas.get(device)
    if arena is None:
        with jax.default_device(device):
            arena = _Arena(library, capacity)
        arenas[device] = arena
    elif arena.capacity < capacity:
        if int(np.asarray(arena.value)[0]):
            raise RuntimeError("Set gpu_heap_bytes for all reduction models before their first simulation.")
        with jax.default_device(device):
            handle = arena.library.reduction_arena_create(capacity)
            if not handle:
                raise RuntimeError("CUDA could not reserve the requested reduction arena budget.")
            arena.library.reduction_arena_free(arena.handle)
            arena.handle = handle
            arena.capacity = capacity
    return arena


class _DeviceState(brainstate.LongTermState):
    """Keep FIFO offsets and their arena alive as long as a runner owns them."""

    def __init__(self, value, call, arena):
        super().__init__(value)
        self.call = call
        self.arena = arena
        self.closed = False

    def close(self):
        # Collection may run during another model's trace. Do not register
        # an already released State as one of that model's inputs.
        if self.closed:
            return
        memory = self.value
        # Cyclic finalization can close the arena before this state. Its
        # allocations are already gone; a release call would dereference 0.
        if not self.arena.handle:
            self.value = None
            self.closed = True
            return
        jax.effects_barrier()
        inputs = (memory["core"], self.arena.value)
        outputs = self.call(
            "reduction_release", tuple(_shape(value) for value in inputs), {i: i for i in range(len(inputs))}
        )(*inputs, **self.release_layout, arena_handle=np.int64(self.arena.handle))
        jax.block_until_ready(outputs)
        self.arena.value = outputs[1]
        self.value = None
        self.closed = True

    def __del__(self):
        # CUDA may already have shut down during interpreter teardown.
        try:
            self.close()
        except Exception:
            pass


@cache
def load_cuda_module(threads, index_bits, high_order):
    name = f"braincell_reduction_dif_{threads}_i{index_bits}_h{int(high_order)}"
    module = brainevent.load_cuda_file(
        Path(__file__).with_name("kernels.cu"),
        name=name,
        extra_cuda_cflags=[
            f"-DREDUCTION_HIGH_ORDER={int(high_order)}",
            "--fmad=true",
            f"-DREDUCTION_BLOCK_THREADS={threads}",
            f"-DREDUCTION_INDEX_BITS={index_bits}",
        ],
        extra_ldflags=["-lcuda"],
        use_fast_math=False,
        # Host-side arena management and stream ordering require ordinary FFI calls.
        allow_cuda_graph=False,
    )
    library = ctypes.CDLL(module.path)
    for symbol in (
        "reduction_neuron_size",
        "reduction_history_size",
        "reduction_arena_control_size",
    ):
        getattr(library, symbol).restype = ctypes.c_size_t
        getattr(library, symbol).argtypes = []
    library.reduction_jobs_size.restype = ctypes.c_size_t
    library.reduction_jobs_size.argtypes = [ctypes.c_int64]
    library.reduction_resident_blocks.restype = ctypes.c_int
    library.reduction_resident_blocks.argtypes = [ctypes.c_bool, ctypes.c_int, ctypes.c_int64]
    library.reduction_configure.restype = ctypes.c_int
    library.reduction_configure.argtypes = [ctypes.c_bool, ctypes.c_int64]
    library.reduction_arena_create.restype = ctypes.c_uint64
    library.reduction_arena_create.argtypes = [ctypes.c_size_t]
    library.reduction_arena_free.restype = None
    library.reduction_arena_free.argtypes = [ctypes.c_uint64]
    library.reduction_stage_create.restype = ctypes.c_uint64
    library.reduction_stage_create.argtypes = [ctypes.c_int]
    library.reduction_stage_free.restype = None
    library.reduction_stage_free.argtypes = [ctypes.c_uint64]
    library.reduction_ma_rebase_configure.restype = ctypes.c_int64
    library.reduction_ma_rebase_configure.argtypes = [ctypes.c_int64]
    library.reduction_panel_configure.restype = ctypes.c_int64
    library.reduction_panel_configure.argtypes = [ctypes.c_int64]
    library.reduction_stage_metadata.restype = ctypes.c_int
    library.reduction_stage_metadata.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return module, library, partial(_call, prefix=name)


def pack_device_table(table):
    data = table
    scalars = (
        data.rest_support.size,
        int(data.response_steps),
        data.time_steps.shape[1],
        data.time_steps.shape[1],
        data.time_steps.shape[1],
        data.rest_grid.size,
        data.post_state_grid.size,
        data.eta.size,
        int(data.rebase_age),
        int(data.candidate_refractory_age),
        int(data.trough_age),
        data.pair_targets.size,
        data.time_lower.shape[1],
    )
    header_size = len(scalars)
    header = np.zeros(header_size + len(_INDICES) + len(_REALS), dtype=np.int64)
    header[:header_size] = scalars
    integers = [header]
    reals = [
        np.asarray(
            [
                data.rest_voltage,
                data.spike_threshold,
                data.confirmation_voltage,
                table.leak_per_ms,
                table.dt_ms,
                0.5 * float(np.min(data.rest_grid) + np.max(data.rest_grid)) - float(data.rest_voltage),
                float(np.max(data.rest_grid) - np.min(data.rest_grid)) / np.sqrt(12.0),
            ],
            dtype=np.float64,
        )
    ]
    position = header_size
    for fields, arrays in ((_INDICES, integers), (_REALS, reals)):
        offsets = {}
        offset = arrays[0].size
        for name in dict.fromkeys(fields):
            values = np.asarray(getattr(table, name), dtype=arrays[0].dtype).ravel()
            offsets[name] = offset
            arrays.append(values)
            offset += values.size
        # Distinct CUDA query axes use the same calibrated grid storage.
        header[position : position + len(fields)] = [offsets[name] for name in fields]
        position += len(fields)
    banks = (
        data.rest_single,
        data.rest_pair,
        data.post_single,
        data.post_pair,
        data.rebase_single,
        data.rebase_pair,
        data.eta,
    )
    # Native FP64 tables are copied bit-for-bit.
    words = []
    for bank in banks[:-1]:
        flat = np.ascontiguousarray(bank, dtype=np.float64).ravel()
        words.append(flat.view(np.uint64))
    words.append(np.ascontiguousarray(banks[-1], dtype=np.float64).view(np.uint64))
    offsets = np.r_[0, np.cumsum([(bank.size + 15) // 16 * 16 for bank in words])]
    curves = np.zeros(offsets[-1], dtype=np.uint64)
    for bank, first in zip(words, offsets[:-1]):
        curves[first : first + bank.size] = bank
    prepared = brainstate.LongTermState(
        tuple(jax.device_put(value) for value in (curves, np.concatenate(integers), np.concatenate(reals)))
    )
    prepared.host_header = header
    prepared.host_scalars = reals[0]
    prepared.bank_offsets = dict(
        zip(("rp_offset", "ps_offset", "pp_offset", "bs_offset", "bp_offset", "eta_offset"), offsets[1:-1])
    )
    return prepared
