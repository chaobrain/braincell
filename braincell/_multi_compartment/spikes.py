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

"""The spike output of :meth:`Cell.run`: how its compiled loop reads the ``"spike"`` port, and the sparse rows.

A run packs the spikes of each step eight rows to a byte, or on GPU every
K steps, and copies a large grid back compacted (CELL-49); after the run
:meth:`_SpikePort.series` turns the packed grid into the sparse rows of
``RunResult.spikes`` on the host.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.network.recording import EventSeries

if TYPE_CHECKING:
    from braincell._multi_compartment.cell import Cell

# The spike output paths of a run (CELL-49). ``None`` chooses by the
# platform of the Cell's spike state: on GPU a run packs the bits every K
# steps and copies a large grid back compacted. A bool forces the choice
# on any backend; the tests force both paths on CPU.
_BLOCKED_SPIKE_OUTPUT: bool | None = None
_COMPACT_SPIKE_COPY: bool | None = None
#: The block sizes a blocked run takes the largest divisor of its step count from.
_BLOCK_STEPS = range(32, 7, -1)
#: A compacted copy has room for one in this many of the grid's 4-byte groups.
_COMPACT_CAPACITY_DIVISOR = 64
#: Smaller grids are copied back whole, which is faster below about 1 MB on GPU.
_COMPACT_MIN_BYTES = 1 << 20


@dataclass(frozen=True)
class _SpikePort:
    """How one run loop reads the Cell's ``"spike"`` port, and its sparse rows.

    The port is the one ``cell.event_outputs["spike"]`` publishes to a
    Network: ``cell.spike`` at the CV holding the root branch midpoint, the
    one CV of a lumped Cell, or a reduction model's event. Detailed and
    lumped spikes are 0 or 1 (the Heaviside forward of ``spk_fun``), so each
    step packs them eight rows to a byte; a reduction model's event is kept
    as it is, since it may count several events in one step.

    ``block_steps`` is K when the loop packs every K steps instead
    (:meth:`read_bits`); ``capacity`` is the number of nonzero 4-byte
    groups of the packed grid the loop copies back compacted
    (:func:`_compact_spike_rows`), or ``None`` to copy it whole.
    """

    cv_id: int
    packed: bool
    block_steps: int | None = None
    capacity: int | None = None

    @classmethod
    def of(cls, rcell: "Cell", *, n_steps: int, path: tuple[bool, bool]) -> "_SpikePort":
        """The port of ``rcell`` for a run of ``n_steps`` on ``path`` (:func:`_spike_output_path`)."""
        packed = not rcell._uses_reduction
        block_steps = capacity = None
        if packed:
            blocked, compact = path
            if blocked:
                block_steps = next((k for k in _BLOCK_STEPS if n_steps % k == 0), None)
            n_bytes = n_steps * -(-_spike_rows(rcell) // 8)
            if compact and n_bytes >= _COMPACT_MIN_BYTES:
                capacity = max(1, -(-n_bytes // 4) // _COMPACT_CAPACITY_DIVISOR)
        return cls(
            cv_id=int(rcell._get_spike_event_source().cv_id),
            packed=packed,
            block_steps=block_steps,
            capacity=capacity,
        )

    def read(self, rcell: "Cell"):
        """This step's port values, inside the loop."""
        spike = rcell.spike.value
        if not self.packed:
            return spike
        if not rcell._is_lumped:
            spike = spike[..., self.cv_id]
        # Pack the stored spike. Left to itself, XLA recomputes it from the
        # voltages before and after the step inside the packing kernel, which
        # keeps a copy of the old voltage alive on every step (CELL-39).
        spike = jax.lax.optimization_barrier(spike)
        return jnp.packbits(spike.reshape(-1) != 0)

    def read_bits(self, rcell: "Cell"):
        """This step's spike bit of each row, as a ``uint32`` 0 or 1, inside a blocked loop.

        The loop ORs it into the row's word, which fuses into the kernel
        that steps the voltage, so no kernel is left to keep the old voltage
        alive and :meth:`read` needs no barrier here (CELL-49).
        """
        spike = rcell.spike.value
        if not rcell._is_lumped:
            spike = spike[..., self.cv_id]
        return (spike.reshape(-1) != 0).astype(jnp.uint32)

    def series(self, rcell: "Cell", times, stop_time, rows) -> EventSeries:
        """Sparse rows in the format of ``NetworkResult.events[population]["spike"]``.

        A step's events take the step's end time (:func:`step_end_times`
        of the loop's ``times`` and ``stop_time``).
        ``rows`` is the loop's spike output: the steps' port values, or
        their packed grid, or ``(grid, count, groups, bytes)`` when
        ``capacity`` is set (:func:`_compact_spike_rows`).
        """
        if rcell._uses_reduction:
            values = np.asarray(rows)
            values = values.reshape(values.shape[0], -1)
            step, source_id = np.nonzero(values)
            count = values[step, source_id]
            n_rows = values.shape[1]
        else:
            n_rows = _spike_rows(rcell)
            step, source_id = self._packed_events(rows)
            count = np.ones(source_id.shape, dtype=np.int64)
        metadata = {"port": "spike", "cv_id": np.full(n_rows, self.cv_id, dtype=np.int64)}
        batch_size = rcell._runtime_batch_size
        if batch_size is not None:
            metadata["batch_index"], metadata["population_index"] = np.divmod(
                np.arange(n_rows, dtype=np.int64), n_rows // batch_size
            )
        from braincell.network.recording import step_end_times

        ends, unit = step_end_times(times, stop_time)
        return EventSeries(
            time=u.Quantity(ends[step], unit),
            source_id=source_id,
            count=count,
            metadata=metadata,
        )

    def _packed_events(self, rows):
        """``(step, source_id)`` of the set bits of the loop's packed grid, in the grid's C order."""
        if self.capacity is not None:
            rows, count, groups, values = rows
            count = int(count)
            if count <= self.capacity:
                return _compacted_grid_events(np.asarray(groups)[:count], np.asarray(values)[:count], rows.shape[1])
        return _grid_events(np.asarray(rows))


def _spike_rows(rcell: "Cell") -> int:
    """The number of rows of a detailed or lumped Cell's spike port: its members, times its batch."""
    shape = rcell.spike.value.shape
    return int(np.prod(shape if rcell._is_lumped else shape[:-1], dtype=np.int64))


def _grid_events(rows):
    """``(step, row)`` of the set bits of the packed grid ``rows``, a ``(steps, bytes)`` array, in C order."""
    # Only the nonzero bytes are expanded to their eight rows. NumPy finds
    # the nonzero entries of a flat boolean array several times faster than
    # those of a 2-D byte array (CELL-39).
    flat = np.flatnonzero(rows != 0)
    return _set_bits(flat, rows.reshape(-1)[flat], rows.shape[1])


def _compacted_grid_events(groups, values, n_bytes: int):
    """:func:`_grid_events` of a grid of ``n_bytes`` columns from its nonzero 4-byte ``groups`` and their ``values``."""
    group, offset = np.nonzero(values)
    flat = groups[group].astype(np.int64) * 4 + offset
    return _set_bits(flat, values[group, offset], n_bytes)


def _set_bits(flat, values, n_bytes: int):
    """``(step, row)`` of the set bits of the bytes ``values`` at C-order positions ``flat`` of a packed grid."""
    step, byte = np.divmod(flat, n_bytes)
    hit, bit = np.nonzero(np.unpackbits(values[:, None], axis=1))
    return step[hit], byte[hit] * 8 + bit


def _on_gpu(value) -> bool:
    """Whether the array (or Quantity) ``value`` lives on a GPU."""
    devices = getattr(u.get_mantissa(value), "devices", None)
    return devices is not None and any(device.platform == "gpu" for device in devices())


def _spike_output_path(rcell: "Cell") -> tuple[bool, bool]:
    """``(blocked, compact)``: whether a run of ``rcell`` may pack its spikes in blocks and copy them compacted.

    Both on GPU and neither elsewhere, unless the module switches force
    them; a reduction model's port is not packed and takes neither.
    """
    on_gpu = False
    if None in (_BLOCKED_SPIKE_OUTPUT, _COMPACT_SPIKE_COPY) and not rcell._uses_reduction:
        on_gpu = _on_gpu(rcell.spike.value)
    blocked = on_gpu if _BLOCKED_SPIKE_OUTPUT is None else bool(_BLOCKED_SPIKE_OUTPUT)
    compact = on_gpu if _COMPACT_SPIKE_COPY is None else bool(_COMPACT_SPIKE_COPY)
    return blocked, compact


def _words_to_rows(words, block_steps: int):
    """The ``(blocks, rows)`` words of a blocked loop as its ``(steps, ceil(rows / 8))`` packed grid.

    Bit ``j`` of a row's word is the row's spike at step ``j`` of the
    block; each step's bytes are ``jnp.packbits`` of its rows, as a loop
    that packs at every step stacks them.
    """
    bits = (words[:, None, :] >> jnp.arange(block_steps, dtype=jnp.uint32)[:, None]) & jnp.uint32(1)
    packed = jnp.packbits(bits, axis=-1)
    return packed.reshape((-1, packed.shape[-1]))


def _compact_spike_rows(rows, capacity: int):
    """``(count, groups, values)``: the nonzero 4-byte groups of the packed grid ``rows``, on the device.

    The grid's bytes in C order make groups of four, the last padded with
    zeros. ``groups`` holds the indices of the first ``capacity`` nonzero
    groups in order and ``values`` their bytes; ``count`` is how many there
    are, so a count above ``capacity`` means the grid is to be copied whole.
    """
    flat = rows.reshape(-1)
    n_groups = -(-flat.shape[0] // 4)
    groups = jnp.pad(flat, (0, 4 * n_groups - flat.shape[0])).reshape((n_groups, 4))
    nonzero = jnp.any(groups != 0, axis=1)
    position = jnp.cumsum(nonzero, dtype=jnp.int32)
    slot = jnp.where(nonzero, position - 1, capacity)
    index = jnp.zeros(capacity, jnp.int32).at[slot].set(jnp.arange(n_groups, dtype=jnp.int32), mode="drop")
    return position[-1], index, groups[index]
