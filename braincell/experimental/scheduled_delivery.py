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

"""Deliver fixed-window scheduled events directly to target layouts.

See docs/design/synapse/proposals/event-delivery-optimization.md.
"""

from dataclasses import dataclass
import time

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.experimental.scheduled_events import prepare_events


class PaddingLimitError(ValueError):
    """Indicate that a fixed-width plan exceeds its declared resource limits."""


@dataclass(frozen=True)
class DeliveryPlan:
    """Hold an immutable, stateless scheduled delivery plan.

    Parameters
    ----------
    method : str
        Compact ``direct`` or fixed-width ``padded``.
    arrays : tuple
        Integer device arrays; pass as dynamic arguments to the compiled driver.
    n_connections, n_targets, start_step, n_steps, block_size : int
        Fixed topology, valid step window and compact block size.
    preparation : dict
        Preparation timings and storage statistics, not peak memory.
    """

    method: str
    arrays: tuple
    n_connections: int
    n_targets: int
    start_step: int
    n_steps: int
    block_size: int
    preparation: dict

    def deliver(self, arrays, step, weights):
        """Return this step's weighted target input without mutating state.

        Parameters
        ----------
        arrays : tuple
            This plan's integer arrays, passed dynamically through JIT.
        step : int or jax.Array
            Absolute step inside the prepared window. Seeking/restarting needs
            no reset because delivery has no cursor.
        weights : brainunit.Quantity
            Current per-connection payloads, shape ``(n_connections,)``.

        Returns
        -------
        brainunit.Quantity
            Shape ``(n_targets,)``, with the same unit as weights.

        Notes
        -----
        Supports first-order reverse differentiation of weights. Source times,
        delays and topology are fixed. Forward-mode and higher-order derivative
        support are not part of the compact plan contract.
        """
        if not isinstance(weights, u.Quantity):
            raise TypeError("weights must carry explicit units")
        if weights.shape != (self.n_connections,):
            raise ValueError("weights must have one entry per connection")
        raw = u.get_mantissa(weights)
        local_step = step - self.start_step
        if self.method == "padded":
            connections, multiplicity, targets = arrays
            ids, count = connections[local_step], multiplicity[local_step]
            value = raw[ids] * count
            result = jnp.zeros(self.n_targets, dtype=value.dtype).at[targets[ids]].add(value)
        else:
            result = _compact_op(self.n_targets, self.block_size)(arrays, local_step, raw)
        return u.Quantity(result, u.get_unit(weights))


def _compact_op(n_targets, block_size):
    """Define a sparse primal and sparse transpose for a fixed layout."""

    def consume(arrays, step, values, transpose=False):
        connections, multiplicity, offsets, targets = arrays
        lo, hi = offsets[step], offsets[step + 1]
        size = targets.size if transpose else n_targets

        def body(carry):
            pos, result = carry
            ids = jax.lax.dynamic_slice_in_dim(connections, pos, block_size)
            count = jax.lax.dynamic_slice_in_dim(multiplicity, pos, block_size)
            valid = pos + jnp.arange(block_size) < hi
            src, dst = (targets[ids], ids) if transpose else (ids, targets[ids])
            value = jnp.where(valid, values[src] * count, 0)
            # Invalid lanes scatter out of bounds rather than contending on 0.
            result = result.at[jnp.where(valid, dst, size)].add(value, mode="drop")
            return pos + block_size, result

        return jax.lax.while_loop(lambda carry: carry[0] < hi, body, (lo, jnp.zeros(size, dtype=values.dtype)))[1]

    @jax.custom_vjp
    def op(arrays, step, weights):
        return consume(arrays, step, weights)

    def forward(arrays, step, weights):
        return consume(arrays, step, weights), (arrays, step)

    def backward(residual, cotangent):
        arrays, step = residual
        return None, None, consume(arrays, step, cotangent, transpose=True)

    op.defvjp(forward, backward)
    return op


def prepare_delivery(
    source,
    source_index,
    target_index,
    *,
    n_targets,
    delay,
    dt,
    n_steps,
    start_step=0,
    method="direct",
    block_size=128,
    max_bytes=256 * 1024**2,
    max_padding_ratio=8.0,
):
    """Prepare compressed scheduled records for direct weighted delivery.

    Parameters
    ----------
    source : braincell.NetStim
        Immutable source schedule.
    source_index, target_index : array-like
        One source and target index per connection.
    n_targets : int
        Positive flattened target layout size.
    delay, dt : brainunit.Quantity
        Nonnegative connection delays and positive scalar time step.
    n_steps, start_step : int, optional
        Positive window length and nonnegative absolute start step.
    method : str, optional
        ``direct`` (default) or ``padded``.
    block_size : int, optional
        Compact-loop block size, default 128.
    max_bytes : int, optional
        Maximum padded device-array bytes, default 256 MiB.
    max_padding_ratio : float, optional
        Maximum padded slots per compressed record, default 8.

    Returns
    -------
    DeliveryPlan
        Stateless plan whose dynamic weights remain differentiable.

    Raises
    ------
    PaddingLimitError
        Fixed-width storage exceeds either limit; no silent fallback occurs.
    TypeError, ValueError
        An index, method, grid, unit or window is invalid.

    Notes
    -----
    Source/grid validation and quantization use ``prepare_events``. Preparation
    still materializes the connection event table. Source, topology, delay, dt
    or window changes require rebuilding; changing weights does not.
    """
    began = time.perf_counter()
    if method not in ("direct", "padded"):
        raise ValueError("method must be direct or padded")
    if isinstance(n_targets, bool) or not isinstance(n_targets, (int, np.integer)) or n_targets < 1:
        raise ValueError("n_targets must be a positive integer")
    targets = np.asarray(target_index)
    if targets.ndim != 1 or targets.dtype.kind not in "iu":
        raise TypeError("target_index must be a one-dimensional integer array")
    if targets.shape != np.shape(source_index) or np.any(targets < 0) or np.any(targets >= n_targets):
        raise ValueError("target indices must match connections and lie in the target layout")
    if not np.isfinite(max_padding_ratio) or max_padding_ratio < 1 or max_bytes < 1:
        raise ValueError("padding limits must be positive, with ratio at least one")
    query = prepare_events(
        source,
        source_index,
        delay=delay,
        dt=dt,
        n_steps=n_steps,
        start_step=start_step,
        method="bucket",
        block_size=block_size,
    )
    compression_start = time.perf_counter()
    raw_events, raw_offsets = (np.asarray(x) for x in query.arrays)
    n_events = int(raw_offsets[-1])
    raw_events = raw_events[:n_events]
    steps = np.repeat(np.arange(n_steps, dtype=np.int32), np.diff(raw_offsets))
    # Existing bucket order is (step, connection, ordinal), so run-length
    # compression preserves the per-connection multiplication of the baseline.
    changed = np.r_[True, (np.diff(raw_events) != 0) | (np.diff(steps) != 0)] if n_events else np.array([], bool)
    begins = np.flatnonzero(changed)
    ids = raw_events[begins]
    unique_steps = steps[begins]
    multiplicity = np.diff(np.r_[begins, n_events]).astype(np.int32)
    per_step = np.bincount(unique_steps, minlength=n_steps)
    offsets = np.r_[0, np.cumsum(per_step)].astype(np.int32)
    width = max(1, int(per_step.max()))
    slots = n_steps * width
    padded_bytes = (2 * slots + targets.size) * np.dtype(np.int32).itemsize
    ratio = slots / max(1, ids.size)
    stats = dict(query.preparation)
    stats.update(
        arrivals=n_events,
        compressed_records=int(ids.size),
        padded_width=width,
        padding_ratio=ratio,
        padded_bytes=padded_bytes,
    )
    if method == "padded" and (padded_bytes > max_bytes or (ids.size and ratio > max_padding_ratio)):
        raise PaddingLimitError(f"padded requires {padded_bytes} bytes, ratio {ratio:.3g}")
    if method == "direct":
        host_arrays = (np.pad(ids, (0, block_size)), np.pad(multiplicity, (0, block_size)), offsets, targets)
    else:
        packed_ids = np.zeros((n_steps, width), dtype=np.int32)
        packed_count = np.zeros_like(packed_ids)
        lane = np.arange(ids.size) - offsets[unique_steps]
        packed_ids[unique_steps, lane] = ids
        packed_count[unique_steps, lane] = multiplicity
        host_arrays = packed_ids, packed_count, targets
    stats["compression_and_pack_s"] = time.perf_counter() - compression_start
    upload_start = time.perf_counter()
    arrays = tuple(jnp.asarray(x, dtype=jnp.int32) for x in host_arrays)
    jax.block_until_ready(arrays)
    stats["direct_upload_s"] = time.perf_counter() - upload_start
    stats["total_prepare_s"] = time.perf_counter() - began
    stats["array_bytes"] = sum(x.size * x.dtype.itemsize for x in arrays)
    return DeliveryPlan(
        method, arrays, targets.size, int(n_targets), int(start_step), int(n_steps), int(block_size), stats
    )
