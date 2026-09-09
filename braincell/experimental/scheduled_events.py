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

"""Experimental fixed-window scheduled event queries.

Design: docs/design/synapse/proposals/event-delivery-optimization.md.
These kernels do not replace Network delivery. Schedules, topology, delay and dt
are immutable for the lifetime of a plan. Only integer scheduling is performed;
weights and synapse dynamics remain outside the scheduling loops.
"""

from dataclasses import dataclass
import time

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.network.event import NetStim, _round_half_up_steps

METHODS = ("current", "scan", "cursor", "bucket")


@dataclass(frozen=True)
class EventPlan:
    """Hold a prepared fixed-window event query.

    Parameters
    ----------
    source : NetStim
        Immutable scheduled source used as the reference implementation.
    source_index : numpy.ndarray
        One source index per connection.
    delay, dt : brainunit.Quantity
        Connection delays and fixed time step.
    method : str
        ``current``, precomputed ``scan``, ``cursor``, or compact ``bucket``.
    start_step, n_steps, block_size : int
        Prepared window and bucket processing block size.
    arrays : tuple
        Device arrays consumed by ``query``. Pass these as dynamic JIT arguments
        when benchmarking to avoid compiling a prerecorded count trajectory.
    preparation : dict
        Preparation phase timings and logical temporary-array sizes.

    Notes
    -----
    Query consecutive steps in the prepared window. To restart or seek forward,
    initialize a new cursor with ``reset``. Rebuild after changing the schedule,
    delay, dt, topology or window. No trainable event-time/delay API is provided.
    """

    source: NetStim
    source_index: np.ndarray
    delay: u.Quantity
    dt: u.Quantity
    method: str
    start_step: int
    n_steps: int
    block_size: int
    arrays: tuple
    preparation: dict

    def reset(self, start_step=None):
        """Return fresh query state at a validated window position.

        Parameters
        ----------
        start_step : int or None
            First step to query; defaults to the prepared window start.

        Returns
        -------
        jax.Array
            Integer connection cursors, or an empty state for other methods.
        """
        start = self.start_step if start_step is None else start_step
        if not isinstance(start, (int, np.integer)) or not self.start_step <= start < self.start_step + self.n_steps:
            raise ValueError("reset position must lie in the prepared window")
        if self.method == "cursor":
            steps, lengths = self.arrays
            return jnp.minimum(jnp.sum(steps < start, axis=1, dtype=jnp.int32), lengths)
        return jnp.empty(0, dtype=jnp.int32)

    def query(self, arrays, state, step):
        """Count arrivals and advance integer state at one consecutive step.

        Parameters
        ----------
        arrays : tuple
            This plan's device arrays, supplied as dynamic compiled arguments.
        state : jax.Array
            State from ``reset`` or the preceding query.
        step : int or jax.Array
            Absolute step within the prepared window.

        Returns
        -------
        tuple
            Updated state and integer counts of shape ``(connections,)``.
        """
        n = self.source_index.size
        if self.method == "current":
            count = self.source.event_count(self.source_index, t=step * self.dt, delay=self.delay, dt=self.dt)
            return state, count.astype(jnp.int32)
        if self.method == "scan":
            return state, jnp.sum(arrays[0] == step, axis=1, dtype=jnp.int32)
        if self.method == "cursor":
            steps, lengths = arrays
            rows = jnp.arange(n)

            def due(p):
                return (p < lengths) & (steps[rows, p] == step)

            end = jax.lax.while_loop(lambda p: jnp.any(due(p)), lambda p: p + due(p).astype(jnp.int32), state)
            return end, jax.lax.stop_gradient(end - state)

        events, offsets = arrays
        lo, hi = offsets[step - self.start_step], offsets[step - self.start_step + 1]
        lanes = jnp.arange(self.block_size)

        def consume(carry):
            pos, counts = carry
            indices = jax.lax.dynamic_slice_in_dim(events, pos, self.block_size)
            counts = counts.at[indices].add((pos + lanes < hi).astype(jnp.int32))
            return pos + self.block_size, counts

        _, counts = jax.lax.while_loop(lambda carry: carry[0] < hi, consume, (lo, jnp.zeros(n, dtype=jnp.int32)))
        return state, jax.lax.stop_gradient(counts)


def prepare_events(source, source_index, *, delay, dt, n_steps, start_step=0, method="cursor", block_size=128):
    """Prepare a scheduled source without changing its delivery semantics.

    Parameters
    ----------
    source : NetStim
        Generated, immutable event schedule.
    source_index : array-like
        Nonempty one-dimensional integer source indices, one per connection.
    delay : brainunit.Quantity
        Nonnegative scalar or per-connection delay.
    dt : brainunit.Quantity
        Positive scalar time step.
    n_steps, start_step : int
        Positive window length and nonnegative absolute starting step.
    method : str
        One of ``current``, ``scan``, ``cursor`` or ``bucket``.
    block_size : int
        Positive fixed block size for compact bucket consumption.

    Returns
    -------
    EventPlan
        Prepared device arrays and query operations.

    Raises
    ------
    TypeError
        Source, indices or physical units have incorrect types.
    ValueError
        Values, shapes, schedule ordering or window bounds are invalid.

    Notes
    -----
    Quantization uses the production JAX expression on the selected device and
    precision, including its half-tie snapping. Compact bucket preparation copies
    the integer arrival table to the host to sort and then transfers the compact
    arrays back. Its preparation time and temporary memory must be accounted for.
    """
    began = time.perf_counter()
    if not isinstance(source, NetStim):
        raise TypeError("source must be a NetStim")
    if method not in METHODS:
        raise ValueError("unknown query method")
    for name, value, lower in (("n_steps", n_steps, 1), ("start_step", start_step, 0), ("block_size", block_size, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"invalid {name}")
    if start_step + n_steps >= np.iinfo(np.int32).max:
        raise ValueError("window exceeds int32 step indexing")
    index = np.asarray(source_index)
    if index.dtype.kind not in "iu":
        raise TypeError("source_index must contain integers")
    if index.ndim != 1 or index.size == 0 or np.any(index < 0) or np.any(index >= source.size):
        raise ValueError("source_index must be a nonempty valid vector")
    index = index.astype(np.int32, copy=True)
    if not isinstance(delay, u.Quantity) or not isinstance(dt, u.Quantity):
        raise TypeError("delay and dt require time units")
    delay_ms = np.asarray(delay.to_decimal(u.ms))
    dt_ms = np.asarray(dt.to_decimal(u.ms))
    if dt_ms.ndim or not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt must be finite, scalar and positive")
    if delay_ms.shape not in ((), index.shape) or np.any(~np.isfinite(delay_ms)) or np.any(delay_ms < 0):
        raise ValueError("delay must be finite, nonnegative and scalar or per-connection")
    clock_dtype = np.dtype(jax.dtypes.canonicalize_dtype(dt_ms.dtype))
    last_time = float(dt_ms) * (start_step + n_steps - 1)
    if clock_dtype.kind in "iu":
        if last_time > np.iinfo(clock_dtype).max:
            raise ValueError("integer time multiplication would overflow; use a floating time step")
        clock_dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    clock_limits = np.finfo(clock_dtype)
    # Four-ULP half-tie snapping reaches a half step at this magnitude.
    # Beyond it even integer step ratios need not map to themselves.
    if start_step + n_steps > 2 ** (clock_limits.nmant - 3):
        raise ValueError("window exceeds the clock precision's half-tie resolution")
    with np.errstate(over="ignore", under="ignore"):
        clock_dt = np.asarray(dt_ms, dtype=clock_dtype)
    if not np.isfinite(clock_dt) or clock_dt <= 0 or last_time > clock_limits.max:
        raise ValueError("time grid is not representable in the selected clock precision")
    times = np.asarray(source._event_times_ms[index])
    mask = np.asarray(source._event_mask[index])
    if times.shape[1] and (np.any(~np.isfinite(times[mask])) or np.any(times[mask] < 0)):
        raise ValueError("valid event times must be finite and nonnegative")
    # NetStim pads only at the end; enforcing this permits one cursor per row.
    if np.any(np.diff(mask.astype(int), axis=1) > 0) or np.any((np.diff(times, axis=1) < 0) & mask[:, 1:]):
        raise ValueError("event schedules must be sorted with trailing padding")
    arrays = ()
    preparation = dict(
        host_validation_s=time.perf_counter() - began,
        quantize_cold_s=0.0,
        device_to_host_s=0.0,
        bucket_build_s=0.0,
        upload_and_pack_s=0.0,
        gathered_source_bytes=times.nbytes + mask.nbytes,
        arrival_table_bytes=0,
    )
    if method != "current":
        began = time.perf_counter()
        delays = jnp.broadcast_to(u.math.asarray(delay_ms), index.shape)
        arrivals = u.math.asarray(times) + delays[:, None]
        quantized = _round_half_up_steps(arrivals / u.math.asarray(dt_ms))
        # Future values can exceed int32; clip before conversion, retaining all
        # in-window steps exactly and using a sentinel for masked entries.
        limit = start_step + n_steps
        steps = jnp.where(jnp.asarray(mask), jnp.minimum(quantized, limit), limit).astype(jnp.int32)
        jax.block_until_ready(steps)
        preparation["quantize_cold_s"] = time.perf_counter() - began
        preparation["arrival_table_bytes"] = steps.size * steps.dtype.itemsize
        began = time.perf_counter()
        if method == "scan":
            arrays = (steps,)
        elif method == "cursor":
            arrays = (
                jnp.pad(steps, ((0, 0), (0, 1)), constant_values=limit),
                jnp.asarray(mask.sum(axis=1), dtype=jnp.int32),
            )
        else:
            host_steps = np.asarray(steps)
            preparation["device_to_host_s"] = time.perf_counter() - began
            began = time.perf_counter()
            connection, ordinal = np.nonzero((host_steps >= start_step) & (host_steps < limit) & mask)
            arrival = host_steps[connection, ordinal]
            order = np.lexsort((ordinal, connection, arrival))
            events = connection[order].astype(np.int32)
            offsets = np.concatenate(([0], np.cumsum(np.bincount(arrival - start_step, minlength=n_steps))))
            if events.size + block_size >= np.iinfo(np.int32).max:
                raise ValueError("bucket exceeds int32 event indexing")
            preparation["bucket_build_s"] = time.perf_counter() - began
            began = time.perf_counter()
            # A trailing block prevents dynamic_slice from clamping its start.
            arrays = (jnp.asarray(np.pad(events, (0, block_size))), jnp.asarray(offsets, dtype=jnp.int32))
        jax.block_until_ready(arrays)
        preparation["upload_and_pack_s"] = time.perf_counter() - began
    return EventPlan(
        source, index, delay, dt, method, int(start_step), int(n_steps), int(block_size), arrays, preparation
    )
