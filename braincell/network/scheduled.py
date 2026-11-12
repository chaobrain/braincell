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

"""Planned delivery of scheduled events for ``Network.run`` and ``prepare_run`` (CELL-9).

By default a Cell counts the arrivals of every scheduled row
(``NetStim``, ``EventSequence``) on every step (``event_count``). With
``scheduled_delivery="direct"`` or ``"padded"`` the Network instead
prepares, on the host and before the compiled loop, a plan of the
arrivals of a window of steps: for each synapse layout, the ``(step,
row)`` records of its scheduled rows without a plasticity rule,
compressed so that one record holds the number of events a row receives
on a step. The plan's integer arrays are arguments of the compiled loop,
so a window of the same length reuses it; on each step the layout reads
the records of that step, weights them with the rows' current weights,
and adds them to its targets.

- ``"direct"`` stores the records in step order with one offset per
  step and reads only the current step's records, in blocks of
  :data:`BLOCK_SIZE`. Its reverse derivative is a ``custom_vjp``, so it
  supports first-order reverse differentiation only.
- ``"padded"`` stores a ``(steps, width)`` matrix, ``width`` being the
  most records any step has, rounded up to a power of two, and reads one
  row of it per step with ordinary JAX operations. It raises
  :class:`PaddingLimitError` when the matrix would take more than
  :data:`MAX_BYTES`, or more than :data:`MAX_PADDING_RATIO` times the
  slots the records need.

The arrival steps are those of the default path (``(time + delay) / dt``
rounded at float64 on the host, CELL-46), and the current step is the
loop's integer step, so a plan delivers each event at the step the
default path does. Rows with a plasticity rule are left to the rule
(``advance_rules``); a ``"sum_count"`` layout also receives the counts,
the plan's multiplicities scattered with unit weight.
"""

from __future__ import annotations

import collections
import contextlib
from dataclasses import dataclass, field
import time

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.network.event import _MAX_STEP, EventSequence, NetStim, _boundary_step, _host_arrival_steps

__all__ = ["METHODS", "PaddingLimitError"]

#: The values of ``scheduled_delivery`` other than ``None``.
METHODS = ("direct", "padded")

#: Records ``"direct"`` reads per loop iteration.
BLOCK_SIZE = 128

#: Largest ``"padded"`` plan of one layout, in bytes.
MAX_BYTES = 256 * 1024**2

#: Largest ratio of a ``"padded"`` plan's slots (steps x needed width) to its records.
MAX_PADDING_RATIO = 8.0

#: Windows a Network keeps the plans of.
CACHE_SIZE = 4

#: ``brainstate.environ`` key of the plans the layouts of a compiled loop read.
_PLANS_KEY = "_braincell_scheduled_plans"

_INT32_MAX = int(np.iinfo(np.int32).max)


class PaddingLimitError(ValueError):
    """A ``"padded"`` plan would exceed its byte or padding-ratio limit.

    Nothing is truncated and no other method is chosen instead: the run
    raises before it starts. ``"direct"`` has no such limit.
    """


def normalize_scheduled_delivery(value, *, prefix: str):
    """Return ``value`` if it is ``None``, ``"direct"`` or ``"padded"``; raise ``ValueError`` otherwise."""
    if value is None or (isinstance(value, str) and value in METHODS):
        return value
    raise ValueError(f"{prefix} scheduled_delivery must be None, 'direct', or 'padded', got {value!r}.")


# ----------------------------------------------------------------------
# Arrival tables


@dataclass(frozen=True)
class _ArrivalTable:
    """The arrival steps of one ``connect`` call's rows, sorted by step.

    Rows with the same source and delay (a group) share their arrivals,
    as in the default path's ``_ArrivalPlan``: the table holds one entry
    per (group, event), and a window expands each entry to its group's
    rows.
    """

    steps: np.ndarray
    group: np.ndarray
    group_offsets: np.ndarray
    group_rows: np.ndarray

    @classmethod
    def build(cls, event_source, event_time_ms, source_index, delay_ms, dt_ms) -> "_ArrivalTable":
        rows = np.asarray(source_index, dtype=np.int64).reshape(-1)
        row_delay = np.broadcast_to(np.asarray(delay_ms, dtype=np.float64), rows.shape)
        delay_bits = row_delay.view(np.uint64).astype(np.int64)
        keys, row_group = np.unique(np.stack([rows, delay_bits], axis=1), axis=0, return_inverse=True)
        row_group = row_group.reshape(-1)
        _, first_row = np.unique(row_group, return_index=True)
        group_source = keys[:, 0]
        group_delay = row_delay[first_row]
        group_rows = np.argsort(row_group, kind="stable").astype(np.int64)
        group_offsets = np.concatenate([[0], np.cumsum(np.bincount(row_group, minlength=len(keys)))])

        order = np.argsort(event_source, kind="stable")
        sorted_source = np.asarray(event_source, dtype=np.int64)[order]
        sorted_time = np.asarray(event_time_ms, dtype=np.float64)[order]
        lo = np.searchsorted(sorted_source, group_source, side="left")
        per_group = np.searchsorted(sorted_source, group_source, side="right") - lo
        n_pairs = int(per_group.sum())
        pair_group = np.repeat(np.arange(len(keys), dtype=np.int64), per_group)
        pair_event = np.arange(n_pairs) + np.repeat(lo - (np.cumsum(per_group) - per_group), per_group)
        steps = _host_arrival_steps(sorted_time[pair_event], group_delay[pair_group], dt_ms).astype(np.int64)
        by_step = np.argsort(steps, kind="stable")
        return cls(steps[by_step], pair_group[by_step], group_offsets, group_rows)

    def last_step(self) -> int:
        """The last step with an arrival, or ``-1``; arrivals clipped to the step limit never come."""
        reachable = self.steps[self.steps < _MAX_STEP]
        return int(reachable[-1]) if reachable.size else -1

    def window(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, rows)`` of the arrivals at the steps ``[start, stop)``, in step order."""
        lo, hi = np.searchsorted(self.steps, [start, stop], side="left")
        steps, group = self.steps[lo:hi], self.group[lo:hi]
        sizes = self.group_offsets[group + 1] - self.group_offsets[group]
        total = int(sizes.sum())
        first = np.repeat(self.group_offsets[group], sizes)
        within = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        return np.repeat(steps, sizes), self.group_rows[first + within]


def _source_events(source) -> tuple[np.ndarray, np.ndarray]:
    """``(source index, time in ms)`` of a scheduled source's generated events."""
    if isinstance(source, NetStim):
        event_source, column = np.nonzero(source._event_mask)
        return event_source, source._event_times_ms[event_source, column]
    if isinstance(source, EventSequence):
        return np.asarray(source.events.source_index), np.asarray(source.events.time.to_decimal(u.ms), dtype=np.float64)
    # A Network registers no other scheduled source.
    raise TypeError(  # pragma: no cover
        f"scheduled_delivery supports NetStim and EventSequence sources, not {type(source).__name__}."
    )


def _schedule_token(source):
    """What a plan of ``source``'s events depends on besides the topology."""
    if isinstance(source, NetStim):
        return source._schedule_key
    return id(source)


# ----------------------------------------------------------------------
# Routes: what one layout's plan covers


@dataclass(frozen=True)
class _Route:
    """The scheduled rows without a rule that one synapse layout of one Cell receives.

    The plan's rows are those of ``call_ids`` in order (the order of
    ``connections._call_items(scheduled=True)``, which the Cell's input
    stage walks too); ``targets`` gives each row's runtime synapse row.
    """

    population: str
    cell: object
    layout_id: int
    call_ids: tuple[int, ...]
    call_rows: tuple[int, ...]
    sources: tuple
    source_index: tuple
    delay_ms: tuple
    targets: np.ndarray
    n_targets: int
    counted: bool


def _routes(populations) -> tuple[_Route, ...]:
    """Every layout of every Cell population that receives scheduled rows without a rule."""
    routes = []
    for name, population in populations.items():
        if population.kind != "cell":
            continue
        cell = population.cell
        runtime = cell._event_runtime()
        store = cell._get_synapse_store()
        items = cell.connections._call_items(scheduled=True)
        for layout in cell._event_layouts():
            if layout.id not in runtime.event_buffers or layout.placement_index is None:
                continue
            calls = [
                (call, view)
                for call, view in items
                if call.plasticity is None and store.layout_id(str(view.synapse_type[0])) == int(layout.id)
            ]
            if not calls:
                continue
            template = runtime.get_event_buffer(layout.id)
            if np.ndim(template) != 1:  # pragma: no cover - event buffers are one row per synapse
                raise NotImplementedError("scheduled_delivery needs one-dimensional event buffers.")
            routes.append(
                _Route(
                    population=name,
                    cell=cell,
                    layout_id=int(layout.id),
                    call_ids=tuple(int(call.id) for call, _ in calls),
                    call_rows=tuple(len(view) for _, view in calls),
                    sources=tuple(view.source for _, view in calls),
                    source_index=tuple(np.asarray(view.source_index, dtype=np.int64) for _, view in calls),
                    delay_ms=tuple(np.asarray(view.delay.to_decimal(u.ms), dtype=np.float64) for _, view in calls),
                    targets=np.concatenate([store.runtime_rows(view.synapse_id).astype(np.int32) for _, view in calls]),
                    n_targets=int(np.shape(template)[0]),
                    counted=cell._event_count_state(layout.id) is not None,
                )
            )
    return tuple(routes)


# ----------------------------------------------------------------------
# Plans


def _power_of_two(value: int) -> int:
    return 1 << max(int(value) - 1, 0).bit_length()


def _compress(steps: np.ndarray, rows: np.ndarray, start: int, n_steps: int):
    """Records ``(step, row)`` in step-then-row order with multiplicities, and the per-step counts."""
    if steps.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, np.zeros(n_steps, dtype=np.int64)
    order = np.lexsort((rows, steps))
    steps, rows = steps[order], rows[order]
    begins = np.flatnonzero(np.r_[True, (np.diff(steps) != 0) | (np.diff(rows) != 0)])
    multiplicity = np.diff(np.r_[begins, steps.size])
    record_steps = steps[begins]
    per_step = np.bincount(record_steps - start, minlength=n_steps)
    return record_steps, rows[begins], multiplicity, per_step


def _layout_arrays(method, route, steps, rows, start, n_steps):
    """Host arrays of one layout's plan for the steps ``[start, start + n_steps)``, and its statistics."""
    record_steps, record_rows, multiplicity, per_step = _compress(steps, rows, start, n_steps)
    records = int(record_rows.size)
    width = max(1, int(per_step.max(initial=0)))
    stats = {"arrivals": int(steps.size), "records": records, "width": width}
    if records + BLOCK_SIZE >= _INT32_MAX:
        raise ValueError(f"scheduled_delivery plan of {records} records exceeds int32 indexing.")
    targets = route.targets
    start_array = np.asarray(start, dtype=np.int32)
    if method == "direct":
        capacity = _power_of_two(records + BLOCK_SIZE)
        ids = np.zeros(capacity, dtype=np.int32)
        ids[:records] = record_rows
        counts = np.zeros(capacity, dtype=np.int32)
        counts[:records] = multiplicity
        offsets = np.concatenate([[0], np.cumsum(per_step)]).astype(np.int32)
        arrays = (ids, counts, offsets, targets, start_array)
    else:
        stored = _power_of_two(width)
        nbytes = (2 * n_steps * stored + targets.size + 1) * 4
        ratio = n_steps * width / records if records else 1.0
        stats.update(stored_width=stored, padding_ratio=ratio)
        if nbytes > MAX_BYTES or ratio > MAX_PADDING_RATIO:
            raise PaddingLimitError(
                f'scheduled_delivery="padded" needs {nbytes} bytes and {ratio:.3g} slots per record for the '
                f"steps [{start}, {start + n_steps}) of population {route.population!r} (limits {MAX_BYTES} bytes, "
                f'ratio {MAX_PADDING_RATIO:g}); use scheduled_delivery="direct", which has no such limit.'
            )
        ids = np.zeros((n_steps, stored), dtype=np.int32)
        counts = np.zeros((n_steps, stored), dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(per_step)])
        lane = np.arange(records) - offsets[record_steps - start]
        ids[record_steps - start, lane] = record_rows
        counts[record_steps - start, lane] = multiplicity
        arrays = (ids, counts, targets, start_array)
    stats["array_bytes"] = int(sum(np.asarray(array).nbytes for array in arrays))
    return arrays, stats


@dataclass
class ScheduledPlans:
    """The plans of one Network: arrival tables, the routes, and the plans of the last windows.

    Built at the first scheduled run of an initialized Network; the
    topology is frozen then, and ``dt`` and the delay rounding are fixed
    by the runtime configuration. Weights are read on every step, so
    changing them never rebuilds a plan.
    """

    method: str
    routes: tuple[_Route, ...]
    dt_ms: float
    tables: dict = field(default_factory=dict)
    windows: collections.OrderedDict = field(default_factory=collections.OrderedDict)
    full: tuple | None = None
    #: Statistics of the plan built last: preparation time, records, bytes.
    preparation: dict = field(default_factory=dict)

    @classmethod
    def build(cls, method, populations, dt_ms) -> "ScheduledPlans":
        return cls(method=method, routes=_routes(populations), dt_ms=float(dt_ms))

    def _tokens(self) -> tuple:
        return tuple(tuple(_schedule_token(source) for source in route.sources) for route in self.routes)

    def _table(self, route: _Route, position: int) -> _ArrivalTable:
        source = route.sources[position]
        key = (route.population, route.layout_id, route.call_ids[position], _schedule_token(source))
        table = self.tables.get(key)
        if table is None:
            # A NetStim whose schedule grew has a new token: drop its old tables.
            stale = [k for k in self.tables if k[:3] == key[:3]]
            for k in stale:
                del self.tables[k]
            event_source, event_time_ms = _source_events(source)
            table = _ArrivalTable.build(
                event_source, event_time_ms, route.source_index[position], route.delay_ms[position], self.dt_ms
            )
            self.tables[key] = table
        return table

    def _route_records(self, route: _Route, start: int, stop: int):
        steps, rows = [], []
        offset = 0
        for position, n_rows in enumerate(route.call_rows):
            call_steps, call_rows = self._table(route, position).window(start, stop)
            steps.append(call_steps)
            rows.append(call_rows + offset)
            offset += n_rows
        return np.concatenate(steps), np.concatenate(rows)

    def window(self, start: int, n_steps: int) -> tuple:
        """Device arrays of the plans for the steps ``[start, start + n_steps)``, cached for the last windows."""
        key = (int(start), int(n_steps), self._tokens())
        arrays = self.windows.get(key)
        if arrays is not None:
            self.windows.move_to_end(key)
            return arrays
        arrays = _build_plans(self, int(start), int(n_steps))
        self.windows[key] = arrays
        while len(self.windows) > CACHE_SIZE:
            self.windows.popitem(last=False)
        return arrays

    def everything(self) -> tuple:
        """Device arrays of plans covering every event: steps ``[0, last arrival + 1)`` of each layout."""
        tokens = self._tokens()
        if self.full is not None and self.full[0] == tokens:
            return self.full[1]
        arrays = _build_plans(self, 0, None)
        self.full = (tokens, arrays)
        return arrays


def _build_plans(plans: ScheduledPlans, start: int, n_steps: int | None) -> tuple:
    """Build every route's plan on the host and upload it; ``n_steps=None`` covers every event."""
    began = time.perf_counter()
    host, stats = [], []
    for route in plans.routes:
        if n_steps is None:
            last = max(plans._table(route, position).last_step() for position in range(len(route.call_rows)))
            span = last + 1 if last >= 0 else 1
        else:
            span = n_steps
        steps, rows = plans._route_records(route, start, start + span)
        arrays, layout_stats = _layout_arrays(plans.method, route, steps, rows, start, span)
        host.append(arrays)
        stats.append(dict(layout_stats, population=route.population, layout_id=route.layout_id, n_steps=span))
    built = time.perf_counter()
    arrays = tuple(tuple(jnp.asarray(array, dtype=jnp.int32) for array in layout) for layout in host)
    jax.block_until_ready(arrays)
    plans.preparation = {
        "method": plans.method,
        "start_step": start,
        "build_s": built - began,
        "upload_s": time.perf_counter() - built,
        "array_bytes": sum(item["array_bytes"] for item in stats),
        "layouts": stats,
    }
    return arrays


def rule_sources(populations) -> set:
    """``id`` of every source a plasticity rule reads: its rows' source and its ``post`` signal."""
    ids = set()
    for population in populations.values():
        if population.kind != "cell":
            continue
        for call in population.cell._get_connection_store()._calls:
            binding = call.plasticity
            if binding is None:
                continue
            ids.add(id(call.source))
            post = getattr(binding, "post", None)
            if post is not None:
                ids.add(id(post.owner))
    return ids


def loop_schedule_key(populations, plans: ScheduledPlans) -> tuple:
    """Schedule tokens of the NetStims a scheduled loop still reads while it is traced.

    A plan is an argument of the loop, so the NetStims only planned rows
    read do not key it: a schedule that grows for a later window then
    reuses the compiled loop.
    """
    planned = {id(source) for route in plans.routes for source in route.sources}
    ruled = rule_sources(populations)
    keys, seen = [], set()
    for population in populations.values():
        source = population.model
        if not isinstance(source, NetStim) or id(source) in seen:
            continue
        seen.add(id(source))
        if id(source) not in planned or id(source) in ruled:
            keys.append(source._schedule_key)
    return tuple(keys)


# ----------------------------------------------------------------------
# Delivery inside the compiled loop


def _direct_op(n_targets: int, block_size: int):
    """The ``"direct"`` delivery, with a sparse transpose for reverse mode.

    ``arrays`` are ``(rows, multiplicities, offsets, targets, start)``:
    the records of step ``start + k`` are ``offsets[k]`` to
    ``offsets[k + 1]``. A step outside the window has none.
    """

    def consume(arrays, step, values, transpose=False):
        connections, multiplicity, offsets, targets, start = arrays
        n_steps = offsets.shape[0] - 1
        local = step - start
        inside = (local >= 0) & (local < n_steps)
        index = jnp.clip(local, 0, n_steps - 1)
        lo = jnp.where(inside, offsets[index], 0)
        hi = jnp.where(inside, offsets[index + 1], 0)
        size = targets.shape[0] if transpose else n_targets

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


def _padded(n_targets: int, arrays, step, values):
    """The ``"padded"`` delivery: one row of the ``(steps, width)`` matrices per step."""
    connections, multiplicity, targets, start = arrays
    n_steps = connections.shape[0]
    local = step - start
    index = jnp.clip(local, 0, n_steps - 1)
    inside = (local >= 0) & (local < n_steps)
    ids = connections[index]
    count = jnp.where(inside, multiplicity[index], 0)
    return jnp.zeros(n_targets, dtype=values.dtype).at[targets[ids]].add(values[ids] * count)


@dataclass(frozen=True)
class _LayoutInput:
    """One layout's plan inside a traced loop: its route and its (traced) arrays."""

    method: str
    route: _Route
    arrays: tuple

    def _deliver(self, values, step):
        if self.method == "direct":
            return _direct_op(self.route.n_targets, BLOCK_SIZE)(self.arrays, step, values)
        return _padded(self.route.n_targets, self.arrays, step, values)

    def deliver(self, output, counts, weights, *, t, dt):
        """Add this step's planned arrivals to ``output`` (and their numbers to ``counts``).

        ``weights`` holds ``(call id, weight in the buffer's unit)`` for
        the layout's scheduled calls without a rule, in input-stage order.
        """
        call_ids = tuple(call_id for call_id, _ in weights)
        if call_ids != self.route.call_ids:  # pragma: no cover - both walk ``_call_items`` in order
            raise RuntimeError(f"scheduled plan covers calls {self.route.call_ids}, the input stage {call_ids}.")
        values = jnp.concatenate(
            [
                jnp.broadcast_to(jnp.asarray(weight), (n_rows,)).reshape(-1)
                for (_, weight), n_rows in zip(weights, self.route.call_rows)
            ]
        )
        step = _boundary_step(t, dt)
        output = output + self._deliver(values, step).astype(output.dtype)
        if counts is not None:
            ones = jnp.ones(values.shape, dtype=counts.dtype)
            counts = counts + jax.lax.stop_gradient(self._deliver(ones, step))
        return output, counts


@contextlib.contextmanager
def scheduled_inputs(method: str, routes: tuple, arrays: tuple):
    """Context in which the layouts of ``routes`` take their scheduled input from ``arrays``."""
    plans = {
        (id(route.cell), route.layout_id): _LayoutInput(method, route, layout_arrays)
        for route, layout_arrays in zip(routes, arrays)
    }
    with brainstate.environ.context(**{_PLANS_KEY: plans}):
        yield


def layout_input(cell, layout_id) -> _LayoutInput | None:
    """The plan of ``cell``'s layout inside a scheduled run loop, else ``None`` (the default path)."""
    plans = brainstate.environ.get(_PLANS_KEY, None)
    if plans is None:
        return None
    return plans.get((id(cell), int(layout_id)))
