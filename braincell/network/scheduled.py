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
  slots the records need. The ratio is not checked for a window whose
  plan takes no more bytes than one the Network already built for the
  layout (CELL-53): a sparse later window, such as the tail of a
  schedule, then runs in the storage an earlier window was accepted in.

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
    raises before it starts. ``"direct"`` has no such limit. A window whose
    plan is no larger than one the Network already built for the same
    layout is not held to the ratio, only to the bytes.
    """


def normalize_scheduled_delivery(value, *, prefix: str):
    """Return ``value`` if it is ``None``, ``"direct"`` or ``"padded"``; raise ``ValueError`` otherwise."""
    if value is None or (isinstance(value, str) and value in METHODS):
        return value
    raise ValueError(f"{prefix} scheduled_delivery must be None, 'direct', or 'padded', got {value!r}.")


# ----------------------------------------------------------------------
# Arrivals


@dataclass(frozen=True)
class _Groups:
    """The rows of one ``connect`` call grouped by source and delay.

    The rows of a group share their arrivals, as in the default path's
    ``_ArrivalPlan``; a window finds each group's arrivals once and
    expands them to the group's rows.
    """

    source: np.ndarray
    delay_ms: np.ndarray
    offsets: np.ndarray
    rows: np.ndarray

    @classmethod
    def build(cls, source_index, delay_ms) -> "_Groups":
        rows = np.asarray(source_index, dtype=np.int64).reshape(-1)
        row_delay = np.broadcast_to(np.asarray(delay_ms, dtype=np.float64), rows.shape)
        delay_bits = row_delay.view(np.uint64).astype(np.int64)
        keys, row_group = np.unique(np.stack([rows, delay_bits], axis=1), axis=0, return_inverse=True)
        row_group = row_group.reshape(-1)
        _, first_row = np.unique(row_group, return_index=True)
        offsets = np.concatenate([[0], np.cumsum(np.bincount(row_group, minlength=len(keys)))])
        return cls(keys[:, 0], row_delay[first_row], offsets, np.argsort(row_group, kind="stable").astype(np.int64))

    def expand(self, steps: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, rows)``: each ``(step, group)`` arrival repeated for every row of the group."""
        sizes = self.offsets[group + 1] - self.offsets[group]
        total = int(sizes.sum())
        first = np.repeat(self.offsets[group], sizes)
        within = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        return np.repeat(steps, sizes), self.rows[first + within]


@dataclass(frozen=True)
class _SequenceArrivals:
    """The arrivals of an EventSequence call: one ``(step, group)`` entry per event, sorted by step.

    An EventSequence never changes, so the table is built once.
    """

    groups: _Groups
    steps: np.ndarray
    group: np.ndarray

    @classmethod
    def build(cls, groups: _Groups, source: EventSequence, dt_ms: float, handover=None) -> "_SequenceArrivals":
        event_source = np.asarray(source.events.source_index, dtype=np.int64)
        event_ms = np.asarray(source.events.time.to_decimal(u.ms), dtype=np.float64)
        order = np.argsort(event_source, kind="stable")
        sorted_source, sorted_ms = event_source[order], event_ms[order]
        lo = np.searchsorted(sorted_source, groups.source, side="left")
        per_group = np.searchsorted(sorted_source, groups.source, side="right") - lo
        pair_group = np.repeat(np.arange(groups.source.size, dtype=np.int64), per_group)
        pair_event = np.arange(int(per_group.sum())) + np.repeat(lo - (np.cumsum(per_group) - per_group), per_group)
        steps = _host_arrival_steps(sorted_ms[pair_event], groups.delay_ms[pair_group], dt_ms, handover)
        steps = steps.astype(np.int64)
        by_step = np.argsort(steps, kind="stable")
        return cls(groups, steps[by_step], pair_group[by_step])

    def window(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, rows)`` of the arrivals at the steps ``[start, stop)``."""
        lo, hi = np.searchsorted(self.steps, [start, stop], side="left")
        return self.groups.expand(self.steps[lo:hi], self.group[lo:hi])

    def every(self, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, rows)`` of every arrival at the steps ``[0, stop)``."""
        return self.window(0, stop)


class _ScheduleArrivals:
    """The arrivals of a NetStim call, read from its generated ``(source, event)`` schedule.

    A source's events are in time order along its row, so each group's
    arrivals are too. A window binary-searches each group's first event
    arriving at or after its start, then reads the next events of every
    group until they arrive at or after its stop: the cost is that of the
    window's arrivals, not of the whole schedule, so it does not grow with
    the time already run. The column a window stops at is kept and tried
    first by the next window (consecutive segments), also across a schedule
    that grew, since a narrower schedule is a prefix of a wider one; it is
    checked before use.
    """

    def __init__(self, groups: _Groups, source: NetStim, dt_ms: float, cursor=None, handover=None):
        self.groups = groups
        self.times = source._event_times_ms
        self.count = np.count_nonzero(source._event_mask, axis=1)[groups.source]
        self.dt_ms = dt_ms
        #: ``(step, column of each group's first arrival at or after it)``.
        self.cursor = cursor
        #: What earlier runs at another ``dt`` took (CELL-57): the steps stay in order.
        self.handover = handover

    def _steps(self, group: np.ndarray, column: np.ndarray) -> np.ndarray:
        times = self.times[self.groups.source[group], column]
        return _host_arrival_steps(times, self.groups.delay_ms[group], self.dt_ms, self.handover).astype(np.int64)

    def _first_columns(self, start: int) -> np.ndarray:
        """The column of each group's first arrival at or after ``start`` (its event count if none)."""
        lo = np.zeros(self.count.shape, dtype=np.int64)
        hi = self.count.astype(np.int64)
        if start <= 0:
            return lo
        if self.cursor is not None and self.cursor[0] == start:
            guess = np.minimum(self.cursor[1], hi)
            ok = np.ones(guess.shape, dtype=bool)
            before = np.flatnonzero(guess > 0)
            ok[before] = self._steps(before, guess[before] - 1) < start
            after = np.flatnonzero(ok & (guess < hi))
            ok[after] = self._steps(after, guess[after]) >= start
            lo[ok] = hi[ok] = guess[ok]
        todo = np.flatnonzero(lo < hi)
        while todo.size:
            mid = (lo[todo] + hi[todo]) // 2
            early = self._steps(todo, mid) < start
            lo[todo[early]] = mid[early] + 1
            hi[todo[~early]] = mid[~early]
            todo = todo[lo[todo] < hi[todo]]
        return lo

    def window(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, rows)`` of the arrivals at the steps ``[start, stop)``."""
        column = self._first_columns(start)
        steps, group = [], []
        active = np.flatnonzero(column < self.count)
        while active.size:
            arrival = self._steps(active, column[active])
            inside = arrival < stop
            active = active[inside]
            arrival = arrival[inside]
            # Only the arrivals an earlier run took are before a start of 0 or less.
            due = arrival >= start
            steps.append(arrival[due])
            group.append(active[due])
            column[active] += 1
            active = active[column[active] < self.count[active]]
        self.cursor = (stop, column)
        if not steps:
            return self.groups.expand(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        return self.groups.expand(np.concatenate(steps), np.concatenate(group))

    def every(self, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, rows)`` of every arrival at the steps ``[0, stop)``, read at once.

        :meth:`window` reads one column of every group per pass, which
        suits a window holding a few events of each group; a plan of the
        whole schedule (``prepare_run``) would pass once per event of its
        longest row. The arrivals are the same.
        """
        count = self.count.astype(np.int64)
        group = np.repeat(np.arange(count.size, dtype=np.int64), count)
        column = np.arange(group.size, dtype=np.int64) - np.repeat(np.cumsum(count) - count, count)
        arrival = self._steps(group, column)
        # Only the arrivals an earlier run took are before step 0.
        due = (arrival >= 0) & (arrival < stop)
        return self.groups.expand(arrival[due], group[due])


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


def _compress(steps: np.ndarray, rows: np.ndarray, start: int, n_steps: int | None):
    """Records ``(step, row)`` in step-then-row order with multiplicities, and the per-step counts.

    The counts are those of the steps ``[start, start + n_steps)``; with
    ``n_steps=None``, of the steps that have records only, which are
    returned last (``None`` otherwise).
    """
    if steps.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        if n_steps is None:
            return empty, empty, empty, empty, empty
        return empty, empty, empty, np.zeros(n_steps, dtype=np.int64), None
    order = np.lexsort((rows, steps))
    steps, rows = steps[order], rows[order]
    begins = np.flatnonzero(np.r_[True, (np.diff(steps) != 0) | (np.diff(rows) != 0)])
    multiplicity = np.diff(np.r_[begins, steps.size])
    record_steps = steps[begins]
    if n_steps is None:
        firsts = np.flatnonzero(np.r_[True, np.diff(record_steps) != 0])
        per_step = np.diff(np.r_[firsts, record_steps.size])
        return record_steps, rows[begins], multiplicity, per_step, record_steps[firsts]
    per_step = np.bincount(record_steps - start, minlength=n_steps)
    return record_steps, rows[begins], multiplicity, per_step, None


def _layout_arrays(method, route, steps, rows, start, n_steps, accepted_bytes=0):
    """Host arrays of one layout's plan for the steps ``[start, start + n_steps)``, and its statistics.

    ``n_steps=None`` stores only the steps that have arrivals (the plan
    of every event, CELL-65): the last array then holds those steps, in
    order, in place of ``start``. ``accepted_bytes`` is the largest
    ``"padded"`` plan built for the layout before: a plan no larger has
    no ratio limit.
    """
    record_steps, record_rows, multiplicity, per_step, stored_steps = _compress(steps, rows, start, n_steps)
    records = int(record_rows.size)
    width = max(1, int(per_step.max(initial=0)))
    stats = {"arrivals": int(steps.size), "records": records, "width": width}
    if records + BLOCK_SIZE >= _INT32_MAX:
        raise ValueError(f"scheduled_delivery plan of {records} records exceeds int32 indexing.")
    targets = route.targets
    if stored_steps is None:
        rows_of_plan = n_steps
        slot = record_steps - start
        last = np.asarray(start, dtype=np.int32)
        steps_text = f"steps [{start}, {start + n_steps})"
    else:
        rows_of_plan = int(stored_steps.size)
        slot = np.repeat(np.arange(rows_of_plan, dtype=np.int64), per_step)
        last = stored_steps.astype(np.int32)
        steps_text = f"{rows_of_plan} steps with arrivals"
    stats["stored_steps"] = rows_of_plan
    if method == "direct":
        capacity = _power_of_two(records + BLOCK_SIZE)
        ids = np.zeros(capacity, dtype=np.int32)
        ids[:records] = record_rows
        counts = np.zeros(capacity, dtype=np.int32)
        counts[:records] = multiplicity
        offsets = np.concatenate([[0], np.cumsum(per_step)]).astype(np.int32)
        arrays = (ids, counts, offsets, targets, last)
    else:
        stored = _power_of_two(width)
        nbytes = (2 * rows_of_plan * stored + targets.size + last.size) * 4
        ratio = rows_of_plan * width / records if records else 1.0
        stats.update(stored_width=stored, padding_ratio=ratio)
        if nbytes > MAX_BYTES or (ratio > MAX_PADDING_RATIO and nbytes > accepted_bytes):
            raise PaddingLimitError(
                f'scheduled_delivery="padded" needs {nbytes} bytes and {ratio:.3g} slots per record for the '
                f"{steps_text} of population {route.population!r} (limits {MAX_BYTES} bytes, "
                f'ratio {MAX_PADDING_RATIO:g}); use scheduled_delivery="direct", which has no such limit.'
            )
        ids = np.zeros((rows_of_plan, stored), dtype=np.int32)
        counts = np.zeros((rows_of_plan, stored), dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(per_step)])
        lane = np.arange(records) - offsets[slot]
        ids[slot, lane] = record_rows
        counts[slot, lane] = multiplicity
        arrays = (ids, counts, targets, last)
    stats["array_bytes"] = int(sum(np.asarray(array).nbytes for array in arrays))
    return arrays, stats


@dataclass
class ScheduledPlans:
    """The plans of one Network: the routes, their arrivals, and the plans of the last windows.

    Built at the first scheduled run of an initialized Network; the
    topology is frozen then, and ``dt`` and the delay rounding are fixed
    by the runtime configuration. Weights are read on every step, so
    changing them never rebuilds a plan.
    """

    method: str
    routes: tuple[_Route, ...]
    dt_ms: float
    #: ``(population, layout, call id)`` -> its rows grouped by source and delay.
    groups: dict = field(default_factory=dict)
    #: ``(population, layout, call id)`` -> ``(schedule token, arrivals)``.
    tables: dict = field(default_factory=dict)
    #: ``(population, layout id)`` -> bytes of the largest ``"padded"`` plan built for it (CELL-53).
    accepted: dict = field(default_factory=dict)
    windows: collections.OrderedDict = field(default_factory=collections.OrderedDict)
    full: tuple | None = None
    #: Statistics of the plan built last: preparation time, records, bytes.
    preparation: dict = field(default_factory=dict)
    #: Population -> the ``ArrivalHandover`` of its Cell's runs before (CELL-57).
    handovers: dict = field(default_factory=dict)

    @classmethod
    def build(cls, method, populations, dt_ms) -> "ScheduledPlans":
        return cls(method=method, routes=_routes(populations), dt_ms=float(dt_ms))

    def _tokens(self) -> tuple:
        tokens = tuple(tuple(_schedule_token(source) for source in route.sources) for route in self.routes)
        if not self.handovers:
            return tokens
        return tokens, tuple(sorted(self.handovers.items()))

    def _table(self, route: _Route, position: int):
        """The arrivals of one call of ``route`` for its source's current schedule."""
        source = route.sources[position]
        key = (route.population, route.layout_id, route.call_ids[position])
        handover = self.handovers.get(route.population)
        token = (_schedule_token(source), handover)
        cached = self.tables.get(key)
        if cached is not None and cached[0] == token:
            return cached[1]
        groups = self.groups.get(key)
        if groups is None:
            groups = _Groups.build(route.source_index[position], route.delay_ms[position])
            self.groups[key] = groups
        if isinstance(source, NetStim):
            # A schedule that grew keeps its first events: reuse where the last window stopped.
            cursor = cached[1].cursor if cached is not None and cached[0][1] == handover else None
            table = _ScheduleArrivals(groups, source, self.dt_ms, cursor, handover)
        elif isinstance(source, EventSequence):
            table = _SequenceArrivals.build(groups, source, self.dt_ms, handover)
        else:  # pragma: no cover - a Network registers no other scheduled source
            raise TypeError(
                f"scheduled_delivery supports NetStim and EventSequence sources, not {type(source).__name__}."
            )
        self.tables[key] = (token, table)
        return table

    def _route_records(self, route: _Route, start: int, stop: int | None):
        """``(steps, rows)`` of ``route``'s arrivals at the steps ``[start, stop)``; ``stop=None``: all of them."""
        steps, rows = [], []
        offset = 0
        for position, n_rows in enumerate(route.call_rows):
            table = self._table(route, position)
            # Arrivals past the step limit never come (``_host_arrival_steps`` clips them to it).
            call_steps, call_rows = table.every(_MAX_STEP) if stop is None else table.window(start, stop)
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
        """Device arrays of plans covering every event, holding only the steps that have arrivals (CELL-65)."""
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
        steps, rows = plans._route_records(route, start, None if n_steps is None else start + n_steps)
        accepted = plans.accepted.get((route.population, route.layout_id), 0)
        arrays, layout_stats = _layout_arrays(plans.method, route, steps, rows, start, n_steps, accepted)
        # The steps a plan of every event spans, to its last arrival.
        span = (int(steps.max()) + 1 if steps.size else 1) if n_steps is None else n_steps
        host.append(arrays)
        stats.append(dict(layout_stats, population=route.population, layout_id=route.layout_id, n_steps=span))
    if plans.method == "padded":
        # Every layout is planned: only now does the run take these plans.
        for item in stats:
            key = (item["population"], item["layout_id"])
            plans.accepted[key] = max(plans.accepted.get(key, 0), item["array_bytes"])
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


def _stored_row(steps, step):
    """``(row, inside)`` of a plan of every event: the row of ``step`` among the stored ``steps``, and whether it is one.

    Such a plan (CELL-65) holds only the steps that have arrivals, in
    order, so a step costs one binary search, unrolled: a compiled
    ``while`` costs a device round trip per iteration on GPU.
    """
    row = jnp.minimum(jnp.searchsorted(steps, step, side="left", method="scan_unrolled"), steps.shape[0] - 1)
    return row, steps[row] == step


def _direct_op(n_targets: int, block_size: int):
    """The ``"direct"`` delivery, with a sparse transpose for reverse mode.

    ``arrays`` are ``(rows, multiplicities, offsets, targets, start)``:
    the records of step ``start + k`` are ``offsets[k]`` to
    ``offsets[k + 1]``. A step outside the window has none. In a plan of
    every event ``start`` is the array of the steps it holds and ``k``
    the position of a step in it (:func:`_stored_row`).
    """

    def consume(arrays, step, values, transpose=False):
        connections, multiplicity, offsets, targets, start = arrays
        n_steps = offsets.shape[0] - 1
        size = targets.shape[0] if transpose else n_targets
        if start.ndim:
            if n_steps == 0:  # a plan of every event, which has none
                return jnp.zeros(size, dtype=values.dtype)
            index, inside = _stored_row(start, step)
        else:
            local = step - start
            inside = (local >= 0) & (local < n_steps)
            index = jnp.clip(local, 0, n_steps - 1)
        lo = jnp.where(inside, offsets[index], 0)
        hi = jnp.where(inside, offsets[index + 1], 0)

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
    """The ``"padded"`` delivery: one row of the ``(steps, width)`` matrices per step.

    In a plan of every event ``start`` is the array of the steps it holds,
    one per row (:func:`_stored_row`).
    """
    connections, multiplicity, targets, start = arrays
    n_steps = connections.shape[0]
    if start.ndim:
        if n_steps == 0:  # a plan of every event, which has none
            return jnp.zeros(n_targets, dtype=values.dtype)
        index, inside = _stored_row(start, step)
    else:
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
