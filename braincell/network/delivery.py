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

"""Runtime synaptic event-delivery helpers.

This module owns the step-time path that moves population spikes into
postsynaptic synapse event buffers. The input is a lowered
``ConnectionBlock`` table, where each row is already a materialized synaptic
contact. The delivery layer preserves per-contact delays for scatter routing,
keeps one history of source event counts per target synapse layout, writes
due arrivals weighted with the weight at delivery, and records the events of
each step into the history. Backends may group contacts internally without
changing queue ownership.

The module does not define synapse models and does not perform topology
lowering. Synapse layout binding, default weight lookup, and delay
quantization belong to :mod:`braincell.network.lowering`.
"""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.network.lowering import ConnectionBlock


@dataclass(frozen=True)
class DeliveryBlock:
    """Runtime delivery routes for one lowered connection block.

    Parameters
    ----------
    source : ConnectionBlock
        Lowered connection block that owns the original synapse layout and
        population metadata.
    delay_steps : int or ndarray of int
        One fixed delay for a grouped backend block, or one delay per contact
        for the canonical scatter representation.
    pre_index : ndarray of int, shape ``(n_contact,)``
        Presynaptic population index for each contact in this delay group.
    flat_target_index : ndarray of int, shape ``(n_contact,)``
        Slot in the postsynaptic synapse layout for each contact.
    weight : object
        Per-contact event payload sliced from ``source.weight`` for this delay
        group. May be a :mod:`brainunit` quantity or an array-like object.

    Notes
    -----
    ``DeliveryBlock`` is static within one run setup. The mutable delayed event
    state is stored separately in :class:`DeliveryState`.
    """

    source: ConnectionBlock
    delay_steps: int | np.ndarray
    pre_index: np.ndarray
    flat_target_index: np.ndarray
    weight: object
    weight_indices: object = None
    #: The consecutive lowered blocks of one queue this route concatenates
    #: (CELL-18b); empty for a route over ``source`` alone.
    members: tuple[ConnectionBlock, ...] = ()

    def __getattribute__(self, name):
        if name == "weight":
            members = object.__getattribute__(self, "members")
            if members:
                return _concatenate_weights(members)
            source = object.__getattribute__(self, "source")
            if getattr(source, "weight_getter", None) is not None:
                indices = object.__getattribute__(self, "weight_indices")
                value = source.weight
                return value if indices is None else value[indices]
        return object.__getattribute__(self, name)


@dataclass(frozen=True)
class _QueuePlan:
    """How one queue keeps and sums its delayed events (CELL-69).

    The queue keeps the event counts of the source members its delayed
    contacts read, at the last ``depth`` boundaries: ``sources[i]`` gives
    ``columns[i]`` (its member ids) to each history row, in that order.

    With one delay per contact (the scatter path), ``lags``, ``columns_read``,
    ``weight_index`` and ``targets`` list the delayed contacts in the order a
    queue of weighted events summed them: longest delay first, then by route
    and by contact. ``depth`` is the longest delay in steps plus one: a
    contact with ``k`` delay steps reads, at the start of step ``m + k +
    1``, the events counted at the end of step ``m``, history row
    ``(cursor + depth - 1 - k) % depth`` (``lags`` holds ``depth - 1 - k``,
    CELL-75), column
    ``columns_read``, the weight ``concatenate(route weights)[weight_index]``
    (``None``: in order), and adds to synapse row ``targets``.

    With one delay per route (the ``brainevent`` path), ``scalar`` lists
    ``(block index, depth - 1 - k, first column, size)`` longest delay first;
    each route reads its source's whole row.
    """

    depth: int = 0
    sources: tuple = ()
    columns: tuple = ()
    weight_blocks: tuple[int, ...] = ()
    weight_index: np.ndarray | None = None
    lags: np.ndarray | None = None
    columns_read: np.ndarray | None = None
    targets: np.ndarray | None = None
    scalar: tuple = ()


@dataclass(frozen=True)
class DeliveryState:
    """Mutable runtime buffers and backend ops for event delivery.

    Parameters
    ----------
    histories : tuple
        One event history per queue with delayed contacts, ``None`` for
        the others: shape ``(depth, width)``, the counts of the source
        members its delayed contacts read at the last ``depth`` boundaries,
        ``depth`` being the queue's longest delay in steps plus one
        (CELL-69, CELL-75).
    history_cursors : tuple
        Per-queue history row the current step writes, ``None`` for a
        queue without a history.
    delivery_ops : tuple
        Backend-specific functions mapping pre spikes to flattened events.
    block_queue_indices : tuple of int
        The queue of each delivery block.
    queue_keys : tuple
        ``(post_population, layout_id)`` of each queue, one per target
        synapse layout.
    queue_sources : tuple of ConnectionBlock
        The first lowered block of each queue: its weight gives the arrival
        unit and dtype.
    queue_plans : tuple of _QueuePlan
        How each queue keeps and sums its delayed events.
    count_dtypes : tuple
        The event-count dtype of each queue whose target layout declares
        ``ScalarEventInput(aggregation="sum_count")``, ``None`` for every
        other queue. Empty when no queue counts (CELL-12b).

    Notes
    -----
    ``DeliveryState`` belongs to the initialized Network runtime. It is reused
    across consecutive :meth:`Network.run` calls so pending events survive a
    split run, and is cleared only by :meth:`Network.reset_state`. The
    histories keep event counts, not weighted events: a delayed event is
    weighted when it arrives, with the weight then (CELL-69).
    """

    histories: tuple
    history_cursors: tuple
    delivery_ops: tuple
    block_queue_indices: tuple[int, ...]
    queue_keys: tuple[tuple[str, int], ...]
    queue_sources: tuple[ConnectionBlock, ...]
    queue_plans: tuple = ()
    count_dtypes: tuple = ()

    def count_dtype(self, queue_index: int):
        """The event-count dtype of one queue, or ``None`` if its target layout does not count."""
        return self.count_dtypes[queue_index] if self.count_dtypes else None


def route_source_event(block: DeliveryBlock) -> object:
    """Read current event values for one live route.

    Parameters
    ----------
    block : DeliveryBlock
        Delivery block whose lowered source names the live
        :class:`~braincell.network.event.EventSource`.

    Returns
    -------
    object
        One event count per source-population member.
    """
    source = block.source.event_source
    return source.current_event_count(np.arange(source.size, dtype=np.int32))


def delivery_blocks(
    blocks: tuple[ConnectionBlock, ...],
    *,
    group_by_delay: bool = True,
) -> tuple[DeliveryBlock, ...]:
    """Build runtime routes, optionally grouping contacts by delay.

    Parameters
    ----------
    blocks : tuple of ConnectionBlock
        Lowered connection blocks. Each block may contain heterogeneous
        per-contact ``delay_steps``.
    group_by_delay : bool, default True
        Split each input block into one fixed-delay block per distinct
        delay. The ``brainevent`` backend needs a scalar delay per block;
        the scatter backend handles the per-contact vector directly.

    Returns
    -------
    tuple of DeliveryBlock
        One block per run of input connections that can share a route
        (see Notes) when ``group_by_delay=False``; otherwise one fixed-delay
        block per input delay group.

    Notes
    -----
    Delay grouping is an operator-lowering choice only. Every returned block
    targeting the same synapse layout writes one shared queue.

    Without grouping, consecutive blocks of one queue (the blocks for one
    ``(post_population, layout_id)``, in order) become one route when they
    read the same event source, carry weights of the same unit and dtype,
    and reach different synapses (CELL-18b). A step then runs one gather and
    one scatter for them instead of one per block. Each synapse of such a
    route receives contacts from one member only, in that member's order,
    so every queue slot and zero-delay drive gets the same additions in the
    same order as from one route per block, on any backend.
    """
    if not group_by_delay:
        return _merged_delivery_blocks(blocks)
    delivery = []
    for block in blocks:
        delay_steps = np.asarray(block.delay_steps, dtype=np.int32)
        if delay_steps.size == 0:
            continue
        # One stable sort lists each delay's contacts in their original order,
        # as a scan per delay did; that scan cost O(contacts x delays) (CELL-18g).
        order = np.argsort(delay_steps, kind="stable")
        delays, starts = np.unique(delay_steps[order], return_index=True)
        weight = block.weight
        for delay_step, contact_indices in zip(delays.tolist(), np.split(order, starts[1:])):
            delivery.append(
                DeliveryBlock(
                    source=block,
                    delay_steps=int(delay_step),
                    pre_index=block.pre_index[contact_indices],
                    flat_target_index=block.synapse_index[contact_indices].astype(np.int32, copy=False),
                    weight=slice_weight(weight, contact_indices),
                    weight_indices=contact_indices,
                )
            )
    return tuple(delivery)


def _merged_delivery_blocks(blocks: tuple[ConnectionBlock, ...]) -> tuple[DeliveryBlock, ...]:
    """One route per run of consecutive blocks of a queue that may share one."""
    targets = [np.asarray(block.synapse_index, dtype=np.int32) for block in blocks]
    n_targets: dict[tuple[str, int], int] = {}
    for block, target in zip(blocks, targets):
        key = (block.post_population, int(block.layout_id))
        n_targets[key] = max(n_targets.get(key, 0), int(target.max()) + 1 if target.size else 0)

    runs: list[list[int]] = []
    open_run: dict[tuple[str, int], tuple[tuple, np.ndarray, list[int]]] = {}
    for position, (block, target) in enumerate(zip(blocks, targets)):
        key = (block.post_population, int(block.layout_id))
        signature = (id(block.event_source), _weight_signature(block.weight))
        current = open_run.get(key)
        if current is not None and current[0] == signature and not current[1][target].any():
            current[1][target] = True
            current[2].append(position)
            continue
        reached = np.zeros(n_targets[key], dtype=bool)
        reached[target] = True
        run = [position]
        runs.append(run)
        open_run[key] = (signature, reached, run)

    delivery = []
    for run in runs:
        first = blocks[run[0]]
        if len(run) == 1:
            delivery.append(
                DeliveryBlock(
                    source=first,
                    delay_steps=np.asarray(first.delay_steps, dtype=np.int32),
                    pre_index=np.asarray(first.pre_index, dtype=np.int32),
                    flat_target_index=targets[run[0]],
                    weight=first.weight,
                )
            )
            continue
        members = tuple(blocks[position] for position in run)
        delivery.append(
            DeliveryBlock(
                source=first,
                delay_steps=np.concatenate([np.asarray(block.delay_steps, dtype=np.int32) for block in members]),
                pre_index=np.concatenate([np.asarray(block.pre_index, dtype=np.int32) for block in members]),
                flat_target_index=np.concatenate([targets[position] for position in run]),
                weight=None,
                members=members,
            )
        )
    return tuple(delivery)


def _weight_signature(weight) -> tuple:
    """Routes concatenate weights only when no value changes on the way."""
    if weight is None:
        return (object(),)
    unit = u.get_unit(weight) if isinstance(weight, u.Quantity) else None
    return (type(weight) is u.Quantity, unit, np.dtype(u.get_mantissa(weight).dtype))


def _concatenate_weights(members: tuple[ConnectionBlock, ...]):
    """The members' per-contact weights, read now, end to end."""
    weights = [member.weight for member in members]
    mantissas = [
        jnp.broadcast_to(jnp.asarray(u.get_mantissa(weight)), (len(member.pre_index),))
        for member, weight in zip(members, weights)
    ]
    if isinstance(weights[0], u.Quantity):
        return u.Quantity(jnp.concatenate(mantissas), u.get_unit(weights[0]))
    return jnp.concatenate(mantissas)


def slice_weight(weight, indices: np.ndarray):
    """Return per-contact weights selected by integer indices.

    Parameters
    ----------
    weight : object
        Per-contact weight array or :mod:`brainunit` quantity.
    indices : ndarray of int
        Contact indices to select.

    Returns
    -------
    object
        Selected weight values, preserving units when ``weight`` is a quantity.
    """
    if isinstance(weight, u.Quantity):
        return u.Quantity(weight.mantissa[indices], weight.unit)
    return np.asarray(weight)[indices]


def zero_arrival(block: ConnectionBlock):
    """Return a zero arrival buffer for one lowered connection block.

    Parameters
    ----------
    block : ConnectionBlock
        Lowered block whose weight dtype/unit and synapse layout size define
        the arrival buffer.

    Returns
    -------
    object
        Zero event vector with shape ``(block.n_active,)``.
    """
    return zeros_like(block.weight, shape=(block.n_active,))


def zeros_like(value, *, shape: tuple[int, ...]):
    """Return zeros of ``shape`` carrying ``value``'s dtype and unit.

    Every weighted buffer in this module -- arrival vectors and the scatter
    accumulator inside a delivery op -- is shaped from an event
    payload that may or may not be a :mod:`brainunit` quantity.

    Parameters
    ----------
    value : object
        Example event payload used to infer dtype and unit.
    shape : tuple of int
        Shape of the returned buffer.

    Returns
    -------
    object
        Zero buffer of ``shape``, a :class:`brainunit.Quantity` when
        ``value`` is one.
    """
    if isinstance(value, u.Quantity):
        return u.math.zeros_like(value, shape=shape)
    return jnp.zeros(shape, dtype=jnp.asarray(value).dtype)


def create_delivery_state(
    blocks: tuple[DeliveryBlock, ...],
    *,
    delivery_ops: tuple,
    counted: frozenset = frozenset(),
) -> DeliveryState:
    """Create persistent mutable state for network event delivery.

    Parameters
    ----------
    blocks : tuple of DeliveryBlock
        Static delivery blocks for the run setup.
    delivery_ops : tuple
        Backend-specific event delivery callables, one per block.
    counted : frozenset of tuple, optional
        The ``(post_population, layout_id)`` queues whose target layout
        declares ``ScalarEventInput(aggregation="sum_count")``; their
        arrivals also carry the per-row event count (CELL-12b).

    Returns
    -------
    DeliveryState
        Event histories, cursors, and delivery operators used during runs.

    Raises
    ------
    ValueError
        If ``delivery_ops`` does not have one operator per delivery block.

    Notes
    -----
    Each queue with delayed contacts keeps one history of the event counts
    of the source members those contacts read, one row deeper than its
    longest delay in steps (CELL-75): ``depth x members`` values in the
    sources' event dtype, where a queue of weighted events, before CELL-69,
    held ``depth x synapse rows``. A route with one delay per contact (the
    scatter path) keeps only the members its delayed contacts read; a
    route with one delay (the ``brainevent`` path) its whole source.

    The history therefore grows with the distinct source members a target
    synapse layout receives from, not with its synapse rows. In float64,
    with delays up to 80 steps (81 rows): 1e6 rows from 1e6 sources onto
    1e4 shared synapses keep 632 MB, where the weighted queue held 6.5 MB;
    1e6 rows from 900 sources onto 1e6 synapses keep 0.59 MB, where it held
    648 MB.
    """
    if len(delivery_ops) != len(blocks):
        raise ValueError(
            f"delivery_ops must have the same length as delivery blocks; got {len(delivery_ops)!r} and {len(blocks)!r}."
        )
    queue_index_by_key: dict[tuple[str, int], int] = {}
    queue_keys: list[tuple[str, int]] = []
    queue_sources: list[ConnectionBlock] = []
    block_queue_indices: list[int] = []
    delayed: list[list[tuple[int, np.ndarray, np.ndarray | None]]] = []
    for block_index, block in enumerate(blocks):
        key = (block.source.post_population, int(block.source.layout_id))
        queue_index = queue_index_by_key.get(key)
        if queue_index is None:
            queue_index = len(queue_keys)
            queue_index_by_key[key] = queue_index
            queue_keys.append(key)
            queue_sources.append(block.source)
            delayed.append([])
        block_queue_indices.append(queue_index)
        delay_steps = np.asarray(block.delay_steps, dtype=np.int32)
        if delay_steps.ndim == 0:
            if int(delay_steps) > 0:
                delayed[queue_index].append((block_index, delay_steps, None))
            continue
        contacts = np.flatnonzero(delay_steps > 0)
        if contacts.size:
            delayed[queue_index].append((block_index, delay_steps[contacts], contacts))
    queue_plans = tuple(_queue_plan(parts, blocks) for parts in delayed)
    histories = []
    for plan in queue_plans:
        if not plan.depth:
            histories.append(None)
            continue
        dtypes = [
            jax.eval_shape(lambda source=source, ids=ids: jnp.asarray(source.current_event_count(ids))).dtype
            for source, ids in zip(plan.sources, plan.columns)
        ]
        width = sum(ids.size for ids in plan.columns)
        histories.append(brainstate.ShortTermState(jnp.zeros((plan.depth, width), dtype=jnp.result_type(*dtypes))))
    history_cursors = tuple(
        None if history is None else brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32)) for history in histories
    )
    count_dtypes = ()
    if any(key in counted for key in queue_keys):
        count_dtypes = tuple(
            jnp.asarray(u.get_mantissa(source.weight)).dtype if key in counted else None
            for key, source in zip(queue_keys, queue_sources)
        )
    return DeliveryState(
        histories=tuple(histories),
        history_cursors=history_cursors,
        delivery_ops=delivery_ops,
        block_queue_indices=tuple(block_queue_indices),
        queue_keys=tuple(queue_keys),
        queue_sources=tuple(queue_sources),
        queue_plans=queue_plans,
        count_dtypes=count_dtypes,
    )


def _queue_plan(parts: list[tuple[int, np.ndarray, np.ndarray | None]], blocks) -> _QueuePlan:
    """The history columns of one queue and its delayed contacts in the order a queue of weighted events summed them."""
    if not parts:
        return _QueuePlan()
    # A contact of k steps reads the events of k + 1 boundaries before (CELL-75).
    depth = max(int(np.max(delay)) for _, delay, _ in parts) + 1
    sources: list = []
    reads: list[list[np.ndarray]] = []
    whole: list[bool] = []
    index_of: dict[int, int] = {}
    for block_index, delay, contacts in parts:
        source = blocks[block_index].source.event_source
        position = index_of.get(id(source))
        if position is None:
            position = index_of[id(source)] = len(sources)
            sources.append(source)
            reads.append([])
            whole.append(False)
        if contacts is None:
            whole[position] = True
        else:
            reads[position].append(np.asarray(blocks[block_index].pre_index, dtype=np.int64)[contacts])
    columns = tuple(
        np.arange(int(source.size), dtype=np.int32) if is_whole else np.unique(np.concatenate(read)).astype(np.int32)
        for source, read, is_whole in zip(sources, reads, whole)
    )
    starts = np.concatenate([[0], np.cumsum([ids.size for ids in columns])]).astype(np.int64)
    if any(contacts is None for _, _, contacts in parts):
        if not all(contacts is None for _, _, contacts in parts):
            raise ValueError("A queue mixes scalar-delay and per-contact-delay routes.")
        scalar = []
        for block_index, delay, _ in parts:
            position = index_of[id(blocks[block_index].source.event_source)]
            scalar.append((block_index, depth - 1 - int(delay), int(starts[position]), int(columns[position].size)))
        # Stable: routes of one delay keep their order.
        scalar.sort(key=lambda item: item[1])
        return _QueuePlan(depth=depth, sources=tuple(sources), columns=columns, scalar=tuple(scalar))
    delays, read, targets, weight_index, weight_blocks = [], [], [], [], []
    offset = 0
    for block_index, delay, contacts in parts:
        block = blocks[block_index]
        position = index_of[id(block.source.event_source)]
        pre = np.asarray(block.pre_index, dtype=np.int64)[contacts]
        delays.append(delay)
        read.append(starts[position] + np.searchsorted(columns[position], pre))
        targets.append(np.asarray(block.flat_target_index, dtype=np.int64)[contacts])
        weight_index.append(offset + contacts)
        weight_blocks.append(block_index)
        offset += len(block.pre_index)
    delays = np.concatenate(delays)
    order = np.argsort(-delays, kind="stable")
    weight_index = np.concatenate(weight_index)[order]
    return _QueuePlan(
        depth=depth,
        sources=tuple(sources),
        columns=columns,
        weight_blocks=tuple(weight_blocks),
        weight_index=None if np.array_equal(weight_index, np.arange(offset)) else weight_index.astype(np.int32),
        lags=(depth - 1 - delays[order]).astype(np.int32),
        columns_read=np.concatenate(read)[order].astype(np.int32),
        targets=np.concatenate(targets)[order].astype(np.int32),
    )


def _route_counts(block: DeliveryBlock, pre_spike, contacts=None, *, dtype):
    """Per-target event counts of one route: ``pre_spike`` gathered per contact and summed per synapse row.

    ``contacts`` selects the contacts that take part (a boolean mask over
    the route's contacts); all of them when ``None``.
    """
    counts = jnp.asarray(pre_spike, dtype=dtype)[jnp.asarray(block.pre_index)]
    target = np.asarray(block.flat_target_index)
    if contacts is not None:
        counts = counts[np.flatnonzero(contacts)]
        target = target[contacts]
    return jnp.zeros((int(block.source.n_active),), dtype=dtype).at[jnp.asarray(target)].add(counts)


def _wrap(index, depth: int):
    """``index % depth`` for a non-negative ``index``, in its integer dtype."""
    index = jnp.asarray(index)
    return jax.lax.rem(index, jnp.asarray(depth, dtype=index.dtype))


def _in_unit(value, unit):
    """``value``'s mantissa in ``unit`` (``None``: a plain array)."""
    if unit is not None:
        value = u.Quantity(value).in_unit(unit)
    return jnp.asarray(u.get_mantissa(value))


def _queue_arrival(blocks, state: DeliveryState, queue_index: int):
    """The weighted arrivals of one queue at this boundary, and their counts for a counting queue."""
    arrival = zero_arrival(state.queue_sources[queue_index])
    unit = u.get_unit(arrival) if isinstance(arrival, u.Quantity) else None
    mantissa = jnp.asarray(u.get_mantissa(arrival))
    count_dtype = state.count_dtype(queue_index)
    count = None if count_dtype is None else jnp.zeros(mantissa.shape, dtype=count_dtype)
    plan = state.queue_plans[queue_index]
    if not plan.depth:
        return arrival, count
    history_state = state.histories[queue_index]
    cursor = state.history_cursors[queue_index].value
    if plan.scalar:
        rows = [
            jax.lax.dynamic_index_in_dim(history_state.value, _wrap(cursor + lag, plan.depth), axis=0, keepdims=False)
            for _, lag, _, _ in plan.scalar
        ]
    else:
        lagged = _wrap(cursor + jnp.asarray(plan.lags), plan.depth)
        rows = [history_state.value[lagged, jnp.asarray(plan.columns_read)]]
    # The reads come before this step's row is written. Nothing else orders
    # them -- the row is the source's events, which need not depend on the
    # arrivals -- and XLA then copied the whole history twice a step to
    # write it in place.
    rows, history_state.value = jax.lax.optimization_barrier((rows, history_state.value))
    if plan.scalar:
        for (block_index, _, start, size), row in zip(plan.scalar, rows):
            pre_spike = row[start : start + size]
            event = state.delivery_ops[block_index](pre_spike).reshape(mantissa.shape)
            # Each route's events are summed before they join the queue, as
            # in a queue of weighted events; without the barrier XLA folds
            # ``sum + zeros.at[i].add(x)`` into ``sum.at[i].add(x)``.
            event = jax.lax.optimization_barrier(_in_unit(event, unit).astype(mantissa.dtype))
            mantissa = mantissa + event
            if count is not None:
                count = count + _route_counts(blocks[block_index], pre_spike, dtype=count.dtype)
    else:
        (pre,) = rows
        weights = [_in_unit(blocks[block_index].weight, unit) for block_index in plan.weight_blocks]
        weight = jnp.concatenate(weights) if len(weights) > 1 else weights[0]
        if plan.weight_index is not None:
            weight = weight[jnp.asarray(plan.weight_index)]
        targets = jnp.asarray(plan.targets)
        mantissa = mantissa.at[targets].add((pre * weight).astype(mantissa.dtype))
        if count is not None:
            count = count.at[targets].add(pre.astype(count.dtype))
    # The target adds the arrival to its synapse state as one value, as it
    # added a queue slot; the barrier keeps XLA from folding the scatter
    # into that sum, which would add the contacts to the state one by one.
    mantissa = jax.lax.optimization_barrier(mantissa)
    return (mantissa if unit is None else u.Quantity(mantissa, unit)), count


def write_arrivals(
    blocks: tuple[DeliveryBlock, ...],
    state: DeliveryState,
    *,
    populations: dict,
) -> None:
    """Write the delayed arrivals of this boundary into target event buffers.

    Parameters
    ----------
    blocks : tuple of DeliveryBlock
        Static delivery blocks for the current run setup: their weights,
        read now, weight the arrivals.
    state : DeliveryState
        Event histories, cursors, and backend operators. The queues, not
        the delivery blocks, are what this walks -- several blocks may share
        one queue.
    populations : dict
        ``population_name -> Population`` mapping used to locate target cells.

    Notes
    -----
    A contact with ``k >= 1`` delay steps delivers at the start of step
    ``m + k + 1`` the events its source counted at the end of step ``m``,
    weighted with the weight at delivery, as NEURON's ``NetCon`` does
    (CELL-69): the dynamics first see the event in step ``m + 1 + k``, as
    in ``Cell.run`` and NEURON (CELL-75). The arrivals of one queue are summed in the order a queue of
    weighted events summed them: longest delay first, then by route and by
    contact. Every queue writes its target's event buffer each step.
    """
    for index, (post_population, layout_id) in enumerate(state.queue_keys):
        arrival, count = _queue_arrival(blocks, state, index)
        cell = populations[post_population].cell
        if count is None:
            cell._write_event_arrival(layout_id, arrival)
        else:
            cell._write_event_arrival(layout_id, arrival, count=count)


def record_source_events(state: DeliveryState) -> None:
    """Write the source events of this boundary into each history's current row.

    Parameters
    ----------
    state : DeliveryState
        Event histories and cursors of the current run setup.
    """
    for index, history in enumerate(state.histories):
        if history is None:
            continue
        plan = state.queue_plans[index]
        counts = [
            jnp.asarray(source.current_event_count(ids), dtype=history.value.dtype)
            for source, ids in zip(plan.sources, plan.columns)
        ]
        row = jnp.concatenate(counts) if len(counts) > 1 else counts[0]
        history.value = jax.lax.dynamic_update_index_in_dim(
            history.value, row, state.history_cursors[index].value, axis=0
        )


def apply_immediate_events(
    blocks: tuple[DeliveryBlock, ...],
    state: DeliveryState,
    *,
    populations: dict,
) -> None:
    """Apply zero-delay population events at their detection boundary.

    A counting queue (a ``"sum_count"`` target layout) also passes the
    per-row number of these events as ``count`` (CELL-12b).
    """
    grouped = {}
    grouped_counts = {}
    for index, block in enumerate(blocks):
        pre_spike = route_source_event(block)
        delay_steps = np.asarray(block.delay_steps, dtype=np.int32)
        n_active = int(block.source.n_active)
        if delay_steps.ndim == 0:
            if int(delay_steps) != 0:
                continue
            immediate = None
            event = state.delivery_ops[index](pre_spike).reshape((n_active,))
        else:
            immediate = delay_steps == 0
            if not np.any(immediate):
                continue
            contact_event = pre_spike[jnp.asarray(block.pre_index)] * block.weight
            event = zeros_like(contact_event, shape=(n_active,))
            event = event.at[jnp.asarray(block.flat_target_index[immediate])].add(contact_event[immediate])
        key = (block.source.post_population, int(block.source.layout_id))
        grouped[key] = event if key not in grouped else grouped[key] + event
        count_dtype = state.count_dtype(state.block_queue_indices[index])
        if count_dtype is not None:
            counts = _route_counts(block, pre_spike, immediate, dtype=count_dtype)
            grouped_counts[key] = counts if key not in grouped_counts else grouped_counts[key] + counts
    for (post_population, layout_id), event in grouped.items():
        cell = populations[post_population].cell
        count = grouped_counts.get((post_population, layout_id))
        if count is None:
            cell._apply_synapse_layout_event_drive(layout_id, event)
        else:
            cell._apply_synapse_layout_event_drive(layout_id, event, count=count)


def advance_delivery_state(state: DeliveryState) -> None:
    """Advance every history cursor by one step.

    Parameters
    ----------
    state : DeliveryState
        Mutable delivery state for the active run.
    """
    for index, cursor in enumerate(state.history_cursors):
        if cursor is not None:
            cursor.value = _wrap(cursor.value + 1, state.queue_plans[index].depth)


def normalize_event_backend(value: str) -> str:
    """Validate the requested event-delivery backend.

    Parameters
    ----------
    value : {"auto", "scatter", "brainevent"}
        Requested event backend.

    Returns
    -------
    str
        The validated backend name.

    Raises
    ------
    ValueError
        If ``value`` is not a supported backend selector.
    """
    if value not in ("auto", "scatter", "brainevent"):
        raise ValueError(f"Network event_backend must be 'auto', 'scatter', or 'brainevent', got {value!r}.")
    return value


def resolve_event_backend(value: str) -> str:
    """Return the concrete delivery backend for one run setup.

    Parameters
    ----------
    value : {"auto", "scatter", "brainevent"}
        Validated backend selector.

    Returns
    -------
    {"scatter", "brainevent"}
        Concrete backend used by delivery operators.

    Raises
    ------
    RuntimeError
        If ``value`` is ``"brainevent"`` but the installed brainevent has no
        ``brainevent.csrmv``. The message points to ``"scatter"``.

    Notes
    -----
    ``"auto"`` resolves to ``"scatter"``. With the default
    ``brainevent_backend="jax_raw"`` the brainevent operator lowers to the
    same gather and scatter-add, so preferring it would change the default
    path without a benefit.
    """
    if value in ("auto", "scatter"):
        return "scatter"
    try:
        import brainevent
    except ImportError as error:
        raise RuntimeError(
            "event_backend='brainevent' requires the brainevent package; use event_backend='scatter' instead."
        ) from error
    if not hasattr(brainevent, "csrmv"):
        raise RuntimeError(
            "event_backend='brainevent' requires brainevent.csrmv, which this brainevent "
            f"{getattr(brainevent, '__version__', '(unknown version)')} does not provide; "
            "use event_backend='scatter' instead."
        )
    return "brainevent"


def make_delivery_op(
    block: DeliveryBlock,
    *,
    pre_size: int,
    backend: str,
    brainevent_backend: str | None = "jax_raw",
):
    """Build a runtime event-delivery operator.

    Parameters
    ----------
    block : DeliveryBlock
        Static sparse delivery block produced during network setup.
    pre_size : int
        Presynaptic population size.
    backend : {"scatter", "brainevent"}
        Delivery backend selected for this run setup.
    brainevent_backend : str or None, optional
        Backend forwarded to ``brainevent.csrmv`` when used.

    Returns
    -------
    Callable
        Function mapping a population-level presynaptic spike vector to a
        flattened postsynaptic event vector.

    Notes
    -----
    The returned callable captures static sparse indices as JAX arrays. The
    scatter path computes ``pre_spike[pre_index] * weight`` and accumulates it
    into ``flat_target_index``. The ``brainevent`` path stores the same
    contacts as a CSR matrix with one row per presynaptic member (contacts
    stably sorted by ``pre_index``; repeated destinations are kept) and
    computes ``brainevent.csrmv(..., transpose=True)``. Its exact bilinear
    derivative uses scatter operations so batching weight and event tangents
    does not depend on the backend's sparse-primitive batching rules.
    """
    target_size = int(block.source.n_active)
    pre_index = jnp.asarray(block.pre_index, dtype=jnp.int32)
    flat_target_index = jnp.asarray(block.flat_target_index, dtype=jnp.int32)
    if backend == "brainevent":
        import brainevent

        static_pre = np.asarray(block.pre_index, dtype=np.int32)
        order = np.argsort(static_pre, kind="stable")
        csr_indices = jnp.asarray(np.asarray(block.flat_target_index, dtype=np.int32)[order])
        csr_indptr = jnp.asarray(
            np.concatenate([[0], np.cumsum(np.bincount(static_pre, minlength=int(pre_size)))]).astype(np.int32)
        )
        csr_order = jnp.asarray(order)

        @jax.custom_jvp
        def _csrmv(weight, pre_spike):
            # A one-element weight is homogeneous for csrmv and needs no reordering.
            csr_weight = weight.reshape(1) if weight.size == 1 else weight[csr_order]
            return brainevent.csrmv(
                csr_weight,
                csr_indices,
                csr_indptr,
                pre_spike,
                shape=(int(pre_size), target_size),
                transpose=True,
                backend=brainevent_backend,
            )

        def _scatter(weight, pre_spike):
            contact_event = pre_spike[pre_index] * weight
            return jnp.zeros((target_size,), dtype=contact_event.dtype).at[flat_target_index].add(contact_event)

        @_csrmv.defjvp
        def _csrmv_jvp(primals, tangents):
            weight, event = primals
            dweight, devent = tangents
            # d(W z) = dW z + W dz, including repeated sparse destinations.
            return _csrmv(weight, event), _scatter(dweight, event) + _scatter(weight, devent)

        def _op(pre_spike):
            weight, weight_unit = u.split_mantissa_unit(block.weight)
            event, event_unit = u.split_mantissa_unit(pre_spike)
            dtype = jnp.result_type(weight, event, float)
            result = _csrmv(jnp.asarray(weight, dtype=dtype), jnp.asarray(event, dtype=dtype))
            return u.maybe_decimal(result * weight_unit * event_unit)

        return _op

    def _op(pre_spike):
        contact_event = pre_spike[pre_index] * block.weight
        out = zeros_like(contact_event, shape=(target_size,))
        return out.at[flat_target_index].add(contact_event)

    return _op
