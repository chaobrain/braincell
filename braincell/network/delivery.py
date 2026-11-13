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
allocates one persistent ring per target synapse layout, writes due arrivals,
and enqueues newly generated events into future slots. Backends may group
contacts internally without changing queue ownership.

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
class _DelayedPart:
    """The contacts of one route that have a positive delay (CELL-69).

    ``delay`` is the route's scalar delay, or one delay per listed contact;
    ``contacts`` lists the route's contacts that have a positive delay
    (``None``: all of them); ``columns`` is each listed contact's column in
    its source's history (``None`` for a scalar-delay route, which reads the
    whole history row).
    """

    block_index: int
    history_index: int
    delay: int | np.ndarray
    contacts: np.ndarray | None = None
    columns: np.ndarray | None = None


@dataclass(frozen=True)
class _QueuePlan:
    """How one queue sums its delayed arrivals at a boundary (CELL-69).

    The sum takes the contacts in the order the queue added them when it
    stored weighted events: by delay, longest first, then by route and by
    contact. ``order`` permutes the concatenated contacts of the per-contact
    parts into that order (``None``: already in it), and ``targets`` is each
    contact's synapse row in that order. Scalar-delay parts are listed
    longest delay first.
    """

    parts: tuple[_DelayedPart, ...] = ()
    order: np.ndarray | None = None
    targets: np.ndarray | None = None


@dataclass(frozen=True)
class DeliveryState:
    """Mutable runtime buffers and backend ops for event delivery.

    Parameters
    ----------
    histories : tuple
        One event history per live source that some contact reads with a
        positive delay: shape ``(depth, columns)``, the source's event counts
        at the last ``depth`` boundaries, ``depth`` being the longest delay
        in steps (CELL-69).
    history_cursors : tuple
        Per-history row the current step writes.
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
        How each queue sums its delayed arrivals.
    history_sources : tuple
        The live event source of each history.
    history_columns : tuple
        The source members each history keeps, ``None`` for all of them.
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
    history_sources: tuple = ()
    history_columns: tuple = ()
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
    targeting the same synapse layout writes one shared ring buffer.

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

    Every buffer in this module -- arrival vectors, ring-buffer queues, and
    the scatter accumulator inside a delivery op -- is shaped from an event
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
    Each live source that a contact reads with a positive delay keeps one
    history of its event counts, as deep as the longest such delay in
    steps. A route with one delay per contact (the scatter path) keeps only
    the source members its delayed contacts read; a scalar-delay route (the
    ``brainevent`` path) reads whole rows.
    """
    if len(delivery_ops) != len(blocks):
        raise ValueError(
            f"delivery_ops must have the same length as delivery blocks; got {len(delivery_ops)!r} and {len(blocks)!r}."
        )
    queue_index_by_key: dict[tuple[str, int], int] = {}
    queue_keys: list[tuple[str, int]] = []
    queue_sources: list[ConnectionBlock] = []
    block_queue_indices: list[int] = []
    history_index_by_source: dict[int, int] = {}
    history_sources: list = []
    history_depths: list[int] = []
    history_reads: list[list[np.ndarray]] = []
    history_whole: list[bool] = []
    delayed: list[tuple[int, int, np.ndarray, np.ndarray | None]] = []
    for block_index, block in enumerate(blocks):
        key = (block.source.post_population, int(block.source.layout_id))
        queue_index = queue_index_by_key.get(key)
        if queue_index is None:
            queue_index = len(queue_keys)
            queue_index_by_key[key] = queue_index
            queue_keys.append(key)
            queue_sources.append(block.source)
        block_queue_indices.append(queue_index)
        delay_steps = np.asarray(block.delay_steps, dtype=np.int32)
        positive = delay_steps > 0
        if not np.any(positive):
            continue
        source = block.source.event_source
        history_index = history_index_by_source.get(id(source))
        if history_index is None:
            history_index = len(history_sources)
            history_index_by_source[id(source)] = history_index
            history_sources.append(source)
            history_depths.append(0)
            history_reads.append([])
            history_whole.append(False)
        history_depths[history_index] = max(history_depths[history_index], int(delay_steps.max()))
        if delay_steps.ndim == 0:
            history_whole[history_index] = True
            delayed.append((block_index, history_index, delay_steps, None))
            continue
        contacts = np.flatnonzero(positive)
        history_reads[history_index].append(np.asarray(block.pre_index, dtype=np.int64)[contacts])
        delayed.append((block_index, history_index, delay_steps, contacts))

    history_columns: list[np.ndarray | None] = []
    for index, source in enumerate(history_sources):
        columns = None
        if not history_whole[index]:
            columns = np.unique(np.concatenate(history_reads[index]))
            if columns.size == int(source.size):
                columns = None
        history_columns.append(columns)

    queue_parts: list[list[_DelayedPart]] = [[] for _ in queue_keys]
    for block_index, history_index, delay_steps, contacts in delayed:
        block = blocks[block_index]
        queue_index = block_queue_indices[block_index]
        if contacts is None:
            queue_parts[queue_index].append(_DelayedPart(block_index, history_index, int(delay_steps)))
            continue
        pre = np.asarray(block.pre_index, dtype=np.int64)[contacts]
        columns = history_columns[history_index]
        column = pre if columns is None else np.searchsorted(columns, pre)
        queue_parts[queue_index].append(
            _DelayedPart(
                block_index,
                history_index,
                delay_steps[contacts],
                contacts=None if contacts.size == delay_steps.size else contacts,
                columns=column.astype(np.int32),
            )
        )
    queue_plans = tuple(_queue_plan(parts, blocks) for parts in queue_parts)

    histories = []
    for index, source in enumerate(history_sources):
        columns = history_columns[index]
        ids = np.arange(int(source.size), dtype=np.int32) if columns is None else columns
        dtype = jax.eval_shape(lambda source=source, ids=ids: jnp.asarray(source.current_event_count(ids))).dtype
        histories.append(brainstate.ShortTermState(jnp.zeros((history_depths[index], ids.size), dtype=dtype)))
    history_cursors = tuple(brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32)) for _ in histories)
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
        history_sources=tuple(history_sources),
        history_columns=tuple(history_columns),
        count_dtypes=count_dtypes,
    )


def _queue_plan(parts: list[_DelayedPart], blocks) -> _QueuePlan:
    """Order one queue's delayed parts as a queue of weighted events summed them."""
    if not parts:
        return _QueuePlan()
    scalar = [part for part in parts if np.ndim(part.delay) == 0]
    if scalar:
        if len(scalar) != len(parts):
            raise ValueError("A queue mixes scalar-delay and per-contact-delay routes.")
        # Stable: routes of one delay keep their order.
        order = sorted(range(len(parts)), key=lambda index: -int(parts[index].delay))
        return _QueuePlan(parts=tuple(parts[index] for index in order))
    delays = np.concatenate([np.asarray(part.delay) for part in parts])
    targets = np.concatenate(
        [
            np.asarray(blocks[part.block_index].flat_target_index)[
                slice(None) if part.contacts is None else part.contacts
            ]
            for part in parts
        ]
    ).astype(np.int32)
    order = np.argsort(-delays, kind="stable")
    if np.array_equal(order, np.arange(order.size)):
        return _QueuePlan(parts=tuple(parts), targets=targets)
    return _QueuePlan(parts=tuple(parts), order=order.astype(np.int32), targets=targets[order])


def _history_rows(state: DeliveryState, index: int, delay):
    """The counts of history ``index`` ``delay`` steps back: one row, or one value per contact."""
    history = state.histories[index].value
    return (state.history_cursors[index].value - delay) % history.shape[0], history


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


def _in_unit(value, unit, dtype):
    """``value``'s mantissa in ``unit`` (``None``: a plain array), cast to ``dtype``."""
    if unit is not None:
        value = u.Quantity(value).in_unit(unit)
    return jnp.asarray(u.get_mantissa(value)).astype(dtype)


def _queue_arrival(blocks, state: DeliveryState, queue_index: int):
    """The weighted arrivals of one queue at this boundary, and their counts for a counting queue."""
    arrival = zero_arrival(state.queue_sources[queue_index])
    unit = u.get_unit(arrival) if isinstance(arrival, u.Quantity) else None
    mantissa = jnp.asarray(u.get_mantissa(arrival))
    count_dtype = state.count_dtype(queue_index)
    count = None if count_dtype is None else jnp.zeros(mantissa.shape, dtype=count_dtype)
    plan = state.queue_plans[queue_index]
    if not plan.parts:
        return arrival, count
    if plan.targets is None:
        for part in plan.parts:
            row, history = _history_rows(state, part.history_index, int(part.delay))
            pre_spike = jax.lax.dynamic_index_in_dim(history, row, axis=0, keepdims=False)
            event = state.delivery_ops[part.block_index](pre_spike).reshape(mantissa.shape)
            # Each route's events are summed before they join the queue, as
            # in a queue of weighted events; without the barrier XLA folds
            # ``sum + zeros.at[i].add(x)`` into ``sum.at[i].add(x)``.
            mantissa = mantissa + jax.lax.optimization_barrier(_in_unit(event, unit, mantissa.dtype))
            if count is not None:
                count = count + _route_counts(blocks[part.block_index], pre_spike, dtype=count.dtype)
    else:
        values, counts = [], []
        for part in plan.parts:
            rows, history = _history_rows(state, part.history_index, jnp.asarray(part.delay))
            pre = history[rows, jnp.asarray(part.columns)]
            weight = blocks[part.block_index].weight
            if part.contacts is not None:
                weight = weight[part.contacts]
            values.append(_in_unit(pre * weight, unit, mantissa.dtype))
            counts.append(pre)
        value = jnp.concatenate(values) if len(values) > 1 else values[0]
        if plan.order is not None:
            value = value[jnp.asarray(plan.order)]
        targets = jnp.asarray(plan.targets)
        mantissa = mantissa.at[targets].add(value)
        if count is not None:
            pre = jnp.concatenate([jnp.asarray(item, dtype=count.dtype) for item in counts])
            if plan.order is not None:
                pre = pre[jnp.asarray(plan.order)]
            count = count.at[targets].add(pre)
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
    ``m + k`` the events its source counted at the end of step ``m``,
    weighted with the weight at delivery, as NEURON's ``NetCon`` does
    (CELL-69). The arrivals of one queue are summed in the order a queue of
    weighted events summed them: longest delay first, then by route and by
    contact; every queue writes its target's event buffer each step.
    """
    for index, (post_population, layout_id) in enumerate(state.queue_keys):
        arrival, count = _queue_arrival(blocks, state, index)
        cell = populations[post_population].cell
        if count is None:
            cell._write_event_arrival(layout_id, arrival)
        else:
            cell._write_event_arrival(layout_id, arrival, count=count)


def record_source_events(state: DeliveryState) -> None:
    """Write each history's source events of this boundary into its current row.

    Parameters
    ----------
    state : DeliveryState
        Event histories and cursors of the current run setup.
    """
    for index, history in enumerate(state.histories):
        source = state.history_sources[index]
        columns = state.history_columns[index]
        ids = np.arange(int(source.size), dtype=np.int32) if columns is None else columns
        current = jnp.asarray(source.current_event_count(ids), dtype=history.value.dtype)
        history.value = jax.lax.dynamic_update_index_in_dim(
            history.value, current, state.history_cursors[index].value, axis=0
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
        cursor.value = (cursor.value + 1) % state.histories[index].value.shape[0]


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
