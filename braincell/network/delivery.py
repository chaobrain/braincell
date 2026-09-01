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
import jax.numpy as jnp
import numpy as np

from .lowering import ConnectionBlock


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
    pre_index, post_index : ndarray of int, shape ``(n_contact,)``
        Presynaptic and postsynaptic population indices for this delay group.
    synapse_index : ndarray of int, shape ``(n_contact,)``
        Local target index in the postsynaptic synapse layout.
    flat_target_index : ndarray of int, shape ``(n_contact,)``
        Flattened target index equal to ``post_index * n_active + synapse_index``.
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
    post_index: np.ndarray
    synapse_index: np.ndarray
    flat_target_index: np.ndarray
    weight: object


@dataclass(frozen=True)
class DeliveryState:
    """Mutable runtime buffers and backend ops for event delivery.

    Parameters
    ----------
    ring_buffers : tuple
        Per-target-layout delayed event buffers.
    ring_cursors : tuple
        Per-target-layout current ring-buffer cursors.
    delivery_ops : tuple
        Backend-specific functions mapping pre spikes to flattened events.

    Notes
    -----
    ``DeliveryState`` belongs to the initialized Network runtime. It is reused
    across consecutive :meth:`Network.run` calls so pending events survive a
    split run, and is cleared only by :meth:`Network.reset_state`.
    """

    ring_buffers: tuple
    ring_cursors: tuple
    delivery_ops: tuple
    block_queue_indices: tuple[int, ...]
    queue_keys: tuple[tuple[str, int], ...]
    queue_sources: tuple[ConnectionBlock, ...]


def population_spike(spike) -> object:
    """Return one spike value per population member.

    Parameters
    ----------
    spike : array-like
        Cell spike buffer. A one-dimensional array is returned unchanged. A
        two-dimensional ``(pop_size, 1)`` buffer is squeezed on the trailing
        singleton axis. Higher-dimensional cell buffers are reduced with a
        logical ``any`` over all non-population axes.

    Returns
    -------
    object
        Population-level spike vector with shape ``(pop_size,)``.
    """
    if spike.ndim == 2 and spike.shape[-1] == 1:
        return spike[..., 0]
    if spike.ndim > 1:
        return jnp.any(spike, axis=tuple(range(1, spike.ndim)))
    return spike


def source_spike(spike, source_cv_id: int) -> object:
    """Return one spike per population member from a single source CV."""
    if spike.ndim < 2:
        return spike
    return spike[..., int(source_cv_id)]


def route_source_event(block: DeliveryBlock, *, populations: dict) -> object:
    """Read current event values for one population or direct live route."""
    if block.source.event_source is not None:
        source = block.source.event_source
        return source.current_event_count(np.arange(source.size, dtype=np.int32))
    pre_cell = populations[block.source.pre_population].cell
    return source_spike(pre_cell.spike.value, block.source.source_cv_id)


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

    Returns
    -------
    tuple of DeliveryBlock
        One block per input connection when ``group_by_delay=False``;
        otherwise one fixed-delay block per input delay group.

    Notes
    -----
    Delay grouping is an operator-lowering choice only. Every returned block
    targeting the same synapse layout writes one shared ring buffer.
    """
    delivery = []
    for block in blocks:
        if not group_by_delay:
            post_index = np.asarray(block.post_index, dtype=np.int32)
            synapse_index = np.asarray(block.synapse_index, dtype=np.int32)
            delivery.append(
                DeliveryBlock(
                    source=block,
                    delay_steps=np.asarray(block.delay_steps, dtype=np.int32),
                    pre_index=np.asarray(block.pre_index, dtype=np.int32),
                    post_index=post_index,
                    synapse_index=synapse_index,
                    flat_target_index=(
                        synapse_index if block.packed else post_index * int(block.n_active) + synapse_index
                    ).astype(np.int32, copy=False),
                    weight=block.weight,
                )
            )
            continue
        for delay_step in sorted(set(np.asarray(block.delay_steps, dtype=np.int32).tolist())):
            mask = np.asarray(block.delay_steps, dtype=np.int32) == int(delay_step)
            contact_indices = np.nonzero(mask)[0]
            post_index = block.post_index[contact_indices]
            synapse_index = block.synapse_index[contact_indices]
            delivery.append(
                DeliveryBlock(
                    source=block,
                    delay_steps=int(delay_step),
                    pre_index=block.pre_index[contact_indices],
                    post_index=post_index,
                    synapse_index=synapse_index,
                    flat_target_index=(
                        synapse_index if block.packed else post_index * int(block.n_active) + synapse_index
                    ).astype(np.int32, copy=False),
                    weight=slice_weight(block.weight, contact_indices),
                )
            )
    return tuple(delivery)


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


def zero_arrival(block: ConnectionBlock, *, post_size: int):
    """Return a zero arrival buffer for one lowered connection block.

    Parameters
    ----------
    block : ConnectionBlock
        Lowered block whose weight dtype/unit and synapse layout size define
        the arrival buffer.
    post_size : int
        Number of cells in the postsynaptic population.

    Returns
    -------
    object
        Zero event matrix with shape ``(post_size, block.n_active)``.
    """
    if block.packed:
        return zeros_like_packed_events(block.weight, n_active=block.n_active)
    return zeros_like_events(block.weight, post_size=post_size, n_active=block.n_active)


def zero_shared_ring_buffer(source: ConnectionBlock, *, depth: int, post_size: int):
    """Return one target-layout queue shared by every incoming route block."""
    shape = (depth, source.n_active) if source.packed else (depth, post_size, source.n_active)
    if isinstance(source.weight, u.Quantity):
        return u.math.zeros_like(source.weight, shape=shape)
    return jnp.zeros(shape, dtype=jnp.asarray(source.weight).dtype)


def zeros_like_packed_events(value, *, n_active: int):
    """Return a zero event vector for packed point instances."""
    if isinstance(value, u.Quantity):
        return u.math.zeros_like(value, shape=(n_active,))
    return jnp.zeros((n_active,), dtype=jnp.asarray(value).dtype)


def zeros_like_events(value, *, post_size: int, n_active: int):
    """Return a zero event buffer matching event payload dtype and unit.

    Parameters
    ----------
    value : object
        Example event payload used to infer dtype and unit.
    post_size : int
        Number of cells in the postsynaptic population.
    n_active : int
        Number of active local targets in the target synapse layout.

    Returns
    -------
    object
        Zero event matrix with shape ``(post_size, n_active)``.
    """
    if isinstance(value, u.Quantity):
        return u.math.zeros_like(value, shape=(post_size, n_active))
    return jnp.zeros((post_size, n_active), dtype=jnp.asarray(value).dtype)


def advance_ring_cursors(ring_buffers, ring_cursors) -> None:
    """Advance all ring-buffer cursors by one step.

    Parameters
    ----------
    ring_buffers : tuple
        Ring buffers whose first axis is the delay queue.
    ring_cursors : tuple
        Mutable cursor states, one per ring buffer.
    """
    for index, state in enumerate(ring_cursors):
        state.value = (state.value + 1) % ring_buffers[index].value.shape[0]


def create_delivery_state(
    blocks: tuple[DeliveryBlock, ...],
    *,
    populations: dict,
    delivery_ops: tuple,
) -> DeliveryState:
    """Create persistent mutable state for network event delivery.

    Parameters
    ----------
    blocks : tuple of DeliveryBlock
        Static delivery blocks for the run setup.
    populations : dict
        ``population_name -> Population`` mapping used to size post buffers.
    delivery_ops : tuple
        Backend-specific event delivery callables, one per block.

    Returns
    -------
    DeliveryState
        Ring buffers, cursors, and delivery operators used during one run.

    Raises
    ------
    ValueError
        If ``delivery_ops`` does not have one operator per delivery block.
    """
    if len(delivery_ops) != len(blocks):
        raise ValueError(
            f"delivery_ops must have the same length as delivery blocks; got {len(delivery_ops)!r} and {len(blocks)!r}."
        )
    queue_index_by_key: dict[tuple[str, int], int] = {}
    queue_keys: list[tuple[str, int]] = []
    queue_sources: list[ConnectionBlock] = []
    queue_depths: list[int] = []
    block_queue_indices: list[int] = []
    for block in blocks:
        key = (block.source.post_population, int(block.source.layout_id))
        queue_index = queue_index_by_key.get(key)
        if queue_index is None:
            queue_index = len(queue_keys)
            queue_index_by_key[key] = queue_index
            queue_keys.append(key)
            queue_sources.append(block.source)
            queue_depths.append(int(np.max(np.asarray(block.delay_steps), initial=0)) + 1)
        else:
            queue_depths[queue_index] = max(
                queue_depths[queue_index],
                int(np.max(np.asarray(block.delay_steps), initial=0)) + 1,
            )
        block_queue_indices.append(queue_index)
    ring_buffers = tuple(
        brainstate.ShortTermState(
            zero_shared_ring_buffer(
                source,
                depth=queue_depths[index],
                post_size=populations[source.post_population].size,
            )
        )
        for index, source in enumerate(queue_sources)
    )
    ring_cursors = tuple(brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32)) for _ in queue_keys)
    return DeliveryState(
        ring_buffers=ring_buffers,
        ring_cursors=ring_cursors,
        delivery_ops=delivery_ops,
        block_queue_indices=tuple(block_queue_indices),
        queue_keys=tuple(queue_keys),
        queue_sources=tuple(queue_sources),
    )


def write_arrivals(
    blocks: tuple[DeliveryBlock, ...],
    state: DeliveryState,
    *,
    populations: dict,
) -> None:
    """Write current delayed arrivals into private target event buffers.

    Parameters
    ----------
    blocks : tuple of DeliveryBlock
        Static delivery blocks for the current run setup.
    state : DeliveryState
        Mutable ring buffers and cursors for the current run.
    populations : dict
        ``population_name -> Population`` mapping used to locate target cells.

    Notes
    -----
    Multiple delivery blocks may target the same postsynaptic synapse layout.
    Their due arrivals are summed before updating the cell runtime
    ``event_buffers[layout_id]`` entry. The consumed ring-buffer slot is
    cleared after it is read.
    """
    for index, key in enumerate(state.queue_keys):
        cursor = state.ring_cursors[index].value
        arrival = state.ring_buffers[index].value[cursor]
        state.ring_buffers[index].value = (
            state.ring_buffers[index]
            .value.at[cursor]
            .set(
                zero_arrival(
                    state.queue_sources[index],
                    post_size=populations[key[0]].size,
                )
            )
        )
        post_population, layout_id = key
        cell = populations[post_population].cell
        cell.runtime.event_buffers[layout_id].value = arrival


def enqueue_future_events(
    blocks: tuple[DeliveryBlock, ...],
    state: DeliveryState,
    *,
    populations: dict,
) -> None:
    """Project current spikes into future ring-buffer slots.

    Parameters
    ----------
    blocks : tuple of DeliveryBlock
        Static delivery blocks for the current run setup.
    state : DeliveryState
        Mutable ring buffers, cursors, and backend operators.
    populations : dict
        ``population_name -> Population`` mapping used to read presynaptic
        spikes and size postsynaptic event matrices.

    Notes
    -----
    Each block reads population-level presynaptic spikes, applies its backend
    operator, reshapes the flattened event vector to ``(n_active,)`` for
    packed layouts or ``(post_size, n_active)`` for broadcast layouts, and
    adds it to the ring-buffer slot ``current_cursor + delay_steps``.
    """
    for index, block in enumerate(blocks):
        pre_spike = route_source_event(block, populations=populations)
        queue_index = state.block_queue_indices[index]
        delay_steps = np.asarray(block.delay_steps, dtype=np.int32)
        if delay_steps.ndim == 0:
            if int(delay_steps) == 0:
                continue
            event = state.delivery_ops[index](pre_spike)
            target_cursor = (state.ring_cursors[queue_index].value + int(delay_steps)) % state.ring_buffers[
                queue_index
            ].value.shape[0]
            state.ring_buffers[queue_index].value = (
                state.ring_buffers[queue_index]
                .value.at[target_cursor]
                .add(
                    event.reshape(
                        (block.source.n_active,)
                        if block.source.packed
                        else (populations[block.source.post_population].size, block.source.n_active)
                    )
                )
            )
            continue

        positive = delay_steps > 0
        if not np.any(positive):
            continue
        contact_event = pre_spike[jnp.asarray(block.pre_index)] * block.weight
        ring = state.ring_buffers[queue_index].value
        flat_ring = ring.reshape((ring.shape[0], -1))
        target_cursor = (state.ring_cursors[queue_index].value + jnp.asarray(delay_steps[positive])) % ring.shape[0]
        flat_ring = flat_ring.at[
            target_cursor,
            jnp.asarray(block.flat_target_index[positive]),
        ].add(contact_event[positive])
        state.ring_buffers[queue_index].value = flat_ring.reshape(ring.shape)


def apply_immediate_events(
    blocks: tuple[DeliveryBlock, ...],
    state: DeliveryState,
    *,
    populations: dict,
) -> None:
    """Apply zero-delay population events at their detection boundary."""
    grouped = {}
    for index, block in enumerate(blocks):
        pre_spike = route_source_event(block, populations=populations)
        delay_steps = np.asarray(block.delay_steps, dtype=np.int32)
        if delay_steps.ndim == 0:
            if int(delay_steps) != 0:
                continue
            event = state.delivery_ops[index](pre_spike).reshape(
                (block.source.n_active,)
                if block.source.packed
                else (populations[block.source.post_population].size, block.source.n_active)
            )
        else:
            immediate = delay_steps == 0
            if not np.any(immediate):
                continue
            contact_event = pre_spike[jnp.asarray(block.pre_index)] * block.weight
            target_size = (
                int(block.source.n_active)
                if block.source.packed
                else populations[block.source.post_population].size * int(block.source.n_active)
            )
            if isinstance(contact_event, u.Quantity):
                event = u.math.zeros_like(contact_event, shape=(target_size,))
            else:
                event = jnp.zeros((target_size,), dtype=jnp.asarray(contact_event).dtype)
            event = event.at[jnp.asarray(block.flat_target_index[immediate])].add(contact_event[immediate])
            event = event.reshape(
                (block.source.n_active,)
                if block.source.packed
                else (populations[block.source.post_population].size, block.source.n_active)
            )
        key = (block.source.post_population, int(block.source.layout_id))
        grouped[key] = event if key not in grouped else grouped[key] + event
    for (post_population, layout_id), event in grouped.items():
        populations[post_population].cell._apply_synapse_layout_event_drive(layout_id, event)


def advance_delivery_state(state: DeliveryState) -> None:
    """Advance all delivery ring-buffer cursors by one step.

    Parameters
    ----------
    state : DeliveryState
        Mutable delivery state for the active run.
    """
    advance_ring_cursors(state.ring_buffers, state.ring_cursors)


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
        If ``value`` is ``"brainevent"`` but ``brainevent.coomv`` is not
        available.

    Notes
    -----
    ``"auto"`` prefers ``brainevent.coomv`` when available and otherwise falls
    back to the JAX scatter implementation.
    """
    if value == "scatter":
        return "scatter"
    try:
        import brainevent  # noqa: F401
    except Exception:
        if value == "brainevent":
            raise
        return "scatter"
    if not hasattr(brainevent, "coomv"):
        if value == "brainevent":
            raise RuntimeError("event_backend='brainevent' requires brainevent.coomv.")
        return "scatter"
    return "brainevent"


def make_delivery_op(
    block: DeliveryBlock,
    *,
    pre_size: int,
    post_size: int,
    backend: str,
    brainevent_backend: str | None = "jax_raw",
):
    """Build a runtime event-delivery operator.

    Parameters
    ----------
    block : DeliveryBlock
        Static sparse delivery block produced during network setup.
    pre_size, post_size : int
        Presynaptic and postsynaptic population sizes.
    backend : {"scatter", "brainevent"}
        Delivery backend selected for this run setup.
    brainevent_backend : str or None, optional
        Backend forwarded to ``brainevent.coomv`` when used.

    Returns
    -------
    Callable
        Function mapping a population-level presynaptic spike vector to a
        flattened postsynaptic event vector.

    Notes
    -----
    The returned callable captures static sparse indices as JAX arrays. The
    scatter path computes ``pre_spike[pre_index] * weight`` and accumulates it
    into ``flat_target_index``. The ``brainevent`` path uses ``brainevent.coomv``
    with the same sparse topology.
    """
    target_size = int(block.source.n_active) if block.source.packed else int(post_size) * int(block.source.n_active)
    pre_index = jnp.asarray(block.pre_index, dtype=jnp.int32)
    flat_target_index = jnp.asarray(block.flat_target_index, dtype=jnp.int32)
    if backend == "brainevent":
        try:
            import brainevent
        except Exception:  # pragma: no cover
            backend = "scatter"
    if backend == "brainevent" and hasattr(brainevent, "coomv"):
        data = block.weight

        def _op(pre_spike):
            return brainevent.coomv(
                data,
                pre_index,
                flat_target_index,
                pre_spike,
                shape=(int(pre_size), target_size),
                transpose=True,
                backend=brainevent_backend,
            )

        return _op

    def _op(pre_spike):
        pre_values = pre_spike[pre_index]
        contact_event = pre_values * block.weight
        if isinstance(contact_event, u.Quantity):
            out = u.math.zeros_like(contact_event, shape=(target_size,))
        else:
            out = jnp.zeros((target_size,), dtype=jnp.asarray(contact_event).dtype)
        return out.at[flat_target_index].add(contact_event)

    return _op
