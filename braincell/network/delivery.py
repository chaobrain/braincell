"""Runtime synaptic event-delivery helpers.

This module owns the step-time path that moves population spikes into
postsynaptic synapse ``pre_spike`` buffers. The input is a lowered
``ConnectionBlock`` table, where each row is already a materialized synaptic
contact. The delivery layer groups those contacts by fixed delay, allocates
per-run delay ring buffers, writes due arrivals into target synapse layouts,
and enqueues newly generated events into future ring-buffer slots.

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
    """Runtime delivery group with one fixed delay step.

    Parameters
    ----------
    source : ConnectionBlock
        Lowered connection block that owns the original synapse layout and
        population metadata.
    delay_steps : int
        Fixed-step delay for every contact in this delivery block.
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
    delay_steps: int
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
        Per-delivery-block delayed event buffers.
    ring_cursors : tuple
        Per-delivery-block current ring-buffer cursors.
    delivery_ops : tuple
        Backend-specific functions mapping pre spikes to flattened events.

    Notes
    -----
    ``DeliveryState`` is allocated for a single :meth:`Network.run` call. It
    contains ring-buffer cursors and delayed events, so it must not be reused
    across separate runs.
    """

    ring_buffers: tuple
    ring_cursors: tuple
    delivery_ops: tuple


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


def delivery_blocks(blocks: tuple[ConnectionBlock, ...]) -> tuple[DeliveryBlock, ...]:
    """Split lowered connection blocks by fixed delay step.

    Parameters
    ----------
    blocks : tuple of ConnectionBlock
        Lowered connection blocks. Each block may contain heterogeneous
        per-contact ``delay_steps``.

    Returns
    -------
    tuple of DeliveryBlock
        Delivery blocks whose contacts all share one integer delay step.

    Notes
    -----
    Splitting by delay lets the runtime maintain one ring buffer depth per
    fixed-delay group while still accepting heterogeneous delays in the input
    connection table.
    """
    delivery = []
    for block in blocks:
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
                    flat_target_index=(post_index * int(block.n_active) + synapse_index).astype(np.int32, copy=False),
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
    return zeros_like_events(block.weight, post_size=post_size, n_active=block.n_active)


def zero_ring_buffer(block: DeliveryBlock, *, post_size: int):
    """Return an empty ring buffer for one delivery block.

    Parameters
    ----------
    block : DeliveryBlock
        Fixed-delay delivery block.
    post_size : int
        Number of cells in the postsynaptic population.

    Returns
    -------
    object
        Zero ring buffer with shape
        ``(block.delay_steps + 1, post_size, block.source.n_active)``.

    Notes
    -----
    The first axis is the delay queue. Events are written into a future cursor
    slot and read back when that slot becomes current.
    """
    depth = int(block.delay_steps) + 1
    if isinstance(block.weight, u.Quantity):
        return u.math.zeros_like(block.weight, shape=(depth, post_size, block.source.n_active))
    return jnp.zeros((depth, post_size, block.source.n_active), dtype=jnp.asarray(block.weight).dtype)


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
    """Create per-run mutable state for event delivery.

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
            "delivery_ops must have the same length as delivery blocks; "
            f"got {len(delivery_ops)!r} and {len(blocks)!r}."
        )
    ring_buffers = tuple(
        brainstate.ShortTermState(
            zero_ring_buffer(
                block,
                post_size=populations[block.source.post_population].size,
            )
        )
        for block in blocks
    )
    ring_cursors = tuple(
        brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))
        for _ in blocks
    )
    return DeliveryState(
        ring_buffers=ring_buffers,
        ring_cursors=ring_cursors,
        delivery_ops=delivery_ops,
    )


def write_arrivals(
    blocks: tuple[DeliveryBlock, ...],
    state: DeliveryState,
    *,
    populations: dict,
) -> None:
    """Write current delayed arrivals into target synapse ``pre_spike`` buffers.

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
    ``state_buffers[(layout_id, "pre_spike")]`` entry. The consumed ring-buffer
    slot is cleared after it is read.
    """
    grouped = {}
    for index, block in enumerate(blocks):
        key = (block.source.post_population, int(block.source.layout_id))
        cursor = state.ring_cursors[index].value
        arrival = state.ring_buffers[index].value[cursor]
        grouped[key] = arrival if key not in grouped else grouped[key] + arrival
        state.ring_buffers[index].value = state.ring_buffers[index].value.at[cursor].set(
            zero_arrival(
                block.source,
                post_size=populations[block.source.post_population].size,
            )
        )
    for (post_population, layout_id), arrival in grouped.items():
        cell = populations[post_population].cell
        cell.runtime.state_buffers[(layout_id, "pre_spike")] = arrival


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
    operator, reshapes the flattened event vector to
    ``(post_size, n_active)``, and adds it to the ring-buffer slot
    ``current_cursor + delay_steps``.
    """
    for index, block in enumerate(blocks):
        pre_cell = populations[block.source.pre_population].cell
        pre_spike = population_spike(pre_cell.spike.value)
        post_size = populations[block.source.post_population].size
        event = state.delivery_ops[index](pre_spike)
        target_cursor = (
            state.ring_cursors[index].value + int(block.delay_steps)
        ) % state.ring_buffers[index].value.shape[0]
        state.ring_buffers[index].value = state.ring_buffers[index].value.at[target_cursor].add(
            event.reshape((post_size, block.source.n_active))
        )


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
        raise ValueError(
            "Network event_backend must be 'auto', 'scatter', or 'brainevent', "
            f"got {value!r}."
        )
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
    target_size = int(post_size) * int(block.source.n_active)
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
