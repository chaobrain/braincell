"""Validation and lowering for network connections."""

from __future__ import annotations

from dataclasses import dataclass

import brainunit as u
import numpy as np

from .core import Connection, Population


@dataclass(frozen=True)
class ConnectionBlock:
    """Runtime-ready sparse contact block."""

    pre_population: str
    post_population: str
    synapse: str
    layout_id: int
    n_active: int
    pre_index: np.ndarray
    post_index: np.ndarray
    synapse_index: np.ndarray
    weight: object
    delay_steps: np.ndarray
    buffer_size: int


def lower_connections(
    populations: dict[str, Population],
    connections: tuple[Connection, ...],
    *,
    dt,
    delay_quantization: str = "ceil",
) -> tuple[ConnectionBlock, ...]:
    """Validate public connections and return runtime blocks."""
    _validate_time_quantity(dt, name="dt")
    delay_quantization = _normalize_delay_quantization(delay_quantization)
    return tuple(
        _lower_connection(populations, connection, dt=dt, delay_quantization=delay_quantization)
        for connection in connections
    )


def _lower_connection(
    populations: dict[str, Population],
    connection: Connection,
    *,
    dt,
    delay_quantization: str,
) -> ConnectionBlock:
    if connection.pre_population not in populations:
        raise KeyError(f"Unknown pre_population {connection.pre_population!r}.")
    if connection.post_population not in populations:
        raise KeyError(f"Unknown post_population {connection.post_population!r}.")

    pre = populations[connection.pre_population]
    post = populations[connection.post_population]
    _validate_indices(connection.pre_index, pre.size, "pre_index")
    _validate_indices(connection.post_index, post.size, "post_index")

    layout_id, n_active, synapse_node = resolve_synapse_layout(post, connection.synapse)
    _validate_indices(connection.synapse_index, n_active, "synapse_index")
    delay_steps = _expand_delay_steps(
        connection.delay,
        dt=dt,
        n_contact=connection.n_contact,
        quantization=delay_quantization,
    )
    if connection.weight is None:
        weight = _default_edge_weight(
            synapse_node,
            connection.post_index,
            connection.synapse_index,
            n_active=n_active,
        )
    else:
        weight = _expand_weight(connection.weight, n_contact=connection.n_contact)
    return ConnectionBlock(
        pre_population=connection.pre_population,
        post_population=connection.post_population,
        synapse=connection.synapse,
        layout_id=layout_id,
        n_active=n_active,
        pre_index=connection.pre_index,
        post_index=connection.post_index,
        synapse_index=connection.synapse_index,
        weight=weight,
        delay_steps=delay_steps,
        buffer_size=int(np.max(delay_steps, initial=1)) + 1,
    )


def resolve_synapse_layout(population: Population, synapse: str) -> tuple[int, int, object]:
    """Return ``(layout_id, n_active, node)`` for a unique synapse layout."""
    runtime = population.cell.runtime
    matches = []
    for layout, node in runtime.iter_synapse_layouts():
        declaration = runtime.get_layout_mechanism(layout.id)
        if declaration.instance_name != synapse:
            continue
        matches.append((layout.id, int(layout.n_active), node))
    if not matches:
        raise KeyError(
            f"Population {population.name!r} has no placed synapse named {synapse!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Population {population.name!r} has multiple synapse layouts named {synapse!r}; "
            "network v1 requires a unique target layout."
        )
    layout_id, n_active, node = matches[0]
    return int(layout_id), int(n_active), node


def _default_edge_weight(synapse_node, post_index: np.ndarray, synapse_index: np.ndarray, *, n_active: int):
    if not hasattr(synapse_node, "weight"):
        raise ValueError(
            "Connection weight is None, but the target synapse has no default "
            "'weight' parameter."
        )
    weight = synapse_node.weight
    if isinstance(weight, u.Quantity):
        decimal = np.asarray(weight.to_decimal(weight.unit))
        if decimal.shape == ():
            return u.Quantity(np.broadcast_to(decimal, post_index.shape).copy(), weight.unit)
        if decimal.ndim >= 2 and decimal.shape[-1] == int(n_active):
            return u.Quantity(decimal[post_index, synapse_index], weight.unit)
        if decimal.shape[-1:] == (1,):
            decimal = decimal[..., 0]
        if decimal.shape[0] != int(n_active):
            return u.Quantity(decimal[post_index], weight.unit)
        return u.Quantity(decimal[synapse_index], weight.unit)
    arr = np.asarray(weight)
    if arr.shape == ():
        return np.broadcast_to(arr, post_index.shape).copy()
    if arr.ndim >= 2 and arr.shape[-1] == int(n_active):
        return arr[post_index, synapse_index]
    if arr.shape[-1:] == (1,):
        arr = arr[..., 0]
    if arr.shape[0] != int(n_active):
        return arr[post_index]
    return arr[synapse_index]


def _validate_indices(indices: np.ndarray, size: int, name: str) -> None:
    if indices.size == 0:
        return
    min_index = int(np.min(indices))
    max_index = int(np.max(indices))
    if min_index < 0 or max_index >= int(size):
        raise IndexError(
            f"Connection {name} out of range for population size {size!r}: "
            f"min={min_index!r}, max={max_index!r}."
        )


def _expand_weight(weight, *, n_contact: int):
    if isinstance(weight, u.Quantity):
        decimal = np.asarray(weight.to_decimal(weight.unit))
        if decimal.shape == ():
            return u.Quantity(np.broadcast_to(decimal, (n_contact,)).copy(), weight.unit)
        if decimal.shape != (n_contact,):
            raise ValueError(
                f"Connection weight must be scalar or shape {(n_contact,)!r}, "
                f"got {decimal.shape!r}."
            )
        return weight

    arr = np.asarray(weight)
    if arr.shape == ():
        return np.broadcast_to(arr, (n_contact,)).copy()
    if arr.shape != (n_contact,):
        raise ValueError(
            f"Connection weight must be scalar or shape {(n_contact,)!r}, got {arr.shape!r}."
        )
    return arr


def _expand_delay_steps(delay, *, dt, n_contact: int, quantization: str = "ceil") -> np.ndarray:
    """Return fixed-step delay offsets for one connection table.

    Parameters
    ----------
    delay : Quantity[time]
        Scalar or per-edge delay.
    dt : Quantity[time]
        Fixed simulation step.
    n_contact : int
        Number of contacts.
    quantization : {"ceil", "floor", "strict"}
        Policy for delays that do not fall on the fixed-step grid.
    """
    _validate_time_quantity(delay, name="delay")
    quantization = _normalize_delay_quantization(quantization)
    delay_ms = np.asarray(delay.to_decimal(u.ms), dtype=float)
    if delay_ms.shape == ():
        delay_ms = np.broadcast_to(delay_ms, (n_contact,)).copy()
    if delay_ms.shape != (n_contact,):
        raise ValueError(
            f"Connection delay must be scalar or shape {(n_contact,)!r}, got {delay_ms.shape!r}."
        )
    if np.any(delay_ms < 0.0):
        raise ValueError("Connection delay must be >= 0.")
    dt_ms = float(np.asarray(dt.to_decimal(u.ms), dtype=float).reshape(()))
    raw_steps = delay_ms / dt_ms
    if quantization == "strict":
        rounded = np.rint(raw_steps).astype(np.int32)
        if not np.allclose(raw_steps, rounded, rtol=1e-9, atol=1e-9):
            raise ValueError(
                "Connection delay must be an integer multiple of dt when "
                "delay_quantization='strict'."
            )
        steps = rounded
    elif quantization == "ceil":
        steps = np.ceil(raw_steps - 1e-12).astype(np.int32)
    elif quantization == "floor":
        steps = np.floor(raw_steps + 1e-12).astype(np.int32)
    else:  # pragma: no cover
        raise ValueError(
            "Connection delay_quantization must be 'ceil', 'floor', or 'strict'."
        )
    return np.maximum(steps, 1)


def _normalize_delay_quantization(value: str) -> str:
    if value not in ("ceil", "floor", "strict"):
        raise ValueError(
            "Network delay_quantization must be 'ceil', 'floor', or 'strict', "
            f"got {value!r}."
        )
    return value


def _validate_time_quantity(value, *, name: str) -> None:
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"Network {name} must be a time quantity, got {value!r}.")
    decimal = np.asarray(value.to_decimal(u.ms), dtype=float)
    if name == "dt" and decimal.shape not in ((), (1,)):
        raise ValueError(f"Network dt must be scalar, got shape {decimal.shape!r}.")
    if name == "dt" and float(decimal.reshape(())) <= 0.0:
        raise ValueError(f"Network dt must be > 0, got {value!r}.")
