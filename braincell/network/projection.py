"""Synaptic projections from reusable edge sets to placed synapse pools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import brainunit as u
import numpy as np

from .connection import Connection, _as_index_array
from .edges import EdgeSet


@dataclass
class Projection:
    """Named mapping from cell-level edges to postsynaptic synapse targets.

    Parameters
    ----------
    name : str
        Projection name.
    edges : str or EdgeSet
        Referenced edge-set name or object.
    synapse_pool : str or sequence of str, optional
        Postsynaptic synapse pool name. A scalar name selects from all active
        points in that placed synapse group. A sequence is accepted as a
        legacy explicit per-contact target list.
    edge_index : array-like of int, optional
        EdgeSet row index per contact. Defaults to all edges once.
    weight : object, optional
        Scalar, per-edge, or per-contact event payload. ``None`` uses the
        placed synapse default weight when available.
    delay : object, optional
        Scalar, per-edge, or per-contact delay.
    contact_pre_index, contact_post_index : array-like of int, optional
        Explicit contact endpoints matched back to the referenced edge set.
    target_policy : {"per_edge", "by_post"}, optional
        Target selection policy for scalar ``synapse_pool``.
    number : int, optional
        Number of synapse targets selected per cell-level edge.
    replace : bool, optional
        Whether target selection can reuse local synapse targets.
    seed : int, optional
        Random seed for reproducible target selection.
    synapse : str or sequence of str, optional
        Compatibility alias for ``synapse_pool``.
    """

    name: str
    edges: str | EdgeSet
    synapse_pool: object | None = None
    edge_index: object | None = None
    weight: object | None = None
    delay: object = field(default_factory=lambda: 0.0 * u.ms)
    contact_pre_index: object | None = None
    contact_post_index: object | None = None
    target_policy: Literal["per_edge", "by_post"] = "per_edge"
    number: int = 1
    replace: bool = True
    seed: int | None = None
    synapse: object | None = None

    def __post_init__(self) -> None:
        self.name = _validate_name(self.name, "name")
        if not isinstance(self.edges, (str, EdgeSet)):
            raise TypeError("Projection edges must be an EdgeSet or edge-set name.")
        if isinstance(self.edges, str):
            self.edges = _validate_name(self.edges, "edges")
        if self.edge_index is not None:
            self.edge_index = _as_index_array(self.edge_index, name="edge_index")
        if self.contact_pre_index is not None:
            self.contact_pre_index = _as_index_array(self.contact_pre_index, name="contact_pre_index")
        if self.contact_post_index is not None:
            self.contact_post_index = _as_index_array(self.contact_post_index, name="contact_post_index")
        if (self.contact_pre_index is None) != (self.contact_post_index is None):
            raise ValueError("Projection contact_pre_index and contact_post_index must be provided together.")
        if self.edge_index is not None and self.contact_pre_index is not None:
            raise ValueError("Projection cannot define both edge_index and explicit pre/post contacts.")
        if (
            self.contact_pre_index is not None
            and self.contact_pre_index.shape != self.contact_post_index.shape
        ):
            raise ValueError(
                "Projection contact_pre_index and contact_post_index must have the same shape; "
                f"got {self.contact_pre_index.shape!r} and {self.contact_post_index.shape!r}."
            )
        if self.synapse_pool is None:
            self.synapse_pool = self.synapse
        elif self.synapse is not None:
            raise ValueError("Projection cannot define both synapse_pool and synapse.")
        if self.synapse_pool is None:
            raise ValueError("Projection requires synapse_pool.")
        self.synapse_pool = _normalize_synapse_pool(self.synapse_pool)
        self.synapse = self.synapse_pool
        self.target_policy = _normalize_target_policy(self.target_policy)
        self.number = _positive_int(self.number, "number")
        self.replace = bool(self.replace)
        if self.seed is not None and not isinstance(self.seed, (int, np.integer)):
            raise TypeError(f"Projection seed must be an integer or None, got {type(self.seed).__name__!s}.")
        if self.seed is not None:
            self.seed = int(self.seed)

    @property
    def edge_set_name(self) -> str:
        """Referenced edge-set name."""
        return self.edges if isinstance(self.edges, str) else self.edges.name

    @property
    def scalar_synapse_pool(self) -> str | None:
        """Return the scalar synapse-pool name, if this projection has one."""
        if self.synapse_pool.shape != ():
            return None
        return str(self.synapse_pool.reshape(()))

    def to_connections(self, edge_set: EdgeSet, *, pool_size: int | None = None) -> tuple[Connection, ...]:
        """Expand this projection into runtime connection declarations."""
        edge_index = self._contact_edge_index(edge_set)
        if edge_index.size:
            min_index = int(np.min(edge_index))
            max_index = int(np.max(edge_index))
            if min_index < 0 or max_index >= edge_set.n_edge:
                raise IndexError(
                    f"Projection {self.name!r} edge_index out of range for "
                    f"EdgeSet {edge_set.name!r}: min={min_index!r}, max={max_index!r}."
                )
        base_pre_index = edge_set.pre_index[edge_index]
        base_post_index = edge_set.post_index[edge_index]
        synapse, synapse_index, pre_index, post_index = self._contact_targets(
            base_pre_index,
            base_post_index,
            pool_size=pool_size,
        )
        number = self.number if self.synapse_pool.shape == () else 1
        weight = _contact_parameter(
            self.weight,
            base_n_edge=edge_index.shape[0],
            n_contact=synapse.shape[0],
            number=number,
        )
        delay = _contact_parameter(
            self.delay,
            base_n_edge=edge_index.shape[0],
            n_contact=synapse.shape[0],
            number=number,
        )

        connections = []
        for target in tuple(dict.fromkeys(synapse.tolist())):
            mask = synapse == target
            indices = np.nonzero(mask)[0]
            connections.append(
                Connection(
                    pre_population=edge_set.pre_population,
                    post_population=edge_set.post_population,
                    pre_index=pre_index[indices],
                    post_index=post_index[indices],
                    synapse=target,
                    synapse_index=synapse_index[indices],
                    weight=_slice_parameter(weight, indices, n_contact=len(synapse)),
                    delay=_slice_parameter(delay, indices, n_contact=len(synapse)),
                )
            )
        return tuple(connections)

    def _contact_edge_index(self, edge_set: EdgeSet) -> np.ndarray:
        if self.contact_pre_index is not None:
            return _match_contacts_to_edges(edge_set, self.contact_pre_index, self.contact_post_index)
        if self.edge_index is None:
            return np.arange(edge_set.n_edge, dtype=np.int32)
        return self.edge_index

    def _contact_targets(
        self,
        pre_index: np.ndarray,
        post_index: np.ndarray,
        *,
        pool_size: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_edge = int(pre_index.shape[0])
        if self.synapse_pool.shape != ():
            if self.number != 1:
                raise ValueError("Projection number > 1 requires scalar synapse_pool.")
            if self.synapse_pool.shape != (n_edge,):
                raise ValueError(
                    f"Projection synapse_pool must be scalar or shape {(n_edge,)!r}, "
                    f"got {self.synapse_pool.shape!r}."
                )
            synapse = self.synapse_pool.astype(object, copy=False)
            synapse_index = np.zeros((n_edge,), dtype=np.int32)
            return synapse, synapse_index, pre_index, post_index

        n_contact = n_edge * self.number
        target = str(self.synapse_pool.reshape(()))
        synapse = np.full((n_contact,), target, dtype=object)
        expanded_pre = np.repeat(pre_index, self.number)
        expanded_post = np.repeat(post_index, self.number)
        if pool_size is None:
            pool_size = 1
        synapse_index = self._select_synapse_indices(
            base_post_index=post_index,
            pool_size=pool_size,
        )
        return synapse, synapse_index, expanded_pre, expanded_post

    def _select_synapse_indices(self, *, base_post_index: np.ndarray, pool_size: int) -> np.ndarray:
        pool_size = _positive_int(pool_size, "pool_size")
        n_contact = int(base_post_index.shape[0]) * self.number
        if n_contact == 0:
            return np.asarray([], dtype=np.int32)
        rng = np.random.default_rng(self.seed)
        if self.target_policy == "per_edge":
            if not self.replace and self.number > pool_size:
                raise ValueError(
                    "Projection target selection with replace=False requires "
                    f"number <= pool_size; got number={self.number!r}, pool_size={pool_size!r}."
                )
            selected = [
                rng.choice(pool_size, size=self.number, replace=self.replace)
                for _ in range(int(base_post_index.shape[0]))
            ]
            return np.asarray(selected, dtype=np.int32).reshape(-1)
        if self.target_policy == "by_post":
            result = np.empty((n_contact,), dtype=np.int32)
            for post in tuple(dict.fromkeys(base_post_index.tolist())):
                edge_positions = np.nonzero(base_post_index == post)[0]
                contact_positions = np.concatenate(
                    [
                        np.arange(edge_pos * self.number, (edge_pos + 1) * self.number)
                        for edge_pos in edge_positions
                    ]
                )
                demand = int(contact_positions.shape[0])
                if not self.replace and demand > pool_size:
                    raise ValueError(
                        "Projection target_policy='by_post' with replace=False requires "
                        f"incoming_edge_count * number <= pool_size for each post; "
                        f"post_index={int(post)!r}, demand={demand!r}, pool_size={pool_size!r}."
                    )
                result[contact_positions] = rng.choice(
                    pool_size,
                    size=demand,
                    replace=self.replace,
                )
            return result
        raise ValueError(f"Unknown target_policy {self.target_policy!r}.")


def contacts(
    name: str,
    edges: str | EdgeSet,
    contact_triples,
    *,
    weight=None,
    delay=0.0 * u.ms,
) -> Projection:
    """Build a projection from explicit ``(pre, post, synapse)`` contacts."""
    arr = np.asarray(contact_triples, dtype=object)
    if arr.size == 0:
        arr = arr.reshape((0, 3))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("projection.contacts(...) expects shape (n_contact, 3).")
    return Projection(
        name=name,
        edges=edges,
        contact_pre_index=arr[:, 0].astype(np.int32),
        contact_post_index=arr[:, 1].astype(np.int32),
        synapse_pool=arr[:, 2],
        weight=weight,
        delay=delay,
    )


def _contact_parameter(value, *, base_n_edge: int, n_contact: int, number: int):
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        decimal = np.asarray(value.to_decimal(value.unit))
        if decimal.shape == ():
            return value
        if decimal.shape == (base_n_edge,):
            return u.Quantity(np.repeat(decimal, number), value.unit)
        if decimal.shape == (n_contact,):
            return value
        raise ValueError(
            "Projection parameter must be scalar, per-edge, or per-contact; "
            f"got {decimal.shape!r}, expected {(base_n_edge,)!r} or {(n_contact,)!r}."
        )
    arr = np.asarray(value)
    if arr.shape == ():
        return value
    if arr.shape == (base_n_edge,):
        return np.repeat(arr, number)
    if arr.shape == (n_contact,):
        return value
    raise ValueError(
        "Projection parameter must be scalar, per-edge, or per-contact; "
        f"got {arr.shape!r}, expected {(base_n_edge,)!r} or {(n_contact,)!r}."
    )


def _normalize_synapse_pool(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=object)
    if arr.shape == ():
        _validate_name(str(arr.reshape(())), "synapse_pool")
        return arr
    if arr.ndim != 1:
        raise ValueError(f"Projection synapse_pool must be scalar or one-dimensional, got {arr.shape!r}.")
    for item in arr:
        _validate_name(item, "synapse_pool")
    return arr


def _slice_parameter(value, indices: np.ndarray, *, n_contact: int):
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        decimal = np.asarray(value.to_decimal(value.unit))
        if decimal.shape == ():
            return value
        if decimal.shape != (n_contact,):
            raise ValueError(
                f"Projection parameter must be scalar or shape {(n_contact,)!r}, "
                f"got {decimal.shape!r}."
            )
        return u.Quantity(decimal[indices], value.unit)
    arr = np.asarray(value)
    if arr.shape == ():
        return value
    if arr.shape != (n_contact,):
        raise ValueError(
            f"Projection parameter must be scalar or shape {(n_contact,)!r}, got {arr.shape!r}."
        )
    return arr[indices]


def _match_contacts_to_edges(edge_set: EdgeSet, pre_index: np.ndarray, post_index: np.ndarray) -> np.ndarray:
    lookup = {
        (int(pre), int(post)): idx
        for idx, (pre, post) in enumerate(zip(edge_set.pre_index, edge_set.post_index))
    }
    result = []
    for pre, post in zip(pre_index, post_index):
        key = (int(pre), int(post))
        if key not in lookup:
            raise KeyError(
                f"Projection contact {(int(pre), int(post))!r} is not present in "
                f"EdgeSet {edge_set.name!r}."
            )
        result.append(lookup[key])
    return np.asarray(result, dtype=np.int32)


def _normalize_target_policy(value: object) -> Literal["per_edge", "by_post"]:
    if value not in {"per_edge", "by_post"}:
        raise ValueError("Projection target_policy must be 'per_edge' or 'by_post'.")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"Projection {name} must be an integer, got {type(value).__name__!s}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"Projection {name} must be > 0, got {value!r}.")
    return value


def _validate_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Projection {field_name} must be a non-empty string.")
    return value
