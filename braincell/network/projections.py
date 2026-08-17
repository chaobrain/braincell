"""Synaptic projections from cell-level edges to one synapse pool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import brainunit as u
import numpy as np

from .core import Connection, _as_index_array
from .edges import EdgeSet
from .pools import SynapsePool
from braincell.filter import LocsetExpr, RootLocation
from braincell.mech import Synapse


@dataclass(frozen=True)
class ProjectionEdgeContext:
    """Selected edge context passed to contact-generation methods.

    Parameters
    ----------
    edge_index : ndarray of int, shape ``(n_edge,)``
        Row ids in the source :class:`EdgeSet`.
    edge_pre_index, edge_post_index : ndarray of int, shape ``(n_edge,)``
        Presynaptic and postsynaptic cell ids for selected edges.
    pre_size, post_size : int
        Population sizes for the selected edge set.
    pool_size : int
        Number of local targets in the projection's postsynaptic synapse pool.
    synapse : str
        Target synapse pool name.
    """

    edge_index: np.ndarray
    edge_pre_index: np.ndarray
    edge_post_index: np.ndarray
    pre_size: int
    post_size: int
    pool_size: int
    synapse: str
    pool_post_index: np.ndarray | None = None

    @property
    def n_edge(self) -> int:
        """Number of selected cell-level edges."""
        return int(self.edge_index.shape[0])


@dataclass
class ContactTable:
    """Materialized contact topology for one projection.

    Parameters
    ----------
    source_edge : array-like of int, shape ``(n_contact,)``
        Index into the selected edge arrays in :class:`ProjectionEdgeContext`.
    synapse_index : array-like of int, shape ``(n_contact,)``
        Local target index in the projection's target synapse pool.

    Notes
    -----
    ``ContactTable`` stores contact topology only. It does not store the
    projection's synapse name, weight, or delay.
    """

    source_edge: object
    synapse_index: object

    def __post_init__(self) -> None:
        self.source_edge = _as_index_array(self.source_edge, name="source_edge")
        self.synapse_index = _as_index_array(self.synapse_index, name="synapse_index")
        if self.source_edge.shape != self.synapse_index.shape:
            raise ValueError(
                "ContactTable source_edge and synapse_index must have the same shape; "
                f"got {self.source_edge.shape!r} and {self.synapse_index.shape!r}."
            )

    @property
    def n_contact(self) -> int:
        """Number of materialized synaptic contacts."""
        return int(self.source_edge.shape[0])

    def __repr__(self) -> str:
        """Return a compact contact-table summary."""
        return f"ContactTable(n_contact={self.n_contact})"

    __str__ = __repr__


@dataclass(frozen=True)
class ProjectionContactContext:
    """Contact context passed to projection parameter callables.

    Parameters
    ----------
    edge_index, edge_pre_index, edge_post_index : ndarray of int
        Selected edge ids and their cell-level endpoints.
    source_edge : ndarray of int, shape ``(n_contact,)``
        Contact-to-selected-edge map.
    contact_pre_index, contact_post_index : ndarray of int, shape ``(n_contact,)``
        Cell-level endpoints for every materialized contact.
    synapse_index : ndarray of int, shape ``(n_contact,)``
        Local target index for every materialized contact.
    pre_size, post_size, pool_size : int
        Population and target-pool sizes.
    synapse : str
        Target synapse pool name.
    """

    edge_index: np.ndarray
    edge_pre_index: np.ndarray
    edge_post_index: np.ndarray
    source_edge: np.ndarray
    contact_pre_index: np.ndarray
    contact_post_index: np.ndarray
    synapse_index: np.ndarray
    pre_size: int
    post_size: int
    pool_size: int
    synapse: str

    @property
    def n_edge(self) -> int:
        """Number of selected cell-level edges."""
        return int(self.edge_index.shape[0])

    @property
    def n_contact(self) -> int:
        """Number of materialized synaptic contacts."""
        return int(self.source_edge.shape[0])


@dataclass(frozen=True)
class ContactMethod:
    """Callable contact-generation method.

    Parameters
    ----------
    builder : Callable
        Callable accepting :class:`ProjectionEdgeContext` and returning a
        :class:`ContactTable`.

    Notes
    -----
    Contact methods are construction-time, host-side functions. They decide
    how selected cell-level edges expand into local synapse contacts. Projection
    parameters such as ``weight`` and ``delay`` are resolved separately after
    contact generation.
    """

    builder: Callable

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError(f"ContactMethod builder must be callable, got {type(self.builder).__name__!s}.")

    def __call__(self, context: ProjectionEdgeContext) -> ContactTable:
        """Return a materialized contact table."""
        contacts = self.builder(context)
        if not isinstance(contacts, ContactTable):
            raise TypeError(f"Projection contact method must return ContactTable, got {type(contacts).__name__!s}.")
        return contacts


@dataclass
class Projection:
    """Named projection from one edge set to one postsynaptic synapse pool.

    Parameters
    ----------
    name : str
        Projection name.
    edges : str or EdgeSet
        Referenced edge-set name or object.
    synapse : str
        Postsynaptic synapse pool name.
    method : ContactMethod or Callable, optional
        Contact-generation method. Defaults to ``per_edge()``.
    edge_index : array-like of int, optional
        Selected EdgeSet rows. Defaults to all edges.
    weight : object or callable, optional
        Scalar, per-edge, per-contact, or callable event payload. ``None``
        uses the placed synapse default weight during lowering.
    delay : object or callable, optional
        Scalar, per-edge, per-contact, or callable delay.
    source : LocsetExpr, optional
        Single presynaptic location used for spike detection. Defaults to the
        midpoint of the morphology root branch.
    """

    name: str
    edges: str | EdgeSet
    synapse: str | Synapse
    pool: SynapsePool | None = None
    method: object | None = None
    edge_index: object | None = None
    weight: object | None = None
    delay: object = field(default_factory=lambda: 0.0 * u.ms)
    source: LocsetExpr = field(default_factory=lambda: RootLocation(0.5))

    def __post_init__(self) -> None:
        self.name = _validate_name(self.name, "name")
        if not isinstance(self.edges, (str, EdgeSet)):
            raise TypeError("Projection edges must be an EdgeSet or edge-set name.")
        if isinstance(self.edges, str):
            self.edges = _validate_name(self.edges, "edges")
        if isinstance(self.synapse, Synapse):
            if self.pool is None:
                raise ValueError("Projection with a Synapse declaration requires pool=synapse_pool(...).")
        elif isinstance(self.synapse, str):
            self.synapse = _validate_name(self.synapse, "synapse")
            if self.pool is not None:
                raise ValueError("Projection pool is only valid when synapse is a Synapse declaration.")
        else:
            raise TypeError("Projection synapse must be a name or braincell.mech.Synapse.")
        if not isinstance(self.source, LocsetExpr):
            raise TypeError(
                "Projection source must be a LocsetExpr resolving to one location, "
                f"got {type(self.source).__name__!s}."
            )
        self.method = _normalize_contact_method(self.method)
        if self.edge_index is not None:
            self.edge_index = _as_index_array(self.edge_index, name="edge_index")

    @property
    def edge_set_name(self) -> str:
        """Referenced edge-set name."""
        return self.edges if isinstance(self.edges, str) else self.edges.name

    @property
    def synapse_name(self) -> str:
        return self.synapse if isinstance(self.synapse, str) else self.synapse.instance_name

    def __repr__(self) -> str:
        """Return a compact projection summary."""
        return (
            f"Projection(name={self.name!r}, edges={self.edge_set_name!r}, "
            f"synapse={self.synapse_name!r}, edge_index={self.edge_index is not None}, "
            f"weight={self.weight is not None}, delay={self.delay is not None})"
        )

    __str__ = __repr__

    def to_connections(
        self,
        edge_set: EdgeSet,
        *,
        pre_size: int | None = None,
        post_size: int | None = None,
        pool_size: int | None = None,
        pool_post_index: object | None = None,
    ) -> tuple[Connection, ...]:
        """Expand this projection into runtime connection declarations.

        Parameters
        ----------
        edge_set : EdgeSet
            Cell-level edge set referenced by this projection.
        pre_size, post_size : int, optional
            Population sizes. Network lowering passes these explicitly; direct
            calls may omit them and infer minimum sizes from ``edge_set``.
        pool_size : int, optional
            Number of local targets in the postsynaptic synapse pool. Defaults
            to one target for direct single-target expansion.

        Returns
        -------
        tuple of Connection
            Runtime connection declarations for the materialized contacts.
        """
        edge_index = self._selected_edge_index(edge_set)
        _validate_edge_index(edge_index, edge_set)
        edge_pre_index = edge_set.pre_index[edge_index]
        edge_post_index = edge_set.post_index[edge_index]
        pre_size = _inferred_size(pre_size, edge_set.pre_index, "pre_size")
        post_size = _inferred_size(post_size, edge_set.post_index, "post_size")
        pool_size = _positive_int(1 if pool_size is None else pool_size, "pool_size")

        pool_owner = None
        if pool_post_index is not None:
            pool_owner = _as_index_array(pool_post_index, name="pool_post_index")
            if pool_owner.shape != (pool_size,):
                raise ValueError(
                    f"pool_post_index must have shape {(pool_size,)!r}, got {pool_owner.shape!r}."
                )
        edge_context = ProjectionEdgeContext(
            edge_index=edge_index,
            edge_pre_index=edge_pre_index,
            edge_post_index=edge_post_index,
            pre_size=pre_size,
            post_size=post_size,
            pool_size=pool_size,
            synapse=self.synapse_name,
            pool_post_index=pool_owner,
        )
        contacts = self.method(edge_context)
        _validate_contacts(contacts, context=edge_context)

        contact_pre_index = edge_pre_index[contacts.source_edge]
        contact_post_index = edge_post_index[contacts.source_edge]
        contact_context = ProjectionContactContext(
            edge_index=edge_index,
            edge_pre_index=edge_pre_index,
            edge_post_index=edge_post_index,
            source_edge=contacts.source_edge,
            contact_pre_index=contact_pre_index,
            contact_post_index=contact_post_index,
            synapse_index=contacts.synapse_index,
            pre_size=pre_size,
            post_size=post_size,
            pool_size=pool_size,
            synapse=self.synapse_name,
        )
        weight = _resolve_contact_parameter(self.weight, contact_context, name="weight")
        delay = _resolve_contact_parameter(self.delay, contact_context, name="delay")

        return (
            Connection(
                pre_population=edge_set.pre_population,
                post_population=edge_set.post_population,
                pre_index=contact_pre_index,
                post_index=contact_post_index,
                synapse=self.synapse_name,
                synapse_index=contacts.synapse_index,
                weight=weight,
                delay=delay,
                source=self.source,
            ),
        )

    def _selected_edge_index(self, edge_set: EdgeSet) -> np.ndarray:
        if self.edge_index is None:
            return np.arange(edge_set.n_edge, dtype=np.int32)
        return self.edge_index


def _normalize_contact_method(value) -> ContactMethod:
    if value is None:
        return per_edge()
    if isinstance(value, ContactMethod):
        return value
    if callable(value):
        return ContactMethod(value)
    raise TypeError(f"Projection method must be a ContactMethod or callable, got {type(value).__name__!s}.")


def _validate_edge_index(edge_index: np.ndarray, edge_set: EdgeSet) -> None:
    if edge_index.size == 0:
        return
    min_index = int(np.min(edge_index))
    max_index = int(np.max(edge_index))
    if min_index < 0 or max_index >= edge_set.n_edge:
        raise IndexError(
            f"Projection edge_index out of range for EdgeSet {edge_set.name!r}: min={min_index!r}, max={max_index!r}."
        )


def _validate_contacts(contacts: ContactTable, *, context: ProjectionEdgeContext) -> None:
    _validate_index_bounds(contacts.source_edge, context.n_edge, "source_edge")
    _validate_index_bounds(contacts.synapse_index, context.pool_size, "synapse_index")
    if contacts.n_contact and context.pool_post_index is not None:
        contact_post = context.edge_post_index[contacts.source_edge]
        target_post = context.pool_post_index[contacts.synapse_index]
        if np.any(contact_post != target_post):
            first = int(np.nonzero(contact_post != target_post)[0][0])
            raise ValueError(
                "Projection contact targets a synapse instance owned by a different "
                f"post cell at contact {first!r}: post_index={int(contact_post[first])!r}, "
                f"target_owner={int(target_post[first])!r}."
            )


def _validate_index_bounds(indices: np.ndarray, size: int, name: str) -> None:
    if indices.size == 0:
        return
    min_index = int(np.min(indices))
    max_index = int(np.max(indices))
    if min_index < 0 or max_index >= int(size):
        raise IndexError(f"Projection {name} out of range for size {size!r}: min={min_index!r}, max={max_index!r}.")


def _resolve_contact_parameter(value, context: ProjectionContactContext, *, name: str):
    if value is None:
        return None
    if callable(value):
        value = value(context)
    if isinstance(value, u.Quantity):
        decimal = np.asarray(value.to_decimal(value.unit))
        if decimal.shape == ():
            return value
        if decimal.shape == (context.n_edge,):
            return u.Quantity(decimal[context.source_edge], value.unit)
        if decimal.shape == (context.n_contact,):
            return value
        raise ValueError(
            f"Projection {name} must be scalar, shape {(context.n_edge,)!r}, "
            f"or shape {(context.n_contact,)!r}; got {decimal.shape!r}."
        )
    arr = np.asarray(value)
    if arr.shape == ():
        return value
    if arr.shape == (context.n_edge,):
        return arr[context.source_edge]
    if arr.shape == (context.n_contact,):
        return value
    raise ValueError(
        f"Projection {name} must be scalar, shape {(context.n_edge,)!r}, "
        f"or shape {(context.n_contact,)!r}; got {arr.shape!r}."
    )


def _normalize_number(value, context: ProjectionEdgeContext) -> np.ndarray:
    if callable(value):
        value = value(context)
    arr = np.asarray(value)
    if arr.shape == ():
        arr = np.broadcast_to(arr, (context.n_edge,)).copy()
    if arr.shape != (context.n_edge,):
        raise ValueError(f"Projection number must be scalar or shape {(context.n_edge,)!r}, got {arr.shape!r}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Projection number must contain integers, got {arr.dtype!r}.")
    arr = arr.astype(np.int32, copy=False)
    if np.any(arr < 0):
        raise ValueError("Projection number must be >= 0.")
    return arr


def _source_edges_from_counts(counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(counts.shape[0], dtype=np.int32), counts)


def _inferred_size(value: int | None, indices: np.ndarray, name: str) -> int:
    if value is not None:
        return _positive_int(value, name)
    if indices.size == 0:
        return 1
    return int(np.max(indices)) + 1


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


# ---------------------------------------------------------------------------
# Extensible contact-generation methods


def per_edge(number=1, *, replace: bool = True, seed: int | None = None) -> ContactMethod:
    """Return a contact method that samples targets independently per edge.

    Parameters
    ----------
    number : int, array-like of int, or callable, optional
        Number of contacts per selected edge. Callables receive
        :class:`ProjectionEdgeContext` and must return a scalar or shape
        ``(n_edge,)`` integer array.
    replace : bool, optional
        Whether each edge may reuse a local synapse target.
    seed : int or None, optional
        Random seed for reproducible target selection.
    """

    def _builder(context: ProjectionEdgeContext) -> ContactTable:
        counts = _normalize_number(number, context)
        source_edge = _source_edges_from_counts(counts)
        if source_edge.size == 0:
            return ContactTable(source_edge, np.asarray([], dtype=np.int32))
        rng = np.random.default_rng(seed)
        selected = []
        for edge_pos, count in enumerate(counts.tolist()):
            if int(count) <= 0:
                continue
            candidates = (
                np.arange(context.pool_size, dtype=np.int32)
                if context.pool_post_index is None
                else np.nonzero(
                    context.pool_post_index == int(context.edge_post_index[edge_pos])
                )[0].astype(np.int32, copy=False)
            )
            if not replace and int(count) > len(candidates):
                raise ValueError(
                    "per_edge(..., replace=False) requires edge number <= the post cell pool size."
                )
            if len(candidates) == 0:
                raise ValueError(
                    f"Projection post_index={int(context.edge_post_index[edge_pos])!r} has no synapse targets."
                )
            selected.append(rng.choice(candidates, size=int(count), replace=replace))
        synapse_index = np.concatenate(selected).astype(np.int32, copy=False)
        return ContactTable(source_edge, synapse_index)

    return ContactMethod(_builder)


def by_post(number=1, *, replace: bool = True, seed: int | None = None) -> ContactMethod:
    """Return a contact method that samples targets jointly per post cell.

    Parameters
    ----------
    number : int, array-like of int, or callable, optional
        Number of contacts per selected edge. Callables receive
        :class:`ProjectionEdgeContext`.
    replace : bool, optional
        Whether contacts landing on the same post cell may reuse local synapse
        targets.
    seed : int or None, optional
        Random seed for reproducible target selection.
    """

    def _builder(context: ProjectionEdgeContext) -> ContactTable:
        counts = _normalize_number(number, context)
        source_edge = _source_edges_from_counts(counts)
        if source_edge.size == 0:
            return ContactTable(source_edge, np.asarray([], dtype=np.int32))
        rng = np.random.default_rng(seed)
        synapse_index = np.empty((source_edge.shape[0],), dtype=np.int32)
        contact_post = context.edge_post_index[source_edge]
        for post in tuple(dict.fromkeys(contact_post.tolist())):
            positions = np.nonzero(contact_post == post)[0]
            demand = int(positions.shape[0])
            candidates = (
                np.arange(context.pool_size, dtype=np.int32)
                if context.pool_post_index is None
                else np.nonzero(context.pool_post_index == int(post))[0].astype(np.int32, copy=False)
            )
            if not replace and demand > len(candidates):
                raise ValueError(
                    "by_post(..., replace=False) requires total contacts per post "
                    f"<= pool_size; post_index={int(post)!r}, demand={demand!r}, "
                    f"pool_size={len(candidates)!r}."
                )
            if len(candidates) == 0:
                raise ValueError(f"Projection post_index={int(post)!r} has no synapse targets.")
            synapse_index[positions] = rng.choice(
                candidates,
                size=demand,
                replace=replace,
            )
        return ContactTable(source_edge, synapse_index)

    return ContactMethod(_builder)


def explicit_contacts(source_edge, synapse_index=0) -> ContactMethod:
    """Return a method from explicit contact topology.

    Parameters
    ----------
    source_edge : array-like of int, shape ``(n_contact,)``
        Index into the selected edge rows.
    synapse_index : int or array-like of int, optional
        Local synapse target index per contact. Scalars are broadcast.
    """

    def _builder(context: ProjectionEdgeContext) -> ContactTable:
        source = _as_index_array(source_edge, name="source_edge")
        target = np.asarray(synapse_index)
        if target.shape == ():
            target = np.broadcast_to(target, source.shape).copy()
        if target.shape != source.shape:
            raise ValueError(
                f"explicit_contacts synapse_index must be scalar or shape {source.shape!r}, got {target.shape!r}."
            )
        return ContactTable(source, target)

    return ContactMethod(_builder)
