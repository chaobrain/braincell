"""Cell-level adjacency builders for :mod:`braincell.network`.

This module only builds connectivity between cells in two named populations.
It does not choose postsynaptic synapse targets and does not attach weights or
delays. Those synapse-level concerns belong to :class:`Projection` and
:class:`Connection`.

All edge-generation methods are construction-time, host-side helpers. Their
outputs are materialized to the same sparse COO-style representation: one
NumPy ``int32`` ``pre_index`` array and one NumPy ``int32`` ``post_index``
array. Each row ``k`` means
``pre_population[pre_index[k]] -> post_population[post_index[k]]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .core import _as_index_array

__all__ = [
    "EdgeMethod",
    "EdgeSet",
    "all_pairs",
    "build",
    "dense",
    "pairs",
    "probability",
]


@dataclass
class EdgeSet:
    """Named sparse cell-to-cell adjacency table.

    Parameters
    ----------
    name : str
        Edge set name in the network.
    pre_population : str
        Presynaptic population name.
    post_population : str
        Postsynaptic population name.
    pre_index : array-like of int
        Presynaptic cell indices, one per cell-level edge.
    post_index : array-like of int
        Postsynaptic cell indices, one per cell-level edge.

    Notes
    -----
    ``EdgeSet`` stores only materialized cell-level adjacency. It has no
    synapse name, local synapse index, weight, or delay. Index arrays are
    normalized to host-side NumPy ``int32`` arrays for static topology
    processing; runtime delivery code is responsible for binding them to JAX
    arrays before calling scatter or ``brainevent`` kernels.
    """

    name: str
    pre_population: str
    post_population: str
    pre_index: object
    post_index: object

    def __post_init__(self) -> None:
        self.name = _validate_name(self.name, "name")
        self.pre_population = _validate_name(self.pre_population, "pre_population")
        self.post_population = _validate_name(self.post_population, "post_population")
        self.pre_index = _as_index_array(self.pre_index, name="pre_index")
        self.post_index = _as_index_array(self.post_index, name="post_index")
        if self.pre_index.shape != self.post_index.shape:
            raise ValueError(
                "EdgeSet pre_index and post_index must have the same shape; "
                f"got {self.pre_index.shape!r} and {self.post_index.shape!r}."
            )

    @property
    def n_edge(self) -> int:
        """Number of cell-level edges."""
        return int(self.pre_index.shape[0])

    def __repr__(self) -> str:
        """Return a compact edge-set summary."""
        return (
            f"EdgeSet(name={self.name!r}, pre_population={self.pre_population!r}, "
            f"post_population={self.post_population!r}, n_edge={self.n_edge})"
        )

    __str__ = __repr__


@dataclass(frozen=True)
class EdgeMethod:
    """Callable cell-adjacency generation method.

    Parameters
    ----------
    builder : Callable
        Callable with signature ``builder(*, n_pre, n_post)`` returning
        ``(pre_index, post_index)``. ``n_pre`` and ``n_post`` are supplied by
        :class:`Network` from the selected population sizes.

    Notes
    -----
    Builder callables are construction-time utilities, not JAX-traced runtime
    functions. They may return lists, tuples, NumPy arrays, or other array-like
    objects. Shared shape, dtype, and bounds validation is performed by
    :func:`build`, and accepted outputs are normalized into ``EdgeSet`` NumPy
    ``int32`` arrays.
    """

    builder: Callable

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError(
                f"EdgeMethod builder must be callable, got {type(self.builder).__name__!s}."
            )

    def __call__(self, *, n_pre: int, n_post: int):
        """Return sparse ``(pre_index, post_index)`` arrays."""
        return self.builder(n_pre=n_pre, n_post=n_post)


def build(
    *,
    name: str,
    pre: str,
    post: str,
    method: EdgeMethod | Callable,
    n_pre: int,
    n_post: int,
) -> EdgeSet:
    """Build a materialized cell-level edge set.

    Parameters
    ----------
    name : str
        Edge set name.
    pre, post : str
        Presynaptic and postsynaptic population names.
    method : EdgeMethod or Callable
        Edge generator. Plain callables must accept keyword arguments
        ``n_pre`` and ``n_post`` and return ``(pre_index, post_index)``.
    n_pre, n_post : int
        Population sizes supplied by :class:`Network`.

    Returns
    -------
    EdgeSet
        Materialized sparse cell-to-cell adjacency.

    Notes
    -----
    ``build`` is the boundary between user-provided edge-generation methods
    and network topology storage. It calls the method, normalizes the returned
    sparse indices through :class:`EdgeSet`, and checks that those indices fit
    within the selected population sizes.
    """
    n_pre = _positive_int(n_pre, "n_pre")
    n_post = _positive_int(n_post, "n_post")
    pre_index, post_index = _call_method(method, n_pre=n_pre, n_post=n_post)
    edge_set = EdgeSet(
        name=name,
        pre_population=pre,
        post_population=post,
        pre_index=pre_index,
        post_index=post_index,
    )
    _validate_edge_bounds(edge_set, n_pre=n_pre, n_post=n_post)
    return edge_set


def _call_method(method: EdgeMethod | Callable, *, n_pre: int, n_post: int):
    if isinstance(method, EdgeMethod):
        result = method(n_pre=n_pre, n_post=n_post)
    elif callable(method):
        result = method(n_pre=n_pre, n_post=n_post)
    else:
        raise TypeError(
            "Network.add_edges(...) method must be an EdgeMethod or callable, "
            f"got {type(method).__name__!s}."
        )
    try:
        pre_index, post_index = result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Edge method must return a pair ``(pre_index, post_index)``."
        ) from exc
    return pre_index, post_index


def _validate_edge_bounds(edge_set: EdgeSet, *, n_pre: int, n_post: int) -> None:
    _validate_index_bounds(edge_set.pre_index, n_pre, "pre_index")
    _validate_index_bounds(edge_set.post_index, n_post, "post_index")


def _validate_index_bounds(indices: np.ndarray, size: int, name: str) -> None:
    if indices.size == 0:
        return
    min_index = int(np.min(indices))
    max_index = int(np.max(indices))
    if min_index < 0 or max_index >= int(size):
        raise IndexError(
            f"EdgeSet {name} out of range for population size {size!r}: "
            f"min={min_index!r}, max={max_index!r}."
        )


def _validate_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"EdgeSet {field_name} must be a non-empty string.")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__!s}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return value


# ---------------------------------------------------------------------------
# Extensible edge-generation methods


def pairs(edge_pairs) -> EdgeMethod:
    """Return a method from explicit sparse ``(pre, post)`` pairs.

    Parameters
    ----------
    edge_pairs : array-like of int, shape ``(n_edge, 2)``
        Explicit cell-level edges.

    Returns
    -------
    EdgeMethod
        Method returning the provided sparse indices.

    Notes
    -----
    ``pairs`` is the lowest-level builder. Other builders ultimately produce
    the same sparse ``(pre_index, post_index)`` representation.
    """

    def _builder(*, n_pre: int, n_post: int):
        _ = (n_pre, n_post)
        arr = np.asarray(edge_pairs)
        if arr.size == 0:
            arr = arr.reshape((0, 2))
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("edges.pairs(...) expects shape (n_edge, 2).")
        return arr[:, 0], arr[:, 1]

    return EdgeMethod(_builder)


def dense(adjacency) -> EdgeMethod:
    """Return a method from a dense adjacency array.

    Parameters
    ----------
    adjacency : array-like
        Dense cell-level adjacency. Nonzero entries are converted to sparse
        ``(pre_index, post_index)`` pairs with :func:`numpy.nonzero`.

    Returns
    -------
    EdgeMethod
        Method returning sparse indices for nonzero adjacency entries.

    Notes
    -----
    The dense adjacency is used only during construction. The resulting
    ``EdgeSet`` stores sparse NumPy ``int32`` indices, not the dense matrix.
    """

    def _builder(*, n_pre: int, n_post: int):
        _ = (n_pre, n_post)
        arr = np.asarray(adjacency)
        if arr.ndim != 2:
            raise ValueError(f"edges.dense(...) expects a two-dimensional array, got {arr.shape!r}.")
        return np.nonzero(arr)

    return EdgeMethod(_builder)


def all_pairs(*, pre_indices="all", post_indices="all", allow_self: bool = True) -> EdgeMethod:
    """Return a method for all selected ``(pre, post)`` cell pairs.

    Parameters
    ----------
    pre_indices, post_indices : "all" or array-like of int, optional
        Presynaptic and postsynaptic cell selections. ``"all"`` expands to the
        full corresponding population size supplied by :class:`Network`.
    allow_self : bool, optional
        Whether to include diagonal ``i -> i`` edges when pre and post
        population sizes match.

    Returns
    -------
    EdgeMethod
        Method returning the selected Cartesian product as sparse indices.

    Notes
    -----
    ``all_pairs`` generates the Cartesian product of selected pre and post
    indices. It is a construction-time helper and does not preserve a lazy
    all-pairs representation after :func:`build`.
    """

    def _builder(*, n_pre: int, n_post: int):
        pre_sel = _normalize_selection(pre_indices, n_pre, name="pre_indices")
        post_sel = _normalize_selection(post_indices, n_post, name="post_indices")
        pre_grid, post_grid = np.meshgrid(pre_sel, post_sel, indexing="ij")
        pre_index = pre_grid.reshape(-1)
        post_index = post_grid.reshape(-1)
        if not allow_self and n_pre == n_post:
            keep = pre_index != post_index
            pre_index = pre_index[keep]
            post_index = post_index[keep]
        return pre_index, post_index

    return EdgeMethod(_builder)


def probability(*, p: float, seed: int | None = None, allow_self: bool = True) -> EdgeMethod:
    """Return a method for independent Bernoulli cell-level edges.

    Parameters
    ----------
    p : float
        Edge probability in ``[0, 1]``.
    seed : int or None, optional
        Random seed for reproducible sampling.
    allow_self : bool, optional
        Whether to include ``i -> i`` edges when pre and post population sizes
        match.

    Returns
    -------
    EdgeMethod
        Method returning sampled sparse indices.

    Notes
    -----
    Sampling is performed during edge-set construction. Reusing the same seed
    creates the same sampled sparse topology for the same population sizes.
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability p must be in [0, 1], got {p!r}.")

    def _builder(*, n_pre: int, n_post: int):
        rng = np.random.default_rng(seed)
        mask = rng.random((n_pre, n_post)) < p
        if not allow_self and n_pre == n_post:
            np.fill_diagonal(mask, False)
        return np.nonzero(mask)

    return EdgeMethod(_builder)


def _normalize_selection(value, size: int, *, name: str) -> np.ndarray:
    if isinstance(value, str):
        if value != "all":
            raise ValueError(f"{name} must be 'all' or integer indices, got {value!r}.")
        return np.arange(size, dtype=np.int32)
    return _as_index_array(value, name=name)
