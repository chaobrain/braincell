"""Reusable cell-level edge sets and user-facing edge builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .connection import _as_index_array

__all__ = [
    "EdgeMethod",
    "EdgeSet",
    "all_to_all",
    "build",
    "pairs",
    "probability",
]


@dataclass(frozen=True)
class EdgeMethod:
    """Pure edge-generation method for :meth:`Network.add_edges`."""

    kind: Literal["all_to_all", "pairs", "probability"]
    params: dict


def all_to_all(*, pre_indices="all", post_indices="all", allow_self: bool = True) -> EdgeMethod:
    """Return an all-to-all edge-generation method."""
    return EdgeMethod(
        "all_to_all",
        {
            "pre_indices": pre_indices,
            "post_indices": post_indices,
            "allow_self": allow_self,
        },
    )


def pairs(edge_pairs) -> EdgeMethod:
    """Return an explicit ``(pre, post)`` pair edge-generation method."""
    return EdgeMethod("pairs", {"pairs": edge_pairs})


def probability(*, p: float, seed: int | None = None, allow_self: bool = True) -> EdgeMethod:
    """Return an independent Bernoulli edge-generation method."""
    return EdgeMethod("probability", {"p": p, "seed": seed, "allow_self": allow_self})


@dataclass
class EdgeSet:
    """Named sparse cell-cell edge table.

    Parameters
    ----------
    name : str
        Edge set name.
    pre_population : str
        Presynaptic population name.
    post_population : str
        Postsynaptic population name.
    pre_index : array-like of int
        Presynaptic population indices, one per cell-level edge.
    post_index : array-like of int
        Postsynaptic population indices, one per cell-level edge.
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


def build(
    *,
    name: str,
    pre: str,
    post: str,
    method: EdgeMethod,
    n_pre: int,
    n_post: int,
) -> EdgeSet:
    """Build an edge set from a user-facing edge-generation method."""
    if not isinstance(method, EdgeMethod):
        raise TypeError(
            f"Network.add_edges(...) method must be an EdgeMethod, got {type(method).__name__!s}."
        )
    if method.kind == "pairs":
        return _pairs_edge_set(name, pre, post, method.params["pairs"])
    if method.kind == "all_to_all":
        return _all_to_all_edge_set(
            name,
            pre,
            post,
            n_pre=n_pre,
            n_post=n_post,
            **method.params,
        )
    if method.kind == "probability":
        return _probability_edge_set(
            name,
            pre,
            post,
            n_pre=n_pre,
            n_post=n_post,
            **method.params,
        )
    raise ValueError(f"Unknown edge method {method.kind!r}.")


def _pairs_edge_set(name: str, pre: str, post: str, edge_pairs) -> EdgeSet:
    """Build an edge set from explicit ``(pre, post)`` index pairs."""
    arr = np.asarray(edge_pairs)
    if arr.size == 0:
        arr = arr.reshape((0, 2))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("edges.pairs(...) expects shape (n_edge, 2).")
    return EdgeSet(
        name=name,
        pre_population=pre,
        post_population=post,
        pre_index=arr[:, 0],
        post_index=arr[:, 1],
    )


def _all_to_all_edge_set(
    name: str,
    pre: str,
    post: str,
    *,
    n_pre: int,
    n_post: int,
    pre_indices="all",
    post_indices="all",
    allow_self: bool = True,
) -> EdgeSet:
    """Build all-to-all cell-level edges over selected indices."""
    pre_sel = _normalize_selection(pre_indices, n_pre, name="pre_indices")
    post_sel = _normalize_selection(post_indices, n_post, name="post_indices")
    pre_grid, post_grid = np.meshgrid(pre_sel, post_sel, indexing="ij")
    pre_index = pre_grid.reshape(-1)
    post_index = post_grid.reshape(-1)
    if not allow_self and pre == post:
        keep = pre_index != post_index
        pre_index = pre_index[keep]
        post_index = post_index[keep]
    return EdgeSet(name, pre, post, pre_index, post_index)


def _probability_edge_set(
    name: str,
    pre: str,
    post: str,
    *,
    n_pre: int,
    n_post: int,
    p: float,
    seed: int | None = None,
    allow_self: bool = True,
) -> EdgeSet:
    """Sample independent Bernoulli cell-level edges."""
    n_pre = _positive_int(n_pre, "n_pre")
    n_post = _positive_int(n_post, "n_post")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability p must be in [0, 1], got {p!r}.")

    rng = np.random.default_rng(seed)
    mask = rng.random((n_pre, n_post)) < p
    if not allow_self and pre == post:
        np.fill_diagonal(mask, False)
    pre_index, post_index = np.nonzero(mask)
    return EdgeSet(name, pre, post, pre_index, post_index)


def _normalize_selection(value, size: int, *, name: str) -> np.ndarray:
    size = _positive_int(size, "size")
    if isinstance(value, str):
        if value != "all":
            raise ValueError(f"{name} must be 'all' or integer indices, got {value!r}.")
        return np.arange(size, dtype=np.int32)
    indices = _as_index_array(value, name=name)
    if indices.size == 0:
        return indices
    min_index = int(np.min(indices))
    max_index = int(np.max(indices))
    if min_index < 0 or max_index >= size:
        raise IndexError(
            f"{name} out of range for population size {size!r}: "
            f"min={min_index!r}, max={max_index!r}."
        )
    return indices


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__!s}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return value


def _validate_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"EdgeSet {field_name} must be a non-empty string.")
    return value
