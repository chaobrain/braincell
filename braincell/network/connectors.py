"""Pure sparse connection builders."""

from __future__ import annotations

import brainunit as u
import numpy as np

from .connection import Connection

__all__ = ["pairs", "probability"]


def pairs(
    pre: str,
    post: str,
    edge_pairs,
    *,
    synapse: str,
    weight=None,
    delay=0.0 * u.ms,
) -> Connection:
    """Build a connection from explicit ``(pre, post)`` index pairs."""
    arr = np.asarray(edge_pairs)
    if arr.size == 0:
        arr = arr.reshape((0, 2))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "connectors.pairs(...) expects an array-like with shape (n_edge, 2)."
        )
    return Connection(
        pre_population=pre,
        post_population=post,
        pre_index=arr[:, 0],
        post_index=arr[:, 1],
        synapse=synapse,
        weight=weight,
        delay=delay,
    )


def probability(
    pre: str,
    post: str,
    *,
    n_pre: int,
    n_post: int,
    p: float,
    synapse: str,
    weight=None,
    seed: int | None = None,
    allow_self: bool = True,
    delay=0.0 * u.ms,
) -> Connection:
    """Sample independent Bernoulli edges between two populations."""
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
    return Connection(
        pre_population=pre,
        post_population=post,
        pre_index=pre_index,
        post_index=post_index,
        synapse=synapse,
        weight=weight,
        delay=delay,
    )


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__!s}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return value
