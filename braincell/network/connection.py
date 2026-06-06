"""Sparse connection declarations."""

from __future__ import annotations

from dataclasses import dataclass, field

import brainunit as u
import numpy as np


@dataclass
class Connection:
    """Inspectible sparse edge table between two named populations.

    Parameters
    ----------
    pre_population : str
        Presynaptic population name.
    post_population : str
        Postsynaptic population name.
    pre_index : array-like of int
        Presynaptic population indices, one per edge.
    post_index : array-like of int
        Postsynaptic population indices, one per edge.
    synapse : str
        Postsynaptic synapse instance name.
    synapse_index : array-like of int, optional
        Local active-point index in the target synapse layout, one per edge.
        Defaults to the first target, which preserves single-point synapse
        behavior.
    weight : object
        Scalar or per-edge event payload. ``None`` uses the placed synapse's
        default ``weight`` parameter when available.
    delay : object
        Scalar time delay. ``0 ms`` uses next-step delivery.
    """

    pre_population: str
    post_population: str
    pre_index: object
    post_index: object
    synapse: str
    weight: object | None = None
    delay: object = field(default_factory=lambda: 0.0 * u.ms)
    synapse_index: object | None = None

    def __post_init__(self) -> None:
        self.pre_population = _validate_name(self.pre_population, "pre_population")
        self.post_population = _validate_name(self.post_population, "post_population")
        self.synapse = _validate_name(self.synapse, "synapse")
        self.pre_index = _as_index_array(self.pre_index, name="pre_index")
        self.post_index = _as_index_array(self.post_index, name="post_index")
        if self.pre_index.shape != self.post_index.shape:
            raise ValueError(
                "Connection pre_index and post_index must have the same shape; "
                f"got {self.pre_index.shape!r} and {self.post_index.shape!r}."
            )
        if self.synapse_index is None:
            self.synapse_index = np.zeros(self.pre_index.shape, dtype=np.int32)
        else:
            self.synapse_index = _as_index_array(self.synapse_index, name="synapse_index")
            if self.synapse_index.shape != self.pre_index.shape:
                raise ValueError(
                    "Connection synapse_index must have the same shape as pre_index; "
                    f"got {self.synapse_index.shape!r} and {self.pre_index.shape!r}."
                )

    @property
    def n_edge(self) -> int:
        """Number of sparse edges in the table."""
        return int(self.pre_index.shape[0])


def _validate_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Connection {field_name} must be a non-empty string.")
    return value


def _as_index_array(value: object, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"Connection {name} must be one-dimensional, got {arr.shape!r}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Connection {name} must contain integers, got {arr.dtype!r}.")
    return arr.astype(np.int32, copy=False)
