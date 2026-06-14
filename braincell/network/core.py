"""Core network declarations shared by topology and runtime layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import brainunit as u
import jax
import numpy as np


@dataclass(frozen=True)
class Population:
    """Named one-dimensional cell population for network indexing.

    ``Population`` is the network-level wrapper around a homogeneous
    ``Cell`` object. BrainCell cells may use multidimensional ``pop_size``
    shapes, but network v1 addresses cells through one-dimensional edge and
    connection indices.

    Parameters
    ----------
    name : str
        Population name.
    cell : object
        Cell-like object exposing one-dimensional ``pop_size`` in network v1.
    """

    name: str
    cell: object

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Population name must be a non-empty string.")
        pop_size = tuple(getattr(self.cell, "pop_size", ()))
        if len(pop_size) != 1:
            # TODO(network): Cell supports multidimensional pop_size, but
            # network v1 uses flat one-dimensional edge indices and does not
            # yet reshape event buffers between flat network indices and
            # cell runtime buffers with shape ``pop_size + (...)``.
            raise ValueError(
                "Network v1 requires one-dimensional cell.pop_size; "
                "Cell supports multidimensional pop_size, but network indexing "
                "does not yet; "
                f"got {pop_size!r}."
            )
        if int(pop_size[0]) <= 0:
            raise ValueError(f"Population size must be > 0, got {pop_size!r}.")

    @property
    def size(self) -> int:
        """Number of cells in the population."""
        return int(tuple(getattr(self.cell, "pop_size"))[0])

    def __repr__(self) -> str:
        """Return a compact population summary."""
        return (
            f"Population(name={self.name!r}, size={self.size}, "
            f"cell={type(self.cell).__name__}, "
            f"initialized={bool(getattr(self.cell, '_initialized', False))})"
        )

    __str__ = __repr__


@dataclass
class Connection:
    """Inspectable sparse contact table between two named populations.

    Parameters
    ----------
    pre_population : str
        Presynaptic population name.
    post_population : str
        Postsynaptic population name.
    pre_index : array-like of int
        Presynaptic population indices, one per contact.
    post_index : array-like of int
        Postsynaptic population indices, one per contact.
    synapse : str
        Postsynaptic synapse pool name.
    synapse_index : array-like of int, optional
        Local active-point index in the target synapse layout, one per contact.
        Defaults to the first target, which preserves single-point synapse
        behavior.
    weight : object
        Scalar or per-contact event payload. ``None`` uses the placed
        synapse's default ``weight`` parameter when available.
    delay : object
        Scalar or per-contact time delay. ``0 ms`` uses next-step delivery.
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
    def n_contact(self) -> int:
        """Number of sparse contacts in the table."""
        return int(self.pre_index.shape[0])

    @property
    def n_edge(self) -> int:
        """Number of sparse contacts.

        Notes
        -----
        This alias is kept for the v1 transition from edge-level to
        contact-level naming.
        """
        return self.n_contact

    def __repr__(self) -> str:
        """Return a compact contact-table summary."""
        return (
            f"Connection(pre_population={self.pre_population!r}, "
            f"post_population={self.post_population!r}, synapse={self.synapse!r}, "
            f"n_contact={self.n_contact})"
        )

    __str__ = __repr__


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NetworkRunResult:
    """Return value of :meth:`braincell.network.Network.run`.

    Attributes
    ----------
    time : brainunit.Quantity
        Step times spanning ``[start_t, start_t + duration)``.
    traces : dict
        ``population_name -> {probe_name: trace}`` mapping.
    spikes : dict
        ``population_name -> spike_trace`` mapping.
    """

    time: object
    traces: dict
    spikes: dict


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
