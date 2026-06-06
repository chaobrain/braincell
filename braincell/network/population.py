"""Population declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Population:
    """Named homogeneous cell population.

    Parameters
    ----------
    name : str
        Population name.
    cell : object
        Cell-like object exposing one-dimensional ``pop_size``.
    """

    name: str
    cell: object

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Population name must be a non-empty string.")
        pop_size = tuple(getattr(self.cell, "pop_size", ()))
        if len(pop_size) != 1:
            raise ValueError(
                "Population cell.pop_size must be one-dimensional in network v1; "
                f"got {pop_size!r}."
            )
        if int(pop_size[0]) <= 0:
            raise ValueError(f"Population size must be > 0, got {pop_size!r}.")

    @property
    def size(self) -> int:
        """Number of cells in the population."""
        return int(tuple(getattr(self.cell, "pop_size"))[0])
