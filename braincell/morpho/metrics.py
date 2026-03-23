from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .morpho import Morpho

Quantity = Any


@dataclass(frozen=True)
class MorphMetrics:
    morpho: Morpho

    def path_length_to_root(self, branch_index: int) -> Quantity:
        raise NotImplementedError

    def shortest_path_length(
        self,
        from_site: tuple[int, float],
        to_site: tuple[int, float],
    ) -> Quantity:
        raise NotImplementedError

    def total_surface_area(self) -> Quantity:
        raise NotImplementedError
