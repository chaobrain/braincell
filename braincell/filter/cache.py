from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Quantity = Any


@dataclass
class SelectionCache:
    tree_distance_to_root: dict[int, Quantity] = field(default_factory=dict)
    euclidean_distance_to_root: dict[int, Quantity] = field(default_factory=dict)
    branch_radius_summary: dict[int, tuple[Quantity, Quantity]] = field(default_factory=dict)
