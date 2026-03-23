from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ALLEN_RGB_BY_TYPE = {
    "soma": (0, 0, 0),
    "axon": (70, 130, 180),
    "basal_dend": (178, 34, 34),
    "basal_dendrite": (178, 34, 34),
    "apical_dend": (255, 127, 80),
    "apical_dendrite": (255, 127, 80),
    "dend": (205, 92, 92),
    "custom": (110, 110, 110),
}


def color_for_branch_type(branch_type: str) -> tuple[int, int, int]:
    return ALLEN_RGB_BY_TYPE.get(branch_type, ALLEN_RGB_BY_TYPE["custom"])


@dataclass(frozen=True)
class BranchPolyline3D:
    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray
    radii_um: np.ndarray


@dataclass(frozen=True)
class BranchTypeBatch3D:
    branch_type: str
    color_rgb: tuple[int, int, int]
    branch_indices: tuple[int, ...]
    branch_names: tuple[str, ...]
    points_um: np.ndarray
    radii_um: np.ndarray
    lines: np.ndarray


@dataclass(frozen=True)
class RenderGeometry3D:
    branches: tuple[BranchPolyline3D, ...]
    batches: tuple[BranchTypeBatch3D, ...]
