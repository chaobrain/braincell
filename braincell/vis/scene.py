# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



from dataclasses import dataclass, field
from typing import Any

import numpy as np

from braincell.filter import LocsetMask, RegionMask
from braincell.morpho import Morpho

ArrayLike = Any


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
class OverlaySpec:
    region: RegionMask | None = None
    locset: LocsetMask | None = None
    values: ArrayLike | None = None


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
class Polyline2D:
    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray
    widths_um: np.ndarray
    color_rgb: tuple[int, int, int]
    draw_order: int = 0


@dataclass(frozen=True)
class Polygon2D:
    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray
    color_rgb: tuple[int, int, int]
    draw_order: int = 0


@dataclass(frozen=True)
class Circle2D:
    center_um: np.ndarray
    radius_um: float
    color_rgb: tuple[int, int, int]
    draw_order: int = 0


@dataclass(frozen=True)
class Label2D:
    text: str
    position_um: np.ndarray
    color_rgb: tuple[int, int, int] = (0, 0, 0)
    draw_order: int = 0


@dataclass(frozen=True)
class RenderScene3D:
    branches: tuple[BranchPolyline3D, ...]
    batches: tuple[BranchTypeBatch3D, ...]


@dataclass(frozen=True)
class RenderScene2D:
    polylines: tuple[Polyline2D, ...] = ()
    polygons: tuple[Polygon2D, ...] = ()
    circles: tuple[Circle2D, ...] = ()
    labels: tuple[Label2D, ...] = ()
    draw_order: tuple[int, ...] = ()
    projection_plane: str | None = None
    mode: str = "projected"


@dataclass(frozen=True)
class RenderRequest:
    morpho: Morpho
    overlay: OverlaySpec = field(default_factory=OverlaySpec)
    dimensionality: str = "3d"
    mode: str | None = None
    scene: RenderScene2D | RenderScene3D | None = None
    notebook: bool | None = None
    jupyter_backend: str | None = None
    return_plotter: bool = False
