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
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
from brainstate.typing import ArrayLike

from .config import (
    alpha_for_2d_line as _alpha_for_2d_line,
    alpha_for_2d_poly as _alpha_for_2d_poly,
    alpha_for_3d_tube as _alpha_for_3d_tube,
    color_for_branch_type as _color_for_branch_type,
)

if TYPE_CHECKING:
    from braincell.filter import LocsetMask, RegionMask
    from braincell.morph import Morphology


def color_for_branch_type(branch_type: str) -> tuple[int, int, int]:
    return _color_for_branch_type(branch_type)


def alpha_for_2d_line() -> float:
    return _alpha_for_2d_line()


def alpha_for_2d_poly() -> float:
    return _alpha_for_2d_poly()


def alpha_for_3d_tube() -> float:
    return _alpha_for_3d_tube()


# ---------------------------------------------------------------------------
# Overlay input spec (what the user passes to plot2d / plot3d)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverlaySpec:
    """User-facing overlay request passed through ``plot2d`` / ``plot3d``.

    The scene builders translate this into concrete overlay *primitives*
    (``HighlightStroke2D`` / ``Marker2D`` / ``HighlightStroke3D`` /
    ``Marker3D``) that the backends then render on top of the base scene.

    Fields are plain masks so callers can build them with
    ``region_expr.evaluate(morpho)`` / ``locset_expr.evaluate(morpho)``.
    """

    region: "RegionMask | None" = None
    locset: "LocsetMask | None" = None
    values: ArrayLike | None = None


# ---------------------------------------------------------------------------
# 3D scene primitives
# ---------------------------------------------------------------------------

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
    opacity: float
    branch_indices: tuple[int, ...]
    branch_names: tuple[str, ...]
    points_um: np.ndarray
    radii_um: np.ndarray
    lines: np.ndarray


@dataclass(frozen=True)
class HighlightStroke3D:
    """Polyline fragment emitted for a region interval overlay in 3D.

    The backend renders this as an accent-colored stroke on top of the
    base tube/skeleton for the affected branch.
    """

    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray
    radii_um: np.ndarray
    color_rgb: tuple[int, int, int]
    opacity: float = 1.0


@dataclass(frozen=True)
class Marker3D:
    """Scatter marker emitted from a locset point in 3D."""

    branch_index: int
    x: float
    position_um: np.ndarray
    color_rgb: tuple[int, int, int]
    radius_um: float = 1.5


# ---------------------------------------------------------------------------
# 2D scene primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Polyline2D:
    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray
    widths_um: np.ndarray
    color_rgb: tuple[int, int, int]
    alpha: float = 1.0
    draw_order: int = 0


@dataclass(frozen=True)
class Polygon2D:
    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray
    color_rgb: tuple[int, int, int]
    alpha: float = 1.0
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
class HighlightStroke2D:
    """Polyline fragment emitted for a region interval overlay in 2D."""

    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray
    color_rgb: tuple[int, int, int]
    linewidth: float
    alpha: float = 1.0
    draw_order: int = 0


@dataclass(frozen=True)
class Marker2D:
    """Scatter marker emitted from a locset point in 2D."""

    branch_index: int
    x: float
    position_um: np.ndarray
    color_rgb: tuple[int, int, int]
    size: float = 30.0
    draw_order: int = 0


# ---------------------------------------------------------------------------
# Scene containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderScene3D:
    branches: tuple[BranchPolyline3D, ...]
    batches: tuple[BranchTypeBatch3D, ...]
    highlight_strokes: tuple[HighlightStroke3D, ...] = ()
    markers: tuple[Marker3D, ...] = ()
    mode: str = "geometry"


@dataclass(frozen=True)
class RenderScene2D:
    polylines: tuple[Polyline2D, ...] = ()
    polygons: tuple[Polygon2D, ...] = ()
    circles: tuple[Circle2D, ...] = ()
    labels: tuple[Label2D, ...] = ()
    highlight_strokes: tuple[HighlightStroke2D, ...] = ()
    markers: tuple[Marker2D, ...] = ()
    draw_order: tuple[int, ...] = ()
    projection_plane: str | None = None
    layout: str = "projected"
    shape: str = "line"


# ---------------------------------------------------------------------------
# Render request — neutral schema with a backend_options escape hatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderRequest:
    """Dispatched to a backend's ``render`` method.

    Backend-specific parameters (matplotlib ``ax``, pyvista ``notebook``,
    ``jupyter_backend``, ``return_plotter``) live in ``backend_options``
    so that adding a new backend does not require editing this schema.
    """

    morpho: "Morphology"
    scene: RenderScene2D | RenderScene3D | None = None
    overlay: OverlaySpec = field(default_factory=OverlaySpec)
    dimensionality: str = "3d"
    mode: str | None = None
    layout: str | None = None
    shape: str | None = None
    backend_options: Mapping[str, Any] = field(default_factory=dict)
