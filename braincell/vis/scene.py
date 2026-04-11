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
from __future__ import annotations
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


# ---------------------------------------------------------------------------
# Value spec — styling for color-by-values overlays (M6 Phase 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueSpec:
    """Styling parameters for a color-by-values overlay.

    Parameters
    ----------
    values : ArrayLike
        Per-element scalar array. The number of elements is interpreted
        against the morphology:

        * ``n_branches`` — one scalar per branch (the whole branch is
          shaded with that single colour);
        * total segment count — one scalar per segment (matplotlib
          ``LineCollection`` / ``PolyCollection`` uses the corresponding
          per-segment colour);
        * total centerline-point count (``segments + branches``) — one
          scalar per polyline point (used directly as
          ``polydata.point_data``).

        Values may carry ``brainunit`` units; the units are used to
        generate a default colourbar label when ``unit_label`` is unset.
    cmap : str
        Matplotlib colormap name, forwarded to
        :class:`matplotlib.cm.ScalarMappable` and PyVista ``add_mesh``.
    vmin, vmax : float or None
        Fixed colour-scale bounds. When either is ``None`` the missing
        bound is derived from the data range.
    norm : object or None
        Optional matplotlib ``Normalize``-compatible object. Takes
        precedence over *vmin*/*vmax* for the 2D backend.
    label : str or None
        Colourbar title. When ``None`` no title is drawn.
    unit_label : str or None
        Optional unit string appended to *label* on the colourbar.
    show_colorbar : bool
        Whether to render a colourbar alongside the scene (matplotlib
        only; PyVista always draws its own scalar bar when scalars are
        present).
    """

    values: ArrayLike
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    norm: Any | None = None
    label: str | None = None
    unit_label: str | None = None
    show_colorbar: bool = True


# ---------------------------------------------------------------------------
# Resolved per-branch value arrays
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BranchValues:
    """Per-branch centerline-point scalar array.

    The scene builders use :class:`BranchValues` as the canonical in-scene
    representation of a color-by-values request. One ``BranchValues``
    entry exists per branch; ``point_values`` stores a scalar for every
    centerline point (one more than the number of segments).
    """

    branch_index: int
    point_values: np.ndarray  # shape (n_points,)

    @property
    def segment_values(self) -> np.ndarray:
        """Per-segment midpoint scalars: ``0.5 * (v[i] + v[i+1])``."""
        if self.point_values.size <= 1:
            return self.point_values.copy()
        return 0.5 * (self.point_values[:-1] + self.point_values[1:])


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

    ``values`` may be either a bare array (interpreted with default
    styling) or a :class:`ValueSpec` carrying colormap / bounds / label
    information.
    """

    region: "RegionMask | None" = None
    locset: "LocsetMask | None" = None
    values: "ValueSpec | ArrayLike | None" = None

    def values_spec(self) -> "ValueSpec | None":
        """Return the normalized :class:`ValueSpec`, or ``None``.

        Accepts either a bare array (upgraded to ``ValueSpec`` with
        default styling) or an already-constructed :class:`ValueSpec`.
        """
        if self.values is None:
            return None
        if isinstance(self.values, ValueSpec):
            return self.values
        return ValueSpec(values=self.values)


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
class ValueBatch3D:
    """Scalar-valued PolyData batch for a color-by-values 3D scene.

    Mirrors :class:`BranchTypeBatch3D` but carries a per-point scalar
    array consumed by ``PyVista.add_mesh(scalars=...)``. One
    ``ValueBatch3D`` is emitted per branch type so that the batch's
    geometry can still be grouped the same way the base renderer does.
    """

    branch_type: str
    branch_indices: tuple[int, ...]
    branch_names: tuple[str, ...]
    points_um: np.ndarray
    radii_um: np.ndarray
    lines: np.ndarray
    point_values: np.ndarray  # shape (n_points,)
    opacity: float


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
class PolylineValues2D:
    """Per-segment scalar-valued polyline for a single branch in 2D.

    Emitted by the scene builder when the caller supplies ``values=``
    and ``shape='line'``. The matplotlib backend vectorizes rendering
    via :class:`matplotlib.collections.LineCollection`, with one
    segment drawn per consecutive pair of points and one scalar per
    segment from :attr:`segment_values`.
    """

    branch_index: int
    branch_name: str
    branch_type: str
    points_um: np.ndarray  # shape (n_points, 2)
    segment_values: np.ndarray  # shape (n_points - 1,)
    widths_um: np.ndarray  # shape (n_points,) — per-point centerline diameter
    draw_order: int = 0


@dataclass(frozen=True)
class PolygonValuesBatch2D:
    """Batched scalar-valued quad polygons for a single branch (frustum).

    For ``shape='frustum'`` each segment is drawn as a trapezoid with
    a per-polygon scalar; using a batched primitive lets the
    matplotlib backend materialise the whole branch as a single
    :class:`matplotlib.collections.PolyCollection`.
    """

    branch_index: int
    branch_name: str
    branch_type: str
    polygons_um: np.ndarray  # shape (n_segments, 4, 2)
    polygon_values: np.ndarray  # shape (n_segments,)
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
    value_batches: tuple[ValueBatch3D, ...] = ()
    value_spec: ValueSpec | None = None
    mode: str = "geometry"


@dataclass(frozen=True)
class RenderScene2D:
    polylines: tuple[Polyline2D, ...] = ()
    polygons: tuple[Polygon2D, ...] = ()
    circles: tuple[Circle2D, ...] = ()
    labels: tuple[Label2D, ...] = ()
    highlight_strokes: tuple[HighlightStroke2D, ...] = ()
    markers: tuple[Marker2D, ...] = ()
    polyline_values: tuple[PolylineValues2D, ...] = ()
    polygon_value_batches: tuple[PolygonValuesBatch2D, ...] = ()
    value_spec: ValueSpec | None = None
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
