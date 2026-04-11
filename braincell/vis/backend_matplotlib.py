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



import importlib.util
import sys
from dataclasses import dataclass

import numpy as np

from ._values import resolved_colorbar_label
from .scene import (
    HighlightStroke2D,
    Marker2D,
    Polygon2D,
    PolygonValuesBatch2D,
    Polyline2D,
    PolylineValues2D,
    RenderRequest,
    RenderScene2D,
    ValueSpec,
)

_BASE_OVERLAY_OFFSET = 10_000  # overlays drawn strictly above all base primitives


@dataclass(frozen=True)
class MatplotlibBackend:
    name: str = "matplotlib"
    supported_scene_kinds: frozenset[str] = frozenset({"2d"})
    background: str = "white"
    show_axes: bool = False

    def available(self) -> bool:
        try:
            return importlib.util.find_spec("matplotlib") is not None
        except ValueError:
            return "matplotlib" in sys.modules

    def render(self, request: RenderRequest) -> object:
        scene = request.scene
        if not isinstance(scene, RenderScene2D):
            raise ValueError("MatplotlibBackend requires RenderScene2D.")
        if not self.available():
            raise RuntimeError("Matplotlib backend is not available. Install matplotlib first.")

        import matplotlib.pyplot as plt

        ax = request.backend_options.get("ax") if request.backend_options else None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        fig.patch.set_facecolor(self.background)
        ax.set_facecolor(self.background)

        # Resolve the normalization once per scene so every value-bearing
        # primitive ends up in the same colormap.
        value_spec = scene.value_spec
        value_norm = _build_norm(scene, value_spec)

        # Render all base primitives in ``draw_order`` so test expectations
        # that inspect `ax.lines` / `ax.patches` match the data order the
        # scene declares, and so later primitives sit above earlier ones.
        # Base polylines use per-segment `ax.plot`; base polygons use
        # individual `Polygon` patches. Value-carrying primitives
        # (`polyline_values` / `polygon_value_batches`) always take the
        # vectorized LineCollection / PolyCollection fast path, which
        # is a 10–50× speedup on large morphologies when the user
        # supplies per-segment scalars.
        polygons = sorted(scene.polygons, key=_primitive_order)
        polylines = sorted(scene.polylines, key=_primitive_order)
        value_polylines = sorted(scene.polyline_values, key=_primitive_order)
        value_polygons = sorted(scene.polygon_value_batches, key=_primitive_order)

        for polygon in polygons:
            _draw_polygon(ax, plt, polygon)
        for polyline in polylines:
            _draw_polyline(ax, polyline)
        for batch in value_polygons:
            _draw_value_polygons(ax, batch, cmap=value_spec.cmap, norm=value_norm)
        for value_polyline in value_polylines:
            _draw_value_polyline(ax, value_polyline, cmap=value_spec.cmap, norm=value_norm)

        for circle in scene.circles:
            color = _rgb_to_float(circle.color_rgb)
            patch = plt.Circle(
                circle.center_um,
                circle.radius_um,
                color=color,
                fill=False,
                zorder=circle.draw_order,
            )
            ax.add_patch(patch)

        for label in scene.labels:
            color = _rgb_to_float(label.color_rgb)
            ax.text(
                label.position_um[0],
                label.position_um[1],
                label.text,
                color=color,
                zorder=label.draw_order,
            )

        for stroke in sorted(scene.highlight_strokes, key=_primitive_order):
            _draw_highlight_stroke(ax, stroke)

        for marker in sorted(scene.markers, key=_primitive_order):
            _draw_marker(ax, marker)

        if value_spec is not None and value_spec.show_colorbar and (value_polylines or value_polygons):
            _draw_colorbar(fig, ax, value_spec=value_spec, norm=value_norm, unit_label=value_spec.unit_label)

        _set_scene_limits(ax, scene)
        ax.set_aspect("equal", adjustable="datalim")
        if not self.show_axes:
            ax.axis("off")
        return ax


def _primitive_order(primitive) -> int:
    return getattr(primitive, "draw_order", 0)


def _rgb_to_float(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(float(channel) / 255.0 for channel in rgb)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Vectorized base primitives (PolyCollection for frustum; per-segment
# ``ax.plot`` for polylines so that `ax.lines` introspection remains
# cheap for tests and downstream tooling).
# ---------------------------------------------------------------------------

def _draw_polyline(ax, polyline: Polyline2D) -> None:
    color = _rgb_to_float(polyline.color_rgb)
    linewidth = max(float(np.mean(polyline.widths_um)), 0.5)
    ax.plot(
        polyline.points_um[:, 0],
        polyline.points_um[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=polyline.alpha,
        zorder=polyline.draw_order,
    )


def _draw_polygon(ax, plt, polygon: Polygon2D) -> None:
    color = _rgb_to_float(polygon.color_rgb)
    patch = plt.Polygon(
        polygon.points_um,
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=polygon.alpha,
        linewidth=1.0,
        zorder=polygon.draw_order,
    )
    ax.add_patch(patch)


# ---------------------------------------------------------------------------
# Color-by-values rendering
# ---------------------------------------------------------------------------

def _build_norm(scene: RenderScene2D, value_spec: ValueSpec | None):
    """Return a matplotlib Normalize covering every value-bearing primitive."""
    if value_spec is None:
        return None
    if value_spec.norm is not None:
        return value_spec.norm

    from matplotlib.colors import Normalize

    all_values: list[float] = []
    for polyline in scene.polyline_values:
        if polyline.segment_values.size:
            all_values.extend(float(v) for v in polyline.segment_values)
    for batch in scene.polygon_value_batches:
        if batch.polygon_values.size:
            all_values.extend(float(v) for v in batch.polygon_values)

    vmin = value_spec.vmin
    vmax = value_spec.vmax
    if vmin is None:
        vmin = float(min(all_values)) if all_values else 0.0
    if vmax is None:
        vmax = float(max(all_values)) if all_values else 1.0
    if vmin == vmax:
        # Avoid degenerate colourmap: pad to ±1 around the single value.
        vmin = vmin - 0.5
        vmax = vmax + 0.5
    return Normalize(vmin=vmin, vmax=vmax)


def _draw_value_polyline(
    ax,
    polyline: PolylineValues2D,
    *,
    cmap: str,
    norm,
) -> None:
    from matplotlib.collections import LineCollection

    pts = np.asarray(polyline.points_um, dtype=float)
    if pts.shape[0] < 2:
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    values = np.asarray(polyline.segment_values, dtype=float)
    widths = np.asarray(polyline.widths_um, dtype=float)
    per_seg_linewidth = np.maximum(
        0.5 * (widths[:-1] + widths[1:]) if widths.size >= 2 else widths,
        0.5,
    )
    lc = LineCollection(
        segments,
        array=values,
        cmap=cmap,
        norm=norm,
        linewidths=per_seg_linewidth,
        zorder=polyline.draw_order,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(lc)


def _draw_value_polygons(
    ax,
    batch: PolygonValuesBatch2D,
    *,
    cmap: str,
    norm,
) -> None:
    from matplotlib.collections import PolyCollection

    polygons_um = np.asarray(batch.polygons_um, dtype=float)
    if polygons_um.size == 0:
        return
    values = np.asarray(batch.polygon_values, dtype=float)
    pc = PolyCollection(
        list(polygons_um),
        array=values,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        zorder=batch.draw_order,
    )
    ax.add_collection(pc)


def _draw_colorbar(fig, ax, *, value_spec: ValueSpec, norm, unit_label: str | None) -> None:
    from matplotlib.cm import ScalarMappable

    mappable = ScalarMappable(norm=norm, cmap=value_spec.cmap)
    mappable.set_array(np.array([]))
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    label = resolved_colorbar_label(value_spec, unit_label)
    if label is not None:
        cbar.set_label(label)


# ---------------------------------------------------------------------------
# Overlays (highlight strokes, markers)
# ---------------------------------------------------------------------------

def _draw_highlight_stroke(ax, stroke: HighlightStroke2D) -> None:
    color = _rgb_to_float(stroke.color_rgb)
    ax.plot(
        stroke.points_um[:, 0],
        stroke.points_um[:, 1],
        color=color,
        linewidth=stroke.linewidth,
        alpha=stroke.alpha,
        zorder=_BASE_OVERLAY_OFFSET + stroke.draw_order,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def _draw_marker(ax, marker: Marker2D) -> None:
    color = _rgb_to_float(marker.color_rgb)
    ax.scatter(
        marker.position_um[0],
        marker.position_um[1],
        s=marker.size,
        c=[color],
        edgecolors="black",
        linewidths=0.5,
        zorder=_BASE_OVERLAY_OFFSET + marker.draw_order,
    )


def _set_scene_limits(ax, scene: RenderScene2D) -> None:
    bounds: list[np.ndarray] = []

    for polyline in scene.polylines:
        if polyline.points_um.size:
            bounds.append(np.asarray(polyline.points_um, dtype=float))

    for polyline in scene.polyline_values:
        if polyline.points_um.size:
            bounds.append(np.asarray(polyline.points_um, dtype=float))

    for polygon in scene.polygons:
        if polygon.points_um.size:
            bounds.append(np.asarray(polygon.points_um, dtype=float))

    for batch in scene.polygon_value_batches:
        if batch.polygons_um.size:
            bounds.append(batch.polygons_um.reshape(-1, 2))

    for circle in scene.circles:
        center = np.asarray(circle.center_um, dtype=float)
        radius = float(circle.radius_um)
        bounds.append(
            np.array(
                [
                    center + np.array([-radius, -radius], dtype=float),
                    center + np.array([radius, radius], dtype=float),
                ]
            )
        )

    for stroke in scene.highlight_strokes:
        if stroke.points_um.size:
            bounds.append(np.asarray(stroke.points_um, dtype=float))

    for marker in scene.markers:
        bounds.append(np.asarray(marker.position_um, dtype=float).reshape(1, -1))

    if not bounds:
        return

    all_points = np.vstack(bounds)
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    span_xy = max_xy - min_xy
    padding_xy = np.maximum(span_xy * 0.05, 1.0)

    ax.set_xlim(float(min_xy[0] - padding_xy[0]), float(max_xy[0] + padding_xy[0]))
    ax.set_ylim(float(min_xy[1] - padding_xy[1]), float(max_xy[1] + padding_xy[1]))
