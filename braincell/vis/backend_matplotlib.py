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

from .scene import (
    HighlightStroke2D,
    Marker2D,
    Polygon2D,
    Polyline2D,
    RenderRequest,
    RenderScene2D,
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

        # Render all base primitives in ``draw_order`` so test expectations
        # that inspect `ax.lines` / `ax.patches` match the data order the
        # scene declares, and so later primitives sit above earlier ones.
        polygons = sorted(scene.polygons, key=_primitive_order)
        polylines = sorted(scene.polylines, key=_primitive_order)
        for polygon in polygons:
            _draw_polygon(ax, plt, polygon)
        for polyline in polylines:
            _draw_polyline(ax, polyline)

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

        _set_scene_limits(ax, scene)
        ax.set_aspect("equal", adjustable="datalim")
        if not self.show_axes:
            ax.axis("off")
        return ax


def _primitive_order(primitive) -> int:
    return getattr(primitive, "draw_order", 0)


def _rgb_to_float(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(float(channel) / 255.0 for channel in rgb)  # type: ignore[return-value]


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

    for polygon in scene.polygons:
        if polygon.points_um.size:
            bounds.append(np.asarray(polygon.points_um, dtype=float))

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
