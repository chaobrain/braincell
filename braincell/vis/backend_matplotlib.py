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

from .scene import RenderRequest, RenderScene2D


@dataclass(frozen=True)
class MatplotlibBackend:
    name: str = "matplotlib"
    scene_kind: str | None = "2d"
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

        fig, ax = plt.subplots()
        fig.patch.set_facecolor(self.background)
        ax.set_facecolor(self.background)

        for polyline in scene.polylines:
            color = tuple(channel / 255.0 for channel in polyline.color_rgb)
            linewidth = max(float(np.mean(polyline.widths_um)), 0.5)
            ax.plot(polyline.points_um[:, 0], polyline.points_um[:, 1], color=color, linewidth=linewidth)

        for polygon in scene.polygons:
            color = tuple(channel / 255.0 for channel in polygon.color_rgb)
            patch = plt.Polygon(
                polygon.points_um,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.3,
                linewidth=1.0,
            )
            ax.add_patch(patch)

        for circle in scene.circles:
            color = tuple(channel / 255.0 for channel in circle.color_rgb)
            patch = plt.Circle(circle.center_um, circle.radius_um, color=color, fill=False)
            ax.add_patch(patch)

        for label in scene.labels:
            color = tuple(channel / 255.0 for channel in label.color_rgb)
            ax.text(label.position_um[0], label.position_um[1], label.text, color=color)

        _set_scene_limits(ax, scene)
        ax.set_aspect("equal", adjustable="datalim")
        if not self.show_axes:
            ax.axis("off")
        return ax


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

    if not bounds:
        return

    all_points = np.vstack(bounds)
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    span_xy = max_xy - min_xy
    padding_xy = np.maximum(span_xy * 0.05, 1.0)

    ax.set_xlim(float(min_xy[0] - padding_xy[0]), float(max_xy[0] + padding_xy[0]))
    ax.set_ylim(float(min_xy[1] - padding_xy[1]), float(max_xy[1] + padding_xy[1]))
