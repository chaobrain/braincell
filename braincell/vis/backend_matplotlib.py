from __future__ import annotations

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

        for circle in scene.circles:
            color = tuple(channel / 255.0 for channel in circle.color_rgb)
            patch = plt.Circle(circle.center_um, circle.radius_um, color=color, fill=False)
            ax.add_patch(patch)

        for label in scene.labels:
            color = tuple(channel / 255.0 for channel in label.color_rgb)
            ax.text(label.position_um[0], label.position_um[1], label.text, color=color)

        ax.set_aspect("equal", adjustable="datalim")
        if not self.show_axes:
            ax.axis("off")
        return ax
