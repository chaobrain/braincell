from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

from .api import RenderRequest


@dataclass(frozen=True)
class PyVistaBackend:
    name: str = "pyvista"
    tube_sides: int = 12
    radius_scale: float = 1.0
    background: str = "white"
    show_axes: bool = True
    plotter_kwargs: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        return importlib.util.find_spec("pyvista") is not None

    def render(self, request: RenderRequest) -> object:
        geometry3d = getattr(request, "geometry3d", None)
        if geometry3d is None:
            raise ValueError("PyVistaBackend requires 3D render geometry.")
        if not self.available():
            raise RuntimeError("PyVista backend is not available. Install pyvista first.")

        import pyvista as pv

        plotter = pv.Plotter(**self.plotter_kwargs)
        plotter.set_background(self.background)
        if self.show_axes:
            plotter.show_axes()

        for batch in geometry3d.batches:
            poly = pv.PolyData()
            poly.points = batch.points_um
            poly.lines = batch.lines
            poly.point_data["radius"] = batch.radii_um * float(self.radius_scale)
            tube = poly.tube(
                scalars="radius",
                absolute=True,
                n_sides=self.tube_sides,
            )
            color = tuple(channel / 255.0 for channel in batch.color_rgb)
            plotter.add_mesh(tube, color=color)

        return plotter
