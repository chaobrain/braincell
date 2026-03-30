from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .scene import RenderRequest, RenderScene3D


@dataclass(frozen=True)
class PyVistaBackend:
    name: str = "pyvista"
    scene_kind: str | None = "3d"
    tube_sides: int = 12
    radius_scale: float = 1.0
    background: str = "white"
    show_axes: bool = True
    plotter_kwargs: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        try:
            return importlib.util.find_spec("pyvista") is not None
        except ValueError:
            return "pyvista" in sys.modules

    def render(self, request: RenderRequest) -> object:
        scene = request.scene
        if not isinstance(scene, RenderScene3D):
            raise ValueError("PyVistaBackend requires RenderScene3D.")
        if not self.available():
            raise RuntimeError("PyVista backend is not available. Install pyvista first.")

        import pyvista as pv

        plotter = pv.Plotter(**self.plotter_kwargs)
        plotter.set_background(self.background)
        if self.show_axes:
            plotter.show_axes()

        for batch in scene.batches:
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

        notebook = request.notebook
        if notebook is None:
            notebook = _running_in_notebook()
        if not notebook:
            return plotter

        diagnostics = _configure_trame_for_notebook(pv)
        attempted = []
        failures: list[tuple[str, str]] = []
        for jupyter_backend in _notebook_backends(request.jupyter_backend):
            _prepare_plotter_for_notebook(plotter)
            attempted.append(jupyter_backend)
            try:
                viewer = plotter.show(
                    jupyter_backend=jupyter_backend,
                    return_viewer=True,
                    auto_close=False,
                )
            except Exception as exc:
                failures.append((jupyter_backend, f"{type(exc).__name__}: {exc}"))
                continue
            if viewer is None:
                failures.append((jupyter_backend, "RuntimeError: PyVista returned no notebook viewer."))
                continue
            if request.return_plotter:
                return plotter
            return viewer

        if hasattr(plotter, "close"):
            plotter.close()
        failure_text = "; ".join(f"{backend}: {message}" for backend, message in failures)
        raise RuntimeError(
            "Interactive notebook rendering failed for PyVista. "
            f"Attempted backends: {', '.join(attempted)}. "
            f"Failures: {failure_text} "
            f"Environment: {diagnostics}. "
            "On a headless remote Jupyter server, prefer jupyter_backend='client' or 'html'. "
            "Server/trame modes may also require a virtual framebuffer. "
            "You can still pass return_plotter=True to receive the raw Plotter."
        )


def _notebook_backends(requested: str | None) -> tuple[str, ...]:
    if requested is None:
        if _headless_display():
            return ("client", "html")
        return ("trame", "html")
    if requested == "trame" and _headless_display():
        return ("client",)
    return (requested,)


def _configure_trame_for_notebook(pv: Any) -> str:
    trame = pv.global_theme.trame
    diagnostics = []

    extension_available = getattr(trame, "jupyter_extension_available", False)
    if extension_available:
        if hasattr(trame, "jupyter_extension_enabled"):
            trame.jupyter_extension_enabled = True
        diagnostics.append("trame_jupyter_extension=enabled")
    else:
        if hasattr(trame, "jupyter_extension_enabled"):
            trame.jupyter_extension_enabled = False
        diagnostics.append("trame_jupyter_extension=missing")

    proxy_available = importlib.util.find_spec("jupyter_server_proxy") is not None
    if proxy_available and hasattr(trame, "server_proxy_enabled"):
        trame.server_proxy_enabled = True
        diagnostics.append("jupyter_server_proxy=enabled")
    elif hasattr(trame, "server_proxy_enabled"):
        trame.server_proxy_enabled = False
        diagnostics.append("jupyter_server_proxy=missing")

    if hasattr(trame, "server_proxy_prefix"):
        trame.server_proxy_prefix = "/proxy/"
    diagnostics.append(f"headless={_headless_display()}")
    diagnostics.append(f"display={os.environ.get('DISPLAY')!r}")
    return ", ".join(diagnostics)


def _headless_display() -> bool:
    return not bool(os.environ.get("DISPLAY"))


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _prepare_plotter_for_notebook(plotter: Any) -> None:
    if getattr(plotter, "_rendered", False):
        return
    if hasattr(plotter, "_on_first_render_request"):
        plotter._on_first_render_request()
    if hasattr(plotter, "render"):
        plotter.render()
