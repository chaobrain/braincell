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
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._values import resolved_colorbar_label
from .hooks import PickInfo, VisHooks
from .scene import RenderRequest, RenderScene3D, ValueBatch3D, ValueSpec

# Attribute name used by the PyVista backend to attach a point→branch
# lookup table to a plotter. Downstream picking callbacks read it via
# ``getattr(plotter, _BC_POINT_BRANCH_MAP, None)``.
_BC_POINT_BRANCH_MAP = "_bc_point_branch_map"


@dataclass(frozen=True)
class PyVistaBackend:
    name: str = "pyvista"
    supported_scene_kinds: frozenset[str] = frozenset({"3d"})
    tube_sides: int = 12
    radius_scale: float = 1.0
    background: str = "white"
    show_axes: bool = True
    plotter_kwargs: dict[str, Any] = field(default_factory=dict)
    skeleton_line_width: float = 2.0

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

        backend_options = request.backend_options or {}
        notebook = backend_options.get("notebook")
        jupyter_backend = backend_options.get("jupyter_backend")
        return_plotter = bool(backend_options.get("return_plotter", False))

        plotter = pv.Plotter(**self.plotter_kwargs)
        plotter.set_background(self.background)
        if self.show_axes:
            plotter.show_axes()

        # Build a point→branch map up front so picking callbacks can
        # resolve clicked points back to (branch_index, branch_name,
        # branch_type). The map aggregates every branch in the scene in
        # the order they will be added to the plotter, matching the
        # concatenation order used when PyVista assembles PolyData.
        point_branch_map = _build_point_branch_map(scene)
        setattr(plotter, _BC_POINT_BRANCH_MAP, point_branch_map)

        mode = scene.mode if scene.mode else "geometry"

        # If a value spec is supplied, render one coloured mesh per value
        # batch instead of the per-type coloured meshes. The base
        # ``scene.batches`` are used for geometry grouping only and are
        # skipped so the two paths don't double-render.
        value_spec = scene.value_spec
        if scene.value_batches and value_spec is not None:
            _render_value_batches_pyvista(
                pv,
                plotter,
                scene.value_batches,
                mode=mode,
                value_spec=value_spec,
                tube_sides=self.tube_sides,
                radius_scale=self.radius_scale,
                skeleton_line_width=self.skeleton_line_width,
            )
        else:
            for batch in scene.batches:
                poly = pv.PolyData()
                poly.points = batch.points_um
                poly.lines = batch.lines
                color = _rgb_to_float(batch.color_rgb)
                if mode == "skeleton":
                    plotter.add_mesh(
                        poly,
                        color=color,
                        opacity=batch.opacity,
                        line_width=self.skeleton_line_width,
                    )
                else:
                    poly.point_data["radius"] = batch.radii_um * float(self.radius_scale)
                    tube = poly.tube(
                        scalars="radius",
                        absolute=True,
                        n_sides=self.tube_sides,
                    )
                    plotter.add_mesh(tube, color=color, opacity=batch.opacity)

        # Overlay: region highlight strokes
        for stroke in scene.highlight_strokes:
            if stroke.points_um.shape[0] < 2:
                continue
            poly = pv.PolyData()
            poly.points = stroke.points_um
            n_points = stroke.points_um.shape[0]
            cell = np.concatenate(
                [np.array([n_points], dtype=np.int64), np.arange(n_points, dtype=np.int64)]
            )
            poly.lines = cell
            color = _rgb_to_float(stroke.color_rgb)
            if mode == "skeleton":
                plotter.add_mesh(
                    poly,
                    color=color,
                    opacity=stroke.opacity,
                    line_width=max(self.skeleton_line_width * 2.0, 3.0),
                )
            else:
                poly.point_data["radius"] = stroke.radii_um * float(self.radius_scale) * 1.15
                tube = poly.tube(
                    scalars="radius",
                    absolute=True,
                    n_sides=self.tube_sides,
                )
                plotter.add_mesh(tube, color=color, opacity=stroke.opacity)

        # Overlay: locset markers as small spheres
        for marker in scene.markers:
            sphere = pv.Sphere(
                radius=float(marker.radius_um),
                center=tuple(float(c) for c in marker.position_um),
            )
            plotter.add_mesh(sphere, color=_rgb_to_float(marker.color_rgb))

        # Optional pick hooks — wired last so every mesh is already on
        # the plotter before picking is enabled.
        hooks = backend_options.get("hooks") if backend_options else None
        if isinstance(hooks, VisHooks) and hooks.is_active():
            connect_pyvista_hooks(plotter, hooks)

        if notebook is None:
            notebook = _running_in_notebook()
        if not notebook:
            return plotter

        diagnostics = _configure_trame_for_notebook(pv)
        attempted = []
        failures: list[tuple[str, str]] = []
        for candidate in _notebook_backends(jupyter_backend):
            _prepare_plotter_for_notebook(plotter)
            attempted.append(candidate)
            try:
                viewer = plotter.show(
                    jupyter_backend=candidate,
                    return_viewer=True,
                    auto_close=False,
                )
            except Exception as exc:
                failures.append((candidate, f"{type(exc).__name__}: {exc}"))
                continue
            if viewer is None:
                failures.append((candidate, "RuntimeError: PyVista returned no notebook viewer."))
                continue
            if return_plotter:
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


def _rgb_to_float(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(float(channel) / 255.0 for channel in rgb)  # type: ignore[return-value]


def _render_value_batches_pyvista(
    pv: Any,
    plotter: Any,
    value_batches: tuple[ValueBatch3D, ...],
    *,
    mode: str,
    value_spec: ValueSpec,
    tube_sides: int,
    radius_scale: float,
    skeleton_line_width: float,
) -> None:
    """Render scalar-valued PolyData batches with a single scalar bar."""
    import numpy as np  # local import keeps module-level surface small

    # Compute a shared vmin/vmax over every batch so the colour scale
    # is consistent across branches/types.
    all_values: list[float] = []
    for batch in value_batches:
        if batch.point_values.size:
            all_values.append(float(np.min(batch.point_values)))
            all_values.append(float(np.max(batch.point_values)))
    vmin = value_spec.vmin if value_spec.vmin is not None else (min(all_values) if all_values else 0.0)
    vmax = value_spec.vmax if value_spec.vmax is not None else (max(all_values) if all_values else 1.0)
    if vmin == vmax:
        vmin = vmin - 0.5
        vmax = vmax + 0.5
    clim = (vmin, vmax)

    title = resolved_colorbar_label(value_spec, value_spec.unit_label)
    scalar_bar_args = {"title": title if title is not None else "values"}

    for index, batch in enumerate(value_batches):
        poly = pv.PolyData()
        poly.points = batch.points_um
        poly.lines = batch.lines
        poly.point_data["values"] = batch.point_values
        first = index == 0
        show_scalar_bar = bool(value_spec.show_colorbar) and first
        if mode == "skeleton":
            plotter.add_mesh(
                poly,
                scalars="values",
                cmap=value_spec.cmap,
                clim=clim,
                opacity=batch.opacity,
                line_width=skeleton_line_width,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args=scalar_bar_args if show_scalar_bar else None,
            )
        else:
            poly.point_data["radius"] = batch.radii_um * float(radius_scale)
            tube = poly.tube(
                scalars="radius",
                absolute=True,
                n_sides=tube_sides,
            )
            plotter.add_mesh(
                tube,
                scalars="values",
                cmap=value_spec.cmap,
                clim=clim,
                opacity=batch.opacity,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args=scalar_bar_args if show_scalar_bar else None,
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


# ---------------------------------------------------------------------------
# Interactive picking
# ---------------------------------------------------------------------------


def _build_point_branch_map(scene: RenderScene3D) -> list[dict[str, Any]]:
    """Build a flat list of per-point pick metadata for a 3D scene.

    Entry ``i`` describes the global ``i``-th point produced when the
    backend renders every branch in order. The PyVista backend uses the
    same ordering for its ``BranchTypeBatch3D`` meshes, so an index
    returned by ``enable_point_picking`` can be resolved directly via
    this table.
    """
    entries: list[dict[str, Any]] = []
    n_branches = len(scene.branches)
    for branch in scene.branches:
        n_points = int(branch.points_um.shape[0])
        for local_index in range(n_points):
            entries.append(
                {
                    "branch_index": int(branch.branch_index),
                    "branch_name": str(branch.branch_name),
                    "branch_type": str(branch.branch_type),
                    "segment_index": max(local_index - 1, 0) if n_points > 1 else 0,
                    "x": float(local_index) / max(n_points - 1, 1),
                    "position_um": np.asarray(branch.points_um[local_index], dtype=float),
                }
            )
    if not entries and n_branches == 0:
        return entries
    return entries


def _pick_info_from_point(
    entries: list[dict[str, Any]],
    point_index: int,
    *,
    artist: Any = None,
) -> PickInfo | None:
    if point_index < 0 or point_index >= len(entries):
        return None
    entry = entries[point_index]
    return PickInfo(
        branch_index=entry["branch_index"],
        branch_name=entry["branch_name"],
        branch_type=entry["branch_type"],
        segment_index=entry.get("segment_index"),
        x=entry.get("x"),
        value=None,
        position_um=entry.get("position_um"),
        artist=artist,
    )


def connect_pyvista_hooks(plotter: Any, hooks: VisHooks) -> None:
    """Wire :class:`VisHooks` callbacks onto an existing PyVista plotter.

    Uses :meth:`pyvista.Plotter.enable_point_picking` when available so
    clicking a point on a branch fires the ``on_pick`` callback with a
    :class:`PickInfo` resolved through the plotter's stored
    ``_bc_point_branch_map``. PyVista does not expose a cheap hover
    event, so ``on_hover`` / ``on_leave`` are silently ignored by this
    backend — callers that need hover semantics should use the
    matplotlib backend.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter produced by :meth:`PyVistaBackend.render`. Must still
        carry the ``_bc_point_branch_map`` attribute so the callback
        can resolve point indices.
    hooks : VisHooks
        Callback bundle. Only ``on_pick`` is honored.

    Notes
    -----
    The callback signature expected by PyVista is a single-argument
    callable that receives the picked ``numpy.ndarray`` point; we wrap
    the user callback so it sees a :class:`PickInfo` instead.
    """
    if not hooks.is_active() or hooks.on_pick is None:
        return
    entries = getattr(plotter, _BC_POINT_BRANCH_MAP, None)
    if not entries:
        return
    if not hasattr(plotter, "enable_point_picking"):
        return
    # Precompute an array of 3D positions so nearest-point lookup is
    # cheap when PyVista passes us a world-space point.
    positions = np.asarray([entry["position_um"] for entry in entries], dtype=float)

    def _on_pv_pick(picked_point, *_args, **_kwargs):
        if picked_point is None:
            return
        try:
            pt = np.asarray(picked_point, dtype=float).reshape(-1)
        except Exception:
            return
        if pt.size < 3 or positions.size == 0:
            return
        diffs = positions[:, : pt.size] - pt[np.newaxis, : pt.size]
        nearest = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        info = _pick_info_from_point(entries, nearest, artist=plotter)
        if info is not None:
            hooks.on_pick(info)

    plotter.enable_point_picking(callback=_on_pv_pick, show_message=False, use_picker=True)
