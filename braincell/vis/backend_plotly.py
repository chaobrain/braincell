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

"""Plotly backend for dependency-light interactive 3D visualization.

PyVista is the canonical interactive 3D backend but it depends on VTK,
which is a heavy install. The Plotly backend covers the common
interactive-notebook use case (rotate, pan, tooltip) without VTK by
rendering branches as ``Scatter3d`` traces.

Color-by-values is supported via the ``line.color`` / ``line.colorscale``
machinery: every value batch becomes a single ``Scatter3d`` trace with a
per-point colour lookup, preserving the shared scalar bar across
branches. Per-branch tooltips surface ``branch_name`` / ``branch_type``
so hover-to-inspect is available without additional callback plumbing.

The backend is gated on ``importlib.util.find_spec("plotly")``; when
``plotly`` is not installed ``available()`` returns ``False`` so the
default :class:`BackendChooser` falls back to PyVista or matplotlib.
"""

import importlib.util
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._values import resolved_colorbar_label
from .scene import RenderRequest, RenderScene3D, ValueSpec


@dataclass(frozen=True)
class PlotlyBackend:
    """Plotly-backed 3D renderer.

    Parameters
    ----------
    name : str
        Backend identifier used by :class:`BackendChooser`.
    supported_scene_kinds : frozenset[str]
        Advertises 3D only; Plotly could serve 2D too but the existing
        matplotlib backend is preferred there.
    line_width : float
        Default line width for branch traces, in pixels. The renderer
        scales this by the average branch radius when colour-by-values
        is off so thicker dendrites read as thicker strokes.
    show_scalar_bar : bool
        Whether to render a colour-bar alongside value-bearing traces.

    Notes
    -----
    The backend returns a ``plotly.graph_objects.Figure``. Callers that
    want the raw figure for further customization can pass
    ``backend_options={'return_figure': True}``, which is the default
    behaviour (there is no plotter-vs-figure distinction on this
    backend). In notebooks the figure renders via Plotly's standard
    mimebundle.
    """

    name: str = "plotly"
    supported_scene_kinds: frozenset[str] = frozenset({"3d"})
    line_width: float = 3.0
    show_scalar_bar: bool = True

    def available(self) -> bool:
        try:
            return importlib.util.find_spec("plotly") is not None
        except ValueError:
            return "plotly" in sys.modules

    def render(self, request: RenderRequest) -> object:
        scene = request.scene
        if not isinstance(scene, RenderScene3D):
            raise ValueError("PlotlyBackend requires RenderScene3D.")
        if not self.available():
            raise RuntimeError("Plotly backend is not available. Install plotly first.")

        import plotly.graph_objects as go

        fig = go.Figure()

        value_spec = scene.value_spec
        if scene.value_batches and value_spec is not None:
            _add_value_traces(fig, go, scene, value_spec=value_spec, show_scalar_bar=self.show_scalar_bar)
        else:
            _add_branch_type_traces(fig, go, scene, line_width=self.line_width)

        # Overlays: highlight strokes and locset markers.
        for stroke in scene.highlight_strokes:
            if stroke.points_um.shape[0] < 2:
                continue
            color = _rgb_to_hex(stroke.color_rgb)
            fig.add_trace(
                go.Scatter3d(
                    x=stroke.points_um[:, 0],
                    y=stroke.points_um[:, 1],
                    z=stroke.points_um[:, 2],
                    mode="lines",
                    line={"color": color, "width": max(self.line_width * 2.0, 5.0)},
                    opacity=float(stroke.opacity),
                    name=f"highlight:{stroke.branch_name}",
                    showlegend=False,
                    hovertemplate=f"{stroke.branch_name}<extra></extra>",
                )
            )

        if scene.markers:
            xs = [float(marker.position_um[0]) for marker in scene.markers]
            ys = [float(marker.position_um[1]) for marker in scene.markers]
            zs = [float(marker.position_um[2]) for marker in scene.markers]
            colors = [_rgb_to_hex(marker.color_rgb) for marker in scene.markers]
            texts = [f"{marker.branch_index}@x={marker.x:.3f}" for marker in scene.markers]
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker={"color": colors, "size": 6.0, "line": {"color": "black", "width": 1.0}},
                    name="locset",
                    showlegend=False,
                    hovertemplate="%{text}<extra></extra>",
                    text=texts,
                )
            )

        fig.update_layout(
            scene={
                "aspectmode": "data",
                "xaxis": {"showbackground": False, "title": "x [µm]"},
                "yaxis": {"showbackground": False, "title": "y [µm]"},
                "zaxis": {"showbackground": False, "title": "z [µm]"},
            },
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            showlegend=False,
        )
        return fig


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _add_branch_type_traces(fig: Any, go: Any, scene: RenderScene3D, *, line_width: float) -> None:
    """Render one ``Scatter3d`` line per branch type with NaN separators."""
    for batch in scene.batches:
        # Insert NaN between branches so Plotly breaks the line — this is
        # Plotly's idiom for drawing multiple disconnected polylines in a
        # single trace, which keeps trace counts bounded.
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        hover_texts: list[str] = []
        offset = 0
        # ``batch.lines`` is the VTK-style ``[n, i0, i1, … i{n-1}]``
        # repeating encoding. Walk it to recover per-branch point ranges.
        i = 0
        branch_slot = 0
        while i < len(batch.lines):
            n = int(batch.lines[i])
            indices = batch.lines[i + 1 : i + 1 + n]
            branch_name = (
                batch.branch_names[branch_slot]
                if branch_slot < len(batch.branch_names)
                else batch.branch_type
            )
            for index in indices:
                pt = batch.points_um[int(index)]
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
                zs.append(float(pt[2]))
                hover_texts.append(f"{branch_name} ({batch.branch_type})")
            # NaN break between branches
            xs.append(float("nan"))
            ys.append(float("nan"))
            zs.append(float("nan"))
            hover_texts.append("")
            i += 1 + n
            branch_slot += 1
            offset += n
        if not xs:
            continue
        color = _rgb_to_hex(batch.color_rgb)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": color, "width": float(line_width)},
                opacity=float(batch.opacity),
                name=batch.branch_type,
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts,
            )
        )


def _add_value_traces(
    fig: Any,
    go: Any,
    scene: RenderScene3D,
    *,
    value_spec: ValueSpec,
    show_scalar_bar: bool,
) -> None:
    """Render value batches as coloured ``Scatter3d`` lines with a shared scale."""
    # Shared colour-scale bounds across every batch.
    vmins: list[float] = []
    vmaxs: list[float] = []
    for batch in scene.value_batches:
        if batch.point_values.size:
            vmins.append(float(np.min(batch.point_values)))
            vmaxs.append(float(np.max(batch.point_values)))
    vmin = value_spec.vmin if value_spec.vmin is not None else (min(vmins) if vmins else 0.0)
    vmax = value_spec.vmax if value_spec.vmax is not None else (max(vmaxs) if vmaxs else 1.0)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    title = resolved_colorbar_label(value_spec, value_spec.unit_label)

    for index, batch in enumerate(scene.value_batches):
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        values: list[float] = []
        hover_texts: list[str] = []
        i = 0
        point_cursor = 0
        branch_slot = 0
        while i < len(batch.lines):
            n = int(batch.lines[i])
            indices = batch.lines[i + 1 : i + 1 + n]
            branch_name = (
                batch.branch_names[branch_slot]
                if branch_slot < len(batch.branch_names)
                else batch.branch_type
            )
            for k, idx in enumerate(indices):
                pt = batch.points_um[int(idx)]
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
                zs.append(float(pt[2]))
                value = float(batch.point_values[int(idx)])
                values.append(value)
                hover_texts.append(f"{branch_name}<br>{value:.3g}")
            xs.append(float("nan"))
            ys.append(float("nan"))
            zs.append(float("nan"))
            values.append(float("nan"))
            hover_texts.append("")
            i += 1 + n
            branch_slot += 1
            point_cursor += n
        if not xs:
            continue
        line_kwargs: dict[str, Any] = {
            "color": values,
            "colorscale": value_spec.cmap,
            "cmin": vmin,
            "cmax": vmax,
            "width": 4.0,
        }
        first = index == 0
        if show_scalar_bar and first and value_spec.show_colorbar:
            colorbar = {"title": {"text": title}} if title is not None else {}
            line_kwargs["colorbar"] = colorbar
            line_kwargs["showscale"] = True
        else:
            line_kwargs["showscale"] = False
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=line_kwargs,
                opacity=float(batch.opacity),
                name=batch.branch_type,
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts,
            )
        )
