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
from .hooks import PickInfo, VisHooks
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

# Attribute name used to attach pick metadata to every drawn artist. Each
# value is either a single dict (for single-segment artists like Polyline2D
# or Polygon2D) or a list of dicts (for LineCollection / PolyCollection
# artists that batch many segments).
_BC_PICK_META = "_bc_pick_meta"


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

        # Wire optional pick / hover callbacks last so every artist is on
        # the axes before the callbacks consult them.
        hooks = request.backend_options.get("hooks") if request.backend_options else None
        if isinstance(hooks, VisHooks) and hooks.is_active():
            connect_hooks(ax, hooks)
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
    (line,) = ax.plot(
        polyline.points_um[:, 0],
        polyline.points_um[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=polyline.alpha,
        zorder=polyline.draw_order,
    )
    setattr(
        line,
        _BC_PICK_META,
        {
            "branch_index": polyline.branch_index,
            "branch_name": polyline.branch_name,
            "branch_type": polyline.branch_type,
            "segment_index": None,
            "points_um": np.asarray(polyline.points_um, dtype=float),
        },
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
    setattr(
        patch,
        _BC_PICK_META,
        {
            "branch_index": polygon.branch_index,
            "branch_name": polygon.branch_name,
            "branch_type": polygon.branch_type,
            "segment_index": None,
        },
    )


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
    # One pick-meta dict per segment so picks report the correct sub-range.
    n_segs = segments.shape[0]
    segment_meta = [
        {
            "branch_index": polyline.branch_index,
            "branch_name": polyline.branch_name,
            "branch_type": polyline.branch_type,
            "segment_index": i,
            "value": float(values[i]) if i < values.size else None,
            "segment_start": segments[i, 0],
            "segment_end": segments[i, 1],
            "x_start": i / n_segs,
            "x_end": (i + 1) / n_segs,
        }
        for i in range(n_segs)
    ]
    setattr(lc, _BC_PICK_META, segment_meta)


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
    n_segs = polygons_um.shape[0]
    segment_meta = [
        {
            "branch_index": batch.branch_index,
            "branch_name": batch.branch_name,
            "branch_type": batch.branch_type,
            "segment_index": i,
            "value": float(values[i]) if i < values.size else None,
            "x_start": i / n_segs,
            "x_end": (i + 1) / n_segs,
        }
        for i in range(n_segs)
    ]
    setattr(pc, _BC_PICK_META, segment_meta)


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


# ---------------------------------------------------------------------------
# Interactive picking / hovering
# ---------------------------------------------------------------------------

def _resolve_pick_meta(artist, event) -> dict | None:
    """Return the pick-metadata dict for a picked artist + pick event.

    Handles both single-segment artists (Line2D / Polygon) and batched
    collections (LineCollection / PolyCollection). For collections the
    index is recovered from ``event.ind``.
    """
    meta = getattr(artist, _BC_PICK_META, None)
    if meta is None:
        return None
    if isinstance(meta, list):
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return None
        index = int(ind[0])
        if index < 0 or index >= len(meta):
            return None
        return meta[index]
    return meta


def _pick_info_from_meta(meta: dict, *, artist, xdata: float | None, ydata: float | None) -> PickInfo:
    position = None
    if xdata is not None and ydata is not None:
        position = np.array([xdata, ydata], dtype=float)
    x_coord = None
    if "x_start" in meta and "x_end" in meta:
        x_coord = 0.5 * (float(meta["x_start"]) + float(meta["x_end"]))
    value = meta.get("value")
    return PickInfo(
        branch_index=int(meta["branch_index"]),
        branch_name=str(meta["branch_name"]),
        branch_type=str(meta["branch_type"]),
        segment_index=meta.get("segment_index"),
        x=x_coord,
        value=float(value) if value is not None else None,
        position_um=position,
        artist=artist,
    )


def _find_hover_meta(ax, event) -> tuple[object, dict] | None:
    """Locate the topmost artist under ``event`` carrying pick metadata."""
    if event.inaxes is not ax:
        return None

    # Sort candidates with the topmost zorder first so hover prefers the
    # artist the user can actually see.
    candidates: list = []
    for coll in ax.collections:
        if hasattr(coll, _BC_PICK_META):
            candidates.append(coll)
    for line in ax.lines:
        if hasattr(line, _BC_PICK_META):
            candidates.append(line)
    for patch in ax.patches:
        if hasattr(patch, _BC_PICK_META):
            candidates.append(patch)

    candidates.sort(key=lambda artist: getattr(artist, "get_zorder", lambda: 0)(), reverse=True)

    for artist in candidates:
        contains, details = artist.contains(event)
        if not contains:
            continue
        meta = getattr(artist, _BC_PICK_META, None)
        if meta is None:
            continue
        if isinstance(meta, list):
            ind = details.get("ind") if isinstance(details, dict) else None
            if ind is None or len(ind) == 0:
                continue
            index = int(ind[0])
            if 0 <= index < len(meta):
                return artist, meta[index]
            continue
        return artist, meta
    return None


def connect_hooks(ax, hooks: VisHooks) -> dict[str, int]:
    """Wire :class:`VisHooks` callbacks onto an existing matplotlib axes.

    Called automatically by :class:`MatplotlibBackend.render` when a
    :class:`VisHooks` is passed through ``backend_options``, but also
    usable directly by callers that build scenes manually. The return
    value is a dict of matplotlib connection ids keyed by
    ``"pick"`` / ``"motion"``, suitable for
    :meth:`Figure.canvas.mpl_disconnect` later.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes. All value-bearing artists on ``ax`` must carry the
        pick metadata produced by :class:`MatplotlibBackend.render`.
    hooks : VisHooks
        The bundle of ``on_pick`` / ``on_hover`` / ``on_leave``
        callbacks to wire up.

    Returns
    -------
    dict
        A ``{event_name: connection_id}`` mapping of the event handlers
        that were actually registered.

    Notes
    -----
    Collections and lines on ``ax`` are made pickable (``set_picker``)
    so that clicks dispatch through the matplotlib ``pick_event`` path.
    The hover path uses ``motion_notify_event`` + ``artist.contains``
    because ``pick_event`` only fires on button clicks.
    """
    if not hooks.is_active():
        return {}

    fig = ax.figure
    connection_ids: dict[str, int] = {}

    if hooks.on_pick is not None:
        for artist in list(ax.collections) + list(ax.lines) + list(ax.patches):
            if not hasattr(artist, _BC_PICK_META):
                continue
            artist.set_picker(True)

        def _on_pick(event):
            artist = event.artist
            meta = _resolve_pick_meta(artist, event)
            if meta is None:
                return
            mouse = getattr(event, "mouseevent", None)
            xdata = getattr(mouse, "xdata", None) if mouse is not None else None
            ydata = getattr(mouse, "ydata", None) if mouse is not None else None
            info = _pick_info_from_meta(meta, artist=artist, xdata=xdata, ydata=ydata)
            hooks.on_pick(info)

        connection_ids["pick"] = fig.canvas.mpl_connect("pick_event", _on_pick)

    if hooks.on_hover is not None or hooks.on_leave is not None:
        state = {"last_key": None}

        def _on_motion(event):
            match = _find_hover_meta(ax, event)
            if match is None:
                if state["last_key"] is not None and hooks.on_leave is not None:
                    hooks.on_leave()
                state["last_key"] = None
                return
            artist, meta = match
            key = (id(artist), meta.get("segment_index"))
            if key == state["last_key"]:
                return
            state["last_key"] = key
            if hooks.on_hover is not None:
                info = _pick_info_from_meta(meta, artist=artist, xdata=event.xdata, ydata=event.ydata)
                hooks.on_hover(info)

        connection_ids["motion"] = fig.canvas.mpl_connect("motion_notify_event", _on_motion)

    return connection_ids
