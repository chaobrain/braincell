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


import brainunit as u
import numpy as np

from braincell.morph.morphology import Morphology
from ._values import resolve_values
from .config import (
    highlight_alpha as _highlight_alpha,
    highlight_color as _highlight_color,
    marker_color as _marker_color,
    marker_size_2d as _marker_size_2d,
)
from .layout import LayoutBranch2D, LayoutConfig, build_layout_branches_2d
from .scene import (
    BranchValues,
    HighlightStroke2D,
    Marker2D,
    OverlaySpec,
    Polygon2D,
    PolygonValuesBatch2D,
    Polyline2D,
    PolylineValues2D,
    RenderScene2D,
    ValueSpec,
    alpha_for_2d_line,
    alpha_for_2d_poly,
    color_for_2d_branch_type,
    edge_color_for_2d_branch_type,
    frustum_edge_linewidth_2d,
)

_PROJECTION_AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def build_render_scene_2d(
    morpho: Morphology,
    *,
    layout: str = "stem",
    shape: str = "frustum",
    projection_plane: str = "xy",
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    overlay: OverlaySpec | None = None,
    layout_config: LayoutConfig | None = None,
) -> RenderScene2D:
    if not isinstance(morpho, Morphology):
        raise TypeError(f"build_render_scene_2d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if layout == "projected":
        if shape != "line":
            raise ValueError("layout='projected' only supports shape='line'.")
        return build_scene2d_projected(morpho, projection_plane=projection_plane, overlay=overlay)
    if shape == "line":
        return build_scene2d_line(
            morpho,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout=layout,
            overlay=overlay,
            layout_config=layout_config,
        )
    if shape == "frustum":
        return build_scene2d_frustum(
            morpho,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout=layout,
            overlay=overlay,
            layout_config=layout_config,
        )
    raise ValueError(f"Unsupported 2D shape {shape!r}.")


def build_scene2d_projected(
    morpho: Morphology,
    *,
    projection_plane: str = "xy",
    overlay: OverlaySpec | None = None,
) -> RenderScene2D:
    try:
        first_axis, second_axis = _PROJECTION_AXES[projection_plane]
    except KeyError as exc:
        raise ValueError(f"Unsupported projection plane {projection_plane!r}.") from exc

    value_spec, per_branch_values, unit_label = _resolve_overlay_values(overlay, morpho)

    polylines: list[Polyline2D] = []
    polyline_values: list[PolylineValues2D] = []
    centerlines: dict[int, _Centerline2D] = {}
    for branch_index in range(len(morpho)):
        branch_view = morpho.branch(index=branch_index)
        branch = branch_view.branch
        if branch.points_proximal is None or branch.points_distal is None:
            raise ValueError(
                f"Branch {branch_view.name!r} lacks complete 3D point geometry and cannot be rendered with "
                "layout='projected'. Use layout='stem' with shape='line' or shape='frustum' instead."
            )
        points_um = np.vstack(
            [
                np.asarray(branch.points_proximal[0].to_decimal(u.um), dtype=float),
                np.asarray(branch.points_distal.to_decimal(u.um), dtype=float),
            ]
        )
        projected_points = points_um[:, (first_axis, second_axis)]
        widths_um = 2.0 * np.concatenate(
            [
                np.atleast_1d(np.asarray(branch.radii_proximal[0].to_decimal(u.um), dtype=float)),
                np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float),
            ]
        )
        if per_branch_values is None:
            polylines.append(
                Polyline2D(
                    branch_index=branch_index,
                    branch_name=branch_view.name,
                    branch_type=branch_view.type,
                    points_um=projected_points,
                    widths_um=widths_um,
                    color_rgb=color_for_2d_branch_type(branch_view.type),
                    alpha=alpha_for_2d_line(),
                    draw_order=branch_index,
                )
            )
        else:
            branch_values = per_branch_values[branch_index]
            polyline_values.append(
                PolylineValues2D(
                    branch_index=branch_index,
                    branch_name=branch_view.name,
                    branch_type=branch_view.type,
                    points_um=projected_points,
                    segment_values=branch_values.segment_values,
                    widths_um=widths_um,
                    draw_order=branch_index,
                )
            )
        centerlines[branch_index] = _Centerline2D.from_points(
            branch_index=branch_index,
            branch_name=branch_view.name,
            branch_type=branch_view.type,
            points_um=projected_points,
            widths_um=widths_um,
        )

    base_count = len(polylines) + len(polyline_values)
    highlight_strokes, markers = _build_overlay_primitives_2d(
        overlay,
        centerlines,
        next_draw_order=base_count,
    )

    draw_order = tuple(
        p.draw_order for p in polylines
    ) + tuple(p.draw_order for p in polyline_values)
    return RenderScene2D(
        polylines=tuple(polylines),
        polyline_values=tuple(polyline_values),
        value_spec=_with_unit_label(value_spec, unit_label),
        highlight_strokes=highlight_strokes,
        markers=markers,
        draw_order=draw_order,
        projection_plane=projection_plane,
        layout="projected",
        shape="line",
    )


def build_scene2d_line(
    morpho: Morphology,
    *,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout: str = "stem",
    overlay: OverlaySpec | None = None,
    layout_config: LayoutConfig | None = None,
) -> RenderScene2D:
    if not isinstance(morpho, Morphology):
        raise TypeError(f"build_scene2d_line(...) expects Morpho, got {type(morpho).__name__!s}.")

    value_spec, per_branch_values, unit_label = _resolve_overlay_values(overlay, morpho)

    polylines: list[Polyline2D] = []
    polyline_values: list[PolylineValues2D] = []
    centerlines: dict[int, _Centerline2D] = {}
    draw_order = 0
    for branch_layout in build_layout_branches_2d(
        morpho,
        mode="tree",
        min_branch_angle_deg=min_branch_angle_deg,
        root_layout=root_layout,
        layout_family=layout,
        layout_config=layout_config,
    ):
        centerlines[branch_layout.branch_index] = _Centerline2D.from_layout(branch_layout)
        n_points = len(branch_layout.segment_points_um)
        if per_branch_values is None:
            for segment_index in range(n_points - 1):
                polylines.append(
                    Polyline2D(
                        branch_index=branch_layout.branch_index,
                        branch_name=branch_layout.branch_name,
                        branch_type=branch_layout.branch_type,
                        points_um=branch_layout.segment_points_um[segment_index: segment_index + 2],
                        widths_um=np.array(
                            [
                                2.0 * float(branch_layout.radii_proximal_um[segment_index]),
                                2.0 * float(branch_layout.radii_distal_um[segment_index]),
                            ],
                            dtype=float,
                        ),
                        color_rgb=color_for_2d_branch_type(branch_layout.branch_type),
                        alpha=alpha_for_2d_line(),
                        draw_order=draw_order,
                    )
                )
                draw_order += 1
        else:
            branch_values = per_branch_values[branch_layout.branch_index]
            widths = np.empty(n_points, dtype=float)
            if n_points > 0:
                widths[0] = 2.0 * float(branch_layout.radii_proximal_um[0])
                widths[1:] = 2.0 * np.asarray(branch_layout.radii_distal_um, dtype=float)
            polyline_values.append(
                PolylineValues2D(
                    branch_index=branch_layout.branch_index,
                    branch_name=branch_layout.branch_name,
                    branch_type=branch_layout.branch_type,
                    points_um=np.asarray(branch_layout.segment_points_um, dtype=float),
                    segment_values=branch_values.segment_values,
                    widths_um=widths,
                    draw_order=draw_order,
                )
            )
            draw_order += 1

    highlight_strokes, markers = _build_overlay_primitives_2d(
        overlay,
        centerlines,
        next_draw_order=draw_order,
    )

    draw_order_tuple = tuple(p.draw_order for p in polylines) + tuple(
        p.draw_order for p in polyline_values
    )
    return RenderScene2D(
        polylines=tuple(polylines),
        polyline_values=tuple(polyline_values),
        value_spec=_with_unit_label(value_spec, unit_label),
        highlight_strokes=highlight_strokes,
        markers=markers,
        draw_order=draw_order_tuple,
        layout=layout,
        shape="line",
    )


def build_scene2d_frustum(
    morpho: Morphology,
    *,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout: str = "stem",
    overlay: OverlaySpec | None = None,
    layout_config: LayoutConfig | None = None,
) -> RenderScene2D:
    if not isinstance(morpho, Morphology):
        raise TypeError(f"build_scene2d_frustum(...) expects Morpho, got {type(morpho).__name__!s}.")

    value_spec, per_branch_values, unit_label = _resolve_overlay_values(overlay, morpho)

    polygons: list[Polygon2D] = []
    polygon_value_batches: list[PolygonValuesBatch2D] = []
    centerlines: dict[int, _Centerline2D] = {}
    draw_order = 0
    for branch_layout in build_layout_branches_2d(
        morpho,
        mode="frustum",
        min_branch_angle_deg=min_branch_angle_deg,
        root_layout=root_layout,
        layout_family=layout,
        layout_config=layout_config,
    ):
        centerlines[branch_layout.branch_index] = _Centerline2D.from_layout(branch_layout)
        n_segments = len(branch_layout.segment_points_um) - 1
        fill_color = color_for_2d_branch_type(branch_layout.branch_type)
        edge_color = edge_color_for_2d_branch_type(branch_layout.branch_type)
        edge_linewidth = frustum_edge_linewidth_2d()

        if per_branch_values is None:
            for segment_index in range(n_segments):
                start_um = branch_layout.segment_points_um[segment_index]
                end_um = branch_layout.segment_points_um[segment_index + 1]
                normal_um = branch_layout.segment_normals_um[segment_index]
                radius_prox_um = float(branch_layout.radii_proximal_um[segment_index])
                radius_dist_um = float(branch_layout.radii_distal_um[segment_index])
                polygon_points_um = np.vstack(
                    [
                        start_um + normal_um * radius_prox_um,
                        end_um + normal_um * radius_dist_um,
                        end_um - normal_um * radius_dist_um,
                        start_um - normal_um * radius_prox_um,
                    ]
                )
                polygons.append(
                    Polygon2D(
                        branch_index=branch_layout.branch_index,
                        branch_name=branch_layout.branch_name,
                        branch_type=branch_layout.branch_type,
                        points_um=polygon_points_um,
                        color_rgb=fill_color,
                        edge_color_rgb=edge_color,
                        edge_linewidth=edge_linewidth,
                        alpha=alpha_for_2d_poly(),
                        draw_order=draw_order,
                    )
                )
                draw_order += 1
        else:
            branch_values = per_branch_values[branch_layout.branch_index]
            if n_segments > 0:
                polygons_um = np.empty((n_segments, 4, 2), dtype=float)
                for segment_index in range(n_segments):
                    start_um = branch_layout.segment_points_um[segment_index]
                    end_um = branch_layout.segment_points_um[segment_index + 1]
                    normal_um = branch_layout.segment_normals_um[segment_index]
                    radius_prox_um = float(branch_layout.radii_proximal_um[segment_index])
                    radius_dist_um = float(branch_layout.radii_distal_um[segment_index])
                    polygons_um[segment_index] = np.vstack(
                        [
                            start_um + normal_um * radius_prox_um,
                            end_um + normal_um * radius_dist_um,
                            end_um - normal_um * radius_dist_um,
                            start_um - normal_um * radius_prox_um,
                        ]
                    )
                polygon_value_batches.append(
                    PolygonValuesBatch2D(
                        branch_index=branch_layout.branch_index,
                        branch_name=branch_layout.branch_name,
                        branch_type=branch_layout.branch_type,
                        polygons_um=polygons_um,
                        polygon_values=branch_values.segment_values,
                        edge_color_rgb=edge_color,
                        edge_linewidth=edge_linewidth,
                        draw_order=draw_order,
                    )
                )
                draw_order += 1

    highlight_strokes, markers = _build_overlay_primitives_2d(
        overlay,
        centerlines,
        next_draw_order=draw_order,
    )

    draw_order_tuple = tuple(p.draw_order for p in polygons) + tuple(
        b.draw_order for b in polygon_value_batches
    )
    return RenderScene2D(
        polygons=tuple(polygons),
        polygon_value_batches=tuple(polygon_value_batches),
        value_spec=_with_unit_label(value_spec, unit_label),
        highlight_strokes=highlight_strokes,
        markers=markers,
        draw_order=draw_order_tuple,
        layout=layout,
        shape="frustum",
    )


def build_projected_scene_2d(morpho: Morphology, *, projection_plane: str = "xy") -> RenderScene2D:
    return build_scene2d_projected(morpho, projection_plane=projection_plane)


# ---------------------------------------------------------------------------
# Overlay resolution helpers
# ---------------------------------------------------------------------------

class _Centerline2D:
    """Per-branch 2D centerline used to resolve overlay (branch, x) coordinates.

    Stores a polyline (``points_um``), per-point widths, cumulative arc
    length, and metadata (branch name/type) so overlays can interpolate
    without re-reading the morphology.
    """

    __slots__ = ("branch_index", "branch_name", "branch_type", "points_um", "widths_um", "cumulative_um", "total_um")

    def __init__(
        self,
        *,
        branch_index: int,
        branch_name: str,
        branch_type: str,
        points_um: np.ndarray,
        widths_um: np.ndarray,
        cumulative_um: np.ndarray,
    ) -> None:
        self.branch_index = branch_index
        self.branch_name = branch_name
        self.branch_type = branch_type
        self.points_um = np.asarray(points_um, dtype=float)
        self.widths_um = np.asarray(widths_um, dtype=float)
        self.cumulative_um = np.asarray(cumulative_um, dtype=float)
        self.total_um = float(self.cumulative_um[-1]) if self.cumulative_um.size else 0.0

    @classmethod
    def from_points(
        cls,
        *,
        branch_index: int,
        branch_name: str,
        branch_type: str,
        points_um: np.ndarray,
        widths_um: np.ndarray,
    ) -> "_Centerline2D":
        pts = np.asarray(points_um, dtype=float)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        return cls(
            branch_index=branch_index,
            branch_name=branch_name,
            branch_type=branch_type,
            points_um=pts,
            widths_um=np.asarray(widths_um, dtype=float),
            cumulative_um=cum,
        )

    @classmethod
    def from_layout(cls, layout: LayoutBranch2D) -> "_Centerline2D":
        widths = np.concatenate(
            [
                np.atleast_1d(2.0 * np.asarray(layout.radii_proximal_um[:1], dtype=float)),
                2.0 * np.asarray(layout.radii_distal_um, dtype=float),
            ]
        )
        return cls(
            branch_index=layout.branch_index,
            branch_name=layout.branch_name,
            branch_type=layout.branch_type,
            points_um=np.asarray(layout.segment_points_um, dtype=float),
            widths_um=widths,
            cumulative_um=np.asarray(layout.cumulative_lengths_um, dtype=float),
        )

    def point_at(self, x: float) -> np.ndarray:
        """Interpolate the 2D position at fractional coordinate *x* ∈ [0, 1]."""
        if self.total_um <= 0.0 or self.points_um.shape[0] == 0:
            return self.points_um[0].copy() if self.points_um.size else np.zeros(2, dtype=float)
        clamped = float(np.clip(x, 0.0, 1.0))
        target = clamped * self.total_um
        idx = int(np.searchsorted(self.cumulative_um[1:], target, side="right"))
        idx = min(max(idx, 0), len(self.points_um) - 2)
        seg_start = float(self.cumulative_um[idx])
        seg_end = float(self.cumulative_um[idx + 1])
        seg_len = seg_end - seg_start
        if seg_len <= 0.0:
            return self.points_um[idx].copy()
        alpha = (target - seg_start) / seg_len
        return (1.0 - alpha) * self.points_um[idx] + alpha * self.points_um[idx + 1]

    def subpolyline(self, prox: float, dist: float) -> np.ndarray:
        """Return the polyline fragment between fractional coordinates *prox* and *dist*.

        The result includes interpolated endpoints plus any intermediate
        polyline vertices that fall strictly between them.
        """
        lo, hi = (float(prox), float(dist)) if prox <= dist else (float(dist), float(prox))
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if self.total_um <= 0.0 or self.points_um.shape[0] == 0:
            return self.points_um[:1].copy() if self.points_um.size else np.zeros((0, 2), dtype=float)
        start = self.point_at(lo)
        end = self.point_at(hi)
        start_arc = lo * self.total_um
        end_arc = hi * self.total_um
        interior_mask = (self.cumulative_um > start_arc + 1e-12) & (self.cumulative_um < end_arc - 1e-12)
        interior = self.points_um[interior_mask]
        return np.vstack([start[None, :], interior, end[None, :]])

    def average_width(self) -> float:
        return float(np.mean(self.widths_um)) if self.widths_um.size else 1.0


def _build_overlay_primitives_2d(
    overlay: OverlaySpec | None,
    centerlines: dict[int, _Centerline2D],
    *,
    next_draw_order: int,
) -> tuple[tuple[HighlightStroke2D, ...], tuple[Marker2D, ...]]:
    if overlay is None:
        return (), ()

    strokes: list[HighlightStroke2D] = []
    markers: list[Marker2D] = []
    order = next_draw_order

    region = overlay.region
    if region is not None:
        color = _highlight_color()
        alpha = _highlight_alpha()
        for branch_index, prox, dist in region.intervals:
            centerline = centerlines.get(int(branch_index))
            if centerline is None:
                continue
            subpoints = centerline.subpolyline(float(prox), float(dist))
            if subpoints.shape[0] < 2:
                continue
            # Overlay strokes are drawn thicker than the underlying line,
            # scaled by the branch's average centerline width.
            linewidth = max(centerline.average_width() * 1.25, 1.5)
            strokes.append(
                HighlightStroke2D(
                    branch_index=centerline.branch_index,
                    branch_name=centerline.branch_name,
                    branch_type=centerline.branch_type,
                    points_um=subpoints,
                    color_rgb=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    draw_order=order,
                )
            )
            order += 1

    locset = overlay.locset
    if locset is not None:
        color = _marker_color()
        size = _marker_size_2d()
        for point in locset.points:
            branch_index, x = int(point[0]), float(point[1])
            centerline = centerlines.get(branch_index)
            if centerline is None:
                continue
            position = centerline.point_at(x)
            markers.append(
                Marker2D(
                    branch_index=branch_index,
                    x=x,
                    position_um=position,
                    color_rgb=color,
                    size=size,
                    draw_order=order,
                )
            )
            order += 1

    return tuple(strokes), tuple(markers)


def _resolve_overlay_values(
    overlay: OverlaySpec | None,
    morpho: Morphology,
) -> tuple[ValueSpec | None, dict[int, BranchValues] | None, str | None]:
    """Return ``(spec, per_branch_values, unit_label)`` or ``(None, None, None)``."""
    if overlay is None:
        return None, None, None
    spec = overlay.values_spec()
    if spec is None:
        return None, None, None
    per_branch, unit_label = resolve_values(morpho, spec)
    return spec, per_branch, unit_label


def _with_unit_label(
    spec: ValueSpec | None,
    unit_label: str | None,
) -> ValueSpec | None:
    """Inject the unit label from the input array into the spec when empty."""
    if spec is None:
        return None
    if unit_label is None or spec.unit_label is not None:
        return spec
    return ValueSpec(
        values=spec.values,
        cmap=spec.cmap,
        vmin=spec.vmin,
        vmax=spec.vmax,
        norm=spec.norm,
        label=spec.label,
        unit_label=unit_label,
        show_colorbar=spec.show_colorbar,
    )
