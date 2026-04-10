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

from braincell.morpho import Morpho
from .layout2d import build_layout_branches_2d
from .scene import alpha_for_2d_line, alpha_for_2d_poly, Polygon2D, Polyline2D, RenderScene2D, color_for_branch_type

_PROJECTION_AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def build_render_scene_2d(
    morpho: Morpho,
    *,
    layout: str = "stem",
    shape: str = "frustum",
    projection_plane: str = "xy",
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
) -> RenderScene2D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_render_scene_2d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if layout == "projected":
        if shape != "line":
            raise ValueError("layout='projected' only supports shape='line'.")
        return build_scene2d_projected(morpho, projection_plane=projection_plane)
    if shape == "line":
        return build_scene2d_line(
            morpho,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout=layout,
        )
    if shape == "frustum":
        return build_scene2d_frustum(
            morpho,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout=layout,
        )
    raise ValueError(f"Unsupported 2D shape {shape!r}.")


def build_scene2d_projected(morpho: Morpho, *, projection_plane: str = "xy") -> RenderScene2D:
    try:
        first_axis, second_axis = _PROJECTION_AXES[projection_plane]
    except KeyError as exc:
        raise ValueError(f"Unsupported projection plane {projection_plane!r}.") from exc

    polylines: list[Polyline2D] = []
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
        polylines.append(
            Polyline2D(
                branch_index=branch_index,
                branch_name=branch_view.name,
                branch_type=branch_view.type,
                points_um=projected_points,
                widths_um=widths_um,
                color_rgb=color_for_branch_type(branch_view.type),
                alpha=alpha_for_2d_line(),
                draw_order=branch_index,
            )
        )

    return RenderScene2D(
        polylines=tuple(polylines),
        draw_order=tuple(polyline.draw_order for polyline in polylines),
        projection_plane=projection_plane,
        layout="projected",
        shape="line",
    )


def build_scene2d_line(
    morpho: Morpho,
    *,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout: str = "stem",
) -> RenderScene2D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_scene2d_line(...) expects Morpho, got {type(morpho).__name__!s}.")

    polylines: list[Polyline2D] = []
    draw_order = 0
    for branch_layout in build_layout_branches_2d(
        morpho,
        mode="tree",
        min_branch_angle_deg=min_branch_angle_deg,
        root_layout=root_layout,
        layout_family=layout,
    ):
        for segment_index in range(len(branch_layout.segment_points_um) - 1):
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
                    color_rgb=color_for_branch_type(branch_layout.branch_type),
                    alpha=alpha_for_2d_line(),
                    draw_order=draw_order,
                )
            )
            draw_order += 1

    return RenderScene2D(
        polylines=tuple(polylines),
        draw_order=tuple(polyline.draw_order for polyline in polylines),
        layout=layout,
        shape="line",
    )


def build_scene2d_frustum(
    morpho: Morpho,
    *,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout: str = "stem",
) -> RenderScene2D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_scene2d_frustum(...) expects Morpho, got {type(morpho).__name__!s}.")

    polygons: list[Polygon2D] = []
    draw_order = 0
    for branch_layout in build_layout_branches_2d(
        morpho,
        mode="frustum",
        min_branch_angle_deg=min_branch_angle_deg,
        root_layout=root_layout,
        layout_family=layout,
    ):
        for segment_index in range(len(branch_layout.segment_points_um) - 1):
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
                    color_rgb=color_for_branch_type(branch_layout.branch_type),
                    alpha=alpha_for_2d_poly(),
                    draw_order=draw_order,
                )
            )
            draw_order += 1

    return RenderScene2D(
        polygons=tuple(polygons),
        draw_order=tuple(polygon.draw_order for polygon in polygons),
        layout=layout,
        shape="frustum",
    )


def build_projected_scene_2d(morpho: Morpho, *, projection_plane: str = "xy") -> RenderScene2D:
    return build_scene2d_projected(morpho, projection_plane=projection_plane)
