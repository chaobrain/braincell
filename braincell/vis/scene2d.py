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



import numpy as np

from braincell._units import u
from braincell.morpho import Morpho

from .layout2d import build_layout_branches_2d
from .scene import Polygon2D, Polyline2D, RenderScene2D, color_for_branch_type


_PROJECTION_AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def build_render_scene_2d(
    morpho: Morpho,
    *,
    mode: str = "projected",
    projection_plane: str = "xy",
) -> RenderScene2D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_render_scene_2d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if mode == "projected":
        return build_scene2d_projected(morpho, projection_plane=projection_plane)
    if mode == "tree":
        return build_scene2d_tree(morpho)
    if mode == "frustum":
        return build_scene2d_frustum(morpho)
    raise ValueError(f"Unsupported 2D mode {mode!r}.")


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
                f"Branch {branch_view.name!r} lacks 3D point geometry and cannot be projected into 2D."
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
                draw_order=branch_index,
            )
        )

    return RenderScene2D(
        polylines=tuple(polylines),
        draw_order=tuple(polyline.draw_order for polyline in polylines),
        projection_plane=projection_plane,
        mode="projected",
    )


def build_scene2d_tree(morpho: Morpho) -> RenderScene2D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_scene2d_tree(...) expects Morpho, got {type(morpho).__name__!s}.")

    polylines: list[Polyline2D] = []
    for layout in build_layout_branches_2d(morpho, mode="tree"):
        width_um = 2.0 * float(layout.radii_proximal_um[0])
        polylines.append(
            Polyline2D(
                branch_index=layout.branch_index,
                branch_name=layout.branch_name,
                branch_type=layout.branch_type,
                points_um=layout.segment_points_um[[0, -1]],
                widths_um=np.array([width_um, width_um], dtype=float),
                color_rgb=color_for_branch_type(layout.branch_type),
                draw_order=layout.branch_index,
            )
        )

    return RenderScene2D(
        polylines=tuple(polylines),
        draw_order=tuple(polyline.draw_order for polyline in polylines),
        mode="tree",
    )


def build_scene2d_frustum(morpho: Morpho) -> RenderScene2D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_scene2d_frustum(...) expects Morpho, got {type(morpho).__name__!s}.")

    polygons: list[Polygon2D] = []
    draw_order = 0
    for layout in build_layout_branches_2d(morpho, mode="frustum"):
        for segment_index in range(len(layout.segment_points_um) - 1):
            start_um = layout.segment_points_um[segment_index]
            end_um = layout.segment_points_um[segment_index + 1]
            radius_prox_um = float(layout.radii_proximal_um[segment_index])
            radius_dist_um = float(layout.radii_distal_um[segment_index])
            polygon_points_um = np.vstack(
                [
                    start_um + layout.normal_um * radius_prox_um,
                    end_um + layout.normal_um * radius_dist_um,
                    end_um - layout.normal_um * radius_dist_um,
                    start_um - layout.normal_um * radius_prox_um,
                ]
            )
            polygons.append(
                Polygon2D(
                    branch_index=layout.branch_index,
                    branch_name=layout.branch_name,
                    branch_type=layout.branch_type,
                    points_um=polygon_points_um,
                    color_rgb=color_for_branch_type(layout.branch_type),
                    draw_order=draw_order,
                )
            )
            draw_order += 1

    return RenderScene2D(
        polygons=tuple(polygons),
        draw_order=tuple(polygon.draw_order for polygon in polygons),
        mode="frustum",
    )


def build_projected_scene_2d(morpho: Morpho, *, projection_plane: str = "xy") -> RenderScene2D:
    return build_scene2d_projected(morpho, projection_plane=projection_plane)
