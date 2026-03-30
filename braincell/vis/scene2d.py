from __future__ import annotations

import numpy as np

from braincell._units import u
from braincell.morpho import Morpho

from .scene import Polyline2D, RenderScene2D, color_for_branch_type


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
    if mode == "layout":
        return build_scene2d_layout(morpho)
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


def build_scene2d_layout(morpho: Morpho) -> RenderScene2D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_scene2d_layout(...) expects Morpho, got {type(morpho).__name__!s}.")
    raise NotImplementedError("2D layout mode is not implemented yet.")


def build_projected_scene_2d(morpho: Morpho, *, projection_plane: str = "xy") -> RenderScene2D:
    return build_scene2d_projected(morpho, projection_plane=projection_plane)
