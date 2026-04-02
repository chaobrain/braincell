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


from collections import OrderedDict

import numpy as np

from braincell._misc import u
from braincell.morpho import Morpho
from .scene import BranchPolyline3D, BranchTypeBatch3D, RenderScene3D, color_for_branch_type


def build_render_scene_3d(morpho: Morpho) -> RenderScene3D:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_render_scene_3d(...) expects Morpho, got {type(morpho).__name__!s}.")

    branches: list[BranchPolyline3D] = []
    for branch_index in range(len(morpho)):
        branch_view = morpho.branch(index=branch_index)
        branch = branch_view.branch
        if branch.points_proximal is None or branch.points_distal is None:
            raise ValueError(
                f"Branch {branch_view.name!r} lacks 3D point geometry and cannot be rendered in 3D."
            )
        points_um = np.vstack(
            [
                np.asarray(branch.points_proximal[0].to_decimal(u.um), dtype=float),
                np.asarray(branch.points_distal.to_decimal(u.um), dtype=float),
            ]
        )
        radii_um = np.concatenate(
            [
                np.atleast_1d(np.asarray(branch.radii_proximal[0].to_decimal(u.um), dtype=float)),
                np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float),
            ]
        )
        if points_um.shape[0] != radii_um.shape[0]:
            raise ValueError(
                f"Branch {branch_view.name!r} produced mismatched point/radius arrays for 3D rendering."
            )
        branches.append(
            BranchPolyline3D(
                branch_index=branch_index,
                branch_name=branch_view.name,
                branch_type=branch_view.type,
                points_um=points_um,
                radii_um=radii_um,
            )
        )

    grouped: OrderedDict[str, list[BranchPolyline3D]] = OrderedDict()
    for branch in branches:
        grouped.setdefault(branch.branch_type, []).append(branch)

    batches: list[BranchTypeBatch3D] = []
    for branch_type, type_branches in grouped.items():
        points_all: list[np.ndarray] = []
        radii_all: list[np.ndarray] = []
        lines_all: list[np.ndarray] = []
        branch_indices: list[int] = []
        branch_names: list[str] = []
        offset = 0
        for branch in type_branches:
            n_points = len(branch.points_um)
            cell = np.arange(offset, offset + n_points, dtype=np.int64)
            cell = np.insert(cell, 0, n_points)
            points_all.append(branch.points_um)
            radii_all.append(branch.radii_um)
            lines_all.append(cell)
            branch_indices.append(branch.branch_index)
            branch_names.append(branch.branch_name)
            offset += n_points
        batches.append(
            BranchTypeBatch3D(
                branch_type=branch_type,
                color_rgb=color_for_branch_type(branch_type),
                branch_indices=tuple(branch_indices),
                branch_names=tuple(branch_names),
                points_um=np.vstack(points_all),
                radii_um=np.concatenate(radii_all),
                lines=np.concatenate(lines_all).astype(np.int64),
            )
        )

    return RenderScene3D(branches=tuple(branches), batches=tuple(batches))
