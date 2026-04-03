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

from .backend import BackendChooser, RenderBackend
from .backend_matplotlib import MatplotlibBackend
from .backend_pyvista import PyVistaBackend
from .compare2d import compare_layouts_2d
from .config import VisDefaults, configure as configure_defaults, get_defaults, reset_defaults, set_defaults
from .layout2d import LayoutBranch2D, build_layout_branches_2d
from .plot2d import plot2d
from .plot3d import plot3d
from .scene import BranchPolyline3D, BranchTypeBatch3D, OverlaySpec, Polygon2D, RenderRequest, RenderScene2D, \
    RenderScene3D
from .scene2d import build_projected_scene_2d, build_render_scene_2d, build_scene2d_frustum, build_scene2d_projected, \
    build_scene2d_tree
from .scene3d import build_render_scene_3d

__all__ = [
    "BackendChooser",
    "BranchPolyline3D",
    "BranchTypeBatch3D",
    "VisDefaults",
    "LayoutBranch2D",
    "MatplotlibBackend",
    "OverlaySpec",
    "Polygon2D",
    "PyVistaBackend",
    "RenderBackend",
    "RenderRequest",
    "RenderScene2D",
    "RenderScene3D",
    "build_layout_branches_2d",
    "build_projected_scene_2d",
    "build_render_scene_2d",
    "build_render_scene_3d",
    "build_scene2d_frustum",
    "build_scene2d_projected",
    "build_scene2d_tree",
    "compare_layouts_2d",
    "configure_defaults",
    "get_defaults",
    "plot2d",
    "plot3d",
    "reset_defaults",
    "set_defaults",
]
