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


from .backend import BackendChooser, validate_backend_for_scene
from .config import resolve_default_2d_layout, resolve_default_2d_shape
from .scene import OverlaySpec, RenderRequest
from .scene2d import build_render_scene_2d


def plot2d(
    morpho,
    *,
    region=None,
    locset=None,
    values=None,
    layout: str | None = None,
    shape: str | None = None,
    backend: str | None = None,
    chooser: BackendChooser | None = None,
    ax=None,
    notebook: bool | None = None,
    jupyter_backend: str | None = None,
    return_plotter: bool = False,
    projection_plane: str = "xy",
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
) -> object:
    from braincell.morph import Morphology

    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot2d(...) expects Morpho, got {type(morpho).__name__!s}.")

    resolved_layout = resolve_default_2d_layout(layout)
    resolved_shape = resolve_default_2d_shape(shape)
    overlay = OverlaySpec(region=region, locset=locset, values=values)
    scene = build_render_scene_2d(
        morpho,
        layout=resolved_layout,
        shape=resolved_shape,
        projection_plane=projection_plane,
        min_branch_angle_deg=min_branch_angle_deg,
        root_layout=root_layout,
        overlay=overlay,
    )
    chooser = chooser or BackendChooser.default()
    backend_options: dict = {}
    if ax is not None:
        backend_options["ax"] = ax
    if notebook is not None:
        backend_options["notebook"] = notebook
    if jupyter_backend is not None:
        backend_options["jupyter_backend"] = jupyter_backend
    if return_plotter:
        backend_options["return_plotter"] = True
    request = RenderRequest(
        morpho=morpho,
        scene=scene,
        overlay=overlay,
        dimensionality="2d",
        layout=resolved_layout,
        shape=resolved_shape,
        backend_options=backend_options,
    )
    backend_impl = chooser.pick(requested=backend, scene_kind="2d")
    validate_backend_for_scene(backend_impl, scene)
    return backend_impl.render(request)
