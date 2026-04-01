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

from __future__ import annotations

from .backend import BackendChooser, validate_backend_for_scene
from .scene import OverlaySpec, RenderRequest
from .scene2d import build_render_scene_2d


def plot2d(
    morpho,
    *,
    region=None,
    locset=None,
    values=None,
    mode: str = "projected",
    backend: str | None = None,
    chooser: BackendChooser | None = None,
    notebook: bool | None = None,
    jupyter_backend: str | None = None,
    return_plotter: bool = False,
    projection_plane: str = "xy",
) -> object:
    from braincell.morpho import Morpho

    if not isinstance(morpho, Morpho):
        raise TypeError(f"plot2d(...) expects Morpho, got {type(morpho).__name__!s}.")

    scene = build_render_scene_2d(morpho, mode=mode, projection_plane=projection_plane)
    chooser = chooser or BackendChooser.default()
    request = RenderRequest(
        morpho=morpho,
        overlay=OverlaySpec(region=region, locset=locset, values=values),
        dimensionality="2d",
        mode=mode,
        scene=scene,
        notebook=notebook,
        jupyter_backend=jupyter_backend,
        return_plotter=return_plotter,
    )
    backend_impl = chooser.pick(requested=backend, scene_kind="2d")
    validate_backend_for_scene(backend_impl, scene)
    return backend_impl.render(request)
