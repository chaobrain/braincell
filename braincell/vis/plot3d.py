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
from .config import SUPPORTED_3D_MODES, resolve_default_3d_mode
from .scene import OverlaySpec, RenderRequest
from .scene3d import build_render_scene_3d


def plot3d(
    morpho,
    *,
    region=None,
    locset=None,
    values=None,
    mode: str | None = None,
    backend: str | None = None,
    chooser: BackendChooser | None = None,
    notebook: bool | None = None,
    jupyter_backend: str | None = None,
    return_plotter: bool = False,
) -> object:
    from braincell.morph import Morphology

    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot3d(...) expects Morpho, got {type(morpho).__name__!s}.")
    resolved_mode = resolve_default_3d_mode(mode)
    if resolved_mode not in SUPPORTED_3D_MODES:
        expected = ", ".join(sorted(repr(item) for item in SUPPORTED_3D_MODES))
        raise ValueError(f"Unsupported 3D mode {resolved_mode!r}. Expected one of {expected}.")

    overlay = OverlaySpec(region=region, locset=locset, values=values)
    scene = build_render_scene_3d(morpho, mode=resolved_mode, overlay=overlay)
    chooser = chooser or BackendChooser.default()
    backend_options: dict = {}
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
        dimensionality="3d",
        mode=resolved_mode,
        backend_options=backend_options,
    )
    backend_impl = chooser.pick(requested=backend, scene_kind="3d")
    validate_backend_for_scene(backend_impl, scene)
    return backend_impl.render(request)
