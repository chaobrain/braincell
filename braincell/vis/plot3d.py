from __future__ import annotations

from .backend import BackendChooser, validate_backend_for_scene
from .scene import OverlaySpec, RenderRequest
from .scene3d import build_render_scene_3d


def plot3d(
    morpho,
    *,
    region=None,
    locset=None,
    values=None,
    mode: str = "geometry",
    backend: str | None = None,
    chooser: BackendChooser | None = None,
    notebook: bool | None = None,
    jupyter_backend: str | None = None,
    return_plotter: bool = False,
) -> object:
    from braincell.morpho import Morpho

    if not isinstance(morpho, Morpho):
        raise TypeError(f"plot3d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if mode != "geometry":
        raise ValueError(f"Unsupported 3D mode {mode!r}. Expected 'geometry'.")

    scene = build_render_scene_3d(morpho)
    chooser = chooser or BackendChooser.default()
    request = RenderRequest(
        morpho=morpho,
        overlay=OverlaySpec(region=region, locset=locset, values=values),
        dimensionality="3d",
        mode=mode,
        scene=scene,
        notebook=notebook,
        jupyter_backend=jupyter_backend,
        return_plotter=return_plotter,
    )
    backend_impl = chooser.pick(requested=backend, scene_kind="3d")
    validate_backend_for_scene(backend_impl, scene)
    return backend_impl.render(request)
