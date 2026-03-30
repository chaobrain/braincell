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
