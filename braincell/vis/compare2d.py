from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .backend import BackendChooser, validate_backend_for_scene
from .scene import OverlaySpec, RenderRequest
from .scene2d import build_render_scene_2d


def compare_layouts_2d(
    morpho,
    *,
    layout_families: Sequence[str] = ("stem", "balloon", "radial_360"),
    mode: str = "tree",
    chooser: BackendChooser | None = None,
    backend: str = "matplotlib",
    axes=None,
    figsize: tuple[float, float] | None = None,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
) -> tuple[object, tuple[object, ...]]:
    from braincell.morpho import Morpho

    if not isinstance(morpho, Morpho):
        raise TypeError(f"compare_layouts_2d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if not layout_families:
        raise ValueError("compare_layouts_2d(...) requires at least one layout family.")

    import matplotlib.pyplot as plt

    layout_families = tuple(layout_families)
    if axes is None:
        ncols = len(layout_families)
        default_figsize = (4.5 * ncols, 4.5)
        fig, ax_array = plt.subplots(1, ncols, figsize=figsize or default_figsize, squeeze=False)
        render_axes = tuple(ax_array[0])
    else:
        render_axes = tuple(np.ravel(np.asarray(axes, dtype=object)))
        if len(render_axes) != len(layout_families):
            raise ValueError(
                f"compare_layouts_2d(...) received {len(render_axes)} axes for {len(layout_families)} layout families."
            )
        fig = render_axes[0].figure

    chooser = chooser or BackendChooser.default()
    backend_impl = chooser.pick(requested=backend, scene_kind="2d")

    for layout_family, ax in zip(layout_families, render_axes):
        scene = build_render_scene_2d(
            morpho,
            mode=mode,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout_family=layout_family,
        )
        validate_backend_for_scene(backend_impl, scene)
        backend_impl.render(
            RenderRequest(
                morpho=morpho,
                overlay=OverlaySpec(),
                dimensionality="2d",
                mode=mode,
                scene=scene,
                ax=ax,
            )
        )
        ax.set_title(_layout_family_title(layout_family))

    return fig, render_axes


def _layout_family_title(layout_family: str) -> str:
    return {
        "stem": "Stem",
        "trunk_first": "Stem",
        "balloon": "Balloon",
        "radial_360": "Radial 360",
    }.get(layout_family, layout_family.replace("_", " ").title())
