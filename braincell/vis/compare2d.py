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

from collections.abc import Sequence

import numpy as np

from .backend import BackendChooser, validate_backend_for_scene
from .layout import LayoutConfig
from .scene import OverlaySpec, RenderRequest
from .scene2d import build_render_scene_2d


def compare_layouts_2d(
    morpho,
    *,
    layouts: Sequence[str] = ("fan", "stem", "balloon", "radial_360"),
    shape: str = "line",
    chooser: BackendChooser | None = None,
    backend: str = "matplotlib",
    axes=None,
    figsize: tuple[float, float] | None = None,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout_config: LayoutConfig | None = None,
) -> tuple[object, tuple[object, ...]]:
    from braincell import Morphology

    if not isinstance(morpho, Morphology):
        raise TypeError(f"compare_layouts_2d(...) expects Morpho, got {type(morpho).__name__!s}.")
    if not layouts:
        raise ValueError("compare_layouts_2d(...) requires at least one layout family.")

    import matplotlib.pyplot as plt

    layouts = tuple(layouts)
    if axes is None:
        ncols = len(layouts)
        default_figsize = (4.5 * ncols, 4.5)
        fig, ax_array = plt.subplots(1, ncols, figsize=figsize or default_figsize, squeeze=False)
        render_axes = tuple(ax_array[0])
    else:
        render_axes = tuple(np.ravel(np.asarray(axes, dtype=object)))
        if len(render_axes) != len(layouts):
            raise ValueError(
                f"compare_layouts_2d(...) received {len(render_axes)} axes for {len(layouts)} layouts."
            )
        fig = render_axes[0].figure

    chooser = chooser or BackendChooser.default()
    backend_impl = chooser.pick(requested=backend, scene_kind="2d")

    for layout, ax in zip(layouts, render_axes):
        scene = build_render_scene_2d(
            morpho,
            layout=layout,
            shape=shape,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout_config=layout_config,
        )
        validate_backend_for_scene(backend_impl, scene)
        backend_impl.render(
            RenderRequest(
                morpho=morpho,
                scene=scene,
                overlay=OverlaySpec(),
                dimensionality="2d",
                layout=layout,
                shape=shape,
                backend_options={"ax": ax},
            )
        )
        ax.set_title(_layout_title(layout))

    return fig, render_axes


def _layout_title(layout: str) -> str:
    return {
        "fan": "Fan",
        "stem": "Stem",
        "trunk_first": "Stem",
        "balloon": "Balloon",
        "radial_360": "Radial 360",
    }.get(layout, layout.replace("_", " ").title())
