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
from .layout import LayoutConfig
from .scene import OverlaySpec, RenderRequest, ValueSpec
from .scene2d import build_render_scene_2d


def plot2d(
    morpho,
    *,
    region=None,
    locset=None,
    values=None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    value_label: str | None = None,
    show_colorbar: bool = True,
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
    layout_config: LayoutConfig | None = None,
) -> object:
    from braincell.morph import Morphology

    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot2d(...) expects Morpho, got {type(morpho).__name__!s}.")

    resolved_layout = resolve_default_2d_layout(layout)
    resolved_shape = resolve_default_2d_shape(shape)
    values_spec = _build_value_spec(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        value_label=value_label,
        show_colorbar=show_colorbar,
    )
    overlay = OverlaySpec(region=region, locset=locset, values=values_spec)
    scene = build_render_scene_2d(
        morpho,
        layout=resolved_layout,
        shape=resolved_shape,
        projection_plane=projection_plane,
        min_branch_angle_deg=min_branch_angle_deg,
        root_layout=root_layout,
        overlay=overlay,
        layout_config=layout_config,
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


def _build_value_spec(
    values,
    *,
    cmap: str | None,
    vmin: float | None,
    vmax: float | None,
    norm,
    value_label: str | None,
    show_colorbar: bool,
) -> ValueSpec | None:
    """Normalize ``plot2d``/``plot3d`` value keywords into a :class:`ValueSpec`."""
    if values is None:
        if any(k is not None for k in (cmap, vmin, vmax, norm, value_label)):
            raise ValueError(
                "values=... is required when passing cmap/vmin/vmax/norm/value_label."
            )
        return None
    if isinstance(values, ValueSpec):
        # Fold caller-supplied style kwargs on top of the spec. A kwarg
        # that's None leaves the spec field untouched.
        return ValueSpec(
            values=values.values,
            cmap=cmap if cmap is not None else values.cmap,
            vmin=vmin if vmin is not None else values.vmin,
            vmax=vmax if vmax is not None else values.vmax,
            norm=norm if norm is not None else values.norm,
            label=value_label if value_label is not None else values.label,
            unit_label=values.unit_label,
            show_colorbar=show_colorbar if show_colorbar is not values.show_colorbar else values.show_colorbar,
        )
    return ValueSpec(
        values=values,
        cmap=cmap if cmap is not None else "viridis",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        label=value_label,
        show_colorbar=show_colorbar,
    )
