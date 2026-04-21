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

"""Time-series trace rendering synchronized with a morphology view.

``plot_traces`` draws a stack of per-location time-series panels and
optionally renders the parent morphology alongside with coloured
markers at each sampled location. The marker colour and the trace
colour always match, so the reader can visually link a spike in the
trace panel to a specific point on the dendrite.

The function keeps its dependency surface minimal: only matplotlib is
required. The layout uses a two-column :class:`~matplotlib.gridspec.GridSpec`
— the left column hosts the morphology view produced by
:func:`plot2d`, and the right column stacks one axis per trace.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from braincell.filter import LocsetMask
from braincell.morph._morphology import Morphology


@dataclass(frozen=True)
class TracesResult:
    """Axes handles returned from :func:`plot_traces`."""

    figure: object
    morpho_axes: object
    trace_axes: tuple[object, ...]


def plot_traces(
    morpho: Morphology,
    time,
    values_over_time,
    *,
    locset: LocsetMask | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence | None = None,
    cmap: str = "tab10",
    layout: str | None = None,
    shape: str | None = None,
    time_unit_label: str | None = None,
    value_unit_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    show_morphology: bool = True,
) -> TracesResult:
    """Plot time-series traces at selected morphology locations.

    Parameters
    ----------
    morpho : Morphology
        Source morphology. Used for both the left-hand morphology
        view (when ``show_morphology=True``) and to validate that the
        number of traces matches the number of locset points.
    time : ArrayLike
        1-D array of timestamps with length ``T``. Units are stripped
        via :mod:`brainunit` for plotting; if no unit is available the
        axis label is left blank unless *time_unit_label* is given.
    values_over_time : ArrayLike
        ``(T, n_locations)`` matrix of trace samples. ``n_locations``
        must equal either ``len(locset.points)`` (if *locset* is
        supplied) or the number of columns of the array (otherwise).
    locset : LocsetMask or None
        Evaluated locset; each point becomes a marker on the
        morphology and a matching trace panel on the right.
    labels : sequence of str or None
        Optional labels per trace. When ``None``, panels are labelled
        ``Loc 0``, ``Loc 1``, ....
    colors : sequence or None
        Optional explicit per-trace colours (any matplotlib colour
        spec). When ``None``, colours are drawn from *cmap*.
    cmap : str
        Matplotlib colormap name used to auto-colour traces when
        *colors* is ``None``.
    layout, shape
        Layout / shape parameters forwarded to :func:`plot2d` for the
        morphology panel.
    time_unit_label, value_unit_label : str or None
        Axis-label unit strings. These override any brainunit units
        attached to *time* / *values_over_time*.
    figsize : tuple or None
        Figure size. When ``None``, defaults to
        ``(10, 2 * n_locations)``.
    sharex : bool
        If ``True`` (default), all trace panels share the time axis.
    show_morphology : bool
        If ``False``, no morphology view is drawn and only the trace
        stack is returned.

    Returns
    -------
    TracesResult
        Figure, morphology axes (``None`` when the view is hidden),
        and the tuple of trace axes.

    Raises
    ------
    ValueError
        If the column count of *values_over_time* does not match the
        number of sampled locations.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot_traces(...) expects Morphology, got {type(morpho).__name__!s}.")

    import matplotlib.pyplot as plt

    t_raw, t_unit_detected = _strip_quantity(time)
    t_raw = np.asarray(t_raw, dtype=float)
    if t_raw.ndim != 1:
        raise ValueError(f"plot_traces(...) expects a 1-D time array, got shape {t_raw.shape!r}.")
    time_label = _axis_label("time", time_unit_label, t_unit_detected)

    values_raw, values_unit_detected = _strip_quantity(values_over_time)
    values_raw = np.asarray(values_raw, dtype=float)
    if values_raw.ndim != 2:
        raise ValueError(
            f"plot_traces(...) expects a 2-D (T, n_locations) values matrix, got shape {values_raw.shape!r}."
        )
    if values_raw.shape[0] != t_raw.shape[0]:
        raise ValueError(
            f"plot_traces(...) time length {t_raw.shape[0]} does not match values shape[0] {values_raw.shape[0]}."
        )
    value_label = _axis_label("value", value_unit_label, values_unit_detected)

    n_traces = values_raw.shape[1]
    if locset is not None and len(locset.points) != n_traces:
        raise ValueError(
            f"plot_traces(...) got {n_traces} traces but locset has {len(locset.points)} points."
        )

    label_list = list(labels) if labels is not None else [f"Loc {i}" for i in range(n_traces)]
    if len(label_list) != n_traces:
        raise ValueError(
            f"plot_traces(...) got {len(label_list)} labels for {n_traces} traces."
        )

    if colors is None:
        colormap = plt.get_cmap(cmap)
        color_list = [colormap(i / max(n_traces - 1, 1)) for i in range(n_traces)]
    else:
        color_list = list(colors)
        if len(color_list) != n_traces:
            raise ValueError(
                f"plot_traces(...) got {len(color_list)} colors for {n_traces} traces."
            )

    figsize = figsize or (10.0, max(2.0 * n_traces, 3.0))
    fig = plt.figure(figsize=figsize)

    if show_morphology:
        gs = fig.add_gridspec(n_traces, 2, width_ratios=(1.0, 1.5))
        morpho_ax = fig.add_subplot(gs[:, 0])
        trace_axes: list = []
        for row in range(n_traces):
            share = trace_axes[0] if (sharex and trace_axes) else None
            trace_axes.append(fig.add_subplot(gs[row, 1], sharex=share))
    else:
        morpho_ax = None
        trace_axes = []
        for row in range(n_traces):
            share = trace_axes[0] if (sharex and trace_axes) else None
            trace_axes.append(fig.add_subplot(n_traces, 1, row + 1, sharex=share))

    if morpho_ax is not None:
        from .plot2d import plot2d

        # Render the morphology view with the locset colour-synced to
        # the traces. We bypass plot2d's own locset overlay (which
        # would use a single marker colour) and instead manually draw
        # per-point markers in the same order as the traces.
        plot2d(morpho, layout=layout, shape=shape, ax=morpho_ax)
        if locset is not None:
            from .scene import OverlaySpec
            from .scene2d import _Centerline2D, build_render_scene_2d

            # Build a scene to recover the per-branch centerlines so
            # we can interpolate each locset point's 2D position.
            scene = build_render_scene_2d(
                morpho,
                layout=layout or "stem",
                shape=shape or "frustum",
                overlay=OverlaySpec(),
            )
            centerlines: dict[int, _Centerline2D] = {}
            for polyline in scene.polylines:
                centerlines.setdefault(
                    polyline.branch_index,
                    _Centerline2D.from_points(
                        branch_index=polyline.branch_index,
                        branch_name=polyline.branch_name,
                        branch_type=polyline.branch_type,
                        points_um=polyline.points_um,
                        widths_um=polyline.widths_um,
                    ),
                )
            for polygon in scene.polygons:
                if polygon.branch_index in centerlines:
                    continue
                pts = np.asarray(polygon.points_um, dtype=float)
                midline = 0.5 * (pts[:2] + pts[-2:][::-1])
                widths = np.full(len(midline), 1.0, dtype=float)
                centerlines[polygon.branch_index] = _Centerline2D.from_points(
                    branch_index=polygon.branch_index,
                    branch_name=polygon.branch_name,
                    branch_type=polygon.branch_type,
                    points_um=midline,
                    widths_um=widths,
                )

            for i, point in enumerate(locset.points):
                branch_index, x = int(point[0]), float(point[1])
                centerline = centerlines.get(branch_index)
                if centerline is None:
                    continue
                pos = centerline.point_at(x)
                morpho_ax.scatter(
                    pos[0],
                    pos[1],
                    s=60.0,
                    c=[color_list[i]],
                    edgecolors="black",
                    linewidths=0.75,
                    zorder=20_000,
                )

    for i, trace_ax in enumerate(trace_axes):
        trace_ax.plot(t_raw, values_raw[:, i], color=color_list[i], linewidth=1.25)
        trace_ax.set_ylabel(label_list[i])
        if value_label is not None:
            trace_ax.set_title(value_label if i == 0 else "")
        for spine in ("top", "right"):
            trace_ax.spines[spine].set_visible(False)
        if i < n_traces - 1:
            trace_ax.tick_params(labelbottom=False)
    if trace_axes:
        trace_axes[-1].set_xlabel(time_label or "")

    fig.tight_layout()
    return TracesResult(
        figure=fig,
        morpho_axes=morpho_ax,
        trace_axes=tuple(trace_axes),
    )


def _strip_quantity(values) -> tuple[np.ndarray, str | None]:
    try:
        import brainunit as u
    except ModuleNotFoundError:  # pragma: no cover
        return np.asarray(values, dtype=float), None
    if isinstance(values, u.Quantity):
        return np.asarray(u.get_mantissa(values), dtype=float), str(u.get_unit(values))
    return np.asarray(values, dtype=float), None


def _axis_label(kind: str, user_label: str | None, detected: str | None) -> str | None:
    if user_label is not None:
        return f"{kind} [{user_label}]"
    if detected is not None:
        return f"{kind} [{detected}]"
    return None
