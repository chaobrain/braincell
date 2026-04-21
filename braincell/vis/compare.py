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

"""Generalized side-by-side comparison plots.

``compare2d.compare_layouts_2d`` renders the *same* morphology under
multiple layout families. The helpers here go the other way: they
render a *list* of morphologies (or a list of value arrays on the same
morphology) side by side on a shared figure. They are thin wrappers
over :func:`plot2d` so every styling knob (layout, shape, overlays,
value spec, publication theme) is available unchanged.

Examples
--------

.. code-block:: python

    >>> from braincell.vis import compare_morphologies, compare_values
    >>> fig, axes = compare_morphologies([tree_a, tree_b], layout="stem")  # doctest: +SKIP
    >>> fig, axes = compare_values(tree, [v_before, v_after])              # doctest: +SKIP
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from .layout import LayoutConfig
from .plot2d import plot2d


def compare_morphologies(
    morphologies: Sequence[Any],
    *,
    titles: Sequence[str] | None = None,
    layout: str | None = None,
    shape: str | None = None,
    align: str | None = "soma",
    figsize: tuple[float, float] | None = None,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout_config: LayoutConfig | None = None,
) -> tuple[object, tuple[object, ...]]:
    """Render several morphologies side by side with a shared layout.

    Parameters
    ----------
    morphologies : sequence of Morphology
        Morphologies to compare. Order left → right matches panel
        order. Must contain at least one element.
    titles : sequence of str or None
        Optional panel titles; defaults to ``morpho.name`` (or
        ``"morpho_{i}"`` for unnamed trees).
    layout : str or None
        Layout family passed to :func:`plot2d` for every panel.
    shape : str or None
        ``"line"`` or ``"frustum"``, forwarded to :func:`plot2d`.
    align : {"soma", "root", None}
        Origin alignment hint. Currently informational — the
        per-morphology layout engine always roots at the soma — but
        recorded on the axes title so scripts can cross-reference a
        stated alignment policy.
    figsize : tuple of floats or None
        Figure size. Defaults to ``(4.5 * n_panels, 4.5)``.
    min_branch_angle_deg, root_layout, layout_config
        Forwarded verbatim to :func:`plot2d`.

    Returns
    -------
    figure, axes : matplotlib.figure.Figure, tuple of Axes
        The composed figure and a tuple of the panel axes, left to
        right.

    Raises
    ------
    ValueError
        If ``morphologies`` is empty.

    Notes
    -----
    All panels share data-space aspect ratio through :func:`plot2d`
    (each call invokes ``ax.set_aspect('equal')``); the wrapper does
    *not* synchronise axis limits across panels because morphologies
    with different footprints would then be rendered at different
    scales.
    """
    if not morphologies:
        raise ValueError("compare_morphologies(...) requires at least one morphology.")

    import matplotlib.pyplot as plt

    n_panels = len(morphologies)
    default_figsize = (4.5 * n_panels, 4.5)
    fig, ax_array = plt.subplots(1, n_panels, figsize=figsize or default_figsize, squeeze=False)
    render_axes = tuple(ax_array[0])

    resolved_titles = _resolve_titles(morphologies, titles)

    for morpho, ax, title in zip(morphologies, render_axes, resolved_titles):
        plot2d(
            morpho,
            layout=layout,
            shape=shape,
            ax=ax,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout_config=layout_config,
        )
        suffix = f" ({align})" if align else ""
        ax.set_title(f"{title}{suffix}")

    return fig, render_axes


def compare_values(
    morpho: Any,
    value_arrays: Sequence[np.ndarray],
    *,
    titles: Sequence[str] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    value_label: str | None = None,
    layout: str | None = None,
    shape: str | None = None,
    figsize: tuple[float, float] | None = None,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout_config: LayoutConfig | None = None,
) -> tuple[object, tuple[object, ...]]:
    """Render the same morphology with multiple value overlays.

    Useful for before/after visualisations: ``compare_values(cell,
    [v_baseline, v_after_stim])`` produces two panels sharing
    morphology geometry with independent colour scales. When
    ``vmin`` / ``vmax`` are explicit they are applied to every panel
    for a shared range; otherwise each panel auto-scales and receives
    its own colourbar.

    Parameters
    ----------
    morpho : Morphology
        Shared morphology for every panel.
    value_arrays : sequence of arrays
        One scalar array per panel. Each array is interpreted through
        the normal :class:`ValueSpec` machinery — per-branch,
        per-segment, or per-centerline-point.
    titles : sequence of str or None
        Optional panel titles.
    cmap, vmin, vmax, value_label
        Forwarded to :func:`plot2d`.
    layout, shape, min_branch_angle_deg, root_layout, layout_config
        Forwarded to :func:`plot2d`.
    figsize : tuple of floats or None
        Figure size. Defaults to ``(4.5 * n_panels, 4.5)``.

    Returns
    -------
    figure, axes : matplotlib.figure.Figure, tuple of Axes
        The composed figure and per-panel axes.

    Raises
    ------
    ValueError
        If ``value_arrays`` is empty.
    """
    if not value_arrays:
        raise ValueError("compare_values(...) requires at least one value array.")

    import matplotlib.pyplot as plt

    n_panels = len(value_arrays)
    default_figsize = (4.5 * n_panels, 4.5)
    fig, ax_array = plt.subplots(1, n_panels, figsize=figsize or default_figsize, squeeze=False)
    render_axes = tuple(ax_array[0])

    resolved_titles = titles if titles is not None else tuple(f"panel {i}" for i in range(n_panels))
    if len(resolved_titles) != n_panels:
        raise ValueError(
            f"compare_values(...) received {len(resolved_titles)} titles for {n_panels} panels."
        )

    for values, ax, title in zip(value_arrays, render_axes, resolved_titles):
        plot2d(
            morpho,
            values=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            value_label=value_label,
            layout=layout,
            shape=shape,
            ax=ax,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout_config=layout_config,
        )
        ax.set_title(str(title))

    return fig, render_axes


def _resolve_titles(
    morphologies: Sequence[Any],
    titles: Sequence[str] | None,
) -> tuple[str, ...]:
    if titles is not None:
        if len(titles) != len(morphologies):
            raise ValueError(
                f"compare_morphologies(...) received {len(titles)} titles for {len(morphologies)} morphologies."
            )
        return tuple(str(t) for t in titles)
    resolved: list[str] = []
    for index, morpho in enumerate(morphologies):
        name = getattr(morpho, "name", None)
        resolved.append(str(name) if name else f"morpho_{index}")
    return tuple(resolved)
