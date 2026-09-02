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

"""Topology plots of a :class:`~braincell.Cell` at three abstraction levels.

A cell can be drawn as a graph at three granularities, from finest to
coarsest:

``"node"``
    One node per runtime point. This is the level the solver works at, so
    it is the only one that can show per-point runtime state.
``"cv"``
    One node per control volume.
``"branch"``
    One node per morphology branch. Topology only — a branch has no single
    runtime value to show.

All three share one entry point, :func:`plot_cell_topology`, because they
share a selector model: highlight by ``region`` / ``locset``, or colour by
``value``. Resolving those selectors against the cell is
:mod:`braincell._multi_compartment.field_resolution`'s job; this module
only decides what to draw and hands the result to the renderer in
:mod:`braincell.vis.point_topology`.
"""

from typing import Any, Literal, get_args

from braincell._multi_compartment import field_resolution
from braincell._multi_compartment.cell import Cell
from braincell.filter import LocsetExpr, LocsetMask, RegionExpr, RegionMask
from .point_topology import CoverageMode, _plot_discrete_topology_graph, plot_point_topology

__all__ = ["plot_cell_topology"]

Level = Literal["node", "cv", "branch"]

_VALID_LEVELS: tuple[str, ...] = get_args(Level)


def plot_cell_topology(
    cell: Cell,
    *,
    level: Level = "node",
    preset: str = "dendrotweaks",
    layout: str | None = None,
    layout_scale: float = 1.0,
    region: RegionExpr | RegionMask | None = None,
    locset: LocsetExpr | LocsetMask | None = None,
    coverage_mode: CoverageMode = "fraction",
    highlight_color: str = "#ef4444",
    value=None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    value_label: str | None = None,
    show_colorbar: bool = True,
    node_color: str | None = None,
    edge_color: str | None = None,
    root_color: str | None = None,
    ax=None,
) -> Any:
    """Draw a cell's topology at the node, CV, or branch level.

    Parameters
    ----------
    cell : braincell.Cell
        Cell to render. ``level="node"`` requires an initialized cell;
        ``"cv"`` and ``"branch"`` also work on a declaration-only cell.
    level : {"node", "cv", "branch"}, optional
        Topology abstraction level. One node per runtime point, per
        control volume, or per morphology branch respectively.
    preset : str, optional
        Name of the built-in topology preset. Presets bundle default
        layout and colour settings.
    layout : str or None, optional
        Explicit layout algorithm override. When ``None``, uses the
        preset's layout.
    layout_scale : float, optional
        Global spacing multiplier for the resolved layout.
    region : braincell.filter.RegionExpr or braincell.filter.RegionMask or None, optional
        Continuous morphology selection to highlight. At ``level="node"``
        and ``"cv"`` the selection is reduced to CV granularity; at
        ``level="node"`` each selected CV is attributed to its midpoint
        point.
    locset : braincell.filter.LocsetExpr or braincell.filter.LocsetMask or None, optional
        Discrete morphology locations to highlight, each mapped to the CV
        that owns it. Not supported at ``level="branch"``.
    coverage_mode : {"fraction", "any", "all"}, optional
        Coverage display rule for ``region``. ``"fraction"`` blends by
        overlap fraction, ``"any"`` highlights any overlap fully, and
        ``"all"`` only highlights fully covered nodes. Locset-backed
        highlights are always full intensity.
    highlight_color : str, optional
        Colour used for highlighted nodes.
    value : object, optional
        Colouring source. Not supported at ``level="branch"``. Supported
        forms are:

        - an array in the level's own space (``n_point`` or ``n_cv``)
        - an array in the other space, which is mapped across
        - ``"V"`` or ``"voltage"``
        - ``("ion", ion_name, field)``
        - ``("channel", class_name, field)``
        - ``("layout_id", layout_id, field)``

        ``value`` is mutually exclusive with ``region`` / ``locset``.
    cmap : str or None, optional
        Matplotlib colormap name used in value mode.
    vmin, vmax : float or None, optional
        Explicit lower and upper bounds for the value colormap.
    norm : matplotlib.colors.Normalize or None, optional
        Explicit normalization object for value mode.
    value_label : str or None, optional
        Colorbar label override. When ``None``, named value selectors
        infer a label automatically.
    show_colorbar : bool, optional
        If ``True`` (default), draw a colorbar in value mode.
    node_color, edge_color, root_color : str or None, optional
        Base style colour overrides.
    ax : matplotlib.axes.Axes or None, optional
        Destination axes. When ``None``, a fresh figure and axes are
        created.

    Returns
    -------
    matplotlib.axes.Axes
        The rendered axes. Call ``matplotlib.pyplot.show()`` to display
        it; like every other ``braincell.vis`` entry point this function
        does not display on your behalf.

    Raises
    ------
    TypeError
        If ``cell`` is not a :class:`~braincell.Cell`.
    ValueError
        If ``level`` is not one of the three known levels; if
        ``level="branch"`` is combined with a parameter it does not
        support; if ``value`` is combined with ``region`` / ``locset``;
        if ``level="cv"`` is used on a cell without exactly one root CV;
        or if a supplied value source cannot be mapped into the level's
        space.
    RuntimeError
        If ``level="node"`` is used before :meth:`~braincell.Cell.init_state`.

    See Also
    --------
    braincell.vis.plot_topology
        Branch-order schematic of a :class:`~braincell.Morphology`, drawn
        without a cell and ignoring segment lengths — a different
        rendering of a different object.
    braincell.vis.plot_point_topology
        The underlying renderer, taking a bare
        :class:`~braincell.NodeTree` and pre-resolved highlight or value
        arrays.

    Notes
    -----
    Highlight mode (``region`` / ``locset``) and value mode (``value``)
    are mutually exclusive. Region and locset mappings use CV midpoint
    semantics, matching the runtime lowering model.

    ``level="branch"`` is topology-only: it accepts ``region`` for
    coverage but rejects ``locset``, ``value``, and every value-colormap
    parameter, because a branch spans many CVs and so carries no single
    runtime value.

    Examples
    --------
    Colour the node topology by membrane voltage:

    .. code-block:: python

        >>> import braincell as bc
        >>> import matplotlib.pyplot as plt
        >>> ax = bc.vis.plot_cell_topology(cell, value="V", cmap="viridis")  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

    Highlight a region at control-volume level:

    .. code-block:: python

        >>> ax = bc.vis.plot_cell_topology(  # doctest: +SKIP
        ...     cell, level="cv", region=bc.filter.BranchSlice(1, 0.0, 0.5)
        ... )

    Show which branches a region covers:

    .. code-block:: python

        >>> ax = bc.vis.plot_cell_topology(cell, level="branch", region=region)  # doctest: +SKIP

    Colour nodes by a channel parameter:

    .. code-block:: python

        >>> ax = bc.vis.plot_cell_topology(  # doctest: +SKIP
        ...     cell, value=("channel", "IL", "g_max")
        ... )
    """
    if not isinstance(cell, Cell):
        raise TypeError(f"plot_cell_topology(...) expects Cell, got {type(cell).__name__!s}.")
    if level not in _VALID_LEVELS:
        options = ", ".join(repr(name) for name in _VALID_LEVELS)
        raise ValueError(f"plot_cell_topology(...) level must be one of {{{options}}}, got {level!r}.")

    if level == "branch":
        return _plot_branch_level(
            cell,
            preset=preset,
            layout=layout,
            layout_scale=layout_scale,
            region=region,
            locset=locset,
            coverage_mode=coverage_mode,
            highlight_color=highlight_color,
            value=value,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            value_label=value_label,
            show_colorbar=show_colorbar,
            node_color=node_color,
            edge_color=edge_color,
            root_color=root_color,
            ax=ax,
        )

    renderer = _plot_node_level if level == "node" else _plot_cv_level
    return renderer(
        cell,
        preset=preset,
        layout=layout,
        layout_scale=layout_scale,
        region=region,
        locset=locset,
        coverage_mode=coverage_mode,
        highlight_color=highlight_color,
        value=value,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        value_label=value_label,
        show_colorbar=show_colorbar,
        node_color=node_color,
        edge_color=edge_color,
        root_color=root_color,
        ax=ax,
    )


def _reject_value_with_highlight(*, caller: str, value, region, locset) -> None:
    if value is not None and (region is not None or locset is not None):
        raise ValueError(f"{caller} does not support value together with region/locset highlighting.")


def _plot_node_level(
    cell,
    *,
    preset,
    layout,
    layout_scale,
    region,
    locset,
    coverage_mode,
    highlight_color,
    value,
    cmap,
    vmin,
    vmax,
    norm,
    value_label,
    show_colorbar,
    node_color,
    edge_color,
    root_color,
    ax,
) -> Any:
    """Render one node per runtime point."""
    caller = "plot_cell_topology(level='node', ...)"
    field_resolution.require_initialized(cell, caller)
    _reject_value_with_highlight(caller=caller, value=value, region=region, locset=locset)

    highlight_fractions = None
    values = None
    resolved_value_label = value_label
    if region is not None or locset is not None:
        highlight_fractions = field_resolution.node_highlight_fractions(
            cell,
            region=region,
            locset=locset,
            caller=caller,
        )
    elif value is not None:
        values, inferred_label = field_resolution.resolve_node_field_values(cell, value, caller=caller)
        if resolved_value_label is None:
            resolved_value_label = inferred_label

    return plot_point_topology(
        cell.node_tree,
        preset=preset,
        layout=layout,
        layout_scale=layout_scale,
        highlight_fractions=highlight_fractions,
        coverage_mode=coverage_mode,
        highlight_color=highlight_color,
        values=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        value_label=resolved_value_label,
        show_colorbar=show_colorbar,
        node_color=node_color,
        edge_color=edge_color,
        root_color=root_color,
        ax=ax,
    )


def _plot_cv_level(
    cell,
    *,
    preset,
    layout,
    layout_scale,
    region,
    locset,
    coverage_mode,
    highlight_color,
    value,
    cmap,
    vmin,
    vmax,
    norm,
    value_label,
    show_colorbar,
    node_color,
    edge_color,
    root_color,
    ax,
) -> Any:
    """Render one node per control volume."""
    caller = "plot_cell_topology(level='cv', ...)"
    cvs = cell.cvs
    root_ids = [cv.id for cv in cvs if cv.parent_cv is None]
    if len(root_ids) != 1:
        raise ValueError(f"{caller} expects exactly one root CV, got {root_ids!r}.")
    _reject_value_with_highlight(caller=caller, value=value, region=region, locset=locset)

    coverage_fractions = None
    values = None
    resolved_value_label = value_label
    if region is not None or locset is not None:
        coverage_fractions = field_resolution.cv_highlight_fractions(
            cell,
            region=region,
            locset=locset,
            caller=caller,
        )
    elif value is not None:
        values, inferred_label = field_resolution.resolve_cv_field_values(cell, value, caller=caller)
        if resolved_value_label is None:
            resolved_value_label = inferred_label

    return _plot_discrete_topology_graph(
        node_ids=tuple(cv.id for cv in cvs),
        edges=tuple((int(cv.parent_cv), int(cv.id)) for cv in cvs if cv.parent_cv is not None),
        root_id=int(root_ids[0]),
        preset=preset,
        layout=layout,
        layout_scale=layout_scale,
        highlight_fractions=coverage_fractions,
        coverage_mode=coverage_mode,
        highlight_color=highlight_color,
        values=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        value_label=resolved_value_label,
        show_colorbar=show_colorbar,
        node_color=node_color,
        edge_color=edge_color,
        root_color=root_color,
        ax=ax,
        caller=caller,
    )


def _plot_branch_level(
    cell,
    *,
    preset,
    layout,
    layout_scale,
    region,
    locset,
    coverage_mode,
    highlight_color,
    value,
    cmap,
    vmin,
    vmax,
    norm,
    value_label,
    show_colorbar,
    node_color,
    edge_color,
    root_color,
    ax,
) -> Any:
    """Render one node per morphology branch.

    Topology only: a branch spans many CVs, so it carries no single
    runtime value and every value-mode parameter is rejected.
    """
    caller = "plot_cell_topology(level='branch', ...)"
    unsupported = [
        name
        for name, supplied in (
            ("locset", locset is not None),
            ("value", value is not None),
            ("cmap", cmap is not None),
            ("vmin", vmin is not None),
            ("vmax", vmax is not None),
            ("norm", norm is not None),
            ("value_label", value_label is not None),
            ("show_colorbar", show_colorbar is not True),
        )
        if supplied
    ]
    if unsupported:
        raise ValueError(
            f"{caller} does not support: {', '.join(unsupported)}. Branch-level rendering is topology-only."
        )

    morpho = cell.morpho
    coverage_fractions = (
        None if region is None else field_resolution.branch_coverage_fractions(cell, region, caller=caller)
    )

    return _plot_discrete_topology_graph(
        node_ids=tuple(branch.index for branch in morpho.branches),
        edges=tuple((edge.parent.index, edge.child.index) for edge in morpho.edges),
        root_id=int(morpho.root.index),
        preset=preset,
        layout=layout,
        layout_scale=layout_scale,
        highlight_fractions=coverage_fractions,
        coverage_mode=coverage_mode,
        highlight_color=highlight_color,
        node_color=node_color,
        edge_color=edge_color,
        root_color=root_color,
        ax=ax,
        caller=caller,
    )
