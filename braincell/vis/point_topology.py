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

"""Node-tree topology plotting.

This module renders :class:`braincell._discretization.base.NodeTree`
instances as pure topology graphs: nodes become points in the plot, node
edges become graph edges, and 2-D coordinates come from a graph layout
algorithm rather than morphology geometry.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from braincell._discretization.base import NodeTree

ColorMode = Literal["solid", "depth", "values"]
CoverageMode = Literal["fraction", "any", "all"]


@dataclass(frozen=True)
class _PointTopologyPreset:
    layout: str
    color_mode: ColorMode
    cmap: str | None
    node_color: str
    edge_color: str
    root_color: str


_PRESETS: dict[str, _PointTopologyPreset] = {
    "dendrotweaks": _PointTopologyPreset(
        layout="twopi",
        color_mode="solid",
        cmap=None,
        node_color="#1b2a7a",
        edge_color="#8c97ab",
        root_color="#f28e2b",
    ),
    "mono": _PointTopologyPreset(
        layout="twopi",
        color_mode="solid",
        cmap=None,
        node_color="#2b2b2b",
        edge_color="#8d8d8d",
        root_color="#000000",
    ),
    "depth": _PointTopologyPreset(
        layout="twopi",
        color_mode="depth",
        cmap="viridis",
        node_color="#1b2a7a",
        edge_color="#8c97ab",
        root_color="#f28e2b",
    ),
}

_GRAPHVIZ_LAYOUTS = {"twopi", "dot", "neato"}
_LAYOUT_ALIASES = {"kamada-kawai": "kamada_kawai"}
_VALID_LAYOUTS = _GRAPHVIZ_LAYOUTS | {"kamada_kawai"}
_VALID_COLOR_MODES = {"solid", "depth", "values"}
_VALID_COVERAGE_MODES = {"fraction", "any", "all"}


def plot_point_topology(
    node_tree: NodeTree,
    *,
    preset: str = "dendrotweaks",
    layout: str | None = None,
    layout_scale: float = 1.0,
    highlight_point_ids=None,
    highlight_fractions: dict[int, float] | None = None,
    coverage_mode: CoverageMode = "fraction",
    highlight_color: str = "#ef4444",
    color_mode: ColorMode | None = None,
    values=None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    value_label: str | None = None,
    value_unit_label: str | None = None,
    show_colorbar: bool = True,
    node_color: str | None = None,
    edge_color: str | None = None,
    root_color: str | None = None,
    ax=None,
) -> Any:
    """Render a :class:`NodeTree` as a topology-only graph.

    Parameters
    ----------
    node_tree : NodeTree
        Declaration-time node tree to render.
    preset : str, optional
        Name of the built-in style preset. Presets bundle default
        layout and colour settings.
    layout : str or None, optional
        Explicit layout algorithm override. When ``None``, uses the
        layout implied by ``preset``.
    layout_scale : float, optional
        Global spacing multiplier for the resolved layout. This scales
        the full node layout; it does not alter pairwise target
        distances.
    highlight_point_ids : iterable of int or None, optional
        Point ids to highlight. Highlight mode is mutually exclusive
        with ``values`` mode in v1.
    highlight_fractions : dict[int, float] or None, optional
        Per-node highlight fractions in ``[0, 1]``. When supplied,
        coverage-style colouring is used instead of discrete point-id
        highlighting.
    coverage_mode : {"fraction", "any", "all"}, optional
        Interpretation of ``highlight_fractions``.
    highlight_color : str, optional
        Colour used for highlighted points.
    color_mode : {"solid", "depth", "values"} or None, optional
        Node colouring mode. ``None`` means "infer from values or
        preset".
    values : array-like or Quantity, optional
        Per-point scalar values for value colouring.
    cmap : str or None, optional
        Matplotlib colormap name used in value mode.
    vmin, vmax : float or None, optional
        Explicit lower and upper bounds for the value colormap.
    norm : matplotlib.colors.Normalize or None, optional
        Explicit normalization object for value mode.
    value_label : str or None, optional
        Colorbar label text.
    value_unit_label : str or None, optional
        Explicit unit label appended to the colorbar label.
    show_colorbar : bool, optional
        If ``True`` (default), draw a colorbar in value mode.
    node_color : str or None, optional
        Base node colour override.
    edge_color : str or None, optional
        Edge colour override.
    root_color : str or None, optional
        Root node colour override.
    ax : matplotlib.axes.Axes or None, optional
        Destination axes. When ``None``, a fresh figure and axes are
        created.

    Returns
    -------
    matplotlib.axes.Axes
        The rendered Matplotlib axes.

    Raises
    ------
    TypeError
        If ``node_tree`` is not a :class:`NodeTree`.
    ValueError
        If the node tree is empty, the layout is invalid, the layout
        scale is invalid, or highlight mode is combined with value
        mode.

    Notes
    -----
    Supported layouts in the current wrapper are ``"twopi"``,
    ``"dot"``, ``"neato"``, and ``"kamada_kawai"``. Graphviz-backed
    layouts fall back to ``"kamada_kawai"`` with a warning when
    Graphviz is unavailable.

    Examples
    --------
    Render with the default preset:

    >>> ax = plot_point_topology(node_tree)  # doctest: +SKIP

    Render with explicit value colouring:

    >>> ax = plot_point_topology(  # doctest: +SKIP
    ...     node_tree,
    ...     values=point_values,
    ...     cmap="plasma",
    ...     value_label="Voltage",
    ... )
    """
    if not isinstance(node_tree, NodeTree):
        raise TypeError(f"plot_point_topology(...) expects NodeTree, got {type(node_tree).__name__!s}.")
    if len(node_tree.nodes) == 0:
        raise ValueError("plot_point_topology(...) requires a non-empty NodeTree.")
    return _plot_discrete_topology_graph(
        node_ids=tuple(node.id for node in node_tree.nodes),
        edges=tuple((edge.parent_node_id, edge.child_node_id) for edge in node_tree.edges),
        root_id=node_tree.root_node_id,
        preset=preset,
        layout=layout,
        layout_scale=layout_scale,
        highlight_point_ids=highlight_point_ids,
        highlight_fractions=highlight_fractions,
        coverage_mode=coverage_mode,
        highlight_color=highlight_color,
        color_mode=color_mode,
        values=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        value_label=value_label,
        value_unit_label=value_unit_label,
        show_colorbar=show_colorbar,
        node_color=node_color,
        edge_color=edge_color,
        root_color=root_color,
        ax=ax,
    )


def _plot_discrete_topology_graph(
    *,
    node_ids: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    root_id: int,
    preset: str = "dendrotweaks",
    layout: str | None = None,
    layout_scale: float = 1.0,
    highlight_point_ids=None,
    highlight_fractions: dict[int, float] | None = None,
    coverage_mode: CoverageMode = "fraction",
    highlight_color: str = "#ef4444",
    color_mode: ColorMode | None = None,
    values=None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    value_label: str | None = None,
    value_unit_label: str | None = None,
    show_colorbar: bool = True,
    node_color: str | None = None,
    edge_color: str | None = None,
    root_color: str | None = None,
    ax=None,
) -> Any:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable

    if ax is None:
        _, ax = plt.subplots()

    preset_spec = _resolve_preset(preset)
    resolved_layout = _normalize_layout(layout or preset_spec.layout)
    resolved_layout_scale = _normalize_layout_scale(layout_scale)
    resolved_color_mode = _resolve_color_mode(
        color_mode,
        values=values,
        default=preset_spec.color_mode,
    )
    resolved_cmap = cmap if cmap is not None else preset_spec.cmap
    resolved_node_color = node_color if node_color is not None else preset_spec.node_color
    resolved_edge_color = edge_color if edge_color is not None else preset_spec.edge_color
    resolved_root_color = root_color if root_color is not None else preset_spec.root_color
    resolved_highlight_ids = _normalize_highlight_point_ids(highlight_point_ids)

    if values is not None and resolved_highlight_ids is not None:
        raise ValueError("plot_point_topology(...) does not support highlight_point_ids together with values.")
    if values is not None and highlight_fractions is not None:
        raise ValueError("Discrete topology renderer does not support values together with coverage highlighting.")

    graph = _build_topology_graph(node_ids=node_ids, edges=edges)
    positions = _resolve_layout_positions(
        graph,
        node_ids=node_ids,
        layout=resolved_layout,
        layout_scale=resolved_layout_scale,
        root_node_id=root_id,
    )
    coordinates = np.asarray([positions[node_id] for node_id in node_ids], dtype=float)
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

    if highlight_fractions is not None:
        intensities = _resolve_coverage_intensities(
            node_ids=node_ids,
            highlight_fractions=highlight_fractions,
            coverage_mode=coverage_mode,
        )
        edge_alpha = 0.35 if np.any(intensities > 0.0) else 0.85
    else:
        intensities = None
        edge_alpha = 0.35 if resolved_highlight_ids is not None else 0.85

    if edges:
        segments = [
            np.asarray([positions[parent_id], positions[child_id]], dtype=float)
            for parent_id, child_id in edges
        ]
        ax.add_collection(
            LineCollection(
                segments,
                colors=resolved_edge_color,
                linewidths=0.9,
                alpha=edge_alpha,
                zorder=1,
            )
        )

    node_sizes = np.full(len(node_ids), _default_node_size(len(node_ids)), dtype=float)
    node_sizes[id_to_index[root_id]] *= 1.8

    if intensities is not None:
        base_colors = _base_node_rgba(
            node_ids=node_ids,
            root_id=root_id,
            node_color=resolved_node_color,
            root_color=resolved_root_color,
        )
        blended = _blend_node_rgba(
            base_colors,
            highlight_color=highlight_color,
            intensities=intensities,
        )
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=node_sizes,
            c=blended,
            edgecolors="none",
            zorder=2,
        )
    elif resolved_highlight_ids is not None:
        base_rgba = np.tile(
            np.asarray(_muted_rgba(resolved_node_color), dtype=float),
            (len(node_ids), 1),
        )
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=node_sizes,
            c=base_rgba,
            edgecolors="none",
            zorder=2,
        )

        highlight_indices = [
            id_to_index[point_id]
            for point_id in resolved_highlight_ids
            if point_id in id_to_index
        ]
        if len(highlight_indices) != len(resolved_highlight_ids):
            missing = sorted(point_id for point_id in resolved_highlight_ids if point_id not in id_to_index)
            raise ValueError(f"highlight_point_ids contains unknown point ids {missing!r}.")

        root_index = id_to_index[root_id]
        if root_index not in highlight_indices:
            ax.scatter(
                [coordinates[root_index, 0]],
                [coordinates[root_index, 1]],
                s=[node_sizes[root_index]],
                c=[mcolors.to_rgba(resolved_root_color)],
                edgecolors="none",
                zorder=3,
            )
        if highlight_indices:
            ax.scatter(
                coordinates[highlight_indices, 0],
                coordinates[highlight_indices, 1],
                s=node_sizes[highlight_indices] * 1.15,
                c=[mcolors.to_rgba(highlight_color)],
                edgecolors="none",
                zorder=4,
            )
    else:
        base_colors, value_norm, inferred_unit_label = _resolve_node_rgba(
            node_ids=node_ids,
            root_id=root_id,
            edges=edges,
            color_mode=resolved_color_mode,
            values=values,
            cmap=resolved_cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            node_color=resolved_node_color,
            root_color=resolved_root_color,
        )
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=node_sizes,
            c=base_colors,
            edgecolors="none",
            zorder=2,
        )
        if values is not None and bool(show_colorbar):
            cmap_name = resolved_cmap if resolved_cmap is not None else "viridis"
            colorbar_label = _resolved_colorbar_label(
                value_label=value_label,
                unit_label=value_unit_label if value_unit_label is not None else inferred_unit_label,
            )
            mappable = ScalarMappable(norm=value_norm, cmap=plt.get_cmap(cmap_name))
            mappable.set_array([])
            cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)

    _apply_axes_style(ax, coordinates)
    return ax


def _resolve_preset(name: str) -> _PointTopologyPreset:
    if name not in _PRESETS:
        options = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unsupported point-topology preset {name!r}; expected one of {options}.")
    return _PRESETS[name]


def _normalize_layout(layout: str) -> str:
    normalized = _LAYOUT_ALIASES.get(layout, layout)
    if normalized not in _VALID_LAYOUTS:
        options = ", ".join(sorted(_VALID_LAYOUTS))
        raise ValueError(f"Unsupported point-topology layout {layout!r}; expected one of {options}.")
    return normalized


def _normalize_layout_scale(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"layout_scale must be a finite positive float, got {value!r}.")
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"layout_scale must be a finite positive float, got {value!r}.")
    return scale


def _normalize_highlight_point_ids(highlight_point_ids) -> set[int] | None:
    if highlight_point_ids is None:
        return None
    return {int(point_id) for point_id in np.asarray(list(highlight_point_ids), dtype=np.int32).reshape(-1)}


def _normalize_coverage_mode(coverage_mode: CoverageMode) -> CoverageMode:
    if coverage_mode not in _VALID_COVERAGE_MODES:
        options = ", ".join(sorted(_VALID_COVERAGE_MODES))
        raise ValueError(f"Unsupported coverage_mode {coverage_mode!r}; expected one of {options}.")
    return coverage_mode


def _resolve_coverage_intensities(
    *,
    node_ids: tuple[int, ...],
    highlight_fractions: dict[int, float],
    coverage_mode: CoverageMode,
) -> np.ndarray:
    resolved_mode = _normalize_coverage_mode(coverage_mode)
    fractions = np.asarray(
        [float(np.clip(highlight_fractions.get(node_id, 0.0), 0.0, 1.0)) for node_id in node_ids],
        dtype=float,
    )
    if resolved_mode == "fraction":
        return fractions
    if resolved_mode == "any":
        return np.where(fractions > 1e-9, 1.0, 0.0)
    return np.where(fractions >= 1.0 - 1e-9, 1.0, 0.0)


def _resolve_color_mode(
    color_mode: ColorMode | None,
    *,
    values,
    default: ColorMode,
) -> ColorMode:
    if color_mode is None:
        return "values" if values is not None else default
    if color_mode not in _VALID_COLOR_MODES:
        options = ", ".join(sorted(_VALID_COLOR_MODES))
        raise ValueError(f"Unsupported point-topology color_mode {color_mode!r}; expected one of {options}.")
    if values is not None and color_mode != "values":
        raise ValueError("values=... requires color_mode='values' or color_mode=None.")
    if values is None and color_mode == "values":
        raise ValueError("color_mode='values' requires values=... .")
    return color_mode


def _build_topology_graph(*, node_ids: tuple[int, ...], edges: tuple[tuple[int, int], ...]) -> object:
    nx = _require_networkx()
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    return graph


def _resolve_layout_positions(
    graph,
    *,
    node_ids: tuple[int, ...],
    layout: str,
    layout_scale: float,
    root_node_id: int,
) -> dict[int, np.ndarray]:
    if layout in _GRAPHVIZ_LAYOUTS:
        try:
            positions = _graphviz_layout_positions(
                graph,
                prog=layout,
                root_node_id=root_node_id,
            )
        except Exception as exc:
            warnings.warn(
                f"Graphviz layout {layout!r} is unavailable ({exc}); falling back to 'kamada_kawai'.",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            return _scale_positions(positions, layout_scale)
    return _kamada_kawai_positions(graph, node_ids=node_ids, layout_scale=layout_scale)


def _graphviz_layout_positions(graph, *, prog: str, root_node_id: int) -> dict[int, np.ndarray]:
    from networkx.drawing.nx_agraph import graphviz_layout

    positions = graphviz_layout(graph, prog=prog, root=root_node_id)
    return {
        node_id: np.asarray(position, dtype=float)
        for node_id, position in positions.items()
    }


def _kamada_kawai_positions(
    graph,
    *,
    node_ids: tuple[int, ...],
    layout_scale: float,
) -> dict[int, np.ndarray]:
    nx = _require_networkx()
    positions = nx.kamada_kawai_layout(graph, scale=layout_scale, center=(0.0, 0.0), dim=2)
    return {
        node_id: np.asarray(positions[node_id], dtype=float)
        for node_id in node_ids
    }


def _scale_positions(
    positions: dict[int, np.ndarray],
    scale: float,
) -> dict[int, np.ndarray]:
    if abs(scale - 1.0) <= 1e-12:
        return {
            node_id: np.asarray(position, dtype=float)
            for node_id, position in positions.items()
        }
    return {
        node_id: np.asarray(position, dtype=float) * scale
        for node_id, position in positions.items()
    }


def _resolve_node_rgba(
    *,
    node_ids: tuple[int, ...],
    root_id: int,
    edges: tuple[tuple[int, int], ...],
    color_mode: ColorMode,
    values,
    cmap: str | None,
    vmin: float | None,
    vmax: float | None,
    norm,
    node_color: str,
    root_color: str,
) -> tuple[np.ndarray, object | None, str | None]:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if color_mode == "solid":
        return _base_node_rgba(
            node_ids=node_ids,
            root_id=root_id,
            node_color=node_color,
            root_color=root_color,
        ), None, None

    if color_mode == "depth":
        scalar_values = _graph_depths(node_ids=node_ids, edges=edges)
        inferred_unit_label = None
    else:
        scalar_values, inferred_unit_label = _normalize_values_array(values, n_points=len(node_ids))

    cmap_name = cmap if cmap is not None else "viridis"
    colormap = plt.get_cmap(cmap_name)
    value_norm = _build_value_norm(
        scalar_values,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    fill_value = 0.0 if getattr(value_norm, "vmin", None) is None else float(value_norm.vmin)
    mapped = np.asarray(colormap(value_norm(np.nan_to_num(scalar_values, nan=fill_value))), dtype=float)
    mapped[~np.isfinite(scalar_values)] = np.asarray(_muted_rgba(node_color), dtype=float)
    root_indices = np.flatnonzero(np.asarray(node_ids, dtype=np.int64) == int(root_id))
    if root_indices.size:
        mapped[root_indices[0]] = np.asarray(mcolors.to_rgba(root_color), dtype=float)
    return mapped, value_norm, inferred_unit_label


def _normalize_values_array(values, *, n_points: int) -> tuple[np.ndarray, str | None]:
    if hasattr(values, "to_decimal") and hasattr(values, "unit"):
        unit = values.unit
        raw = np.asarray(values.to_decimal(unit), dtype=float)
        unit_label = str(unit)
    else:
        raw = np.asarray(values, dtype=float)
        unit_label = None
    if raw.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {raw.shape!r}.")
    array = raw.reshape(-1)
    if array.shape != (n_points,):
        raise ValueError(
            f"values must have shape ({n_points},), got {array.shape!r}."
        )
    return array, unit_label


def _build_value_norm(values: np.ndarray, *, vmin: float | None, vmax: float | None, norm):
    import matplotlib.colors as mcolors

    if norm is not None:
        return norm
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        lo = 0.0 if vmin is None else float(vmin)
        hi = 1.0 if vmax is None else float(vmax)
    else:
        lo = float(np.min(finite)) if vmin is None else float(vmin)
        hi = float(np.max(finite)) if vmax is None else float(vmax)
    if hi - lo <= 1e-12:
        hi = lo + 1.0
    return mcolors.Normalize(vmin=lo, vmax=hi)


def _normalize_scalar_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros_like(values, dtype=float)
    finite_values = values[finite_mask]
    lo = float(np.min(finite_values))
    hi = float(np.max(finite_values))
    if hi - lo <= 1e-12:
        out = np.zeros_like(values, dtype=float)
        out[~finite_mask] = 0.0
        return out
    out = np.zeros_like(values, dtype=float)
    out[finite_mask] = (finite_values - lo) / (hi - lo)
    return out


def _graph_depths(*, node_ids: tuple[int, ...], edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not node_ids:
        return np.zeros((0,), dtype=float)
    adjacency: dict[int, list[int]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for parent_id, child_id in edges:
        adjacency[parent_id].append(child_id)
        indegree[child_id] += 1
    roots = [node_id for node_id in node_ids if indegree[node_id] == 0]
    root_id = roots[0]
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    depths = np.full(len(node_ids), -1.0, dtype=float)
    depths[id_to_index[root_id]] = 0.0
    queue = [root_id]
    while queue:
        node_id = queue.pop(0)
        parent_depth = depths[id_to_index[node_id]]
        for child_id in adjacency[node_id]:
            depths[id_to_index[child_id]] = parent_depth + 1.0
            queue.append(child_id)
    return depths


def _default_node_size(n_points: int) -> float:
    return float(np.clip(420.0 / max(np.sqrt(n_points), 1.0), 12.0, 80.0))


def _muted_rgba(color: str) -> tuple[float, float, float, float]:
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, 0.22)


def _base_node_rgba(
    *,
    node_ids: tuple[int, ...],
    root_id: int,
    node_color: str,
    root_color: str,
) -> np.ndarray:
    import matplotlib.colors as mcolors

    base = np.tile(np.asarray(mcolors.to_rgba(node_color), dtype=float), (len(node_ids), 1))
    root_indices = np.flatnonzero(np.asarray(node_ids, dtype=np.int64) == int(root_id))
    if root_indices.size:
        base[root_indices[0]] = np.asarray(mcolors.to_rgba(root_color), dtype=float)
    return base


def _blend_node_rgba(
    base_rgba: np.ndarray,
    *,
    highlight_color: str,
    intensities: np.ndarray,
) -> np.ndarray:
    import matplotlib.colors as mcolors

    out = np.asarray(base_rgba, dtype=float).copy()
    target = np.asarray(mcolors.to_rgba(highlight_color), dtype=float)
    weights = np.clip(np.asarray(intensities, dtype=float).reshape(-1, 1), 0.0, 1.0)
    out = (1.0 - weights) * out + weights * target[None, :]
    out[:, 3] = 1.0
    return out


def _resolved_colorbar_label(*, value_label: str | None, unit_label: str | None) -> str | None:
    if value_label and unit_label:
        return f"{value_label} [{unit_label}]"
    if value_label:
        return value_label
    if unit_label:
        return f"[{unit_label}]"
    return None


def _apply_axes_style(ax, coordinates: np.ndarray) -> None:
    if coordinates.shape[0] == 1:
        center_x, center_y = coordinates[0]
        ax.set_xlim(center_x - 1.0, center_x + 1.0)
        ax.set_ylim(center_y - 1.0, center_y + 1.0)
    else:
        x_values = coordinates[:, 0]
        y_values = coordinates[:, 1]
        x_span = float(np.max(x_values) - np.min(x_values))
        y_span = float(np.max(y_values) - np.min(y_values))
        pad_x = max(0.05 * x_span, 0.5)
        pad_y = max(0.05 * y_span, 0.5)
        ax.set_xlim(float(np.min(x_values) - pad_x), float(np.max(x_values) + pad_x))
        ax.set_ylim(float(np.min(y_values) - pad_y), float(np.max(y_values) + pad_y))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)


def _require_networkx():
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised via dependency failures
        raise ImportError(
            "plot_point_topology(...) requires networkx. Install braincell[vis] or add networkx manually."
        ) from exc
    return nx
