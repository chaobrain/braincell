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

"""Morphometry / topology plots.

Phase 3 ships four commonly-requested morphology summary views:

* :func:`plot_dendrogram` — horizontal-tree schematic with branch
  length on the x-axis and leaves stacked on the y-axis.
* :func:`plot_topology` — graph-layout schematic that ignores branch
  lengths entirely and surfaces branching topology.
* :func:`plot_sholl` — concentric Sholl-intersection profile around a
  chosen centre (defaults to the soma origin).
* :func:`plot_branch_order_histogram` — distribution of branch counts
  per branch order.

All four accept an optional ``ax=`` matplotlib Axes so callers can
assemble multi-panel figures. They return the Axes they drew into so
tests can assert on line / bar counts without caring about figure
bookkeeping.
"""

from dataclasses import dataclass
from typing import Any

import brainunit as u
import numpy as np

from braincell.morph import MorphoBranch
from braincell.morph.morphology import Morphology
from .config import color_for_branch_type


@dataclass(frozen=True)
class ShollProfile:
    """Result of a Sholl analysis.

    ``radii`` and ``intersections`` are parallel arrays.
    ``intersections[i]`` counts how many branch segments cross the
    sphere / circle of radius ``radii[i]`` around the analysis centre.
    """

    radii_um: np.ndarray
    intersections: np.ndarray


# ---------------------------------------------------------------------------
# Dendrogram
# ---------------------------------------------------------------------------

def plot_dendrogram(
    morpho: Morphology,
    *,
    ax=None,
    color_by_type: bool = True,
    linewidth: float = 1.5,
) -> Any:
    """Render a left-to-right dendrogram of the morphology.

    Each branch is drawn as a horizontal stroke whose length equals
    the branch's own total length. Children are stacked vertically
    and connected to their parent by a short vertical segment at the
    attachment x-coordinate.

    Parameters
    ----------
    morpho : Morphology
        Morphology to render.
    ax : matplotlib Axes or None
        Destination axes. When ``None``, a fresh figure is created.
    color_by_type : bool
        If ``True`` (default), each branch is coloured using the
        shared :func:`color_for_branch_type` palette. Otherwise all
        strokes are black.
    linewidth : float
        Stroke width for the horizontal branch lines.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot_dendrogram(...) expects Morphology, got {type(morpho).__name__!s}.")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    y_positions = _assign_dendrogram_y(morpho.root)

    def _walk(node: MorphoBranch, x_start: float) -> None:
        length_um = float(node.length.to_decimal(u.um))
        y = y_positions[node.index]
        color = _rgb_to_float(color_for_branch_type(node.type)) if color_by_type else (0.0, 0.0, 0.0)
        ax.plot(
            [x_start, x_start + length_um],
            [y, y],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
        )
        # Vertical connector between first/last child and the parent.
        children = node.children
        if children:
            child_ys = [y_positions[child.index] for child in children]
            ax.plot(
                [x_start + length_um, x_start + length_um],
                [min(child_ys), max(child_ys)],
                color=color,
                linewidth=linewidth * 0.75,
            )
            for child in children:
                _walk(child, x_start + length_um)

    _walk(morpho.root, 0.0)
    ax.set_xlabel("path length from root [µm]")
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return ax


def _assign_dendrogram_y(root: MorphoBranch) -> dict[int, float]:
    """Return per-branch y positions in a compact leaf-first layout."""
    positions: dict[int, float] = {}
    cursor: list[float] = [0.0]

    def _visit(node: MorphoBranch) -> float:
        if node.n_children == 0:
            y = cursor[0]
            cursor[0] += 1.0
            positions[node.index] = y
            return y
        child_ys = [_visit(child) for child in node.children]
        y = 0.5 * (min(child_ys) + max(child_ys))
        positions[node.index] = y
        return y

    _visit(root)
    return positions


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def plot_topology(
    morpho: Morphology,
    *,
    ax=None,
    color_by_type: bool = True,
) -> Any:
    """Render a topology-only schematic (length-ignored) of the morphology.

    Unlike :func:`plot_dendrogram`, this view ignores physical length
    entirely: depth is used as the x-coordinate, so branching order
    is the only thing the reader cares about. Useful for comparing
    the shapes of morphologies whose scales differ by orders of
    magnitude.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot_topology(...) expects Morphology, got {type(morpho).__name__!s}.")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    y_positions = _assign_dendrogram_y(morpho.root)
    depths = _branch_depths(morpho.root)

    def _walk(node: MorphoBranch) -> None:
        y = y_positions[node.index]
        x = float(depths[node.index])
        color = _rgb_to_float(color_for_branch_type(node.type)) if color_by_type else (0.0, 0.0, 0.0)
        ax.plot([x, x + 1.0], [y, y], color=color, linewidth=1.5, solid_capstyle="round")
        if node.n_children:
            child_ys = [y_positions[child.index] for child in node.children]
            ax.plot(
                [x + 1.0, x + 1.0],
                [min(child_ys), max(child_ys)],
                color=color,
                linewidth=1.0,
            )
            for child in node.children:
                _walk(child)

    _walk(morpho.root)
    ax.set_xlabel("branch order")
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return ax


def _branch_depths(root: MorphoBranch) -> dict[int, int]:
    depths: dict[int, int] = {}

    def _visit(node: MorphoBranch, depth: int) -> None:
        depths[node.index] = depth
        for child in node.children:
            _visit(child, depth + 1)

    _visit(root, 0)
    return depths


# ---------------------------------------------------------------------------
# Sholl analysis
# ---------------------------------------------------------------------------

def compute_sholl_profile(
    morpho: Morphology,
    *,
    step_um: float = 10.0,
    max_radius_um: float | None = None,
    center_um: np.ndarray | None = None,
) -> ShollProfile:
    """Compute a Sholl-intersection profile around a chosen centre.

    The algorithm samples concentric spheres / circles at
    ``step_um`` increments and counts how many branch segments have
    one endpoint inside and one endpoint outside the sphere. For
    length-only morphologies (without explicit points) the
    computation falls back to path distance from the root along the
    tree, which is a reasonable proxy for dendritic coverage.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"compute_sholl_profile(...) expects Morphology, got {type(morpho).__name__!s}.")
    if step_um <= 0.0:
        raise ValueError(f"step_um must be > 0, got {step_um!r}.")

    if morpho.has_full_point_geometry:
        distances_start, distances_end = _euclidean_segment_distances(morpho, center_um)
    else:
        distances_start, distances_end = _path_segment_distances(morpho)

    if distances_start.size == 0:
        return ShollProfile(radii_um=np.array([], dtype=float), intersections=np.array([], dtype=int))

    upper = max_radius_um if max_radius_um is not None else float(max(distances_start.max(), distances_end.max()))
    if upper <= 0:
        return ShollProfile(
            radii_um=np.array([step_um], dtype=float),
            intersections=np.array([0], dtype=int),
        )
    radii = np.arange(step_um, upper + step_um, step_um)
    counts = np.zeros(radii.shape, dtype=int)
    for i, r in enumerate(radii):
        inside = distances_start < r
        outside = distances_end >= r
        counts[i] = int(np.count_nonzero(inside & outside))
    return ShollProfile(radii_um=radii, intersections=counts)


def plot_sholl(
    morpho: Morphology,
    *,
    ax=None,
    step_um: float = 10.0,
    max_radius_um: float | None = None,
    color: Any = "tab:blue",
) -> Any:
    """Plot a Sholl-intersection curve.

    Parameters are the same as :func:`compute_sholl_profile`; *ax*
    and *color* control the matplotlib output.
    """
    import matplotlib.pyplot as plt

    profile = compute_sholl_profile(
        morpho,
        step_um=step_um,
        max_radius_um=max_radius_um,
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(profile.radii_um, profile.intersections, color=color, linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("radius [µm]")
    ax.set_ylabel("intersections")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def _euclidean_segment_distances(
    morpho: Morphology,
    center_um: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    origin = _resolve_center(morpho, center_um)
    starts: list[float] = []
    ends: list[float] = []
    for branch_view in morpho.branches:
        branch = branch_view.branch
        if branch.points_proximal is None or branch.points_distal is None:
            continue
        proximal = np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float)
        distal = np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)
        # Build one contiguous (n_points, 3) array.
        pts = np.vstack([proximal[:1], distal])
        dist = np.linalg.norm(pts - origin[None, :], axis=1)
        starts.extend(dist[:-1].tolist())
        ends.extend(dist[1:].tolist())
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def _path_segment_distances(morpho: Morphology) -> tuple[np.ndarray, np.ndarray]:
    cumulative: dict[int, float] = {morpho.root.index: 0.0}
    starts: list[float] = []
    ends: list[float] = []

    def _visit(node: MorphoBranch, parent_end_um: float) -> None:
        lengths = np.asarray(node.lengths.to_decimal(u.um), dtype=float)
        running = parent_end_um
        for seg_len in lengths:
            starts.append(running)
            running = running + float(seg_len)
            ends.append(running)
        cumulative[node.index] = running
        for child in node.children:
            _visit(child, cumulative[node.index])

    _visit(morpho.root, 0.0)
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def _resolve_center(morpho: Morphology, center_um: np.ndarray | None) -> np.ndarray:
    if center_um is not None:
        return np.asarray(center_um, dtype=float)
    # Default to the proximal point of the root branch.
    root = morpho.root.branch
    if root.points_proximal is not None:
        return np.asarray(root.points_proximal[0].to_decimal(u.um), dtype=float)
    return np.zeros(3, dtype=float)


# ---------------------------------------------------------------------------
# Branch-order histogram
# ---------------------------------------------------------------------------

def plot_branch_order_histogram(
    morpho: Morphology,
    *,
    ax=None,
    color: Any = "tab:gray",
) -> Any:
    """Bar chart of the number of branches per branch order.

    ``branch_order`` is the depth of the branch in the tree, with the
    root at order 0.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot_branch_order_histogram(...) expects Morphology, got {type(morpho).__name__!s}.")
    import matplotlib.pyplot as plt

    depths = _branch_depths(morpho.root)
    if not depths:
        raise ValueError("plot_branch_order_histogram(...) requires a non-empty morphology.")
    max_order = max(depths.values())
    counts = np.zeros(max_order + 1, dtype=int)
    for depth in depths.values():
        counts[depth] += 1

    if ax is None:
        _, ax = plt.subplots()
    ax.bar(np.arange(max_order + 1), counts, color=color, edgecolor="black")
    ax.set_xlabel("branch order")
    ax.set_ylabel("count")
    ax.set_xticks(np.arange(max_order + 1))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def _rgb_to_float(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(float(channel) / 255.0 for channel in rgb)  # type: ignore[return-value]
