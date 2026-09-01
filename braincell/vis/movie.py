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

"""Time-varying colour-by-values animation (``plot_movie``).

``plot_movie`` is the Phase 3 entry point for rendering voltage or
calcium traces on a morphology over time. It builds the scene once and
re-writes only the value arrays per frame, which avoids rebuilding
layouts or re-uploading PyVista meshes for every timestep.

Two dispatch modes are supported:

- ``dimensionality='2d'`` — uses :mod:`matplotlib.animation.FuncAnimation`
  and mutates the underlying :class:`~matplotlib.collections.LineCollection`
  / :class:`~matplotlib.collections.PolyCollection` scalars in place.
- ``dimensionality='3d'`` — uses ``pyvista.Plotter.open_movie`` and
  updates :class:`~pyvista.PolyData` point data per frame.

Both paths accept the same ``values_over_time`` shape: ``(T, N)`` where
``N`` is one of the shapes supported by :class:`~braincell.vis.scene.ValueSpec`
(per-branch, per-segment, or per-centerline-point). The colour scale is
fixed across all frames; callers should pass ``vmin`` / ``vmax``
explicitly if they want tight bounds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from braincell.morph.morphology import Morphology
from ._values import ValueLayout, _strip_quantity, resolve_value_limits
from .layout import LayoutConfig
from .scene import ValueSpec as _ValueSpec


@dataclass(frozen=True)
class MovieResult:
    """Artifacts returned from :func:`plot_movie`.

    Attributes
    ----------
    animation : object or None
        The matplotlib ``FuncAnimation`` (2D) or PyVista ``Plotter``
        (3D), retained so notebooks can display it or tests can
        inspect it.
    frames : int
        Number of rendered frames.
    output_path : Path or None
        On-disk location if the caller passed ``out=`` (e.g. an MP4 or
        GIF path), otherwise ``None``.
    """

    animation: Any
    frames: int
    output_path: Path | None


def plot_movie(
    morpho: Morphology,
    values_over_time,
    *,
    dt=None,
    dimensionality: str = "2d",
    out: Path | str | None = None,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    value_label: str | None = None,
    layout: str | None = None,
    shape: str | None = None,
    layout_config: LayoutConfig | None = None,
    mode: str | None = None,
    ax=None,
    figsize: tuple[float, float] | None = None,
) -> MovieResult:
    """Render a time-varying colour-by-values animation over a morphology.

    Parameters
    ----------
    morpho : Morphology
        Morphology the values are defined against.
    values_over_time : ArrayLike
        A ``(T, N)`` array where ``T`` is the number of frames and
        ``N`` matches one of the supported shapes for
        :class:`ValueSpec` (``n_branches``, total segment count, or
        total centerline-point count). Units attached via
        :mod:`brainunit` are stripped and used as the default
        colour-bar label.
    dt : Quantity or None
        Optional timestep between frames. Only used to display the
        current time in the figure title.
    dimensionality : {'2d', '3d'}
        Target backend. 2D uses matplotlib ``FuncAnimation``; 3D uses
        PyVista's movie writer.
    out : str or Path or None
        If set, the animation is written to this path (``.mp4`` / ``.gif``
        for 2D, ``.mp4`` for 3D). The directory must already exist.
    fps : int
        Frames per second for the output file.
    cmap, vmin, vmax, value_label
        Forwarded to :class:`ValueSpec` and applied uniformly across
        every frame.
    layout, shape, layout_config, mode
        Layout / rendering knobs forwarded to the scene builder. Match
        the semantics of :func:`plot2d` / :func:`plot3d`.
    ax : matplotlib Axes or None
        Optional Axes to render into. Ignored in 3D mode.
    figsize : tuple or None
        Matplotlib figure size. Ignored when *ax* is supplied.

    Returns
    -------
    MovieResult
        The underlying animation / plotter, frame count, and output
        path (if any).

    Raises
    ------
    ValueError
        If *values_over_time* is not 2-D or if the frame count is zero.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"plot_movie(...) expects Morphology, got {type(morpho).__name__!s}.")

    arr, _ = _strip_quantity(values_over_time)
    if arr.ndim != 2:
        raise ValueError(f"plot_movie(...) expects a 2-D (T, N) values array, got shape {arr.shape!r}.")
    n_frames = arr.shape[0]
    if n_frames == 0:
        raise ValueError("plot_movie(...) requires at least one frame.")

    # Resolve a shared vmin/vmax across every frame so the colour scale
    # is fixed for the whole animation.
    vmin, vmax = resolve_value_limits(_ValueSpec(values=arr, vmin=vmin, vmax=vmax), (arr,))

    if dimensionality == "2d":
        return _plot_movie_2d(
            morpho,
            arr,
            dt=dt,
            out=Path(out) if out is not None else None,
            fps=fps,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            value_label=value_label,
            layout=layout,
            shape=shape,
            layout_config=layout_config,
            ax=ax,
            figsize=figsize,
        )
    if dimensionality == "3d":
        return _plot_movie_3d(
            morpho,
            arr,
            dt=dt,
            out=Path(out) if out is not None else None,
            fps=fps,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            value_label=value_label,
            mode=mode,
        )
    raise ValueError(f"plot_movie(...) dimensionality must be '2d' or '3d', got {dimensionality!r}.")


# ---------------------------------------------------------------------------
# 2D movie
# ---------------------------------------------------------------------------


def _plot_movie_2d(
    morpho: Morphology,
    values: np.ndarray,
    *,
    dt,
    out: Path | None,
    fps: int,
    cmap: str,
    vmin: float,
    vmax: float,
    value_label: str | None,
    layout: str | None,
    shape: str | None,
    layout_config: LayoutConfig | None,
    ax,
    figsize,
) -> MovieResult:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection, PolyCollection

    from .plot2d import plot2d

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    initial_values = values[0]
    plot2d(
        morpho,
        values=initial_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        value_label=value_label,
        layout=layout,
        shape=shape,
        layout_config=layout_config,
        ax=ax,
    )

    # Grab the value-bearing collections that plot2d just added so we can
    # mutate their arrays in place for every subsequent frame.
    value_collections = [
        collection
        for collection in ax.collections
        if isinstance(collection, (LineCollection, PolyCollection)) and collection.get_array() is not None
    ]

    # Value collections come back in the order the scene builder
    # produced them: one per branch, in branch-index order. Neither that
    # order nor the per-branch segment counts change between frames, so
    # the whole structural derivation happens once, here.
    value_layout = ValueLayout.from_morphology(morpho)
    scene_order_branch_indices = value_layout.branch_indices

    title_obj = ax.set_title("") if dt is None else ax.set_title(_format_time(0))

    def _update(frame_index: int):
        # Resolved lazily per frame rather than materialising all T
        # frames upfront: FuncAnimation only ever needs one at a time,
        # and precomputing held T x n_branches arrays alive for the
        # lifetime of the animation. Only the slicing is per frame —
        # the branch structure comes from the hoisted ``value_layout``.
        per_branch = value_layout.expand(values[frame_index])
        for collection, branch_index in zip(value_collections, scene_order_branch_indices):
            collection.set_array(per_branch[branch_index].segment_values)
        if dt is not None:
            title_obj.set_text(_format_time(frame_index))
        return (*value_collections, title_obj)

    animation = FuncAnimation(
        fig,
        _update,
        frames=values.shape[0],
        interval=1000.0 / max(int(fps), 1),
        blit=False,
        repeat=False,
    )

    output_path = None
    if out is not None:
        _save_animation_2d(animation, out, fps=fps)
        output_path = out

    return MovieResult(animation=animation, frames=values.shape[0], output_path=output_path)


def _format_time(frame_index: int) -> str:
    # NOTE: this renders the frame index, not the elapsed time — the
    # ``dt`` the caller passes is not folded in. Kept as-is because
    # changing it would change every rendered title.
    return f"t = {frame_index} × dt"


def _save_animation_2d(animation, out: Path, *, fps: int) -> None:
    """Write the matplotlib animation to disk, selecting the writer by suffix."""
    writer = "pillow" if out.suffix.lower() == ".gif" else "ffmpeg"
    animation.save(str(out), writer=writer, fps=fps)


# ---------------------------------------------------------------------------
# 3D movie
# ---------------------------------------------------------------------------


def _plot_movie_3d(
    morpho: Morphology,
    values: np.ndarray,
    *,
    dt,
    out: Path | None,
    fps: int,
    cmap: str,
    vmin: float,
    vmax: float,
    value_label: str | None,
    mode: str | None,
) -> MovieResult:
    """3D ``plot_movie`` dispatch.

    The 3D path renders each branch as a fat-line PolyData and mutates
    ``polydata.point_data['values']`` per frame. Tubes are intentionally
    skipped here: after :meth:`pyvista.PolyData.tube` the scalar is
    expanded to ``n_sides * n_polydata_points`` samples, which would
    require a manual index remap every frame. Fat lines avoid the issue
    and stay fast enough for the typical time-series visualization
    use-case.
    """
    import pyvista as pv

    from .scene3d import build_render_scene_3d

    scene = build_render_scene_3d(morpho, mode=mode or "skeleton")
    if not scene.branches:
        raise ValueError("plot_movie(dimensionality='3d') requires a non-empty morphology.")

    plotter = pv.Plotter(off_screen=out is not None)
    meshes: list[tuple[Any, tuple[int, ...]]] = []
    # Branch order and per-branch segment counts are the same for every
    # frame; only the scalars change. Derive them once.
    value_layout = ValueLayout.from_morphology(morpho)
    initial_values = value_layout.expand(values[0])
    first = True
    for batch in scene.batches:
        poly = pv.PolyData()
        poly.points = batch.points_um
        poly.lines = batch.lines
        point_values = np.concatenate([initial_values[branch_idx].point_values for branch_idx in batch.branch_indices])
        poly.point_data["values"] = point_values
        plotter.add_mesh(
            poly,
            scalars="values",
            cmap=cmap,
            clim=(vmin, vmax),
            line_width=6.0,
            show_scalar_bar=first and (value_label is not None),
            scalar_bar_args={"title": value_label} if first and value_label else None,
        )
        first = False
        meshes.append((poly, batch.branch_indices))

    output_path = None
    if out is not None:
        plotter.open_movie(str(out), framerate=fps)
        for frame_index in range(values.shape[0]):
            frame = value_layout.expand(values[frame_index])
            for poly, branch_indices in meshes:
                poly.point_data["values"] = np.concatenate(
                    [frame[branch_idx].point_values for branch_idx in branch_indices]
                )
            plotter.write_frame()
        plotter.close()
        output_path = out

    return MovieResult(animation=plotter, frames=values.shape[0], output_path=output_path)
