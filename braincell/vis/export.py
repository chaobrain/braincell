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

"""Unified figure export for the :mod:`braincell.vis` backends.

``save_figure`` takes the return value of :func:`plot2d` or :func:`plot3d`
(a matplotlib ``Axes``/``Figure``, a PyVista ``Plotter``, or a
Plotly ``Figure``) and writes it to disk with a single call. It is a
thin type-dispatching wrapper — its job is to hide which backend
produced the object so downstream scripts don't branch on
``isinstance``.

The file format is inferred from the path suffix; callers can override
with ``format=...`` when matplotlib or Plotly need a hint. ``dpi`` and
``transparent`` are forwarded where meaningful.

Examples
--------

.. code-block:: python

    >>> import braincell.vis as vis
    >>> ax = vis.plot2d(tree, layout="stem")            # doctest: +SKIP
    >>> vis.save_figure(ax, "soma.pdf", dpi=300)        # doctest: +SKIP
    >>> plotter = vis.plot3d(tree)                      # doctest: +SKIP
    >>> vis.save_figure(plotter, "tree.png")            # doctest: +SKIP
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


_PNG_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
_VECTOR_SUFFIXES = {".pdf", ".svg", ".eps", ".ps"}
_HTML_SUFFIXES = {".html", ".htm"}


def save_figure(
    figure: Any,
    path: str | os.PathLike,
    *,
    dpi: int | None = None,
    transparent: bool = False,
    format: str | None = None,
) -> Path:
    """Save a vis backend return value to disk.

    Parameters
    ----------
    figure : object
        Return value from :func:`plot2d`, :func:`plot3d`, or a raw
        matplotlib / PyVista / Plotly handle. The specific types
        recognised are:

        * ``matplotlib.axes.Axes`` — the owning figure is saved.
        * ``matplotlib.figure.Figure`` — saved directly.
        * ``pyvista.Plotter`` — screenshot for raster formats, VTK /
          HTML export for vector formats.
        * ``plotly.graph_objects.Figure`` — written via
          ``figure.write_image`` (raster) or ``figure.write_html``.
    path : str or os.PathLike
        Destination. Missing parent directories are **not** created
        implicitly — the caller owns filesystem layout.
    dpi : int or None
        DPI hint forwarded to matplotlib and PyVista. Plotly ignores
        it (use width/height on the figure instead).
    transparent : bool
        Whether to request a transparent background. Matplotlib
        honours this directly. PyVista uses its off-screen transparent
        path. Plotly is ignored.
    format : str or None
        Explicit format override. When ``None`` the format is inferred
        from the path suffix.

    Returns
    -------
    path : pathlib.Path
        The path that was written, for convenient chaining.

    Raises
    ------
    TypeError
        If the figure type is not recognised.
    ValueError
        If a vector path is requested for a backend that cannot serve
        it (e.g. PyVista for ``.pdf``).

    Notes
    -----
    The dispatch is intentionally conservative — it only recognises
    types it can handle natively. Pass ``figure.savefig(...)`` (for
    matplotlib) or the backend-specific API directly when you need
    finer control.
    """
    out_path = Path(path)
    suffix = out_path.suffix.lower()
    inferred_format = format or (suffix.lstrip(".") if suffix else None)

    handler = _dispatch(figure)
    handler(figure, out_path, suffix=suffix, format=inferred_format, dpi=dpi, transparent=transparent)
    return out_path


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatch(figure: Any):
    # Matplotlib Axes / Figure
    if _is_matplotlib_artist(figure):
        return _save_matplotlib
    if _is_pyvista_plotter(figure):
        return _save_pyvista
    if _is_plotly_figure(figure):
        return _save_plotly
    raise TypeError(
        f"save_figure(...) does not know how to save {type(figure).__name__!s}. "
        "Pass a matplotlib Axes/Figure, pyvista Plotter, or plotly Figure."
    )


def _is_matplotlib_artist(obj: Any) -> bool:
    if "matplotlib" not in sys.modules:
        return False
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    return isinstance(obj, (Axes, Figure))


def _is_pyvista_plotter(obj: Any) -> bool:
    if importlib.util.find_spec("pyvista") is None:
        return False
    if "pyvista" not in sys.modules:
        # Avoid triggering a heavy pyvista import just for instance check.
        return obj.__class__.__module__.startswith("pyvista")
    import pyvista as pv

    return isinstance(obj, pv.Plotter)


def _is_plotly_figure(obj: Any) -> bool:
    if importlib.util.find_spec("plotly") is None:
        return False
    module = type(obj).__module__
    if not module.startswith("plotly"):
        return False
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    return isinstance(obj, go.Figure)


# ---------------------------------------------------------------------------
# Backend-specific savers
# ---------------------------------------------------------------------------


def _save_matplotlib(
    figure: Any,
    path: Path,
    *,
    suffix: str,
    format: str | None,
    dpi: int | None,
    transparent: bool,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if isinstance(figure, Axes):
        fig: Figure = figure.figure
    else:
        fig = figure

    kwargs: dict[str, Any] = {"transparent": bool(transparent)}
    if dpi is not None:
        kwargs["dpi"] = int(dpi)
    if format is not None:
        kwargs["format"] = format
    fig.savefig(path, **kwargs)


def _save_pyvista(
    figure: Any,
    path: Path,
    *,
    suffix: str,
    format: str | None,
    dpi: int | None,
    transparent: bool,
) -> None:
    # PyVista handles raster screenshots and HTML exports natively.
    if suffix in _HTML_SUFFIXES and hasattr(figure, "export_html"):
        figure.export_html(str(path))
        return
    if suffix in _VECTOR_SUFFIXES:
        # Try VTK's vector exporter; if unsupported, fall back to raster.
        if hasattr(figure, "save_graphic"):
            figure.save_graphic(str(path))
            return
        raise ValueError(
            f"PyVista cannot save vector format {suffix!r}. Use a raster suffix "
            "(.png, .jpg, .tif, ...) or install vtk[extra] for save_graphic support."
        )
    # Default: raster screenshot.
    if hasattr(figure, "screenshot"):
        screenshot_kwargs: dict[str, Any] = {"transparent_background": bool(transparent)}
        if dpi is not None:
            screenshot_kwargs["scale"] = max(int(dpi // 100), 1)
        figure.screenshot(str(path), **screenshot_kwargs)
        return
    raise TypeError("PyVista plotter does not expose a screenshot/save_graphic method.")


def _save_plotly(
    figure: Any,
    path: Path,
    *,
    suffix: str,
    format: str | None,
    dpi: int | None,
    transparent: bool,
) -> None:
    if suffix in _HTML_SUFFIXES and hasattr(figure, "write_html"):
        figure.write_html(str(path))
        return
    if hasattr(figure, "write_image"):
        figure.write_image(str(path))
        return
    raise TypeError("Plotly figure does not expose a write_image/write_html method.")
