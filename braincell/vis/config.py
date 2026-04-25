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

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Iterator

DEFAULT_BRANCH_TYPE_COLORS = {
    "soma": (47, 49, 54),
    "axon": (108, 142, 173),
    "basal_dendrite": (190, 134, 111),
    "apical_dendrite": (214, 173, 98),
    "dendrite": (158, 170, 132),
    "custom": (154, 164, 175),
}
DEFAULT_2D_EDGE_DARKEN_FACTOR = 0.72
DEFAULT_2D_FRUSTUM_EDGE_LINEWIDTH = 0.9
DEFAULT_HIGHLIGHT_COLOR = (255, 215, 0)  # gold — used for region overlays
DEFAULT_MARKER_COLOR = (30, 144, 255)  # dodger blue — used for locset overlays

SUPPORTED_2D_LAYOUTS = frozenset({"projected", "fan", "stem", "balloon", "radial_360"})
SUPPORTED_2D_SHAPES = frozenset({"line", "frustum"})
SUPPORTED_3D_MODES = frozenset({"geometry", "skeleton"})


@dataclass
class VisDefaults:
    layout_2d_default: str = "fan"
    shape_2d_default: str = "frustum"
    mode_3d_default: str = "geometry"
    branch_type_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_BRANCH_TYPE_COLORS)
    )
    branch_type_edge_colors_2d: dict[str, tuple[int, int, int]] | None = None
    alpha_2d: float = 0.8
    alpha_2d_poly: float | None = None
    alpha_2d_line: float | None = None
    frustum_edge_linewidth_2d: float = DEFAULT_2D_FRUSTUM_EDGE_LINEWIDTH
    alpha_3d_tube: float = 1.0
    highlight_color: tuple[int, int, int] = DEFAULT_HIGHLIGHT_COLOR
    highlight_alpha: float = 0.9
    marker_color: tuple[int, int, int] = DEFAULT_MARKER_COLOR
    marker_size_2d: float = 36.0
    marker_radius_3d_um: float = 1.5


_VIS_DEFAULTS = VisDefaults()


def get_defaults() -> VisDefaults:
    return _copy_defaults(_VIS_DEFAULTS)


def reset_defaults() -> VisDefaults:
    global _VIS_DEFAULTS
    _VIS_DEFAULTS = VisDefaults()
    return get_defaults()


def configure(
    *,
    layout_2d_default: str | None = None,
    shape_2d_default: str | None = None,
    mode_3d_default: str | None = None,
    branch_type_colors: dict[str, object] | None = None,
    branch_type_edge_colors_2d: dict[str, object] | None = None,
    replace_branch_type_colors: bool = False,
    replace_branch_type_edge_colors_2d: bool = False,
    alpha_2d: float | None = None,
    alpha_2d_poly: float | None = None,
    alpha_2d_line: float | None = None,
    frustum_edge_linewidth_2d: float | None = None,
    alpha_3d_tube: float | None = None,
    highlight_color: object | None = None,
    highlight_alpha: float | None = None,
    marker_color: object | None = None,
    marker_size_2d: float | None = None,
    marker_radius_3d_um: float | None = None,
) -> VisDefaults:
    global _VIS_DEFAULTS

    updated = get_defaults()
    if layout_2d_default is not None:
        updated.layout_2d_default = _normalize_mode(
            layout_2d_default,
            supported=SUPPORTED_2D_LAYOUTS,
            label="2D layout",
        )
    if shape_2d_default is not None:
        updated.shape_2d_default = _normalize_mode(
            shape_2d_default,
            supported=SUPPORTED_2D_SHAPES,
            label="2D shape",
        )
    if mode_3d_default is not None:
        updated.mode_3d_default = _normalize_mode(mode_3d_default, supported=SUPPORTED_3D_MODES, label="3D")
    if branch_type_colors is not None:
        normalized = {str(branch_type): _normalize_color(color) for branch_type, color in branch_type_colors.items()}
        if replace_branch_type_colors:
            updated.branch_type_colors = normalized
        else:
            merged = dict(updated.branch_type_colors)
            merged.update(normalized)
            updated.branch_type_colors = merged
    if branch_type_edge_colors_2d is not None:
        normalized_edge_2d = {
            str(branch_type): _normalize_color(color)
            for branch_type, color in branch_type_edge_colors_2d.items()
        }
        if replace_branch_type_edge_colors_2d or updated.branch_type_edge_colors_2d is None:
            updated.branch_type_edge_colors_2d = normalized_edge_2d
        else:
            merged_edge_2d = dict(updated.branch_type_edge_colors_2d)
            merged_edge_2d.update(normalized_edge_2d)
            updated.branch_type_edge_colors_2d = merged_edge_2d
    if alpha_2d is not None:
        updated.alpha_2d = _normalize_alpha(alpha_2d, label="alpha_2d")
    if alpha_2d_poly is not None:
        updated.alpha_2d_poly = _normalize_alpha(alpha_2d_poly, label="alpha_2d_poly")
    if alpha_2d_line is not None:
        updated.alpha_2d_line = _normalize_alpha(alpha_2d_line, label="alpha_2d_line")
    if frustum_edge_linewidth_2d is not None:
        value = float(frustum_edge_linewidth_2d)
        if value < 0.0:
            raise ValueError(
                f"frustum_edge_linewidth_2d must be >= 0, got {frustum_edge_linewidth_2d!r}."
            )
        updated.frustum_edge_linewidth_2d = value
    if alpha_3d_tube is not None:
        updated.alpha_3d_tube = _normalize_alpha(alpha_3d_tube, label="alpha_3d_tube")
    if highlight_color is not None:
        updated.highlight_color = _normalize_color(highlight_color)
    if highlight_alpha is not None:
        updated.highlight_alpha = _normalize_alpha(highlight_alpha, label="highlight_alpha")
    if marker_color is not None:
        updated.marker_color = _normalize_color(marker_color)
    if marker_size_2d is not None:
        value = float(marker_size_2d)
        if value <= 0.0:
            raise ValueError(f"marker_size_2d must be > 0, got {marker_size_2d!r}.")
        updated.marker_size_2d = value
    if marker_radius_3d_um is not None:
        value = float(marker_radius_3d_um)
        if value <= 0.0:
            raise ValueError(f"marker_radius_3d_um must be > 0, got {marker_radius_3d_um!r}.")
        updated.marker_radius_3d_um = value

    _VIS_DEFAULTS = updated
    return get_defaults()


set_defaults = configure


# ---------------------------------------------------------------------------
# Publication theme preset
# ---------------------------------------------------------------------------


# Branch colours chosen for a publication-ready palette — higher
# contrast, print-friendly, and accessible to the common forms of
# colour blindness (the palette is adapted from Paul Tol's "muted"
# cycle: https://personal.sron.nl/~pault/). The keys mirror the
# default ``DEFAULT_BRANCH_TYPE_COLORS`` so callers can diff the two
# presets side by side.
PUBLICATION_BRANCH_TYPE_COLORS = {
    "soma": (17, 17, 17),
    "axon": (51, 34, 136),
    "basal_dendrite": (136, 34, 85),
    "apical_dendrite": (204, 102, 119),
    "dendrite": (170, 68, 153),
    "custom": (68, 68, 68),
}


# Matplotlib rcParams applied when the publication theme is active.
# The aim is LaTeX-style output: serif font, thicker lines, no grid,
# tight margins. Values are chosen for raster export at 300 dpi and
# look good in PDF / PNG / SVG without further tweaking.
PUBLICATION_RC_PARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size": 11.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 11.0,
    "axes.linewidth": 1.2,
    "axes.grid": False,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.6,
    "lines.antialiased": True,
    "figure.dpi": 150.0,
    "savefig.dpi": 300.0,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
}


@dataclass(frozen=True)
class PublicationTheme:
    """Publication-quality styling preset for :mod:`braincell.vis`.

    Bundles a :class:`VisDefaults` diff (serif-friendly branch colours,
    thicker strokes) with a matplotlib ``rcParams`` block so a single
    context manager flips both at once.

    Parameters
    ----------
    branch_type_colors : mapping
        Branch-type colour overrides shared by 2D and 3D. Defaults to
        :data:`PUBLICATION_BRANCH_TYPE_COLORS`.
    branch_type_edge_colors_2d : mapping or None
        Optional 2D frustum border colour overrides. When ``None``,
        border colours are derived automatically from the fill colours.
    rc_params : mapping
        Matplotlib ``rcParams`` applied inside the theme. Defaults to
        :data:`PUBLICATION_RC_PARAMS`.
    alpha_2d : float
        Shared 2D opacity used by both line and frustum rendering.
    frustum_edge_linewidth_2d : float
        Border linewidth used by 2D frustum rendering.
    alpha_3d_tube : float
        Tube alpha used by the 3D backend. Defaults to ``1.0``.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.vis import publication_theme, plot2d
        >>> with publication_theme():                 # doctest: +SKIP
        ...     plot2d(tree, layout='stem')           # doctest: +SKIP
    """

    branch_type_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(PUBLICATION_BRANCH_TYPE_COLORS)
    )
    branch_type_edge_colors_2d: dict[str, tuple[int, int, int]] | None = None
    rc_params: dict[str, object] = field(default_factory=lambda: dict(PUBLICATION_RC_PARAMS))
    alpha_2d: float = 0.7
    frustum_edge_linewidth_2d: float = DEFAULT_2D_FRUSTUM_EDGE_LINEWIDTH
    alpha_3d_tube: float = 1.0


@contextmanager
def publication_theme(
    preset: PublicationTheme | None = None,
    *,
    rc_overrides: dict[str, object] | None = None,
) -> Iterator[VisDefaults]:
    """Activate the :class:`PublicationTheme` for the duration of a block.

    On entry this context manager applies the preset's branch colours
    and alphas to :data:`_VIS_DEFAULTS` **and** patches matplotlib's
    ``rcParams`` with the preset's styling block. On exit both are
    restored, even if the body raised.

    Parameters
    ----------
    preset : PublicationTheme or None
        Theme to activate. When ``None`` the default instance is used.
    rc_overrides : mapping or None
        Extra ``rcParams`` applied on top of the preset. Useful for
        one-off tweaks without subclassing the preset.

    Yields
    ------
    VisDefaults
        A snapshot of the vis defaults active inside the block.

    Notes
    -----
    The matplotlib patch goes through
    :func:`matplotlib.rcParams.update` and is restored from the
    original values — this is the same mechanism :func:`plt.rc_context`
    uses internally, but scoped to the keys the preset actually sets
    so unrelated rc state is left alone.
    """
    global _VIS_DEFAULTS

    active = preset or PublicationTheme()
    vis_snapshot = _copy_defaults(_VIS_DEFAULTS)

    # Apply vis defaults.
    configure(
        branch_type_colors=dict(active.branch_type_colors),
        branch_type_edge_colors_2d=(
            dict(active.branch_type_edge_colors_2d)
            if active.branch_type_edge_colors_2d is not None
            else None
        ),
        alpha_2d=active.alpha_2d,
        frustum_edge_linewidth_2d=active.frustum_edge_linewidth_2d,
        alpha_3d_tube=active.alpha_3d_tube,
    )

    # Apply matplotlib rcParams lazily — only import if available.
    import importlib.util

    mpl_available = importlib.util.find_spec("matplotlib") is not None
    rc_snapshot: dict[str, object] = {}
    if mpl_available:
        import matplotlib as mpl

        merged_rc = dict(active.rc_params)
        if rc_overrides is not None:
            merged_rc.update(rc_overrides)
        # Skip unknown keys so tests never crash on older matplotlib.
        valid_keys = set(mpl.rcParams.keys())
        effective_rc = {k: v for k, v in merged_rc.items() if k in valid_keys}
        rc_snapshot = {key: mpl.rcParams[key] for key in effective_rc}
        mpl.rcParams.update(effective_rc)

    try:
        yield get_defaults()
    finally:
        _VIS_DEFAULTS = vis_snapshot
        if mpl_available and rc_snapshot:
            import matplotlib as mpl

            mpl.rcParams.update(rc_snapshot)


@contextmanager
def theme(**overrides: object) -> Iterator[VisDefaults]:
    """Temporarily override visualization defaults for the duration of a block.

    All keyword arguments are forwarded to :func:`configure`. On exit the
    previous defaults are restored, even if the block raised.

    Parameters
    ----------
    **overrides
        Any keyword accepted by :func:`configure`, e.g. ``layout_2d_default``,
        ``branch_type_colors``, ``alpha_2d_line``, ``highlight_color``.

    Yields
    ------
    VisDefaults
        A snapshot of the effective defaults inside the ``with`` block.

    Examples
    --------

    .. code-block:: python

        >>> import braincell
        >>> with braincell.vis.theme(branch_type_colors={"axon": "#ff0000"}):
        ...     morpho.vis2d()  # doctest: +SKIP
        >>> morpho.vis2d()       # doctest: +SKIP  — original colors restored
    """
    global _VIS_DEFAULTS

    snapshot = _copy_defaults(_VIS_DEFAULTS)
    try:
        configure(**overrides)  # type: ignore[arg-type]
        yield get_defaults()
    finally:
        _VIS_DEFAULTS = snapshot


def resolve_default_2d_layout(layout: str | None) -> str:
    return _VIS_DEFAULTS.layout_2d_default if layout is None else layout


def resolve_default_2d_shape(shape: str | None) -> str:
    return _VIS_DEFAULTS.shape_2d_default if shape is None else shape


def resolve_default_3d_mode(mode: str | None) -> str:
    return _VIS_DEFAULTS.mode_3d_default if mode is None else mode


def color_for_branch_type(branch_type: str) -> tuple[int, int, int]:
    return _VIS_DEFAULTS.branch_type_colors.get(
        branch_type,
        _VIS_DEFAULTS.branch_type_colors.get("custom", (110, 110, 110)),
    )


def color_for_2d_branch_type(branch_type: str) -> tuple[int, int, int]:
    return color_for_branch_type(branch_type)


def edge_color_for_2d_branch_type(branch_type: str) -> tuple[int, int, int]:
    fill_color = color_for_2d_branch_type(branch_type)
    if _VIS_DEFAULTS.branch_type_edge_colors_2d is not None:
        return _VIS_DEFAULTS.branch_type_edge_colors_2d.get(
            branch_type,
            _VIS_DEFAULTS.branch_type_edge_colors_2d.get(
                "custom",
                _darken_color(fill_color, factor=DEFAULT_2D_EDGE_DARKEN_FACTOR),
            ),
        )
    return _darken_color(fill_color, factor=DEFAULT_2D_EDGE_DARKEN_FACTOR)


def alpha_for_2d() -> float:
    return _VIS_DEFAULTS.alpha_2d


def alpha_for_2d_poly() -> float:
    if _VIS_DEFAULTS.alpha_2d_poly is not None:
        return _VIS_DEFAULTS.alpha_2d_poly
    return _VIS_DEFAULTS.alpha_2d


def alpha_for_2d_line() -> float:
    if _VIS_DEFAULTS.alpha_2d_line is not None:
        return _VIS_DEFAULTS.alpha_2d_line
    return _VIS_DEFAULTS.alpha_2d


def frustum_edge_linewidth_2d() -> float:
    return _VIS_DEFAULTS.frustum_edge_linewidth_2d


def alpha_for_3d_tube() -> float:
    return _VIS_DEFAULTS.alpha_3d_tube


def highlight_color() -> tuple[int, int, int]:
    return _VIS_DEFAULTS.highlight_color


def highlight_alpha() -> float:
    return _VIS_DEFAULTS.highlight_alpha


def marker_color() -> tuple[int, int, int]:
    return _VIS_DEFAULTS.marker_color


def marker_size_2d() -> float:
    return _VIS_DEFAULTS.marker_size_2d


def marker_radius_3d_um() -> float:
    return _VIS_DEFAULTS.marker_radius_3d_um


def _copy_defaults(defaults: VisDefaults) -> VisDefaults:
    return VisDefaults(
        layout_2d_default=defaults.layout_2d_default,
        shape_2d_default=defaults.shape_2d_default,
        mode_3d_default=defaults.mode_3d_default,
        branch_type_colors=dict(defaults.branch_type_colors),
        branch_type_edge_colors_2d=(
            dict(defaults.branch_type_edge_colors_2d)
            if defaults.branch_type_edge_colors_2d is not None
            else None
        ),
        alpha_2d=defaults.alpha_2d,
        alpha_2d_poly=defaults.alpha_2d_poly,
        alpha_2d_line=defaults.alpha_2d_line,
        frustum_edge_linewidth_2d=defaults.frustum_edge_linewidth_2d,
        alpha_3d_tube=defaults.alpha_3d_tube,
        highlight_color=defaults.highlight_color,
        highlight_alpha=defaults.highlight_alpha,
        marker_color=defaults.marker_color,
        marker_size_2d=defaults.marker_size_2d,
        marker_radius_3d_um=defaults.marker_radius_3d_um,
    )


def _normalize_mode(mode: str, *, supported: frozenset[str], label: str) -> str:
    if mode not in supported:
        expected = ", ".join(sorted(repr(item) for item in supported))
        raise ValueError(f"Unsupported default {label} mode {mode!r}. Expected one of {expected}.")
    return mode


def _normalize_alpha(alpha: float, *, label: str) -> float:
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0.0 and 1.0, got {alpha!r}.")
    return value


def _normalize_color(color: object) -> tuple[int, int, int]:
    if isinstance(color, str):
        text = color.strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) != 6:
            raise ValueError(f"Hex colors must be in '#RRGGBB' format, got {color!r}.")
        try:
            return tuple(int(text[index: index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError(f"Hex colors must be in '#RRGGBB' format, got {color!r}.") from exc

    if not isinstance(color, Iterable):
        raise TypeError(f"Color must be an RGB iterable or '#RRGGBB' string, got {type(color).__name__!s}.")

    channels = tuple(float(channel) for channel in color)
    if len(channels) != 3:
        raise ValueError(f"Color iterable must contain exactly 3 channels, got {len(channels)}.")
    if all(0.0 <= channel <= 1.0 for channel in channels):
        scaled = tuple(int(round(channel * 255.0)) for channel in channels)
    else:
        scaled = tuple(int(round(channel)) for channel in channels)
    if not all(0 <= channel <= 255 for channel in scaled):
        raise ValueError(f"RGB channels must be between 0 and 255, got {color!r}.")
    return scaled


def _darken_color(color: tuple[int, int, int], *, factor: float) -> tuple[int, int, int]:
    return tuple(
        max(0, min(255, int(round(channel * factor))))
        for channel in color
    )
