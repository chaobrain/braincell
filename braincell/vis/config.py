from dataclasses import dataclass, field
from typing import Iterable


DEFAULT_BRANCH_TYPE_COLORS = {
    "soma": (0, 0, 0),
    "axon": (70, 130, 180),
    "basal_dendrite": (178, 34, 34),
    "apical_dendrite": (255, 127, 80),
    "dendrite": (205, 92, 92),
    "custom": (110, 110, 110),
}
SUPPORTED_2D_MODES = frozenset({"projected", "tree", "frustum"})
SUPPORTED_3D_MODES = frozenset({"geometry"})


@dataclass
class VisDefaults:
    mode_2d_default: str = "frustum"
    mode_3d_default: str = "geometry"
    branch_type_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_BRANCH_TYPE_COLORS)
    )
    alpha_2d_poly: float = 0.3
    alpha_2d_line: float = 1.0
    alpha_3d_tube: float = 1.0


_VIS_DEFAULTS = VisDefaults()


def get_defaults() -> VisDefaults:
    return _copy_defaults(_VIS_DEFAULTS)


def reset_defaults() -> VisDefaults:
    global _VIS_DEFAULTS
    _VIS_DEFAULTS = VisDefaults()
    return get_defaults()


def configure(
    *,
    mode_2d_default: str | None = None,
    mode_3d_default: str | None = None,
    branch_type_colors: dict[str, object] | None = None,
    replace_branch_type_colors: bool = False,
    alpha_2d_poly: float | None = None,
    alpha_2d_line: float | None = None,
    alpha_3d_tube: float | None = None,
) -> VisDefaults:
    global _VIS_DEFAULTS

    updated = get_defaults()
    if mode_2d_default is not None:
        updated.mode_2d_default = _normalize_mode(mode_2d_default, supported=SUPPORTED_2D_MODES, label="2D")
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
    if alpha_2d_poly is not None:
        updated.alpha_2d_poly = _normalize_alpha(alpha_2d_poly, label="alpha_2d_poly")
    if alpha_2d_line is not None:
        updated.alpha_2d_line = _normalize_alpha(alpha_2d_line, label="alpha_2d_line")
    if alpha_3d_tube is not None:
        updated.alpha_3d_tube = _normalize_alpha(alpha_3d_tube, label="alpha_3d_tube")

    _VIS_DEFAULTS = updated
    return get_defaults()


set_defaults = configure


def resolve_default_2d_mode(mode: str | None) -> str:
    return _VIS_DEFAULTS.mode_2d_default if mode is None else mode


def resolve_default_3d_mode(mode: str | None) -> str:
    return _VIS_DEFAULTS.mode_3d_default if mode is None else mode


def color_for_branch_type(branch_type: str) -> tuple[int, int, int]:
    return _VIS_DEFAULTS.branch_type_colors.get(
        branch_type,
        _VIS_DEFAULTS.branch_type_colors.get("custom", (110, 110, 110)),
    )


def alpha_for_2d_poly() -> float:
    return _VIS_DEFAULTS.alpha_2d_poly


def alpha_for_2d_line() -> float:
    return _VIS_DEFAULTS.alpha_2d_line


def alpha_for_3d_tube() -> float:
    return _VIS_DEFAULTS.alpha_3d_tube


def _copy_defaults(defaults: VisDefaults) -> VisDefaults:
    return VisDefaults(
        mode_2d_default=defaults.mode_2d_default,
        mode_3d_default=defaults.mode_3d_default,
        branch_type_colors=dict(defaults.branch_type_colors),
        alpha_2d_poly=defaults.alpha_2d_poly,
        alpha_2d_line=defaults.alpha_2d_line,
        alpha_3d_tube=defaults.alpha_3d_tube,
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


__all__ = [
    "DEFAULT_BRANCH_TYPE_COLORS",
    "SUPPORTED_2D_MODES",
    "SUPPORTED_3D_MODES",
    "VisDefaults",
    "alpha_for_2d_line",
    "alpha_for_2d_poly",
    "alpha_for_3d_tube",
    "color_for_branch_type",
    "configure",
    "get_defaults",
    "reset_defaults",
    "resolve_default_2d_mode",
    "resolve_default_3d_mode",
    "set_defaults",
]
