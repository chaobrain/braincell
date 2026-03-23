from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from braincell.filter import LocsetMask, RegionMask
from braincell.morpho import Morpho
from .backends import BackendChooser
from .geometry import build_render_geometry_3d
from .types import RenderGeometry3D

ArrayLike = Any


@dataclass(frozen=True)
class OverlaySpec:
    region: RegionMask | None = None
    locset: LocsetMask | None = None
    values: ArrayLike | None = None


@dataclass(frozen=True)
class RenderRequest:
    morpho: Morpho
    overlay: OverlaySpec = OverlaySpec()
    dimensionality: str = "auto"
    geometry3d: RenderGeometry3D | None = None


def plot(
    morpho: Morpho,
    *,
    region: RegionMask | None = None,
    locset: LocsetMask | None = None,
    values: ArrayLike | None = None,
    dimensionality: str = "auto",
    backend: str | None = None,
    chooser: BackendChooser | None = None,
) -> object:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"plot(...) expects Morpho, got {type(morpho).__name__!s}.")
    chooser = chooser or BackendChooser.default()
    backend_impl = chooser.pick(requested=backend)
    resolved_dimensionality = dimensionality
    if resolved_dimensionality == "auto" and backend_impl.name == "pyvista":
        resolved_dimensionality = "3d"
    geometry3d = build_render_geometry_3d(morpho) if resolved_dimensionality == "3d" else None
    request = RenderRequest(
        morpho=morpho,
        overlay=OverlaySpec(region=region, locset=locset, values=values),
        dimensionality=resolved_dimensionality,
        geometry3d=geometry3d,
    )
    return backend_impl.render(request)
