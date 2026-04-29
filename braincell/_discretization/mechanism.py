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

"""Declaration-rule normalization and CV mechanism lowering.

This module owns the declaration-side mechanism pipeline:

- normalize ``paint(...)`` and ``place(...)`` inputs into frozen rule
  records
- merge successive declarations with deterministic overwrite behavior
- lower those declarations onto per-CV mechanism buckets
"""

from dataclasses import dataclass, field, replace
from typing import Literal

import brainunit as u

from braincell.filter import AllRegion, LocsetExpr, RegionExpr
from braincell.filter.cache import SelectionCache
from braincell.mech import (
    CableProperty,
    CurrentProbe,
    Density,
    MechanismProbe,
    Point,
    StateProbe,
)
from braincell.morph.morphology import Morphology
from .base import CVPointMechanism, Position
from .geometry import (
    CVGeometryResult,
    EPS_AREA_UM2,
    EPS_PARAM,
    _GeoCV,
    _build_frusta,
    _lateral_area_um2,
)

__all__ = [
    "PaintRule",
    "PlaceRule",
    "build_cv_mechanisms",
    "default_paint_rules",
    "merge_paint_rules",
    "merge_place_rules",
    "normalize_paint_rules",
    "normalize_place_rule",
]

_DEFAULT_CABLE = CableProperty(
    resting_potential=-65.0 * u.mV,
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
)


@dataclass(frozen=True)
class PaintRule:
    """Normalized ``Cell.paint(...)`` declaration.

    Attributes
    ----------
    region : RegionExpr
        Region expression being painted.
    mechanism : CableProperty or Density
        Mechanism declaration applied over that region.
    """

    region: RegionExpr
    mechanism: CableProperty | Density


@dataclass(frozen=True)
class PlaceRule:
    """Normalized ``Cell.place(...)`` declaration.

    Attributes
    ----------
    locset : LocsetExpr
        Location expression being targeted.
    mechanisms : tuple of Point
        Point-mechanism declarations applied at each resolved location.
    site : {"mid"}, optional
        Reserved placement-site tag used by the current lowering model.
    """

    locset: LocsetExpr
    mechanisms: tuple[Point, ...]
    site: Literal["mid"] = "mid"


@dataclass
class _MechBucket:
    cable: CableProperty
    density_by_key: dict[tuple[int, str, object], Density]
    points: list[Point]
    point_roles: list[CVPointMechanism] = field(default_factory=list)


class _RegionCache:
    """Per-build cache of region / locset evaluation outputs."""

    def __init__(self, morpho: Morphology) -> None:
        self._morpho = morpho
        self._selection = SelectionCache()
        self._region_by_id: dict[int, dict[int, tuple[tuple[float, float], ...]]] = {}
        self._locset_by_id: dict[int, tuple[tuple[int, float, str], ...]] = {}

    def intervals(
        self, region: RegionExpr
    ) -> dict[int, tuple[tuple[float, float], ...]]:
        key = id(region)
        cached = self._region_by_id.get(key)
        if cached is not None:
            return cached
        mask = region.evaluate(self._morpho, cache=self._selection)
        grouped: dict[int, list[tuple[float, float]]] = {}
        for branch_id, prox, dist in mask.intervals:
            grouped.setdefault(int(branch_id), []).append((float(prox), float(dist)))
        result = {bid: tuple(ranges) for bid, ranges in grouped.items()}
        self._region_by_id[key] = result
        return result

    def points(self, locset: LocsetExpr) -> tuple[tuple[int, float, str], ...]:
        key = id(locset)
        cached = self._locset_by_id.get(key)
        if cached is not None:
            return cached
        mask = locset.evaluate(self._morpho, cache=self._selection)
        result = tuple(
            (int(point[0]), float(point[1]), str(name))
            for point, name in zip(mask.points, mask.display_names)
        )
        self._locset_by_id[key] = result
        return result


def default_paint_rules() -> tuple[PaintRule, ...]:
    """Return the default global cable-property rule.

    Returns
    -------
    tuple of PaintRule
        One global rule applying the package default cable properties to
        ``AllRegion()``.
    """
    return (PaintRule(region=AllRegion(), mechanism=_DEFAULT_CABLE),)


def normalize_paint_rules(
    region: RegionExpr,
    mechanisms: tuple[object, ...],
) -> tuple[PaintRule, ...]:
    """Normalize one ``Cell.paint(...)`` call into paint rules.

    Parameters
    ----------
    region : RegionExpr
        Region expression being painted.
    mechanisms : tuple of object
        Candidate mechanism declarations. Each item must be a
        :class:`CableProperty` or :class:`Density`.

    Returns
    -------
    tuple of PaintRule
        One normalized rule per mechanism argument.

    Raises
    ------
    TypeError
        If ``region`` is not a region expression or if any mechanism is
        not cable-like or density-like.
    ValueError
        If no mechanisms are supplied.
    """
    if not isinstance(region, RegionExpr):
        raise TypeError(
            f"Cell.paint(...) expects RegionExpr, got {type(region).__name__!s}."
        )
    if len(mechanisms) == 0:
        raise ValueError("Cell.paint(...) expects at least one mechanism.")

    rules: list[PaintRule] = []
    for mechanism in mechanisms:
        if isinstance(mechanism, (CableProperty, Density)):
            rules.append(PaintRule(region=region, mechanism=mechanism))
            continue
        raise TypeError(
            "Cell.paint(...) mechanisms must be CableProperty or Density "
            f"(use braincell.mech.Channel / Ion), got {type(mechanism).__name__!s}."
        )
    return tuple(rules)


def normalize_place_rule(
    locset: LocsetExpr,
    mechanisms: tuple[object, ...],
) -> PlaceRule:
    """Normalize one ``Cell.place(...)`` call into a place rule.

    Parameters
    ----------
    locset : LocsetExpr
        Location expression being targeted.
    mechanisms : tuple of object
        Candidate point-mechanism declarations.

    Returns
    -------
    PlaceRule
        Frozen normalized place rule.

    Raises
    ------
    TypeError
        If ``locset`` is not a locset expression or any mechanism is
        not a point declaration.
    ValueError
        If no mechanisms are supplied.
    """
    if not isinstance(locset, LocsetExpr):
        raise TypeError(
            f"Cell.place(...) expects LocsetExpr, got {type(locset).__name__!s}."
        )
    if len(mechanisms) == 0:
        raise ValueError("Cell.place(...) expects at least one point mechanism.")

    normalized: list[Point] = []
    for mechanism in mechanisms:
        if not isinstance(mechanism, Point):
            raise TypeError(
                "Cell.place(...) mechanisms must be Point instances, "
                f"got {type(mechanism).__name__!s}."
            )
        normalized.append(mechanism)
    return PlaceRule(locset=locset, mechanisms=tuple(normalized), site="mid")


def _paint_key(rule: PaintRule) -> tuple[object, str, str, object]:
    if isinstance(rule.mechanism, CableProperty):
        return (rule.region, "cable", "", None)
    return (
        rule.region,
        "density",
        rule.mechanism.class_name,
        getattr(rule.mechanism, "name", None),
    )


def merge_paint_rules(
    existing: tuple[PaintRule, ...],
    incoming: tuple[PaintRule, ...],
) -> tuple[PaintRule, ...]:
    """Merge normalized paint rules with overwrite-on-identity semantics.

    Parameters
    ----------
    existing : tuple of PaintRule
        Existing accumulated paint rules.
    incoming : tuple of PaintRule
        Newly normalized paint rules.

    Returns
    -------
    tuple of PaintRule
        Merged rules where later declarations replace earlier rules with
        the same effective paint identity.
    """
    merged: list[PaintRule] = list(existing)
    for rule in incoming:
        new_key = _paint_key(rule)
        merged = [item for item in merged if _paint_key(item) != new_key]
        merged.append(rule)
    return tuple(merged)


def merge_place_rules(
    existing: tuple[PlaceRule, ...],
    incoming: tuple[PlaceRule, ...],
) -> tuple[PlaceRule, ...]:
    """Merge normalized place rules without duplicating exact matches.

    Parameters
    ----------
    existing : tuple of PlaceRule
        Existing accumulated place rules.
    incoming : tuple of PlaceRule
        Newly normalized place rules.

    Returns
    -------
    tuple of PlaceRule
        Merged place-rule sequence.
    """
    merged: list[PlaceRule] = list(existing)
    for rule in incoming:
        if rule in merged:
            continue
        merged.append(rule)
    return tuple(merged)


def _interval_contains(
    intervals: tuple[tuple[float, float], ...],
    x: float,
    *,
    epsilon: float,
) -> bool:
    for left, right in intervals:
        if left - epsilon <= x <= right + epsilon:
            return True
    return False


def _coverage_fraction(
    morpho: Morphology,
    geo: _GeoCV,
    intervals: tuple[tuple[float, float], ...],
    *,
    frusta_builder=None,
) -> float:
    if geo.lateral_area_um2 <= EPS_AREA_UM2:
        return 0.0
    branch = morpho.branches[geo.branch_id]
    build = frusta_builder if frusta_builder is not None else _build_frusta
    overlap = 0.0
    for left, right in intervals:
        start = max(geo.prox, float(left))
        end = min(geo.dist, float(right))
        if end - start <= EPS_PARAM:
            continue
        overlap += _lateral_area_um2(build(branch, prox=start, dist=end))
    return max(0.0, min(1.0, overlap / geo.lateral_area_um2))


def _apply_density(
    bucket: _MechBucket,
    mechanism: Density,
    *,
    region_key: object,
    fraction: float,
) -> None:
    key = (
        id(region_key),
        mechanism.class_name,
        getattr(mechanism, "name", None),
    )
    if mechanism.category != "channel":
        bucket.density_by_key[key] = mechanism
        return
    if fraction >= 1.0 - EPS_PARAM:
        bucket.density_by_key[key] = mechanism
    else:
        bucket.density_by_key[key] = mechanism.with_coverage(fraction)


def _resolve_point_name(mechanism: Point, *, display_name: str) -> Point:
    if isinstance(mechanism, StateProbe):
        if mechanism.name is not None:
            return mechanism
        return replace(mechanism, name=f"{display_name}_{mechanism.field}")
    if isinstance(mechanism, MechanismProbe):
        if mechanism.name is not None:
            return mechanism
        return replace(
            mechanism,
            name=f"{display_name}_{mechanism.mechanism}_{mechanism.field}",
        )
    if isinstance(mechanism, CurrentProbe):
        if mechanism.name is not None:
            return mechanism
        suffix = (
            f"{mechanism.mechanism}_current"
            if mechanism.mechanism is not None
            else f"{mechanism.ion}_current"
        )
        return replace(mechanism, name=f"{display_name}_{suffix}")
    return mechanism


def _apply_place(
    bucket: _MechBucket,
    mechanism: Point,
    *,
    display_name: str,
    seen_names: set[str],
    position: Position = "mid",
) -> Point:
    named = _resolve_point_name(mechanism, display_name=display_name)
    auto_generated = (
        getattr(mechanism, "name", None) is None
        and getattr(named, "name", None) is not None
    )
    if auto_generated:
        candidate_name = named.name
        if candidate_name in seen_names:
            raise ValueError(
                f"Duplicate auto-generated point-mechanism name "
                f"{candidate_name!r} from place rule at {display_name!r}. "
                "Supply an explicit name= argument to disambiguate."
            )
        seen_names.add(candidate_name)
    bucket.points.append(named)
    bucket.point_roles.append(CVPointMechanism(position=position, mechanism=named))
    return named


def _init_bucket() -> _MechBucket:
    return _MechBucket(
        cable=_DEFAULT_CABLE,
        density_by_key={},
        points=[],
        point_roles=[],
    )


def _position_for_geo(geo: _GeoCV, *, x: float) -> Position:
    if x <= geo.prox + EPS_PARAM:
        return "prox"
    if x >= geo.dist - EPS_PARAM:
        return "dist"
    return "mid"


def build_cv_mechanisms(
    morpho: Morphology,
    geometry: CVGeometryResult,
    *,
    paint_rules: tuple[PaintRule, ...],
    place_rules: tuple[PlaceRule, ...],
) -> list[_MechBucket]:
    """Lower normalized declaration rules onto per-CV mechanism buckets.

    Parameters
    ----------
    morpho : Morphology
        Morphology whose filters and locsets are being evaluated.
    geometry : CVGeometryResult
        Geometry-stage payload defining the current CV tiling.
    paint_rules : tuple of PaintRule
        Normalized region-based declarations.
    place_rules : tuple of PlaceRule
        Normalized locset-based declarations.

    Returns
    -------
    list of _MechBucket
        One mutable bucket per CV, later consumed by the base
        discretization assembly step.

    Notes
    -----
    Density-like mechanisms are assigned by region overlap, cable
    properties by midpoint ownership, and point mechanisms by resolved
    locset ownership.
    """
    geos = geometry.geos
    buckets = [_init_bucket() for _ in geos]
    cache = _RegionCache(morpho)

    frusta_cache: dict[tuple[int, float, float], object] = {}

    def _cached_frusta(branch, *, prox, dist):
        key = (id(branch), round(float(prox), 9), round(float(dist), 9))
        cached = frusta_cache.get(key)
        if cached is None:
            cached = _build_frusta(branch, prox=prox, dist=dist)
            frusta_cache[key] = cached
        return cached

    for rule in paint_rules:
        intervals_by_branch = cache.intervals(rule.region)
        mechanism = rule.mechanism

        for branch_id, cv_ids in enumerate(geometry.branch_to_cv_ids):
            intervals = intervals_by_branch.get(branch_id, ())
            if len(intervals) == 0:
                continue
            for cv_id in cv_ids:
                geo = geos[cv_id]
                bucket = buckets[cv_id]

                if isinstance(mechanism, CableProperty):
                    if not _interval_contains(
                        intervals,
                        geo.midpoint,
                        epsilon=EPS_PARAM,
                    ):
                        continue
                    bucket.cable = mechanism
                    continue

                fraction = _coverage_fraction(
                    morpho,
                    geo,
                    intervals,
                    frusta_builder=_cached_frusta,
                )
                if fraction <= EPS_PARAM:
                    continue
                _apply_density(
                    bucket,
                    mechanism,
                    region_key=rule.region,
                    fraction=fraction,
                )

    seen_names: set[str] = set()
    for rule in place_rules:
        for branch_id, x, display_name in cache.points(rule.locset):
            ids = geometry.cv_ids(branch_id)
            if not ids:
                continue
            cv_id = geometry.locate_cv(branch_id=branch_id, x=x)
            geo = geos[cv_id]
            position = _position_for_geo(geo, x=float(x))
            for mechanism in rule.mechanisms:
                _apply_place(
                    buckets[cv_id],
                    mechanism,
                    display_name=display_name,
                    seen_names=seen_names,
                    position=position,
                )

    return buckets
