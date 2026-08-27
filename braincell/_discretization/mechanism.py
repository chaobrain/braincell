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
import numpy as np

from braincell.filter import AllRegion, LocsetBatch, LocsetExpr, LocsetMask, RegionExpr
from braincell.filter.cache import SelectionCache
from braincell.mech import (
    CableProperty,
    CVContext,
    CurrentProbe,
    Density,
    MechanismProbe,
    Point,
    StateProbe,
)
from braincell.morph.morphology import Morphology
from .base import CVPointMechanism, Position
from .context import build_cv_contexts
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
    membrane_capacitance=1.0 * (u.uF / u.cm**2),
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
    locset : LocsetExpr or LocsetMask or LocsetBatch
        Location expression being targeted.
    mechanisms : tuple of Point
        Point-mechanism declarations applied at each resolved location.
    site : {"mid"}, optional
        Reserved placement-site tag used by the current lowering model.
    """

    locset: LocsetExpr | LocsetMask | LocsetBatch
    mechanisms: tuple[Point, ...]
    site: Literal["mid"] = "mid"
    population_indices: tuple[int, ...] | None = None
    aligned: bool = False


@dataclass
class _MechBucket:
    cable: CableProperty
    density_by_key: dict[tuple[int, str, object], Density]
    points: list[Point]
    point_roles: list[CVPointMechanism] = field(default_factory=list)


@dataclass(frozen=True)
class _RegionCVCoverage:
    """Per-region coverage cache in CV order."""

    cable_contains_midpoint: tuple[bool, ...]
    density_fraction: tuple[float, ...]


class _RegionCache:
    """Per-build cache of region / locset evaluation outputs."""

    def __init__(self, morpho: Morphology) -> None:
        self._morpho = morpho
        self._selection = SelectionCache()
        self._region_by_id: dict[int, dict[int, tuple[tuple[float, float], ...]]] = {}
        self._locset_by_id: dict[int, tuple[tuple[int, float, str], ...]] = {}

    def intervals(self, region: RegionExpr) -> dict[int, tuple[tuple[float, float], ...]]:
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

    def points(self, locset: LocsetExpr | LocsetMask) -> tuple[tuple[int, float, str], ...]:
        key = id(locset)
        cached = self._locset_by_id.get(key)
        if cached is not None:
            return cached
        mask = locset.evaluate(self._morpho, cache=self._selection) if isinstance(locset, LocsetExpr) else locset
        names = mask.display_names
        if names is None:
            names = tuple(
                f"{self._morpho.branch(index=int(branch_id)).name}({float(branch_x):g})"
                for branch_id, branch_x in mask.points
            )
        result = tuple(
            (int(branch_id), float(branch_x), str(name))
            for branch_id, branch_x, name in zip(mask.branch_id, mask.branch_x, names)
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
        raise TypeError(f"Cell.paint(...) expects RegionExpr, got {type(region).__name__!s}.")
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
    locset: LocsetExpr | LocsetMask | LocsetBatch,
    mechanisms: tuple[object, ...],
    *,
    population_indices: tuple[int, ...] | None = None,
    aligned: bool = False,
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
    if not isinstance(locset, (LocsetExpr, LocsetMask, LocsetBatch)):
        raise TypeError(
            f"Cell.place(...) expects LocsetExpr, LocsetMask, or LocsetBatch, got {type(locset).__name__!s}."
        )
    if aligned and not isinstance(locset, LocsetBatch):
        raise TypeError("Aligned Cell.place(...) rules require LocsetBatch.")
    if isinstance(locset, LocsetBatch) and not aligned:
        raise ValueError("LocsetBatch placement must be aligned with population members.")
    if len(mechanisms) == 0:
        raise ValueError("Cell.place(...) expects at least one point mechanism.")

    normalized: list[Point] = []
    for mechanism in mechanisms:
        if not isinstance(mechanism, Point):
            raise TypeError(f"Cell.place(...) mechanisms must be Point instances, got {type(mechanism).__name__!s}.")
        normalized.append(mechanism)
    return PlaceRule(
        locset=locset,
        mechanisms=tuple(normalized),
        site="mid",
        population_indices=population_indices,
        aligned=aligned,
    )


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
    """Merge cable rules by identity and append density declarations.

    Parameters
    ----------
    existing : tuple of PaintRule
        Existing accumulated paint rules.
    incoming : tuple of PaintRule
        Newly normalized paint rules.

    Returns
    -------
    tuple of PaintRule
        Cable rules retain overwrite-on-identical-region behavior. Density
        rules remain independent declarations so CV-level identity conflicts
        can be validated after continuous regions are resolved.
    """
    merged: list[PaintRule] = list(existing)
    for rule in incoming:
        if not isinstance(rule.mechanism, CableProperty):
            merged.append(rule)
            continue
        new_key = _paint_key(rule)
        merged = [item for item in merged if _paint_key(item) != new_key]
        merged.append(rule)
    return tuple(merged)


def merge_place_rules(
    existing: tuple[PlaceRule, ...],
    incoming: tuple[PlaceRule, ...],
) -> tuple[PlaceRule, ...]:
    """Append normalized place rules in declaration order.

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
    return tuple(existing) + tuple(incoming)


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
    branch=None,
    frusta_builder=None,
) -> float:
    if geo.lateral_area_um2 <= EPS_AREA_UM2:
        return 0.0
    if branch is None:
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


def _region_cv_coverage(
    intervals_by_branch: dict[int, tuple[tuple[float, float], ...]],
    geometry: CVGeometryResult,
    *,
    morpho: Morphology,
    branches,
    frusta_builder,
) -> _RegionCVCoverage:
    cable_contains = [False] * len(geometry.geos)
    density_fraction = [0.0] * len(geometry.geos)

    for branch_id, cv_ids in enumerate(geometry.branch_to_cv_ids):
        intervals = intervals_by_branch.get(branch_id, ())
        if len(intervals) == 0:
            continue
        branch = branches[branch_id]
        for cv_id in cv_ids:
            geo = geometry.geos[cv_id]
            cable_contains[cv_id] = _interval_contains(
                intervals,
                geo.midpoint,
                epsilon=EPS_PARAM,
            )
            density_fraction[cv_id] = _coverage_fraction(
                morpho,
                geo,
                intervals,
                branch=branch,
                frusta_builder=frusta_builder,
            )

    return _RegionCVCoverage(
        cable_contains_midpoint=tuple(cable_contains),
        density_fraction=tuple(density_fraction),
    )


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
        suffix = f"{mechanism.mechanism}_current" if mechanism.mechanism is not None else f"{mechanism.ion}_current"
        return replace(mechanism, name=f"{display_name}_{suffix}")
    return mechanism


def _apply_place(
    bucket: _MechBucket,
    mechanism: Point,
    *,
    display_name: str,
    placement_id: int = 0,
    branch_id: int = 0,
    branch_x: float = 0.5,
    seen_names: set[str],
    position: Position = "mid",
    population_index: int | None = None,
) -> Point:
    named = _resolve_point_name(mechanism, display_name=display_name)
    auto_generated = getattr(mechanism, "name", None) is None and getattr(named, "name", None) is not None
    if auto_generated:
        candidate_name = named.name
        if candidate_name in seen_names:
            candidate_name = f"{candidate_name}__placement_{int(placement_id)}"
            named = replace(named, name=candidate_name)
        seen_names.add(candidate_name)
    bucket.points.append(named)
    bucket.point_roles.append(
        CVPointMechanism(
            placement_id=int(placement_id),
            position=position,
            mechanism=named,
            display_name=str(display_name),
            branch_id=int(branch_id),
            branch_x=float(branch_x),
            population_index=population_index,
        )
    )
    return named


def _init_bucket() -> _MechBucket:
    return _MechBucket(
        cable=_DEFAULT_CABLE,
        density_by_key={},
        points=[],
        point_roles=[],
    )


def _resolve_cable_property(
    cable: CableProperty,
    context: CVContext,
) -> CableProperty:
    return CableProperty(
        resting_potential=_resolve_cable_field(
            cable.resting_potential,
            context,
            name="resting_potential",
            unit=u.mV,
        ),
        membrane_capacitance=_resolve_cable_field(
            cable.membrane_capacitance,
            context,
            name="membrane_capacitance",
            unit=u.uF / u.cm**2,
        ),
        axial_resistivity=_resolve_cable_field(
            cable.axial_resistivity,
            context,
            name="axial_resistivity",
            unit=u.ohm * u.cm,
        ),
        temperature=_resolve_cable_field(
            cable.temperature,
            context,
            name="temperature",
            unit=u.kelvin,
        ),
    )


def _resolve_cable_field(value, context: CVContext, *, name: str, unit) -> u.Quantity:
    resolved = value(context) if callable(value) else value
    if not hasattr(resolved, "to_decimal") or not callable(getattr(resolved, "to_decimal")):
        raise TypeError(f"CableProperty.{name} callable must return a Quantity, got {resolved!r}.")
    decimal = np.asarray(resolved.to_decimal(unit), dtype=float)
    if decimal.shape not in ((), (1,)):
        raise TypeError(f"CableProperty.{name} must resolve to a scalar Quantity, got shape {decimal.shape!r}.")
    return u.Quantity(float(decimal.reshape(())), unit)


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
    cv_contexts: tuple[CVContext, ...] | None = None,
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
    cv_contexts : tuple of CVContext or None, optional
        Precomputed contexts for ``geometry``. When omitted they are built
        from the supplied morphology and geometry.

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
    if cv_contexts is None:
        cv_contexts = build_cv_contexts(morpho, geos)
    if len(cv_contexts) != len(geos):
        raise ValueError(
            "cv_contexts must contain exactly one context per geometry CV; "
            f"got {len(cv_contexts)!r} contexts for {len(geos)!r} CVs."
        )
    branches = morpho.branches
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

    coverage_cache: dict[int, _RegionCVCoverage] = {}

    density_type_by_name: dict[tuple[str, str], str] = {}
    for rule in paint_rules:
        mechanism = rule.mechanism
        if isinstance(mechanism, CableProperty):
            continue
        name_key = (str(mechanism.category), str(mechanism.instance_name))
        previous_type = density_type_by_name.setdefault(name_key, str(mechanism.class_name))
        if previous_type != mechanism.class_name:
            raise ValueError(
                f"{mechanism.category.title()} name {mechanism.instance_name!r} cannot denote both "
                f"{previous_type!r} and {mechanism.class_name!r}. Use distinct names."
            )

    active_density_owner: dict[tuple[int, str, str], tuple[str, int]] = {}
    for rule_index, rule in enumerate(paint_rules):
        intervals_by_branch = cache.intervals(rule.region)
        mechanism = rule.mechanism
        region_key = id(rule.region)
        coverage = coverage_cache.get(region_key)
        if coverage is None:
            coverage = _region_cv_coverage(
                intervals_by_branch,
                geometry,
                morpho=morpho,
                branches=branches,
                frusta_builder=_cached_frusta,
            )
            coverage_cache[region_key] = coverage

        if isinstance(mechanism, CableProperty):
            for cv_id, contains in enumerate(coverage.cable_contains_midpoint):
                if contains:
                    buckets[cv_id].cable = _resolve_cable_property(
                        mechanism,
                        cv_contexts[cv_id],
                    )
            continue

        for cv_id, fraction in enumerate(coverage.density_fraction):
            if fraction <= EPS_PARAM:
                continue
            owner_key = (int(cv_id), str(mechanism.category), str(mechanism.instance_name))
            previous = active_density_owner.get(owner_key)
            if previous is not None:
                previous_type, previous_rule = previous
                raise ValueError(
                    f"{mechanism.category.title()} {mechanism.instance_name!r} ({mechanism.class_name}) "
                    f"is painted more than once on CV {cv_id}; declarations {previous_rule} and "
                    f"{rule_index} overlap after discretization. Combine their Region expressions, "
                    "use a finer CV policy, or use distinct mechanism names."
                )
            active_density_owner[owner_key] = (str(mechanism.class_name), int(rule_index))
            _apply_density(
                buckets[cv_id],
                mechanism,
                region_key=rule.region,
                fraction=fraction,
            )

    seen_names: set[str] = set()
    placement_id = 0
    for rule in place_rules:
        if rule.aligned:
            assert isinstance(rule.locset, LocsetBatch)
            assert rule.population_indices is not None
            location_groups = zip(
                (cache.points(rule.locset[index]) for index in range(len(rule.locset))),
                ((int(index),) for index in rule.population_indices),
            )
        else:
            assert not isinstance(rule.locset, LocsetBatch)
            population_indices = (
                (None,) if rule.population_indices is None else tuple(int(index) for index in rule.population_indices)
            )
            location_groups = ((cache.points(rule.locset), population_indices),)

        for locations, population_indices in location_groups:
            for branch_id, x, display_name in locations:
                ids = geometry.cv_ids(branch_id)
                if not ids:
                    continue
                cv_id = geometry.locate_cv(branch_id=branch_id, x=x)
                geo = geos[cv_id]
                position = _position_for_geo(geo, x=float(x))
                for population_index in population_indices:
                    for mechanism in rule.mechanisms:
                        _apply_place(
                            buckets[cv_id],
                            mechanism,
                            display_name=display_name,
                            placement_id=placement_id,
                            branch_id=branch_id,
                            branch_x=x,
                            seen_names=seen_names,
                            position=position,
                            population_index=population_index,
                        )
                        placement_id += 1

    return buckets
