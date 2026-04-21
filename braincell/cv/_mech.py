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


from dataclasses import dataclass, replace

import brainunit as u
import numpy as np

from braincell.filter import AllRegion, LocsetExpr, RegionExpr
from braincell.mech import (
    CableProperty,
    CurrentProbe,
    Density,
    MechanismProbe,
    Point,
    StateProbe,
)
from braincell.morph._morphology import Morphology
from ._geo import CVGeo, EPSILON, interval_lateral_area, map_point_to_cv

_DEFAULT_CABLE = CableProperty(
    resting_potential=-65.0 * u.mV,
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
)


# This module has one job: translate frontend ``paint`` / ``place``
# declarations into per-CV mechanism payloads during ``Cell`` rebuild. The
# results are still declaration-layer data; runtime lowering happens later in
# ``runtime.py``.


@dataclass(frozen=True)
class PaintRule:
    """Normalized ``Cell.paint(...)`` declaration.

    Each rule binds one :class:`RegionExpr` to either cable properties or a
    density mechanism. ``Cell`` stores these rules in declaration order and
    applies them during lazy rebuild. Cable rules overwrite midpoint cable
    values, while density rules append scaled mechanisms based on CV coverage.
    """
    region: RegionExpr
    mechanism: CableProperty | Density


@dataclass(frozen=True)
class PlaceRule:
    """Normalized ``Cell.place(...)`` declaration.

    A place rule maps one :class:`LocsetExpr` to a tuple of point mechanisms.
    The current cell frontend only supports ``site='mid'``, so evaluated points
    are attached to the midpoint of the CV selected by :func:`map_point_to_cv`.
    """
    locset: LocsetExpr
    mechanisms: tuple[Point, ...]
    site: str = "mid"


@dataclass
class CVMech:
    """Mutable mechanism accumulator used only during ``Cell`` rebuild.

    ``CVMech`` starts from default cable properties, then ``apply_paint_rules``
    and ``apply_place_rules`` mutate it in place for one CV. After all rules are
    applied, :func:`assemble_cv` freezes the result into the immutable
    user-facing :class:`CV`.
    """
    cm: object
    ra: object
    v: object
    temp: u.Quantity[u.kelvin]
    density_mech: list[Density]
    point_mech: list[Point]


def default_paint_rules() -> tuple[PaintRule, ...]:
    return (
        PaintRule(region=AllRegion(), mechanism=_DEFAULT_CABLE),
    )


def init_cv_mech(n_cv: int) -> list[CVMech]:
    return [
        CVMech(
            cm=_DEFAULT_CABLE.membrane_capacitance,
            ra=_DEFAULT_CABLE.axial_resistivity,
            v=_DEFAULT_CABLE.resting_potential,
            temp=_DEFAULT_CABLE.temperature,
            density_mech=[],
            point_mech=[],
        )
        for _ in range(n_cv)
    ]


def normalize_paint_rules(
    region: RegionExpr, mechanisms: tuple[object, ...]
) -> tuple[PaintRule, ...]:
    # Rule normalization keeps the public ``Cell.paint(...)`` surface flexible
    # while making later rebuild code operate on one predictable internal shape.
    if not isinstance(region, RegionExpr):
        raise TypeError(
            f"Cell.paint(...) expects RegionExpr, got {type(region).__name__!s}."
        )
    if len(mechanisms) == 0:
        raise ValueError("Cell.paint(...) expects at least one mechanism.")

    new_rules: list[PaintRule] = []
    for mechanism in mechanisms:
        if isinstance(mechanism, CableProperty):
            new_rules.append(PaintRule(region=region, mechanism=mechanism))
            continue
        if isinstance(mechanism, Density):
            new_rules.append(PaintRule(region=region, mechanism=mechanism))
            continue
        raise TypeError(
            "Cell.paint(...) mechanisms must be CableProperty or "
            "Density (use braincell.mech.Channel / Ion), "
            f"got {type(mechanism).__name__!s}."
        )
    return tuple(new_rules)


def merge_paint_rules(
    existing: tuple[PaintRule, ...],
    incoming: tuple[PaintRule, ...],
) -> tuple[PaintRule, ...]:
    # Cable-property rules overwrite previous cable defaults on the same region,
    # while density mechanisms accumulate in declaration order.
    merged = list(existing)
    for rule in incoming:
        if isinstance(rule.mechanism, CableProperty):
            merged = [
                item
                for item in merged
                if not (
                    isinstance(item.mechanism, CableProperty)
                    and item.region == rule.region
                )
            ]
        merged.append(rule)
    return tuple(merged)


def normalize_place_rule(
    locset: LocsetExpr, mechanisms: tuple[object, ...]
) -> PlaceRule:
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


def apply_paint_rules(
    morpho: Morphology,
    *,
    cvs: tuple[CVGeo, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    paint_rules: tuple[PaintRule, ...],
    mechs: list[CVMech],
) -> None:
    # Paint rules are applied onto mutable ``CVMech`` buckets so coverage logic
    # and cable overwrites can be resolved before the final immutable ``CV`` is built.
    for rule in paint_rules:
        mask = rule.region.evaluate(morpho)
        intervals_by_branch = _group_intervals_by_branch(mask.intervals)
        mechanism = rule.mechanism

        for branch_id, cv_ids in cv_ids_by_branch.items():
            intervals = intervals_by_branch.get(branch_id, ())
            if len(intervals) == 0:
                continue

            for cv_id in cv_ids:
                cv_geo = cvs[cv_id]
                mech = mechs[cv_id]

                if isinstance(mechanism, CableProperty):
                    if not _contains_coord(cv_geo.midpoint, intervals, epsilon=EPSILON):
                        continue
                    mech.cm = mechanism.membrane_capacitance
                    mech.ra = mechanism.axial_resistivity
                    mech.v = mechanism.resting_potential
                    mech.temp = mechanism.temperature
                    continue

                if isinstance(mechanism, Density):
                    area_fraction = _coverage_area_fraction(
                        morpho,
                        cv_geo=cv_geo,
                        intervals=intervals,
                    )
                    if area_fraction <= EPSILON:
                        continue
                    scaled = _scale_density_for_coverage(
                        mechanism,
                        area_fraction=area_fraction,
                    )
                    mech.density_mech.append(scaled)
                    continue

                raise TypeError(
                    f"Unsupported paint mechanism type {type(mechanism).__name__!s}."
                )


def apply_place_rules(
    morpho: Morphology,
    *,
    cvs: tuple[CVGeo, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    place_rules: tuple[PlaceRule, ...],
    mechs: list[CVMech],
) -> None:
    # Place rules do not instantiate runtime nodes here. They only attach point
    # mechanism declarations to the CV whose midpoint owns the requested location.
    for rule in place_rules:
        if rule.site != "mid":
            raise ValueError(f"Unsupported place site {rule.site!r}; only 'mid' is allowed.")
        mask = rule.locset.evaluate(morpho)
        for point, display_name in zip(mask.points, mask.display_names):
            cv_id = map_point_to_cv(
                point,
                cvs=cvs,
                cv_ids_by_branch=cv_ids_by_branch,
                epsilon=EPSILON,
            )
            if cv_id is None:
                continue
            mechs[cv_id].point_mech.extend(
                _resolve_point_mechanism_name(mechanism, display_name=display_name)
                for mechanism in rule.mechanisms
            )


def _contains_coord(
    x: float,
    intervals: tuple[tuple[float, float], ...],
    *,
    epsilon: float = EPSILON,
) -> bool:
    for left, right in intervals:
        if left - epsilon <= x <= right + epsilon:
            return True
    return False


def _group_intervals_by_branch(
    intervals: tuple[tuple[int, float, float], ...],
) -> dict[int, tuple[tuple[float, float], ...]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for branch, prox, dist in intervals:
        grouped.setdefault(int(branch), []).append((float(prox), float(dist)))
    return {branch: tuple(ranges) for branch, ranges in grouped.items()}


def _resolve_point_mechanism_name(mechanism: Point, *, display_name: str) -> Point:
    if isinstance(mechanism, StateProbe):
        return mechanism if mechanism.name is not None else replace(
            mechanism,
            name=f"{display_name}_{mechanism.field}",
        )
    if isinstance(mechanism, MechanismProbe):
        return mechanism if mechanism.name is not None else replace(
            mechanism,
            name=f"{display_name}_{mechanism.mechanism}_{mechanism.field}",
        )
    if isinstance(mechanism, CurrentProbe):
        suffix = (
            f"{mechanism.mechanism}_current"
            if mechanism.mechanism is not None else f"{mechanism.ion}_current"
        )
        return mechanism if mechanism.name is not None else replace(
            mechanism,
            name=f"{display_name}_{suffix}",
        )
    return mechanism


def _coverage_area_fraction(
    morpho: Morphology,
    *,
    cv_geo: CVGeo,
    intervals: tuple[tuple[float, float], ...],
) -> float:
    total_area_um2 = float(np.asarray(cv_geo.lateral_area.to_decimal(u.um ** 2), dtype=float))
    if total_area_um2 <= EPSILON:
        return 0.0

    branch = morpho.branches[cv_geo.branch_id]
    overlap_area_um2 = 0.0
    for left, right in intervals:
        start = max(cv_geo.prox, float(left))
        end = min(cv_geo.dist, float(right))
        if end - start <= EPSILON:
            continue
        overlap_area = interval_lateral_area(branch, prox=start, dist=end)
        overlap_area_um2 += float(np.asarray(overlap_area.to_decimal(u.um ** 2), dtype=float))

    return max(0.0, min(1.0, overlap_area_um2 / total_area_um2))


def _scale_density_for_coverage(
    mechanism: Density,
    *,
    area_fraction: float,
) -> Density:
    """Scale a partial-CV channel's conductance by its coverage fraction.

    The scaling only applies to ``"channel"`` category mechanisms. Ions
    are not conductance-bearing so they are returned unchanged.

    When the channel exposes a ``g_max`` parameter, we multiply it in
    place and leave ``coverage_area_fraction`` at 1.0. Otherwise we
    stash the fraction in :attr:`Density.coverage_area_fraction`
    so later passes can consume it. Either way the fraction is tracked
    as a first-class field, not as a pseudo-parameter smuggled through
    ``params``.
    """
    if mechanism.category != "channel":
        return mechanism
    if area_fraction >= 1.0 - EPSILON:
        return mechanism
    if "g_max" in mechanism.params:
        try:
            scaled = mechanism.params["g_max"] * float(area_fraction)
        except Exception:
            return mechanism.with_coverage(area_fraction)
        return mechanism.with_params(g_max=scaled)
    return mechanism.with_coverage(area_fraction)
