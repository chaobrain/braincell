from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from braincell._units import u
from braincell.filter import AllRegion, LocsetExpr, RegionExpr
from braincell.mech import (
    CableProperties,
    CurrentClamp,
    DensityMechanism,
    GapJunctionMechanism,
    ProbeMechanism,
    SynapseMechanism,
)
from braincell.morpho import Morpho
from .cv_geo import CVGeo, EPSILON, interval_lateral_area, map_point_to_cv

PointMechanismRuntime = (
    SynapseMechanism,
    GapJunctionMechanism,
    CurrentClamp,
    ProbeMechanism,
)

_DEFAULT_CABLE = CableProperties(
    resting_potential=-65.0 * u.mV,
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
)


@dataclass(frozen=True)
class PaintRule:
    region: RegionExpr
    mechanism: CableProperties | DensityMechanism


@dataclass(frozen=True)
class PlaceRule:
    locset: LocsetExpr
    mechanisms: tuple[object, ...]
    site: str = "mid"


@dataclass
class CVMech:
    cm: object
    ra: object
    v: object
    temp: u.Quantity[u.kelvin]
    density_mech: list[DensityMechanism]
    point_mech: list[object]


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


def normalize_paint_rules(region: RegionExpr, mechanisms: tuple[object, ...]) -> tuple[PaintRule, ...]:
    if not isinstance(region, RegionExpr):
        raise TypeError(f"Cell.paint(...) expects RegionExpr, got {type(region).__name__!s}.")
    if len(mechanisms) == 0:
        raise ValueError("Cell.paint(...) expects at least one mechanism.")

    new_rules: list[PaintRule] = []
    for mechanism in mechanisms:
        if isinstance(mechanism, CableProperties):
            new_rules.append(
                PaintRule(
                    region=region,
                    mechanism=_normalize_cable_properties(mechanism),
                )
            )
            continue
        if isinstance(mechanism, DensityMechanism):
            new_rules.append(PaintRule(region=region, mechanism=mechanism))
            continue
        raise TypeError(
            "Cell.paint(...) mechanisms must be CableProperties or DensityMechanism, "
            f"got {type(mechanism).__name__!s}."
        )
    return tuple(new_rules)


def merge_paint_rules(
    existing: tuple[PaintRule, ...],
    incoming: tuple[PaintRule, ...],
) -> tuple[PaintRule, ...]:
    merged = list(existing)
    for rule in incoming:
        if isinstance(rule.mechanism, CableProperties):
            merged = [
                item
                for item in merged
                if not (
                    isinstance(item.mechanism, CableProperties)
                    and item.region == rule.region
                )
            ]
        merged.append(rule)
    return tuple(merged)


def normalize_place_rule(locset: LocsetExpr, mechanisms: tuple[object, ...]) -> PlaceRule:
    if not isinstance(locset, LocsetExpr):
        raise TypeError(f"Cell.place(...) expects LocsetExpr, got {type(locset).__name__!s}.")
    if len(mechanisms) == 0:
        raise ValueError("Cell.place(...) expects at least one point mechanism.")

    normalized: list[object] = []
    for mechanism in mechanisms:
        if not isinstance(mechanism, PointMechanismRuntime):
            raise TypeError(
                "Cell.place(...) mechanisms must be point mechanisms, "
                f"got {type(mechanism).__name__!s}."
            )
        normalized.append(mechanism)
    return PlaceRule(locset=locset, mechanisms=tuple(normalized), site="mid")


def apply_paint_rules(
    morpho: Morpho,
    *,
    cvs: tuple[CVGeo, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    paint_rules: tuple[PaintRule, ...],
    mechs: list[CVMech],
) -> None:
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

                if isinstance(mechanism, CableProperties):
                    if not _contains_coord(cv_geo.midpoint, intervals, epsilon=EPSILON):
                        continue
                    mech.cm = mechanism.membrane_capacitance
                    mech.ra = mechanism.axial_resistivity
                    mech.v = mechanism.resting_potential
                    mech.temp = mechanism.temperature
                    continue

                if isinstance(mechanism, DensityMechanism):
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

                raise TypeError(f"Unsupported paint mechanism type {type(mechanism).__name__!s}.")


def apply_place_rules(
    morpho: Morpho,
    *,
    cvs: tuple[CVGeo, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    place_rules: tuple[PlaceRule, ...],
    mechs: list[CVMech],
) -> None:
    for rule in place_rules:
        if rule.site != "mid":
            raise ValueError(f"Unsupported place site {rule.site!r}; only 'mid' is allowed.")
        points = rule.locset.evaluate(morpho).points
        for point in points:
            cv_id = map_point_to_cv(
                point,
                cvs=cvs,
                cv_ids_by_branch=cv_ids_by_branch,
                epsilon=EPSILON,
            )
            if cv_id is None:
                continue
            mechs[cv_id].point_mech.extend(rule.mechanisms)


def _normalize_cable_properties(mechanism: CableProperties) -> CableProperties:
    return CableProperties(
        resting_potential=mechanism.resting_potential,
        membrane_capacitance=mechanism.membrane_capacitance,
        axial_resistivity=mechanism.axial_resistivity,
        temperature=_coerce_temperature(
            mechanism.temperature,
            name="CableProperties.temperature",
        ),
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


def _coverage_area_fraction(
    morpho: Morpho,
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
    mechanism: DensityMechanism,
    *,
    area_fraction: float,
) -> DensityMechanism:
    if mechanism.channel_type is None:
        return mechanism
    if area_fraction >= 1.0 - EPSILON:
        return mechanism

    updated: list[tuple[str, Any]] = []
    scaled_gmax = False
    for key, value in mechanism.params:
        if key == "g_max":
            scaled_gmax = True
            try:
                updated.append((key, value * float(area_fraction)))
            except Exception:
                updated.append((key, value))
                updated.append(("coverage_area_fraction", float(area_fraction)))
            continue
        updated.append((key, value))

    if not scaled_gmax:
        updated.append(("coverage_area_fraction", float(area_fraction)))

    return DensityMechanism(
        ion_type=mechanism.ion_type,
        channel_type=mechanism.channel_type,
        params=tuple(updated),
    )


def _coerce_temperature(value: object, *, name: str) -> u.Quantity[u.kelvin]:
    if not hasattr(value, "to_decimal") or not callable(getattr(value, "to_decimal")):
        raise TypeError(f"{name} must be a temperature Quantity, got {value!r}.")
    decimal = np.asarray(value.to_decimal(u.kelvin), dtype=float)
    if decimal.ndim != 0:
        raise TypeError(f"{name} must be scalar temperature Quantity, got shape {decimal.shape!r}.")
    return u.Quantity(float(decimal), u.kelvin)
