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

"""Pure-functional control-volume lowering.

One entry point: :func:`lower`. All helpers in this module are pure
functions operating on immutable or locally-scoped data. Internal
types ``_Frustum``, ``_GeoCV``, ``_MechBucket`` never leave this
module — the final ``tuple[CV, ...]`` is the only output.
"""

from dataclasses import dataclass, replace
from typing import Literal, TYPE_CHECKING

import brainunit as u
import numpy as np

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
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology
from .policy import CVPolicy

if TYPE_CHECKING:
    from .base import CV

__all__ = [
    "PaintRule",
    "PlaceRule",
    "default_paint_rules",
    "normalize_paint_rules",
    "normalize_place_rule",
    "merge_paint_rules",
    "merge_place_rules",
    "lower",
]

EPS_PARAM = 1e-9         # normalized x ∈ [0, 1]
EPS_LEN_UM = 1e-6        # μm lengths
EPS_AREA_UM2 = 1e-9      # μm² areas

_DEFAULT_CABLE = CableProperty(
    resting_potential=-65.0 * u.mV,
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
)


@dataclass(frozen=True)
class PaintRule:
    """Normalized ``Cell.paint(...)`` declaration.

    Binds one :class:`RegionExpr` to either :class:`CableProperty` or
    :class:`Density`. Stored in declaration order by ``Cell``.
    """

    region: RegionExpr
    mechanism: CableProperty | Density


@dataclass(frozen=True)
class PlaceRule:
    """Normalized ``Cell.place(...)`` declaration.

    Maps one :class:`LocsetExpr` to a tuple of point mechanisms. Only
    ``site='mid'`` is currently supported.
    """

    locset: LocsetExpr
    mechanisms: tuple[Point, ...]
    site: Literal["mid"] = "mid"


@dataclass(frozen=True)
class _Frustum:
    prox: float
    dist: float
    length_um: float
    r_prox_um: float
    r_dist_um: float
    point_prox_um: "np.ndarray | None"
    point_dist_um: "np.ndarray | None"


@dataclass(frozen=True)
class _GeoCV:
    id: int
    branch_id: int
    branch_type: str
    prox: float
    dist: float
    midpoint: float
    parent_cv: int | None
    children_cv: tuple[int, ...]
    length_um: float
    lateral_area_um2: float
    axial_factor_total_per_cm: float
    axial_factor_prox_per_cm: float
    axial_factor_dist_per_cm: float
    r_prox_um: float
    r_mid_um: float
    r_dist_um: float


@dataclass
class _MechBucket:
    cable: CableProperty
    density_by_key: dict[tuple[int, str, object], Density]
    points: list[Point]


# =============================================================================
# Rule normalization / merging
# =============================================================================


def default_paint_rules() -> tuple[PaintRule, ...]:
    return (PaintRule(region=AllRegion(), mechanism=_DEFAULT_CABLE),)


def normalize_paint_rules(
    region: RegionExpr,
    mechanisms: tuple[object, ...],
) -> tuple[PaintRule, ...]:
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
    """Key for last-wins dedup.

    Cable rules collide per region. Density rules collide only when
    the *user-facing name* also matches — so two channels of the same
    class painted with distinct ``name=`` attrs stay distinct.
    """
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
    """Append incoming paint rules with last-wins dedup.

    Two rules collide when they share the same region AND the same kind:
    - CableProperty on the same region → incoming replaces existing.
    - Density with the same ``class_name`` on the same region → incoming
      replaces existing.

    Rules targeting different regions, or two Density rules for different
    channel classes, all survive.
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
    """Append incoming place rules, dropping exact duplicates."""
    merged: list[PlaceRule] = list(existing)
    for rule in incoming:
        if rule in merged:
            continue
        merged.append(rule)
    return tuple(merged)


# =============================================================================
# Region / locset cache
# =============================================================================


class _RegionCache:
    """Per-build cache of region / locset evaluation outputs.

    Keyed by ``id(expr)`` so even non-hashable exprs work. Sharing a
    single :class:`SelectionCache` for morphology-derived intermediates
    (distance-to-root etc.) across all lookups.
    """

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


# =============================================================================
# Frustum math
# =============================================================================


def _build_frusta(
    branch: Branch,
    *,
    prox: float,
    dist: float,
) -> tuple[_Frustum, ...]:
    """Clip ``branch`` to normalized ``[prox, dist]`` and return frustum slices.

    Preserves segment breaks from the morphology, linearly interpolates radii
    and 3D point geometry at clip boundaries, and drops zero-length interior
    segments. Raises ``ValueError`` for invalid bounds, non-positive radii, or
    branches of zero length.
    """
    prox_f = float(prox)
    dist_f = float(dist)
    if not (0.0 - EPS_PARAM <= prox_f < dist_f - EPS_PARAM and dist_f <= 1.0 + EPS_PARAM):
        raise ValueError(
            f"CV bounds must satisfy 0 <= prox < dist <= 1, got {(prox, dist)!r}."
        )
    prox_f = max(0.0, min(1.0, prox_f))
    dist_f = max(0.0, min(1.0, dist_f))

    lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
    radii_prox_um = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
    radii_dist_um = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)

    if np.any(radii_prox_um <= 0.0) or np.any(radii_dist_um <= 0.0):
        raise ValueError(
            f"All branch radii must be > 0 (branch type={branch.type!r})."
        )

    total_length_um = float(np.sum(lengths_um))
    if total_length_um <= EPS_LEN_UM:
        raise ValueError(
            f"Branch total length must be > {EPS_LEN_UM} μm "
            f"(got {total_length_um} μm, type={branch.type!r})."
        )

    points_proximal = (
        np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float)
        if branch.points_proximal is not None
        else None
    )
    points_distal = (
        np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)
        if branch.points_distal is not None
        else None
    )

    start_um = prox_f * total_length_um
    end_um = dist_f * total_length_um
    segment_starts_um = np.concatenate(([0.0], np.cumsum(lengths_um)[:-1]))
    segment_ends_um = segment_starts_um + lengths_um

    frusta: list[_Frustum] = []
    for seg_idx, seg_length_um in enumerate(lengths_um):
        seg_start_um = float(segment_starts_um[seg_idx])
        seg_end_um = float(segment_ends_um[seg_idx])
        if seg_length_um <= EPS_LEN_UM:
            continue

        left_um = max(seg_start_um, start_um)
        right_um = min(seg_end_um, end_um)
        if right_um - left_um <= EPS_LEN_UM:
            continue

        t0 = (left_um - seg_start_um) / float(seg_length_um)
        t1 = (right_um - seg_start_um) / float(seg_length_um)
        r_seg_prox = float(radii_prox_um[seg_idx])
        r_seg_dist = float(radii_dist_um[seg_idx])
        r0_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t0
        r1_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t1

        x0 = max(prox_f, min(dist_f, left_um / total_length_um))
        x1 = max(prox_f, min(dist_f, right_um / total_length_um))
        point0 = None
        point1 = None
        if points_proximal is not None and points_distal is not None:
            p_prox = points_proximal[seg_idx]
            p_dist = points_distal[seg_idx]
            point0 = p_prox + (p_dist - p_prox) * t0
            point1 = p_prox + (p_dist - p_prox) * t1

        frusta.append(
            _Frustum(
                prox=float(x0),
                dist=float(x1),
                length_um=float(right_um - left_um),
                r_prox_um=r0_um,
                r_dist_um=r1_um,
                point_prox_um=point0,
                point_dist_um=point1,
            )
        )

    if len(frusta) == 0:
        raise ValueError(
            f"CV [{prox}, {dist}] produced no frusta on branch of length "
            f"{total_length_um} μm."
        )
    return tuple(frusta)


def _boundary_radii_um(frusta: tuple[_Frustum, ...]) -> tuple[float, float]:
    if len(frusta) == 0:
        raise ValueError("Cannot resolve boundary radii from empty frusta.")
    return frusta[0].r_prox_um, frusta[-1].r_dist_um


def _lateral_area_um2(frusta: tuple[_Frustum, ...]) -> float:
    total = 0.0
    pi = float(np.pi)
    for piece in frusta:
        slant = float(
            np.sqrt(piece.length_um ** 2 + (piece.r_dist_um - piece.r_prox_um) ** 2)
        )
        total += pi * (piece.r_prox_um + piece.r_dist_um) * slant
    return total


def _axial_factor_per_cm(frusta: tuple[_Frustum, ...]) -> float:
    """Return Σ L_cm / (π · r_prox_cm · r_dist_cm) across the frusta."""
    factor = 0.0
    pi = float(np.pi)
    for piece in frusta:
        length_cm = piece.length_um * 1e-4
        r0_cm = piece.r_prox_um * 1e-4
        r1_cm = piece.r_dist_um * 1e-4
        if r0_cm <= 0.0 or r1_cm <= 0.0:
            raise ValueError(
                "Axial factor requires strictly positive radii; validation slipped."
            )
        factor += length_cm / (pi * r0_cm * r1_cm)
    return factor


def _midpoint_radius_um(frusta: tuple[_Frustum, ...]) -> float:
    if len(frusta) == 0:
        raise ValueError("Cannot resolve midpoint radius from empty frusta.")
    total_length_um = sum(piece.length_um for piece in frusta)
    if total_length_um <= EPS_LEN_UM:
        return 0.5 * (frusta[0].r_prox_um + frusta[-1].r_dist_um)
    target = 0.5 * total_length_um
    walked = 0.0
    for piece in frusta:
        next_walked = walked + piece.length_um
        if next_walked >= target - EPS_LEN_UM:
            if piece.length_um <= EPS_LEN_UM:
                return 0.5 * (piece.r_prox_um + piece.r_dist_um)
            ratio = max(0.0, min(1.0, (target - walked) / piece.length_um))
            return piece.r_prox_um + (piece.r_dist_um - piece.r_prox_um) * ratio
        walked = next_walked
    return frusta[-1].r_dist_um


def _split_frusta(
    frusta: tuple[_Frustum, ...],
    *,
    x: float,
) -> tuple[tuple[_Frustum, ...], tuple[_Frustum, ...]]:
    left: list[_Frustum] = []
    right: list[_Frustum] = []
    for piece in frusta:
        p0 = piece.prox
        p1 = piece.dist
        if p1 <= x + EPS_PARAM:
            left.append(piece)
            continue
        if p0 >= x - EPS_PARAM:
            right.append(piece)
            continue
        span = p1 - p0
        if span <= EPS_PARAM:
            continue
        ratio = max(0.0, min(1.0, (x - p0) / span))

        length_left = piece.length_um * ratio
        length_right = piece.length_um - length_left
        r_mid = piece.r_prox_um + (piece.r_dist_um - piece.r_prox_um) * ratio
        point_mid = None
        if piece.point_prox_um is not None and piece.point_dist_um is not None:
            point_mid = piece.point_prox_um + (piece.point_dist_um - piece.point_prox_um) * ratio

        if length_left > EPS_LEN_UM:
            left.append(
                _Frustum(
                    prox=p0, dist=x, length_um=length_left,
                    r_prox_um=piece.r_prox_um, r_dist_um=r_mid,
                    point_prox_um=piece.point_prox_um, point_dist_um=point_mid,
                )
            )
        if length_right > EPS_LEN_UM:
            right.append(
                _Frustum(
                    prox=x, dist=p1, length_um=length_right,
                    r_prox_um=r_mid, r_dist_um=piece.r_dist_um,
                    point_prox_um=point_mid, point_dist_um=piece.point_dist_um,
                )
            )
    return tuple(left), tuple(right)


# =============================================================================
# Validators
# =============================================================================


def _validate_morpho(morpho: Morphology) -> None:
    if not isinstance(morpho, Morphology):
        raise TypeError(
            f"Expected Morphology, got {type(morpho).__name__!s}."
        )
    for branch_id, branch in enumerate(morpho.branches):
        lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
        if float(np.sum(lengths_um)) <= EPS_LEN_UM:
            raise ValueError(
                f"Branch {branch_id} (type={branch.type!r}) has total length "
                f"<= {EPS_LEN_UM} μm; morphology rejected."
            )
        radii_prox = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
        radii_dist = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
        if np.any(radii_prox <= 0.0) or np.any(radii_dist <= 0.0):
            raise ValueError(
                f"Branch {branch_id} (type={branch.type!r}) has non-positive "
                "radii; morphology rejected."
            )


def _validate_bounds(
    bounds_by_branch: tuple[tuple[tuple[float, float], ...], ...],
    morpho: Morphology,
) -> None:
    if len(bounds_by_branch) != len(morpho.branches):
        raise ValueError(
            f"CV bounds length {len(bounds_by_branch)} does not match "
            f"branch count {len(morpho.branches)}."
        )
    for branch_id, branch_bounds in enumerate(bounds_by_branch):
        if len(branch_bounds) == 0:
            raise ValueError(f"Branch {branch_id} has no CV bounds.")
        prev_dist = 0.0
        for i, (prox, dist) in enumerate(branch_bounds):
            prox_f = float(prox)
            dist_f = float(dist)
            if not (
                0.0 - EPS_PARAM <= prox_f < dist_f - EPS_PARAM
                and dist_f <= 1.0 + EPS_PARAM
            ):
                raise ValueError(
                    f"Branch {branch_id} CV {i} has invalid bounds "
                    f"(prox={prox_f}, dist={dist_f}); must satisfy "
                    "0 <= prox < dist <= 1."
                )
            if i == 0 and abs(prox_f - 0.0) > EPS_PARAM:
                raise ValueError(
                    f"Branch {branch_id} first CV must start at 0.0, got {prox_f}."
                )
            if i > 0 and abs(prox_f - prev_dist) > EPS_PARAM:
                raise ValueError(
                    f"Branch {branch_id} CV {i} prox={prox_f} does not meet "
                    f"previous dist={prev_dist} (overlap or gap)."
                )
            prev_dist = dist_f
        if abs(prev_dist - 1.0) > EPS_PARAM:
            raise ValueError(
                f"Branch {branch_id} last CV must end at 1.0, got {prev_dist}."
            )


def _validate_connectivity(
    geos: tuple[_GeoCV, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    morpho: Morphology,
) -> None:
    n = len(geos)
    for geo in geos:
        if geo.parent_cv is not None and not (0 <= geo.parent_cv < n):
            raise ValueError(
                f"CV {geo.id} has out-of-range parent_cv {geo.parent_cv}."
            )
        for child in geo.children_cv:
            if not (0 <= child < n):
                raise ValueError(
                    f"CV {geo.id} has out-of-range child_cv {child}."
                )
    for branch_id, ids in cv_ids_by_branch.items():
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            if geos[right_id].parent_cv != left_id:
                raise ValueError(
                    f"CV {right_id} on branch {branch_id} expects parent "
                    f"{left_id}, got {geos[right_id].parent_cv}."
                )
    root_ids = cv_ids_by_branch[morpho.root.index]
    if geos[root_ids[0]].parent_cv is not None:
        raise ValueError(
            f"Root-branch first CV {root_ids[0]} must have parent_cv=None, "
            f"got {geos[root_ids[0]].parent_cv}."
        )
    # DFS cycle detection (follows parent pointers).
    visited: set[int] = set()
    for start in range(n):
        if start in visited:
            continue
        path: list[int] = []
        path_set: set[int] = set()
        cursor: int | None = start
        while cursor is not None and cursor not in visited:
            if cursor in path_set:
                raise ValueError(f"Cycle detected through CV {cursor}.")
            path_set.add(cursor)
            path.append(cursor)
            cursor = geos[cursor].parent_cv
        visited.update(path)


# =============================================================================
# Geometry build
# =============================================================================


def _locate_cv_on_branch(
    ids: tuple[int, ...],
    geos: list[_GeoCV] | tuple[_GeoCV, ...],
    *,
    x: float,
) -> int:
    if x <= 0.0 + EPS_PARAM:
        return ids[0]
    if x >= 1.0 - EPS_PARAM:
        return ids[-1]
    for cv_id in ids:
        geo = geos[cv_id]
        if geo.prox - EPS_PARAM <= x < geo.dist - EPS_PARAM:
            return cv_id
    for cv_id in ids:
        geo = geos[cv_id]
        if abs(x - geo.dist) <= EPS_PARAM:
            return cv_id
    raise ValueError(
        f"x={x!r} not owned by any CV in branch; "
        f"bounds are {[(geos[i].prox, geos[i].dist) for i in ids]!r}."
    )


def _build_geo(
    morpho: Morphology,
    bounds_by_branch: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[_GeoCV, ...], dict[int, tuple[int, ...]]]:
    cv_ids_by_branch: dict[int, tuple[int, ...]] = {}
    geos: list[_GeoCV] = []
    parent_by_cv: list[int | None] = []
    children_by_cv: list[list[int]] = []

    cv_id = 0
    for branch_id, branch in enumerate(morpho.branches):
        branch_bounds = bounds_by_branch[branch_id]
        ids: list[int] = []
        for prox, dist in branch_bounds:
            prox_f = float(prox)
            dist_f = float(dist)
            frusta = _build_frusta(branch, prox=prox_f, dist=dist_f)
            length_um = sum(p.length_um for p in frusta)
            area_um2 = _lateral_area_um2(frusta)
            factor_total = _axial_factor_per_cm(frusta)
            midpoint = 0.5 * (prox_f + dist_f)
            left, right = _split_frusta(frusta, x=midpoint)
            factor_prox = _axial_factor_per_cm(left) if left else 0.0
            factor_dist = _axial_factor_per_cm(right) if right else 0.0
            r_prox, r_dist = _boundary_radii_um(frusta)
            r_mid = _midpoint_radius_um(frusta)

            geos.append(
                _GeoCV(
                    id=cv_id,
                    branch_id=branch_id,
                    branch_type=branch.type,
                    prox=prox_f,
                    dist=dist_f,
                    midpoint=midpoint,
                    parent_cv=None,
                    children_cv=(),
                    length_um=length_um,
                    lateral_area_um2=area_um2,
                    axial_factor_total_per_cm=factor_total,
                    axial_factor_prox_per_cm=factor_prox,
                    axial_factor_dist_per_cm=factor_dist,
                    r_prox_um=r_prox,
                    r_mid_um=r_mid,
                    r_dist_um=r_dist,
                )
            )
            parent_by_cv.append(None)
            children_by_cv.append([])
            ids.append(cv_id)
            cv_id += 1
        cv_ids_by_branch[branch_id] = tuple(ids)

    # Chain CVs within each branch.
    for ids in cv_ids_by_branch.values():
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            parent_by_cv[right_id] = left_id
            children_by_cv[left_id].append(right_id)

    # Wire branch-root CVs to parent branches in traversal order.
    for edge in morpho.edges:
        parent_ids = cv_ids_by_branch[edge.parent.index]
        child_ids = cv_ids_by_branch[edge.child.index]
        parent_cv = _locate_cv_on_branch(parent_ids, geos, x=float(edge.parent_x))
        child_cv = child_ids[0]
        if parent_by_cv[child_cv] is None:
            parent_by_cv[child_cv] = parent_cv
        if child_cv not in children_by_cv[parent_cv]:
            children_by_cv[parent_cv].append(child_cv)

    finalized = tuple(
        _GeoCV(
            id=geo.id,
            branch_id=geo.branch_id,
            branch_type=geo.branch_type,
            prox=geo.prox,
            dist=geo.dist,
            midpoint=geo.midpoint,
            parent_cv=parent_by_cv[geo.id],
            children_cv=tuple(children_by_cv[geo.id]),
            length_um=geo.length_um,
            lateral_area_um2=geo.lateral_area_um2,
            axial_factor_total_per_cm=geo.axial_factor_total_per_cm,
            axial_factor_prox_per_cm=geo.axial_factor_prox_per_cm,
            axial_factor_dist_per_cm=geo.axial_factor_dist_per_cm,
            r_prox_um=geo.r_prox_um,
            r_mid_um=geo.r_mid_um,
            r_dist_um=geo.r_dist_um,
        )
        for geo in geos
    )
    return finalized, cv_ids_by_branch


# =============================================================================
# Mechanism lowering
# =============================================================================


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
) -> None:
    named = _resolve_point_name(mechanism, display_name=display_name)
    # Only enforce uniqueness on auto-generated names — explicit user-supplied
    # names pass through so the runtime layer can surface duplicates later.
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


def _init_bucket() -> _MechBucket:
    return _MechBucket(
        cable=_DEFAULT_CABLE,
        density_by_key={},
        points=[],
    )


def _build_mech(
    morpho: Morphology,
    geos: tuple[_GeoCV, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    *,
    paint_rules: tuple[PaintRule, ...],
    place_rules: tuple[PlaceRule, ...],
    cache: _RegionCache,
) -> list[_MechBucket]:
    buckets = [_init_bucket() for _ in geos]

    frusta_cache: dict = {}

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

        for branch_id, cv_ids in cv_ids_by_branch.items():
            intervals = intervals_by_branch.get(branch_id, ())
            if len(intervals) == 0:
                continue
            for cv_id in cv_ids:
                geo = geos[cv_id]
                bucket = buckets[cv_id]

                if isinstance(mechanism, CableProperty):
                    if not _interval_contains(
                        intervals, geo.midpoint, epsilon=EPS_PARAM
                    ):
                        continue
                    bucket.cable = mechanism
                    continue

                fraction = _coverage_fraction(
                    morpho, geo, intervals,
                    frusta_builder=_cached_frusta,
                )
                if fraction <= EPS_PARAM:
                    continue
                _apply_density(
                    bucket, mechanism,
                    region_key=rule.region, fraction=fraction,
                )

    seen_names: set[str] = set()
    for rule in place_rules:
        if rule.site != "mid":
            raise ValueError(
                f"Unsupported place site {rule.site!r}; only 'mid' is allowed."
            )
        for branch_id, x, display_name in cache.points(rule.locset):
            ids = cv_ids_by_branch.get(branch_id)
            if not ids:
                continue
            cv_id = _locate_cv_on_branch(ids, geos, x=x)
            for mechanism in rule.mechanisms:
                _apply_place(
                    buckets[cv_id], mechanism,
                    display_name=display_name, seen_names=seen_names,
                )

    return buckets


def _validate_names(buckets: list[_MechBucket]) -> None:
    """No-op placeholder; duplicate user-named points surface at runtime."""
    return None


# =============================================================================
# Composer + assembly
# =============================================================================


def _assemble(geo: _GeoCV, bucket: _MechBucket) -> "CV":
    from .base import CV  # local import: base.py is created in Phase 6
    cable = bucket.cable
    ra = cable.axial_resistivity
    ra_ohm_cm = float(np.asarray(ra.to_decimal(u.ohm * u.cm), dtype=float))
    return CV(
        id=geo.id,
        branch_id=geo.branch_id,
        branch_type=geo.branch_type,
        prox=geo.prox,
        dist=geo.dist,
        parent_cv=geo.parent_cv,
        children_cv=geo.children_cv,
        length=u.Quantity(geo.length_um, u.um),
        area=u.Quantity(geo.lateral_area_um2, u.um ** 2),
        cm=cable.membrane_capacitance,
        ra=cable.axial_resistivity,
        v=cable.resting_potential,
        temp=cable.temperature,
        r_axial=u.Quantity(ra_ohm_cm * geo.axial_factor_total_per_cm, u.ohm),
        r_axial_prox=u.Quantity(ra_ohm_cm * geo.axial_factor_prox_per_cm, u.ohm),
        r_axial_dist=u.Quantity(ra_ohm_cm * geo.axial_factor_dist_per_cm, u.ohm),
        radius_prox=u.Quantity(geo.r_prox_um, u.um),
        radius_mid=u.Quantity(geo.r_mid_um, u.um),
        radius_dist=u.Quantity(geo.r_dist_um, u.um),
        density_mech=tuple(bucket.density_by_key.values()),
        point_mech=tuple(bucket.points),
    )


def lower(
    morpho: Morphology,
    *,
    policy: CVPolicy,
    paint_rules: tuple[PaintRule, ...],
    place_rules: tuple[PlaceRule, ...],
) -> tuple["CV", ...]:
    """Lower a morphology + policy + rules into a frozen ``tuple[CV, ...]``.

    Pipeline:
    1. Validate ``morpho``.
    2. Resolve CV bounds from ``policy``; validate them.
    3. Build per-branch geometry (``_build_geo``) and connectivity.
    4. Evaluate paint and place rules against the geometry
       (``_build_mech``), caching region/locset lookups.
    5. Run a final global name-uniqueness check.
    6. Freeze into ``CV`` records (``_assemble``).
    """
    if not isinstance(policy, CVPolicy):
        raise TypeError(
            f"lower(...) expects a CVPolicy, got {type(policy).__name__!s}."
        )
    _validate_morpho(morpho)
    cache = _RegionCache(morpho)
    bounds = policy.resolve_cv_bounds(morpho, paint_rules=paint_rules)
    _validate_bounds(bounds, morpho)
    geos, ids_by_branch = _build_geo(morpho, bounds)
    _validate_connectivity(geos, ids_by_branch, morpho)
    buckets = _build_mech(
        morpho, geos, ids_by_branch,
        paint_rules=paint_rules, place_rules=place_rules, cache=cache,
    )
    _validate_names(buckets)
    return tuple(_assemble(g, b) for g, b in zip(geos, buckets))
