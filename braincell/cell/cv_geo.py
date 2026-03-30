from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from braincell._units import u
from braincell.morpho import Branch, Morpho

Quantity = Any
EPSILON = 1e-12


@dataclass(frozen=True)
class CVFrustum:
    prox: float
    dist: float
    length: Quantity
    radius_prox: Quantity
    radius_dist: Quantity
    point_prox: Quantity | None
    point_dist: Quantity | None


@dataclass(frozen=True)
class CVGeo:
    id: int
    branch_id: int
    branch_type: str
    prox: float
    dist: float
    midpoint: float
    parent_cv: int | None
    children_cv: tuple[int, ...]
    length: Quantity
    lateral_area: Quantity
    frusta: tuple[CVFrustum, ...]
    axial_factor_total: Quantity
    axial_factor_prox: Quantity
    axial_factor_dist: Quantity


def build_cv_geo(
    morpho: Morpho,
    *,
    policy: object,
) -> tuple[tuple[CVGeo, ...], dict[int, tuple[int, ...]]]:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_cv_geo(...) expects Morpho, got {type(morpho).__name__!s}.")
    n_by_branch = _resolve_cv_counts(morpho, policy=policy)
    cv_ids_by_branch: dict[int, tuple[int, ...]] = {}
    temp_geo: list[CVGeo] = []

    cv_id = 0
    for branch_id, branch in enumerate(morpho.branches):
        n_per_branch = n_by_branch[branch_id]
        ids: list[int] = []
        for offset in range(n_per_branch):
            prox = float(offset) / float(n_per_branch)
            dist = float(offset + 1) / float(n_per_branch)
            midpoint = 0.5 * (prox + dist)
            (
                frusta,
                length,
                lateral_area,
                axial_factor_total,
                axial_factor_prox,
                axial_factor_dist,
            ) = _build_geo_fields(branch, prox=prox, dist=dist, midpoint=midpoint)
            temp_geo.append(
                CVGeo(
                    id=cv_id,
                    branch_id=branch_id,
                    branch_type=branch.type,
                    prox=prox,
                    dist=dist,
                    midpoint=midpoint,
                    parent_cv=None,
                    children_cv=(),
                    length=length,
                    lateral_area=lateral_area,
                    frusta=frusta,
                    axial_factor_total=axial_factor_total,
                    axial_factor_prox=axial_factor_prox,
                    axial_factor_dist=axial_factor_dist,
                )
            )
            ids.append(cv_id)
            cv_id += 1
        cv_ids_by_branch[branch_id] = tuple(ids)

    parent_by_cv: list[int | None] = [None for _ in temp_geo]
    children_by_cv: list[list[int]] = [[] for _ in temp_geo]

    # Chain CVs on each branch.
    for ids in cv_ids_by_branch.values():
        for left, right in zip(ids[:-1], ids[1:]):
            parent_by_cv[right] = left
            children_by_cv[left].append(right)

    # Attach branch roots to their parent branch CV.
    for edge in morpho.edges:
        parent_ids = cv_ids_by_branch[edge.parent.index]
        child_ids = cv_ids_by_branch[edge.child.index]
        parent_cv = _locate_branch_cv_by_x(
            parent_ids,
            temp_geo,
            x=edge.parent_x,
            epsilon=EPSILON,
        )
        child_cv = _locate_branch_cv_by_x(
            child_ids,
            temp_geo,
            x=edge.child_x,
            epsilon=EPSILON,
        )
        if parent_by_cv[child_cv] is None:
            parent_by_cv[child_cv] = parent_cv
        if child_cv not in children_by_cv[parent_cv]:
            children_by_cv[parent_cv].append(child_cv)

    cvs = tuple(
        CVGeo(
            id=cv.id,
            branch_id=cv.branch_id,
            branch_type=cv.branch_type,
            prox=cv.prox,
            dist=cv.dist,
            midpoint=cv.midpoint,
            parent_cv=parent_by_cv[cv.id],
            children_cv=tuple(sorted(children_by_cv[cv.id])),
            length=cv.length,
            lateral_area=cv.lateral_area,
            frusta=cv.frusta,
            axial_factor_total=cv.axial_factor_total,
            axial_factor_prox=cv.axial_factor_prox,
            axial_factor_dist=cv.axial_factor_dist,
        )
        for cv in temp_geo
    )
    return cvs, cv_ids_by_branch


def _resolve_cv_counts(morpho: Morpho, *, policy: object) -> tuple[int, ...]:
    mode = getattr(policy, "mode", None)
    if not isinstance(mode, str):
        raise TypeError(
            "build_cv_geo(...) expects a CVPolicy-like object with string field 'mode'."
        )
    if mode == "cv_per_branch":
        return _resolve_cv_per_branch_counts(morpho, policy=policy)
    if mode == "max_cv_len":
        return _resolve_max_cv_len_counts(morpho, policy=policy)
    raise ValueError(
        f"Unsupported cv policy mode {mode!r}. Supported modes are 'cv_per_branch' and 'max_cv_len'."
    )


def _resolve_cv_per_branch_counts(morpho: Morpho, *, policy: object) -> tuple[int, ...]:
    cv_per_branch = getattr(policy, "cv_per_branch", None)
    if isinstance(cv_per_branch, bool) or not isinstance(cv_per_branch, int):
        raise TypeError(f"cv_per_branch must be integer, got {cv_per_branch!r}.")
    if cv_per_branch <= 0:
        raise ValueError(f"cv_per_branch must be > 0, got {cv_per_branch!r}.")
    n_per_branch = int(cv_per_branch)
    return tuple(n_per_branch for _ in morpho.branches)


def _resolve_max_cv_len_counts(morpho: Morpho, *, policy: object) -> tuple[int, ...]:
    max_cv_len = getattr(policy, "max_cv_len", None)
    if not hasattr(max_cv_len, "to_decimal"):
        raise TypeError(f"max_cv_len must be a length Quantity, got {max_cv_len!r}.")
    try:
        max_len_um = float(np.asarray(max_cv_len.to_decimal(u.um), dtype=float))
    except Exception as exc:  # pragma: no cover - defensive for foreign quantity types
        raise TypeError(f"max_cv_len must be a length Quantity, got {max_cv_len!r}.") from exc
    if not np.isfinite(max_len_um) or max_len_um <= 0.0:
        raise ValueError(f"max_cv_len must be > 0, got {max_cv_len!r}.")

    return tuple(
        _strict_count_from_max_len_um(branch=branch, max_len_um=max_len_um)
        for branch in morpho.branches
    )


def _strict_count_from_max_len_um(*, branch: Branch, max_len_um: float) -> int:
    branch_len_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    ratio = branch_len_um / max_len_um
    # Keep CV length <= max_len_um while damping tiny floating point overshoot near integers.
    n_cv = int(np.ceil(ratio - EPSILON))
    return max(1, n_cv)


def map_point_to_cv(
    point: tuple[int, float],
    *,
    cvs: tuple[CVGeo, ...],
    cv_ids_by_branch: dict[int, tuple[int, ...]],
    epsilon: float = EPSILON,
) -> int | None:
    branch_id, x = point
    ids = cv_ids_by_branch.get(int(branch_id))
    if not ids:
        return None
    return _locate_branch_cv_by_x(ids, cvs, x=float(x), epsilon=epsilon)


def axial_resistance_from_factor(ra: Quantity, *, factor: Quantity) -> Quantity:
    ra_ohm_cm = float(np.asarray(ra.to_decimal(u.ohm * u.cm), dtype=float))
    factor_per_cm = float(np.asarray(factor.to_decimal(u.cm ** -1), dtype=float))
    return u.Quantity(ra_ohm_cm * factor_per_cm, u.ohm)


def interval_lateral_area(
    branch: Branch,
    *,
    prox: float,
    dist: float,
    epsilon: float = EPSILON,
) -> Quantity:
    frusta = _build_frusta(branch, prox=float(prox), dist=float(dist), epsilon=epsilon)
    return u.Quantity(_frusta_lateral_area(frusta), u.um ** 2)


def _build_geo_fields(
    branch: Branch,
    *,
    prox: float,
    dist: float,
    midpoint: float,
    epsilon: float = EPSILON,
) -> tuple[
    tuple[CVFrustum, ...],
    Quantity,
    Quantity,
    Quantity,
    Quantity,
    Quantity,
]:
    frusta = _build_frusta(branch, prox=prox, dist=dist, epsilon=epsilon)
    area = _frusta_lateral_area(frusta)
    prox_frusta, dist_frusta = _split_frusta(frusta, x=midpoint, epsilon=epsilon)
    return (
        frusta,
        u.Quantity(_frusta_total_length_um(frusta), u.um),
        u.Quantity(area, u.um ** 2),
        u.Quantity(_frusta_axial_factor_per_cm(frusta), u.cm ** -1),
        u.Quantity(_frusta_axial_factor_per_cm(prox_frusta), u.cm ** -1),
        u.Quantity(_frusta_axial_factor_per_cm(dist_frusta), u.cm ** -1),
    )


def _build_frusta(
    branch: Branch,
    *,
    prox: float,
    dist: float,
    epsilon: float,
) -> tuple[CVFrustum, ...]:
    if not (0.0 <= float(prox) < float(dist) <= 1.0):
        raise ValueError(f"CV bounds must satisfy 0 <= prox < dist <= 1, got {(prox, dist)!r}.")

    lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
    radii_prox_um = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
    radii_dist_um = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
    total_length_um = float(np.sum(lengths_um))
    if total_length_um <= epsilon:
        return ()

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

    start_um = prox * total_length_um
    end_um = dist * total_length_um
    segment_starts_um = np.concatenate(([0.0], np.cumsum(lengths_um)[:-1]))
    segment_ends_um = segment_starts_um + lengths_um

    frusta: list[CVFrustum] = []
    for seg_idx, seg_length_um in enumerate(lengths_um):
        seg_start_um = float(segment_starts_um[seg_idx])
        seg_end_um = float(segment_ends_um[seg_idx])

        if seg_length_um <= epsilon:
            boundary_x = 1.0 if total_length_um <= epsilon else seg_start_um / total_length_um
            if not _interval_owns_boundary(prox=prox, dist=dist, x=boundary_x, epsilon=epsilon):
                continue

            point = None
            if points_proximal is not None and points_distal is not None:
                point = u.Quantity(points_proximal[seg_idx], u.um)

            frusta.append(
                CVFrustum(
                    prox=float(boundary_x),
                    dist=float(boundary_x),
                    length=u.Quantity(0.0, u.um),
                    radius_prox=u.Quantity(float(radii_prox_um[seg_idx]), u.um),
                    radius_dist=u.Quantity(float(radii_dist_um[seg_idx]), u.um),
                    point_prox=point,
                    point_dist=point,
                )
            )
            continue

        left_um = max(seg_start_um, start_um)
        right_um = min(seg_end_um, end_um)
        if right_um - left_um <= epsilon:
            continue

        t0 = (left_um - seg_start_um) / float(seg_length_um)
        t1 = (right_um - seg_start_um) / float(seg_length_um)
        r_seg_prox = float(radii_prox_um[seg_idx])
        r_seg_dist = float(radii_dist_um[seg_idx])
        r0_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t0
        r1_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t1

        x0 = max(prox, min(dist, left_um / total_length_um))
        x1 = max(prox, min(dist, right_um / total_length_um))
        point0 = None
        point1 = None
        if points_proximal is not None and points_distal is not None:
            seg_p0 = points_proximal[seg_idx]
            seg_p1 = points_distal[seg_idx]
            point0 = u.Quantity(seg_p0 + (seg_p1 - seg_p0) * t0, u.um)
            point1 = u.Quantity(seg_p0 + (seg_p1 - seg_p0) * t1, u.um)

        frusta.append(
            CVFrustum(
                prox=float(x0),
                dist=float(x1),
                length=u.Quantity(float(right_um - left_um), u.um),
                radius_prox=u.Quantity(r0_um, u.um),
                radius_dist=u.Quantity(r1_um, u.um),
                point_prox=point0,
                point_dist=point1,
            )
        )
    return tuple(frusta)


def _interval_owns_boundary(*, prox: float, dist: float, x: float, epsilon: float) -> bool:
    if x < prox - epsilon:
        return False
    if x < dist - epsilon:
        return True
    return dist >= 1.0 - epsilon and x <= dist + epsilon


def _frusta_total_length_um(frusta: tuple[CVFrustum, ...]) -> float:
    return float(
        sum(float(np.asarray(piece.length.to_decimal(u.um), dtype=float)) for piece in frusta)
    )


def _frusta_lateral_area(frusta: tuple[CVFrustum, ...]) -> float:
    total_area_um2 = 0.0
    for piece in frusta:
        length_um = float(np.asarray(piece.length.to_decimal(u.um), dtype=float))
        r0_um = float(np.asarray(piece.radius_prox.to_decimal(u.um), dtype=float))
        r1_um = float(np.asarray(piece.radius_dist.to_decimal(u.um), dtype=float))
        slant_um = float(np.sqrt(length_um * length_um + (r1_um - r0_um) ** 2))
        total_area_um2 += float(np.pi * (r0_um + r1_um) * slant_um)
    return total_area_um2


def _frusta_axial_factor_per_cm(frusta: tuple[CVFrustum, ...]) -> float:
    factor_per_cm = 0.0
    for piece in frusta:
        length_cm = float(np.asarray(piece.length.to_decimal(u.cm), dtype=float))
        r0_cm = float(np.asarray(piece.radius_prox.to_decimal(u.cm), dtype=float))
        r1_cm = float(np.asarray(piece.radius_dist.to_decimal(u.cm), dtype=float))
        if r0_cm <= 0.0 or r1_cm <= 0.0:
            return float(np.inf)
        factor_per_cm += length_cm / (float(np.pi) * r0_cm * r1_cm)
    return factor_per_cm


def _split_frusta(
    frusta: tuple[CVFrustum, ...],
    *,
    x: float,
    epsilon: float,
) -> tuple[tuple[CVFrustum, ...], tuple[CVFrustum, ...]]:
    left: list[CVFrustum] = []
    right: list[CVFrustum] = []
    for piece in frusta:
        p0 = float(piece.prox)
        p1 = float(piece.dist)
        if p1 <= x + epsilon:
            left.append(piece)
            continue
        if p0 >= x - epsilon:
            right.append(piece)
            continue
        if p1 - p0 <= epsilon:
            continue

        ratio = (x - p0) / (p1 - p0)
        ratio = max(0.0, min(1.0, ratio))

        piece_length_um = float(np.asarray(piece.length.to_decimal(u.um), dtype=float))
        r0_um = float(np.asarray(piece.radius_prox.to_decimal(u.um), dtype=float))
        r1_um = float(np.asarray(piece.radius_dist.to_decimal(u.um), dtype=float))
        r_mid_um = r0_um + (r1_um - r0_um) * ratio

        left_len_um = piece_length_um * ratio
        right_len_um = piece_length_um - left_len_um

        point_mid = None
        if piece.point_prox is not None and piece.point_dist is not None:
            p_prox = np.asarray(piece.point_prox.to_decimal(u.um), dtype=float)
            p_dist = np.asarray(piece.point_dist.to_decimal(u.um), dtype=float)
            point_mid = u.Quantity(p_prox + (p_dist - p_prox) * ratio, u.um)

        if left_len_um > epsilon:
            left.append(
                CVFrustum(
                    prox=p0,
                    dist=float(x),
                    length=u.Quantity(left_len_um, u.um),
                    radius_prox=u.Quantity(r0_um, u.um),
                    radius_dist=u.Quantity(r_mid_um, u.um),
                    point_prox=piece.point_prox,
                    point_dist=point_mid,
                )
            )
        if right_len_um > epsilon:
            right.append(
                CVFrustum(
                    prox=float(x),
                    dist=p1,
                    length=u.Quantity(right_len_um, u.um),
                    radius_prox=u.Quantity(r_mid_um, u.um),
                    radius_dist=u.Quantity(r1_um, u.um),
                    point_prox=point_mid,
                    point_dist=piece.point_dist,
                )
            )
    return tuple(left), tuple(right)


def _locate_branch_cv_by_x(
    ids: tuple[int, ...],
    cvs: tuple[CVGeo, ...] | list[CVGeo],
    *,
    x: float,
    epsilon: float,
) -> int:
    if x <= 0.0 + epsilon:
        return ids[0]
    if x >= 1.0 - epsilon:
        return ids[-1]

    # Right-side ownership on boundaries: x == dist belongs to the next CV.
    for cv_id in ids:
        cv = cvs[cv_id]
        if x >= cv.prox - epsilon and x < cv.dist - epsilon:
            return cv_id
    return ids[-1]
