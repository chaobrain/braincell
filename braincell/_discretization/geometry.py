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

"""Geometry helpers for static CV discretization.

This module turns branch-wise normalized CV bounds into validated,
geometry-rich CV fragments. The resulting objects are still declaration-
time data: they carry geometric and topological facts, but no runtime
state or instantiated mechanism objects.
"""

from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology

EPS_PARAM = 1e-9
EPS_LEN_UM = 1e-6
EPS_AREA_UM2 = 1e-9

__all__ = [
    "CVGeometryResult",
    "EPS_AREA_UM2",
    "EPS_LEN_UM",
    "EPS_PARAM",
    "_Frustum",
    "_GeoCV",
    "_build_frusta",
    "_lateral_area_um2",
    "build_cv_geometry",
    "locate_cv_on_branch",
    "validate_bounds",
    "validate_connectivity",
    "validate_morphology",
]


@dataclass(frozen=True)
class _Frustum:
    """One frustum-like geometry slice clipped from a branch segment.

    Attributes
    ----------
    prox : float
        Proximal normalized branch coordinate of this slice.
    dist : float
        Distal normalized branch coordinate of this slice.
    length_um : float
        Physical slice length in micrometers.
    r_prox_um : float
        Proximal radius in micrometers.
    r_dist_um : float
        Distal radius in micrometers.
    point_prox_um : numpy.ndarray or None
        Optional proximal 3-D point in micrometers, when source branch
        coordinates are available.
    point_dist_um : numpy.ndarray or None
        Optional distal 3-D point in micrometers, when source branch
        coordinates are available.
    """

    prox: float
    dist: float
    length_um: float
    r_prox_um: float
    r_dist_um: float
    point_prox_um: "np.ndarray | None"
    point_dist_um: "np.ndarray | None"


@dataclass(frozen=True)
class _GeoCV:
    """Geometry-first intermediate record for one control volume.

    Attributes
    ----------
    id : int
        Stable CV id.
    branch_id : int
        Owning morphology branch id.
    branch_type : str
        Owning branch type.
    prox : float
        Proximal normalized branch coordinate.
    dist : float
        Distal normalized branch coordinate.
    midpoint : float
        Midpoint coordinate in normalized branch coordinates.
    parent_cv : int or None
        Parent CV id, or ``None`` for the root CV.
    children_cv : tuple of int
        Child CV ids.
    length_um : float
        CV cable length in micrometers.
    lateral_area_um2 : float
        CV membrane lateral area in square micrometers.
    axial_factor_total_per_cm : float
        End-to-end axial-resistance geometric factor.
    axial_factor_prox_per_cm : float
        Geometric factor from midpoint to proximal side.
    axial_factor_dist_per_cm : float
        Geometric factor from midpoint to distal side.
    r_prox_um : float
        Proximal radius in micrometers.
    r_mid_um : float
        Midpoint radius in micrometers.
    diam_arc_mean_um : float
        Arc-length-weighted mean diameter in micrometers.
    r_dist_um : float
        Distal radius in micrometers.
    """

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
    diam_arc_mean_um: float
    r_dist_um: float


@dataclass(frozen=True)
class CVGeometryResult:
    """Validated CV geometry payload consumed by build and mechanism lowering.

    Attributes
    ----------
    geos : tuple of _GeoCV
        Finalized geometry records in CV id order.
    branch_to_cv_ids : tuple of tuple of int
        For each morphology branch, the ordered CV ids that tile that
        branch.
    """

    geos: tuple[_GeoCV, ...]
    branch_to_cv_ids: tuple[tuple[int, ...], ...]

    def cv_ids(self, branch_id: int) -> tuple[int, ...]:
        """Return ordered CV ids on one morphology branch.

        Parameters
        ----------
        branch_id : int
            Morphology branch id.

        Returns
        -------
        tuple of int
            CV ids tiling that branch.
        """
        return self.branch_to_cv_ids[int(branch_id)]

    def locate_cv(self, *, branch_id: int, x: float) -> int:
        """Locate the CV that owns one normalized branch coordinate.

        Parameters
        ----------
        branch_id : int
            Morphology branch id.
        x : float
            Normalized branch coordinate.

        Returns
        -------
        int
            Id of the owning CV.
        """
        return locate_cv_on_branch(
            self.cv_ids(int(branch_id)),
            self.geos,
            x=float(x),
        )


def _owns_zero_length_jump(
    *,
    position_um: float,
    start_um: float,
    end_um: float,
) -> bool:
    """Return whether a zero-length jump belongs to ``[start_um, end_um]``."""
    if position_um < start_um - EPS_LEN_UM or position_um > end_um + EPS_LEN_UM:
        return False
    if position_um <= start_um + EPS_LEN_UM:
        return start_um <= EPS_LEN_UM
    return position_um <= end_um + EPS_LEN_UM


def _build_frusta(
    branch: Branch,
    *,
    prox: float,
    dist: float,
) -> tuple[_Frustum, ...]:
    """Clip ``branch`` to normalized ``[prox, dist]`` and return frustum slices."""
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
            r_seg_prox = float(radii_prox_um[seg_idx])
            r_seg_dist = float(radii_dist_um[seg_idx])
            if np.isclose(r_seg_prox, r_seg_dist):
                continue
            if not _owns_zero_length_jump(
                position_um=seg_start_um,
                start_um=start_um,
                end_um=end_um,
            ):
                continue
            x_jump = max(prox_f, min(dist_f, seg_start_um / total_length_um))
            point_jump = None
            if points_proximal is not None and points_distal is not None:
                point_jump = points_distal[seg_idx]
            frusta.append(
                _Frustum(
                    prox=float(x_jump),
                    dist=float(x_jump),
                    length_um=0.0,
                    r_prox_um=r_seg_prox,
                    r_dist_um=r_seg_dist,
                    point_prox_um=point_jump,
                    point_dist_um=point_jump,
                )
            )
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


def _arc_weighted_mean_diam_um(frusta: tuple[_Frustum, ...]) -> float:
    """Return the arc-length-weighted mean diameter across ``frusta``."""
    total_length_um = sum(piece.length_um for piece in frusta)
    if total_length_um <= EPS_LEN_UM:
        return frusta[0].r_prox_um + frusta[-1].r_dist_um
    weighted = 0.0
    for piece in frusta:
        weighted += (piece.r_prox_um + piece.r_dist_um) * piece.length_um
    return weighted / total_length_um


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
                    prox=p0,
                    dist=x,
                    length_um=length_left,
                    r_prox_um=piece.r_prox_um,
                    r_dist_um=r_mid,
                    point_prox_um=piece.point_prox_um,
                    point_dist_um=point_mid,
                )
            )
        if length_right > EPS_LEN_UM:
            right.append(
                _Frustum(
                    prox=x,
                    dist=p1,
                    length_um=length_right,
                    r_prox_um=r_mid,
                    r_dist_um=piece.r_dist_um,
                    point_prox_um=point_mid,
                    point_dist_um=piece.point_dist_um,
                )
            )
    return tuple(left), tuple(right)


def validate_morphology(morpho: Morphology) -> None:
    """Validate geometry preconditions required by CV discretization.

    Parameters
    ----------
    morpho : Morphology
        Morphology to validate.

    Raises
    ------
    TypeError
        If ``morpho`` is not a :class:`Morphology`.
    ValueError
        If any branch has non-positive total length or non-positive
        radii.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"Expected Morphology, got {type(morpho).__name__!s}.")
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


def validate_bounds(
    bounds_by_branch: tuple[tuple[tuple[float, float], ...], ...],
    morpho: Morphology,
) -> None:
    """Validate branch-wise CV bounds before geometry assembly.

    Parameters
    ----------
    bounds_by_branch : tuple of tuple of tuple of float
        Normalized ``(prox, dist)`` intervals for each branch.
    morpho : Morphology
        Morphology whose branches are being tiled.

    Raises
    ------
    ValueError
        If any branch has missing coverage, overlaps, gaps, invalid
        interval ordering, or a branch-count mismatch.
    """
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


def validate_connectivity(
    geos: tuple[_GeoCV, ...],
    branch_to_cv_ids: tuple[tuple[int, ...], ...],
    morpho: Morphology,
) -> None:
    """Validate parent/child connectivity for assembled geometry records.

    Parameters
    ----------
    geos : tuple of _GeoCV
        Geometry records to validate.
    branch_to_cv_ids : tuple of tuple of int
        Per-branch CV id tilings.
    morpho : Morphology
        Source morphology used to define root and branch attachments.

    Raises
    ------
    ValueError
        If CV ids are out of range, parent/child relations are
        inconsistent, or a cycle is detected.
    """
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
    for branch_id, ids in enumerate(branch_to_cv_ids):
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            if geos[right_id].parent_cv != left_id:
                raise ValueError(
                    f"CV {right_id} on branch {branch_id} expects parent "
                    f"{left_id}, got {geos[right_id].parent_cv}."
                )
    root_ids = branch_to_cv_ids[morpho.root.index]
    if geos[root_ids[0]].parent_cv is not None:
        raise ValueError(
            f"Root-branch first CV {root_ids[0]} must have parent_cv=None, "
            f"got {geos[root_ids[0]].parent_cv}."
        )
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


def locate_cv_on_branch(
    ids: tuple[int, ...],
    geos: list[_GeoCV] | tuple[_GeoCV, ...],
    *,
    x: float,
) -> int:
    """Return the CV id that owns one branch coordinate.

    Parameters
    ----------
    ids : tuple of int
        Ordered CV ids on one branch.
    geos : sequence of _GeoCV
        Geometry records indexed by CV id.
    x : float
        Normalized branch coordinate.

    Returns
    -------
    int
        Owning CV id.

    Raises
    ------
    ValueError
        If ``x`` falls in no CV interval.
    """
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


def build_cv_geometry(
    morpho: Morphology,
    bounds_by_branch: tuple[tuple[tuple[float, float], ...], ...],
) -> CVGeometryResult:
    """Build validated geometry records for all control volumes.

    Parameters
    ----------
    morpho : Morphology
        Morphology to discretize.
    bounds_by_branch : tuple of tuple of tuple of float
        Normalized ``(prox, dist)`` bounds for each branch.

    Returns
    -------
    CVGeometryResult
        Finalized geometry payload used by later mechanism and
        discretization assembly stages.

    Notes
    -----
    This function is the geometry-stage entry point. It validates the
    morphology and bounds, clips branch segments into frusta, computes
    per-CV geometric summaries, and resolves parent/child connectivity
    across both intra-branch tilings and inter-branch attachments.
    """
    validate_morphology(morpho)
    validate_bounds(bounds_by_branch, morpho)

    branch_to_cv_ids_lists: list[tuple[int, ...]] = []
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
                    diam_arc_mean_um=_arc_weighted_mean_diam_um(frusta),
                    r_dist_um=r_dist,
                )
            )
            parent_by_cv.append(None)
            children_by_cv.append([])
            ids.append(cv_id)
            cv_id += 1
        branch_to_cv_ids_lists.append(tuple(ids))

    branch_to_cv_ids = tuple(branch_to_cv_ids_lists)

    for ids in branch_to_cv_ids:
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            parent_by_cv[right_id] = left_id
            children_by_cv[left_id].append(right_id)

    for edge in morpho.edges:
        parent_ids = branch_to_cv_ids[edge.parent.index]
        child_ids = branch_to_cv_ids[edge.child.index]
        parent_cv = locate_cv_on_branch(parent_ids, geos, x=float(edge.parent_x))
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
            diam_arc_mean_um=geo.diam_arc_mean_um,
            r_dist_um=geo.r_dist_um,
        )
        for geo in geos
    )
    validate_connectivity(finalized, branch_to_cv_ids, morpho)
    return CVGeometryResult(
        geos=finalized,
        branch_to_cv_ids=branch_to_cv_ids,
    )
