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


from numbers import Integral, Real

import brainunit as u
import numpy as np

from braincell._misc import normalize_param, scalar_decimal
from braincell.morph.morphology import MorphoBranch, Morphology

Interval = tuple[int, float, float]
Location = tuple[int, float]
EPSILON = 1e-12
#: Relative slack in StepSamples' step count, so a step that divides the
#: total length up to rounding (0.3 um / 0.1 um = 2.9999999999999996) still
#: takes its last step.
STEP_COUNT_SLACK = 1e-9
_BRANCH_METRIC_PROPERTIES = {"length", "mean_radius", "diam_arc_mean", "area", "volume"}
_UNSUPPORTED_BRANCH_METRIC_PROPERTIES = {"max_radius", "min_radius"}

__all__ = [
    "EPSILON",
    "branch_slice_intervals",
    "coerce_positive_count",
    "branch_in_intervals",
    "branch_range_intervals",
    "normalize_region_intervals",
    "union_region_intervals",
    "intersect_region_intervals",
    "difference_region_intervals",
    "complement_region_intervals",
    "fork_points_locations",
    "terminal_locations",
    "uniform_samples_from_region",
    "random_samples_from_region",
    "normalize_locset_points",
    "unique_locset_points",
    "concat_locset_points",
    "union_locset_points",
    "intersect_locset_points",
    "difference_locset_points",
]


def _is_quantity(value: object) -> bool:
    return hasattr(value, "to_decimal") and callable(getattr(value, "to_decimal"))


def _coerce_values(name: str, value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)):
        values = (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = (value,)
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")
    return values


def _broadcast_values(name: str, values: tuple[object, ...], size: int) -> tuple[object, ...]:
    if len(values) == size:
        return values
    if len(values) == 1:
        return values * size
    raise ValueError(f"{name} has length {len(values)!r}, which cannot be broadcast to {size!r}.")


def _coerce_property_name(property_name: object) -> str:
    if not isinstance(property_name, str) or not property_name:
        raise TypeError("property must be a non-empty string.")
    return property_name


def _coerce_branch_index(value: object, n_branches: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"branch_index values must be integers, got {value!r}.")
    index = int(value)
    if not 0 <= index < n_branches:
        raise IndexError(f"Branch index {index!r} is out of range [0, {n_branches}).")
    return index


def _coerce_norm_coord(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} values must be real numbers, got {value!r}.")
    return float(value)


def _coerce_quantity_scalar(value: object, *, property_name: str) -> object:
    if not _is_quantity(value):
        raise TypeError(f"{property_name} is not quantity-like.")
    mantissa = np.asarray(u.get_mantissa(value), dtype=float)
    if mantissa.ndim == 0:
        return value
    if mantissa.ndim == 1 and mantissa.size == 1:
        return u.Quantity(mantissa.reshape(()), u.get_unit(value))
    raise TypeError(f"{property_name} must be scalar-valued, got shape {mantissa.shape!r}.")


def _coerce_filterable_scalar(property_name: str, value: object) -> object:
    if _is_quantity(value):
        return _coerce_quantity_scalar(value, property_name=property_name)
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"Property {property_name!r} is not scalar-filterable; got {type(value).__name__!s}.")


def _resolve_branch_property(branch_view: MorphoBranch, property_name: str) -> object:
    # ``MorphoBranch.parent_id`` is the parent's *node id* and is ``None`` at
    # the root, while a region filter wants the parent *branch index* with -1
    # as the root sentinel. That mismatch is why this one name cannot fall
    # through to the generic ``getattr`` path below; every other branch fact
    # this layer used to re-derive is now published by ``MorphoBranch``.
    if property_name == "parent_id":
        parent = branch_view.parent
        return -1 if parent is None else parent.index
    if property_name in _BRANCH_METRIC_PROPERTIES:
        return getattr(branch_view, property_name)
    if property_name in _UNSUPPORTED_BRANCH_METRIC_PROPERTIES:
        raise ValueError(f"Branch property {property_name!r} is not supported in this version.")
    if hasattr(branch_view, property_name):
        return _coerce_filterable_scalar(property_name, getattr(branch_view, property_name))
    raise ValueError(f"Unknown branch property {property_name!r}.")


def _to_decimal_scalar(value: object, *, unit: object, name: str) -> float:
    scalar = _coerce_quantity_scalar(value, property_name=name)
    return scalar_decimal(scalar, unit)


def _parse_closed(closed: object) -> str:
    allowed = {"both", "left", "right", "neither"}
    if not isinstance(closed, str) or closed not in allowed:
        raise ValueError(f"closed must be one of {sorted(allowed)!r}, got {closed!r}.")
    return closed


def _value_in_range(value: float, lo: float | None, hi: float | None, closed: str) -> bool:
    """Whether ``value`` lies within ``[lo, hi]``, each end open or closed per ``closed``; ``None`` is unbounded."""
    if lo is not None and not (value >= lo if _left_closed(closed) else value > lo):
        return False
    if hi is not None and not (value <= hi if _right_closed(closed) else value < hi):
        return False
    return True


def _linear_piece_select(
    x0: float,
    x1: float,
    f0: float,
    f1: float,
    lo: float | None,
    hi: float | None,
    closed: str,
) -> tuple[float, float] | None:
    """The part of ``[x0, x1]`` where ``f``, linear from ``f0`` to ``f1``, lies within the bounds.

    Returns ``(start, end)`` or ``None``. Where ``f`` varies, a bound is met
    at a single point, so ``closed`` matters only when ``f0 == f1``: the
    whole piece is then in or out, decided by :func:`_value_in_range`. A
    piece that touches the range at one point is ``None``.
    """
    if f0 == f1:
        return (x0, x1) if _value_in_range(f0, lo, hi, closed) else None
    slope = f1 - f0
    t_start, t_end = 0.0, 1.0
    for bound, is_lower in ((lo, True), (hi, False)):
        if bound is None:
            continue
        t = (bound - f0) / slope
        # Rising f enters the range at a lower bound and leaves at an upper one.
        if (slope > 0.0) == is_lower:
            t_start = max(t_start, t)
        else:
            t_end = min(t_end, t)
    if t_end <= t_start:
        return None
    return (x0 + t_start * (x1 - x0), x0 + t_end * (x1 - x0))


def _quadratic_sublevel(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    t_lo: np.ndarray | float,
    t_hi: np.ndarray | float,
    *,
    strict: bool,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise ``{t in [t_lo, t_hi] : a t^2 + b t + c <= 0}`` (``< 0`` if ``strict``), for ``a >= 0``.

    Returns ``(lo, hi)``; the set is empty where ``hi <= lo``. A
    coefficient smaller than ``1e-12 * scale`` counts as zero, so a nearly
    flat quadratic is solved as linear and a nearly flat line as constant.
    Roots use ``q = -(b + sign(b) sqrt(b^2 - 4ac)) / 2``, ``q / a`` and
    ``c / q``, which avoids cancelling the two terms of the usual formula.
    """
    tiny = 1e-12 * scale
    lo = np.full(a.shape, np.inf)
    hi = np.full(a.shape, -np.inf)
    quadratic = a > tiny
    disc = b * b - 4.0 * a * c
    real = quadratic & (disc >= 0.0)
    root = np.sqrt(np.where(real, disc, 0.0))
    q = -0.5 * (b + np.where(b >= 0.0, root, -root))
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = q / np.where(quadratic, a, 1.0)
        # q == 0 only when b == 0 and c == 0: a double root at 0.
        r2 = np.where(q != 0.0, c / np.where(q != 0.0, q, 1.0), 0.0)
    lo = np.where(real, np.minimum(r1, r2), lo)
    hi = np.where(real, np.maximum(r1, r2), hi)
    linear = ~quadratic & (np.abs(b) > tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = -c / np.where(linear, b, 1.0)
    lo = np.where(linear, np.where(b > 0.0, -np.inf, crossing), lo)
    hi = np.where(linear, np.where(b > 0.0, crossing, np.inf), hi)
    constant = ~quadratic & ~linear
    inside = (c < 0.0) if strict else (c <= 0.0)
    lo = np.where(constant, np.where(inside, -np.inf, np.inf), lo)
    hi = np.where(constant, np.where(inside, np.inf, -np.inf), hi)
    return np.maximum(lo, t_lo), np.minimum(hi, t_hi)


def _segment_distance_sublevel(
    starts: np.ndarray,
    steps: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    *,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """For each line segment ``p(t) = starts + t * steps``, ``t in [0, 1]``, where it is within ``radius`` of ``[a, b]``.

    ``starts`` and ``steps`` are ``(n, 3)``; ``a`` and ``b`` one reference
    segment's ends (``a == b`` is a point). The distance to a segment is
    convex along a line, so each result is one ``t``-interval ``(lo, hi)``,
    empty where ``hi <= lo``: the hull of three quadratic pieces, where the
    nearest point is ``a``, inside ``[a, b]``, or ``b``. ``strict`` uses
    ``< radius`` instead of ``<= radius``.
    """
    vv = np.einsum("ij,ij->i", steps, steps)
    scale = np.maximum(vv, 1.0)
    w = b - a
    ww = float(w @ w)
    d = starts - a
    dd = np.einsum("ij,ij->i", d, d)
    dv = np.einsum("ij,ij->i", d, steps)
    r2 = radius * radius
    if ww == 0.0:
        return _quadratic_sublevel(vv, 2.0 * dv, dd - r2, 0.0, 1.0, strict=strict, scale=scale)
    dw = d @ w
    vw = steps @ w
    # s(t) = s0 + ds t is the position of the nearest point of the infinite line through [a, b].
    s0 = dw / ww
    ds = vw / ww
    rising, falling = ds > 0.0, ds < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(ds != 0.0, -s0 / np.where(ds != 0.0, ds, 1.0), np.nan)
        t1 = np.where(ds != 0.0, (1.0 - s0) / np.where(ds != 0.0, ds, 1.0), np.nan)
    # Where ds == 0 the nearest point is the same for every t: one piece covers [0, 1], the others nothing.
    before, after = s0 < 0.0, s0 > 1.0
    within = ~before & ~after
    a_lo = np.where(rising, 0.0, np.where(falling, t0, np.where(before, 0.0, 1.0)))
    a_hi = np.where(rising, t0, np.where(falling, 1.0, np.where(before, 1.0, 0.0)))
    b_lo = np.where(rising, t1, np.where(falling, 0.0, np.where(after, 0.0, 1.0)))
    b_hi = np.where(rising, 1.0, np.where(falling, t1, np.where(after, 1.0, 0.0)))
    m_lo = np.where(rising, t0, np.where(falling, t1, np.where(within, 0.0, 1.0)))
    m_hi = np.where(rising, t1, np.where(falling, t0, np.where(within, 1.0, 0.0)))
    e = starts - b
    pieces = (
        _quadratic_sublevel(
            vv, 2.0 * dv, dd - r2, np.clip(a_lo, 0.0, 1.0), np.clip(a_hi, 0.0, 1.0), strict=strict, scale=scale
        ),
        _quadratic_sublevel(
            vv,
            2.0 * np.einsum("ij,ij->i", e, steps),
            np.einsum("ij,ij->i", e, e) - r2,
            np.clip(b_lo, 0.0, 1.0),
            np.clip(b_hi, 0.0, 1.0),
            strict=strict,
            scale=scale,
        ),
        _quadratic_sublevel(
            np.maximum(vv - vw * vw / ww, 0.0),
            2.0 * (dv - dw * vw / ww),
            dd - dw * dw / ww - r2,
            np.clip(m_lo, 0.0, 1.0),
            np.clip(m_hi, 0.0, 1.0),
            strict=strict,
            scale=scale,
        ),
    )
    los = np.stack([piece[0] for piece in pieces])
    his = np.stack([piece[1] for piece in pieces])
    nonempty = his > los
    return np.min(np.where(nonempty, los, np.inf), axis=0), np.max(np.where(nonempty, his, -np.inf), axis=0)


def _left_closed(closed: str) -> bool:
    return closed in ("left", "both")


def _right_closed(closed: str) -> bool:
    return closed in ("right", "both")


def _parse_bounds(bounds: object) -> tuple[object | None, object | None]:
    if _is_quantity(bounds):
        mantissa = np.asarray(u.get_mantissa(bounds), dtype=float)
        if mantissa.ndim != 1 or mantissa.size != 2:
            raise TypeError("Quantity bounds must be a one-dimensional vector with exactly 2 values.")
        unit = u.get_unit(bounds)
        return u.Quantity(float(mantissa[0]), unit), u.Quantity(float(mantissa[1]), unit)
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        return bounds[0], bounds[1]
    raise TypeError("bounds must be either (low, high) tuple/list or a 2-element quantity vector.")


def _normalize_numeric_bound(bound: object | None, *, property_name: str) -> float | None:
    if bound is None:
        return None
    if _is_quantity(bound):
        raise ValueError(f"Property {property_name!r} is not quantity-valued; quantity bounds are not allowed.")
    if isinstance(bound, bool) or not isinstance(bound, Real):
        raise TypeError(f"{property_name} bounds must be numeric.")
    return float(bound)


def _normalize_quantity_bound(
    bound: object | None,
    *,
    property_name: str,
    unit: object,
) -> float | None:
    if bound is None:
        return None
    if not _is_quantity(bound):
        raise ValueError(f"Property {property_name!r} is quantity-valued; bounds must include units.")
    return _to_decimal_scalar(bound, unit=unit, name=property_name)


def _matches_range(
    value: object,
    *,
    low: object | None,
    high: object | None,
    closed: str,
    property_name: str,
    bounds_by_unit: dict[object, tuple[float | None, float | None]] | None = None,
) -> bool:
    if _is_quantity(value):
        unit = u.get_unit(value)
        current = _to_decimal_scalar(value, unit=unit, name=property_name)
        # Same per-unit memo as _matches_in: the bounds do not vary across
        # branches, but the unit they must be decoded into comes from each
        # branch's value, so the cache is keyed on it rather than removed.
        cached = None if bounds_by_unit is None else bounds_by_unit.get(unit)
        if cached is None:
            low_value = _normalize_quantity_bound(low, property_name=property_name, unit=unit)
            high_value = _normalize_quantity_bound(high, property_name=property_name, unit=unit)
            if bounds_by_unit is not None:
                bounds_by_unit[unit] = (low_value, high_value)
        else:
            low_value, high_value = cached
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Property {property_name!r} does not support numeric range filtering.")
        current = float(value)
        low_value = _normalize_numeric_bound(low, property_name=property_name)
        high_value = _normalize_numeric_bound(high, property_name=property_name)

    if low_value is not None and high_value is not None and low_value > high_value:
        raise ValueError(f"Lower bound must be <= upper bound for property {property_name!r}.")

    if low_value is not None:
        if _left_closed(closed):
            if current < low_value:
                return False
        elif current <= low_value:
            return False
    if high_value is not None:
        if _right_closed(closed):
            if current > high_value:
                return False
        elif current >= high_value:
            return False
    return True


def _normalize_candidates(candidates: tuple[object, ...], *, unit: object, property_name: str) -> set[float]:
    """Decode every candidate into ``unit``, rejecting non-numeric entries."""
    normalized: set[float] = set()
    for candidate in candidates:
        if _is_quantity(candidate):
            normalized.add(_to_decimal_scalar(candidate, unit=unit, name=property_name))
        elif isinstance(candidate, bool) or not isinstance(candidate, Real):
            raise TypeError(f"Property {property_name!r} is quantity-valued; candidates must be numeric/quantity.")
        else:
            normalized.add(float(candidate))
    return normalized


def _matches_in(
    value: object,
    *,
    candidates: tuple[object, ...],
    property_name: str,
    normalized_by_unit: dict[object, set[float]] | None = None,
) -> bool:
    if _is_quantity(value):
        unit = u.get_unit(value)
        current = _to_decimal_scalar(value, unit=unit, name=property_name)
        # Keyed by unit rather than hoisted outright: the unit comes from the
        # branch's own value, so a property returning mixed units still gets
        # a correctly decoded candidate set -- it just gets one per unit
        # instead of one per branch.
        if normalized_by_unit is None:
            normalized = _normalize_candidates(candidates, unit=unit, property_name=property_name)
        else:
            normalized = normalized_by_unit.get(unit)
            if normalized is None:
                normalized = _normalize_candidates(candidates, unit=unit, property_name=property_name)
                normalized_by_unit[unit] = normalized
        return current in normalized
    return any(value == candidate for candidate in candidates)


def branch_slice_intervals(
    morpho: Morphology,
    *,
    branch_index: object,
    prox: object,
    dist: object,
) -> tuple[tuple[int, float, float], ...]:
    branch_values = _coerce_values("branch_index", branch_index)
    prox_values = _coerce_values("prox", prox)
    dist_values = _coerce_values("dist", dist)

    n_values = max(len(branch_values), len(prox_values), len(dist_values))
    branch_values = _broadcast_values("branch_index", branch_values, n_values)
    prox_values = _broadcast_values("prox", prox_values, n_values)
    dist_values = _broadcast_values("dist", dist_values, n_values)

    n_branches = len(morpho.branches)
    intervals: list[tuple[int, float, float]] = []
    for raw_branch, raw_prox, raw_dist in zip(branch_values, prox_values, dist_values):
        idx = _coerce_branch_index(raw_branch, n_branches)
        prox_value = _coerce_norm_coord("prox", raw_prox)
        dist_value = _coerce_norm_coord("dist", raw_dist)
        if not (0.0 <= prox_value < dist_value <= 1.0):
            raise ValueError(
                f"BranchSlice expects 0.0 <= prox < dist <= 1.0, got prox={prox_value!r}, dist={dist_value!r}."
            )
        intervals.append((idx, prox_value, dist_value))
    return tuple(intervals)


def branch_in_intervals(
    morpho: Morphology,
    *,
    property_name: object,
    values: object,
) -> tuple[tuple[int, float, float], ...]:
    prop = _coerce_property_name(property_name)
    candidates = _coerce_values("values", values)
    normalized_by_unit: dict[object, set[float]] = {}
    intervals: list[tuple[int, float, float]] = []
    for index, branch_view in enumerate(morpho.branches):
        value = _resolve_branch_property(branch_view, prop)
        if _matches_in(
            value,
            candidates=candidates,
            property_name=prop,
            normalized_by_unit=normalized_by_unit,
        ):
            intervals.append((index, 0.0, 1.0))
    return tuple(intervals)


def branch_range_intervals(
    morpho: Morphology,
    *,
    property_name: object,
    bounds: object,
    closed: object,
) -> tuple[tuple[int, float, float], ...]:
    prop = _coerce_property_name(property_name)
    low, high = _parse_bounds(bounds)
    closed_mode = _parse_closed(closed)

    bounds_by_unit: dict[object, tuple[float | None, float | None]] = {}
    intervals: list[tuple[int, float, float]] = []
    for index, branch_view in enumerate(morpho.branches):
        value = _resolve_branch_property(branch_view, prop)
        if _matches_range(
            value,
            low=low,
            high=high,
            closed=closed_mode,
            property_name=prop,
            bounds_by_unit=bounds_by_unit,
        ):
            intervals.append((index, 0.0, 1.0))
    return tuple(intervals)


def _clip_norm_x(value: float, *, epsilon: float) -> float:
    if value < 0.0 and value >= -epsilon:
        return 0.0
    if value > 1.0 and value <= 1.0 + epsilon:
        return 1.0
    return value


def normalize_region_intervals(
    intervals: tuple[Interval, ...] | list[Interval],
    *,
    epsilon: float = EPSILON,
) -> tuple[Interval, ...]:
    grouped: dict[int, list[tuple[float, float]]] = {}

    for raw_branch, raw_prox, raw_dist in intervals:
        if not isinstance(raw_branch, int):
            raise TypeError(f"Branch index must be int, got {type(raw_branch).__name__!s}.")
        branch = int(raw_branch)
        prox = _clip_norm_x(float(raw_prox), epsilon=epsilon)
        dist = _clip_norm_x(float(raw_dist), epsilon=epsilon)
        if prox < 0.0 - epsilon or dist > 1.0 + epsilon:
            raise ValueError(f"Interval coordinates must be within [0, 1], got prox={prox!r}, dist={dist!r}.")
        if dist - prox <= epsilon:
            continue
        grouped.setdefault(branch, []).append((prox, dist))

    normalized: list[Interval] = []
    for branch in sorted(grouped):
        ranges = sorted(grouped[branch], key=lambda pair: (pair[0], pair[1]))
        current_start, current_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= current_end + epsilon:
                if end > current_end:
                    current_end = end
                continue
            if current_end - current_start > epsilon:
                normalized.append((branch, current_start, current_end))
            current_start, current_end = start, end
        if current_end - current_start > epsilon:
            normalized.append((branch, current_start, current_end))

    return tuple(normalized)


def _group_by_branch(
    intervals: tuple[Interval, ...] | list[Interval],
    *,
    epsilon: float,
) -> dict[int, list[tuple[float, float]]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for branch, prox, dist in normalize_region_intervals(intervals, epsilon=epsilon):
        grouped.setdefault(branch, []).append((prox, dist))
    return grouped


def union_region_intervals(
    left: tuple[Interval, ...] | list[Interval],
    right: tuple[Interval, ...] | list[Interval],
    *,
    epsilon: float = EPSILON,
) -> tuple[Interval, ...]:
    return normalize_region_intervals(tuple(left) + tuple(right), epsilon=epsilon)


def _intersect_branch_ranges(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    *,
    epsilon: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        ls, le = left[i]
        rs, re = right[j]
        start = max(ls, rs)
        end = min(le, re)
        if end - start > epsilon:
            out.append((start, end))
        if le <= re + epsilon:
            i += 1
        else:
            j += 1
    return out


def intersect_region_intervals(
    left: tuple[Interval, ...] | list[Interval],
    right: tuple[Interval, ...] | list[Interval],
    *,
    epsilon: float = EPSILON,
) -> tuple[Interval, ...]:
    left_group = _group_by_branch(left, epsilon=epsilon)
    right_group = _group_by_branch(right, epsilon=epsilon)
    branches = sorted(set(left_group) & set(right_group))
    out: list[Interval] = []
    for branch in branches:
        for start, end in _intersect_branch_ranges(
            left_group[branch],
            right_group[branch],
            epsilon=epsilon,
        ):
            out.append((branch, start, end))
    return normalize_region_intervals(out, epsilon=epsilon)


def _difference_branch_ranges(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    *,
    epsilon: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    j = 0
    for ls, le in left:
        while j < len(right) and right[j][1] <= ls + epsilon:
            j += 1
        cursor = ls
        k = j
        while k < len(right) and right[k][0] < le - epsilon:
            rs, re = right[k]
            if rs > cursor + epsilon:
                out.append((cursor, min(rs, le)))
            if re > cursor:
                cursor = re
            if cursor >= le - epsilon:
                break
            k += 1
        if cursor < le - epsilon:
            out.append((cursor, le))
    return out


def difference_region_intervals(
    left: tuple[Interval, ...] | list[Interval],
    right: tuple[Interval, ...] | list[Interval],
    *,
    epsilon: float = EPSILON,
) -> tuple[Interval, ...]:
    left_group = _group_by_branch(left, epsilon=epsilon)
    right_group = _group_by_branch(right, epsilon=epsilon)
    out: list[Interval] = []
    for branch in sorted(left_group):
        right_ranges = right_group.get(branch, [])
        if not right_ranges:
            for start, end in left_group[branch]:
                out.append((branch, start, end))
            continue
        for start, end in _difference_branch_ranges(
            left_group[branch],
            right_ranges,
            epsilon=epsilon,
        ):
            out.append((branch, start, end))
    return normalize_region_intervals(out, epsilon=epsilon)


def complement_region_intervals(
    intervals: tuple[Interval, ...] | list[Interval],
    *,
    n_branches: int,
    epsilon: float = EPSILON,
) -> tuple[Interval, ...]:
    universe = tuple((idx, 0.0, 1.0) for idx in range(n_branches))
    return difference_region_intervals(universe, intervals, epsilon=epsilon)


def _round_digits_from_epsilon(epsilon: float) -> int:
    if epsilon <= 0.0:
        return 12
    return max(0, int(round(-np.log10(epsilon))))


def _normalize_loc_x(value: float, *, epsilon: float) -> float:
    x = _clip_norm_x(float(value), epsilon=epsilon)
    if not np.isfinite(x):
        raise ValueError(f"Locset coordinate x must be finite, got {value!r}.")
    if x < 0.0 - epsilon or x > 1.0 + epsilon:
        raise ValueError(f"Locset coordinate x must be within [0, 1], got {value!r}.")
    return x


def normalize_locset_points(
    points: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    """Validate and canonicalize locations without changing row identity."""
    digits = _round_digits_from_epsilon(epsilon)
    normalized: list[Location] = []
    for raw_branch, raw_x in points:
        if isinstance(raw_branch, bool) or not isinstance(raw_branch, Integral):
            raise TypeError(f"Branch index must be int, got {type(raw_branch).__name__!s}.")
        if isinstance(raw_x, bool) or not isinstance(raw_x, Real):
            raise TypeError(f"Locset coordinate x must be a real number, got {type(raw_x).__name__!s}.")
        branch = int(raw_branch)
        if branch < 0:
            raise ValueError(f"Branch index must be non-negative, got {branch!r}.")
        if branch > np.iinfo(np.int64).max:
            raise ValueError(f"Branch index exceeds the int64 range, got {branch!r}.")
        x = _normalize_loc_x(raw_x, epsilon=epsilon)
        normalized.append((branch, round(x, digits)))
    return tuple(normalized)


def unique_locset_points(
    points: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    """Return canonical locations with stable first-occurrence deduplication."""
    result: list[Location] = []
    seen: set[Location] = set()
    for point in normalize_locset_points(points, epsilon=epsilon):
        if point in seen:
            continue
        seen.add(point)
        result.append(point)
    return tuple(result)


def concat_locset_points(
    left: tuple[Location, ...] | list[Location],
    right: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    """Concatenate canonical location rows while preserving duplicates."""
    return normalize_locset_points(tuple(left) + tuple(right), epsilon=epsilon)


def union_locset_points(
    left: tuple[Location, ...] | list[Location],
    right: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    return unique_locset_points(tuple(left) + tuple(right), epsilon=epsilon)


def intersect_locset_points(
    left: tuple[Location, ...] | list[Location],
    right: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    right_support = set(unique_locset_points(right, epsilon=epsilon))
    return tuple(point for point in unique_locset_points(left, epsilon=epsilon) if point in right_support)


def difference_locset_points(
    left: tuple[Location, ...] | list[Location],
    right: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    right_support = set(unique_locset_points(right, epsilon=epsilon))
    return tuple(point for point in unique_locset_points(left, epsilon=epsilon) if point not in right_support)


def fork_points_locations(morpho: Morphology, *, epsilon: float = EPSILON) -> tuple[Location, ...]:
    """Return topology junctions incident to at least three branches."""
    location_parent: dict[Location, Location] = {}
    location_order: list[Location] = []

    def add(point: Location) -> Location:
        normalized = normalize_locset_points((point,), epsilon=epsilon)[0]
        if normalized not in location_parent:
            location_parent[normalized] = normalized
            location_order.append(normalized)
        return normalized

    def find(point: Location) -> Location:
        root = point
        while location_parent[root] != root:
            root = location_parent[root]
        while location_parent[point] != point:
            parent = location_parent[point]
            location_parent[point] = root
            point = parent
        return root

    def union(left: Location, right: Location) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            location_parent[right_root] = left_root

    for edge in morpho.edges:
        parent = add((edge.parent.index, float(edge.parent_x)))
        child = add((edge.child.index, float(edge.child_x)))
        union(parent, child)

    groups: dict[Location, list[Location]] = {}
    for point in location_order:
        groups.setdefault(find(point), []).append(point)

    branch_depth: dict[int, int] = {}
    for branch in morpho.branch_by_order(order="depth"):
        branch_depth[branch.index] = 0 if branch.parent is None else branch_depth[branch.parent.index] + 1

    forks: list[Location] = []
    for points in groups.values():
        if len({branch_id for branch_id, _ in points}) < 3:
            continue
        canonical = min(points, key=lambda point: (branch_depth[point[0]], point[0], point[1]))
        forks.append(canonical)
    return tuple(forks)


def terminal_locations(morpho: Morphology, *, epsilon: float = EPSILON) -> tuple[Location, ...]:
    points: list[Location] = [
        (branch_idx, 1.0) for branch_idx, branch in enumerate(morpho.branches) if branch.n_children == 0
    ]
    return normalize_locset_points(points, epsilon=epsilon)


def _interval_measure_um(morpho: Morphology, branch: int, prox: float, dist: float) -> float:
    total_length_um = float(np.asarray(morpho.branches[branch].length.to_decimal(u.um), dtype=float))
    return (dist - prox) * total_length_um


def _sample_entries(
    morpho: Morphology,
    intervals: tuple[Interval, ...],
    *,
    epsilon: float,
) -> list[tuple[int, float, float, float]]:
    entries: list[tuple[int, float, float, float]] = []
    for branch, prox, dist in normalize_region_intervals(intervals, epsilon=epsilon):
        length_um = _interval_measure_um(morpho, branch, prox, dist)
        if length_um > epsilon:
            entries.append((branch, prox, dist, length_um))
    return entries


def coerce_positive_count(name: str, count: object) -> int:
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError(f"{name} must be an integer, got {count!r}.")
    value = int(count)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return value


def _length_scalar_um(
    value: object,
    *,
    name: str,
    positive: bool = False,
    allow_none: bool = False,
) -> float | None:
    """Check a scalar length quantity and return it as a float in micrometres.

    ``None`` raises ``TypeError`` unless ``allow_none``; a bare number or a
    quantity of the wrong unit raises ``TypeError`` and a non-scalar one
    ``ValueError`` (both from ``normalize_param``; a one-element array is a
    scalar). A non-finite value raises ``ValueError``, as does a value
    ``<= 0`` when ``positive`` and ``< 0`` otherwise.
    """
    if value is None and allow_none:
        return None
    quantity = normalize_param(value, name=name, unit=u.um, shape=())
    length_um = float(np.asarray(quantity.to_decimal(u.um)).reshape(()))
    if not np.isfinite(length_um):
        hint = "; use None for no bound" if allow_none else ""
        raise ValueError(f"{name} must be finite{hint}, got {value!r}.")
    if positive and length_um <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    if length_um < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}.")
    return length_um


def _points_at_arc_lengths(
    entries: list[tuple[int, float, float, float]],
    targets: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    """Map arc lengths along the concatenated ``_sample_entries`` to locations.

    Shared by ``UniformSamples`` and ``StepSamples``, so the two agree when a
    step divides the total length. A target past the end clips to the end
    of the last interval.
    """
    lengths = np.asarray([item[3] for item in entries], dtype=float)
    cumulative = np.cumsum(lengths)
    points: list[Location] = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right"))
        if idx >= len(entries):
            idx = len(entries) - 1
        branch, prox, dist, length_um = entries[idx]
        prev = 0.0 if idx == 0 else float(cumulative[idx - 1])
        offset = max(0.0, min(target - prev, length_um))
        # _sample_entries only appends when length_um > epsilon, so there is
        # no zero-length entry to guard against here.
        ratio = offset / length_um
        x = prox + ratio * (dist - prox)
        points.append((branch, _normalize_loc_x(x, epsilon=epsilon)))
    return normalize_locset_points(points, epsilon=epsilon)


def _total_length_um(entries: list[tuple[int, float, float, float]]) -> float:
    # The last cumulative sum, not np.sum: the same number the arc-length
    # mapping ends on.
    return float(np.cumsum(np.asarray([item[3] for item in entries], dtype=float))[-1])


def uniform_samples_from_region(
    morpho: Morphology,
    *,
    intervals: tuple[Interval, ...],
    count: object,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    n = coerce_positive_count("count", count)
    entries = _sample_entries(morpho, intervals, epsilon=epsilon)
    if not entries:
        raise ValueError("UniformSamples cannot sample from an empty or zero-length region.")
    total = _total_length_um(entries)
    targets = ((np.arange(n, dtype=float) + 0.5) / float(n)) * total
    return _points_at_arc_lengths(entries, targets, epsilon=epsilon)


def _step_samples_from_region(
    morpho: Morphology,
    *,
    intervals: tuple[Interval, ...],
    step_um: float,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    """Locations at arc lengths ``(k + 1/2) * step_um`` along the concatenated region.

    ``n = floor(T / step_um * (1 + STEP_COUNT_SLACK))`` locations, ``T`` the
    total length; a final partial step gets none.
    """
    entries = _sample_entries(morpho, intervals, epsilon=epsilon)
    if not entries:
        raise ValueError("StepSamples cannot sample from an empty or zero-length region.")
    total = _total_length_um(entries)
    n = int(np.floor(total / step_um * (1.0 + STEP_COUNT_SLACK)))
    if n == 0:
        raise ValueError(
            f"StepSamples step {step_um!r} um exceeds the region's total length {total!r} um; "
            "use a smaller step or UniformSamples(region, 1)."
        )
    targets = (np.arange(n, dtype=float) + 0.5) * step_um
    return _points_at_arc_lengths(entries, targets, epsilon=epsilon)


def random_samples_from_region(
    morpho: Morphology,
    *,
    intervals: tuple[Interval, ...],
    count: object,
    seed: object,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    n = coerce_positive_count("count", count)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"seed must be an integer, got {seed!r}.")
    entries = _sample_entries(morpho, intervals, epsilon=epsilon)
    if not entries:
        raise ValueError("RandomSamples cannot sample from an empty or zero-length region.")

    lengths = np.asarray([item[3] for item in entries], dtype=float)
    probs = lengths / np.sum(lengths)
    rng = np.random.default_rng(int(seed))
    interval_ids = rng.choice(len(entries), size=n, p=probs)
    uvals = rng.random(n)

    points: list[Location] = []
    for idx, uval in zip(interval_ids, uvals):
        branch, prox, dist, _ = entries[int(idx)]
        x = prox + float(uval) * (dist - prox)
        points.append((branch, _normalize_loc_x(x, epsilon=epsilon)))
    return normalize_locset_points(points, epsilon=epsilon)
