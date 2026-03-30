from __future__ import annotations

from numbers import Integral, Real

import brainunit as u
import numpy as np

from ..morpho import Morpho

Interval = tuple[int, float, float]
Location = tuple[int, float]
EPSILON = 1e-12

__all__ = [
    "EPSILON",
    "branch_slice_intervals",
    "branch_in_intervals",
    "branch_range_intervals",
    "normalize_region_intervals",
    "union_region_intervals",
    "intersect_region_intervals",
    "difference_region_intervals",
    "complement_region_intervals",
    "branch_points_locations",
    "terminal_locations",
    "uniform_samples_from_region",
    "random_samples_from_region",
    "normalize_locset_points",
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
    raise ValueError(
        f"{name} has length {len(values)!r}, which cannot be broadcast to {size!r}."
    )


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
    raise TypeError(
        f"Property {property_name!r} is not scalar-filterable; got {type(value).__name__!s}."
    )


def _resolve_branch_property(morpho: Morpho, branch_index: int, property_name: str) -> object:
    branch_view = morpho.branch(index=branch_index)

    if property_name == "branch_id":
        return branch_index
    if property_name == "parent_id":
        return -1 if branch_view.parent is None else branch_view.parent.index
    if property_name == "n_children":
        return branch_view.n_children
    if property_name == "branch_order":
        return len(morpho.path_to_root(branch_index)) - 1
    if property_name == "n_tapers":
        return branch_view.n_segments
    if property_name == "length":
        return branch_view.length
    if property_name in {"area", "volume", "max_radius", "min_radius"}:
        raise ValueError(
            f"Branch property {property_name!r} is not supported in this version."
        )
    if hasattr(branch_view, property_name):
        return _coerce_filterable_scalar(property_name, getattr(branch_view, property_name))
    raise ValueError(f"Unknown branch property {property_name!r}.")


def _to_decimal_scalar(value: object, *, unit: object, name: str) -> float:
    scalar = _coerce_quantity_scalar(value, property_name=name)
    return float(np.asarray(scalar.to_decimal(unit), dtype=float))


def _parse_closed(closed: object) -> str:
    allowed = {"both", "left", "right", "neither"}
    if not isinstance(closed, str) or closed not in allowed:
        raise ValueError(f"closed must be one of {sorted(allowed)!r}, got {closed!r}.")
    return closed


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
    raise TypeError(
        "bounds must be either (low, high) tuple/list or a 2-element quantity vector."
    )


def _normalize_numeric_bound(bound: object | None, *, property_name: str) -> float | None:
    if bound is None:
        return None
    if _is_quantity(bound):
        raise ValueError(
            f"Property {property_name!r} is not quantity-valued; quantity bounds are not allowed."
        )
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
        raise ValueError(
            f"Property {property_name!r} is quantity-valued; bounds must include units."
        )
    return _to_decimal_scalar(bound, unit=unit, name=property_name)


def _matches_range(value: object, *, low: object | None, high: object | None, closed: str, property_name: str) -> bool:
    if _is_quantity(value):
        unit = u.get_unit(value)
        current = _to_decimal_scalar(value, unit=unit, name=property_name)
        low_value = _normalize_quantity_bound(low, property_name=property_name, unit=unit)
        high_value = _normalize_quantity_bound(high, property_name=property_name, unit=unit)
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Property {property_name!r} does not support numeric range filtering."
            )
        current = float(value)
        low_value = _normalize_numeric_bound(low, property_name=property_name)
        high_value = _normalize_numeric_bound(high, property_name=property_name)

    if low_value is not None and high_value is not None and low_value > high_value:
        raise ValueError(
            f"Lower bound must be <= upper bound for property {property_name!r}."
        )

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


def _matches_in(value: object, *, candidates: tuple[object, ...], property_name: str) -> bool:
    if _is_quantity(value):
        unit = u.get_unit(value)
        current = _to_decimal_scalar(value, unit=unit, name=property_name)
        normalized: set[float] = set()
        for candidate in candidates:
            if _is_quantity(candidate):
                normalized.add(_to_decimal_scalar(candidate, unit=unit, name=property_name))
            elif isinstance(candidate, bool) or not isinstance(candidate, Real):
                raise TypeError(
                    f"Property {property_name!r} is quantity-valued; candidates must be numeric/quantity."
                )
            else:
                normalized.add(float(candidate))
        return current in normalized
    return any(value == candidate for candidate in candidates)


def branch_slice_intervals(
    morpho: Morpho,
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
                "BranchSlice expects 0.0 <= prox < dist <= 1.0, "
                f"got prox={prox_value!r}, dist={dist_value!r}."
            )
        intervals.append((idx, prox_value, dist_value))
    return tuple(intervals)


def branch_in_intervals(
    morpho: Morpho,
    *,
    property_name: object,
    values: object,
) -> tuple[tuple[int, float, float], ...]:
    prop = _coerce_property_name(property_name)
    candidates = _coerce_values("values", values)
    intervals: list[tuple[int, float, float]] = []
    for index, _ in enumerate(morpho.branches):
        value = _resolve_branch_property(morpho, index, prop)
        if _matches_in(value, candidates=candidates, property_name=prop):
            intervals.append((index, 0.0, 1.0))
    return tuple(intervals)


def branch_range_intervals(
    morpho: Morpho,
    *,
    property_name: object,
    bounds: object,
    closed: object,
) -> tuple[tuple[int, float, float], ...]:
    prop = _coerce_property_name(property_name)
    low, high = _parse_bounds(bounds)
    closed_mode = _parse_closed(closed)

    intervals: list[tuple[int, float, float]] = []
    for index, _ in enumerate(morpho.branches):
        value = _resolve_branch_property(morpho, index, prop)
        if _matches_range(
            value,
            low=low,
            high=high,
            closed=closed_mode,
            property_name=prop,
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
            raise ValueError(
                f"Interval coordinates must be within [0, 1], got prox={prox!r}, dist={dist!r}."
            )
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
    if x < 0.0 - epsilon or x > 1.0 + epsilon:
        raise ValueError(f"Locset coordinate x must be within [0, 1], got {value!r}.")
    return x


def normalize_locset_points(
    points: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    digits = _round_digits_from_epsilon(epsilon)
    normalized: set[Location] = set()
    for raw_branch, raw_x in points:
        if not isinstance(raw_branch, int):
            raise TypeError(f"Branch index must be int, got {type(raw_branch).__name__!s}.")
        branch = int(raw_branch)
        x = _normalize_loc_x(float(raw_x), epsilon=epsilon)
        normalized.add((branch, round(x, digits)))
    return tuple(sorted(normalized, key=lambda item: (item[0], item[1])))


def union_locset_points(
    left: tuple[Location, ...] | list[Location],
    right: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    return normalize_locset_points(tuple(left) + tuple(right), epsilon=epsilon)


def intersect_locset_points(
    left: tuple[Location, ...] | list[Location],
    right: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    left_norm = set(normalize_locset_points(left, epsilon=epsilon))
    right_norm = set(normalize_locset_points(right, epsilon=epsilon))
    return tuple(sorted(left_norm & right_norm, key=lambda item: (item[0], item[1])))


def difference_locset_points(
    left: tuple[Location, ...] | list[Location],
    right: tuple[Location, ...] | list[Location],
    *,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    left_norm = set(normalize_locset_points(left, epsilon=epsilon))
    right_norm = set(normalize_locset_points(right, epsilon=epsilon))
    return tuple(sorted(left_norm - right_norm, key=lambda item: (item[0], item[1])))


def branch_points_locations(morpho: Morpho, *, epsilon: float = EPSILON) -> tuple[Location, ...]:
    points: list[Location] = []
    for parent_branch in range(len(morpho.branches)):
        children = morpho.branch(index=parent_branch).children
        if len(children) < 2:
            continue
        for child in children:
            parent_x = child.parent_x
            if parent_x is None:
                continue
            points.append((parent_branch, float(parent_x)))
    return normalize_locset_points(points, epsilon=epsilon)


def terminal_locations(morpho: Morpho, *, epsilon: float = EPSILON) -> tuple[Location, ...]:
    points: list[Location] = []
    for branch_idx in range(len(morpho.branches)):
        if len(morpho.branch(index=branch_idx).children) == 0:
            points.append((branch_idx, 1.0))
    return normalize_locset_points(points, epsilon=epsilon)


def _interval_measure_um(morpho: Morpho, branch: int, prox: float, dist: float) -> float:
    total_length_um = float(np.asarray(morpho.branches[branch].length.to_decimal(u.um), dtype=float))
    return (dist - prox) * total_length_um


def _sample_entries(
    morpho: Morpho,
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


def _coerce_positive_count(name: str, count: object) -> int:
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError(f"{name} must be an integer, got {count!r}.")
    value = int(count)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return value


def uniform_samples_from_region(
    morpho: Morpho,
    *,
    intervals: tuple[Interval, ...],
    count: object,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    n = _coerce_positive_count("count", count)
    entries = _sample_entries(morpho, intervals, epsilon=epsilon)
    if not entries:
        return ()

    lengths = np.asarray([item[3] for item in entries], dtype=float)
    cumulative = np.cumsum(lengths)
    total = float(cumulative[-1])
    targets = ((np.arange(n, dtype=float) + 0.5) / float(n)) * total

    points: list[Location] = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right"))
        if idx >= len(entries):
            idx = len(entries) - 1
        branch, prox, dist, length_um = entries[idx]
        prev = 0.0 if idx == 0 else float(cumulative[idx - 1])
        offset = max(0.0, min(target - prev, length_um))
        ratio = 0.0 if length_um <= epsilon else (offset / length_um)
        x = prox + ratio * (dist - prox)
        points.append((branch, _normalize_loc_x(x, epsilon=epsilon)))
    return normalize_locset_points(points, epsilon=epsilon)


def random_samples_from_region(
    morpho: Morpho,
    *,
    intervals: tuple[Interval, ...],
    count: object,
    seed: object,
    epsilon: float = EPSILON,
) -> tuple[Location, ...]:
    n = _coerce_positive_count("count", count)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"seed must be an integer, got {seed!r}.")
    entries = _sample_entries(morpho, intervals, epsilon=epsilon)
    if not entries:
        return ()

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
