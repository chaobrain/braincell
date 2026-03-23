"""Shared `brainunit` helpers used by the layered skeleton.

The public pattern is: each parameter first goes through `normalize_param(...)`
to validate shape, unit compatibility, and numeric bounds against a chosen base
unit. Callers then operate on the normalized quantity directly with
`brainunit` operations; this module stays focused on parameter normalization
and validation.
"""

from __future__ import annotations

from typing import Any

import brainunit as u
import numpy as np


_BOUND_OPERATORS = ("ge", "gt", "le", "lt")


def is_quantity(value: object) -> bool:
    """Best-effort check for a brainunit quantity object."""

    return hasattr(value, "to_decimal") and callable(getattr(value, "to_decimal"))


def mantissa(value: object) -> Any:
    """Return a NumPy mantissa array from a quantity-like value."""

    return np.asarray(u.get_mantissa(value), dtype=float)


def _decimal_data(value: object, *, unit: Any, name: str) -> object:
    """Convert nested quantity leaves into raw decimals in the target unit."""

    if is_quantity(value):
        u.fail_for_dimension_mismatch(
            value,
            unit,
            f"{name} must have unit compatible with {unit}.",
        )
        return value.to_decimal(unit)
    if isinstance(value, (list, tuple)):
        return [_decimal_data(item, unit=unit, name=name) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _decimal_data(value.tolist(), unit=unit, name=name)
        except TypeError:
            pass
    return value


def _normalize_quantity(param: object, *, unit: Any, name: str) -> Any:
    """Convert raw or quantity input into a normalized quantity in `unit`."""

    try:
        decimal = _decimal_data(param, unit=unit, name=name)
        arr = np.asarray(decimal, dtype=float)
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    return u.Quantity(arr, unit)


def _check_shape(array: Any, *, name: str, shape: int | tuple[int | None, ...] | None) -> None:
    """Validate a mantissa array against a compact shape specification."""

    if shape is None:
        return
    if isinstance(shape, int):
        if array.ndim != shape:
            raise ValueError(f"{name} must be {shape}D, got shape {array.shape!r}.")
        return

    expected = tuple(shape)
    if array.ndim != len(expected):
        raise ValueError(f"{name} must have shape {expected!r}, got {array.shape!r}.")
    for axis, (actual, target) in enumerate(zip(array.shape, expected)):
        if target is not None and actual != target:
            raise ValueError(
                f"{name} axis {axis} must have size {target}, got shape {array.shape!r}."
            )


def _normalize_bound(bound: object, *, unit: Any, name: str) -> Any:
    """Convert a bound into raw decimal data in the target unit."""

    if is_quantity(bound):
        u.fail_for_dimension_mismatch(
            bound,
            unit,
            f"{name} bound must have unit compatible with {unit}.",
        )
        return np.asarray(bound.to_decimal(unit), dtype=float)
    try:
        return np.asarray(bound, dtype=float)
    except ValueError as exc:
        raise ValueError(f"{name} bounds must be numeric.") from exc


def _check_bounds(array: Any, *, name: str, unit: Any, bounds: dict[str, object] | None) -> None:
    """Validate a mantissa array against scalar or broadcastable bounds."""

    if not bounds:
        return

    invalid = tuple(key for key in bounds if key not in _BOUND_OPERATORS)
    if invalid:
        raise ValueError(f"{name} received unsupported bound keys {invalid!r}.")

    comparators = {
        "ge": (lambda lhs, rhs: lhs >= rhs, ">="),
        "gt": (lambda lhs, rhs: lhs > rhs, ">"),
        "le": (lambda lhs, rhs: lhs <= rhs, "<="),
        "lt": (lambda lhs, rhs: lhs < rhs, "<"),
    }
    for key in _BOUND_OPERATORS:
        if key not in bounds:
            continue
        bound = _normalize_bound(bounds[key], unit=unit, name=name)
        compare, symbol = comparators[key]
        if not np.all(compare(array, bound)):
            raise ValueError(f"{name} must satisfy {symbol} {bound!r} in base unit {unit}.")


def normalize_param(
    param: object,
    *,
    name: str,
    unit: Any,
    shape: int | tuple[int | None, ...] | None = None,
    bounds: dict[str, object] | None = None,
    allow_none: bool = False,
) -> Any:
    """Normalize one parameter to a base unit and validate shape and values.

    Rules:
    - unitless numeric inputs are interpreted in the provided `unit`
    - quantity inputs must have compatible dimensions and are converted to `unit`
    - mantissas are normalized to `numpy.ndarray`
    - shape and numeric bounds are checked after conversion into `unit`
    """

    if param is None:
        if allow_none:
            return None
        raise TypeError(f"{name} cannot be None.")

    quantity = _normalize_quantity(param, unit=unit, name=name)
    array = mantissa(quantity)
    _check_shape(array, name=name, shape=shape)
    _check_bounds(array, name=name, unit=unit, bounds=bounds)
    return quantity


def segment_lengths_from_points(points: object) -> Any:
    """Compute segment lengths from a normalized point matrix quantity."""

    return u.math.linalg.norm(points[1:] - points[:-1], axis=1)
