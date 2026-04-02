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

"""Shared unit-normalization helpers for explicit `brainunit` inputs."""



from typing import Any

import brainunit as u
import numpy as np

_BOUND_OPERATORS = ("ge", "gt", "le", "lt")


def _to_unit(param: object, name: str, unit: Any) -> np.ndarray:
    """Convert a quantity-like value to a NumPy array in the target unit."""

    try:
       return np.asarray(param.to_decimal(unit))
    except Exception as exc:
        raise TypeError(f"{name} must satisfy unit {unit}.") from exc


def _to_shape(
    array: np.ndarray,
    *,
    name: str,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    """Check and optionally reshape array to match the shape specification."""

    if shape is None:
        return array

    n_expected_dims = len(shape)
    n_actual_dims = array.ndim

    if n_actual_dims < n_expected_dims:
        new_shape = (1,) * (n_expected_dims - n_actual_dims) + array.shape
        array = array.reshape(new_shape)
        n_actual_dims = array.ndim

    if n_actual_dims > n_expected_dims:
        extra_dims = array.shape[: n_actual_dims - n_expected_dims]
        if all(d == 1 for d in extra_dims):
            array = array.reshape(array.shape[n_actual_dims - n_expected_dims :])
        else:
            raise ValueError(
                f"{name}: expected {n_expected_dims}D, got {n_actual_dims}D with shape {array.shape}"
            )

    for i, (expected, actual) in enumerate(zip(shape, array.shape)):
        if expected is not None and expected != actual:
            raise ValueError(f"{name}: expected dimension {i} to be {expected}, got {actual}")

    return array


def _normalize_bound(bound: object, *, unit: Any, name: str) -> np.ndarray:
    """Convert a quantity bound into a NumPy value in `unit`."""

    try:
        return _to_unit(bound, name, unit)
    except TypeError as exc:
        raise ValueError(f"{name} bounds must satisfy unit {unit}.") from exc


def _check_bounds(array: np.ndarray, *, name: str, unit: Any, bounds: dict[str, object] | None) -> None:
    """Validate a normalized NumPy array against simple scalar comparisons."""

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
            raise ValueError(f"{name} must satisfy {symbol} {bound!r}.")


def normalize_param(
    param: object,
    *,
    name: str,
    unit: Any,
    shape: int | tuple[int | None, ...] | None = None,
    bounds: dict[str, object] | None = None,
    allow_none: bool = False,
) -> Any:
    """Normalize one explicit-unit parameter and validate shape and bounds."""

    if param is None:
        if allow_none:
            return None
        raise TypeError(f"{name} cannot be None.")

    if isinstance(shape, int):
        shape = (shape,)

    array = _to_unit(param, name, unit)
    array = _to_shape(array, name=name, shape=shape)
    _check_bounds(array, name=name, unit=unit, bounds=bounds)
    return u.Quantity(array, unit)
