"""CV ↔ point-space conversion helpers and brainunit vectorisation.

All the CV/point scatter-gather primitives live here so that
:mod:`braincell._compute.runtime` does not reach into
:mod:`braincell._multi_compartment` and vice-versa. Runtime re-imports
what it needs at the top of :mod:`braincell._compute.runtime`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import brainunit as u
import numpy as np

if TYPE_CHECKING:
    from braincell._compute.runtime import CellRuntimeState

__all__ = [
    "attach_runtime_ion_geometry",
    "cv_to_point",
    "cv_value_vector",
    "fill_like",
    "gather_midpoint_values",
    "is_python_zero",
    "matches_last_dim",
    "point_to_cv",
    "quantity_vector",
    "scatter_cv_geometry",
    "scatter_midpoint_values",
]


def quantity_vector(
    values: list[object], *, shape: tuple[int, ...] | None = None
) -> object:
    """Stack a list of brainunit quantities (or plain numbers) into one array."""
    if len(values) == 0:
        return values
    first = values[0]
    target_shape = (len(values),) if shape is None else shape
    if hasattr(first, "unit"):
        decimals = [item.to_decimal(first.unit) for item in values]
        return u.Quantity(u.math.asarray(decimals).reshape(target_shape), first.unit)
    return u.math.asarray(values).reshape(target_shape)


def fill_like(shape: tuple[int, ...], value: object) -> object:
    """Broadcast ``value`` to ``shape`` preserving its brainunit type."""
    values = [value for _ in range(int(np.prod(shape, dtype=int)))]
    return quantity_vector(values, shape=shape)


def cv_value_vector(cell: "object", *, attr_name: str) -> object:
    """Stack per-CV attribute values from ``cell.cvs`` into one quantity vector."""
    return quantity_vector([getattr(cv, attr_name) for cv in cell.cvs])


def scatter_midpoint_values(
    *, values: object, point_ids: np.ndarray, n_point: int
) -> object:
    """Scatter a ``(..., n_cv)`` array onto ``point_ids`` within a ``(..., n_point)`` output."""
    if hasattr(values, "unit"):
        unit = values.unit
        mantissa = u.math.asarray(values.to_decimal(unit))
        base_shape = mantissa.shape[:-1] + (n_point,)
        out = u.math.zeros(base_shape, dtype=mantissa.dtype)
        out = out.at[..., point_ids].set(mantissa)
        return u.Quantity(out, unit)
    array = u.math.asarray(values)
    base_shape = array.shape[:-1] + (n_point,)
    out = u.math.zeros(base_shape, dtype=array.dtype)
    return out.at[..., point_ids].set(array)


def gather_midpoint_values(values: object, *, point_ids: np.ndarray) -> object:
    """Gather ``(..., n_point)`` values at ``point_ids`` → ``(..., n_cv)``."""
    return values[..., point_ids]


def matches_last_dim(value: object, size: int) -> bool:
    """True when ``value.shape[-1] == size`` (for a shape-bearing object)."""
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return False
    return int(shape[-1]) == int(size)


def is_python_zero(value: object) -> bool:
    """True for a bare Python ``0`` / ``0.0`` (not ``False``)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == 0
    )


def scatter_cv_geometry(
    *,
    cvs: tuple[object, ...],
    attr_name: str,
    point_ids: np.ndarray,
    n_point: int,
) -> object:
    """Gather a per-CV attribute and scatter it onto the point-space vector."""
    values = quantity_vector([getattr(cv, attr_name) for cv in cvs])
    return scatter_midpoint_values(
        values=values, point_ids=point_ids, n_point=n_point
    )


def attach_runtime_ion_geometry(
    *,
    ions: dict[str, object],
    cvs: tuple[object, ...],
    point_ids: np.ndarray,
    n_point: int,
) -> None:
    """Assign length / area / diam_mid / radii to every runtime-ion container."""
    length = scatter_cv_geometry(cvs=cvs, attr_name="length", point_ids=point_ids, n_point=n_point)
    area = scatter_cv_geometry(cvs=cvs, attr_name="area", point_ids=point_ids, n_point=n_point)
    diam_mid = scatter_cv_geometry(cvs=cvs, attr_name="diam_mid", point_ids=point_ids, n_point=n_point)
    radius_prox = scatter_cv_geometry(cvs=cvs, attr_name="radius_prox", point_ids=point_ids, n_point=n_point)
    radius_dist = scatter_cv_geometry(cvs=cvs, attr_name="radius_dist", point_ids=point_ids, n_point=n_point)

    for ion in ions.values():
        setattr(ion, "length", length)
        setattr(ion, "area", area)
        setattr(ion, "diam_mid", diam_mid)
        setattr(ion, "radius_prox", radius_prox)
        setattr(ion, "radius_dist", radius_dist)


def cv_to_point(values, runtime: "CellRuntimeState"):
    """Scatter a ``(..., n_cv)`` array onto CV midpoints in point space."""
    return scatter_midpoint_values(
        values=values,
        point_ids=runtime.node_tree.cv_to_mid_node_id,
        n_point=runtime.n_point,
    )


def point_to_cv(values, runtime: "CellRuntimeState"):
    """Gather a ``(..., n_point)`` array at CV midpoints → ``(..., n_cv)``."""
    return gather_midpoint_values(
        values, point_ids=runtime.node_tree.cv_to_mid_node_id
    )
