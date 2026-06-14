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
    "broadcast_to_shape",
    "expand_with_batch_axis",
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
    """Stack homogeneous scalar values into one array.

    Parameters
    ----------
    values : list of object
        Values to stack. All entries should either be plain numeric
        scalars or same-unit brainunit quantities.
    shape : tuple of int, optional
        Target shape for the stacked output. When omitted, the result
        is one-dimensional with length ``len(values)``.

    Returns
    -------
    object
        A plain array or :class:`brainunit.Quantity` containing the
        stacked values.
    """
    if len(values) == 0:
        return values
    first = values[0]
    target_shape = (len(values),) if shape is None else shape
    if hasattr(first, "unit"):
        decimals = [item.to_decimal(first.unit) for item in values]
        return u.Quantity(u.math.asarray(decimals).reshape(target_shape), first.unit)
    return u.math.asarray(values).reshape(target_shape)


def broadcast_to_shape(value: object, shape: tuple[int, ...], *, name: str = "value") -> object:
    """Broadcast ``value`` to ``shape`` while preserving brainunit quantities.

    Parameters
    ----------
    value : object
        Scalar or array-like value to broadcast.
    shape : tuple[int, ...]
        Target shape.
    name : str, optional
        Label used in error messages.

    Returns
    -------
    object
        ``value`` broadcast to ``shape``.

    Raises
    ------
    ValueError
        If ``value`` cannot be broadcast to ``shape``.
    """
    if hasattr(value, "unit"):
        unit = value.unit
        mantissa = u.math.asarray(value.to_decimal(unit))
        try:
            out = u.math.broadcast_to(mantissa, shape)
        except Exception as exc:
            raise ValueError(
                f"{name} with shape {getattr(mantissa, 'shape', None)!r} "
                f"cannot be broadcast to target shape {shape!r}."
            ) from exc
        return u.Quantity(out, unit)

    array = u.math.asarray(value)
    try:
        return u.math.broadcast_to(array, shape)
    except Exception as exc:
        raise ValueError(
            f"{name} with shape {getattr(array, 'shape', None)!r} "
            f"cannot be broadcast to target shape {shape!r}."
        ) from exc


def fill_like(shape: tuple[int, ...], value: object) -> object:
    """Broadcast ``value`` to ``shape`` preserving its brainunit type.

    Parameters
    ----------
    shape : tuple of int
        Target shape.
    value : object
        Scalar or array-like value to broadcast.

    Returns
    -------
    object
        Broadcast value with the requested shape.
    """
    return broadcast_to_shape(value, shape, name="fill_like(value)")


def expand_with_batch_axis(value: object, batch_size: int | None, *, name: str = "value") -> object:
    """Prepend one leading batch axis to an already-shaped value.

    Parameters
    ----------
    value : object
        Value that already has its population/spatial shape.
    batch_size : int or None
        Optional leading batch size. ``None`` returns ``value``
        unchanged.
    name : str, optional
        Label used in validation errors.

    Returns
    -------
    object
        ``value`` with a leading batch axis when requested.
    """
    if batch_size is None:
        return value
    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, bool):
        raise TypeError(f"{name} batch_size must be int or None, got {batch_size!r}.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"{name} batch_size must be > 0, got {batch_size!r}.")

    if hasattr(value, "unit"):
        unit = value.unit
        mantissa = u.math.asarray(value.to_decimal(unit))
        out = u.math.broadcast_to(mantissa, (batch_size,) + tuple(mantissa.shape))
        return u.Quantity(out, unit)

    array = u.math.asarray(value)
    return u.math.broadcast_to(array, (batch_size,) + tuple(array.shape))


def cv_value_vector(cell: "object", *, attr_name: str) -> object:
    """Return one per-CV attribute vector, broadcast across population dims.

    Parameters
    ----------
    cell : object
        Cell-like object exposing ``cvs`` and optionally ``pop_size``.
    attr_name : str
        CV attribute to gather.

    Returns
    -------
    object
        Quantity or plain array with shape ``pop_size + (n_cv,)`` when
        ``cell.pop_size`` is non-empty, otherwise ``(n_cv,)``.
    """
    values = quantity_vector([getattr(cv, attr_name) for cv in cell.cvs])
    pop_size = tuple(getattr(cell, "pop_size", ()))
    if len(pop_size) == 0:
        return values
    return broadcast_to_shape(
        values,
        pop_size + (len(cell.cvs),),
        name=f"cv_value_vector({attr_name})",
    )


def scatter_midpoint_values(
    *, values: object, point_ids: np.ndarray, n_point: int
) -> object:
    """Scatter CV-midpoint values into point space.

    Parameters
    ----------
    values : object
        CV-space value with shape ``(..., n_cv)``.
    point_ids : ndarray
        Midpoint point ids for each CV.
    n_point : int
        Total number of point-space rows.

    Returns
    -------
    object
        Point-space value with shape ``(..., n_point)``.
    """
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
    """Gather point-space midpoint values back into CV space.

    Parameters
    ----------
    values : object
        Point-space value with shape ``(..., n_point)``.
    point_ids : ndarray
        Midpoint point ids for each CV.

    Returns
    -------
    object
        CV-space value with shape ``(..., n_cv)``.
    """
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
    """Gather one per-CV geometry attribute and scatter it to point space.

    Parameters
    ----------
    cvs : tuple
        Control-volume declarations.
    attr_name : str
        Geometry attribute name to read from each CV.
    point_ids : ndarray
        Midpoint point ids for each CV.
    n_point : int
        Total number of point-space rows.

    Returns
    -------
    object
        Point-space geometry value with shape ``(n_point,)``.
    """
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
    """Attach point-space geometry arrays to runtime ion containers.

    Parameters
    ----------
    ions : dict[str, object]
        Runtime ion containers.
    cvs : tuple
        Control-volume declarations.
    point_ids : ndarray
        Midpoint point ids for each CV.
    n_point : int
        Total number of point-space rows.

    Notes
    -----
    Geometry is first assembled in one-cell point space, then broadcast
    to ``ion.varshape[:-1] + (n_point,)`` so homogeneous population
    ions receive the same geometry at every population index.
    """
    for ion in ions.values():
        pop_size = tuple(getattr(ion, "varshape", ())[:-1])
        point_shape = pop_size + (n_point,)
        length = broadcast_to_shape(
            scatter_cv_geometry(cvs=cvs, attr_name="length", point_ids=point_ids, n_point=n_point),
            point_shape,
            name="ion.length",
        )
        area = broadcast_to_shape(
            scatter_cv_geometry(cvs=cvs, attr_name="area", point_ids=point_ids, n_point=n_point),
            point_shape,
            name="ion.area",
        )
        diam_mid = broadcast_to_shape(
            scatter_cv_geometry(cvs=cvs, attr_name="diam_mid", point_ids=point_ids, n_point=n_point),
            point_shape,
            name="ion.diam_mid",
        )
        diam_arc_mean = broadcast_to_shape(
            scatter_cv_geometry(cvs=cvs, attr_name="diam_arc_mean", point_ids=point_ids, n_point=n_point),
            point_shape,
            name="ion.diam_arc_mean",
        )
        radius_prox = broadcast_to_shape(
            scatter_cv_geometry(cvs=cvs, attr_name="radius_prox", point_ids=point_ids, n_point=n_point),
            point_shape,
            name="ion.radius_prox",
        )
        radius_dist = broadcast_to_shape(
            scatter_cv_geometry(cvs=cvs, attr_name="radius_dist", point_ids=point_ids, n_point=n_point),
            point_shape,
            name="ion.radius_dist",
        )
        setattr(ion, "length", length)
        setattr(ion, "area", area)
        setattr(ion, "diam_mid", diam_mid)
        setattr(ion, "diam_arc_mean", diam_arc_mean)
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
