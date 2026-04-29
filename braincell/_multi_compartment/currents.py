"""Membrane-current summation pipeline for :class:`Cell`.

Responsibilities:

1. Normalize user-supplied external current (``I_ext``) into point-space
   current density, rejecting shapes that would NaN (bug #2).
2. Route the external density through :meth:`sum_current_inputs` as
   ``init`` (single-compartment pattern), so registered current-input
   callables accumulate on top and external is never dropped (bug #1).
3. Add clamp density from the precomputed :class:`ClampActiveTable`.
4. Iterate channel currents.
5. Bridge point-space sum back to CV-space for the voltage update.
"""

from typing import TYPE_CHECKING

import brainunit as u
import jax.numpy as jnp

from braincell._base import IonChannel
from braincell._compute.runtime import CellRuntimeState
from . import bridge

if TYPE_CHECKING:
    from .cell import Cell

__all__ = ["total_membrane_current"]

_CURRENT_DENSITY = u.nA / u.cm ** 2


def total_membrane_current(
    host: "Cell",
    *,
    V_cv,
    I_ext,
    t,
):
    """Return ``(..., n_cv)`` membrane current density in ``nA/cm^2``."""
    runtime = host.runtime
    point_V = bridge.cv_to_point(V_cv, runtime)

    I_ext_density = _normalize_ext_to_point_density(I_ext, runtime)
    I_point = host.sum_current_inputs(I_ext_density, point_V)

    I_point = I_point + _clamp_density(runtime, t=t)

    for key, ch in host.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
        try:
            contrib = ch.current(point_V)
        except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
            raise ValueError(
                f"Error computing current for ion channel {key!r}:\n{ch}\nError: {exc}"
            ) from exc
        if contrib is None:
            continue
        I_point = I_point + contrib

    return bridge.point_to_cv(I_point, runtime)


def _normalize_ext_to_point_density(value, runtime: CellRuntimeState):
    """Convert every supported ``I_ext`` shape to ``(n_point,) nA/cm^2``.

    Accepts:

    - python ``0`` / ``0.0``             -> zeros
    - scalar current density             -> broadcast
    - scalar total current (nA)          -> divide by cv area, scatter
    - ``(n_cv,)`` current density        -> scatter
    - ``(n_cv,)`` total current (nA)     -> divide, scatter
    - ``(n_point,)`` current density     -> pass through
    - ``(n_point,)`` total current (nA)  -> **raise** ValueError

    Plain array/scalar without units is treated as current density.
    """
    if _is_python_zero(value):
        return u.Quantity(jnp.zeros((runtime.n_point,), dtype=float), _CURRENT_DENSITY)

    if not isinstance(value, u.Quantity):
        broadcast = jnp.broadcast_to(jnp.asarray(value), (runtime.n_point,))
        return u.Quantity(broadcast, _CURRENT_DENSITY)

    is_density = value.has_same_unit(1.0 * _CURRENT_DENSITY)
    is_total = value.has_same_unit(1.0 * u.nA)
    shape = tuple(getattr(value, "shape", ()))

    if is_total and shape == (runtime.n_point,):
        raise ValueError(
            "I_ext supplied as (n_point,)-shaped total current (nA) is ambiguous; "
            "pass (n_cv,) or provide current density (nA/cm^2) instead."
        )

    if is_density:
        if shape == (runtime.n_point,):
            return value.in_unit(_CURRENT_DENSITY)
        if shape == (runtime.n_cv,):
            return bridge.cv_to_point(value.in_unit(_CURRENT_DENSITY), runtime)
        if shape == ():
            broadcast = jnp.broadcast_to(
                jnp.asarray(value.to_decimal(_CURRENT_DENSITY)),
                (runtime.n_point,),
            )
            return u.Quantity(broadcast, _CURRENT_DENSITY)

    if is_total:
        cv_area = runtime.cv_area
        if cv_area is None:
            raise ValueError(
                "Cannot convert total current (nA) to density: runtime.cv_area is None."
            )
        if shape == ():
            cv_density = (value / cv_area).in_unit(_CURRENT_DENSITY)
            return bridge.cv_to_point(cv_density, runtime)
        if shape == (runtime.n_cv,):
            cv_density = (value / cv_area).in_unit(_CURRENT_DENSITY)
            return bridge.cv_to_point(cv_density, runtime)

    raise ValueError(
        f"Unsupported I_ext shape/unit: shape={shape!r}, unit={getattr(value, 'unit', None)!r}. "
        "Accepted shapes: (), (n_cv,), (n_point,) with density units (nA/cm^2); "
        "or (), (n_cv,) with total-current units (nA)."
    )


def _clamp_density(runtime: CellRuntimeState, *, t):
    """Return ``(n_point,) nA/cm^2`` clamp current density.

    Reads the pre-built :class:`ClampActiveTable`; no layout iteration
    in the hot path.
    """
    table = runtime.clamp_active_table
    if table is None:
        return u.Quantity(jnp.zeros((runtime.n_point,), dtype=float), _CURRENT_DENSITY)

    currents_nA = runtime.evaluate_point_clamps(t=t).to_decimal(u.nA)
    active_density = currents_nA[table.ids] / table.area
    density = jnp.zeros((runtime.n_point,), dtype=float)
    density = density.at[table.ids].set(active_density)
    return u.Quantity(density, _CURRENT_DENSITY)


def _is_python_zero(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == 0
    )
