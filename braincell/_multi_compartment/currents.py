"""Membrane-current summation pipeline for :class:`Cell`.

Responsibilities:

1. Seed point-space membrane current density with zeros.
2. Add registered current-input callables via :meth:`sum_current_inputs`.
3. Add clamp density from the precomputed :class:`ClampActiveTable`.
4. Iterate channel currents.
5. Bridge point-space sum back to CV-space for the voltage update.
"""

from typing import TYPE_CHECKING

import brainunit as u
import jax.numpy as jnp

from braincell._base import IonChannel, Synapse as RuntimeSynapse
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
    t,
):
    """Return ``(..., n_cv)`` membrane current density in ``nA/cm^2``."""
    runtime = host.runtime
    point_V = bridge.cv_to_point(V_cv, runtime)

    zero_density = u.Quantity(
        jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=float),
        _CURRENT_DENSITY,
    )
    I_point = host.sum_current_inputs(zero_density, point_V)

    I_point = I_point + _clamp_density(runtime, t=t)

    for key, ch in host.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
        if isinstance(ch, RuntimeSynapse):
            layout_id = _layout_id_from_current_key(key)
            layout = runtime.layouts[layout_id]
            contrib_point = _synapse_contrib_to_point(runtime, layout, ch, point_V)
            if contrib_point is None:
                continue
            I_point = I_point + contrib_point
            continue
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


def _layout_id_from_current_key(key) -> int:
    if len(key) == 0:
        raise ValueError(f"Expected runtime object key ending with 'layout_<id>', got {key!r}.")
    last = key[-1]
    if not isinstance(last, str) or not last.startswith("layout_"):
        raise ValueError(f"Expected runtime object key ending with 'layout_<id>', got {key!r}.")
    return int(last.split("_", 1)[1])


def _synapse_contrib_to_point(runtime: CellRuntimeState, layout, syn, point_V):
    try:
        contrib = syn.current(point_V[..., layout.point_index])
    except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
        raise ValueError(
            f"Error computing current for synapse layout {layout.id!r}:\n{syn}\nError: {exc}"
        ) from exc
    if contrib is None:
        return None
    if layout.point_index is None:
        raise ValueError(f"Synapse layout {layout.id!r} is missing point_index.")
    syn_contrib = contrib
    if getattr(syn, "current_units", None) == "total":
        point_area = runtime.point_area[..., layout.point_index]
        syn_contrib = syn_contrib / point_area
    if getattr(syn, "current_sign", None) == "neuron":
        syn_contrib = -syn_contrib
    if hasattr(contrib, "unit"):
        contrib_point = u.Quantity(
            jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=u.get_mantissa(syn_contrib).dtype),
            syn_contrib.unit,
        )
        return contrib_point.at[..., layout.point_index].add(syn_contrib)
    contrib_point = jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=jnp.asarray(syn_contrib).dtype)
    return contrib_point.at[..., layout.point_index].add(syn_contrib)


def _clamp_density(runtime: CellRuntimeState, *, t):
    """Return ``(..., n_point) nA/cm^2`` clamp current density.

    Reads the pre-built :class:`ClampActiveTable`; no layout iteration
    in the hot path.

    Parameters
    ----------
    runtime : CellRuntimeState
        Runtime object that owns the clamp layouts and active-table.
    t : Quantity[time]
        Current simulation time.

    Returns
    -------
    Quantity
        Clamp current density in point space with shape
        ``runtime.pop_size + (runtime.n_point,)``.
    """
    table = runtime.clamp_active_table
    if table is None:
        return u.Quantity(jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=float), _CURRENT_DENSITY)

    currents_nA = runtime.evaluate_point_clamps(t=t).to_decimal(u.nA)
    active_density = currents_nA[..., table.ids] / table.area
    density = jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=float)
    density = density.at[..., table.ids].set(active_density)
    return u.Quantity(density, _CURRENT_DENSITY)
