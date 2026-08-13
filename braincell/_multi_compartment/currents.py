"""Membrane-current summation pipeline for :class:`Cell`.

Responsibilities:

1. Seed point-space membrane current density with zeros.
2. Add registered current-input callables via :meth:`sum_current_inputs`.
3. Add clamp density from the precomputed clamp routing table.
4. Iterate channel currents.
5. Bridge point-space sum back to CV-space for the voltage update.
"""

import os
from typing import TYPE_CHECKING

import brainunit as u
import jax
import jax.numpy as jnp

from braincell._base import IonChannel, Synapse as RuntimeSynapse
from braincell._compute.runtime import CellRuntimeState
from . import bridge

if TYPE_CHECKING:
    from .cell import Cell

__all__ = ["total_membrane_current", "total_membrane_current_point"]

_CURRENT_DENSITY = u.nA / u.cm**2


def total_membrane_current(
    host: "Cell",
    *,
    V_cv,
    t,
):
    """Return ``(..., n_cv)`` membrane current density in ``nA/cm^2``."""
    runtime = host.runtime
    with jax.named_scope("braincell:membrane_current:cv_to_point"):
        point_V = bridge.cv_to_point(V_cv, runtime)

    I_point = total_membrane_current_point(host, point_V=point_V, t=t)

    with jax.named_scope("braincell:membrane_current:point_to_cv"):
        return bridge.point_to_cv(I_point, runtime)


def total_membrane_current_point(
    host: "Cell",
    *,
    point_V,
    t,
):
    """Return ``(..., n_point)`` membrane current density in ``nA/cm^2``."""
    runtime = host.runtime
    with jax.named_scope("braincell:membrane_current:current_inputs"):
        zero_density = u.Quantity(
            jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=float),
            _CURRENT_DENSITY,
        )
        I_point = host.sum_current_inputs(zero_density, point_V)

    with jax.named_scope("braincell:membrane_current:clamp_density"):
        I_point = I_point + _clamp_density(runtime, t=t)

    with jax.named_scope("braincell:membrane_current:channel_currents"):
        for key, ch in host.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            with jax.named_scope(_scope_name("braincell:membrane_current:channel", key, ch)):
                if isinstance(ch, RuntimeSynapse):
                    layout_id = _layout_id_from_current_key(key)
                    layout = runtime.layouts[layout_id]
                    contrib_point = _synapse_contrib_to_point(runtime, layout, ch, point_V)
                    if contrib_point is None:
                        continue
                    I_point = I_point + contrib_point
                    continue
                try:
                    contrib = jax.named_call(
                        ch.current,
                        name=_call_name("braincell:membrane_current:channel_current", key, ch),
                    )(point_V)
                except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
                    raise ValueError(f"Error computing current for ion channel {key!r}:\n{ch}\nError: {exc}") from exc
                if contrib is None:
                    continue
                I_point = I_point + _profile_barrier_current(contrib)

    return I_point


def _layout_id_from_current_key(key) -> int:
    if len(key) == 0:
        raise ValueError(f"Expected runtime object key ending with 'layout_<id>', got {key!r}.")
    last = key[-1]
    if not isinstance(last, str) or not last.startswith("layout_"):
        raise ValueError(f"Expected runtime object key ending with 'layout_<id>', got {key!r}.")
    return int(last.split("_", 1)[1])


def _scope_name(prefix: str, path, node) -> str:
    """Build a stable, profiler-safe internal JAX scope name."""
    path_name = "_".join(str(part) for part in path) if path else "root"
    class_name = type(getattr(node, "_channel", node)).__name__
    raw = f"{prefix}:{path_name}:{class_name}"
    cleaned = "".join(ch if ch.isalnum() or ch in ":_" else "_" for ch in raw)
    return cleaned[:180]


def _call_name(prefix: str, path, node) -> str:
    """Build a profiler-safe ``jax.named_call`` name."""
    return _scope_name(prefix, path, node).replace(":", "_")


def _profile_barrier_current(current):
    """Optionally split membrane-current HLO for profiler attribution."""
    if os.environ.get("BRAINCELL_PROFILE_SPLIT_CURRENTS") != "1":
        return current
    if hasattr(current, "unit"):
        return u.Quantity(jax.lax.optimization_barrier(u.get_mantissa(current)), current.unit)
    return jax.lax.optimization_barrier(current)


def _synapse_contrib_to_point(runtime: CellRuntimeState, layout, syn, point_V):
    try:
        contrib = jax.named_call(
            syn.current,
            name=_call_name("braincell:membrane_current:synapse_current", (f"layout_{layout.id}",), syn),
        )(point_V[..., layout.point_index])
    except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
        raise ValueError(f"Error computing current for synapse layout {layout.id!r}:\n{syn}\nError: {exc}") from exc
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

    Reads the pre-built midpoint entries from the clamp routing table; no
    layout iteration is needed in the hot path.

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
    table = runtime.clamp_routing_table
    if table is None or len(table.midpoint_ids) == 0:
        return u.Quantity(jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=float), _CURRENT_DENSITY)

    currents_nA = runtime.evaluate_point_clamps(t=t, point_ids=table.midpoint_ids).to_decimal(u.nA)
    active_density = currents_nA[..., table.midpoint_ids] / table.midpoint_area
    density = jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=float)
    density = density.at[..., table.midpoint_ids].set(active_density)
    return u.Quantity(density, _CURRENT_DENSITY)
