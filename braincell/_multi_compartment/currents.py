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

"""Membrane-current summation pipeline for :class:`Cell`.

Responsibilities:

1. Evaluate sparse inputs, clamps, and placed synapses in point space.
2. Gather their contribution onto CV midpoint rows.
3. Evaluate painted density channels directly in CV space.
4. Return the combined CV-space current for the voltage update.
"""

import os
from typing import TYPE_CHECKING

import brainunit as u
import jax
import jax.numpy as jnp

from braincell._base import IonChannel, Synapse as RuntimeSynapse
from braincell._compute.state import CellRuntimeState
from braincell._compute import bridge

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
    with jax.named_scope("braincell:membrane_current:cv_to_point_for_point_mechanisms"):
        point_V = bridge.cv_to_point(V_cv, runtime)
    I_point = _point_mechanism_current(host, point_V=point_V, t=t)
    with jax.named_scope("braincell:membrane_current:point_to_cv"):
        I_cv = bridge.point_to_cv(I_point, runtime)
    return I_cv + _density_current_cv(host, V_cv=V_cv)


def total_membrane_current_point(
    host: "Cell",
    *,
    point_V,
    t,
):
    """Return ``(..., n_point)`` membrane current density in ``nA/cm^2``."""
    total_cv = total_membrane_current(
        host,
        V_cv=bridge.point_to_cv(point_V, host.runtime),
        t=t,
    )
    return bridge.cv_to_point(total_cv, host.runtime)


def _point_mechanism_current(host: "Cell", *, point_V, t):
    runtime = host.runtime
    with jax.named_scope("braincell:membrane_current:current_inputs"):
        zero_density = u.Quantity(
            jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=float),
            _CURRENT_DENSITY,
        )
        I_point = host.sum_current_inputs(zero_density, point_V)

    with jax.named_scope("braincell:membrane_current:clamp_density"):
        I_point = I_point + _clamp_density(runtime, t=t)

    with jax.named_scope("braincell:membrane_current:point_synapse_currents"):
        for key, ch in host.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(ch, RuntimeSynapse):
                continue
            with jax.named_scope(_scope_name("braincell:membrane_current:channel", key, ch)):
                layout_id = _layout_id_from_current_key(key)
                layout = runtime.layouts[layout_id]
                contrib_point = _synapse_contrib_to_point(runtime, layout, ch, point_V)
                if contrib_point is None:
                    continue
                I_point = I_point + contrib_point

    return I_point


def _density_current_cv(host: "Cell", *, V_cv):
    current = None
    with jax.named_scope("braincell:membrane_current:density_currents"):
        for key, channel in host.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if isinstance(channel, RuntimeSynapse):
                continue
            try:
                contribution = jax.named_call(
                    channel.current,
                    name=_call_name("braincell:membrane_current:channel_current", key, channel),
                )(V_cv)
            except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
                raise ValueError(f"Error computing current for ion channel {key!r}:\n{channel}\nError: {exc}") from exc
            if contribution is None:
                continue
            contribution = _profile_barrier_current(contribution)
            current = contribution if current is None else current + contribution
    if current is not None:
        return current
    return u.Quantity(jnp.zeros(V_cv.shape, dtype=float), _CURRENT_DENSITY)


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
    if layout.point_index is None:
        raise ValueError(f"Synapse layout {layout.id!r} is missing point_index.")
    if layout.population_index is None:
        local_voltage = point_V[..., layout.point_index]
    else:
        local_voltage = point_V[..., layout.population_index, layout.point_index]
    try:
        contrib = jax.named_call(
            syn.current,
            name=_call_name("braincell:membrane_current:synapse_current", (f"layout_{layout.id}",), syn),
        )(local_voltage)
    except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
        raise ValueError(f"Error computing current for synapse layout {layout.id!r}:\n{syn}\nError: {exc}") from exc
    if contrib is None:
        return None
    point_area = runtime.point_area[..., layout.point_index]
    syn_contrib = contrib / point_area
    if hasattr(contrib, "unit"):
        contrib_point = u.Quantity(
            jnp.zeros(point_V.shape, dtype=u.get_mantissa(syn_contrib).dtype),
            syn_contrib.unit,
        )
        if layout.population_index is not None:
            return contrib_point.at[..., layout.population_index, layout.point_index].add(syn_contrib)
        return contrib_point.at[..., layout.point_index].add(syn_contrib)
    contrib_point = jnp.zeros(point_V.shape, dtype=jnp.asarray(syn_contrib).dtype)
    if layout.population_index is not None:
        return contrib_point.at[..., layout.population_index, layout.point_index].add(syn_contrib)
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
