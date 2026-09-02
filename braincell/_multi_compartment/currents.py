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

1. Seed point-space membrane current density with zeros.
2. Add registered current-input callables via :meth:`sum_current_inputs`.
3. Add clamp density from the precomputed clamp routing table.
4. Iterate channel currents.
5. Bridge point-space sum back to CV-space for the voltage update.
"""

from typing import TYPE_CHECKING

import brainunit as u
import jax
import jax.numpy as jnp

from braincell._base_channel import IonChannel, Synapse as RuntimeSynapse
from braincell._misc import (
    profile_barrier_current as _profile_barrier_current,
    profiler_call_name as _call_name,
    profiler_scope_name as _scope_name,
)
from braincell._compute.layouts import layout_id_from_key
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
                    layout_id = layout_id_from_key(key)
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


def _synapse_contrib_to_point(runtime: CellRuntimeState, layout, syn, point_V):
    local_voltage = layout.gather_points(point_V)
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
    else:
        contrib_point = jnp.zeros(point_V.shape, dtype=jnp.asarray(syn_contrib).dtype)
    return layout.scatter_add_points(contrib_point, syn_contrib)


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
