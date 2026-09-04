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

1. Preserve the CV-space current path used by derivative solvers.
2. Evaluate the staggered/DHS membrane rate directly in full point space.
3. Keep density mechanisms on CV midpoints and point mechanisms at their
   declared electrical points.
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

__all__ = ["total_membrane_current", "total_membrane_current_point", "total_membrane_rate_point"]

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


def total_membrane_rate_point(host: "Cell", *, point_V, point_capacitance, t):
    """Return the staggered membrane contribution on every point-tree row.

    Density currents are evaluated only at CV midpoints. Synapses and clamps
    remain absolute point currents and are normalized by the same row scale as
    the DHS axial equation, including on algebraic boundary rows.
    """

    runtime = host.runtime
    V_cv = bridge.point_to_cv(point_V, runtime)
    with jax.named_scope("braincell:point_membrane_rate:density"):
        rate_cv = _density_current_cv(host, V_cv=V_cv) / host.C

    with jax.named_scope("braincell:point_membrane_rate:current_inputs"):
        zero_density = u.Quantity(
            jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=u.get_mantissa(point_V).dtype),
            _CURRENT_DENSITY,
        )
        input_density = host.sum_current_inputs(zero_density, point_V)
        rate_cv = rate_cv + bridge.point_to_cv(input_density, runtime) / host.C

    rate_point = bridge.cv_to_point(rate_cv, runtime)
    point_current = host._solver_clamp_point_current(t=t)
    with jax.named_scope("braincell:point_membrane_rate:synapses"):
        for key, synapse in host.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(synapse, RuntimeSynapse):
                continue
            layout_id = layout_id_from_key(key)
            layout = runtime.layouts[layout_id]
            contribution = _synapse_absolute_current_point(runtime, layout, synapse, point_V)
            if contribution is not None:
                point_current = point_current + contribution

    return rate_point + point_current / point_capacitance


def _point_mechanism_current(host: "Cell", *, point_V, t):
    runtime = host.runtime
    with jax.named_scope("braincell:membrane_current:current_inputs"):
        zero_density = u.Quantity(
            jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=u.get_mantissa(point_V).dtype),
            _CURRENT_DENSITY,
        )
        I_point = host.sum_current_inputs(zero_density, point_V)

    with jax.named_scope("braincell:membrane_current:clamp_density"):
        I_point = I_point + _clamp_density(host, t=t, dtype=u.get_mantissa(point_V).dtype)

    with jax.named_scope("braincell:membrane_current:point_synapse_currents"):
        for key, ch in host.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(ch, RuntimeSynapse):
                continue
            with jax.named_scope(_scope_name("braincell:membrane_current:channel", key, ch)):
                layout_id = layout_id_from_key(key)
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
    return u.Quantity(jnp.zeros(V_cv.shape, dtype=u.get_mantissa(V_cv).dtype), _CURRENT_DENSITY)


def _synapse_contrib_to_point(runtime: CellRuntimeState, layout, syn, point_V):
    contrib_point = _synapse_absolute_current_point(runtime, layout, syn, point_V)
    if contrib_point is None:
        return None
    point_area = runtime.point_area
    return contrib_point / point_area


def _synapse_absolute_current_point(runtime: CellRuntimeState, layout, syn, point_V):
    """Scatter one runtime synapse's inward-positive absolute point current."""

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
    if hasattr(contrib, "unit"):
        contrib_point = u.Quantity(
            jnp.zeros(point_V.shape, dtype=u.get_mantissa(contrib).dtype),
            contrib.unit,
        )
    else:
        contrib_point = jnp.zeros(point_V.shape, dtype=jnp.asarray(contrib).dtype)
    return layout.scatter_add_points(contrib_point, contrib)


def _clamp_density(host: "Cell", *, t, dtype):
    """Return ``(..., n_point) nA/cm^2`` clamp current density.

    Reads the pre-built midpoint entries from the clamp routing table; no
    layout iteration is needed in the hot path.

    Parameters
    ----------
    host : Cell
        Cell owning the prepared per-step clamp current and routing table.
    t : Quantity[time]
        Solver stage time. The prepared clamp value remains fixed for the
        enclosing main step.

    Returns
    -------
    Quantity
        Clamp current density in point space with shape
        ``runtime.pop_size + (runtime.n_point,)``.
    """
    runtime = host.runtime
    table = runtime.clamp_routing_table
    if table is None or len(table.midpoint_ids) == 0:
        return u.Quantity(jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=dtype), _CURRENT_DENSITY)

    currents_nA = host._solver_clamp_point_current(t=t).to_decimal(u.nA)
    active_density = currents_nA[..., table.midpoint_ids] / table.midpoint_area
    density = jnp.zeros(runtime.pop_size + (runtime.n_point,), dtype=dtype)
    density = density.at[..., table.midpoint_ids].set(active_density)
    return u.Quantity(density, _CURRENT_DENSITY)
