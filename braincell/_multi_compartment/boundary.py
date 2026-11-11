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

"""Algebraic boundary rows for the derivative solvers of a cable :class:`Cell`.

A branch end or branch point is a point without membrane capacitance: its row
of the point-tree cable equation is algebraic,

.. math::

    0 = \\sum_k g_{bk}\\,(V_k - V_b) + I_b(V_b),

with ``k`` the CV midpoints joined to ``b`` by an edge of conductance
``g_bk`` and ``I_b`` the inward current of the clamps and synapses placed at
``b``. The CV axial operator of the derivative path eliminates these rows
assuming no input, which puts every boundary at the conductance-weighted mean
``V_b0`` of its neighbours. For a boundary carrying an input, this module
solves the row at each derivative evaluation and returns the missing term
``g_bk (V_b - V_b0) / C_k`` of each neighbouring CV. Two algebraic points are
never adjacent (every branch has at least one CV), so the rows decouple and
each is a scalar equation.

A model with no input on any algebraic point never reaches this code, so its
numerics are those of the bare CV operator.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute.cable import CableTopology, edge_conductance
from braincell._multi_compartment import currents

if TYPE_CHECKING:
    from braincell._compute.state import CellRuntimeState
    from braincell._multi_compartment.cell import Cell

__all__ = [
    "NEWTON_STEPS",
    "BoundaryRows",
    "build_boundary_rows",
    "point_voltage",
    "solve_boundary",
    "stage_point_voltage",
]

#: Newton steps per solve. A boundary current affine in ``V_b`` (every built-in
#: clamp and synapse) is solved exactly by the first step. For a nonlinear one,
#: the start value is off by ``O(dt)`` within a step and ``K`` steps leave an
#: ``O(dt ** 2 ** K)`` error; ``2 ** 3 = 8`` covers the highest-order
#: registered Runge-Kutta method. A fixed count (no ``while_loop``) keeps the
#: solve differentiable in reverse mode.
NEWTON_STEPS = 3

#: ``nA / mS`` in ``mV``.
_NA_PER_MS_IN_MV = 1e-3


@dataclass(frozen=True)
class BoundaryRows:
    """Static description of the algebraic points that carry a point input.

    Attributes
    ----------
    input_ids : numpy.ndarray
        ``(n_b,)`` sorted point ids of the algebraic points (not a CV
        midpoint) that carry a clamp or a synapse.
    edge_index : numpy.ndarray
        ``(n_e,)`` index into ``node_tree.edges`` of each edge with one end in
        ``input_ids``.
    edge_slot : numpy.ndarray
        ``(n_e,)`` position in ``input_ids`` of that end.
    edge_cv : numpy.ndarray
        ``(n_e,)`` CV whose midpoint is the other end.
    point_source_cv : numpy.ndarray
        ``(n_point,)`` CV whose voltage a point takes when no boundary value is
        tracked: its own CV for a midpoint, its representative CV otherwise.
    synapse_layout_ids : frozenset of int
        Synapse layouts with at least one row on ``input_ids``.
    topology : CableTopology or None
        Edge routing in point-id order, ``None`` without inputs.
    """

    input_ids: np.ndarray
    edge_index: np.ndarray
    edge_slot: np.ndarray
    edge_cv: np.ndarray
    point_source_cv: np.ndarray
    synapse_layout_ids: frozenset
    topology: CableTopology | None

    @property
    def has_inputs(self) -> bool:
        """Whether any algebraic point carries a clamp or a synapse."""
        return self.input_ids.size > 0


def build_boundary_rows(runtime: "CellRuntimeState", cvs) -> BoundaryRows:
    """Find the algebraic points with inputs and the edges that join them to CVs.

    Algebraic points are the points that are not a CV midpoint. A midpoint
    where a child branch attaches has ``Node.kind == "boundary"`` but keeps its
    membrane capacitance, so it is not one of them.

    Raises
    ------
    RuntimeError
        If an edge joins two algebraic points; the cable discretization never
        produces one.
    """
    node_tree = runtime.node_tree
    n_point = int(runtime.n_point)
    midpoints = np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)
    cv_of_point = np.full((n_point,), -1, dtype=np.int32)
    cv_of_point[midpoints] = np.arange(midpoints.size, dtype=np.int32)
    point_source_cv = np.asarray(runtime.point_to_representative_cv_np, dtype=np.int32).copy()
    point_source_cv[midpoints] = np.arange(midpoints.size, dtype=np.int32)

    inputs: set[int] = set()
    table = runtime.clamp_routing_table
    if table is not None:
        inputs.update(int(pid) for pid in np.asarray(table.boundary_ids).tolist())
    layout_ids = set()
    for layout, _ in runtime.iter_synapse_layouts():
        on_boundary = {int(pid) for pid in np.asarray(layout.point_index).tolist() if cv_of_point[pid] < 0}
        if on_boundary:
            inputs.update(on_boundary)
            layout_ids.add(int(layout.id))
    input_ids = np.asarray(sorted(inputs), dtype=np.int32)
    slot_of = {int(pid): slot for slot, pid in enumerate(input_ids.tolist())}

    edge_index, edge_slot, edge_cv = [], [], []
    for index, edge in enumerate(node_tree.edges):
        ends = (int(edge.parent_node_id), int(edge.child_node_id))
        for end, other in (ends, ends[::-1]):
            if end not in slot_of:
                continue
            if cv_of_point[other] < 0:
                raise RuntimeError(f"Point-tree edge {edge.id!r} joins two algebraic points ({end}, {other}).")
            edge_index.append(index)
            edge_slot.append(slot_of[end])
            edge_cv.append(int(cv_of_point[other]))

    topology = None
    if input_ids.size:
        topology = CableTopology.build(
            cvs=cvs,
            node_tree=node_tree,
            point_id_to_row=np.arange(n_point, dtype=np.int32),
        )
    return BoundaryRows(
        input_ids=input_ids,
        edge_index=np.asarray(edge_index, dtype=np.int32),
        edge_slot=np.asarray(edge_slot, dtype=np.int32),
        edge_cv=np.asarray(edge_cv, dtype=np.int32),
        point_source_cv=point_source_cv,
        synapse_layout_ids=frozenset(layout_ids),
        topology=topology,
    )


def solve_boundary(cell: "Cell", rows: BoundaryRows, V_cv, *, t, guess):
    """Solve the algebraic rows that carry an input at CV voltage ``V_cv``.

    Parameters
    ----------
    cell : Cell
        Initialized cable cell; conductances and capacitances are read from
        its current cable arrays, so runtime and trainable geometry apply.
    rows : BoundaryRows
        Output of :func:`build_boundary_rows` with ``rows.has_inputs``.
    V_cv : Quantity
        ``(..., n_cv)`` CV voltage.
    t : Quantity[time]
        Time passed to the clamp lookup.
    guess : array_like
        ``(..., n_b)`` start value in mV at ``rows.input_ids``.

    Returns
    -------
    correction : Quantity
        ``(..., n_cv)`` term in mV/ms to add to the derivative of ``V_cv``.
    V_b : jax.Array
        ``(..., n_b)`` boundary voltage in mV.
    """
    cable = cell._runtime.current_cable()
    g = edge_conductance(cable, rows.topology).to_decimal(u.mS)[..., rows.edge_index]
    capacitance = (cable.area * cable.cm).to_decimal(u.uF)[..., rows.edge_cv]
    V = jnp.asarray(V_cv.to_decimal(u.mV))
    n_b = rows.input_ids.size
    slot = rows.edge_slot

    # Total conductance and no-input voltage of every row. The weights are
    # exactly 1 at a branch tip, so there V_b0 is the neighbour's voltage.
    total = jnp.zeros(g.shape[:-1] + (n_b,), dtype=g.dtype).at[..., slot].add(g)
    weighted = (g / total[..., slot]) * V[..., rows.edge_cv]
    V_b0 = jnp.zeros(weighted.shape[:-1] + (n_b,), dtype=weighted.dtype).at[..., slot].add(weighted)

    base = V[..., rows.point_source_cv]

    def boundary_current(V_b):
        point = base.at[..., rows.input_ids].set(V_b)
        current = currents.point_input_current(
            cell,
            point_V=u.Quantity(point, u.mV),
            t=t,
            layout_ids=rows.synapse_layout_ids,
        )
        return current.to_decimal(u.nA)[..., rows.input_ids]

    # Newton on the offset from V_b0, delta = V_b - V_b0:
    #   total * delta = 1e-3 * I(V_b0 + delta)
    # Written this way a row whose input is zero gives delta == 0 exactly.
    delta = jnp.asarray(guess, dtype=V_b0.dtype) - V_b0
    for _ in range(NEWTON_STEPS):
        current, slope = jax.jvp(boundary_current, (V_b0 + delta,), (jnp.ones_like(delta),))
        delta = _NA_PER_MS_IN_MV * (current - slope * delta) / (total - _NA_PER_MS_IN_MV * slope)

    # mS * mV / uF = mV / ms
    flux = g * delta[..., slot] / capacitance
    correction = jnp.zeros(flux.shape[:-1] + V.shape[-1:], dtype=flux.dtype).at[..., rows.edge_cv].add(flux)
    return u.Quantity(correction, u.mV / u.ms), V_b0 + delta


def point_voltage(rows: BoundaryRows, V_cv, cached):
    """Return the point voltage the mechanisms, events and records read.

    Midpoints take their CV's voltage, algebraic points with an input the
    cached boundary voltage, and the other algebraic points the voltage of
    their representative CV (no mechanism sits there to read it).
    """
    point = u.math.asarray(V_cv)[..., rows.point_source_cv]
    if not rows.has_inputs:
        return point
    return point.at[..., rows.input_ids].set(cached[..., rows.input_ids])


def stage_point_voltage(rows: BoundaryRows, V_cv, V_b):
    """Return the point voltage of one derivative stage.

    As :func:`point_voltage`, but the algebraic points with an input carry
    ``V_b``, the ``(..., n_b)`` voltage in mV that :func:`solve_boundary`
    solved at ``V_cv``, instead of the cached value.
    """
    point = u.math.asarray(V_cv)[..., rows.point_source_cv]
    return point.at[..., rows.input_ids].set(u.Quantity(V_b, u.mV))
