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

"""Continuous cable arrays and differentiable node-tree coefficient assembly.

Host lowering still chooses CVs and computes their reference geometry. This
module starts at those physical values; it does not differentiate morphology
clipping or define trainable geometry ownership. Integer edge roles remain host
metadata, while capacitances and conductances are computed from array inputs.
"""

from dataclasses import dataclass
from typing import NamedTuple

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np


class CableArrays(NamedTuple):
    """Physical arrays with CVs on the last axis and optional population axes.

    ``area`` is membrane area, ``cm`` is capacitance density, and the two
    resistance arrays describe the actual proximal/distal half-CV geometry.
    Leading axes broadcast across fields; a shared ``(n_cv,)`` reference can
    therefore be combined with population-specific capacitance or resistance.
    """

    area: object
    cm: object
    resistance_prox: object
    resistance_dist: object

    @classmethod
    def from_cvs(cls, cvs):
        """Transfer complete host vectors, never one device scalar per CV."""
        fields = (("area", u.cm**2), ("cm", u.uF / u.cm**2), ("r_axial_prox", u.ohm), ("r_axial_dist", u.ohm))
        host = np.asarray(
            [[getattr(cv, name).to_decimal(unit) for cv in cvs] for name, unit in fields],
            dtype=brainstate.environ.dftype(),
        )
        packed = jnp.asarray(host)
        return cls(*(u.Quantity(packed[i], unit) for i, (_, unit) in enumerate(fields)))


@dataclass(frozen=True)
class CableTopology:
    """Integer routing for a fixed point tree in solver row order."""

    n_point: int
    cv_rows: np.ndarray
    parent_rows: np.ndarray
    child_rows: np.ndarray
    role_half_indices: np.ndarray
    role_edge_ids: np.ndarray
    series_edges: np.ndarray

    @classmethod
    def build(cls, *, cvs, node_tree, point_id_to_row):
        point_id_to_row = np.asarray(point_id_to_row, dtype=np.int32)
        parent, child, halves, edge_ids, series = [], [], [], [], []
        n_cv = len(cvs)
        for edge_index, edge in enumerate(node_tree.edges):
            if not edge.roles:
                raise ValueError(f"Point-tree edge {edge.id!r} has no CV edge roles.")
            parent.append(point_id_to_row[edge.parent_node_id])
            child.append(point_id_to_row[edge.child_node_id])
            branch_ids = {cvs[role.cv_id].branch_id for role in edge.roles}
            series.append(len(edge.roles) > 1 and len(branch_ids) == 1)
            for role in edge.roles:
                halves.append(role.cv_id + (n_cv if role.half == "dist" else 0))
                edge_ids.append(edge_index)
        return cls(
            n_point=len(node_tree.nodes),
            cv_rows=point_id_to_row[node_tree.cv_to_mid_node_id],
            parent_rows=np.asarray(parent, dtype=np.int32),
            child_rows=np.asarray(child, dtype=np.int32),
            role_half_indices=np.asarray(halves, dtype=np.int32),
            role_edge_ids=np.asarray(edge_ids, dtype=np.int32),
            series_edges=np.asarray(series, dtype=bool),
        )


def axial_coefficients(cable: CableArrays, topology: CableTopology):
    """Return row capacitance (μF) and directed edge coefficients (ms⁻¹).

    Same-branch adjacent half-CVs combine in series; the remaining roles use
    parallel conductances, preserving the existing point-tree assembly rule.
    Algebraic rows use an arbitrary 1 μF scaling and have no membrane storage.
    Only initial host lowering may extract numeric Python values. This function
    is pure array arithmetic and supports JAX differentiation of every input.
    """
    capacitance, prox, dist = jnp.broadcast_arrays(
        (cable.area * cable.cm).to_decimal(u.uF),
        cable.resistance_prox.to_decimal(u.ohm),
        cable.resistance_dist.to_decimal(u.ohm),
    )
    batch_shape = capacitance.shape[:-1]
    row_c = jnp.ones(batch_shape + (topology.n_point,), dtype=capacitance.dtype)
    row_c = row_c.at[..., topology.cv_rows].set(capacitance)
    resistance = jnp.concatenate((prox, dist), axis=-1)
    role_r = resistance[..., topology.role_half_indices]
    n_edge = len(topology.parent_rows)
    sum_r = jnp.zeros(batch_shape + (n_edge,), dtype=resistance.dtype).at[..., topology.role_edge_ids].add(role_r)
    sum_g = jnp.zeros(batch_shape + (n_edge,), dtype=resistance.dtype).at[..., topology.role_edge_ids].add(1.0 / role_r)
    # S / μF converts to 1/ms by 1e3. Resistance inputs must be positive.
    conductance_ms = 1e3 * jnp.where(topology.series_edges, 1.0 / sum_r, sum_g)
    parent = conductance_ms / row_c[..., topology.parent_rows]
    child = conductance_ms / row_c[..., topology.child_rows]
    return u.Quantity(row_c, u.uF), u.Quantity(parent, u.ms**-1), u.Quantity(child, u.ms**-1)


def axial_matrix(cable: CableArrays, topology: CableTopology):
    """Assemble the mixed node-tree operator in ms⁻¹ without host conversion."""
    row_c, parent, child = axial_coefficients(cable, topology)
    p, c = topology.parent_rows, topology.child_rows
    parent, child = parent.to_decimal(u.ms**-1), child.to_decimal(u.ms**-1)
    matrix = jnp.zeros(parent.shape[:-1] + (topology.n_point, topology.n_point), dtype=parent.dtype)
    matrix = matrix.at[..., p, p].add(parent).at[..., p, c].add(-parent)
    matrix = matrix.at[..., c, c].add(child).at[..., c, p].add(-child)
    return u.Quantity(matrix, u.ms**-1), row_c
