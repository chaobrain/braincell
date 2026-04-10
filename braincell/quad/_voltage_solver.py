# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Voltage solver for the active point-tree runtime.

Type responsibilities in this file:

- ``u.Quantity`` stays at the solver boundary where physical units matter.
  Examples: ``target.V.value``, CV capacitance, axial resistance, membrane
  derivative linearization.
- ``np.ndarray`` is only used for static topology metadata that is built once
  from the point tree or inside ``ensure_compile_time_eval`` blocks.
  Examples: row lookup tables, DHS edge ordering, parent lookup.
- ``jnp.ndarray`` is used for the numerical hot path executed every step.
  Examples: diagonal/off-diagonal coefficients, right-hand side vectors, and
  the DHS forward/backward elimination kernels.
"""

from dataclasses import dataclass

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import set_module_as
from ._registry import register_integrator

__all__ = [
    "dhs_voltage_step",
]


@dataclass(frozen=True)
class DHSStaticCache:
    n_point: int
    dynamic_rows_np: np.ndarray
    dynamic_rows_jnp: jnp.ndarray
    diag_base_jnp: jnp.ndarray
    diag_base_with_sentinel_jnp: jnp.ndarray
    lowers_with_sentinel_jnp: jnp.ndarray
    uppers_with_sentinel_jnp: jnp.ndarray
    parent_lookup_jnp: jnp.ndarray
    edges_jnp: jnp.ndarray
    level_offsets_np: np.ndarray
    backsub_indices_jnp: jnp.ndarray


@dataclass(frozen=True)
class DHSNumericState:
    diags: jnp.ndarray
    solves: jnp.ndarray
    lowers: jnp.ndarray
    uppers: jnp.ndarray
    parent_lookup: jnp.ndarray
    edges: jnp.ndarray
    dynamic_rows: jnp.ndarray
    n_point: int


@register_integrator(
    "dhs_voltage",
    category="voltage",
    description="Implicit-Euler DHS voltage solve on the active point-tree runtime.",
)
@set_module_as("braincell")
def dhs_voltage_step(target, t, dt, *args):
    """Advance midpoint voltages by one implicit-Euler DHS step.

    The public cell voltage lives on CV midpoints with shape ``[..., n_cv]``.
    DHS solves the linear system on point-tree rows with shape ``[batch, n_point]``
    plus one sentinel row used by the recursive doubling back-substitution.
    """
    if not hasattr(target, "point_tree") or not hasattr(target, "point_scheduling"):
        raise TypeError(f"dhs_voltage_step(...) requires a point-tree aware target, got {type(target)}.")

    point_tree = target.point_tree()
    scheduling = target.point_scheduling(algorithm="dhs")
    system = _point_tree_linear_system(target, point_tree=point_tree, scheduling=scheduling, dt=dt)
    static_cache = _get_dhs_static_cache(target, system)

    # Boundary quantities keep units. The hot path below converts them once to
    # fixed numerical units before entering the JAX kernels.
    V_n = target.V.value
    linear, const = _linear_and_const_term(target, V_n, *args)
    numeric = _build_dhs_numeric_state(
        V_n,
        linear,
        const,
        dt=dt,
        static_cache=static_cache,
    )
    diags, solves = comp_triang_raw(
        numeric.diags,
        numeric.solves,
        numeric.lowers,
        numeric.uppers,
        numeric.edges,
        static_cache.level_offsets_np,
    )
    solves = comp_backsub_raw(
        diags,
        solves,
        numeric.lowers,
        static_cache.backsub_indices_jnp,
    )
    target.V.value = _restore_midpoint_voltage(
        solves,
        dynamic_rows=numeric.dynamic_rows,
        target_shape=target.V.value.shape,
    )


def _point_tree_linear_system(target, *, point_tree, scheduling, dt) -> dict[str, object]:
    """Build the static axial system from the point tree.

    Returned arrays are static topology metadata:

    - ``dynamic_rows``: midpoint row indices
    - ``diag_base``: dimensionless diagonal coefficients ``dt * axial_diag``
    - ``lowers``/``uppers``: dimensionless off-diagonal coefficients
    - ``parent_lookup``/``edges``/``level_size``: integer scheduling metadata
    """
    n_point, dynamic_rows, axial_matrix = _build_point_tree_axial_matrix(
        target,
        point_tree=point_tree,
        point_id_to_row=scheduling.point_id_to_row,
    )
    dt_ms = _scalar_decimal(dt, u.ms)

    diag_base = dt_ms * np.diag(axial_matrix)
    lowers = np.zeros((n_point,), dtype=float)
    uppers = np.zeros((n_point,), dtype=float)
    for row, parent_row in enumerate(scheduling.parent_rows.tolist()):
        if parent_row < 0:
            continue
        lowers[row] = dt_ms * axial_matrix[row, parent_row]
        uppers[row] = dt_ms * axial_matrix[parent_row, row]

    parent_lookup = np.empty((n_point + 1,), dtype=np.int32)
    spurious_row = n_point
    parent_lookup[:n_point] = np.where(scheduling.parent_rows >= 0, scheduling.parent_rows, spurious_row)
    parent_lookup[spurious_row] = spurious_row
    edges, level_size = _build_dhs_edge_order(scheduling)

    return {
        "n_point": n_point,
        "dynamic_rows": dynamic_rows,
        "diag_base": diag_base,
        "lowers": lowers,
        "uppers": uppers,
        "edges": edges,
        "level_size": level_size,
        "parent_lookup": parent_lookup,
    }


def _build_point_tree_axial_matrix(target, *, point_tree, point_id_to_row) -> tuple[int, np.ndarray, np.ndarray]:
    """Assemble the mixed point-tree axial operator in ``ms^-1``."""
    n_point = len(point_tree.points)
    point_id_to_row = np.asarray(point_id_to_row, dtype=np.int32)
    cv_row_by_cv = point_id_to_row[point_tree.cv_midpoint_point_id]
    dynamic_rows = np.asarray([int(cv_row_by_cv[cv_id]) for cv_id in range(len(target.cvs))], dtype=np.int32)
    row_capacitance = _row_capacitance_scale(target, dynamic_rows=dynamic_rows, n_point=n_point)
    axial_matrix = np.zeros((n_point, n_point), dtype=float)

    for edge in point_tree.edges:
        parent_row = int(point_id_to_row[edge.parent_point_id])
        child_row = int(point_id_to_row[edge.child_point_id])
        conductance = _edge_conductance(edge=edge, cvs=target.cvs)

        # Dynamic rows use physical membrane capacitance. Algebraic boundary rows
        # use an arbitrary nonzero scale because the row is only used as a
        # constraint during static reduction.
        parent_coeff = _scalar_decimal(conductance / row_capacitance[parent_row], u.ms ** -1)
        child_coeff = _scalar_decimal(conductance / row_capacitance[child_row], u.ms ** -1)

        axial_matrix[parent_row, parent_row] += parent_coeff
        axial_matrix[parent_row, child_row] -= parent_coeff
        axial_matrix[child_row, child_row] += child_coeff
        axial_matrix[child_row, parent_row] -= child_coeff
    return n_point, dynamic_rows, axial_matrix


def _build_cv_axial_operator(target, *, point_tree, scheduling) -> np.ndarray:
    """Reduce the mixed point-tree axial system to a CV-midpoint operator."""
    _, dynamic_rows, axial_matrix = _build_point_tree_axial_matrix(
        target,
        point_tree=point_tree,
        point_id_to_row=scheduling.point_id_to_row,
    )
    dynamic_rows = np.asarray(dynamic_rows, dtype=np.int32)
    algebraic_rows = np.asarray(
        [row for row in range(axial_matrix.shape[0]) if row not in set(dynamic_rows.tolist())],
        dtype=np.int32,
    )
    if algebraic_rows.size == 0:
        reduced = axial_matrix[np.ix_(dynamic_rows, dynamic_rows)]
    else:
        dynamic_dynamic = axial_matrix[np.ix_(dynamic_rows, dynamic_rows)]
        dynamic_algebraic = axial_matrix[np.ix_(dynamic_rows, algebraic_rows)]
        algebraic_dynamic = axial_matrix[np.ix_(algebraic_rows, dynamic_rows)]
        algebraic_algebraic = axial_matrix[np.ix_(algebraic_rows, algebraic_rows)]
        reduced = dynamic_dynamic - dynamic_algebraic @ np.linalg.solve(algebraic_algebraic, algebraic_dynamic)
    return np.asarray(reduced, dtype=np.float32)


def _build_dhs_static_cache(system: dict[str, object]) -> DHSStaticCache:
    n_point = int(system["n_point"])
    dynamic_rows_np = np.asarray(system["dynamic_rows"], dtype=np.int32)
    diag_base_jnp = jnp.asarray(system["diag_base"], dtype=jnp.float32)
    level_size_np = np.asarray(system["level_size"], dtype=np.int32)
    level_offsets_np = np.cumsum(np.insert(level_size_np, 0, 0)).astype(np.int32, copy=False)
    parent_lookup_np = np.asarray(system["parent_lookup"], dtype=np.int32)
    diag_base_with_sentinel_jnp = jnp.concatenate(
        [diag_base_jnp, jnp.ones((1,), dtype=diag_base_jnp.dtype)],
        axis=0,
    )
    lower_base = jnp.asarray(system["lowers"], dtype=jnp.float32)
    upper_base = jnp.asarray(system["uppers"], dtype=jnp.float32)
    return DHSStaticCache(
        n_point=n_point,
        dynamic_rows_np=dynamic_rows_np,
        dynamic_rows_jnp=jnp.asarray(dynamic_rows_np),
        diag_base_jnp=diag_base_jnp,
        diag_base_with_sentinel_jnp=diag_base_with_sentinel_jnp,
        lowers_with_sentinel_jnp=jnp.concatenate([lower_base, jnp.zeros((1,), dtype=lower_base.dtype)], axis=0),
        uppers_with_sentinel_jnp=jnp.concatenate([upper_base, jnp.zeros((1,), dtype=upper_base.dtype)], axis=0),
        parent_lookup_jnp=jnp.asarray(parent_lookup_np),
        edges_jnp=jnp.asarray(system["edges"]),
        level_offsets_np=level_offsets_np,
        backsub_indices_jnp=jnp.asarray(_build_backsub_indices(parent_lookup_np, n_nodes=n_point)),
    )


def _get_dhs_static_cache(target, system: dict[str, object]) -> DHSStaticCache:
    runtime = getattr(target, "_compiled_runtime", None)
    cache = getattr(runtime, "dhs_static_cache", None)
    if cache is not None:
        return cache
    return _build_dhs_static_cache(system)


def _build_dhs_numeric_state(V_n, linear, const, *, dt, static_cache: DHSStaticCache) -> DHSNumericState:
    V_n, linear, const = [x.reshape((-1, V_n.shape[-1])) for x in (V_n, linear, const)]
    batch_size = V_n.shape[0]
    n_point = static_cache.n_point

    rhs_midpoint_mv = _to_decimal(V_n + dt * const, u.mV)
    linear_ms_inv = _to_decimal(linear, u.ms ** -1)
    dt_ms = _scalar_decimal(dt, u.ms)

    diags = jnp.broadcast_to(static_cache.diag_base_with_sentinel_jnp[None, :], (batch_size, n_point + 1))
    diags = diags.at[:, static_cache.dynamic_rows_jnp].add(1.0 - dt_ms * linear_ms_inv)

    solves = jnp.zeros((batch_size, n_point + 1), dtype=rhs_midpoint_mv.dtype)
    solves = solves.at[:, static_cache.dynamic_rows_jnp].set(rhs_midpoint_mv)

    return DHSNumericState(
        diags=diags,
        solves=solves,
        lowers=static_cache.lowers_with_sentinel_jnp,
        uppers=static_cache.uppers_with_sentinel_jnp,
        parent_lookup=static_cache.parent_lookup_jnp,
        edges=static_cache.edges_jnp,
        dynamic_rows=static_cache.dynamic_rows_jnp,
        n_point=n_point,
    )


def _restore_midpoint_voltage(solves: jnp.ndarray, *, dynamic_rows: jnp.ndarray, target_shape: tuple[int, ...]) -> object:
    return solves[:, dynamic_rows].reshape(target_shape) * u.mV


def _edge_conductance(*, edge, cvs) -> object:
    """Sum all half-CV conductances attached to one point-tree edge."""
    conductance = None
    for role in edge.cv_edges:
        cv = cvs[role.cv_id]
        resistance = cv.r_axial_prox if role.half == "prox" else cv.r_axial_dist
        value = 1.0 / resistance
        conductance = value if conductance is None else (conductance + value)
    if conductance is None:
        raise ValueError(f"Point-tree edge {edge.id!r} has no CV edge roles.")
    return conductance


def _row_capacitance_scale(target, *, dynamic_rows: np.ndarray, n_point: int) -> list[object]:
    """Return per-row membrane capacitance used in static axial assembly.

    Only CV midpoint rows carry membrane capacitance. Boundary rows are algebraic
    rows, so a dummy capacitance is used there because the corresponding axial
    coefficients are never consumed as membrane rows.
    """
    midpoint_capacitances = [cv.area * cv.cm for cv in target.cvs]
    if len(midpoint_capacitances) == 0:
        raise ValueError("Point-tree linear system requires at least one CV.")

    row_capacitance: list[object] = [1.0 * u.uF for _ in range(n_point)]
    for cv_id, row in enumerate(dynamic_rows.tolist()):
        row_capacitance[int(row)] = midpoint_capacitances[cv_id]
    return row_capacitance


def _build_dhs_edge_order(scheduling) -> tuple[np.ndarray, np.ndarray]:
    """Build leaf-to-root DHS elimination groups as static integer arrays."""
    edge_pairs: list[list[int]] = []
    level_size: list[int] = []
    for group in reversed(scheduling.groups):
        level_edges = []
        for row in group.tolist():
            parent_row = int(scheduling.parent_rows[row])
            if parent_row >= 0:
                level_edges.append([int(row), parent_row])
        if level_edges:
            edge_pairs.extend(level_edges)
            level_size.append(len(level_edges))

    if edge_pairs:
        return np.asarray(edge_pairs, dtype=np.int32), np.asarray(level_size, dtype=np.int32)
    return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)


def _check_comp_triang(diags, solves, lowers, uppers, edges):
    """Kernel contract check: only raw numerical arrays are allowed here."""
    assert not isinstance(diags, u.Quantity)
    assert not isinstance(solves, u.Quantity)
    assert not isinstance(lowers, u.Quantity)
    assert not isinstance(uppers, u.Quantity)
    assert not isinstance(edges, u.Quantity)
    assert diags.ndim == 2
    assert solves.ndim == 2
    assert lowers.ndim == 1
    assert uppers.ndim == 1
    assert lowers.shape[0] == diags.shape[1]
    assert uppers.shape[0] == diags.shape[1]
    assert edges.ndim == 2 and edges.shape[1] == 2


def comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets):
    """DHS forward elimination on raw ``jnp.ndarray`` inputs."""
    _check_comp_triang(diags, solves, lowers, uppers, edges)
    for i in range(level_offsets.shape[0] - 1):
        children = edges[level_offsets[i]:level_offsets[i + 1], 0]
        parent = edges[level_offsets[i]:level_offsets[i + 1], 1]
        lower_val = lowers[children]
        upper_val = uppers[children]
        child_diag = diags[:, children]
        child_solve = solves[:, children]

        multiplier = upper_val / child_diag
        diags = diags.at[:, parent].add(-lower_val * multiplier)
        solves = solves.at[:, parent].add(-child_solve * multiplier)
    return diags, solves


def _check_comp_backsub(diags, solves, lowers, backsub_indices):
    """Kernel contract check: recursive doubling only accepts raw arrays."""
    assert not isinstance(diags, u.Quantity)
    assert not isinstance(solves, u.Quantity)
    assert not isinstance(lowers, u.Quantity)
    assert not isinstance(backsub_indices, u.Quantity)
    assert diags.ndim == 2
    assert solves.ndim == 2
    assert lowers.ndim == 1
    assert diags.shape == solves.shape
    assert lowers.shape[0] == diags.shape[1]
    assert backsub_indices.ndim == 2
    assert backsub_indices.shape[1] == diags.shape[1]


def _build_backsub_indices(parent_lookup: np.ndarray, *, n_nodes: int) -> np.ndarray:
    """Precompute recursive-doubling ancestor jumps as static metadata."""
    parent_lookup = np.asarray(parent_lookup, dtype=np.int32)
    indices = []
    old_step = 0
    new_step = 1
    k_step_parent = np.arange(n_nodes + 1, dtype=np.int32)
    while new_step <= max(1, n_nodes):
        for _ in range(new_step - old_step):
            k_step_parent = parent_lookup[k_step_parent]
        old_step = new_step
        new_step = 2 * new_step
        indices.append(k_step_parent)
    return np.asarray(indices, dtype=np.int32)


def comp_backsub_raw(
    diags,
    solves,
    lowers,
    backsub_indices,
):
    """DHS recursive-doubling back substitution on raw ``jnp.ndarray`` inputs."""
    _check_comp_backsub(diags, solves, lowers, backsub_indices)
    lowers = lowers.at[0].set(0.0)
    lower_effect = -lowers / diags
    solve_effect = solves / diags

    for i in range(backsub_indices.shape[0]):
        k_step_parent = backsub_indices[i]
        solve_effect = solve_effect + lower_effect * solve_effect[:, k_step_parent]
        lower_effect = lower_effect * lower_effect[:, k_step_parent]

    return solve_effect


def _linear_and_const_term(target, V_n, *args):
    """Linearize membrane dynamics around ``V_n``.

    Returns two boundary quantities with units:

    - ``linear`` in ``ms^-1``
    - ``const`` in voltage/time
    """
    if hasattr(target, "_voltage_linearizer"):
        linearizer = target._voltage_linearizer()
    else:
        linearizer = brainstate.transform.vector_grad(
            target.compute_membrane_derivative,
            argnums=0,
            return_value=True,
            unit_aware=False,
        )
    linear, derivative = linearizer(V_n, *args)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    const = derivative - V_n * linear
    return linear, const


def _to_decimal(value: object, unit: object) -> jnp.ndarray:
    """Convert a quantity-like value to a hot-path ``jnp.ndarray``."""
    return jnp.asarray(value.to_decimal(unit), dtype=float)


def _scalar_decimal(value: object, unit: object) -> float:
    """Convert a scalar quantity to Python float for static assembly."""
    return float(np.asarray(value.to_decimal(unit), dtype=float))
