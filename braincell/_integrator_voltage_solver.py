# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""
Implementation of the backward Euler integrator for voltage dynamics in multicompartment models.
"""

import brainevent
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from ._misc import set_module_as
from ._protocol import DiffEqModule


@set_module_as('braincell')
def dhs_voltage_step(target, t, dt, *args):
    """
    Implicit euler solver with the `dendritic hierarchical scheduling` (DHS, Zhang et al., 2023).
    """

    # membrane potential at time n
    V_n = target.V.value
    diags = target.diags
    uppers = target.uppers
    lowers = target.lowers
    parent_lookup = target.parent_lookup
    internal_node_inds = target.internal_node_inds
    flipped_comp_edges, padded_edges, edge_masks = target.flipped_comp_edges

    n_nodes = len(diags)

    # linear and constant term
    linear, const = _linear_and_const_term(target, V_n, *args)
    V_linear = u.math.zeros(n_nodes) * u.get_unit(linear)
    V_linear = V_linear.at[internal_node_inds].set(-linear.ravel())
    V_const = u.math.zeros(n_nodes) * u.get_unit(const)
    V_const = V_const.at[internal_node_inds].set(const.ravel())
    V = u.math.zeros(n_nodes) * u.get_unit(V_n)
    V = V.at[internal_node_inds].set(V_n.ravel())

    # diags = I + dt * (G_diags + linear_term)
    diags = dt * (diags + V_linear)
    diags = diags.at[internal_node_inds].add(1.0)
    solves = V + dt * V_const
    uppers = dt * uppers
    lowers = dt * lowers

    # Add a spurious compartment that is modified by the masking.
    diags = u.math.concatenate([diags, u.math.asarray([1.0 * u.get_unit(diags)])])
    solves = u.math.concatenate([solves, u.math.asarray([0.0 * u.get_unit(solves)])])
    uppers = u.math.concatenate([uppers, u.math.asarray([0.0 * u.get_unit(uppers)])])
    lowers = u.math.concatenate([lowers, u.math.asarray([0.0 * u.get_unit(lowers)])])

    # Triangulate by unrolling the loop of the levels.
    steps = len(flipped_comp_edges)
    for i in range(steps):
        diags, solves = _comp_based_triang(i, (diags, solves, lowers, uppers, flipped_comp_edges))

    # Back substitute with recursive doubling.
    diags, solves = _comp_based_backsub_recursive_doubling(
        diags, solves, lowers, steps, n_nodes, parent_lookup
    )

    target.V.value = solves[internal_node_inds].reshape((1, -1))


def _dhs_matrix(target: DiffEqModule):
    with (jax.ensure_compile_time_eval()):
        Gmat_sorted, parent_rows, dhs_groups, segment2rowid, flipped_comp_edges = target.get_dhs_info(
            max_group_size=16, show=False
        )

        cm_segmid = target.cm
        area_segmid = target.area

        cm = jnp.ones(len(parent_rows)) * u.get_unit(cm_segmid)
        area = jnp.ones(len(parent_rows)) * u.get_unit(area_segmid)
        seg_mid_ids = jnp.array(list(segment2rowid.values()))
        cm[seg_mid_ids] = cm_segmid
        area[seg_mid_ids] = area_segmid

        Gmat_sorted = -Gmat_sorted / (cm * area)[:, u.math.newaxis]

        n = len(parent_rows)
        lowers = u.math.zeros(n) * u.get_unit(Gmat_sorted)
        uppers = u.math.zeros(n) * u.get_unit(Gmat_sorted)

        for i in range(n):
            p = parent_rows[i]
            if p == -1:
                lowers = lowers.at[i].set(0 * u.get_unit(Gmat_sorted))
                uppers = uppers.at[i].set(0 * u.get_unit(Gmat_sorted))
            else:
                lowers = lowers.at[i].set(Gmat_sorted[i, p])
                uppers = uppers.at[i].set(Gmat_sorted[p, i])

        diags = u.math.diag(Gmat_sorted)

    return diags, uppers, lowers, parent_rows, dhs_groups, segment2rowid, flipped_comp_edges, Gmat_sorted


def _comp_based_triang(index, carry):
    """Triangulate the quasi-tridiagonal system compartment by compartment."""
    diags, solves, lowers, uppers, flipped_comp_edges = carry

    # `flipped_comp_edges` has shape `(num_levels, num_comps_per_level, 2)`. We first
    # get the relevant level with `[index]` and then we get all children and parents
    # in the level.
    comp_edge = flipped_comp_edges[index]
    child = comp_edge[:, 0]
    parent = comp_edge[:, 1]
    lower_val = lowers[child]
    upper_val = uppers[child]
    child_diag = diags[child]
    child_solve = solves[child]

    # Factor that the child row has to be multiplied by.
    multiplier = upper_val / child_diag
    # Updates to diagonal and solve
    diags = diags.at[parent].add(-lower_val * multiplier)
    # jax.debug.print('diags_step= {}',diags)
    solves = solves.at[parent].add(-child_solve * multiplier)

    return (diags, solves)


def _comp_based_backsub_recursive_doubling(
    diags,
    solves,
    lowers,
    steps: int,
    n_nodes: int,
    parent_lookup: jnp.ndarray,
):
    """Backsubstitute with recursive doubling.

    This function contains a lot of math, so I will describe what is going on here:

    The matrix describes a system like:
    diag[n] * x[n] + lower[n] * x[parent] = solve[n]

    We rephrase this as:
    x[n] = solve[n]/diag[n] - lower[n]/diag[n] * x[parent].

    and we call variables as follows:
    solve/diag => solve_effect
    -lower/diag => lower_effect

    This gives:
    x[n] = solve_effect[n] + lower_effect[n] * x[parent].

    Recursive doubling solves this equation for `x` in log_2(N) steps. How?

    (1) Notice that lower_effect[n]=0, because x[0] has no parent.

    (2) In the first step, recursive doubling substitutes x[parent] into
    every equation. This leads to something like:
    x[n] = solve_effect[n] + lower_effect[n] * (solve_effect[parent] + ...
    ...lower_effect[parent] * x[parent[parent]])

    Abbreviate this as:
    new_solve_effect[n] = solve_effect[n] + lower_effect[n] * solve_effect[parent]
    new_lower_effect[n] = lower_effect[n] + lower_effect[parent]
    x[n] = new_solve_effect[n] + new_lower_effect[n] * x[parent[parent]]
    Importantly, every node n is now a function of its two-step parent.

    (3) In the next step, recursive doubling substitutes x[parent[parent]].
    Since x[parent[parent]] already depends on its own _two-step_ parent,
    every node then depends on its four step parent. This introduces the
    log_2 scaling.

    (4) The algorithm terminates when all `new_lower_effect=0`. This
    naturally happens because `lower_effect[0]=0`, and the recursion
    keeps multiplying new_lower_effect with the `lower_effect[parent]`.
    """

    # Why `lowers = lowers.at[0].set(0.0)`? During triangulation (and the
    # cpu-optimized solver), we never access `lowers[0]`. Its value should
    # be zero (because the zero-eth compartment does not have a `lower`), but
    # it is not for coding convenience in the other solvers. For the recursive
    # doubling solver below, we do use lowers[0], so we set it to the value
    # it should have anyways: 0.
    lowers = lowers.at[0].set(0.0 * u.get_unit(lowers))

    # Rephrase the equations as a recursion.
    # x[n] = solve[n]/diag[n] - lower[n]/diag[n] * x[parent].
    # x[n] = solve_effect[n] + lower_effect[n] * x[parent].
    lower_effect = -lowers / diags
    solve_effect = solves / diags

    step = 1
    while step <= steps:
        # For each node, get its k-step parent, where k=`step`.
        k_step_parent = u.math.arange(n_nodes + 1)
        for _ in range(step):
            k_step_parent = parent_lookup[k_step_parent]

        # Update.
        solve_effect = lower_effect * solve_effect[k_step_parent] + solve_effect
        lower_effect *= lower_effect[k_step_parent]
        step *= 2

    # We have to return a `diags` because the final solution is computed as
    # `solves/diags` (see `step_voltage_implicit_with_dhs_solve`). For recursive
    # doubling, the solution should just be `solve_effect`, so we define diags as
    # 1.0 so the division has no effect.
    diags = u.math.ones_like(solve_effect) * u.get_unit(solve_effect)
    solves = solve_effect
    return diags, solves


@set_module_as('braincell')
def dense_voltage_step():
    """
    Implicit euler solver implementation by solving the dense matrix system.
    """
    pass


def _dense_solve_v(
    Laplacian_matrix: brainstate.typing.ArrayLike,
    D_linear: brainstate.typing.ArrayLike,
    D_const: brainstate.typing.ArrayLike,
    dt: brainstate.typing.ArrayLike,
    V_n: brainstate.typing.ArrayLike
):
    """
    Set the left-hand side (lhs) and right-hand side (rhs) of the implicit equation:
    V^{n+1} (I + dt*(L_matrix + D_linear)) = V^{n} + dt*D_const

    Parameters:
    - Laplacian_matrix: The Laplacian matrix L describing diffusion between compartments
    - D_linear: Diagonal matrix of linear coefficients for voltage-dependent currents
                D_linear = diag(∑g_i^{t+dt}) where g_i^t are time-dependent conductances
    - D_const: Vector of constant terms from voltage-independent currents
               D_const = ∑(g_i^{t+dt}·E_i) +I^{t+dt}_ext where E_i are reversal potentials
    - V_n: Membrane potential vector at current time step n

    Returns:
    - V^{n+1} = lhs^{-1} * rhs

    Notes:
    - This function constructs the matrices for solving the next time step
      in a compartmental model using an implicit Euler method.
    - The Laplacian matrix accounts for passive diffusion between compartments.
    - D_linear and D_const incorporate active membrane currents (ionic, synaptic, external).
    - The implicit formulation ensures numerical stability for stiff systems.
    """

    # Compute the left-hand side matrix
    # lhs = I + dt*(Laplacian_matrix + D_linear)
    n_compartments = Laplacian_matrix.shape[0]

    # dense method
    I_matrix = jnp.eye(n_compartments)
    lhs = I_matrix + dt * (Laplacian_matrix + u.math.diag(D_linear))
    rhs = V_n + dt * D_const
    print(lhs.shape, rhs.shape)
    result = u.math.linalg.solve(lhs, rhs)
    return result


@set_module_as('braincell')
def sparse_voltage_step(target, t, dt, *args):
    """
    Implicit euler solver implementation by solving the sparse matrix system.
    """
    from ._multi_compartment import MultiCompartment
    assert isinstance(target, MultiCompartment), (
        'The target should be a MultiCompartment for the sparse integrator. '
    )

    # membrane potential at time n
    V_n = target.V.value

    # laplacian matrix
    L_matrix = _laplacian_matrix(target)

    # linear and constant term
    linear, const = _linear_and_const_term(target, V_n, *args)

    # solve the membrane potential at time n+1
    # -linear cause from left to right, the sign changed
    target.V.value = _sparse_solve_v(L_matrix, -linear, const, dt, V_n)


def _sparse_solve_v(
    Laplacian_matrix: brainevent.CSR,
    D_linear,
    D_const,
    dt: brainstate.typing.ArrayLike,
    V_n: brainstate.typing.ArrayLike
):
    r"""
    Set the left-hand side (lhs) and right-hand side (rhs) of the implicit equation:

    $$
    V^{n+1} (I + dt*(\mathrm{L_matrix} + \mathrm{D_linear})) = V^{n} + dt*\mathrm{D_const}
    $$

    Parameters:
    - Laplacian_matrix: The Laplacian matrix L describing diffusion between compartments
    - D_linear: Diagonal matrix of linear coefficients for voltage-dependent currents
                D_linear = diag(∑g_i^{t+dt}) where g_i^t are time-dependent conductances
    - D_const: Vector of constant terms from voltage-independent currents
               D_const = ∑(g_i^{t+dt}·E_i) +I^{t+dt}_ext where E_i are reversal potentials
    - V_n: Membrane potential vector at current time step n

    Returns:
    - V^{n+1} = lhs^{-1} * rhs

    Notes:
    - This function constructs the matrices for solving the next time step
      in a compartmental model using an implicit Euler method.
    - The Laplacian matrix accounts for passive diffusion between compartments.
    - D_linear and D_const incorporate active membrane currents (ionic, synaptic, external).
    - The implicit formulation ensures numerical stability for stiff systems.
    """

    # Compute the left-hand side matrix
    # lhs = I + dt*(Laplacian_matrix + D_linear)
    lhs = (dt * Laplacian_matrix).diag_add(dt * D_linear.reshape(-1) + 1)

    # Compute the right-hand side vector: rhs = V_n + dt*D_const
    rhs = V_n + dt * D_const
    result = lhs.solve(rhs.reshape(-1)).reshape((1, -1))
    return result


def _laplacian_matrix(target: DiffEqModule) -> brainevent.CSR:
    """
    Construct the Laplacian matrix L = diag(G'*1) - G' for the given target,
    where G' = G/(area*cm) is the normalized conductance matrix.

    Parameters:
        target: A DiffEqModule instance containing compartmental model parameters

    Returns:
        L_matrix: The Laplacian matrix representing the conductance term
                  of the compartmental model's differential equations

    Notes:
        - Computes the Laplacian matrix which describes the electrical conductance
          between compartments in a compartmental model.
        - The diagonal elements are set to the sum of the respective row's
          off-diagonal elements to ensure conservation of current.
        - The normalization by (area*cm) accounts for compartment geometry and membrane properties.
    """
    from ._multi_compartment import MultiCompartment
    target: MultiCompartment

    with jax.ensure_compile_time_eval():
        # Extract model parameters
        cm = target.cm
        area = target.area
        G_matrix = target.conductance_matrix  # TODO
        n_compartment = target.n_compartment

        # Compute negative normalized conductance matrix: element-wise division by (cm * area)
        L_matrix = -G_matrix / (cm * area)[:, u.math.newaxis]

        # Set diagonal elements to enforce Kirchhoff's current law
        # This constructs the Laplacian matrix L
        L_matrix = L_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(L_matrix, axis=1))

        # convert to CSR format
        L_matrix = brainevent.CSR.fromdense(L_matrix)

    return L_matrix


def _linear_and_const_term(target: DiffEqModule, V_n, *args):
    """
    get the linear and constant term of voltage.
    """
    from ._multi_compartment import MultiCompartment
    assert isinstance(target, MultiCompartment), (
        'The target should be a MultiCompartment for the sparse integrator. '
    )

    # compute the linear and derivative term
    linear, derivative = brainstate.transform.vector_grad(
        target.compute_membrane_derivative, argnums=0, return_value=True, unit_aware=False,
    )(V_n, *args)

    # Convert linearization to a unit-aware quantity
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))

    # Compute constant term
    const = derivative - V_n * linear
    return linear, const
