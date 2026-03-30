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

# -*- coding: utf-8 -*-

"""
Implementation of the backward Euler integrator for voltage dynamics in multicompartment models.
"""

import brainevent
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import set_module_as
from ._protocol import DiffEqModule


@set_module_as('braincell')
def dhs_voltage_step(target, t, dt, *args):
    """
    Implicit Euler solver for multi-compartment neurons using 
    Dendritic Hierarchical Scheduling (DHS).

    Purpose:
        Advance the membrane potential V by one timestep dt 
        with an implicit scheme tailored for tree-structured morphologies.

    Steps:
        1. Ensure branch-tree representation is available, and extract
           morphology metadata: diagonals, upper/lower off-diagonals,
           parent indices, internal node indices, and flipped edges.
        2. Extract the current membrane potential V and compute 
           linear and constant contributions from all ion channels and synapses.
        3. Reshape all vectors to consistent batch shape (P, Nseg) and 
           scatter values into full node arrays including boundary nodes.
        4. Assemble the implicit Euler system:
               - Scale diagonals, uppers, and lowers by dt.
               - Add unit diagonal to internal nodes to include I-term.
               - Construct the RHS vector for the linear system.
        5. Append a "virtual" or spurious node to improve numerical stability 
           and simplify boundary handling.
        6. Solve the linear system for all populations in parallel 
           using DHS forward elimination + back-substitution (via vmap).
        7. Extract internal node results and write them back to 
           target.V.value, preserving the original shape.
    """

    # --- Step 1: Extract morphology and solver metadata ---
    if not hasattr(target.morphology, 'branch_tree'):
        with jax.ensure_compile_time_eval():
            target.morphology.to_branch_tree()
    tree = target.morphology.branch_tree
    diags, uppers, lowers, parent_lookup, internal_node_inds, flipped_comp_edges, edges, level_sizes, level_start = (
        tree.diags,
        tree.uppers,
        tree.lowers,
        tree.parent_lookup,
        tree.internal_node_inds,
        tree.flipped_comp_edges,
        tree.edges,
        tree.level_size,
        tree.level_start
    )
    n_nodes = len(diags)  # total number of nodes including boundaries
    # --- Step 2: Get current membrane potential and compute linear/constant terms ---
    V_n = target.V.value  # (P, Nseg)
    linear, const = _linear_and_const_term(target, V_n, *args)

    # --- Step 3: Reshape vectors and scatter to full node arrays ---
    V_n, linear, const = [x.reshape((-1, V_n.shape[-1])) for x in (V_n, linear, const)]
    P = V_n.shape[0]  # population size
    V, V_linear, V_const = [
        u.math.zeros((P, n_nodes), unit=u.get_unit(val)).at[:, internal_node_inds].set(val)
        for val in (V_n, -linear, const)
    ]

    # --- Step 4: Build implicit Euler system matrices ---
    diags = (dt * (diags + V_linear)).at[:, internal_node_inds].add(1.0)  # scale diagonals + unit I-term
    solves = V + dt * V_const  # RHS vector
    uppers = dt * uppers  # scale upper off-diagonal
    lowers = dt * lowers  # scale lower off-diagonal

    # --- Step 5: Append virtual/spurious compartment for stability ---
    diags = u.math.concatenate([diags, u.math.ones((P, 1)) * u.get_unit(diags)], axis=1)
    solves = u.math.concatenate([solves, u.math.zeros((P, 1)) * u.get_unit(solves)], axis=1)
    lowers = u.math.concatenate([lowers, u.math.zeros((), dtype=lowers.dtype)])
    uppers = u.math.concatenate([uppers, u.math.zeros((), dtype=uppers.dtype)])

    # --- Step 6: Solve the linear system for all populations in batch ---
    solves = solves.to_decimal(u.mV)
    n_steps = len(flipped_comp_edges)
    # diags, solves = comp_triang_call(diags, solves, lowers, uppers, edges)
    # solves = comp_backsub_call(diags, solves, lowers, parent_lookup, n_nodes=n_nodes, n_steps=n_steps)
    # diags, solves = comp_triang_call_v2(diags, solves, lowers, uppers, edges, level_sizes)
    diags, solves = comp_triang_raw(diags, solves, lowers, uppers, edges, level_sizes)
    solves = comp_backsub_raw(diags, solves, lowers, parent_lookup, n_nodes=n_nodes, n_steps=n_steps)
    # --- Step 7: Write back results for internal nodes only ---
    target.V.value = solves[:, internal_node_inds].reshape(target.V.value.shape) * u.mV


def _check_comp_triang(diags, solves, lowers, uppers, edges):
    assert not isinstance(diags, u.Quantity)
    assert not isinstance(solves, u.Quantity)
    assert not isinstance(lowers, u.Quantity)
    assert not isinstance(uppers, u.Quantity)
    assert not isinstance(edges, u.Quantity)
    assert diags.ndim == 2, 'diags should be 2D'
    assert solves.ndim == 2, 'solves should be 2D'
    assert lowers.ndim == 1, 'lowers should be 1D'
    assert uppers.ndim == 1, 'uppers should be 1D'

    assert lowers.shape[0] == diags.shape[1], 'lowers should have same length as diags.shape[1]'
    assert uppers.shape[0] == diags.shape[1], 'uppers should have same length as diags.shape[1]'
    assert edges.ndim == 2 and edges.shape[1] == 2


def comp_triang_raw(diags, solves, lowers, uppers, edges, level_sizes):
    _check_comp_triang(diags, solves, lowers, uppers, edges)

    with jax.ensure_compile_time_eval():
        level_sizes = np.cumsum(np.insert(level_sizes, 0, 0))
    for i in range(level_sizes.shape[0] - 1):
        children = edges[level_sizes[i]:level_sizes[i + 1], 0]
        parent = edges[level_sizes[i]:level_sizes[i + 1], 1]
        lower_val = lowers[children]
        upper_val = uppers[children]
        child_diag = diags[:, children]
        child_solve = solves[:, children]

        # Factor that the child row has to be multiplied by.
        multiplier = upper_val / child_diag

        # Updates to diagonal and solve
        diags = diags.at[:, parent].add(-lower_val * multiplier)
        solves = solves.at[:, parent].add(-child_solve * multiplier)
    return diags, solves


def _check_comp_backsub(diags, solves, lowers, parent_lookup):
    assert not isinstance(diags, u.Quantity)
    assert not isinstance(solves, u.Quantity)
    assert not isinstance(lowers, u.Quantity)
    assert not isinstance(parent_lookup, u.Quantity)
    assert diags.ndim == 2, 'diags should be 2D'
    assert solves.ndim == 2, 'solves should be 2D'
    assert lowers.ndim == 1, 'lowers should be 1D'
    assert diags.shape == solves.shape, 'diags and solves should have the same shape'
    assert lowers.shape[0] == diags.shape[1], 'lowers should have same length as diags.shape[1]'
    assert parent_lookup.ndim == 1, 'parent_lookup should be 1D'
    assert parent_lookup.shape[0] == diags.shape[1], 'parent_lookup should have same length as diags.shape[1]'


def _get_index_comp_backsub(parent_lookup, n_steps, n_nodes):
    with jax.ensure_compile_time_eval():
        parent_lookup = np.asarray(parent_lookup)
        indices = []
        old_step = 0
        new_step = 1
        k_step_parent = np.arange(n_nodes + 1)
        while new_step <= n_steps:
            for _ in range(new_step - old_step):
                k_step_parent = parent_lookup[k_step_parent]
            old_step = new_step
            new_step = 2 * new_step
            indices.append(k_step_parent)
        indices = np.asarray(indices)
    return indices


def comp_backsub_raw(
    diags,
    solves,
    lowers,
    parent_lookup,
    *,
    n_nodes: int,
    n_steps: int,
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
    _check_comp_backsub(diags, solves, lowers, parent_lookup)
    indices = _get_index_comp_backsub(parent_lookup, n_steps, n_nodes)

    # Why `lowers = lowers.at[0].set(0.0)`? During triangulation (and the
    # cpu-optimized solver), we never access `lowers[0]`. Its value should
    # be zero (because the zero-eth compartment does not have a `lower`), but
    # it is not for coding convenience in the other solvers. For the recursive
    # doubling solver below, we do use lowers[0], so we set it to the value
    # it should have anyways: 0.
    lowers = lowers.at[0].set(0.0)

    # Rephrase the equations as a recursion.
    # x[n] = solve[n]/diag[n] - lower[n]/diag[n] * x[parent].
    # x[n] = solve_effect[n] + lower_effect[n] * x[parent].
    lower_effect = -lowers / diags
    solve_effect = solves / diags

    for i in range(indices.shape[0]):
        k_step_parent = indices[i]
        solve_effect = solve_effect + lower_effect * solve_effect[:, k_step_parent]
        lower_effect = lower_effect * lower_effect[:, k_step_parent]

    # We have to return a `diags` because the final solution is computed as
    # `solves/diags` (see `step_voltage_implicit_with_dhs_solve`). For recursive
    # doubling, the solution should just be `solve_effect`, so we define diags as
    # 1.0 so the division has no effect.
    return solve_effect


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
    from braincell._multi_compartment import MultiCompartment
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
    from braincell._multi_compartment import MultiCompartment
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
    from braincell._multi_compartment import MultiCompartment
    assert isinstance(target, MultiCompartment), 'The target should be a MultiCompartment for the sparse integrator.'

    # compute the linear and derivative term
    linear, derivative = brainstate.transform.vector_grad(
        target.compute_membrane_derivative, argnums=0, return_value=True, unit_aware=False,
    )(V_n, *args)

    # Convert linearization to a unit-aware quantity
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))

    # Compute constant term
    const = derivative - V_n * linear
    return linear, const  # [n_neuron, n_segments]
