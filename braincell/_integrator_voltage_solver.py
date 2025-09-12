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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import brainevent
from brainevent._misc import generate_block_dim
from brainevent._compatible_import import pallas as pl

from ._integrator_protocol import DiffEqModule
from ._misc import set_module_as


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
        (u.math.zeros((P, n_nodes)) * u.get_unit(val)).at[:, internal_node_inds].set(val)
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
    
    # # --- Step 6: Solve the linear system for all populations in batch ---
    # solves = jax.vmap(
    #     solve_one,
    #     in_axes=(0, 0, None, None, None, None, None),  # batch over population
    #     out_axes=0
    # )(diags, solves, lowers, uppers, flipped_comp_edges, n_nodes, parent_lookup)
    
    # # # # --- Step 7: Write back results for internal nodes only ---
    # target.V.value = solves[:, internal_node_inds].reshape(target.V.value.shape)

    # --- Alternative DHS solver using custom pallas and warp kernels ---
    diags, solves = u.get_magnitude(diags), u.get_magnitude(solves)
    diags, solves = comp_based_triang_call(diags, solves, lowers, uppers, edges)

    solves = jax.vmap(
        solve_one_kernel,
        in_axes=(0, 0, None, None, None, None, None),  # batch over population
        out_axes=0
    )(diags, solves, lowers, uppers, flipped_comp_edges, n_nodes, parent_lookup)
    # solves = comp_based_backsub_call(solves/diags, lowers/diags, parent_lookup, n_nodes = n_nodes+1, n_steps=len(flipped_comp_edges))
    #steps = len(flipped_comp_edges)
    #solves = _comp_based_backsub_recursive_doubling(diags, solves, lowers, steps, n_nodes, parent_lookup)
    target.V.value = solves[:, internal_node_inds].reshape(target.V.value.shape)* u.mV


def pallas_kernel_generator(
    diags_info: jax.ShapeDtypeStruct,
    **kwargs
):
    block_size = generate_block_dim(diags_info.shape[0], maximum=512)
    #block_size = 32
    n_neuron_loop = pl.cdiv(diags_info.shape[0], block_size)

    def kernel(
        diags_ref,  # [n_neuron, n_nodes]
        solves_ref,  # [n_neuron, n_nodes]
        lowers_ref,  # [ n_nodes]
        uppers_ref,  # [ n_nodes]
        edges_ref,  # [ n_nodes-1, 2]
        # outs
        out_diags_ref,  # [n_neuron, n_nodes]
        out_solves_ref,  # [n_neuron, n_nodes]
    ):
        i_neuron_block = pl.program_id(0)
        i_neuron = i_neuron_block * block_size
        mask = jnp.arange(block_size) + i_neuron < diags_info.shape[0]

        def edge_loop_fn(i_edge, _):
            child = edges_ref[i_edge, 0]
            parent = edges_ref[i_edge, 1]
            lower_val = lowers_ref[child]
            upper_val = uppers_ref[child]

            child_diag = pl.load(diags_ref, (pl.dslice(i_neuron, block_size), child), mask=mask)
            child_solve = pl.load(solves_ref, (pl.dslice(i_neuron, block_size), child), mask=mask)

            # Factor that the child row has to be multiplied by.
            multiplier = upper_val / child_diag

            pl.atomic_add(out_diags_ref,
                          (pl.dslice(i_neuron, block_size), parent),
                          -lower_val * multiplier,
                          mask=mask)
            pl.atomic_add(out_solves_ref,
                          (pl.dslice(i_neuron, block_size), parent),
                          -child_solve * multiplier,
                          mask=mask)

        jax.lax.fori_loop(0, edges_ref.shape[0], edge_loop_fn, None)

    return brainevent.pallas_kernel(
        kernel,
        tile=(n_neuron_loop,),
        input_output_aliases={0: 0, 1: 1},
        outs=kwargs['outs']
    )


comp_based_triang = brainevent.XLACustomKernel('comp_based_triang')
comp_based_triang.def_gpu_kernel(pallas=pallas_kernel_generator)


def comp_based_triang_call(
    diags,  # [n_neuron, n_nodes]
    solves,  # [n_neuron, n_nodes]
    lowers,  # [ n_nodes]
    uppers,  # [ n_nodes]
    edges,  # [ n_nodes-1, 2]
):
    return comp_based_triang(
        diags,
        solves,
        lowers,
        uppers,
        edges,
        diags_info=jax.ShapeDtypeStruct(diags.shape, diags.dtype),
        outs=(jax.ShapeDtypeStruct(diags.shape, diags.dtype),
              jax.ShapeDtypeStruct(solves.shape, solves.dtype))
    )

def warp_kernel_generator(
    solves_info: jax.ShapeDtypeStruct,
    lowers_info: jax.ShapeDtypeStruct,
    parent_lookup_info: jax.ShapeDtypeStruct,
    n_nodes: int,
    n_steps: int,
    **kwargs
):
    import warp
    WARP_TILE_SIZE = warp.constant(solves_info.shape[1])
    def kernel(
        solves_ref: brainevent.jaxinfo_to_warpinfo(solves_info),
        lowers_ref: brainevent.jaxinfo_to_warpinfo(lowers_info),
        parent_lookup_ref: brainevent.jaxinfo_to_warpinfo(parent_lookup_info),
        out_solve_ref: brainevent.jaxinfo_to_warpinfo(solves_info),
    ):
        i_neuron = warp.tid()

        solves = warp.tile_load(solves_ref[i_neuron], WARP_TILE_SIZE)
        lowers = warp.tile_load(lowers_ref[i_neuron],  WARP_TILE_SIZE)
        parent_lookup = warp.tile_load(parent_lookup_ref, WARP_TILE_SIZE)

        # diags = warp.tile_load(diags_ref, shape=(1, n_nodes), offset=(i_neuron, 0))
        # solves = warp.tile_load(solves_ref, shape=(1, n_nodes,), offset=(i_neuron, 0))
        # lowers = warp.tile_load(lowers_ref, shape=(n_nodes,), offset=(0,))
        # parent_lookup = warp.tile_load(parent_lookup_ref, shape=(n_nodes,), offset=(0,))

        lowers[0] = 0.0
        step = 1
        while step <= n_steps:
            # For each node, get its k-step parent, where k=`step`.
            k_step_parent = warp.tile_arange(n_nodes)
            for _ in range(step):
                k_step_parent = parent_lookup[k_step_parent]

            # Update.
            solves = lowers * solves[k_step_parent] + solves
            lowers *= lowers[k_step_parent]
            step *= 2

        warp.tile_store(out_solve_ref[i_neuron], solves)

    return brainevent.warp_kernel(kernel, dim=(solves_info.shape[0], ),)

comp_based_backsub = brainevent.XLACustomKernel('comp_based_backsub')
comp_based_backsub.def_gpu_kernel(warp=warp_kernel_generator)


def comp_based_backsub_call(
    solves,
    lowers,
    parent_lookup,
    *,
    n_nodes: int,
    n_steps: int,
):
    return comp_based_backsub(
        solves, lowers, parent_lookup,
        n_nodes=n_nodes,
        n_steps=n_steps,
        solves_info=jax.ShapeDtypeStruct(solves.shape, solves.dtype),
        lowers_info=jax.ShapeDtypeStruct(lowers.shape, lowers.dtype),
        parent_lookup_info=jax.ShapeDtypeStruct(parent_lookup.shape, parent_lookup.dtype),
        outs=(jax.ShapeDtypeStruct(solves.shape, solves.dtype), )
    )[0]


# def solve_one_kernel(diags, solves, lowers, uppers, n_nodes, parent_lookup, edges, n_steps):
#     #diags, solves = run_all_levels_single_kernel(edges, diags, solves, uppers, lowers)
#     solves = _comp_based_backsub_recursive_doubling(
#         diags, solves, lowers, n_steps, n_nodes, parent_lookup
#     )
#     return solves
#
# def kernel_all_levels_serial(
#     edges_ref, diags_ref, solves_ref, uppers_ref, lowers_ref, num_edges_ref, out_diags_ref,
#     out_solves_ref,
# ):
#     E = pl.load(num_edges_ref, ())
#
#     def forward_body(e, _):
#         c = pl.load(edges_ref, (e, 0))
#         p = pl.load(edges_ref, (e, 1))
#         valid = p >= 0
#
#         upper = pl.load(uppers_ref, c, mask=valid, other=0.0)
#         lower = pl.load(lowers_ref, c, mask=valid, other=0.0)
#         diag_c = pl.load(diags_ref, c, mask=valid, other=1.0)
#         solve_c = pl.load(solves_ref, c, mask=valid, other=0.0)
#
#         mul = upper / diag_c
#         diag_p = pl.load(out_diags_ref, p, mask=valid, other=0.0)
#         solve_p = pl.load(out_solves_ref, p, mask=valid, other=0.0)
#
#         pl.store(out_diags_ref, p, diag_p - lower * mul, mask=valid)
#         pl.store(out_solves_ref, p, solve_p - solve_c * mul, mask=valid)
#         return None
#
#     jax.lax.fori_loop(0, E, forward_body, None)
#
# def run_all_levels_single_kernel(edges, diags, solves, uppers, lowers):
#     num_levels = jnp.asarray(diags.shape[0], dtype=jnp.int32)
#
#     call_all = pl.pallas_call(
#         kernel_all_levels_serial,
#         out_shape=(
#             jax.ShapeDtypeStruct(diags.shape, diags.dtype),
#             jax.ShapeDtypeStruct(solves.shape, solves.dtype),
#         ),
#         grid=(1,),
#         input_output_aliases={1: 0, 2: 1},
#     )
#     return call_all(edges, diags, solves, uppers, lowers, num_levels)


def solve_one_kernel(diags, solves, lowers, uppers, flipped_comp_edges, n_nodes, parent_lookup):
    # diags: [n_neuron, n_nodes]
    # solves: [n_neuron, n_nodes]
    # lowers: [n_nodes]
    # uppers: [n_nodes]
    # flipped_comp_edges: [num_levels, num_comps_per_level, 2]
    # n_nodes: int
    # parent_lookup: [n_nodes]
    steps = len(flipped_comp_edges)
    solves = _comp_based_backsub_recursive_doubling(diags, solves, lowers, steps, n_nodes, parent_lookup)
    return solves

def solve_one(diags, solves, lowers, uppers, flipped_comp_edges, n_nodes, parent_lookup):
    # diags: [n_neuron, n_nodes]
    # solves: [n_neuron, n_nodes]
    # lowers: [n_nodes]
    # uppers: [n_nodes]
    # flipped_comp_edges: [num_levels, num_comps_per_level, 2]
    # n_nodes: int
    # parent_lookup: [n_nodes]
    steps = len(flipped_comp_edges)
    for i in range(steps):
        diags, solves = _comp_based_triang(diags, solves, lowers, uppers, flipped_comp_edges[i])
    solves = _comp_based_backsub_recursive_doubling(diags, solves, lowers, steps, n_nodes, parent_lookup)
    return solves


def _comp_based_triang(diags, solves, lowers, uppers, comp_edge):
    """
    Triangulate the quasi-tridiagonal system compartment by compartment.
    """

    # `flipped_comp_edges` has shape `(num_levels, num_comps_per_level, 2)`. We first
    # get the relevant level with `[index]` and then we get all children and parents
    # in the level.
    child = comp_edge[:, 0]  # vector
    lower_val = lowers[child]
    upper_val = uppers[child]
    child_diag = diags[child]
    child_solve = solves[child]

    # Factor that the child row has to be multiplied by.
    multiplier = upper_val / child_diag

    # Updates to diagonal and solve
    parent = comp_edge[:, 1]
    diags = diags.at[parent].add(-lower_val * multiplier)
    solves = solves.at[parent].add(-child_solve * multiplier)
    return diags, solves


def _comp_based_backsub_recursive_doubling(
    diags,
    solves,
    lowers,
    steps: int,
    n_nodes: int,
    parent_lookup: np.ndarray,
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
