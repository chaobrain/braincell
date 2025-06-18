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
def dhs_voltage_step():
    """
    Implicit euler solver with the `dendritic hierarchical scheduling` (DHS, Zhang et al., 2023).
    """
    pass


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
    """
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
