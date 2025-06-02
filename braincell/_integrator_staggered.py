# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import functools
from typing import Dict

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from ._base import HHTypedNeuron
from ._integrator_util import apply_standard_solver_step, jacrev_last_dim, _check_diffeq_state_derivative
from ._misc import set_module_as
from ._integrator_runge_kutta import euler_step, rk4_step
from ._integrator_exp_euler import ind_exp_euler_step
from ._integrator_exp_euler import _exponential_euler
from ._protocol import DiffEqState, DiffEqModule
from ._typing import Path, T, DT

__all__ = [
    'staggered_step',
]

@set_module_as('braincell')
def linear_and_const_term(target: DiffEqModule, t: T, dt: DT, *args):
    '''
    get the linear_and_constant_term of ion
    '''
    assert isinstance(target, DiffEqModule), (
        f"The target should be a {DiffEqModule.__name__}. "
        f"But got {type(target)} instead."
    )

    # Retrieve all states from the target module
    all_states = brainstate.graph.states(target)

    # Split states into differential equation states and other states
    diffeq_states, other_states = all_states.split(DiffEqState, ...)

    # Split states into v differential equation states and other differential equation states
    v_diffeq_states = {k: v for k, v in diffeq_states.items() if k == ('V',)}
    other_diffeq_states = {k: v for k, v in diffeq_states.items() if k != ('V',)}

    # Collect all state object ids for trace validation
    all_state_ids = {id(st) for st in all_states.values()}

    def vector_field(
        diffeq_state_key: Path,
        diffeq_state_val: brainstate.typing.ArrayLike,
        other_diffeq_state_vals: Dict,
        other_state_vals: Dict,
    ):
        """
        Compute the derivative for a single DiffEqState, given its value and the values of other states.

        Parameters
        ----------
        diffeq_state_key : Path
            The key identifying the current DiffEqState.
        diffeq_state_val : ArrayLike
            The value of the current DiffEqState.
        other_diffeq_state_vals : dict
            Values of other DiffEqStates.
        other_state_vals : dict
            Values of other (non-differential) states.

        Returns
        -------
        tuple
            (diffeq_state_derivative, other_state_vals_out)
        """
        # Ensure the state value is a supported floating point type
        dtype = u.math.get_dtype(diffeq_state_val)
        if dtype not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
            raise ValueError(
                f'The input data type should be float64, float32, float16, or bfloat16 '
                f'when using Exponential Euler method. But we got {dtype}.'
            )

        with brainstate.StateTraceStack() as trace:
            # Assign the current and other state values
            all_states[diffeq_state_key].value = diffeq_state_val
            for key, val in other_diffeq_state_vals.items():
                all_states[key].value = val
            for key, val in other_state_vals.items():
                all_states[key].value = val

            # Pre-integration hook (e.g., update gating variables)
            target.pre_integral(*args)

            # Compute derivatives for all states
            target.compute_derivative(*args)

            # Post-integration hook (e.g., apply constraints)
            target.post_integral(*args)

            # Validate and retrieve the derivative for the current state
            _check_diffeq_state_derivative(all_states[diffeq_state_key], dt)  # THIS is important.
            diffeq_state_derivative = all_states[diffeq_state_key].derivative
            # Collect updated values for other states
            other_state_vals_out = {key: other_states[key].value for key in other_state_vals.keys()}

        # Ensure all states in the trace are known
        for st in trace.states:
            if id(st) not in all_state_ids:
                raise ValueError(f'State {st} is not in the state list.')

        return diffeq_state_derivative, other_state_vals_out

    # Prepare dictionaries of current state values
    other_state_vals = {k: v.value for k, v in other_states.items()}
    v_diffeq_state_vals = {k: v.value for k, v in v_diffeq_states.items()}
    other_diffeq_state_vals = {k: v.value for k, v in other_diffeq_states.items()}
    assert len(diffeq_states) > 0, "No DiffEqState found in the target module."

    # Compute the linearization (Jacobian), derivative, and auxiliary outputs
    key = ('V',)
    linear, derivative, aux = brainstate.transform.vector_grad(
            functools.partial(vector_field, key), argnums=0, return_value=True, has_aux=True, unit_aware=False,
)(          v_diffeq_state_vals[key],  # V DiffEqState value
            other_diffeq_state_vals,  # Other DiffEqState values
            other_state_vals,)  # Other state values

    # Convert linearization to a unit-aware quantity
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    # Compute constant term
    const = derivative - v_diffeq_state_vals[key] * linear

    # Update other states with auxiliary outputs
    for k, st in other_states.items():
        st.value = aux[k]

    return linear, const

@set_module_as('braincell')
def Laplacian_matrix(target: DiffEqModule):
    '''
    Construct the Laplacian matrix L = diag(G'*1) - G' for the given target,
    where G' = G/(area*cm) is the normalized conductance matrix.
    
    Parameters:
    - target: A DiffEqModule instance containing compartmental model parameters
    
    Returns:
    - L_matrix: The Laplacian matrix representing the conductance term
                of the compartmental model's differential equations

    Notes:
    - Computes the Laplacian matrix which describes the electrical conductance
      between compartments in a compartmental model.
    - The diagonal elements are set to the sum of the respective row's
      off-diagonal elements to ensure conservation of current.
    - The normalization by (area*cm) accounts for compartment geometry and membrane properties.
    '''
    with jax.ensure_compile_time_eval():
        # Extract model parameters
        n_compartment, cm, area, G_matrix = target.n_compartment, target.cm, target.area, target.conductance_matrix
        
        # Compute negative normalized conductance matrix: element-wise division by (cm * area)
        L_matrix = -G_matrix / (cm * area)[:, u.math.newaxis]
        
        # Set diagonal elements to enforce Kirchhoff's current law
        # This constructs the Laplacian matrix L
        L_matrix = L_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(L_matrix, axis=1))
        
    return L_matrix

@set_module_as('braincell')
def solve_v(Laplacian_matrix, D_linear, D_const, dt, V_n):
    '''
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
    ''' 
    
    # Compute the left-hand side matrix
    # lhs = I + dt*(Laplacian_matrix + D_linear)
    n_compartments = Laplacian_matrix.shape[0]
    identity_matrix = jnp.eye(n_compartments)
    lhs = identity_matrix + dt * (Laplacian_matrix + D_linear)
    
    # Compute the right-hand side vector
    # rhs = V_n + dt*D_const
    rhs = V_n + dt * D_const
    
    return u.math.linalg.solve(lhs, rhs)


@set_module_as('braincell')
def staggered_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args
):
    dt = brainstate.environ.get_dt()
    def update_v():
        
        V_n = target.V.value
        L_matrix = Laplacian_matrix(target)
        linear, const = linear_and_const_term(target, t, dt, *args)
        
        D_linear, D_const = u.math.diag(linear), const
        ## -D_linear cause from left to right, the sign changed
        target.V.value = solve_v(L_matrix, -D_linear, D_const, dt, V_n)
    

    # update v
    for _ in range(len(target.pop_size)):
        update_v_batched = brainstate.augment.vmap(update_v, in_states=target.states())
    update_v_batched()
    
    # update nonv
    V_n = target.V.value

    # #exp_euler
    # euler_step(
    # target,
    # t,
    # *args,
    # )

    #ind_exp_euler
    ind_exp_euler_step(
    target,
    t,
    dt,
    *args,
    )

    # #rk4
    # rk4_step(
    # target,
    # t,
    # *args,
    # )

    # apply_standard_solver_step(
    # _exponential_euler,
    # target,
    # t,
    # *args,
    # merging_method='stack'
    # )

    target.V.value = V_n