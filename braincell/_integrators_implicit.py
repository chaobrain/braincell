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

from __future__ import annotations

import brainstate
import brainunit as u
import jax.numpy as jnp

from scipy.integrate import solve_ivp


from ._integrators_util import apply_standard_solver_step
from ._misc import set_module_as
from ._protocol import DiffEqModule

__all__ = [
    'implicit_euler_step',
]


def newton_method(f, y0, t, dt, tol=1e-5, max_iter=100, order = 1, args=()):
    r"""
    Newton's method for solving the implicit equations arising from the Crank - Nicolson method for ordinary differential equations (ODEs).

    The Crank - Nicolson method is a finite - difference method used for numerically solving ODEs of the form \(\frac{dy}{dt}=f(t,y)\).
    Given the current state \(y_0\) at time \(t\), this function uses Newton's method to find the next state \(y\) at time \(t + dt\)
    by solving the implicit equation \(y - y_0-\frac{dt}{2}(f(t,y_0)+f(t + dt,y)) = 0\).

    Parameters:
        f : callable
            Function representing the ODE or implicit equation.
        y0 : array_like
            Initial guess for the solution.
        t : float
            Current time.
        dt : float
            Time step.
        tol : float, optional
            Convergence tolerance for the solution. Default is 1e-5.
        max_iter : int, optional
            Maximum number of iterations. Default is 100.
        order : int, optional
            Order of the integration method. If order = 1, use explicit Euler. If order = 2, use Crank - Nicolson.
        args : tuple, optional
            Additional arguments passed to the function f.

    Returns:
        y : ndarray
            Solution array, shape (n,).
    """
    def g(t, y, *args):
        if order == 1:
            return y - y0 - dt * f(t + dt, y, *args)
        elif order == 2:
            return y - y0 - 0.5*dt * (f(t, y0, *args) + f(t + dt, y, *args))
        else:
            raise ValueError("Only order 1 or 2 is supported.")

    def cond_fun(carry):
        i, _, A, df = carry
        condition = jnp.logical_or(jnp.linalg.norm(A) < tol, jnp.linalg.norm(df) < tol)
        return jnp.logical_and(i < max_iter, jnp.logical_not(condition))

    def body_fun(carry):
        i, y1, _, _ = carry
        A, df = brainstate.augment.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=False)(y1)
        new_y1 = y1 - jnp.linalg.solve(A, df)
        return (i + 1, new_y1, A, df)

    def body_fun_modified(carry):
        i, y1, A, _ = carry
        df = g(t, y1, *args)
        new_y1 = y1 - jnp.linalg.solve(A, df)
        return (i + 1, new_y1, A, df)
    
    dt = u.get_magnitude(dt)
    init_guess = y0 + dt*f(t, y0, *args)
    A, df= brainstate.augment.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=False)(init_guess)
    init_carry = (0, init_guess, A, df)

    if modified ==True:
        _, result, _, _ = jax.lax.while_loop(cond_fun, body_fun_modified, init_carry)
    else:
        _, result, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        
    return result

def solve_ivp_method(f, y0, t, dt, method, args=()):
    sol = solve_ivp(lambda t, y: f(t, y, *args), [t, t+dt], y0, t_eval=[t + dt], method=method)
    return sol.y.flatten()

def _implicit_euler_for_axial_current(A, y0, dt, inv_A=None):
    r"""
    Implicit Euler Integrator for linear ODEs of the form:

    $$
    u_{n+1} = u_{n} + h_n \cdot A \cdot u_{n+1}
    $$

    Rearranging this equation:
    $$
    (I - h_n \cdot A) \cdot u_{n+1} = u_n
    $$

    Parameters:
        A : ndarray
            The coefficient matrix (linear matrix), shape (n, n).
        y0 : array_like
            Initial condition, shape (n,).
        dt : float
            Time step.
        inv_A : ndarray, optional
            The inverse of the matrix (I - dt * A), shape (n, n). If provided, it will be used for solving.

    Returns:
        y1 : ndarray
            Solution array at the next time step, shape (n,).
    """
    dt = u.get_magnitude(dt)
    n = y0.shape[-1]
    I = jnp.eye(n)
    M = I - dt * A

    def solve_with_inv():
        return jnp.dot(inv_A, y0)

    def solve_with_solve():
        return jnp.linalg.solve(M, y0)

    is_tracer = isinstance(A, jax.core.Tracer)
    has_inv_A = inv_A is not None
    condition = jnp.logical_and(jnp.logical_not(is_tracer), has_inv_A)

    y1 = jax.lax.cond(condition, solve_with_inv, solve_with_solve)

    return y1


def _crank_nicolson_for_axial_current(A, y0, dt):
    r"""
    Crank-Nicolson Integrator for linear ODEs of the form:

    $$
    \frac{dy}{dt} = A y
    $$

    The Crank-Nicolson method is a combination of the implicit and explicit methods:
    $$
    y_{n+1} = y_n + \frac{dt}{2} \cdot A \cdot y_{n+1} + \frac{dt}{2} \cdot A \cdot y_n
    $$

    Rearranged as:
    $$
    (I - \frac{dt}{2} \cdot A) \cdot y_{n+1} = (I + \frac{dt}{2} \cdot A) \cdot y_n
    $$

    Parameters:
        A : ndarray
            The coefficient matrix (linear matrix), shape (n, n).
        y0 : array_like
            Initial condition, shape (n,).
        t : float
            Current time (not used directly in this linear case but kept for consistency with ODE format).
        dt : float
            Time step.
        args : tuple, optional
            Additional arguments for the function (not used in this linear case).

    Returns:
        y1 : ndarray
            Solution array at the next time step, shape (n,).
    """
    dt = u.get_magnitude(dt)
    n = y0.shape[-1]
    I = jnp.eye(n)

    lhs = I - 0.5 * dt * A
    rhs = (I + 0.5 * dt * A) @ y0

    y1 = jnp.linalg.solve(lhs, rhs)
    return y1


@set_module_as('braincell')
def implicit_euler_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args
):
    """
    Applies the implicit Euler method to solve a differential equation.

    This function uses the Newton method to solve the implicit equation
    arising from the implicit Euler discretization of the ODE.

    Parameters:
    -----------
    target : DiffEqModule
        The differential equation module to be solved.
    t : u.Quantity[u.second]
        The current time in the simulation.
    *args : 
        Additional arguments to be passed to the differential equation.
    """
    apply_standard_solver_step(
        newton_method, target, t, *args
    )

def construct_A(target):
    """
    Construct the matrix A for the axial current of a multi-compartment neuron, which satisfies the differential equation dV/dt = AV.

    Parameters:
    target (object): An object containing relevant information about the multi-compartment neuron. It should have the following attributes:
        - n_compartment (int): The number of compartments in the neuron.
        - connection (array): An array of connection information. Each row represents a connection, containing two elements which are the indices of the pre-synaptic and post-synaptic compartments respectively.
        - cm (float): The membrane capacitance.
        - A (array): An array of the area of each compartment.
        - resistances (array): An array of the axial resistances.

    Returns:
    A_matrix (array): The constructed matrix A for the axial current.
    """
    n_compartment = target.n_compartment
    connection = u.math.array(target.connection) 
    cm = target.cm
    A = target.A
    R_axial = target.resistances

    pre_ids, post_ids = connection[:, 0], connection[:, 1]
    

    adj_matrix = u.math.zeros((n_compartment, n_compartment)).at[pre_ids, post_ids].set(1)
    R_matrix = u.math.zeros((n_compartment, n_compartment)).at[pre_ids, post_ids].set(1/R_axial) 
    adj_matrix = adj_matrix + adj_matrix.T
    R_matrix = R_matrix + R_matrix.T

    A_matrix =  R_matrix /(cm *A[:,u.math.newaxis] )
    A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(A_matrix, axis=1))

    '''
    R_matrix = coo_matrix((R_axial, (pre_ids, post_ids)), shape = (n_compartment, n_compartment))
    A_matrix = coo_matrix(A_matrix)
    '''

    return A_matrix

def splitting_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args
):
    """
    Applies the splitting solver method to solve a differential equation.

    This function uses the Newton method to solve the implicit equation
    arising from the implicit Euler discretization of the ODE.

    Parameters
    ----------
    target : DiffEqModule
        The differential equation module to be solved.
    t : u.Quantity[u.second]
        The current time in the simulation.
    *args :
        Additional arguments to be passed to the differential equation.
    """
    from braincell.neuron.multi_compartment import MultiCompartment

    if isinstance(target, MultiCompartment):
        pass
        # first step, extracting the axial current matrix and solve AV_{n+1} = b(V_n) with Crank–Nicolson method

        V_n = target.V.value
        A_matrix = construct_A(target)
        target.V.value = _crank_nicolson_for_axial_current(A_matrix, V_n, dt)
        # second step
        with brainstate.environ.context(compute_axial_current=False):
            integral = lambda: apply_standard_solver_step(
                newton_method,
                target,
                t,
                *args
            )
            for _ in range(len(target.pop_size+1)):
                integral = brainstate.augment.vmap(integral, in_states=target.states())
            integral()

    else:
        apply_standard_solver_step(
            newton_method, target, t, *args
        )
