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
    def g_1(t, y, *args):
        return y - y0 - dt * f(t + dt, y, *args)

    def g_2(t, y, *args):
        return y - y0 - 0.5 * dt * (f(t, y0, *args) + f(t + dt, y, *args))

    def g(t, y, *args):
        branches = [g_1, g_2]
        index = jnp.clip(order - 1, 0, 1)
        return jax.lax.switch(index, branches, t, y, *args)

    def cond_fun(carry):
        i, y1, A, df = carry
        condition = jnp.logical_or(jnp.linalg.norm(A) < tol, jnp.linalg.norm(df) < tol)
        return jnp.logical_and(i < max_iter, jnp.logical_not(condition))

    def body_fun(carry):
        i, y1, _, _ = carry
        A, df = brainstate.augment.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=False)(y1)
        new_y1 = y1 - jnp.linalg.solve(A, df)
        return (i + 1, new_y1, A, df)
    
    dt = u.get_magnitude(dt)
    A, df= brainstate.augment.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=False)(y0)
    init_carry = (0, y0, A, df)
    _, result, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return result

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

def construct_CN(target):
    r'''
    The function constructs the Crank-Nicolson (CN) method matrices and right-hand side vectors
    for a system of equations that models a network of nodes connected by resistances. 
    The CN method is commonly used to numerically solve differential equations, and in this case,
    it is applied to systems involving voltages and resistances.

    The equation solved by this function is derived from the Crank-Nicolson method applied to a network
    described by voltages \( V_j^{n+1} \) and resistances \( R_{k,j} \). The general form of the equation
    is:

    \[
    V_j^{n+1} \left( 1 + \frac{\Delta t}{2 c_m A_j} \sum_{k \rightarrow j} \frac{1}{R_{k,j}} \right)
    - \frac{\Delta t}{2 c_m A_j} \sum_{k \rightarrow j} \frac{V_k^{n+1}}{R_{k,j}} = 
    V_j^n \left( 1 - \frac{\Delta t}{2 c_m A_j} \sum_{k \rightarrow j} \frac{1}{R_{k,j}} \right)
    + \frac{\Delta t}{2 c_m A_j} \sum_{k \rightarrow j} \frac{V_k^n}{R_{k,j}}
    \]

    Where:
    - \( V_j^{n+1} \) is the voltage at node \( j \) at the next time step \( n+1 \).
    - \( V_j^n \) is the voltage at node \( j \) at the current time step \( n \).
    - \( R_{k,j} \) represents the resistance between node \( k \) and node \( j \).
    - \( A_j \) is the surface area associated with the node \( j \).
    - \( c_m \) is the capacitance for node \( j \), which is typically given as a matrix.
    - \( \Delta t \) is the time step size.

    The above equation can be interpreted as follows:
    - The left-hand side (LHS) consists of terms involving the voltage at the next time step \( V_j^{n+1} \), which is influenced by both the current voltage \( V_j^n \) and the neighboring nodes' voltages \( V_k^n \).
    - The right-hand side (RHS) contains terms for the current voltages \( V_j^n \), and the sum over the neighboring nodes' voltages \( V_k^n \) weighted by the corresponding resistances.

    The matrix `A_matrix` and the vector `b_vector` are constructed to solve this equation numerically for each time step, using the Crank-Nicolson method.

    '''
    connection = target.connetion
    cm = target.cm
    A = target.A
    R = target.resistances
    inv_R = jnp.where(connection == 1, 1 / R, 0)
    sum_inv_R = jnp.sum(inv_R, axis=1)

    # Construction of A
    A_matrix = jnp.eye(n) - (dt / (2 * cm * A[:, jnp.newaxis])) * inv_R
    A_matrix = A_matrix.at[jnp.diag_indices(n)].set(1 + (dt / (2 * cm * A)) * sum_inv_R)
    # Construction of b
    term1 = Vn * (1 - (dt / (2 * cm * A)) * sum_inv_R)
    term2 = (dt / (2 * cm * A[:, jnp.newaxis])) * (Vn * inv_R)
    b_vector = term1 + jnp.sum(term2, axis=1)

    return A_matrix, b_vector

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

        # second step
        with brainstate.environ.context(compute_axial_current=False):
            integral = lambda: apply_standard_solver_step(
                newton_method,
                target,
                t,
                *args
            )
            for _ in range(len(target.pop_size)):
                integral = brainstate.augment.vmap(integral, in_states=target.states())
            integral()

    else:
        apply_standard_solver_step(
            newton_method, target, t, *args
        )
