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

from typing import Callable, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp

from ._integrators_util import apply_standard_solver_step
from ._misc import set_module_as
from ._protocol import DiffEqModule

__all__ = [
    'implicit_euler_step',
    'splitting_step',
]


def _newton_method(f, y0, t, dt, modified=False, tol=1e-5, max_iter=100, order=2, args=()):
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
            return y - y0 - dt * f(t + dt, y, *args)[0]
        elif order == 2:
            return y - y0 - 0.5 * dt * (f(t, y0, *args)[0] + f(t + dt, y, *args)[0])
        else:
            raise ValueError("Only order 1 or 2 is supported.")

    def cond_fun(carry):
        i, _, cond = carry
        # condition = u.math.logical_or(u.math.linalg.norm(A) < tol, u.math.linalg.norm(df) < tol)
        return u.math.logical_and(i < max_iter, u.math.logical_not(cond))

    def body_fun(carry):
        i, y1, _, _ = carry
        A, df = brainstate.augment.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=False)(y1)
        # df: [n_neuron, n_compartment, M]
        # A: [n_neuron, n_compartment, M, M]
        # df: [n_neuron * n_compartment, M]
        # A: [n_neuron * n_compartment, M, M]

        # y1: [n_neuron * n_compartment, M]

        condition = u.math.logical_or(u.math.linalg.norm(A) < tol, u.math.linalg.norm(df) < tol)
        new_y1 = y1 - u.math.linalg.solve(A, df)
        return (i + 1, new_y1, condition)

    def body_fun_modified(carry):
        i, y1, A, _ = carry
        df = g(t, y1, *args)
        new_y1 = y1 - u.math.linalg.solve(A, df)
        return (i + 1, new_y1, A, df)

    dt = u.get_magnitude(dt)
    t = u.get_magnitude(t)
    init_guess = y0  # + dt*f(t, y0, *args)[0]
    init_carry = (0, init_guess, True)
    '''
    if not modified:
        n, result, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    else:
        n, result, _, df = jax.lax.while_loop(cond_fun, body_fun_modified, init_carry)
    '''
    n, result, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    aux = {}
    # print(result)
    # print(modified)
    return result, aux


def _newton_method_manual_parallel(
    f,
    y0,
    t,
    dt,
    modified=False,
    tol=1e-5,
    max_iter=100,
    order=2,
    args=()
):
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
        modified: bool
            If True, use the modified Newton's method.
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
            return y - y0 - dt * f(t + dt, y, *args)[0]
        elif order == 2:
            return y - y0 - 0.5 * dt * (f(t, y0, *args)[0] + f(t + dt, y, *args)[0])
        else:
            raise ValueError("Only order 1 or 2 is supported.")

    def cond_fun(carry):
        i, _, cond = carry
        return u.math.logical_and(i < max_iter, u.math.logical_not(cond))

    def body_fun(carry):
        i, y1, _, _ = carry
        # df: [*pop_size, n_compartment, M]
        # A: [*pop_size, n_compartment, M, M]
        A, df = _jacrev_last_dim(lambda y: g(t, y, *args), y1)
        shape = df.shape
        # df: [n_neuron * n_compartment, M]
        # A: [n_neuron * n_compartment, M, M]
        A = A.reshape(-1, shape[-2], shape[-1])
        df = df.reshape(-1, shape[-1])

        # y1: [n_neuron * n_compartment, M]
        condition = u.math.alltrue(
            jax.vmap(lambda A_, df_: u.math.logical_or(
                u.math.linalg.norm(A_) < tol,
                u.math.linalg.norm(df_) < tol
            ))(A, df)
        )
        solve = jax.vmap(lambda A_, df_: u.math.linalg.solve(A_, df_))(A, df)
        solve = solve.reshape(*shape)
        new_y1 = y1 - solve
        return (i + 1, new_y1, condition)

    def body_fun_modified(carry):
        i, y1, A = carry
        df = g(t, y1, *args)
        new_y1 = y1 - u.math.linalg.solve(A, df)
        return (i + 1, new_y1, A)

    dt = u.get_magnitude(dt)
    t = u.get_magnitude(t)
    init_guess = y0  # + dt*f(t, y0, *args)[0]
    init_carry = (0, init_guess, True)
    n, result, _ = jax.lax.while_loop(
        cond_fun,
        body_fun_modified if modified else body_fun,
        init_carry
    )
    aux = {}
    return result, aux


def solve_ivp_method(f, y0, t, dt, args=()):
    dt = u.get_magnitude(dt)
    t = u.get_magnitude(t)
    sol = solve_ivp(lambda t, y: f(t, y, *args)[0], [t, t + dt], y0, t_eval=[t + dt], method='LSODA')
    aux = {}
    return sol.y.flatten(), aux


def _implicit_euler_for_axial_current(A, y0, dt):
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
    # dt = u.get_magnitude(dt)
    n = y0.shape[-1]
    I = u.math.eye(n)
    M = I - dt * A
    y1 = u.math.linalg.solve(M, y0)

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
    n = y0.shape[-1]
    I = u.math.eye(n) * u.get_unit(0.5 * dt * A)

    lhs = I - 0.5 * dt * A
    rhs = (I + 0.5 * dt * A) @ y0
    y1 = u.math.linalg.solve(lhs, rhs)
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
        _newton_method, target, t, *args
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
    with jax.ensure_compile_time_eval():
        n_compartment = target.n_compartment
        connection = u.math.array(target.connection)
        cm = target.cm
        A = target.A
        R_axial = target.resistances

        pre_ids, post_ids = connection[:, 0], connection[:, 1]

        adj_matrix = u.math.zeros((n_compartment, n_compartment)).at[pre_ids, post_ids].set(1)
        R_matrix = u.math.zeros((n_compartment, n_compartment)).at[pre_ids, post_ids].set(
            1 / u.get_magnitude(R_axial)) / u.get_unit(R_axial)
        adj_matrix = adj_matrix + adj_matrix.T
        R_matrix = R_matrix + R_matrix.T

        A_matrix = R_matrix / (cm * A[:, u.math.newaxis])
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(A_matrix, axis=1))

        '''
        R_matrix = coo_matrix((R_axial, (pre_ids, post_ids)), shape = (n_compartment, n_compartment))
        A_matrix = coo_matrix(A_matrix)
        '''

    return A_matrix


@set_module_as('braincell')
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
        # first step, extracting the axial current matrix and solve AV_{n+1} = b(V_n) with Crank–Nicolson method
        def solve_axial():
            dt = brainstate.environ.get_dt()
            V_n = target.V.value
            A_matrix = construct_A(target)
            target.V.value = _crank_nicolson_for_axial_current(A_matrix, V_n, dt)

        for _ in range(len(target.pop_size)):
            integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
        integral()

        # second step

        with brainstate.environ.context(compute_axial_current=False):
            apply_standard_solver_step(
                _newton_method_manual_parallel,
                target,
                t,
                *args,
                merging_method='stack'
            )

            # ralston4_step(
            #     target,
            #     t,
            #     *args,
            # )


    else:
        apply_standard_solver_step(
            _newton_method, target, t, *args
        )


def _jacrev_last_dim(
    fn: Callable[[...], jax.Array],
    hid_vals: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute the Jacobian of a function with respect to its last dimension.

    This function calculates the Jacobian matrix of the given function 'fn'
    with respect to the last dimension of the input 'hid_vals'. It uses
    JAX's vector-Jacobian product (vjp) and vmap for efficient computation.

    Args:
        fn (Callable[[...], jax.Array]): The function for which to compute
            the Jacobian. It should take a JAX array as input and return
            a JAX array.
        hid_vals (jax.Array): The input values for which to compute the
            Jacobian. The last dimension is considered as the dimension
            of interest.

    Returns:
        jax.Array: The Jacobian matrix. Its shape is (*varshape, num_state, num_state),
        where varshape is the shape of the input excluding the last dimension,
        and num_state is the size of the last dimension.

    Raises:
        AssertionError: If the number of input and output states are not the same.
    """
    new_hid_vals, f_vjp = jax.vjp(fn, hid_vals)
    num_state = new_hid_vals.shape[-1]
    varshape = new_hid_vals.shape[:-1]
    assert num_state == hid_vals.shape[-1], 'Error: the number of input/output states should be the same.'
    g_primals = u.math.broadcast_to(u.math.eye(num_state), (*varshape, num_state, num_state))
    jac = jax.vmap(f_vjp, in_axes=-2, out_axes=-2)(g_primals)
    return jac[0], new_hid_vals
