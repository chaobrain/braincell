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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import scipy.sparse as sp
from jax.experimental import sparse
from jax.scipy.linalg import lu_factor, lu_solve

from ._integrator_exp_euler import _exponential_euler
from ._integrator_runge_kutta import rk4_step
from ._integrator_util import apply_standard_solver_step, jacrev_last_dim
from ._misc import set_module_as
from ._protocol import DiffEqModule
from ._typing import T, DT

__all__ = [
    'implicit_euler_step',
    'splitting_step',
    'implicit_rk4_step',
    'implicit_exp_euler_step',
    'cn_rk4_step',
    'cn_exp_euler_step',
    'exp_exp_euler_step',
]


def _newton_method(f, y0, t, dt, args=(), modified=False, tol=1e-5, max_iter=100, order=2):
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
        # jax.debug.print("arg = {a}", a = args)
        if order == 1:
            return y - y0 - dt * f(t + dt, y, *args)[0]
        elif order == 2:
            return y - y0 - 0.5 * dt * (f(t, y0, *args)[0] + f(t + dt, y, *args)[0])
        else:
            raise ValueError("Only order 1 or 2 is supported.")

    def cond_fun(carry):
        i, _, cond = carry
        # condition = u.math.logical_or(u.math.linalg.norm(A) < tol, u.math.linalg.norm(df) < tol)
        return u.math.logical_and(i < max_iter, cond)

    def body_fun(carry):
        i, y1, _ = carry
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
    n, result, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    aux = {}
    return result, aux


def _newton_method_manual_parallel(
    f,
    y0,
    t,
    dt,
    args=(),
    modified=False,
    tol=1e-5,
    max_iter=100,
    order=2,

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
            Order of the integration method. If order = 1, use implicit Euler. If order = 2, use Crank - Nicolson.
        args : tuple, optional
            Additional arguments passed to the function f.

    Returns:
        y : ndarray
            Solution array, shape (n,).
    """

    def g(t, y, *args):
        # jax.debug.print("arg ={a}", a = args)
        if order == 1:
            return y - y0 - dt * f(t + dt, y, *args)[0]
        elif order == 2:
            return y - y0 - 0.5 * dt * (f(t, y0, *args)[0] + f(t + dt, y, *args)[0])
        else:
            raise ValueError("Only order 1 or 2 is supported.")

    def cond_fun(carry):
        i, _, cond = carry
        return u.math.logical_and(i < max_iter, cond)

    def body_fun(carry):
        i, y1, _ = carry
        # df: [*pop_size, n_compartment, M]
        # A: [*pop_size, n_compartment, M, M]
        A, df = jacrev_last_dim(lambda y: g(t, y, *args), y1)

        shape = df.shape
        # df: [n_neuron * n_compartment, M]
        # A: [n_neuron * n_compartment, M, M]
        A = A.reshape((A.shape[0] * A.shape[1],) + A.shape[2:])
        df = df.reshape((df.shape[0] * df.shape[1],) + df.shape[2:])

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
    init_guess = y0 + dt * f(t, y0, *args)[0]
    init_carry = (0, init_guess, True)
    n, result, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        init_carry
    )
    aux = {}
    return result, aux


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

    Parameters
    ----------
    A : ndarray
        The coefficient matrix (linear matrix), shape (n, n).
    y0 : array_like
        Initial condition, shape (n,).
    dt : float
        Time step.
    inv_A : ndarray, optional
        The inverse of the matrix (I - dt * A), shape (n, n). If provided, it will be used for solving.

    Returns
    -------
    y1 : ndarray
        Solution array at the next time step, shape (n,).
    """
    with jax.ensure_compile_time_eval():
        n = y0.shape[-1]
        I = u.math.eye(n)
        lhs = I - dt * A
        rhs = y0
        y1 = u.math.linalg.solve(lhs, rhs)

    # # residual
    # residual = rhs - lhs @ y1
    # residual_norm = jnp.linalg.norm(u.get_magnitude(residual))
    # jax.debug.print('Residual norm = {a}', a = residual_norm)
    # jax.debug.print('Relative error = {a}', a = relative_error)
    # cond = jnp.linalg.cond(u.get_magnitude(lhs))
    # jax.debug.print('cond = {a}', a = cond)

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

    Parameters
    ----------
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

    Returns
    -------
    y1 : ndarray
        Solution array at the next time step, shape (n,).
    """
    with jax.ensure_compile_time_eval():
        n = y0.shape[-1]
        I = u.math.eye(n)
        alpha = 1
        lhs = (I - alpha * dt * A)
        rhs = (I + (1 - alpha) * dt * A) @ y0
        y1 = u.math.linalg.solve(lhs, rhs)

    # # residual
    # residual = rhs - lhs @ y1
    # residual_norm = jnp.linalg.norm(u.get_magnitude(residual))
    # jax.debug.print('Residual norm = {a}', a = residual_norm)
    # jax.debug.print('Relative error = {a}', a = relative_error)
    # cond = jnp.linalg.cond(u.get_magnitude(lhs))
    # jax.debug.print('cond = {a}', a = cond)
    # jax.debug.print('I = {a}',a = I)
    # jax.debug.print('lhs = {a}',a = lhs)
    # cond = jnp.linalg.cond(u.get_magnitude(lhs))
    # jax.debug.print('cond = {a}', a = cond)
    return y1


@set_module_as('braincell')
def implicit_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
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
    dt : u.Quantity[u.second]
        The numerical time step for the integration.
    *args : 
        Additional arguments to be passed to the differential equation.
    """
    apply_standard_solver_step(
        _newton_method, target, t, dt, *args
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
        ## load the param
        n_compartment = target.n_compartment
        cm = target.cm
        A = target.area
        G_matrix = target.conductance_matrix()
        Gl = target.Gl

        # jax.debug.print('Area = {a}', a = A)
        # jax.debug.print('cm = {a}', a = cm)
        ## create the A_matrix
        cm_A = cm * A

        A_matrix = G_matrix / (cm_A[:, u.math.newaxis])
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(A_matrix, axis=1))
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].add(-Gl / cm)

    return A_matrix


def construct_lhs(target):
    with jax.ensure_compile_time_eval():
        dt = brainstate.environ.get_dt()
        n_compartment = target.n_compartment
        cm = target.cm
        A = target.area
        G_matrix = target.conductance_matrix()
        Gl = target.Gl

        # jax.debug.print('Area = {a}', a = A)
        # jax.debug.print('cm = {a}', a = cm)
        ## create the A_matrix
        cm_A = cm * A

        A_matrix = G_matrix / (cm_A[:, u.math.newaxis])
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(A_matrix, axis=1))
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].add(-Gl / cm)

        I = u.math.eye(n_compartment)
        lhs = I - dt * A_matrix
        return lhs


def construct_lhs_sparse(target):
    with jax.ensure_compile_time_eval():
        dt = brainstate.environ.get_dt()
        n_compartment = target.n_compartment
        cm = target.cm
        A = target.area
        G_matrix = target.conductance_matrix()
        Gl = target.Gl

        # jax.debug.print('Area = {a}', a = A)
        # jax.debug.print('cm = {a}', a = cm)
        ## create the A_matrix
        cm_A = cm * A

        A_matrix = G_matrix / (cm_A[:, u.math.newaxis])
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(A_matrix, axis=1))
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].add(-Gl / cm)

        I = u.math.eye(n_compartment)
        lhs = I - dt * A_matrix
        lhs_dense_np = jnp.array(lhs)

        lhs_sparse_scipy = sp.csr_matrix(lhs_dense_np)

        data = jnp.array(lhs_sparse_scipy.data)
        indices = jnp.array(lhs_sparse_scipy.indices)
        indptr = jnp.array(lhs_sparse_scipy.indptr)

        return data, indices, indptr


def construct_lu(target):
    with jax.ensure_compile_time_eval():
        dt = brainstate.environ.get_dt()
        n_compartment = target.n_compartment
        cm = target.cm
        A = target.area
        G_matrix = target.conductance_matrix()
        Gl = target.Gl

        # jax.debug.print('Area = {a}', a = A)
        # jax.debug.print('cm = {a}', a = cm)
        ## create the A_matrix
        cm_A = cm * A

        A_matrix = G_matrix / (cm_A[:, u.math.newaxis])
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(A_matrix, axis=1))
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].add(-Gl / cm)
        I = u.math.eye(n_compartment)
        lhs = I - dt * A_matrix
        lu, piv = lu_factor(lhs)

        return lu, piv


def construct_lu_sparse(target):
    with jax.ensure_compile_time_eval():
        dt = brainstate.environ.get_dt()
        n_compartment = target.n_compartment
        cm = target.cm
        A = target.area
        G_matrix = target.conductance_matrix()
        Gl = target.Gl

        cm_A = cm * A

        A_matrix = G_matrix / (cm_A[:, u.math.newaxis])
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].set(-u.math.sum(A_matrix, axis=1))
        A_matrix = A_matrix.at[jnp.diag_indices(n_compartment)].add(-Gl / cm)
        I = u.math.eye(n_compartment)
        lhs = I - dt * A_matrix
        lhs_bcoo = sparse.BCOO.fromdense(lhs)

        return


@set_module_as('braincell')
def splitting_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
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
    dt : u.Quantity[u.second]
        The numerical time step for the integration.
    *args :
        Additional arguments to be passed to the differential equation.
    """
    from braincell._multi_compartment_test import MultiCompartment_test

    if isinstance(target, MultiCompartment_test):

        def solve_axial():
            # dt = brainstate.environ.get_dt()
            V_n = u.get_magnitude(target.V.value)
            # V_n = target.V.value

            # A_matrix = construct_A(target)
            # target.V.value = _implicit_euler_for_axial_current(A_matrix, V_n, dt)

            # lhs = construct_lhs(target)
            # target.V.value = u.math.linalg.solve(lhs, V_n)

            # data, indices, indptr = construct_lhs_sparse(target)
            # target.V.value = sparse.linalg.spsolve(data, indices, indptr, V_n.reshape(-1), tol=1e-6, reorder=1).reshape(1,-1) * u.mV

            lu, piv = construct_lu(target)
            target.V.value = lu_solve((lu, piv), V_n) * u.mV

            # lu, piv = construct_lu_sparse(target)
            # target.V.value =sparse.lu_solve(lu, piv, V_n )* u.mV

        '''
        for _ in range(len(target.pop_size)):
            integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
        integral()
        '''

        ## time
        # s1t1 = time.time()

        with brainstate.environ.context(compute_axial_current=False):
            apply_standard_solver_step(_newton_method_manual_parallel, target, t, dt, *args, merging='stack')
        for _ in range(len(target.pop_size)):
            integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
        integral()
        # jax.debug.print('step2 cost {a}',a = time.time() - s2t1)

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)


@set_module_as('braincell')
def cn_rk4_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    from braincell._multi_compartment import MultiCompartment
    assert isinstance(target, MultiCompartment)

    def solve_axial():
        V_n = target.V.value
        A_matrix = construct_A(target)
        target.V.value = _crank_nicolson_for_axial_current(A_matrix, V_n, dt)

    with brainstate.environ.context(compute_axial_current=False):
        rk4_step(target, t, dt, *args, )
    for _ in range(len(target.pop_size)):
        integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
    integral()


@set_module_as('braincell')
def cn_exp_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    from braincell._multi_compartment import MultiCompartment
    assert isinstance(target, MultiCompartment)

    with brainstate.environ.context(compute_axial_current=False):
        apply_standard_solver_step(_exponential_euler, target, t, dt, *args, merging='stack')

    def solve_axial():
        V_n = target.V.value
        A_matrix = construct_A(target)
        target.V.value = _crank_nicolson_for_axial_current(A_matrix, V_n, dt)

    for _ in range(len(target.pop_size)):
        integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
    integral()


@set_module_as('braincell')
def implicit_rk4_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
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
    dt : u.Quantity[u.second]
        The numerical time step for the integration.
    *args :
        Additional arguments to be passed to the differential equation.
    """
    from braincell._multi_compartment import MultiCompartment

    if isinstance(target, MultiCompartment):
        with brainstate.environ.context(compute_axial_current=False):
            rk4_step(target, t, dt, *args, )

        def solve_axial():
            V_n = target.V.value
            A_matrix = construct_A(target)
            target.V.value = _implicit_euler_for_axial_current(A_matrix, V_n, dt)

        for _ in range(len(target.pop_size)):
            integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
        integral()

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)


@set_module_as('braincell')
def implicit_exp_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
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
    dt : u.Quantity[u.second]
        The numerical time step for the integration.
    *args :
        Additional arguments to be passed to the differential equation.
    """
    from braincell._multi_compartment import MultiCompartment

    if isinstance(target, MultiCompartment):

        with brainstate.environ.context(compute_axial_current=False):
            apply_standard_solver_step(_exponential_euler, target, t, dt, *args, merging='stack')

        def solve_axial():
            V_n = target.V.value
            A_matrix = construct_A(target)
            target.V.value = _implicit_euler_for_axial_current(A_matrix, V_n, dt)

        for _ in range(len(target.pop_size)):
            integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
        integral()

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)


@set_module_as('braincell')
def exp_exp_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
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
    dt : u.Quantity[u.second]
        The numerical time step for the integration.
    *args :
        Additional arguments to be passed to the differential equation.
    """
    from braincell._multi_compartment import MultiCompartment

    if isinstance(target, MultiCompartment):

        with brainstate.environ.context(compute_axial_current=False):
            apply_standard_solver_step(_exponential_euler, target, t, dt, *args, merging='stack')

        def solve_axial():
            V_n = target.V.value
            A_matrix = construct_A(target)
            # jax.debug.print("A = {a}",a=A_matrix)
            target.V.value = _implicit_euler_for_axial_current(A_matrix, V_n, dt)  # expm(dt*A_matrix)@V_n

        for _ in range(len(target.pop_size)):
            integral = brainstate.augment.vmap(solve_axial, in_states=target.states())
        integral()

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)
