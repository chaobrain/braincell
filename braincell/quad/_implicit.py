# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from braincell._misc import set_module_as
from braincell._typing import T, DT
from ._exp_euler import _exponential_euler
from .protocol import DiffEqModule
from ._registry import register_integrator
from ._runge_kutta import rk4_step
from ._util import apply_standard_solver_step, jacrev_last_dim

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
        A, df = brainstate.transform.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=False)(y1)
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


@register_integrator(
    "implicit_euler",
    category="implicit",
    order=1,
    description="Implicit Euler via Newton iteration.",
)
@set_module_as('braincell.quad')
def implicit_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    r"""Advance one step with the implicit (backward) Euler method.

    Solves

    .. math::

        y_{n+1} = y_n + \Delta t \, f(t_{n+1}, y_{n+1})

    by Newton iteration on the residual
    :math:`g(y) = y - y_n - \Delta t \, f(t + \Delta t, y)`. Each
    iteration assembles the full Jacobian
    :math:`J = \partial g / \partial y` and updates
    :math:`y \leftarrow y - J^{-1} g(y)` until either the residual norm or
    the Jacobian norm falls below ``1e-5`` or 100 iterations have been
    spent.

    Implicit Euler is :math:`L`-stable, so it tolerates arbitrarily large
    time steps on stiff problems at the cost of damping high-frequency
    components. Local truncation error is :math:`O(\Delta t^2)`; global
    error is :math:`O(\Delta t)`.

    Parameters
    ----------
    target : DiffEqModule
        The module whose :class:`DiffEqState` leaves are advanced.
    t : Quantity[time]
        Current simulation time.
    dt : Quantity[time]
        Time step. Must carry units of time (e.g. ``0.025 * u.ms``).
    *args
        Extra positional arguments forwarded to ``target``'s
        ``compute_derivative`` and ``pre/post_integral`` hooks.

    Returns
    -------
    None
        ``target``'s differential states are updated in place.

    See Also
    --------
    backward_euler_step : Single-Jacobian linearized backward Euler
        (one Newton iteration).
    splitting_step : Backward Euler for the cable equation paired with a
        Newton solve for the gating variables.
    cn_exp_euler_step : Crank-Nicolson cable solve combined with
        exponential Euler gating.
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

        # create the A_matrix
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


@register_integrator(
    "splitting",
    category="implicit",
    description="Operator-splitting solver pairing implicit axial currents with Newton-based gating updates.",
)
@set_module_as('braincell.quad')
def splitting_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    r"""Advance a multi-compartment cell with operator-splitting.

    Two complementary updates are applied within a single ``dt``:

    1. **Channels and concentrations.** With axial currents temporarily
       disabled (``compute_axial_current=False``), every
       :class:`DiffEqState` in *target* is advanced by the manually
       parallelised Newton iteration in
       :func:`_newton_method_manual_parallel`. This handles ion channel
       gating variables, calcium pools, and any other non-voltage state.
    2. **Cable voltage.** A dense LU factorisation of the implicit-Euler
       cable matrix :math:`I - \Delta t A` is constructed once via
       :func:`construct_lu` and reused with :func:`jax.scipy.linalg.lu_solve`
       to solve for the new midpoint voltages. The factorisation lives
       inside ``ensure_compile_time_eval`` so it is hoisted out of the
       JIT cache.

    For non-:class:`Cell` targets the splitting collapses to a single
    Newton solve over the whole state vector.

    Splitting recovers full-cable stability while keeping the channel
    update embarrassingly parallel across compartments. It is the
    historical NEURON-style integrator and is provided here mainly for
    comparison against :func:`staggered_step`, which uses a sparser DHS
    cable solve.

    Parameters
    ----------
    target : DiffEqModule
        The module to advance. When *target* is a
        :class:`braincell.Cell`, the splitting branch is used;
        otherwise the routine falls back to a plain Newton step.
    t : Quantity[time]
        Current simulation time.
    dt : Quantity[time]
        Time step. Must carry units of time.
    *args
        Extra positional arguments forwarded to ``target``'s
        ``compute_derivative`` and ``pre/post_integral`` hooks.

    Returns
    -------
    None
        ``target``'s state — voltage and channel/ion variables — is
        updated in place.

    See Also
    --------
    staggered_step : Sparser DHS-based splitting suitable for large trees.
    cn_rk4_step, implicit_rk4_step, implicit_exp_euler_step : Other
        cable/gating splitting recipes.
    """
    from braincell._multi_compartment import Cell

    if isinstance(target, Cell):

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
            integral = brainstate.transform.vmap2(solve_axial, in_states=target.states())
        integral()
        '''

        ## time
        # s1t1 = time.time()

        with brainstate.environ.context(compute_axial_current=False):
            apply_standard_solver_step(_newton_method_manual_parallel, target, t, dt, *args, merging='stack')
        for _ in range(len(target.pop_size)):
            integral = brainstate.transform.vmap2(solve_axial, in_states=target.states())
        integral()
        # jax.debug.print('step2 cost {a}',a = time.time() - s2t1)

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)


@register_integrator(
    "cn_rk4",
    category="implicit",
    description="Crank-Nicolson axial currents combined with explicit RK4 gating updates.",
)
@set_module_as('braincell.quad')
def cn_rk4_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    r"""Advance a cell with Crank-Nicolson voltage and explicit RK4 channels.

    Performs a two-stage operator-splitting update inside one ``dt``:

    1. **Channels and concentrations.** With axial currents temporarily
       disabled, every non-voltage :class:`DiffEqState` is advanced by
       :func:`rk4_step` (classical four-stage fourth-order Runge-Kutta).
    2. **Cable voltage.** A Crank-Nicolson step is then applied to the
       linear axial system :math:`dV/dt = A V` via
       :func:`_crank_nicolson_for_axial_current`, which solves
       :math:`(I - \tfrac{\Delta t}{2} A) V_{n+1} = (I + \tfrac{\Delta t}{2} A) V_n`.

    The voltage solve is second-order accurate and unconditionally stable;
    the channel update is fourth-order accurate. The combined scheme is
    therefore second-order overall and well suited to dendrites where the
    cable equation dominates the stiffness budget.

    Parameters
    ----------
    target : Cell
        Multi-compartment cell to advance.
    t : Quantity[time]
        Current simulation time.
    dt : Quantity[time]
        Time step.
    *args
        Extra positional arguments forwarded to the channel and voltage
        solvers.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    AssertionError
        If *target* is not a :class:`braincell.Cell`.

    See Also
    --------
    cn_exp_euler_step : Same Crank-Nicolson voltage solve paired with
        exponential Euler channel updates.
    implicit_rk4_step : Implicit Euler voltage solve paired with RK4
        channels.
    """

    def solve_axial():
        V_n = target.V.value
        A_matrix = construct_A(target)
        target.V.value = _crank_nicolson_for_axial_current(A_matrix, V_n, dt)

    with brainstate.environ.context(compute_axial_current=False):
        rk4_step(target, t, dt, *args, )
    for _ in range(len(target.pop_size)):
        integral = brainstate.transform.vmap2(solve_axial, in_states=target.states())
    integral()


@register_integrator(
    "cn_exp_euler",
    category="implicit",
    description="Crank-Nicolson axial currents combined with exponential Euler gating updates.",
)
@set_module_as('braincell.quad')
def cn_exp_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    r"""Advance a cell with Crank-Nicolson voltage and exponential Euler channels.

    Operator-splitting update inside one ``dt``:

    1. **Channels and concentrations.** With axial currents disabled, all
       non-voltage :class:`DiffEqState` leaves are advanced by the coupled
       exponential Euler update from :func:`_exponential_euler` (the same
       linearised matrix-exponential step used by :func:`exp_euler_step`).
    2. **Cable voltage.** The linear axial system is then advanced by a
       Crank-Nicolson half-implicit step,
       :math:`(I - \tfrac{\Delta t}{2} A) V_{n+1} = (I + \tfrac{\Delta t}{2} A) V_n`.

    The Crank-Nicolson voltage solve is second-order accurate and
    unconditionally stable. Pairing it with exponential Euler for the
    channels is the recommended choice when the channel kinetics are very
    stiff (e.g. fast sodium activation) — exponential Euler captures the
    local exponential decay exactly while Crank-Nicolson keeps the cable
    update centred in time.

    Parameters
    ----------
    target : Cell
        Multi-compartment cell to advance.
    t : Quantity[time]
        Current simulation time.
    dt : Quantity[time]
        Time step.
    *args
        Extra positional arguments forwarded to the channel and voltage
        solvers.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    Raises
    ------
    AssertionError
        If *target* is not a :class:`braincell.Cell`.

    See Also
    --------
    cn_rk4_step : Same Crank-Nicolson voltage solve paired with classical
        RK4 channel updates.
    implicit_exp_euler_step : Implicit Euler voltage solve paired with
        exponential Euler channel updates.
    """

    with brainstate.environ.context(compute_axial_current=False):
        apply_standard_solver_step(_exponential_euler, target, t, dt, *args, merging='stack')

    def solve_axial():
        V_n = target.V.value
        A_matrix = construct_A(target)
        target.V.value = _crank_nicolson_for_axial_current(A_matrix, V_n, dt)

    for _ in range(len(target.pop_size)):
        integral = brainstate.transform.vmap2(solve_axial, in_states=target.states())
    integral()


@register_integrator(
    "implicit_rk4",
    category="implicit",
    order=4,
    description="Implicit axial currents combined with explicit RK4 gating updates.",
)
@set_module_as('braincell.quad')
def implicit_rk4_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    r"""Advance a cell with implicit Euler voltage and explicit RK4 channels.

    Operator-splitting update inside one ``dt`` for a multi-compartment
    cell:

    1. **Channels and concentrations.** With axial currents temporarily
       disabled, every non-voltage :class:`DiffEqState` is advanced by
       :func:`rk4_step`.
    2. **Cable voltage.** The linear axial system :math:`dV/dt = A V` is
       then advanced by one implicit-Euler solve via
       :func:`_implicit_euler_for_axial_current`,
       :math:`(I - \Delta t A) V_{n+1} = V_n`.

    For non-:class:`Cell` targets the routine falls back to a single
    Newton-based implicit Euler step on the full state vector.

    The implicit Euler voltage solve is :math:`L`-stable and damps
    high-frequency cable modes; the explicit RK4 channel update gives the
    rest of the system fourth-order accuracy. Use this scheme when the
    cable equation is the only severely stiff component and you want
    extra accuracy on smooth gating dynamics.

    Parameters
    ----------
    target : DiffEqModule
        The module to advance. Splitting is used when *target* is a
        :class:`braincell.Cell`; otherwise the routine reduces to a
        plain Newton step.
    t : Quantity[time]
        Current simulation time.
    dt : Quantity[time]
        Time step.
    *args
        Extra positional arguments forwarded to the channel and voltage
        solvers.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    See Also
    --------
    cn_rk4_step : Crank-Nicolson voltage solve paired with RK4 channels.
    implicit_exp_euler_step : Implicit Euler voltage solve paired with
        exponential Euler channels.
    """
    from braincell._multi_compartment import Cell

    if isinstance(target, Cell):
        with brainstate.environ.context(compute_axial_current=False):
            rk4_step(target, t, dt, *args, )

        def solve_axial():
            V_n = target.V.value
            A_matrix = construct_A(target)
            target.V.value = _implicit_euler_for_axial_current(A_matrix, V_n, dt)

        for _ in range(len(target.pop_size)):
            integral = brainstate.transform.vmap2(solve_axial, in_states=target.states())
        integral()

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)


@register_integrator(
    "implicit_exp_euler",
    category="implicit",
    description="Implicit axial currents combined with exponential Euler gating updates.",
)
@set_module_as('braincell.quad')
def implicit_exp_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    r"""Advance a cell with implicit Euler voltage and exponential Euler channels.

    Operator-splitting update inside one ``dt`` for a multi-compartment
    cell:

    1. **Channels and concentrations.** With axial currents temporarily
       disabled, every non-voltage :class:`DiffEqState` is advanced by
       the coupled exponential Euler update from
       :func:`_exponential_euler`.
    2. **Cable voltage.** The linear axial system :math:`dV/dt = A V` is
       then advanced by one implicit-Euler solve via
       :func:`_implicit_euler_for_axial_current`,
       :math:`(I - \Delta t A) V_{n+1} = V_n`.

    For non-:class:`Cell` targets the routine falls back to a single
    Newton-based implicit Euler step on the full state vector.

    The combination of an :math:`L`-stable cable solve and an
    :math:`A`-stable channel solve makes this scheme robust at large time
    steps and is the recommended choice for multi-compartment Hodgkin-Huxley
    models when accuracy on smooth dynamics is less critical than stability.

    Parameters
    ----------
    target : DiffEqModule
        The module to advance.
    t : Quantity[time]
        Current simulation time.
    dt : Quantity[time]
        Time step.
    *args
        Extra positional arguments forwarded to the channel and voltage
        solvers.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    See Also
    --------
    implicit_rk4_step : Implicit Euler voltage paired with RK4 channels.
    cn_exp_euler_step : Crank-Nicolson voltage paired with exponential
        Euler channels.
    """
    from braincell._multi_compartment import Cell

    if isinstance(target, Cell):

        with brainstate.environ.context(compute_axial_current=False):
            apply_standard_solver_step(_exponential_euler, target, t, dt, *args, merging='stack')

        def solve_axial():
            V_n = target.V.value
            A_matrix = construct_A(target)
            target.V.value = _implicit_euler_for_axial_current(A_matrix, V_n, dt)

        for _ in range(len(target.pop_size)):
            integral = brainstate.transform.vmap2(solve_axial, in_states=target.states())
        integral()

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)


@register_integrator(
    "exp_exp_euler",
    category="exponential",
    description="Exponential axial integration paired with exponential Euler gating updates.",
)
@set_module_as('braincell.quad')
def exp_exp_euler_step(
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args
):
    r"""Advance a cell with exponential cable and exponential Euler channels.

    Operator-splitting update inside one ``dt`` for a multi-compartment
    cell:

    1. **Channels and concentrations.** With axial currents temporarily
       disabled, every non-voltage :class:`DiffEqState` is advanced by
       the coupled exponential Euler update from
       :func:`_exponential_euler`.
    2. **Cable voltage.** The linear axial system :math:`dV/dt = A V` is
       then advanced by one step of the implicit Euler solver
       :func:`_implicit_euler_for_axial_current`. Despite the function
       name, the *cable* update used here is the same implicit Euler
       solve as :func:`implicit_exp_euler_step`; the ``exp_exp`` label
       refers to using exponential Euler on **both** the channel update
       and (conceptually) on the linear cable equation, where one step
       of implicit Euler approximates :math:`e^{\Delta t A} V_n`.

    For non-:class:`Cell` targets the routine falls back to a single
    Newton solve.

    Parameters
    ----------
    target : DiffEqModule
        The module to advance.
    t : Quantity[time]
        Current simulation time.
    dt : Quantity[time]
        Time step.
    *args
        Extra positional arguments forwarded to the channel and voltage
        solvers.

    Returns
    -------
    None
        ``target``'s state is updated in place.

    See Also
    --------
    implicit_exp_euler_step : Same composition under a more descriptive
        name.
    cn_exp_euler_step : Crank-Nicolson voltage paired with exponential
        Euler channels.
    """
    from braincell._multi_compartment import Cell

    if isinstance(target, Cell):

        with brainstate.environ.context(compute_axial_current=False):
            apply_standard_solver_step(_exponential_euler, target, t, dt, *args, merging='stack')

        def solve_axial():
            V_n = target.V.value
            A_matrix = construct_A(target)
            # jax.debug.print("A = {a}",a=A_matrix)
            target.V.value = _implicit_euler_for_axial_current(A_matrix, V_n, dt)  # expm(dt*A_matrix)@V_n

        for _ in range(len(target.pop_size)):
            integral = brainstate.transform.vmap2(solve_axial, in_states=target.states())
        integral()

    else:
        apply_standard_solver_step(_newton_method, target, t, dt, *args)
