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

from ._misc import set_module_as
from ._protocol import DiffEqModule

__all__ = [
]


def newton_method(f, y0, t, dt, tol=1e-5, max_iter=100, args=()):
    r"""
    Newton's method for solving implicit equations of the form f(y) = 0.

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
        args : tuple, optional
            Additional arguments passed to the function f.

    Returns:
        y : ndarray
            Solution array, shape (n,).
    """
    dt = u.get_magnitude(dt)
    y1 = y0

    def g(t, y, *args):
        return y - y0 - dt * f(t + dt, y, *args)

    for i in range(max_iter):
        A, df, aux = brainstate.augment.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=True)(y1)

        if jnp.linalg.norm(A) < tol or jnp.linalg.norm(df) < tol:
            return y1

            # Update the guess using Newton's method: y1 = y1 - (Jacobian)^-1 * g(y1)
        y1 = y1 - jnp.linalg.solve(A, df)

    return y1


import jax.numpy as jnp


def implicit_euler(A, y0, t, dt, args=()):
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
        t : float
            Current time (not used directly in the linear case but kept for consistency with ODE format).
        dt : float
            Time step.
        args : tuple, optional
            Additional arguments for the function f (not used in this linear case).

    Returns:
        y1 : ndarray
            Solution array at the next time step, shape (n,).
    """

    dt = u.get_magnitude(dt)
    n = y0.shape[-1]
    I = jnp.eye(n)
    y1 = jnp.linalg.solve(I - dt * A, y0)
    return y1


def crank_nicolson(A, y0, t, dt, args=()):
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
def newton_step(target: DiffEqModule, t: u.Quantity[u.second], *args):
    """
    The newton method for the differential equations.
    """
    # pre integral
    dt = brainstate.environ.get_dt()
    target.pre_integral(*args)
    dimensionless_fn, diffeq_states, other_states = _transform_diffeq_module_into_dimensionless_fn(target)

    # one-step integration
    diffeq_vals, other_vals = newton_method(
        dimensionless_fn,
        _dict_state_to_arr(diffeq_states),
        t,
        dt,
        args
    )

    # post integral
    _assign_arr_to_states(diffeq_vals, diffeq_states)
    for key, val in other_vals.items():
        other_states[key].value = val
