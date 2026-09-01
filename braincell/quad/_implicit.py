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

"""Implicit (backward) integration for :mod:`braincell.quad`.

Exposes a single step function, :func:`implicit_euler_step`, which advances
any :class:`~braincell.quad.protocol.DiffEqModule` by one backward-Euler /
Crank-Nicolson step solved with Newton iteration.
"""

import brainstate
import brainunit as u

from braincell._misc import set_module_as
from braincell._typing import T, DT
from .protocol import DiffEqModule
from ._registry import register_integrator
from ._util import apply_standard_solver_step

__all__ = [
    'implicit_euler_step',
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
    n, result, _ = brainstate.transform.while_loop(cond_fun, body_fun, init_carry)
    aux = {}
    return result, aux


@register_integrator(
    "implicit_euler",
    category="implicit",
    order=1,
    description="Implicit Euler via Newton iteration.",
)
@set_module_as('braincell.quad')
def implicit_euler_step(target: DiffEqModule, t: T, dt: DT, *args):
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
    staggered_step : Staggered voltage / channel splitting, the solver
        multi-compartment :class:`braincell.Cell` uses by default.

    Notes
    -----
    This step takes ``t`` and ``dt`` explicitly, so it is **not**
    selectable through ``solver="implicit_euler"``: the model hosts call
    ``self.solver(self)`` / ``self.solver(self, I_ext)`` and read the time
    arguments from :mod:`brainstate.environ`. Call it directly, or pick a
    ``(target, *args)`` integrator such as ``"backward_euler"``. The
    registry records this as ``IntegratorEntry.requires_time_args``.
    """
    apply_standard_solver_step(_newton_method, target, t, dt, *args)
