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
from braincell._typing import Args, Aux, DT, T, VectorField, Y0, Y1
from .protocol import DiffEqModule
from ._registry import register_integrator
from ._util import apply_standard_solver_step, environ_time

__all__ = [
    'implicit_euler_step',
]


def _newton_method(
    f: VectorField,
    y0: Y0,
    t: T,
    dt: DT,
    args: Args = (),
    tol: float = 1e-5,
    max_iter: int = 100,
) -> tuple[Y1, Aux]:
    r"""
    Solve one trapezoidal (Crank-Nicolson) step by Newton iteration.

    For an ODE :math:`dy/dt = f(t, y)` and a current state :math:`y_0` at time
    :math:`t`, this finds :math:`y` at :math:`t + \Delta t` by solving the
    implicit residual

    .. math::

        g(y) = y - y_0 - \tfrac{\Delta t}{2}
               \bigl(f(t, y_0) + f(t + \Delta t, y)\bigr) = 0,

    updating :math:`y \leftarrow y - J^{-1} g(y)` with
    :math:`J = \partial g / \partial y` assembled by forward-mode AD.

    Parameters
    ----------
    f : Callable
        The vector field ``f(t, y, *args)``, returning ``dy/dt`` and an
        auxiliary output.
    y0 : jax.Array
        The current state, used both as the initial guess and as the anchor of
        the residual.
    t : u.Quantity[u.second]
        The current time.
    dt : u.Quantity[u.second]
        The integration time step.
    args : tuple, optional
        Extra positional arguments forwarded to ``f``.
    tol : float, optional
        Convergence tolerance. Iteration stops once either the residual norm
        or the Jacobian norm falls below this value. Default 1e-5.
    max_iter : int, optional
        Maximum number of Newton iterations. Default 100.

    Returns
    -------
    y1 : jax.Array
        The updated state.
    aux : dict
        Empty; this solver produces no auxiliary output.
    """

    # The residual works on bare magnitudes; strip before anything closes over
    # ``t`` / ``dt`` so every evaluation below sees the same values.
    dt = u.get_magnitude(dt)
    t = u.get_magnitude(t)

    # f(t, y0) does not depend on the loop carry, and XLA does not hoist
    # loop-invariant code out of a while body, so evaluating it inside ``g``
    # cost a second full vector-field evaluation on every Newton iteration.
    f0 = f(t, y0, *args)[0]

    def g(t, y, *args):
        return y - y0 - 0.5 * dt * (f0 + f(t + dt, y, *args)[0])

    def cond_fun(carry):
        i, _, cond = carry
        return u.math.logical_and(i < max_iter, cond)

    def body_fun(carry):
        i, y1, _ = carry
        # df: [n_neuron * n_compartment, M];  A: [n_neuron * n_compartment, M, M]
        A, df = brainstate.transform.jacfwd(lambda y: g(t, y, *args), return_value=True, has_aux=False)(y1)
        condition = u.math.logical_or(u.math.linalg.norm(A) < tol, u.math.linalg.norm(df) < tol)
        new_y1 = y1 - u.math.linalg.solve(A, df)
        return (i + 1, new_y1, condition)

    _, result, _ = brainstate.transform.while_loop(cond_fun, body_fun, (0, y0, True))
    return result, {}


@register_integrator(
    "implicit_euler",
    category="implicit",
    order=1,
    description="Implicit Euler via Newton iteration.",
)
@set_module_as('braincell.quad')
def implicit_euler_step(target: DiffEqModule, *args, t: T = None, dt: DT = None):
    r"""Advance one step with an implicit Newton-solved step.

    Solves the trapezoidal (Crank-Nicolson) residual

    .. math::

        g(y) = y - y_n - \frac{\Delta t}{2}
               \bigl(f(t_n, y_n) + f(t_{n+1}, y)\bigr) = 0

    by Newton iteration. Each iteration assembles the full Jacobian
    :math:`J = \partial g / \partial y` and updates
    :math:`y \leftarrow y - J^{-1} g(y)` until either the residual norm or
    the Jacobian norm falls below ``1e-5`` or 100 iterations have been
    spent.

    The trapezoidal rule is A-stable with local truncation error
    :math:`O(\Delta t^3)` and global error :math:`O(\Delta t^2)`. Unlike an
    L-stable scheme it does not damp high-frequency components, so very stiff
    problems may ring rather than decay.

    Parameters
    ----------
    target : DiffEqModule
        The module whose :class:`DiffEqState` leaves are advanced.
    *args
        Extra positional arguments forwarded to ``target``'s
        ``compute_derivative`` and ``pre/post_integral`` hooks.
    t : Quantity[time], optional
        Current simulation time. Defaults to the value in the active
        :mod:`brainstate.environ` context.
    dt : Quantity[time], optional
        Time step; must carry units of time (e.g. ``0.025 * u.ms``). Defaults
        to the value in the active :mod:`brainstate.environ` context.

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
    ``t`` and ``dt`` are keyword-only and optional, so this step matches the
    ``(target, *args)`` convention the model hosts call
    (``self.solver(self)`` / ``self.solver(self, I_ext)``) and is selectable
    through ``solver="implicit_euler"``.

    The registered name and ``order=1`` metadata describe implicit Euler, but
    the residual solved here is the trapezoidal one, as the equations above
    and ``_implicit_test.py`` both show. Reconciling the two means either
    renaming a public solver or changing the scheme, so it is left to a
    deliberate decision rather than settled by a refactor.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import implicit_euler_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
        ...     implicit_euler_step(my_neuron)  # doctest: +SKIP
    """
    t, dt = environ_time(target, t, dt)
    apply_standard_solver_step(_newton_method, target, t, dt, *args)
