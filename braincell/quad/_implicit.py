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

"""Trapezoidal (Crank-Nicolson) integration for :mod:`braincell.quad`.

Exposes a single step function, :func:`crank_nicolson_step`, which advances
any :class:`~braincell.quad.protocol.DiffEqModule` by one trapezoidal step
solved with Newton iteration on each neuron's block of states.
"""

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell._misc import set_module_as
from braincell._typing import Args, Aux, DT, T, VectorField, Y0, Y1
from braincell.quad.protocol import DiffEqModule
from braincell.quad._registry import register_module_step
from braincell.quad._util import apply_standard_solver_step, environ_time, jacrev_last_dim

__all__ = [
    'crank_nicolson_step',
]

#: Newton iterations stop after this many steps even if not converged.
_MAX_ITER = 20


def _newton_trapezoid(
    f: VectorField,
    y0: Y0,
    t: T,
    dt: DT,
    args: Args = (),
    max_iter: int = _MAX_ITER,
) -> tuple[Y1, Aux]:
    r"""
    Solve one trapezoidal (Crank-Nicolson) step by Newton iteration.

    For an ODE :math:`dy/dt = f(t, y)` and a current state :math:`y_0` at time
    :math:`t`, this finds :math:`y` at :math:`t + \Delta t` by solving

    .. math::

        g(y) = y - y_0 - \tfrac{\Delta t}{2}
               \bigl(f(t, y_0) + f(t + \Delta t, y)\bigr) = 0,

    updating :math:`y \leftarrow y - J^{-1} g(y)`. ``y0`` has shape
    ``(..., M)``: :math:`J` is one ``M x M`` block per leading index
    (reverse-mode AD over the last axis), never a dense matrix over the
    whole population.

    Iteration stops when :math:`\max |\Delta y| / (1 + |y|) \le
    \varepsilon^{3/4}` (:math:`\varepsilon` is the machine epsilon of the
    state dtype) or after ``max_iter`` iterations.

    Parameters
    ----------
    f : Callable
        The vector field ``f(t, y, *args)``, returning ``dy/dt`` and an
        auxiliary output.
    y0 : jax.Array
        The current state, shape ``(..., n_state)``; also the initial guess.
    t : u.Quantity[u.second]
        The current time.
    dt : u.Quantity[u.second]
        The integration time step.
    args : tuple, optional
        Extra positional arguments forwarded to ``f``.
    max_iter : int, optional
        Maximum number of Newton iterations. Default 20.

    Returns
    -------
    y1 : jax.Array
        The updated state.
    aux : Any
        The auxiliary output of ``f`` at the last Newton iterate.
    """
    dtype = y0.dtype
    dt = jnp.asarray(u.get_magnitude(dt), dtype=dtype)
    t = u.get_magnitude(t)
    tol = jnp.finfo(dtype).eps ** 0.75

    # f(t, y0) does not depend on the Newton iterate; evaluate it once.
    f0, aux0 = f(t, y0, *args)

    def g(y):
        fy, aux = f(t + dt, y, *args)
        return y - y0 - 0.5 * dt * (f0 + fy), aux

    def cond_fun(carry):
        i, _, err, _ = carry
        return jnp.logical_and(i < max_iter, err > tol)

    def body_fun(carry):
        i, y, _, _ = carry
        # Evaluating ``f`` writes the module's states; restore them so no
        # tracer of this loop body leaks into the states.
        with brainstate.StateTraceStack() as trace:
            J, r, aux = jacrev_last_dim(g, y, has_aux=True)
        for st, val in zip(trace.states, trace.original_state_values):
            st.restore_value(val)
        dy = jnp.linalg.solve(J, r[..., None])[..., 0]
        y_new = y - dy
        err = jnp.max(jnp.abs(dy) / (1.0 + jnp.abs(y_new)))
        return i + 1, y_new, err, aux

    init = (0, y0, jnp.asarray(jnp.inf, dtype=dtype), aux0)
    _, y1, _, aux = brainstate.transform.while_loop(cond_fun, body_fun, init)
    return y1, aux


@register_module_step(
    "crank_nicolson",
    category="implicit",
    order=2,
    description="Trapezoidal (Crank-Nicolson) rule, Newton iteration per neuron.",
)
@set_module_as('braincell.quad')
def crank_nicolson_step(target: DiffEqModule, *args, t: T = None, dt: DT = None):
    r"""Advance one step with the trapezoidal (Crank-Nicolson) rule.

    Solves

    .. math::

        g(y) = y - y_n - \frac{\Delta t}{2}
               \bigl(f(t_n, y_n) + f(t_{n+1}, y)\bigr) = 0

    by Newton iteration. The states of ``target`` are merged along the last
    axis as it declares in ``diffeq_state_merging``: ``'stack'`` (the
    default, e.g. the engine of a lumped :class:`braincell.Cell`) gives one
    ``n_state x n_state`` Jacobian per neuron, ``'concat'`` (a cable
    :class:`braincell.Cell`) one per cell over all of its compartments,
    axial coupling included. Each Newton iteration solves these systems in
    a batch. Iteration stops
    when :math:`\max |\Delta y| / (1 + |y|) \le \varepsilon^{3/4}`
    (:math:`\varepsilon` is the machine epsilon of the state dtype) or after
    20 iterations.

    The trapezoidal rule is A-stable with global error
    :math:`O(\Delta t^2)`. Unlike an L-stable scheme it does not damp
    high-frequency components, so very stiff problems may ring rather than
    decay; :func:`backward_euler_step` damps them.

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

    Raises
    ------
    TypeError
        If ``target`` is not a :class:`DiffEqModule`.

    See Also
    --------
    backward_euler_step : Single-Jacobian linearized backward Euler
        (first order, L-stable).
    staggered_step : Staggered voltage / channel splitting, the solver
        multi-compartment :class:`braincell.Cell` uses by default.

    Notes
    -----
    This step was registered as ``implicit_euler`` (``implicit_euler_step``)
    in braincell <= 0.2, where the Newton loop stopped after one iteration.
    The name now raises; results on nonlinear models differ from the old
    step because the iteration converges.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import crank_nicolson_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
        ...     crank_nicolson_step(my_neuron)  # doctest: +SKIP
    """
    if not isinstance(target, DiffEqModule):
        raise TypeError(f"The target should be a {DiffEqModule.__name__}. But got {type(target)} instead.")
    t, dt = environ_time(target, t, dt)
    apply_standard_solver_step(_newton_trapezoid, target, t, dt, *args, merging=target.diffeq_state_merging)
