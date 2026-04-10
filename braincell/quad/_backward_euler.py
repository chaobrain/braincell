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

from braincell._misc import set_module_as
from ._protocol import DiffEqModule
from ._registry import register_integrator
from ._util import apply_standard_solver_step, jacrev_last_dim

__all__ = [
    'backward_euler_step',
]


def _backward_euler(f, y0, t, dt, args=()):
    """
    One step of implicit backward Euler method for ODE integration.
    Linearize the system at the current state using the Jacobian.

    Args:
        f: Callable function f(t, y, *args) returning dy/dt (and optional aux)
        y0: current state, shape (..., M)
        t: current time
        dt: time step
        args: additional arguments passed to f

    Returns:
        y1: updated state after one backward Euler step
        aux: optional auxiliary output from f
    """
    dt = u.get_magnitude(dt)

    # Compute Jacobian A = df/dy and function value df = f(y0)
    A, df, aux = jacrev_last_dim(lambda y: f(t, y, *args), y0, has_aux=True)

    # Flatten batch dimensions
    A = A.reshape((-1, A.shape[-2], A.shape[-1]))  # (B, M, M)
    df = df.reshape((-1, df.shape[-1]))  # (B, M)

    n = y0.shape[-1]
    I = jnp.eye(n)

    # Solve linear system: (I - dt * A) @ Δy = dt * df
    LHS = I - dt * A
    RHS = dt * df
    updates = jax.scipy.linalg.solve(LHS, RHS.reshape(-1, n, 1)).reshape(y0.shape)

    # Compute the new state
    y1 = y0 + updates
    return y1, aux


@register_integrator(
    "backward_euler",
    category="implicit",
    order=1,
    description="Backward (implicit) Euler method via local Jacobian linearization.",
)
@set_module_as('braincell')
def backward_euler_step(target: DiffEqModule, *args):
    r"""Advance one step with the linearised backward (implicit) Euler method.

    Backward Euler discretises an ODE :math:`dy/dt = f(t, y)` as

    .. math::

        y_{n+1} = y_n + \Delta t \, f(t_{n+1}, y_{n+1}),

    which is implicit in :math:`y_{n+1}`. Rather than running a Newton
    solver to convergence (see :func:`implicit_euler_step` for that
    variant), this routine takes a single Newton step from a local
    Jacobian:

    .. math::

        J = \frac{\partial f}{\partial y}\bigg|_{y_n}, \qquad
        (I - \Delta t \, J)\, \Delta y = \Delta t \, f(t_n, y_n), \qquad
        y_{n+1} = y_n + \Delta y.

    The result is the so-called *Rosenbrock* / *linearly implicit Euler*
    update — first-order accurate, :math:`L`-stable, and considerably
    cheaper than full Newton because the Jacobian is built once per step
    and the linear system is solved by a batched
    :func:`jax.scipy.linalg.solve`. It is the recommended choice for
    medium-stiff Hodgkin-Huxley models when matrix-exponential schemes
    such as :func:`exp_euler_step` are too expensive.

    Parameters
    ----------
    target : DiffEqModule
        The module whose differential states are to be advanced. Must be
        an :class:`HHTypedNeuron` (single compartment or multi-compartment).
    *args
        Extra positional arguments forwarded to ``target``'s
        :meth:`pre_integral`, :meth:`compute_derivative`, and
        :meth:`post_integral` hooks.

    Returns
    -------
    None
        ``target``'s differential states are updated in place.

    Raises
    ------
    AssertionError
        Raised inside :func:`apply_standard_solver_step` if *target* is
        not a :class:`DiffEqModule`.

    See Also
    --------
    implicit_euler_step : Full Newton iteration on the same residual.
    exp_euler_step : Matrix-exponential exponential Euler step.

    Notes
    -----
    The current time and step size are read from the active
    :mod:`brainstate.environ` context. State leaves are stacked along the
    last axis (``merging='stack'``) before the linear solve.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import backward_euler_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
        ...     backward_euler_step(my_neuron, input_current)  # doctest: +SKIP
    """

    t = brainstate.environ.get('t')
    dt = brainstate.environ.get('dt')

    apply_standard_solver_step(
        _backward_euler,
        target,
        t,
        dt,
        *args,
        merging='stack'  # [n_neuron, n_state]
    )
