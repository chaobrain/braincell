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
from ._integrator_protocol import DiffEqModule
from ._integrator_util import apply_standard_solver_step, jacrev_last_dim

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


@set_module_as('braincell')
def backward_euler_step(target: DiffEqModule, *args):
    r"""
    Perform a backward (implicit) Euler step for solving differential equations.

    This function applies the backward Euler method to solve differential equations
    for a given target module. It can handle both single neurons and populations of neurons.

    Mathematical Description
    -------------------------
    The backward Euler method is used to solve differential equations of the form:

    $$
    \frac{dy}{dt} = f(y, t)
    $$

    The implicit Euler scheme is given by:

    $$
    y_{n+1} = y_n + \Delta t \, f(y_{n+1}, t_{n+1})
    $$

    which is implicit in $y_{n+1}$. In practice, the system is linearized using the Jacobian:

    $$
    (I - \Delta t J) \Delta y = \Delta t f(y_n, t_n), \quad y_{n+1} = y_n + \Delta y
    $$

    Parameters
    ----------
    target : DiffEqModule
        The target module containing the differential equations to be solved.
        Must be an instance of HHTypedNeuron.
    *args :
        Additional arguments to be passed to the underlying implementation.

    Raises
    ------
    AssertionError
        If the target is not an instance of :class:`HHTypedNeuron`.

    Notes
    -----
    This function uses vectorization (vmap) to handle populations of neurons efficiently.
    The actual computation of the backward Euler step is performed in the
    `_backward_euler` function, which this function wraps and potentially
    vectorizes for population-level computations.
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
