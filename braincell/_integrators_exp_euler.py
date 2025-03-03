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
from jax.scipy.linalg import expm

from ._base import HHTypedNeuron
from ._integrators_util import apply_standard_solver_step
from ._misc import set_module_as
from ._protocol import DiffEqModule

__all__ = [
    'exp_euler_step',
]


def _exponential_euler(f, y0, t, dt, args=()):
    dt = u.get_magnitude(dt)
    A, df, aux = brainstate.augment.jacfwd(lambda y: f(t, y, *args), return_value=True, has_aux=True)(y0)

    # Compute exp(hA) and phi(hA)
    n = y0.shape[-1]
    exp_hA = expm(dt * A)  # Matrix exponential
    phi_hA = jnp.linalg.solve(A, (exp_hA - jnp.eye(n)))
    # # Regularize if A is ill-conditioned
    # phi_hA = (
    #     jnp.linalg.solve(A, (exp_hA - jnp.eye(n)))
    #     if jnp.linalg.cond(A) < 1e12 else
    #     jnp.eye(n)
    # )
    y1 = y0 + phi_hA @ df
    return y1, aux


@set_module_as('braincell')
def _exp_euler_step_impl(target: DiffEqModule, t: u.Quantity[u.second], *args):
    r"""
    Exponential Euler Integrator for multidimensional ODEs.

    $$
    {\hat {u}}_{n+1}=u_{n}+h_{n}\ \varphi _{1}(h_{n}L_{n})f(u_{n}),
    $$

    where

    $$
    \varphi _{1}(z)={\frac {e^{z}-1}{z}},
    $$

    Parameters
    ----------
    f : callable
        Nonlinear/time-dependent part of the ODE, g(t, y, *args).
        Note here `y` should be dimensionless, `args` can be arrays with physical units.
    y0 : array_like
        Initial condition, shape (n,).
    t : float
        Current time.
    dt : float
        Time step.
    args : tuple, optional
        Additional arguments for the function f.

    Returns
    -------
    y : ndarray
        Solution array, shape (m, n).
    """
    return apply_standard_solver_step(
        _exponential_euler,
        target,
        t,
        *args,
        merging_method='stack'
    )


@set_module_as('braincell')
def exp_euler_step(
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args
):
    r"""
    Perform an exponential Euler step for solving differential equations.

    This function applies the exponential Euler method to solve differential equations
    for a given target module. It can handle both single neurons and populations of neurons.

    Mathematical Description
    -------------------------
    The exponential Euler method is used to solve differential equations of the form:

    $$
    \frac{dy}{dt} = Ay + f(y, t)
    $$

    where $A$ is a linear operator and $f(y, t)$ is a nonlinear function.

    The exponential Euler scheme is given by:

    $$
    y_{n+1} = e^{A\Delta t}y_n + \Delta t\varphi_1(A\Delta t)f(y_n, t_n)
    $$

    where $\varphi_1(z)$ is the first order exponential integrator function defined as:

    $$
    \varphi_1(z) = \frac{e^z - 1}{z}
    $$

    This method is particularly effective for stiff problems where $A$ represents
    the stiff linear part of the system.

    Parameters
    ----------
    target : DiffEqModule
        The target module containing the differential equations to be solved.
        Must be an instance of HHTypedNeuron.
    t : u.Quantity[u.second]
        The current time point in the simulation.
    *args : 
        Additional arguments to be passed to the underlying implementation.

    Raises
    ------
    AssertionError
        If the target is not an instance of :class:`HHTypedNeuron`.

    Notes
    -----
    This function uses vectorization (vmap) to handle populations of neurons efficiently.
    The actual computation of the exponential Euler step is performed in the
    `_exp_euler_step_impl` function, which this function wraps and potentially
    vectorizes for population-level computations.
    """
    assert isinstance(target, HHTypedNeuron), ("The target should be a HHTypedNeuron. "
                                               f"But got {type(target)} instead.")
    integral = lambda: _exp_euler_step_impl(target, t, *args)
    for _ in range(len(target.pop_size)):
        integral = brainstate.augment.vmap(integral, in_states=target.states())
    return integral()
