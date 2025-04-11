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

import brainunit as u
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from ._base import HHTypedNeuron
from ._integrator_util import apply_standard_solver_step, jacrev_last_dim
from ._misc import set_module_as
from ._protocol import DiffEqModule

__all__ = [
    'exp_euler_step',
]


def _exponential_euler(f, y0, t, dt, args=()):
    dt = u.get_magnitude(dt)
    A, df, aux = jacrev_last_dim(lambda y: f(t, y, *args), y0, has_aux=True)

    # reshape A from "[..., M, M]" to "[-1, M, M]"
    A = A.reshape((-1, A.shape[-2], A.shape[-1]))

    # reshape df from "[..., M]" to "[-1, M]"
    df = df.reshape((-1, df.shape[-1]))

    # Compute exp(hA) and phi(hA)
    n = y0.shape[-1]
    I = jnp.eye(n)
    updates = jax.vmap(
        lambda A_, df_:
        (
            jnp.linalg.solve(
                A_,
                (
                    expm(dt * A_)  # Matrix exponential
                    - I
                )
            ) @ df_
        )
    )(A, df)
    updates = updates.reshape(y0.shape)

    # Compute the new state
    y1 = y0 + updates
    return y1, aux


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
    assert isinstance(target, HHTypedNeuron), (
        f"The target should be a {HHTypedNeuron.__name__}. "
        f"But got {type(target)} instead."
    )
    from braincell._single_compartment import SingleCompartment
    from braincell._multi_compartment import MultiCompartment

    if isinstance(target, SingleCompartment):
        apply_standard_solver_step(
            _exponential_euler,
            target,
            t,
            *args,
            merging_method='stack'
        )

    elif isinstance(target, MultiCompartment):
        apply_standard_solver_step(
            _exponential_euler,
            target,
            t,
            *args,
            merging_method='concat'
        )

    else:
        raise ValueError(f"Unknown target type: {type(target)}")
