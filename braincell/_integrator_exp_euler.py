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

import functools
from typing import Dict

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from ._base import HHTypedNeuron
from ._integrator_util import apply_standard_solver_step, jacrev_last_dim, _check_diffeq_state_derivative
from ._misc import set_module_as
from ._protocol import DiffEqState, DiffEqModule
from ._typing import Path, T, DT

__all__ = [
    'exp_euler_step',
    'ind_exp_euler_step',
]


def power_iteration_expm(A, num_steps=20, method='scipy'):
    """
    A naive implementation of matrix exponential using the power series definition.
    This is for demonstration and is not numerically stable or efficient for general use.
    """
    if method == 'scipy':
        return expm(A)
    elif method == 'approx':
        n = A.shape[0]
        result = jnp.eye(n, dtype=A.dtype)
        term = jnp.eye(n, dtype=A.dtype)
        for k in range(1, num_steps + 1):
            term = term @ A / k
            result = result + term
        return result
    else:
        raise ValueError('Unsupported method "{}"'.format(method))


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
                    power_iteration_expm(dt * A_, method='scipy')  # Matrix exponential
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
def exp_euler_step(target: DiffEqModule, t: T, dt: DT, *args):
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
    dt : u.Quantity[u.second]
        The numerical time step of the integration step.
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
            dt,
            *args,
            merging='stack'
        )

    elif isinstance(target, MultiCompartment):
        apply_standard_solver_step(
            _exponential_euler,
            target,
            t,
            dt,
            *args,
            merging='concat'
        )

    else:
        raise ValueError(f"Unknown target type: {type(target)}")


@set_module_as('braincell')
def ind_exp_euler_step(target: DiffEqModule, t: T, dt: DT, *args, excluded_paths=()):
    """
    Perform an independent exponential Euler integration step for each DiffEqState in the target module.

    This function applies the exponential Euler method to each differential equation state (DiffEqState)
    in the target module independently, rather than as a coupled system. This is in contrast to
    :func:`exp_euler_step`, which typically handles the system as a whole (potentially vectorized for populations).
    The independent approach is useful when the states are weakly coupled or can be updated separately.

    Comparison with :func:`exp_euler_step`:

    - :func:`exp_euler_step` applies the exponential Euler method to the entire system, handling all states together,
      which is suitable for tightly coupled systems or when vectorization is desired.
    - :func:`ind_exp_euler_step` updates each DiffEqState independently, which can be more efficient or appropriate
      for loosely coupled or independent states, but may not capture interactions between states as accurately.

    Parameters
    ----------
    target : DiffEqModule
        The module containing the differential equation states to be integrated.
        Must be an instance of HHTypedNeuron.
    t : T
        The current simulation time.
    dt : DT
        The integration time step.
    args : Any
        Additional arguments passed to the module's integration hooks.

    Notes
    -----
    - The function uses `brainstate.transform.vector_grad` to compute the linearization and derivative
      for each state, and applies the exponential Euler update formula using the `exprel` function.
    - State values are updated in-place, and auxiliary state values are handled for consistency.
    - Data type checks ensure compatibility with JAX and the exponential Euler method.

    Raises
    ------
    AssertionError
        If the target is not an instance of :class:`HHTypedNeuron`.
    ValueError
        If the input data type is not a supported floating point type.
        If a state in the trace is not found in the state list.
    """
    assert isinstance(target, DiffEqModule), (
        f"The target should be a {DiffEqModule.__name__}. "
        f"But got {type(target)} instead."
    )

    # Retrieve all states from the target module
    all_states = brainstate.graph.states(target)

    # Split states into differential equation states and other states
    diffeq_states, other_states = all_states.split(DiffEqState, ...)

    # Collect all state object ids for trace validation
    all_state_ids = {id(st) for st in all_states.values()}

    def vector_field(
        diffeq_state_key: Path,
        diffeq_state_val: brainstate.typing.ArrayLike,
        other_diffeq_state_vals: Dict,
        other_state_vals: Dict,
    ):
        """
        Compute the derivative for a single DiffEqState, given its value and the values of other states.

        Parameters
        ----------
        diffeq_state_key : Path
            The key identifying the current DiffEqState.
        diffeq_state_val : ArrayLike
            The value of the current DiffEqState.
        other_diffeq_state_vals : dict
            Values of other DiffEqStates.
        other_state_vals : dict
            Values of other (non-differential) states.

        Returns
        -------
        tuple
            (diffeq_state_derivative, other_state_vals_out)
        """
        # Ensure the state value is a supported floating point type
        dtype = u.math.get_dtype(diffeq_state_val)
        if dtype not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
            raise ValueError(
                f'The input data type should be float64, float32, float16, or bfloat16 '
                f'when using Exponential Euler method. But we got {dtype}.'
            )

        with brainstate.StateTraceStack() as trace:
            # Assign the current and other state values
            all_states[diffeq_state_key].value = diffeq_state_val
            for key, val in other_diffeq_state_vals.items():
                all_states[key].value = val
            for key, val in other_state_vals.items():
                all_states[key].value = val

            # Pre-integration hook (e.g., update gating variables)
            target.pre_integral(*args)

            # Compute derivatives for all states
            target.compute_derivative(*args)

            # Post-integration hook (e.g., apply constraints)
            target.post_integral(*args)

            # Validate and retrieve the derivative for the current state
            _check_diffeq_state_derivative(all_states[diffeq_state_key], dt)  # THIS is important.
            diffeq_state_derivative = all_states[diffeq_state_key].derivative
            # Collect updated values for other states
            other_state_vals_out = {key: other_states[key].value for key in other_state_vals.keys()}

        # Ensure all states in the trace are known
        for st in trace.states:
            if id(st) not in all_state_ids:
                raise ValueError(f'State {st} is not in the state list.')

        return diffeq_state_derivative, other_state_vals_out

    # Prepare dictionaries of current state values
    other_state_vals = {k: v.value for k, v in other_states.items()}
    diffeq_state_vals = {k: v.value for k, v in diffeq_states.items()}
    assert len(diffeq_states) > 0, "No DiffEqState found in the target module."

    # data to capture the integrated values of DiffEqStates
    integrated_diffeq_state_vals = dict()

    # Iterate over each DiffEqState and apply the exponential Euler update independently
    i = 0
    for key in diffeq_states.keys():
        if key in excluded_paths:
            continue

        # Compute the linearization (Jacobian), derivative, and auxiliary outputs
        linear, derivative, aux = brainstate.transform.vector_grad(
            functools.partial(vector_field, key), argnums=0, return_value=True, has_aux=True, unit_aware=False,
        )(
            diffeq_state_vals[key],  # Current DiffEqState value
            {k: v for k, v in diffeq_state_vals.items() if k != key},  # Other DiffEqState values
            other_state_vals,  # Other state values
        )
        # Convert linearization to a unit-aware quantity
        linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
        # Compute the exponential relative function phi(dt * linear)
        phi = u.math.exprel(dt * linear)
        # Apply the exponential Euler update formula
        integrated_diffeq_state_vals[key] = all_states[key].value + dt * phi * derivative

        if i == 0:
            # Update other states with auxiliary outputs (only on first iteration)
            for k, st in other_states.items():
                st.value = aux[k]

        i += 1

    # Assign the integrated values back to the corresponding DiffEqStates
    for k, st in diffeq_states.items():
        st.value = integrated_diffeq_state_vals[k]


@set_module_as('braincell')
def ind_exp_euler_step(target: DiffEqModule, t: T, dt: DT, *args, excluded_paths=()):
    """
    Perform an independent exponential Euler integration step for each DiffEqState in the target module.

    This function applies the exponential Euler method to each differential equation state (DiffEqState)
    in the target module independently, rather than as a coupled system. This is in contrast to
    :func:`exp_euler_step`, which typically handles the system as a whole (potentially vectorized for populations).
    The independent approach is useful when the states are weakly coupled or can be updated separately.

    Comparison with :func:`exp_euler_step`:

    - :func:`exp_euler_step` applies the exponential Euler method to the entire system, handling all states together,
      which is suitable for tightly coupled systems or when vectorization is desired.
    - :func:`ind_exp_euler_step` updates each DiffEqState independently, which can be more efficient or appropriate
      for loosely coupled or independent states, but may not capture interactions between states as accurately.

    Parameters
    ----------
    target : DiffEqModule
        The module containing the differential equation states to be integrated.
        Must be an instance of HHTypedNeuron.
    t : T
        The current simulation time.
    dt : DT
        The integration time step.
    args : Any
        Additional arguments passed to the module's integration hooks.
    excluded_paths: tuple
        The path to exclude from the integration step. This is useful for skipping certain states.

    Notes
    -----
    - The function uses `brainstate.transform.vector_grad` to compute the linearization and derivative
      for each state, and applies the exponential Euler update formula using the `exprel` function.
    - State values are updated in-place, and auxiliary state values are handled for consistency.
    - Data type checks ensure compatibility with JAX and the exponential Euler method.

    Raises
    ------
    AssertionError
        If the target is not an instance of :class:`HHTypedNeuron`.
    ValueError
        If the input data type is not a supported floating point type.
        If a state in the trace is not found in the state list.
    """
    assert isinstance(target, DiffEqModule), (
        f"The target should be a {DiffEqModule.__name__}. "
        f"But got {type(target)} instead."
    )

    # Retrieve all states from the target module
    all_states = brainstate.graph.states(target)

    # Split states into differential equation states and other states
    diffeq_states, other_states = all_states.split(DiffEqState, ...)

    # Collect all state object ids for trace validation
    all_state_ids = {id(st) for st in all_states.values()}

    def vector_field(
        diffeq_state_key: Path,
        diffeq_state_val: brainstate.typing.ArrayLike,
        other_diffeq_state_vals: Dict,
        other_state_vals: Dict,
    ):
        """
        Compute the derivative for a single DiffEqState, given its value and the values of other states.

        Parameters
        ----------
        diffeq_state_key : Path
            The key identifying the current DiffEqState.
        diffeq_state_val : ArrayLike
            The value of the current DiffEqState.
        other_diffeq_state_vals : dict
            Values of other DiffEqStates.
        other_state_vals : dict
            Values of other (non-differential) states.

        Returns
        -------
        tuple
            (diffeq_state_derivative, other_state_vals_out)
        """
        # Ensure the state value is a supported floating point type
        dtype = u.math.get_dtype(diffeq_state_val)
        if dtype not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
            raise ValueError(
                f'The input data type should be float64, float32, float16, or bfloat16 '
                f'when using Exponential Euler method. But we got {dtype}.'
            )

        with brainstate.StateTraceStack() as trace:
            # Assign the current and other state values
            all_states[diffeq_state_key].value = diffeq_state_val
            for key, val in other_diffeq_state_vals.items():
                all_states[key].value = val
            for key, val in other_state_vals.items():
                all_states[key].value = val

            # Pre-integration hook (e.g., update gating variables)
            target.pre_integral(*args)

            # Compute derivatives for all states
            target.compute_derivative(*args)

            # Post-integration hook (e.g., apply constraints)
            target.post_integral(*args)

            # Validate and retrieve the derivative for the current state
            _check_diffeq_state_derivative(all_states[diffeq_state_key], dt)  # THIS is important.
            diffeq_state_derivative = all_states[diffeq_state_key].derivative
            # Collect updated values for other states
            other_state_vals_out = {key: other_states[key].value for key in other_state_vals.keys()}

        # Ensure all states in the trace are known
        for st in trace.states:
            if id(st) not in all_state_ids:
                raise ValueError(f'State {st} is not in the state list.')

        return diffeq_state_derivative, other_state_vals_out

    # Prepare dictionaries of current state values
    other_state_vals = {k: v.value for k, v in other_states.items()}
    diffeq_state_vals = {k: v.value for k, v in diffeq_states.items()}
    assert len(diffeq_states) > 0, "No DiffEqState found in the target module."

    # data to capture the integrated values of DiffEqStates
    integrated_diffeq_state_vals = dict()
    
    # Iterate over each DiffEqState and apply the exponential Euler update independently
    i = 0
    for key in diffeq_states.keys():
        if key in excluded_paths:
            continue

        # Compute the linearization (Jacobian), derivative, and auxiliary outputs
        linear, derivative, aux = brainstate.transform.vector_grad(
            functools.partial(vector_field, key), argnums=0, return_value=True, has_aux=True, unit_aware=False,
        )(
            diffeq_state_vals[key],  # Current DiffEqState value
            {k: v for k, v in diffeq_state_vals.items() if k != key},  # Other DiffEqState values
            other_state_vals,  # Other state values
        )
        # Convert linearization to a unit-aware quantity
        linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
        # Compute the exponential relative function phi(dt * linear)
        phi = u.math.exprel(dt * linear)
        # Apply the exponential Euler update formula
        integrated_diffeq_state_vals[key] = all_states[key].value + dt * phi * derivative

        if i == 0:
            # Update other states with auxiliary outputs (only on first iteration)
            for k, st in other_states.items():
                st.value = aux[k]

        i += 1

    # Assign the integrated values back to the corresponding DiffEqStates
    for k, st in diffeq_states.items():
        if k in excluded_paths:
            continue
        st.value = integrated_diffeq_state_vals[k]


