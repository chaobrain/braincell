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

import functools
from typing import Dict

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from braincell._misc import set_module_as
from braincell._typing import Path
from ._protocol import DiffEqModule
from ._registry import register_integrator
from ._util import (
    apply_standard_solver_step,
    jacrev_last_dim,
    _check_diffeq_state_derivative,
    split_diffeq_states,
)

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
    dtype = y0.dtype
    dt = jnp.asarray(u.get_magnitude(dt), dtype=dtype)
    A, df, aux = jacrev_last_dim(lambda y: f(t, y, *args), y0, has_aux=True)

    # reshape A from "[..., M, M]" to "[-1, M, M]"
    A = jnp.asarray(A, dtype=dtype).reshape((-1, A.shape[-2], A.shape[-1]))

    # reshape df from "[..., M]" to "[-1, M]"
    df = jnp.asarray(df, dtype=dtype).reshape((-1, df.shape[-1]))

    # Compute exp(hA) and phi(hA)
    n = y0.shape[-1]
    I = jnp.eye(n, dtype=dtype)
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


@register_integrator(
    "exp_euler",
    category="exponential",
    order=1,
    description="Coupled exponential Euler step linearizing the full state vector.",
)
@set_module_as('braincell.quad')
def exp_euler_step(target: DiffEqModule, *args):
    r"""Advance one step with the (coupled) exponential Euler method.

    The exponential Euler method targets semi-linear ODEs of the form

    .. math::

        \frac{dy}{dt} = A(y_n)\, y + g(t, y),

    where the local Jacobian :math:`A(y_n) = \partial f / \partial y` is
    treated implicitly via the matrix exponential while the residual
    :math:`g` is held frozen at :math:`y_n`. The update reads

    .. math::

        y_{n+1} = y_n + \Delta t \, \varphi_1(\Delta t \, A)\, f(t_n, y_n),
        \qquad
        \varphi_1(z) = \frac{e^{z} - 1}{z}.

    Equivalently, with :math:`A` the local linearization,

    .. math::

        y_{n+1} = A^{-1}\!\left(e^{\Delta t A} - I\right) f(t_n, y_n) + y_n.

    Because the linear part is integrated exactly, the scheme is
    A-stable for the local linearization and remains accurate where
    forward Euler would blow up. It is the workhorse explicit-style
    integrator for Hodgkin-Huxley type membranes.

    Unlike :func:`ind_exp_euler_step`, this routine treats the entire
    state vector as a single coupled block: it builds the dense local
    Jacobian over all :class:`DiffEqState` leaves, computes its matrix
    exponential, and applies it in one shot. That captures cross-state
    coupling exactly to first order in :math:`\Delta t` but costs
    :math:`O(M^3)` per step in the state dimension :math:`M`.

    Parameters
    ----------
    target : DiffEqModule
        The neuron model to advance. Must be an :class:`HHTypedNeuron`
        subclass — currently :class:`SingleCompartment` (states are
        stacked along ``[n_neuron, n_state]``) or
        :class:`braincell.Cell` (states are concatenated along
        ``[n_neuron, n_compartment * n_state]``).
    *args
        Extra positional arguments forwarded to ``target``'s
        :meth:`pre_integral`, :meth:`compute_derivative`, and
        :meth:`post_integral` hooks (typically the input current
        for this step).

    Returns
    -------
    None
        ``target``'s differential states are updated in place.

    Raises
    ------
    AssertionError
        If *target* is not an :class:`HHTypedNeuron`.
    ValueError
        If *target* is an :class:`HHTypedNeuron` of an unsupported subtype.

    See Also
    --------
    ind_exp_euler_step : State-by-state variant that linearizes each
        :class:`DiffEqState` independently.
    backward_euler_step : Linearized backward Euler counterpart.

    Notes
    -----
    The current time and step size are read from the active
    :mod:`brainstate.environ` context.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import exp_euler_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
        ...     exp_euler_step(my_neuron, input_current)    # doctest: +SKIP
    """
    from braincell._base import HHTypedNeuron
    from braincell._multi_compartment import Cell
    from braincell._single_compartment import SingleCompartment
    assert isinstance(target, HHTypedNeuron), (
        f"The target should be a {HHTypedNeuron.__name__}. "
        f"But got {type(target)} instead."
    )
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')

    if isinstance(target, SingleCompartment):
        apply_standard_solver_step(
            _exponential_euler,
            target,
            t,
            dt,
            *args,
            merging='stack'  # [n_neuron, n_state]
        )

    elif isinstance(target, Cell):
        apply_standard_solver_step(
            _exponential_euler,
            target,
            t,
            dt,
            *args,
            merging='concat'  # [n_neuron, n_compartment * n_state]
        )

    else:
        raise ValueError(f"Unknown target type: {type(target)}")


@register_integrator(
    "ind_exp_euler",
    category="exponential",
    order=1,
    description="Independent exponential Euler step (per-state linearization).",
)
@set_module_as('braincell.quad')
def ind_exp_euler_step(target: DiffEqModule, *args, excluded_paths=()):
    r"""Advance each :class:`DiffEqState` independently with exponential Euler.

    This is the *decoupled* sibling of :func:`exp_euler_step`. Instead of
    building one global Jacobian over the full state vector, the routine
    iterates over every :class:`DiffEqState` :math:`y^{(k)}` in *target*
    and treats the others as frozen at their current values, fitting the
    local linearization

    .. math::

        \frac{d y^{(k)}}{dt} \approx \lambda^{(k)} y^{(k)} + b^{(k)}

    via :func:`brainstate.transform.vector_grad` and applying the
    component-wise exponential Euler update

    .. math::

        y^{(k)}_{n+1} = y^{(k)}_n
            + \Delta t \, \varphi_1\!\left(\Delta t \, \lambda^{(k)}\right) f^{(k)}(t_n, y_n),

    using :func:`brainunit.math.exprel` to evaluate
    :math:`\varphi_1(z) = (e^{z} - 1)/z` accurately near :math:`z = 0`.

    The trade-off compared with :func:`exp_euler_step`:

    - :func:`exp_euler_step` is more accurate when states are tightly
      coupled, because it builds the full :math:`M \times M` Jacobian and
      uses a true matrix exponential.
    - :func:`ind_exp_euler_step` is much cheaper for large state vectors
      and is the right choice when each variable is mostly self-coupled
      (the typical pattern for HH-style gating variables and ion
      concentrations) and especially when the voltage equation is being
      solved by a separate solver (see :func:`staggered_step`).

    Parameters
    ----------
    target : DiffEqModule
        The module whose :class:`DiffEqState` leaves will be advanced.
    *args
        Extra positional arguments forwarded to ``target``'s
        :meth:`pre_integral`, :meth:`compute_derivative`, and
        :meth:`post_integral` hooks.
    excluded_paths : tuple of tuple, optional
        Iterable of state paths to skip. Each entry is a tuple of attribute
        names identifying a :class:`DiffEqState` inside *target*'s state
        graph. The classic use is ``excluded_paths=[('V',)]`` from
        :func:`staggered_step`, which leaves the membrane voltage
        untouched so the upstream cable solve is preserved.

    Returns
    -------
    None
        Differential states are updated in place; auxiliary (non-DiffEq)
        states are written from the trace captured during the first
        Jacobian evaluation.

    Raises
    ------
    AssertionError
        If *target* is not a :class:`DiffEqModule`, or if it has no
        :class:`DiffEqState` leaves.
    ValueError
        If a state value uses an unsupported (non-floating) dtype, or if
        an unknown state appears in the trace.

    See Also
    --------
    exp_euler_step : Coupled (full-Jacobian) exponential Euler.
    staggered_step : Operator-splitting scheme that combines DHS for the
        voltage equation with this routine for everything else.

    Notes
    -----
    The current time and step size are read from the active
    :mod:`brainstate.environ` context.

    Examples
    --------

    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> from braincell.quad import ind_exp_euler_step
        >>> with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
        ...     ind_exp_euler_step(
        ...         my_cell, input_current,
        ...         excluded_paths=[('V',)],
        ...     )                                           # doctest: +SKIP
    """
    assert isinstance(target, DiffEqModule), (
        f"The target should be a {DiffEqModule.__name__}. "
        f"But got {type(target)} instead."
    )
    t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    dt = brainstate.environ.get('dt')

    # Pre-integration hook (e.g., update gating variables)
    target.pre_integral(*args)

    # Retrieve all states from the target module
    all_states, diffeq_states, other_states = split_diffeq_states(target)

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

            # Compute derivatives for all states
            target.compute_derivative(*args)

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
            functools.partial(vector_field, key),
            argnums=0,
            return_value=True,
            has_aux=True,
            unit_aware=False,
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

    # Post-integration hook (e.g., apply constraints)
    target.post_integral(*args)
