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

from typing import Dict, Any, Callable, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from ._protocol import DiffEqState, DiffEqModule


def _check_diffeq_state_derivative(
    st: DiffEqState,
    dt: u.Quantity
):
    a = u.get_unit(st.derivative) * u.get_unit(dt)
    b = u.get_unit(st.value)
    assert a.has_same_dim(b), f'Unit mismatch. Got {a} != {b}'
    if isinstance(st.derivative, u.Quantity):
        st.derivative = st.derivative.in_unit(u.get_unit(st.value) / u.get_unit(dt))


def _merging(leaves, method: str):
    if method == 'concat':
        return jnp.concatenate(leaves, axis=-1)
    elif method == 'stack':
        return jnp.stack(leaves, axis=-1)
    else:
        raise ValueError(f'Unknown method: {method}')


def _dict_derivative_to_arr(
    a_dict: Dict[Any, DiffEqState],
    method: str = 'concat',
):
    a_dict = {key: val.derivative for key, val in a_dict.items()}
    leaves = jax.tree.leaves(a_dict)
    return _merging(leaves, method=method)


def _dict_state_to_arr(
    a_dict: Dict[Any, brainstate.State],
    method: str = 'concat',
):
    a_dict = {key: val.value for key, val in a_dict.items()}
    leaves = jax.tree.leaves(a_dict)
    return _merging(leaves, method=method)


def _assign_arr_to_states(
    vals: jax.Array,
    states: Dict[Any, brainstate.State],
    method: str = 'concat',
):
    leaves, tree_def = jax.tree.flatten({key: state.value for key, state in states.items()})
    index = 0
    vals_like_leaves = []
    for leaf in leaves:
        if method == 'stack':
            vals_like_leaves.append(vals[..., index])
            index += 1
        elif method == 'concat':
            vals_like_leaves.append(vals[..., index: index + leaf.shape[-1]])
            index += leaf.shape[-1]
        else:
            raise ValueError(f'Unknown method: {method}')
    vals_like_states = jax.tree.unflatten(tree_def, vals_like_leaves)
    for key, state_val in vals_like_states.items():
        states[key].value = state_val


def _transform_diffeq_module_into_dimensionless_fn(
    target: DiffEqModule,
    method: str = 'concat'
):
    assert method in ['concat', 'stack'], f'Unknown method: {method}'

    all_states = brainstate.graph.states(target)
    diffeq_states, other_states = all_states.split(DiffEqState, ...)
    all_state_ids = {id(st) for st in all_states.values()}

    def vector_field(t, y_dimensionless, *args):
        with brainstate.StateTraceStack() as trace:

            # y: dimensionless states
            _assign_arr_to_states(y_dimensionless, diffeq_states, method=method)
            target.compute_derivative(*args)

            # derivative_arr: dimensionless derivatives
            for st in diffeq_states.values():
                _check_diffeq_state_derivative(st, brainstate.environ.get_dt())
            derivative_dimensionless = _dict_derivative_to_arr(diffeq_states, method=method)
            other_vals = {key: st.value for key, st in other_states.items()}

        # check if all states exist in the trace
        for st in trace.states:
            if id(st) not in all_state_ids:
                raise ValueError(f'State {st} is not in the state list.')
        return derivative_dimensionless, other_vals

    return vector_field, diffeq_states, other_states


def apply_standard_solver_step(
    solver_step: Callable[[Callable, jax.Array, u.Quantity[u.second], u.Quantity[u.second], Any], Any],
    target: DiffEqModule,
    t: u.Quantity[u.second],
    *args,
    merging_method: str = 'concat',
):
    """
    Apply a standard solver step to the given differential equation module.

    This function performs a single step of numerical integration for a differential equation
    system. It handles pre-integration preparation, the actual integration step, and
    post-integration updates.

    Parameters
    ----------
    solver_step : Callable[[Callable, jax.Array, u.Quantity[u.second], u.Quantity[u.second], Any], Any]
        The solver step function that performs the actual numerical integration.
    target : DiffEqModule
        The differential equation module to be integrated.
    t : u.Quantity[u.second]
        The current time of the integration.
    *args : Any
        Additional arguments to be passed to the pre_integral, post_integral, and compute_derivative methods.
    merging_method: str
        The merging method to be used when converting states to arrays.

        - 'concat': Concatenate the states along the last dimension.
        - 'stack': Stack the states along the last dimension.

    Returns
    -------
    None
        This function updates the states of the target module in-place and does not return a value.
    """

    assert merging_method in ['concat', 'stack'], f'Unknown merging method: {merging_method}'

    # pre integral
    dt = u.get_magnitude(brainstate.environ.get_dt())
    target.pre_integral(*args)
    dimensionless_fn, diffeq_states, other_states = (
        _transform_diffeq_module_into_dimensionless_fn(
            target,
            method=merging_method,
        )
    )

    # one-step integration
    diffeq_vals, other_vals = solver_step(
        dimensionless_fn,
        _dict_state_to_arr(diffeq_states, method=merging_method),
        t,
        dt,
        args
    )

    # post integral
    _assign_arr_to_states(diffeq_vals, diffeq_states, method=merging_method)
    for key, val in other_vals.items():
        other_states[key].value = val
    target.post_integral(*args)


def jacrev_last_dim(
    fn: Callable[[...], jax.Array] | Callable[[...], Tuple[jax.Array, Any]],
    hid_vals: jax.Array,
    has_aux: bool = False,
) -> Tuple[jax.Array, jax.Array] | Tuple[jax.Array, jax.Array, Any]:
    """
    Compute the reverse-mode Jacobian of a function with respect to its last dimension.

    This function calculates the Jacobian matrix of the given function `fn`
    with respect to the last dimension of the input `hid_vals`. It uses
    JAX's reverse-mode automatic differentiation (jacrev) for efficient computation.

    Args:
        fn (Callable[[...], jax.Array] | Callable[[...], Tuple[jax.Array, Any]]):
            The function for which to compute the Jacobian. It can either return a single
            JAX array or a tuple containing a JAX array and auxiliary values.
        hid_vals (jax.Array):
            The input values for which to compute the Jacobian. The last dimension is
            considered as the dimension of interest.
        has_aux (bool, optional):
            Whether the function `fn` returns auxiliary values. Defaults to False.

    Returns:
        Tuple[jax.Array, jax.Array] | Tuple[jax.Array, jax.Array, Any]:
            If `has_aux` is False, returns a tuple containing the Jacobian matrix and the
            output of the function `fn`. If `has_aux` is True, returns a tuple containing
            the Jacobian matrix, the output of the function `fn`, and the auxiliary values.

    Raises:
        AssertionError:
            If the number of input and output states are not the same.
    """
    if has_aux:
        new_hid_vals, f_vjp, aux = jax.vjp(fn, hid_vals, has_aux=True)
    else:
        new_hid_vals, f_vjp = jax.vjp(fn, hid_vals)
    num_state = new_hid_vals.shape[-1]
    varshape = new_hid_vals.shape[:-1]
    assert num_state == hid_vals.shape[-1], 'Error: the number of input/output states should be the same.'
    g_primals = u.math.broadcast_to(u.math.eye(num_state), (*varshape, num_state, num_state))
    jac = jax.vmap(f_vjp, in_axes=-2, out_axes=-2)(g_primals)
    if has_aux:
        return jac[0], new_hid_vals, aux
    else:
        return jac[0], new_hid_vals
