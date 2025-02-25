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

from typing import Dict, Any, Callable

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from ._protocol import DiffEqState, DiffEqModule


def _check_diffeq_state_derivative(st: DiffEqState, dt):
    a = u.get_unit(st.derivative) * u.get_unit(dt)
    b = u.get_unit(st.value)
    assert a.has_same_dim(b), f'Unit mismatch. Got {a} != {b}'
    if isinstance(st.derivative, u.Quantity):
        st.derivative = st.derivative.in_unit(u.get_unit(st.value) / u.get_unit(dt))


def _dict_derivative_to_arr(a_dict: Dict[Any, DiffEqState]):
    a_dict = {key: val.derivative for key, val in a_dict.items()}
    leaves = jax.tree.leaves(a_dict)
    leaves = [jnp.expand_dims(leaf, axis=0) if leaf.ndim == 0 else leaf
              for leaf in leaves]
    return jnp.concatenate(leaves, axis=-1)


def _dict_state_to_arr(a_dict: Dict[Any, brainstate.State]):
    a_dict = {key: val.value for key, val in a_dict.items()}
    leaves = jax.tree.leaves(a_dict)
    leaves = [jnp.expand_dims(leaf, axis=0) if leaf.ndim == 0 else leaf for leaf in leaves]
    return jnp.concatenate(leaves, axis=-1)


def _assign_arr_to_states(vals: jax.Array, states: Dict[Any, brainstate.State]):
    leaves, tree_def = jax.tree.flatten({key: state.value for key, state in states.items()})
    index = 0
    vals_like_leaves = []
    for leaf in leaves:
        if leaf.ndim == 0:
            vals_like_leaves.append(vals[..., index])
            index += 1
        else:
            vals_like_leaves.append(vals[..., index: index + leaf.shape[-1]])
            index += leaf.shape[-1]
    vals_like_states = jax.tree.unflatten(tree_def, vals_like_leaves)
    for key, state_val in vals_like_states.items():
        states[key].value = state_val


def _transform_diffeq_module_into_dimensionless_fn(target: DiffEqModule):
    all_states = brainstate.graph.states(target)
    diffeq_states, other_states = all_states.split(DiffEqState, ...)
    all_state_ids = {id(st) for st in all_states}

    def vector_field(t, y_dimensionless, *args):
        with brainstate.StateTraceStack() as trace:

            # y: dimensionless states
            _assign_arr_to_states(y_dimensionless, diffeq_states)
            target.compute_derivative(*args)

            # derivative_arr: dimensionless derivatives
            for st in diffeq_states.values():
                _check_diffeq_state_derivative(st, brainstate.environ.get_dt())
            derivative_dimensionless = _dict_derivative_to_arr(diffeq_states)
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
    *args
):
    """
    The explicit Euler step for the differential equations.
    """
    # pre integral
    dt = u.get_magnitude(brainstate.environ.get_dt())
    target.pre_integral(*args)
    dimensionless_fn, diffeq_states, other_states = _transform_diffeq_module_into_dimensionless_fn(target)

    # one-step integration
    diffeq_vals, other_vals = solver_step(
        dimensionless_fn,
        _dict_state_to_arr(diffeq_states),
        t,
        dt,
        args
    )

    # post integral
    _assign_arr_to_states(diffeq_vals, diffeq_states)
    for key, val in other_vals.items():
        other_states[key].value = val
    target.post_integral(*args)
