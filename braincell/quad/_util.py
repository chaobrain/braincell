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
from typing import Dict, Any, Callable, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braincell._typing import T, DT, Y0, Y1, Aux, Jacobian, VectorField, Args
from .protocol import DiffEqState, DiffEqModule, IndependentIntegration


def environ_time(target: DiffEqModule, t: T = None, dt: DT = None) -> Tuple[T, DT]:
    """
    Resolve the current simulation time and time step.

    Every registered integrator step is invoked by its host as
    ``self.solver(self, *args)``, with no explicit time arguments, so each one
    has to recover ``t`` and ``dt`` from :mod:`brainstate.environ`. This is the
    single definition of how that is done.

    Parameters
    ----------
    target : DiffEqModule
        The module being integrated. Supplies the ``current_time`` fallback
        used when ``t`` is absent from the environment.
    t : u.Quantity[u.second], optional
        An explicit time that overrides the environment. ``None`` (the
        default) means "read from the environment".
    dt : u.Quantity[u.second], optional
        An explicit time step that overrides the environment. ``None`` (the
        default) means "read from the environment".

    Returns
    -------
    t : u.Quantity[u.second]
        The current simulation time. When read from the environment it falls
        back to ``target.current_time`` and then to ``0.0 * u.ms``, so the
        result always carries a time unit.
    dt : u.Quantity[u.second]
        The integration time step.
    """
    if t is None:
        t = brainstate.environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))
    if dt is None:
        dt = brainstate.environ.get('dt')
    return t, dt


def _has_path_prefix(path, prefixes):
    for prefix in prefixes:
        if path[: len(prefix)] == prefix:
            return True
    return False


def _filter_diffeq(independent_modules, excluded_paths, path, value):
    direct_owner_path = path[:-1]
    if direct_owner_path in independent_modules:
        return False
    if _has_path_prefix(path, excluded_paths):
        return False
    return isinstance(value, DiffEqState)


def split_diffeq_states(module: DiffEqModule, *, excluded_paths=()):
    """
    Splits the states of a differential equation module into three categories:
    all states, states to be integrated (diffeq_states), and other states.

    This function traverses the given `DiffEqModule` and identifies all its states.
    It then separates these states into:
      - `diffeq_states`: States that are instances of `DiffEqState` and are not part of
        any submodule of type :class:`IndependentIntegration`.
      - `other_states`: All remaining states.
      - `all_states`: A dictionary of all states in the module.

    The separation is useful for numerical integration routines, where only certain
    states (those representing differential equations) should be integrated, while
    others are treated differently.

    Parameters
    ----------
    module : DiffEqModule
        The differential equation module whose states are to be split.
    excluded_paths : tuple of tuple of str, optional
        State paths to keep out of ``diffeq_states``. A path matches when it is
        a prefix of the state's path, so ``(('channels',),)`` excludes every
        state owned by the ``channels`` subtree. Excluded states still appear in
        ``all_states`` and ``other_states``.

    Returns
    -------
    all_states : Dict[Any, brainstate.State]
        A dictionary of all states in the module.
    diffeq_states : Dict[Any, DiffEqState]
        A dictionary of states that are subject to integration (excluding those in
        :class:`IndependentIntegration` modules).
    other_states : Dict[Any, brainstate.State]
        A dictionary of all other states.

    Notes
    -----
    - States directly owned by submodules of type
      :class:`IndependentIntegration` are excluded from `diffeq_states` to
      allow for independent integration strategies. Child modules remain
      eligible unless they are independently integrated themselves or matched
      by ``excluded_paths``.
    - The function relies on the module's state graph and a custom filter to
      distinguish between state types.

    Examples
    --------
    >>> all_states, diffeq_states, other_states = split_diffeq_states(my_module)
    """
    # Exclude states directly owned by IndependentIntegration modules.
    independent_modules = brainstate.graph.nodes(module, IndependentIntegration, allowed_hierarchy=(1, 1000000000000))
    all_states = brainstate.graph.states(module)
    excluded_paths = tuple(tuple(path) for path in excluded_paths)
    diffeq_states, other_states = all_states.split(
        functools.partial(_filter_diffeq, independent_modules, excluded_paths), ...
    )
    return all_states, diffeq_states, other_states


def _check_diffeq_state_derivative(state: DiffEqState, dt: u.Quantity):
    def _fn_checking(state_val, state_derivative):
        a = u.get_unit(state_derivative) * u.get_unit(dt)
        b = u.get_unit(state_val)
        assert a.has_same_dim(b), f'Unit mismatch. Got {a} != {b}'
        if isinstance(state_derivative, u.Quantity):
            return state_derivative.in_unit(u.get_unit(state_val) / u.get_unit(dt))
        else:
            return state_derivative

    state.derivative = jax.tree.map(_fn_checking, state.value, state.derivative, is_leaf=u.math.is_quantity)


def _check_merging(method: str) -> str:
    """Validate a state-merging method, returning it unchanged."""
    if method not in ('concat', 'stack'):
        raise ValueError(f"Unknown merging method: {method!r}. Expected 'concat' or 'stack'.")
    return method


def _merging(leaves, method: str):
    if _check_merging(method) == 'concat':
        return jnp.concatenate(leaves, axis=-1)
    return jnp.stack(leaves, axis=-1)


def _dict_attr_to_arr(
    a_dict: Dict[Any, brainstate.State],
    attr: str,
    method: str = 'concat',
):
    leaves = jax.tree.leaves({key: getattr(val, attr) for key, val in a_dict.items()})
    return _merging(leaves, method=method)


def _assign_arr_to_states(
    vals: jax.Array,
    states: Dict[Any, brainstate.State],
    method: str = 'concat',
):
    stacked = _check_merging(method) == 'stack'
    leaves, tree_def = jax.tree.flatten({key: state.value for key, state in states.items()})
    index = 0
    vals_like_leaves = []
    for leaf in leaves:
        if stacked:
            vals_like_leaves.append(vals[..., index])
            index += 1
        else:
            width = leaf.shape[-1]
            vals_like_leaves.append(vals[..., index : index + width])
            index += width
    vals_like_states = jax.tree.unflatten(tree_def, vals_like_leaves)
    for key, state_val in vals_like_states.items():
        states[key].value = state_val


def _transform_diffeq_module_into_dimensionless_fn(
    target: DiffEqModule,
    dt: DT,
    method: str = 'concat',
    excluded_paths=(),
):
    _check_merging(method)
    all_states, diffeq_states, other_states = split_diffeq_states(target, excluded_paths=excluded_paths)
    all_state_ids = {id(st) for st in all_states.values()}

    def vector_field(t, y_dimensionless, *args):
        with brainstate.StateTraceStack() as trace:
            # y: dimensionless states
            _assign_arr_to_states(y_dimensionless, diffeq_states, method=method)
            target.compute_derivative(*args)

            # derivative_arr: dimensionless derivatives
            for st in diffeq_states.values():
                _check_diffeq_state_derivative(st, dt)  # THIS is important.
            derivative_dimensionless = _dict_attr_to_arr(diffeq_states, 'derivative', method=method)
            other_vals = {key: st.value for key, st in other_states.items()}

        # check if all states exist in the trace
        for st in trace.states:
            if id(st) not in all_state_ids:
                raise ValueError(f'State {st} is not in the state list.')
        return derivative_dimensionless, other_vals

    return vector_field, diffeq_states, other_states


def apply_standard_solver_step(
    solver_step: Callable[[VectorField, Y0, T, DT, Args], Tuple[Y1, Aux]],
    target: DiffEqModule,
    t: T,
    dt: DT,
    *args,
    merging: str = 'concat',
    excluded_paths=(),
):
    """
    Apply a standard solver step to the given differential equation module.

    This function performs a single step of numerical integration for a differential equation
    system. It handles pre-integration preparation, the actual integration step, and
    post-integration updates.

    The ``solver_step`` should have the following signature::

        solver_step(f, y0, t, dt, args) -> (y1, aux)

    - ``f`` is the function representing the system of differential equations.
    - ``y0`` is the current state of the system.
    - ``t`` is the current time.
    - ``dt`` is the time step for the integration.
    - ``args`` are additional arguments to be passed to the function ``f``.

    Parameters
    ----------
    solver_step : Callable[[Callable, jax.Array, u.Quantity[u.second], u.Quantity[u.second], Any], Any]
        The solver step function that performs the actual numerical integration.
    target : DiffEqModule
        The differential equation module to be integrated.
    t : u.Quantity[u.second]
        The current time of the integration.
    dt: u.Quantity[u.second]
        The time step of the integration.
    *args : Any
        Additional arguments to be passed to the pre_integral, post_integral, and compute_derivative methods.
    merging: str
        The merging method to be used when converting states to arrays.

        - 'concat': Concatenate the states along the last dimension.
        - 'stack': Stack the states along the last dimension.
    """
    # Raised rather than asserted so that ``python -O`` preserves the contract.
    if not isinstance(target, DiffEqModule):
        raise TypeError(f'Target must be a DiffEqModule, but got {type(target)}')
    if not callable(solver_step):
        raise TypeError(f'Solver step must be callable, but got {type(solver_step)}')
    _check_merging(merging)

    # pre integral
    target.pre_integral(*args)
    dimensionless_fn, diffeq_states, other_states = _transform_diffeq_module_into_dimensionless_fn(
        target,
        dt=dt,
        method=merging,
        excluded_paths=excluded_paths,
    )

    # one-step integration
    diffeq_vals, other_vals = solver_step(
        dimensionless_fn, _dict_attr_to_arr(diffeq_states, 'value', method=merging), t, dt, args
    )

    # post integral
    _assign_arr_to_states(diffeq_vals, diffeq_states, method=merging)
    for key, val in other_vals.items():
        other_states[key].value = val
    target.post_integral(*args)


def jacrev_last_dim(
    fn: Callable[[Y0], Y1] | Callable[[Y0], Tuple[Y1, Aux]],
    hid_vals: Y0,
    has_aux: bool = False,
) -> Tuple[Jacobian, Y1] | Tuple[Jacobian, Y1, Aux]:
    """
    Compute the reverse-mode Jacobian of a function with respect to its last dimension.

    This function calculates the Jacobian matrix of the given function `fn`
    with respect to the last dimension of the input `hid_vals`. It uses
    JAX's reverse-mode automatic differentiation (jacrev) for efficient computation.

    Parameters
    ----------
    fn : Callable
        The function for which to compute the Jacobian. It can either return a
        single JAX array or a tuple containing a JAX array and auxiliary values.
    hid_vals : jax.Array
        The input values for which to compute the Jacobian. The last dimension
        is considered as the dimension of interest.
    has_aux : bool, optional
        Whether the function ``fn`` returns auxiliary values. Defaults to False.

    Returns
    -------
    tuple
        If ``has_aux`` is False, a tuple containing the Jacobian matrix and the
        output of ``fn``. If ``has_aux`` is True, a tuple containing the
        Jacobian matrix, the output of ``fn``, and the auxiliary values.

    Raises
    ------
    AssertionError
        If the number of input and output states are not the same.
    """
    if has_aux:
        new_hid_vals, f_vjp, aux = jax.vjp(fn, hid_vals, has_aux=True)
    else:
        new_hid_vals, f_vjp = jax.vjp(fn, hid_vals)
    num_state = new_hid_vals.shape[-1]
    varshape = new_hid_vals.shape[:-1]
    assert num_state == hid_vals.shape[-1], 'Error: the number of input/output states should be the same.'
    g_primals = u.math.broadcast_to(
        u.math.eye(num_state, dtype=u.math.get_dtype(new_hid_vals)),
        (*varshape, num_state, num_state),
    )
    jac = jax.vmap(f_vjp, in_axes=-2, out_axes=-2)(g_primals)
    if has_aux:
        return jac[0], new_hid_vals, aux
    else:
        return jac[0], new_hid_vals


def linearize_flat(
    f: VectorField,
    y0: Y0,
    t: T,
    dt: DT,
    args: Args = (),
) -> Tuple[jax.Array, Jacobian, jax.Array, Aux]:
    """
    Linearize a vector field about ``y0`` and fold the leading axes into one batch axis.

    This is the shared prologue of the locally linearised solvers
    (:func:`~braincell.quad._backward_euler._backward_euler` and
    :func:`~braincell.quad._exp_euler._exponential_euler`): it strips the unit
    from ``dt``, builds the local Jacobian, and reshapes both the Jacobian and
    the derivative so that everything ahead of the state axis becomes a single
    batch axis.

    Parameters
    ----------
    f : Callable
        The vector field ``f(t, y, *args)``, returning ``(dy/dt, aux)``.
    y0 : jax.Array
        The current state, shape ``(..., M)``.
    t : u.Quantity[u.second]
        The current time.
    dt : u.Quantity[u.second]
        The integration time step.
    args : tuple, optional
        Extra positional arguments forwarded to ``f``.

    Returns
    -------
    dt : jax.Array
        The magnitude of ``dt``, cast to ``y0``'s dtype.
    A : jax.Array
        The Jacobian :math:`\\partial f / \\partial y` at ``y0``, shape ``(B, M, M)``.
    df : jax.Array
        The vector field evaluated at ``y0``, shape ``(B, M)``.
    aux : Any
        The auxiliary output of ``f``.
    """
    dtype = y0.dtype
    dt = jnp.asarray(u.get_magnitude(dt), dtype=dtype)
    A, df, aux = jacrev_last_dim(lambda y: f(t, y, *args), y0, has_aux=True)
    # reshape A from "[..., M, M]" to "[-1, M, M]" and df from "[..., M]" to "[-1, M]"
    A = jnp.asarray(A, dtype=dtype).reshape((-1, A.shape[-2], A.shape[-1]))
    df = jnp.asarray(df, dtype=dtype).reshape((-1, df.shape[-1]))
    return dt, A, df, aux
