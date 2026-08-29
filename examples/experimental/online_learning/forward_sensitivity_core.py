# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Low-level exact forward sensitivity for a functionalized BrainCell step.

This experimental layer owns state tracing, parameter coordinates, full and
compact sensitivity recurrence, and parameter-dependent initialization. Use
``rollout_gradients`` for the higher-level training-facing engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np


class FunctionalStep(NamedTuple):
    """A stateful step exposed as an explicit state-value transition."""

    function: brainstate.transform.StatefulFunction
    state_trace: brainstate.StateTraceStack
    parameter_names: tuple[str, ...]
    parameter_states: tuple[brainstate.ParamState, ...]
    parameter_indices: tuple[int, ...]

    def state_values(self):
        """Return current values in the state order expected by ``jaxpr_call``."""
        return self.state_trace.get_state_values()

    def call(self, state_values, step_data):
        """Evaluate the pure state-value transition without mutating model states."""
        return self.function.jaxpr_call(state_values, step_data)


class ForwardSensitivityResult(NamedTuple):
    """Outputs and final carry of an exact forward-sensitivity rollout."""

    final_state_values: Any
    final_state_tangents: Any
    final_gradient: Any
    losses: Any
    local_gradients: Any
    prefix_gradients: Any


@dataclass(frozen=True)
class ParameterCoordinates:
    """Coordinate metadata for array-valued optimizer roots."""

    names: tuple[str, ...]
    state_indices: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    slices: tuple[slice, ...]
    size: int

    def seed(self, state_values):
        """Return one full-state tangent direction per parameter coordinate."""
        tangents = list(_broadcast_zero_tangents(state_values, self.size))
        for state_index, shape, coordinate_slice in zip(self.state_indices, self.shapes, self.slices):
            value = state_values[state_index]
            leaves, treedef = jax.tree.flatten(value)
            if len(leaves) != 1:
                raise ValueError("Experimental parameter coordinates require one array leaf per optimizer root.")
            leaf = leaves[0]
            coordinate_count = coordinate_slice.stop - coordinate_slice.start
            flat = jnp.zeros((self.size, coordinate_count), dtype=leaf.dtype)
            directions = jnp.arange(coordinate_slice.start, coordinate_slice.stop)
            flat = flat.at[directions, jnp.arange(coordinate_count)].set(1.0)
            tangents[state_index] = treedef.unflatten((flat.reshape((self.size,) + shape),))
        return tuple(tangents)

    def flatten(self, parameter_values: Mapping[str, object]):
        """Flatten a complete stable-name parameter mapping in coordinate order."""
        if tuple(parameter_values) != self.names:
            raise KeyError(
                f"Parameter mapping keys/order differ: expected {self.names!r}, got {tuple(parameter_values)!r}."
            )
        leaves = []
        for name, shape in zip(self.names, self.shapes):
            value_leaves = jax.tree.leaves(parameter_values[name])
            if len(value_leaves) != 1 or tuple(value_leaves[0].shape) != shape:
                raise ValueError(f"Parameter {name!r} no longer matches coordinate shape {shape!r}.")
            leaves.append(jnp.ravel(value_leaves[0]))
        return jnp.concatenate(leaves) if leaves else jnp.asarray([])

    def unflatten(self, coordinate_values) -> dict[str, object]:
        """Restore a coordinate vector to stable-name array leaves."""
        if tuple(coordinate_values.shape) != (self.size,):
            raise ValueError(f"Coordinate vector must have shape ({self.size},), got {coordinate_values.shape!r}.")
        return {
            name: coordinate_values[coordinate_slice].reshape(shape)
            for name, shape, coordinate_slice in zip(self.names, self.shapes, self.slices)
        }


@dataclass(frozen=True)
class ActiveStateSelection:
    """Select active flattened coordinates from one traced state object."""

    name: str
    state: brainstate.State
    indices: tuple[int, ...]


@dataclass(frozen=True)
class _ResolvedActiveStateSelection:
    name: str
    state_index: int
    indices: tuple[int, ...]
    coordinate_slice: slice


@dataclass(frozen=True)
class ActiveStateProjection:
    """Project full runtime tangents to a compact active-state matrix."""

    selections: tuple[_ResolvedActiveStateSelection, ...]
    size: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(selection.name for selection in self.selections)

    def values(self, state_values):
        """Return active primal coordinates as one flat vector."""
        coordinates = []
        for selection in self.selections:
            leaf = _single_array_leaf(state_values[selection.state_index], label=selection.name)
            coordinates.append(jnp.ravel(leaf)[jnp.asarray(selection.indices)])
        return jnp.concatenate(coordinates) if coordinates else jnp.asarray([])

    def extract_tangents(self, full_state_tangents):
        """Gather active coordinates from a direction-leading tangent tree."""
        direction_count = _direction_count(full_state_tangents)
        coordinates = []
        for selection in self.selections:
            leaf = _single_array_leaf(full_state_tangents[selection.state_index], label=selection.name)
            flattened = leaf.reshape((direction_count, -1))
            coordinates.append(flattened[:, jnp.asarray(selection.indices)])
        if not coordinates:
            return jnp.zeros((direction_count, 0))
        return jnp.concatenate(coordinates, axis=1)

    def embed_tangents(
        self,
        state_values,
        active_state_tangents,
        parameter_coordinates: ParameterCoordinates,
    ):
        """Scatter compact state sensitivity and parameter basis into full tangents."""
        expected = (parameter_coordinates.size, self.size)
        if tuple(active_state_tangents.shape) != expected:
            raise ValueError(
                f"Active state tangents must have shape {expected!r}, got {active_state_tangents.shape!r}."
            )
        tangents = list(parameter_coordinates.seed(state_values))
        for selection in self.selections:
            value = state_values[selection.state_index]
            leaves, treedef = jax.tree.flatten(value)
            if len(leaves) != 1:
                raise ValueError(f"Active state {selection.name!r} must contain one array leaf.")
            leaf = leaves[0]
            tangent = jnp.zeros((parameter_coordinates.size, leaf.size), dtype=leaf.dtype)
            tangent = tangent.at[:, jnp.asarray(selection.indices)].set(
                active_state_tangents[:, selection.coordinate_slice]
            )
            tangents[selection.state_index] = treedef.unflatten(
                (tangent.reshape((parameter_coordinates.size,) + leaf.shape),)
            )
        return tuple(tangents)


def build_stateful_step(
    step_fn: Callable[[Any], Any],
    example_step_data: Any,
    parameter_states: Mapping[str, brainstate.ParamState],
    *,
    require_scalar_output: bool = True,
) -> FunctionalStep:
    """Functionalize one stateful step and locate its trainable state leaves.

    Parameters
    ----------
    step_fn : callable
        Stateful function accepting one per-step data PyTree and returning a
        scalar local loss. It should materialize trainable runtime parameters
        before advancing the Cell.
    example_step_data : PyTree
        Shape/dtype example used to trace ``step_fn``.
    parameter_states : mapping of str to brainstate.ParamState
        Stable parameter names and the original optimizer states.
    require_scalar_output : bool, default True
        Require the step output to be one scalar local loss. Set to ``False``
        for a trajectory-observation step whose output may be a PyTree.

    Returns
    -------
    FunctionalStep
        Pure state-value transition and stable state metadata.

    Raises
    ------
    TypeError
        If a supplied parameter is not a ``ParamState``.
    ValueError
        If no parameter is supplied, a parameter is not read by the step, or
        the local loss is not scalar.
    """
    if not parameter_states:
        raise ValueError("Forward sensitivity requires at least one parameter state.")
    for name, state in parameter_states.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter direction names must be non-empty strings.")
        if not isinstance(state, brainstate.ParamState):
            raise TypeError(f"Parameter {name!r} must be a brainstate.ParamState.")

    function = brainstate.transform.StatefulFunction(step_fn, return_only_write=False)
    function.make_jaxpr(example_step_data)
    state_trace = function.get_state_trace(example_step_data)
    traced_states = tuple(state_trace.states)
    names = tuple(parameter_states)
    states = tuple(parameter_states.values())
    indices = tuple(_state_index(traced_states, state, name=name) for name, state in parameter_states.items())

    output_shape, _ = function.get_out_shapes(example_step_data)
    if require_scalar_output and tuple(getattr(output_shape, "shape", ())) != ():
        raise ValueError(
            f"Forward-sensitivity step must return a scalar local loss, got shape "
            f"{getattr(output_shape, 'shape', ())!r}."
        )
    return FunctionalStep(function, state_trace, names, states, indices)


def build_parameter_coordinates(
    functional_step: FunctionalStep,
    state_values=None,
) -> ParameterCoordinates:
    """Describe every scalar coordinate of the step's optimizer roots.

    Parameters
    ----------
    functional_step : FunctionalStep
        Functionalized transition whose parameter states define the roots.
    state_values : PyTree, optional
        State values in trace order. Current values are read when omitted.

    Returns
    -------
    ParameterCoordinates
        Stable root shapes, coordinate slices and total direction count.

    Raises
    ------
    TypeError
        If a parameter leaf does not have an inexact dtype.
    ValueError
        If a root contains more than one array leaf.
    """
    state_values = functional_step.state_values() if state_values is None else state_values
    shapes = []
    slices = []
    offset = 0
    for name, state_index in zip(functional_step.parameter_names, functional_step.parameter_indices):
        leaf = _single_array_leaf(state_values[state_index], label=f"parameter {name}")
        if not np.issubdtype(np.dtype(leaf.dtype), np.inexact):
            raise TypeError(f"Parameter {name!r} must have an inexact dtype.")
        shape = tuple(leaf.shape)
        size = int(leaf.size)
        shapes.append(shape)
        slices.append(slice(offset, offset + size))
        offset += size
    return ParameterCoordinates(
        names=functional_step.parameter_names,
        state_indices=functional_step.parameter_indices,
        shapes=tuple(shapes),
        slices=tuple(slices),
        size=offset,
    )


def build_active_state_projection(
    functional_step: FunctionalStep,
    selections,
    state_values=None,
) -> ActiveStateProjection:
    """Resolve explicit active-state selections against a traced transition.

    Parameters
    ----------
    functional_step : FunctionalStep
        Functionalized transition whose state order is authoritative.
    selections : iterable of ActiveStateSelection
        Named state objects and flattened active coordinate indices.
    state_values : PyTree, optional
        State values in trace order. Current values are read when omitted.

    Returns
    -------
    ActiveStateProjection
        Compact gather/scatter metadata in the supplied selection order.

    Raises
    ------
    TypeError
        If a selection is not an ``ActiveStateSelection``.
    ValueError
        If names repeat, a state is not traced, or an index is invalid.
    """
    state_values = functional_step.state_values() if state_values is None else state_values
    traced_states = tuple(functional_step.state_trace.states)
    resolved = []
    seen_names = set()
    seen_coordinates = set()
    offset = 0
    for selection in selections:
        if not isinstance(selection, ActiveStateSelection):
            raise TypeError("Active state selections must be ActiveStateSelection instances.")
        if not selection.name or selection.name in seen_names:
            raise ValueError(f"Active state selection names must be unique and non-empty: {selection.name!r}.")
        seen_names.add(selection.name)
        state_index = _state_index(traced_states, selection.state, name=selection.name)
        leaf = _single_array_leaf(state_values[state_index], label=selection.name)
        indices = tuple(int(index) for index in selection.indices)
        if not indices:
            raise ValueError(f"Active state selection {selection.name!r} must not be empty.")
        if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= leaf.size:
            raise ValueError(
                f"Active indices for {selection.name!r} must be unique and within [0, {leaf.size}); got {indices!r}."
            )
        overlap = {(state_index, index) for index in indices}.intersection(seen_coordinates)
        if overlap:
            raise ValueError(f"Active state coordinates are selected more than once: {tuple(sorted(overlap))!r}.")
        seen_coordinates.update((state_index, index) for index in indices)
        resolved.append(
            _ResolvedActiveStateSelection(
                selection.name,
                state_index,
                indices,
                slice(offset, offset + len(indices)),
            )
        )
        offset += len(indices)
    if not resolved:
        raise ValueError("Active state projection requires at least one selection.")
    return ActiveStateProjection(tuple(resolved), offset)


def seed_scalar_parameter_directions(functional_step: FunctionalStep, state_values):
    """Create one coordinate tangent direction per scalar optimizer root.

    Parameters
    ----------
    functional_step : FunctionalStep
        Functionalized step whose parameter leaves define the directions.
    state_values : PyTree
        Initial values ordered like ``functional_step.state_trace.states``.

    Returns
    -------
    PyTree
        State tangents with a leading parameter-direction axis.

    Raises
    ------
    ValueError
        If a selected optimizer root is not scalar.

    Notes
    -----
    Vector-valued roots need explicit, unit-aware tangent directions. This
    experimental helper deliberately does not define a flattening contract.
    """
    coordinates = build_parameter_coordinates(functional_step, state_values)
    if any(shape != () for shape in coordinates.shapes):
        first = next(name for name, shape in zip(coordinates.names, coordinates.shapes) if shape != ())
        raise ValueError(f"Parameter {first!r} is not scalar; provide explicit tangent directions.")
    return coordinates.seed(state_values)


def seed_parameter_coordinate_directions(functional_step: FunctionalStep, state_values):
    """Create one full-state tangent direction per optimizer coordinate."""
    return build_parameter_coordinates(functional_step, state_values).seed(state_values)


def forward_sensitivity_step(
    functional_step: FunctionalStep,
    state_values,
    state_tangents,
    step_data,
):
    """Advance primal state and all forward parameter directions by one step.

    Parameters
    ----------
    functional_step : FunctionalStep
        Pure state-value transition.
    state_values : PyTree
        Current primal state values.
    state_tangents : PyTree
        Current state sensitivities with a leading direction axis.
    step_data : PyTree
        Current fixed input and target data.

    Returns
    -------
    tuple
        ``(next_state_values, next_state_tangents, local_loss,
        local_gradient_contribution)``.
    """

    def transition(values):
        return functional_step.call(values, step_data)

    (next_state_values, local_loss), linear_map = jax.linearize(transition, state_values)
    next_state_tangents, local_gradient = jax.vmap(linear_map)(state_tangents)
    return next_state_values, next_state_tangents, local_loss, local_gradient


def compact_forward_sensitivity_step(
    functional_step: FunctionalStep,
    projection: ActiveStateProjection,
    parameter_coordinates: ParameterCoordinates,
    state_values,
    active_state_tangents,
    step_data,
):
    """Advance one step while carrying only selected active-state sensitivity."""

    def transition(values):
        return functional_step.call(values, step_data)

    full_state_tangents = projection.embed_tangents(
        state_values,
        active_state_tangents,
        parameter_coordinates,
    )
    (next_state_values, local_loss), linear_map = jax.linearize(transition, state_values)
    next_full_state_tangents, local_gradient = jax.vmap(linear_map)(full_state_tangents)
    next_active_state_tangents = projection.extract_tangents(next_full_state_tangents)
    return next_state_values, next_active_state_tangents, local_loss, local_gradient


def forward_sensitivity_rollout(
    functional_step: FunctionalStep,
    initial_state_values,
    initial_state_tangents,
    step_data,
) -> ForwardSensitivityResult:
    """Scan a fixed-parameter rollout while accumulating exact prefix gradients.

    Parameters
    ----------
    functional_step : FunctionalStep
        Pure state-value transition.
    initial_state_values : PyTree
        State values at the start of the rollout.
    initial_state_tangents : PyTree
        Initial state sensitivity with a leading parameter-direction axis.
    step_data : PyTree
        Time-major fixed inputs and targets.

    Returns
    -------
    ForwardSensitivityResult
        Final carry, per-step losses and gradient contributions, and exact
        prefix gradients.
    """
    direction_count = _direction_count(initial_state_tangents)
    gradient_dtype = _parameter_dtype(functional_step, initial_state_values)
    initial_gradient = jnp.zeros((direction_count,), dtype=gradient_dtype)

    def scan_step(carry, data):
        state_values, state_tangents, gradient = carry
        next_values, next_tangents, loss, local_gradient = forward_sensitivity_step(
            functional_step,
            state_values,
            state_tangents,
            data,
        )
        gradient = gradient + local_gradient
        return (next_values, next_tangents, gradient), (loss, local_gradient, gradient)

    final_carry, outputs = brainstate.transform.scan(
        scan_step,
        (initial_state_values, initial_state_tangents, initial_gradient),
        step_data,
    )
    final_values, final_tangents, final_gradient = final_carry
    losses, local_gradients, prefix_gradients = outputs
    return ForwardSensitivityResult(
        final_values,
        final_tangents,
        final_gradient,
        losses,
        local_gradients,
        prefix_gradients,
    )


def compact_forward_sensitivity_rollout(
    functional_step: FunctionalStep,
    projection: ActiveStateProjection,
    parameter_coordinates: ParameterCoordinates,
    initial_state_values,
    initial_active_state_tangents,
    step_data,
) -> ForwardSensitivityResult:
    """Scan an exact rollout with a compact ``(parameter, active_state)`` carry."""
    expected = (parameter_coordinates.size, projection.size)
    if tuple(initial_active_state_tangents.shape) != expected:
        raise ValueError(
            f"Initial active-state tangents must have shape {expected!r}, got {initial_active_state_tangents.shape!r}."
        )
    gradient_dtype = _parameter_dtype(functional_step, initial_state_values)
    initial_gradient = jnp.zeros((parameter_coordinates.size,), dtype=gradient_dtype)

    def scan_step(carry, data):
        state_values, active_state_tangents, gradient = carry
        next_values, next_tangents, loss, local_gradient = compact_forward_sensitivity_step(
            functional_step,
            projection,
            parameter_coordinates,
            state_values,
            active_state_tangents,
            data,
        )
        gradient = gradient + local_gradient
        return (next_values, next_tangents, gradient), (loss, local_gradient, gradient)

    final_carry, outputs = brainstate.transform.scan(
        scan_step,
        (initial_state_values, initial_active_state_tangents, initial_gradient),
        step_data,
    )
    final_values, final_tangents, final_gradient = final_carry
    losses, local_gradients, prefix_gradients = outputs
    return ForwardSensitivityResult(
        final_values,
        final_tangents,
        final_gradient,
        losses,
        local_gradients,
        prefix_gradients,
    )


def initialize_forward_sensitivity(
    initializer_step: FunctionalStep,
    functional_step: FunctionalStep,
    initializer_data,
):
    """Differentiate reset/materialization and align its output to a rollout step.

    Parameters
    ----------
    initializer_step : FunctionalStep
        Functionalized initializer returning a scalar dummy loss. Its parameter
        names and ordering must match ``functional_step``.
    functional_step : FunctionalStep
        Rollout transition whose state ordering defines the returned values.
    initializer_data : PyTree
        Dynamic data accepted by the initializer, commonly ``None``.

    Returns
    -------
    tuple
        ``(initial_state_values, initial_state_tangents)`` aligned to
        ``functional_step.state_trace.states``.

    Raises
    ------
    ValueError
        If initializer and rollout parameter identities or ordering differ.
    """
    if initializer_step.parameter_names != functional_step.parameter_names:
        raise ValueError("Initializer and rollout step must use the same parameter names and ordering.")
    if any(
        initializer is not rollout
        for initializer, rollout in zip(initializer_step.parameter_states, functional_step.parameter_states)
    ):
        raise ValueError("Initializer and rollout step must reference the same ParamState objects.")

    initializer_values = initializer_step.state_values()
    initializer_tangents = seed_parameter_coordinate_directions(initializer_step, initializer_values)
    reset_values, reset_tangents, _dummy_loss, _dummy_gradient = forward_sensitivity_step(
        initializer_step,
        initializer_values,
        initializer_tangents,
        initializer_data,
    )
    return _align_state_carry(
        initializer_step,
        functional_step,
        reset_values,
        reset_tangents,
    )


def initialize_compact_forward_sensitivity(
    initializer_step: FunctionalStep,
    functional_step: FunctionalStep,
    projection: ActiveStateProjection,
    initializer_data,
):
    """Differentiate initialization and return projected active-state sensitivity."""
    initial_values, full_initial_tangents = initialize_forward_sensitivity(
        initializer_step,
        functional_step,
        initializer_data,
    )
    return initial_values, projection.extract_tangents(full_initial_tangents)


def bptt_reference_loss(functional_step: FunctionalStep, initial_state_values, step_data):
    """Return the summed local loss for reverse-mode reference differentiation."""

    def scan_step(state_values, data):
        next_state_values, local_loss = functional_step.call(state_values, data)
        return next_state_values, local_loss

    _, losses = brainstate.transform.scan(scan_step, initial_state_values, step_data)
    return jnp.sum(losses)


def initialized_bptt_reference_loss(
    initializer_step: FunctionalStep,
    functional_step: FunctionalStep,
    initializer_state_values,
    initializer_data,
    step_data,
):
    """Return a reverse-mode reference loss including parameter-dependent reset."""
    reset_values, _ = initializer_step.call(initializer_state_values, initializer_data)
    aligned_values = _align_state_values(initializer_step, functional_step, reset_values)
    return bptt_reference_loss(functional_step, aligned_values, step_data)


def select_parameter_derivatives(functional_step: FunctionalStep, derivative_state_values):
    """Select derivatives corresponding to the functional step's optimizer roots."""
    return {
        name: derivative_state_values[index]
        for name, index in zip(functional_step.parameter_names, functional_step.parameter_indices)
    }


def _state_index(traced_states, expected_state, *, name: str) -> int:
    matches = [index for index, state in enumerate(traced_states) if state is expected_state]
    if len(matches) != 1:
        raise ValueError(
            f"Parameter {name!r} must be read exactly once by the functionalized step; "
            f"found {len(matches)} matching state entries."
        )
    return matches[0]


def _zero_tangent_tree(values):
    return jax.tree.map(_zero_tangent_leaf, values)


def _broadcast_zero_tangents(values, direction_count: int):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (direction_count,) + value.shape),
        _zero_tangent_tree(values),
    )


def _zero_tangent_leaf(value):
    dtype = getattr(value, "dtype", None)
    if dtype is not None and not np.issubdtype(np.dtype(dtype), np.inexact):
        return jnp.zeros_like(value, dtype=jax.dtypes.float0)
    return jnp.zeros_like(value)


def _single_array_leaf(value, *, label: str):
    leaves = jax.tree.leaves(value)
    if len(leaves) != 1:
        raise ValueError(f"{label} must contain exactly one array leaf; got {len(leaves)}.")
    leaf = leaves[0]
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        raise TypeError(f"{label} must contain an array leaf.")
    return leaf


def _align_state_values(source_step, target_step, source_values):
    source_indices = {id(state): index for index, state in enumerate(source_step.state_trace.states)}
    target_current = target_step.state_values()
    return tuple(
        source_values[source_indices[id(state)]] if id(state) in source_indices else current
        for state, current in zip(target_step.state_trace.states, target_current)
    )


def _align_state_carry(source_step, target_step, source_values, source_tangents):
    source_indices = {id(state): index for index, state in enumerate(source_step.state_trace.states)}
    target_current = target_step.state_values()
    direction_count = _direction_count(source_tangents)
    aligned_values = []
    aligned_tangents = []
    for state, current in zip(target_step.state_trace.states, target_current):
        source_index = source_indices.get(id(state))
        if source_index is not None:
            aligned_values.append(source_values[source_index])
            aligned_tangents.append(source_tangents[source_index])
            continue
        zero = _zero_tangent_tree(current)
        aligned_values.append(current)
        aligned_tangents.append(
            jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (direction_count,) + leaf.shape),
                zero,
            )
        )
    return tuple(aligned_values), tuple(aligned_tangents)


def _parameter_dtype(functional_step: FunctionalStep, state_values):
    dtypes = []
    for index in functional_step.parameter_indices:
        leaves = jax.tree.leaves(state_values[index])
        if len(leaves) != 1:
            raise ValueError("Scalar parameter roots must contain exactly one array leaf.")
        dtype = getattr(leaves[0], "dtype", None)
        if dtype is None or not np.issubdtype(np.dtype(dtype), np.inexact):
            raise TypeError("Forward-sensitivity parameter roots must have an inexact dtype.")
        dtypes.append(dtype)
    return jnp.result_type(*dtypes)


def _direction_count(state_tangents) -> int:
    leaves = jax.tree.leaves(state_tangents)
    if not leaves:
        raise ValueError("State tangent PyTree must contain at least one array leaf.")
    counts = {int(leaf.shape[0]) for leaf in leaves}
    if len(counts) != 1:
        raise ValueError("Every state tangent leaf must use the same leading direction axis.")
    count = counts.pop()
    if count < 1:
        raise ValueError("Forward sensitivity requires at least one tangent direction.")
    return count
