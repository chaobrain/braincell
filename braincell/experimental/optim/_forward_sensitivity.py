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
``gradients`` for the higher-level training-facing engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from typing import Any, NamedTuple
import weakref

import brainstate
import jax
from jax.extend import source_info_util
import jax.numpy as jnp
import numpy as np

from braincell._misc import SOLVER_START_TAG


class OperatorInput(NamedTuple):
    """The target's axial operator, traced as an explicit input of a step.

    ``builder`` builds the operator leaves the solver reads from the states
    listed in ``builder_trace``; ``value_indices`` locates each of those
    states in the functional step's state values. The step was traced with
    ``(step_data, operator_leaves)`` as its data, so a rollout builds the
    operator once from its initial values and passes it to every step. The
    geometry and the optimizer roots do not change within a rollout (the
    engine keeps per-step building when the user's step writes them), so
    the operator built from the initial values is the one every step would
    build.
    """

    builder: brainstate.transform.StatefulFunction
    builder_trace: brainstate.StateTraceStack
    value_indices: tuple[int, ...]

    def build(self, step_state_values):
        """Return the operator leaves for ``step_state_values`` (functional-step order)."""
        return self.build_from(self.select(step_state_values))

    def select(self, step_state_values):
        """The builder's inputs: the values of the states it reads, in its own order."""
        return tuple(step_state_values[index] for index in self.value_indices)

    def build_from(self, builder_values):
        """Return the operator leaves for the builder's own input values."""
        _new_values, leaves = self.builder.jaxpr_call(builder_values, None)
        return leaves


class FunctionalStep(NamedTuple):
    """A stateful step exposed as an explicit state-value transition.

    ``materialized_states`` lists ``(index, label)`` for each state that
    holds trainable values written before the rollout rather than by the
    step (an engine's ``materialization_mode == "rollout"``): the step reads
    the parameters only through these states, so a compact carry must keep
    their sensitivity (see :func:`build_active_state_projection`).

    ``example_data`` holds the shapes and dtypes (``jax.ShapeDtypeStruct``)
    of the data the step was traced with, ``(step_data, operator_leaves)``
    for a step with an operator input; the compact-carry check traces the
    step's tangent map from it. ``state_labels`` names each traced state
    (``ion_channels.CalciumDetailed.Ci``, say) for that check's messages;
    the engines fill it, a step built by :func:`build_stateful_step` leaves
    it empty. ``rematerialized_states`` locates the states the step
    rewrites from the optimizer roots before anything reads them (an
    engine's ``materialization_mode == "step"``): their stored values do not
    reach the result, so the compact-carry check does not ask for them.
    """

    function: brainstate.transform.StatefulFunction
    state_trace: brainstate.StateTraceStack
    parameter_names: tuple[str, ...]
    parameter_states: tuple[brainstate.ParamState, ...]
    parameter_indices: tuple[int, ...]
    operator: OperatorInput | None = None
    materialized_states: tuple[tuple[int, str], ...] = ()
    example_data: Any = None
    state_labels: tuple[str, ...] = ()
    rematerialized_states: tuple[int, ...] = ()

    def state_values(self):
        """Return current values in the state order expected by ``jaxpr_call``."""
        return self.state_trace.get_state_values()

    def call(self, state_values, step_data):
        """Evaluate the pure state-value transition without mutating model states.

        A step with an operator input builds the operator from
        ``state_values`` here, once per call; rollouts build it once and use
        :meth:`call_with_operator`.
        """
        if self.operator is None:
            return self.function.jaxpr_call(state_values, step_data)
        return self.call_with_operator(state_values, step_data, self.operator.build(state_values))

    def call_with_operator(self, state_values, step_data, operator):
        """Evaluate the transition with prebuilt operator leaves (``operator`` is ``None`` without one)."""
        if self.operator is None:
            return self.function.jaxpr_call(state_values, step_data)
        return self.function.jaxpr_call(state_values, (step_data, operator))

    def build_operator(self, state_values):
        """The operator leaves for ``state_values``, or ``None`` for a step without an operator input."""
        return None if self.operator is None else self.operator.build(state_values)

    def build_operator_with_tangents(self, state_values, state_tangents):
        """The operator leaves and their tangents along every direction of ``state_tangents``.

        The rollout's operator depends on the parameters through the states
        it is built from; its tangent is computed once here and passed with
        the operator to every step, so forward sensitivity keeps that path.
        """
        if self.operator is None:
            return None, None
        # Linearize in the builder's inputs only: the other states (some
        # integer or boolean) do not reach the operator.
        operator, linear_map = jax.linearize(self.operator.build_from, self.operator.select(state_values))
        return operator, jax.vmap(linear_map)(self.operator.select(state_tangents))


class ForwardSensitivityResult(NamedTuple):
    """Outputs and final carry of an exact forward-sensitivity rollout."""

    final_state_values: Any
    final_state_tangents: Any
    final_gradient: Any
    losses: Any
    local_gradients: Any
    prefix_gradients: Any


@dataclass(frozen=True)
class StateSensitivityEntry:
    """Static description of one traced state and its tangent storage."""

    index: int
    state_type: str
    value_shapes: tuple[tuple[int, ...], ...]
    value_dtypes: tuple[str, ...]
    tangent_shapes: tuple[tuple[int, ...], ...]
    tangent_dtypes: tuple[str, ...]
    value_bytes: int
    tangent_bytes: int
    has_float0_tangent: bool
    is_parameter: bool


def inspect_state_sensitivity(functional_step: FunctionalStep, state_values=None, state_tangents=None):
    """Describe traced state and tangent leaves without executing a rollout.

    The returned entries distinguish optimizer roots from runtime and
    auxiliary states. ``tangent_bytes`` includes the leading parameter
    direction axis and therefore reports logical carry storage, not peak
    device memory. This is an inspection helper for exact and compact RTRL.
    """
    values = functional_step.state_values() if state_values is None else state_values
    tangents = None if state_tangents is None else state_tangents
    parameter_indices = set(functional_step.parameter_indices)
    entries = []
    for index, value in enumerate(values):
        value_leaves = tuple(jax.tree.leaves(value))
        tangent_leaves = () if tangents is None else tuple(jax.tree.leaves(tangents[index]))
        value_shapes = tuple(tuple(getattr(leaf, "shape", ())) for leaf in value_leaves)
        value_dtypes = tuple(str(getattr(leaf, "dtype", None)) for leaf in value_leaves)
        tangent_shapes = tuple(tuple(getattr(leaf, "shape", ())) for leaf in tangent_leaves)
        tangent_dtypes = tuple(str(getattr(leaf, "dtype", None)) for leaf in tangent_leaves)
        value_bytes = sum(int(getattr(leaf, "nbytes", 0)) for leaf in value_leaves)
        tangent_bytes = sum(int(getattr(leaf, "nbytes", 0)) for leaf in tangent_leaves)
        entries.append(
            StateSensitivityEntry(
                index=index,
                state_type=type(functional_step.state_trace.states[index]).__name__,
                value_shapes=value_shapes,
                value_dtypes=value_dtypes,
                tangent_shapes=tangent_shapes,
                tangent_dtypes=tangent_dtypes,
                value_bytes=value_bytes,
                tangent_bytes=tangent_bytes,
                has_float0_tangent=any("float0" in dtype for dtype in tangent_dtypes),
                is_parameter=index in parameter_indices,
            )
        )
    return tuple(entries)


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
    """Project full runtime tangents to a compact active-state matrix.

    ``truncated_indices`` locate the traced states whose sensitivity the
    caller drops on purpose (``truncated_states`` of
    :func:`build_active_state_projection`).
    """

    selections: tuple[_ResolvedActiveStateSelection, ...]
    size: int
    truncated_indices: tuple[int, ...] = ()

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
    return FunctionalStep(function, state_trace, names, states, indices, example_data=_abstract_tree(example_step_data))


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
    *,
    truncated_states=(),
) -> ActiveStateProjection:
    """Resolve explicit active-state selections against a traced transition.

    A compact carry gives every coordinate outside the selections a zero
    tangent at the start of each step, so its result is exact only when no
    such coordinate has a sensitivity that reaches a selected state or the
    step's output. This function checks that from the step's tangent map
    (traced once per step and cached) and refuses selections that fail it.

    Parameters
    ----------
    functional_step : FunctionalStep
        Functionalized transition whose state order is authoritative.
    selections : iterable of ActiveStateSelection
        Named state objects and flattened active coordinate indices.
    state_values : PyTree, optional
        State values in trace order. Current values are read when omitted.
    truncated_states : iterable of brainstate.State, optional
        Traced states whose sensitivity the caller drops on purpose (a
        truncated, approximate recurrence): the check does not refuse
        leaving them out, wholly or partly. The materialized-state check
        still applies to them.

    Returns
    -------
    ActiveStateProjection
        Compact gather/scatter metadata in the supplied selection order.

    Raises
    ------
    TypeError
        If a selection is not an ``ActiveStateSelection``, or a truncated
        state is not a ``brainstate.State``.
    ValueError
        If names repeat, a state is not traced, or an index is invalid. Also
        if the selections leave out a coordinate of a state in
        ``functional_step.materialized_states``: the step reads the
        trainable parameters only through those states, so a compact carry
        without them would drop the parameters' sensitivity and return a
        zero gradient for them. Also if they leave out, wholly or partly, a
        state whose sensitivity can be non-zero (the step writes it from
        values that depend on the parameters) and that feeds a selected
        state or the step's output, an unselected calcium concentration,
        say: its dropped sensitivity would silently change the gradient.
        States tagged ``SOLVER_START_TAG`` (a Cell's point-tree voltages,
        which the step reads only as its solver's start value) are not
        refused.
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
    truncated = set()
    for state in truncated_states:
        if not isinstance(state, brainstate.State):
            raise TypeError("Truncated states must be brainstate.State instances.")
        matches = [index for index, traced in enumerate(traced_states) if traced is state]
        if len(matches) != 1:
            raise ValueError(f"A truncated state must be traced once by the functional step; found {len(matches)}.")
        truncated.add(matches[0])
    projection = ActiveStateProjection(tuple(resolved), offset, tuple(sorted(truncated)))
    _check_active_states(functional_step, projection, state_values)
    return projection


def _check_active_states(functional_step: FunctionalStep, projection, state_values, step_data=None) -> None:
    """Raise unless a compact carry with ``projection`` keeps every sensitivity its result depends on.

    Two checks: the states materialized before the rollout
    (:func:`_check_materialized_states_active`), then the states the step
    writes (:func:`_check_dropped_sensitivity`). ``step_data`` is one step's
    data, read only for its shape when the step records no ``example_data``.
    """
    _check_materialized_states_active(functional_step, projection, state_values)
    _check_dropped_sensitivity(functional_step, projection, state_values, step_data)


def _check_materialized_states_active(functional_step: FunctionalStep, projection, state_values) -> None:
    """Raise unless ``projection`` keeps every coordinate of the step's materialized states.

    The full sensitivity of such a state is its initial one, set by the
    initializer from the optimizer roots; the step itself never writes it.
    A compact carry gives every unselected state a zero tangent at each
    step, which for these states silently removes the parameters from the
    gradient.
    """
    if not functional_step.materialized_states:
        return
    selected: dict[int, set[int]] = {}
    for selection in projection.selections:
        selected.setdefault(selection.state_index, set()).update(selection.indices)
    missing = []
    for index, label in functional_step.materialized_states:
        size = _single_array_leaf(state_values[index], label=label).size
        count = len(selected.get(index, ()))
        if count < size:
            missing.append(f"{label} (state_trace.states[{index}], {count} of {size} coordinates selected)")
    if missing:
        raise ValueError(
            f"The active states leave out {', '.join(missing)}. The trainable parameters "
            f"{functional_step.parameter_names!r} reach the step only through these states, which hold "
            "values materialized before the rollout, so the compact gradient would omit them (zero when "
            "nothing else depends on the parameters). Select every coordinate of each such state, or call "
            "the target's trainables.materialize() inside the step so that the step reads the optimizer roots."
        )


def _check_dropped_sensitivity(functional_step: FunctionalStep, projection, state_values, step_data=None) -> None:
    """Raise if the projection drops a sensitivity that reaches a selected state or the step's output.

    A compact carry starts each step with a zero tangent for every
    coordinate it does not select. That is exact for a dropped state whose
    true tangent is zero throughout (it never depends on the parameters),
    or whose tangent never reaches a selected state or the step's output.
    The check works per traced state from the step's tangent dependencies
    (:func:`_step_tangent_dependencies`), a structural over-approximation,
    so it may refuse a selection whose dropped part is numerically zero but
    never passes one whose dropped part is structurally present:

    - a state's tangent can be non-zero if it is an optimizer root, a
      materialized state, or the step writes it from a state that can be
      non-zero (or from the operator input, whose tangent carries the
      parameters' geometry sensitivity);
    - a dropped state matters if its tangent can be non-zero and it feeds a
      selected state's new value, the step's output, or another dropped
      state that matters.

    Optimizer roots are seeded at every step, so they are never dropped.
    The caller's ``truncated_indices`` and states tagged
    ``SOLVER_START_TAG`` are not reported, and dependencies through them are
    not followed.
    """
    data = functional_step.example_data
    if data is None:
        if step_data is None:
            # A hand-built FunctionalStep without example data: the compact
            # entries check it with their own data.
            return
        data = _abstract_tree(step_data)
        if functional_step.operator is not None:
            data = (data, jax.eval_shape(functional_step.build_operator, state_values))
    dependencies = _step_tangent_dependencies(functional_step, state_values, data)
    state_bits, new_state_deps = dependencies.state_bits, dependencies.new_state_deps
    count = len(state_bits)
    sizes = [sum(int(np.size(leaf)) for leaf in jax.tree.leaves(value)) for value in state_values]
    selected: dict[int, set[int]] = {}
    for selection in projection.selections:
        selected.setdefault(selection.state_index, set()).update(selection.indices)
    roots = set(functional_step.parameter_indices)
    exempt = set(projection.truncated_indices) | set(functional_step.rematerialized_states)
    exempt.update(
        index
        for index, state in enumerate(functional_step.state_trace.states)
        if SOLVER_START_TAG in (getattr(state, "tag", None) or ())
    )

    # States whose tangent can be non-zero.
    sources = roots | {index for index, _label in functional_step.materialized_states}
    live_bits = dependencies.operator_bits
    for index in sources:
        live_bits |= state_bits[index]
    live = set(sources)
    changed = True
    while changed:
        changed = False
        for index in range(count):
            if index not in live and state_bits[index] and new_state_deps[index] & live_bits:
                live.add(index)
                live_bits |= state_bits[index]
                changed = True

    # Dropped states that matter: the tangent can be non-zero and reaches a
    # selected state, the output, or another dropped state that matters.
    dropped = {
        index
        for index in live
        if state_bits[index]
        and index not in roots
        and index not in exempt
        and len(selected.get(index, ())) < sizes[index]
    }
    target_bits = dependencies.output_deps
    for index in selected:
        target_bits |= new_state_deps[index]
    flagged = set()
    changed = True
    while changed:
        changed = False
        for index in sorted(dropped - flagged):
            if state_bits[index] & target_bits:
                flagged.add(index)
                target_bits |= new_state_deps[index]
                changed = True
    if not flagged:
        return
    labels = functional_step.state_labels
    missing = []
    for index in sorted(flagged):
        label = labels[index] if index < len(labels) and labels[index] else ""
        if not label:
            label = _state_origin(functional_step.state_trace.states[index])
        missing.append(
            f"{label} (state_trace.states[{index}], {len(selected.get(index, ()))} of {sizes[index]} "
            "coordinates selected)"
        )
    raise ValueError(
        f"The active states leave out {', '.join(missing)}. The step writes these states from values that "
        f"depend on the trainable parameters {functional_step.parameter_names!r}, and they feed a selected "
        "state or the step's output, so the compact gradient would silently omit part of it. Select every "
        "coordinate of each such state, or pass the states as truncated_states to drop their sensitivity "
        "on purpose."
    )


def _state_origin(state) -> str:
    """``ShortTermState created at <file>:<line> (<function>)``, for a state without a label."""
    try:
        origin = source_info_util.summarize(state.source_info)
    except Exception:  # pragma: no cover - a State without creation info
        return type(state).__name__
    return f"{type(state).__name__} created at {origin}"


class _StepDependencies(NamedTuple):
    """Structural tangent dependencies of one functional step.

    Each inexact leaf of the state values (in trace order) and then of the
    operator input has one bit. ``state_bits[i]`` holds state ``i``'s bits,
    ``operator_bits`` the operator's; ``new_state_deps[i]`` holds the input
    bits state ``i``'s new value depends on, ``output_deps`` those of the
    step's output.
    """

    state_bits: tuple[int, ...]
    operator_bits: int
    new_state_deps: tuple[int, ...]
    output_deps: int


#: Per StatefulFunction, the dependencies for each abstract signature of its
#: state values and data. The trace costs about as much as tracing the step
#: once; the compact entries check again at every call.
_DEPENDENCY_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _step_tangent_dependencies(functional_step: FunctionalStep, state_values, data) -> _StepDependencies:
    """Trace the step's tangent map on abstract values and return its structural dependencies.

    ``data`` holds the shapes the step's function takes: ``(step_data,
    operator_leaves)`` with an operator input. The tangent map is the
    JAX-visible linearization (``jax.jvp`` of the step), so derivatives JAX
    knows are zero (comparisons, ``stop_gradient``, integer arithmetic) add
    no dependency.
    """
    leaves, treedef = jax.tree.flatten(state_values)
    abstract = [jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)) for leaf in leaves]
    data_leaves, data_def = jax.tree.flatten(data)
    key = (
        treedef,
        tuple((value.shape, str(value.dtype)) for value in abstract),
        data_def,
        tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in data_leaves),
    )
    per_function = _DEPENDENCY_CACHE.setdefault(functional_step.function, {})
    cached = per_function.get(key)
    if cached is not None:
        return cached

    has_operator = functional_step.operator is not None
    if has_operator:
        step_data, operator = data
        operator_abstract, operator_def = jax.tree.flatten(operator)
    else:
        step_data, operator_abstract, operator_def = data, [], None
    inexact = [index for index, value in enumerate(abstract) if jnp.issubdtype(value.dtype, jnp.inexact)]
    operator_inexact = [
        index for index, value in enumerate(operator_abstract) if jnp.issubdtype(value.dtype, jnp.inexact)
    ]

    def full_tangents(values, positions, given):
        tangents = [np.zeros(value.shape, dtype=jax.dtypes.float0) for value in values]
        for position, tangent in zip(positions, given):
            tangents[position] = tangent
        return tangents

    def tangent_map(primals, tangents, operator_primals, operator_tangents, step_data):
        def transition(state_leaves, operator_leaves):
            values = treedef.unflatten(state_leaves)
            data = (step_data, operator_def.unflatten(operator_leaves)) if has_operator else step_data
            new_values, output = functional_step.function.jaxpr_call(values, data)
            return jax.tree.leaves(new_values), jax.tree.leaves(output)

        _primal_out, tangent_out = jax.jvp(
            transition,
            (list(primals), list(operator_primals)),
            (full_tangents(abstract, inexact, tangents), full_tangents(operator_abstract, operator_inexact, operator_tangents)),
        )
        return tangent_out

    closed = jax.make_jaxpr(tangent_map)(
        abstract,
        [abstract[index] for index in inexact],
        operator_abstract,
        [operator_abstract[index] for index in operator_inexact],
        step_data,
    )
    jaxpr = closed.jaxpr
    n_primal, n_tangent, n_operator = len(abstract), len(inexact), len(operator_abstract)
    in_deps = [0] * len(jaxpr.invars)
    for bit in range(n_tangent):
        in_deps[n_primal + bit] = 1 << bit
    for bit in range(len(operator_inexact)):
        in_deps[n_primal + n_tangent + n_operator + bit] = 1 << (n_tangent + bit)
    out_deps = _jaxpr_dependencies(jaxpr, in_deps)
    new_leaf_deps, output_leaf_deps = out_deps[: len(leaves)], out_deps[len(leaves) :]

    leaf_counts = [len(jax.tree.leaves(value)) for value in state_values]
    leaf_state = [index for index, leaf_count in enumerate(leaf_counts) for _ in range(leaf_count)]
    state_bits = [0] * len(leaf_counts)
    for bit, leaf in enumerate(inexact):
        state_bits[leaf_state[leaf]] |= 1 << bit
    new_state_deps = [0] * len(leaf_counts)
    for leaf, dependency in enumerate(new_leaf_deps):
        new_state_deps[leaf_state[leaf]] |= dependency
    output_deps = 0
    for dependency in output_leaf_deps:
        output_deps |= dependency
    result = _StepDependencies(
        tuple(state_bits),
        ((1 << len(operator_inexact)) - 1) << n_tangent,
        tuple(new_state_deps),
        output_deps,
    )
    per_function[key] = result
    return result


def _jaxpr_dependencies(jaxpr, in_deps):
    """Propagate input dependency bitmasks through ``jaxpr``; return one bitmask per output."""
    env = {}

    def read(var):
        # Literals carry no dependency.
        return env.get(var, 0) if not hasattr(var, "val") else 0

    for var, dependency in zip(jaxpr.invars, in_deps):
        env[var] = dependency
    for equation in jaxpr.eqns:
        outputs = _equation_dependencies(equation, [read(var) for var in equation.invars])
        for var, dependency in zip(equation.outvars, outputs):
            env[var] = dependency
    return [read(var) for var in jaxpr.outvars]


def _open_jaxpr(value):
    return value.jaxpr if hasattr(value, "consts") else value


def _equation_dependencies(equation, inputs):
    """Dependencies of one equation's outputs on its inputs' bitmasks.

    Call-like primitives recurse into their jaxpr; ``scan`` and ``while``
    iterate their carry to a fixed point; ``cond`` joins its branches. Any
    other primitive, and any of these whose parameters do not have the
    expected layout, makes every output depend on every input: an
    over-approximation, which can refuse a selection but never pass a wrong
    one.
    """
    count = len(equation.outvars)
    if not any(inputs):
        return [0] * count
    name = equation.primitive.name
    params = equation.params
    union = 0
    for dependency in inputs:
        union |= dependency
    if name == "scan":
        body = _open_jaxpr(params["jaxpr"])
        layout = _scan_layout(params)
        if layout is not None and len(body.invars) == len(inputs) and len(body.outvars) == count:
            n_consts, n_carry = layout
            consts, carry, xs = (
                inputs[:n_consts],
                list(inputs[n_consts : n_consts + n_carry]),
                inputs[n_consts + n_carry :],
            )
            while True:
                outputs = _jaxpr_dependencies(body, consts + carry + xs)
                joined = [old | new for old, new in zip(carry, outputs[:n_carry])]
                if joined == carry:
                    return joined + outputs[n_carry:]
                carry = joined
        return [union] * count
    if name == "while":
        body = _open_jaxpr(params["body_jaxpr"])
        n_cond, n_body = params.get("cond_nconsts"), params.get("body_nconsts")
        if n_cond is not None and n_body is not None and len(body.invars) == len(inputs) - n_cond:
            consts, carry = inputs[n_cond : n_cond + n_body], list(inputs[n_cond + n_body :])
            if len(body.outvars) == len(carry) == count:
                while True:
                    outputs = _jaxpr_dependencies(body, consts + carry)
                    joined = [old | new for old, new in zip(carry, outputs)]
                    if joined == carry:
                        return joined
                    carry = joined
        return [union] * count
    if name == "cond":
        # The branch index is an integer: no tangent flows from it.
        joined = [0] * count
        for branch in params["branches"]:
            body = _open_jaxpr(branch)
            if len(body.invars) != len(inputs) - 1 or len(body.outvars) != count:
                return [union] * count
            joined = [old | new for old, new in zip(joined, _jaxpr_dependencies(body, inputs[1:]))]
        return joined
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        inner = params.get(key)
        if inner is not None and (hasattr(inner, "eqns") or hasattr(inner, "consts")):
            body = _open_jaxpr(inner)
            if len(body.invars) == len(inputs) and len(body.outvars) == count:
                return _jaxpr_dependencies(body, inputs)
            break
    return [union] * count


def _scan_layout(params):
    """``(number of consts, number of carries)`` of a ``scan``, or ``None`` if the parameters are unknown."""
    if "num_consts" in params and "num_carry" in params:
        return int(params["num_consts"]), int(params["num_carry"])
    layout = params.get("ft_in")
    # jax >= 0.12 records (consts, carry, xs) as a flat tree.
    parts = getattr(layout, "elts", layout)
    try:
        return len(parts[0]), len(parts[1])
    except (TypeError, IndexError):
        return None


def _abstract_tree(tree):
    """``jax.ShapeDtypeStruct`` leaves with the shapes, dtypes and weak types of ``tree``'s leaves."""

    def abstract(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return leaf
        aval = jax.typeof(leaf)
        return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)

    return jax.tree.map(abstract, tree)


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
    operator=None,
    operator_tangents=None,
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

    if operator is not None:
        return _forward_sensitivity_step_with_operator(
            functional_step, state_values, state_tangents, step_data, operator, operator_tangents
        )

    def transition(values):
        return functional_step.call(values, step_data)

    if os.environ.get("BRAINCELL_RTRL_JVP_MODE", "linearize") == "direct":

        def one_direction(tangent):
            return jax.jvp(transition, (state_values,), (tangent,))

        (next_state_values, local_loss), (next_state_tangents, local_gradient) = jax.vmap(
            one_direction,
            out_axes=((None, None), (0, 0)),
        )(state_tangents)
    else:
        (next_state_values, local_loss), linear_map = jax.linearize(transition, state_values)
        next_state_tangents, local_gradient = jax.vmap(linear_map)(state_tangents)
    return next_state_values, next_state_tangents, local_loss, local_gradient


def _forward_sensitivity_step_with_operator(
    functional_step, state_values, state_tangents, step_data, operator, operator_tangents
):
    """:func:`forward_sensitivity_step` with prebuilt operator leaves and their tangents."""

    def transition(values, op):
        return functional_step.call_with_operator(values, step_data, op)

    if os.environ.get("BRAINCELL_RTRL_JVP_MODE", "linearize") == "direct":

        def one_direction(tangent, operator_tangent):
            return jax.jvp(transition, (state_values, operator), (tangent, operator_tangent))

        (next_state_values, local_loss), (next_state_tangents, local_gradient) = jax.vmap(
            one_direction,
            out_axes=((None, None), (0, 0)),
        )(state_tangents, operator_tangents)
    else:
        (next_state_values, local_loss), linear_map = jax.linearize(transition, state_values, operator)
        next_state_tangents, local_gradient = jax.vmap(linear_map)(state_tangents, operator_tangents)
    return next_state_values, next_state_tangents, local_loss, local_gradient


def compact_forward_sensitivity_step(
    functional_step: FunctionalStep,
    projection: ActiveStateProjection,
    parameter_coordinates: ParameterCoordinates,
    state_values,
    active_state_tangents,
    step_data,
    operator=None,
    operator_tangents=None,
):
    """Advance one step while carrying only selected active-state sensitivity.

    Raises
    ------
    ValueError
        As :func:`build_active_state_projection`, if ``projection`` leaves out
        a coordinate of a state in ``functional_step.materialized_states``.
    """
    _check_active_states(functional_step, projection, state_values, step_data)
    full_state_tangents = projection.embed_tangents(
        state_values,
        active_state_tangents,
        parameter_coordinates,
    )
    if operator is not None:
        next_state_values, next_full_state_tangents, local_loss, local_gradient = (
            _forward_sensitivity_step_with_operator(
                functional_step, state_values, full_state_tangents, step_data, operator, operator_tangents
            )
        )
        next_active_state_tangents = projection.extract_tangents(next_full_state_tangents)
        return next_state_values, next_active_state_tangents, local_loss, local_gradient

    def transition(values):
        return functional_step.call(values, step_data)

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
    operator, operator_tangents = functional_step.build_operator_with_tangents(
        initial_state_values, initial_state_tangents
    )

    def scan_step(carry, data):
        state_values, state_tangents, gradient = carry
        next_values, next_tangents, loss, local_gradient = forward_sensitivity_step(
            functional_step,
            state_values,
            state_tangents,
            data,
            operator,
            operator_tangents,
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
    """Scan an exact rollout with a compact ``(parameter, active_state)`` carry.

    The result is exact when the active states carry every sensitivity the
    loss depends on: the selection must include each state whose tangent is
    not zero, since every other state starts each step with a zero tangent.
    States materialized before the rollout are checked (below); a state the
    step itself writes and the selection leaves out (an unselected calcium
    concentration, say) is not detected and silently drops its part of the
    gradient (OPT-18).

    Raises
    ------
    ValueError
        If ``initial_active_state_tangents`` has the wrong shape, or, as
        :func:`build_active_state_projection`, if ``projection`` leaves out a
        coordinate of a state in ``functional_step.materialized_states``.
    """
    expected = (parameter_coordinates.size, projection.size)
    if tuple(initial_active_state_tangents.shape) != expected:
        raise ValueError(
            f"Initial active-state tangents must have shape {expected!r}, got {initial_active_state_tangents.shape!r}."
        )
    _check_active_states(
        functional_step,
        projection,
        initial_state_values,
        jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf)[1:], jnp.result_type(leaf)), step_data),
    )
    gradient_dtype = _parameter_dtype(functional_step, initial_state_values)
    initial_gradient = jnp.zeros((parameter_coordinates.size,), dtype=gradient_dtype)
    operator, operator_tangents = functional_step.build_operator_with_tangents(
        initial_state_values,
        projection.embed_tangents(initial_state_values, initial_active_state_tangents, parameter_coordinates),
    )

    def scan_step(carry, data):
        state_values, active_state_tangents, gradient = carry
        next_values, next_tangents, loss, local_gradient = compact_forward_sensitivity_step(
            functional_step,
            projection,
            parameter_coordinates,
            state_values,
            active_state_tangents,
            data,
            operator,
            operator_tangents,
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

    operator = functional_step.build_operator(initial_state_values)

    def scan_step(state_values, data):
        next_state_values, local_loss = functional_step.call_with_operator(state_values, data, operator)
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
