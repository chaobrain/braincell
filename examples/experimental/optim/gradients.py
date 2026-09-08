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

"""Training-facing experimental BPTT and exact full-RTRL rollout engines.

The implementation builds on ``_forward_sensitivity`` and adds automatic
Cell trainable discovery, reset/materialization, common gradient results, and a
separately compiled diagnostic path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import brainstate
import jax
import jax.numpy as jnp

from examples.experimental.optim._forward_sensitivity import (
    FunctionalStep,
    ParameterCoordinates,
    _align_state_carry,
    build_parameter_coordinates,
    build_stateful_step,
    forward_sensitivity_step,
)

__all__ = [
    "FullRTRLDiagnostic",
    "RolloutGradientEngine",
    "RolloutGradientResult",
    "TrajectoryGradientEngine",
    "TrajectoryGradientResult",
    "build_rollout_value_and_grad",
    "build_trajectory_value_and_grad",
]


class RolloutGradientResult(NamedTuple):
    """Common result returned by BPTT and full RTRL.

    Attributes
    ----------
    losses : array-like
        Scalar local losses stacked along the rollout time axis.
    loss : array-like
        Sum of ``losses``.
    gradients : dict[str, array-like]
        Total gradient keyed like ``target.trainables.parameters().states()``.
    """

    losses: object
    loss: object
    gradients: dict[str, object]


class TrajectoryGradientResult(NamedTuple):
    """Loss and optimizer-compatible gradients for one trajectory objective."""

    loss: object
    gradients: dict[str, object]


class FullRTRLDiagnostic(NamedTuple):
    """Sampled decomposition of an exact full-RTRL rollout.

    ``sensitivity`` and ``learning_signal`` preserve the full state PyTree.
    Their leaves have leading sample and parameter-direction axes respectively:
    sensitivity leaves use ``(n_sample, n_parameter, ...)`` while learning
    signal leaves use ``(n_sample, ...)``.
    """

    at: object
    losses: object
    sensitivity: object
    learning_signal: object
    direct_gradients: object
    eligibility_gradients: object
    local_gradients: object
    prefix_gradients: object
    decomposition_residual: object
    loss: object
    gradients: dict[str, object]


class RolloutGradientEngine:
    """Differentiate a stateful fixed-shape rollout with BPTT or full RTRL.

    This class is experimental. It deliberately keeps the full state traced by
    :class:`brainstate.transform.StatefulFunction` in RTRL mode, making it the
    conservative correctness path for stateful mechanisms.

    Parameters
    ----------
    target : object
        Initialized target exposing ``trainables`` and ``reset_state()``.
    step : callable
        Callable accepting one time slice, advancing ``target`` exactly once,
        and returning a scalar additive local loss.
    method : {"bptt", "rtrl"}
        Rollout differentiation order.
    initializer : callable, optional
        Custom zero-argument reset hook. By default ``target.reset_state()`` is
        called. Trainable parameters are materialized before either hook.
    parameters : mapping, optional
        Stable-name mapping of :class:`brainstate.ParamState` objects. Defaults
        to ``target.trainables.parameters().states()``.
    """

    def __init__(
        self,
        target,
        *,
        step: Callable[[object], object],
        method: str = "rtrl",
        initializer: Callable[[], None] | None = None,
        parameters: Mapping[str, brainstate.ParamState] | None = None,
    ) -> None:
        if method not in {"bptt", "rtrl"}:
            raise ValueError(f"method must be 'bptt' or 'rtrl', got {method!r}.")
        if not callable(step):
            raise TypeError("step must be callable.")
        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be callable or None.")
        if parameters is None:
            try:
                parameters = target.trainables.parameters().states()
            except AttributeError as exc:
                raise TypeError("parameters=None requires target.trainables.parameters().states().") from exc
        self.target = target
        self.step = step
        self.method = method
        self.initializer = initializer
        self.parameter_states = _validate_parameters(parameters)
        self._initializer_step: FunctionalStep | None = None
        self._functional_step: FunctionalStep | None = None
        self._initializer_coordinates: ParameterCoordinates | None = None
        self._parameter_coordinates: ParameterCoordinates | None = None
        self._initializer_values: object | None = None

    @property
    def prepared(self) -> bool:
        """Whether the stateful transition has been traced."""
        return self._functional_step is not None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return optimizer root names in stable coordinate order."""
        return tuple(self.parameter_states)

    def prepare(self, example_step_data) -> "RolloutGradientEngine":
        """Trace the initializer and one rollout step.

        Repeated calls are idempotent. A prepared engine is tied to the traced
        target state objects and step-data structure.
        """
        if self.prepared:
            return self

        def initialize_and_zero(_):
            self.target.trainables.materialize()
            if self.initializer is None:
                self.target.reset_state()
            else:
                self.initializer()
            return jnp.asarray(0.0)

        def materialized_step(data):
            self.target.trainables.materialize()
            return self.step(data)

        self._initializer_step = build_stateful_step(
            initialize_and_zero,
            None,
            self.parameter_states,
        )
        self._functional_step = build_stateful_step(
            materialized_step,
            example_step_data,
            self.parameter_states,
        )
        self._parameter_coordinates = build_parameter_coordinates(self._functional_step)
        initializer_coordinates = build_parameter_coordinates(self._initializer_step)
        if (
            initializer_coordinates.names != self._parameter_coordinates.names
            or initializer_coordinates.shapes != self._parameter_coordinates.shapes
        ):
            raise ValueError("Initializer and rollout parameter coordinates must match.")
        self._initializer_coordinates = initializer_coordinates
        self._initializer_values = self._initializer_step.state_values()
        return self

    def __call__(self, step_data) -> RolloutGradientResult:
        """Return per-step losses, summed loss, and optimizer-compatible gradients."""
        self._ensure_prepared(step_data)
        roots = tuple(state.value for state in self.parameter_states.values())
        if self.method == "bptt":
            return self._bptt(roots, step_data)
        return self._rtrl(roots, step_data)

    def diagnose(self, step_data, *, at: Sequence[int] | None = None) -> FullRTRLDiagnostic:
        """Return sampled ``S/L/D`` decomposition from a separate RTRL path.

        Parameters
        ----------
        step_data : PyTree
            Time-major rollout inputs.
        at : sequence of int, optional
            Sorted unique zero-based step indices to inspect. ``None`` records
            every step through one scan, which is appropriate for short
            diagnostic rollouts but stores ``O(T * N_state * N_parameter)``
            sensitivity output.

        Returns
        -------
        FullRTRLDiagnostic
            Sampled full-state sensitivities, learning signals, and gradient
            decomposition, plus the final total loss and parameter gradient.
        """
        if self.method != "rtrl":
            raise ValueError("S/L diagnostics require method='rtrl'.")
        length = _time_axis_length(step_data)
        self._ensure_prepared(step_data)
        roots = tuple(state.value for state in self.parameter_states.values())
        if at is None:
            return self._rtrl_diagnostic_all(roots, step_data, length=length)
        sample_indices = _validate_sample_indices(at, length=length)
        return self._rtrl_diagnostic(roots, step_data, sample_indices)

    def _ensure_prepared(self, step_data) -> None:
        length = _time_axis_length(step_data)
        if length < 1:
            raise ValueError("step_data must contain at least one time step.")
        if not self.prepared:
            self.prepare(jax.tree.map(lambda leaf: leaf[0], step_data))

    def _initial_full_carry(self, roots):
        initializer_step, functional_step, _coordinates = self._parts()
        initializer_values = _replace_parameter_values(
            self._initializer_values,
            initializer_step.parameter_indices,
            roots,
        )
        initializer_tangents = self._initializer_coordinates.seed(initializer_values)
        reset_values, reset_tangents, _dummy_loss, _dummy_gradient = forward_sensitivity_step(
            initializer_step,
            initializer_values,
            initializer_tangents,
            None,
        )
        return _align_state_carry(
            initializer_step,
            functional_step,
            reset_values,
            reset_tangents,
        )

    def _initial_primal_values(self, roots):
        initializer_step, functional_step, _coordinates = self._parts()
        initializer_values = _replace_parameter_values(
            self._initializer_values,
            initializer_step.parameter_indices,
            roots,
        )
        reset_values, _dummy_loss = initializer_step.call(initializer_values, None)
        return _align_state_values(initializer_step, functional_step, reset_values)

    def _bptt(self, roots, step_data) -> RolloutGradientResult:
        functional_step = self._functional_step

        def objective(parameter_values, data):
            values = self._initial_primal_values(parameter_values)

            def scan_step(current_values, item):
                next_values, local_loss = functional_step.call(current_values, item)
                return next_values, local_loss

            _, losses = jax.lax.scan(scan_step, values, data)
            return jnp.sum(losses), losses

        (loss, losses), root_gradients = jax.value_and_grad(objective, argnums=0, has_aux=True)(roots, step_data)
        return RolloutGradientResult(
            losses=losses,
            loss=loss,
            gradients=dict(zip(self.parameter_names, root_gradients)),
        )

    def _rtrl(self, roots, step_data) -> RolloutGradientResult:
        functional_step = self._functional_step
        coordinates = self._parameter_coordinates
        values, tangents = self._initial_full_carry(roots)
        gradient = jnp.zeros((coordinates.size,), dtype=_coordinate_dtype(roots))

        def scan_step(carry, item):
            current_values, current_tangents, current_gradient = carry
            next_values, next_tangents, local_loss, local_gradient = forward_sensitivity_step(
                functional_step,
                current_values,
                current_tangents,
                item,
            )
            return (
                next_values,
                next_tangents,
                current_gradient + local_gradient,
            ), local_loss

        (_, _, gradient), losses = jax.lax.scan(
            scan_step,
            (values, tangents, gradient),
            step_data,
        )
        return RolloutGradientResult(
            losses=losses,
            loss=jnp.sum(losses),
            gradients=coordinates.unflatten(gradient),
        )

    def _rtrl_diagnostic(self, roots, step_data, sample_indices) -> FullRTRLDiagnostic:
        functional_step = self._functional_step
        coordinates = self._parameter_coordinates
        carry = self._initial_diagnostic_carry(roots)
        cursor = 0
        samples = []

        def ordinary_step(carry, item):
            next_carry, _sample = _diagnostic_step(
                functional_step,
                coordinates,
                carry,
                item,
            )
            return next_carry, None

        for sample_index in sample_indices:
            if sample_index > cursor:
                segment = jax.tree.map(lambda leaf, lo=cursor, hi=sample_index: leaf[lo:hi], step_data)
                carry, _ = jax.lax.scan(ordinary_step, carry, segment)
            item = jax.tree.map(lambda leaf, index=sample_index: leaf[index], step_data)
            carry, sample = _diagnostic_step(
                functional_step,
                coordinates,
                carry,
                item,
            )
            samples.append(sample)
            cursor = sample_index + 1

        length = _time_axis_length(step_data)
        if cursor < length:
            segment = jax.tree.map(lambda leaf, lo=cursor: leaf[lo:], step_data)
            carry, _ = jax.lax.scan(ordinary_step, carry, segment)
        stacked = tuple(jax.tree.map(lambda *leaves: jnp.stack(leaves), *items) for items in zip(*samples))
        return _build_diagnostic_result(
            coordinates,
            at=sample_indices,
            samples=stacked,
            final_carry=carry,
        )

    def _rtrl_diagnostic_all(self, roots, step_data, *, length: int) -> FullRTRLDiagnostic:
        functional_step = self._functional_step
        coordinates = self._parameter_coordinates

        def scan_step(carry, item):
            return _diagnostic_step(functional_step, coordinates, carry, item)

        final_carry, samples = jax.lax.scan(
            scan_step,
            self._initial_diagnostic_carry(roots),
            step_data,
        )
        return _build_diagnostic_result(
            coordinates,
            at=tuple(range(length)),
            samples=samples,
            final_carry=final_carry,
        )

    def _initial_diagnostic_carry(self, roots):
        coordinates = self._parameter_coordinates
        values, tangents = self._initial_full_carry(roots)
        dtype = _coordinate_dtype(roots)
        return (
            values,
            tangents,
            jnp.zeros((coordinates.size,), dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        )

    def _parts(self):
        if not self.prepared:
            raise RuntimeError("RolloutGradientEngine must be prepared before use.")
        return self._initializer_step, self._functional_step, self._parameter_coordinates


class TrajectoryGradientEngine:
    """Differentiate an arbitrary scalar loss over a complete observation trace.

    BPTT differentiates through one complete rollout. Exact RTRL uses two
    passes: a primal observation rollout followed by a replay that propagates
    full state sensitivity and contracts each observation tangent with the
    loss-derived learning signal.
    """

    def __init__(
        self,
        target,
        *,
        step: Callable[[object], object],
        loss: Callable[[object, object], object],
        method: str = "rtrl",
        initializer: Callable[[], None] | None = None,
        parameters: Mapping[str, brainstate.ParamState] | None = None,
    ) -> None:
        if method not in {"bptt", "rtrl"}:
            raise ValueError(f"method must be 'bptt' or 'rtrl', got {method!r}.")
        if not callable(step) or not callable(loss):
            raise TypeError("step and loss must be callable.")
        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be callable or None.")
        if parameters is None:
            try:
                parameters = target.trainables.parameters().states()
            except AttributeError as exc:
                raise TypeError("parameters=None requires target.trainables.parameters().states().") from exc
        self.target = target
        self.step = step
        self.loss = loss
        self.method = method
        self.initializer = initializer
        self.parameter_states = _validate_parameters(parameters)
        self._initializer_step: FunctionalStep | None = None
        self._functional_step: FunctionalStep | None = None
        self._initializer_coordinates: ParameterCoordinates | None = None
        self._parameter_coordinates: ParameterCoordinates | None = None
        self._initializer_values: object | None = None

    @property
    def prepared(self) -> bool:
        return self._functional_step is not None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameter_states)

    def prepare(self, example_step_data) -> "TrajectoryGradientEngine":
        """Trace reset and one observation-producing transition."""
        if self.prepared:
            return self

        def initialize_and_zero(_):
            self.target.trainables.materialize()
            if self.initializer is None:
                self.target.reset_state()
            else:
                self.initializer()
            return jnp.asarray(0.0)

        def materialized_step(data):
            self.target.trainables.materialize()
            return self.step(data)

        self._initializer_step = build_stateful_step(initialize_and_zero, None, self.parameter_states)
        self._functional_step = build_stateful_step(
            materialized_step,
            example_step_data,
            self.parameter_states,
            require_scalar_output=False,
        )
        self._parameter_coordinates = build_parameter_coordinates(self._functional_step)
        initializer_coordinates = build_parameter_coordinates(self._initializer_step)
        if (
            initializer_coordinates.names != self._parameter_coordinates.names
            or initializer_coordinates.shapes != self._parameter_coordinates.shapes
        ):
            raise ValueError("Initializer and rollout parameter coordinates must match.")
        self._initializer_coordinates = initializer_coordinates
        self._initializer_values = self._initializer_step.state_values()
        return self

    def __call__(self, step_data) -> TrajectoryGradientResult:
        """Return only the scalar trajectory loss and named total gradients."""
        self._ensure_prepared(step_data)
        roots = tuple(state.value for state in self.parameter_states.values())
        if self.method == "bptt":
            return self._bptt(roots, step_data)
        return self._rtrl_two_pass(roots, step_data)

    def _ensure_prepared(self, step_data) -> None:
        length = _time_axis_length(step_data)
        if length < 1:
            raise ValueError("step_data must contain at least one time step.")
        if not self.prepared:
            self.prepare(jax.tree.map(lambda leaf: leaf[0], step_data))

    def _initial_primal_values(self, roots):
        initializer_values = _replace_parameter_values(
            self._initializer_values,
            self._initializer_step.parameter_indices,
            roots,
        )
        reset_values, _dummy_loss = self._initializer_step.call(initializer_values, None)
        return _align_state_values(self._initializer_step, self._functional_step, reset_values)

    def _initial_full_carry(self, roots):
        initializer_values = _replace_parameter_values(
            self._initializer_values,
            self._initializer_step.parameter_indices,
            roots,
        )
        initializer_tangents = self._initializer_coordinates.seed(initializer_values)
        reset_values, reset_tangents, _dummy_loss, _dummy_gradient = forward_sensitivity_step(
            self._initializer_step,
            initializer_values,
            initializer_tangents,
            None,
        )
        return _align_state_carry(
            self._initializer_step,
            self._functional_step,
            reset_values,
            reset_tangents,
        )

    def _observation_rollout(self, roots, step_data):
        values = self._initial_primal_values(roots)

        def scan_step(current_values, item):
            next_values, observation = self._functional_step.call(current_values, item)
            return next_values, observation

        _, observations = jax.lax.scan(scan_step, values, step_data)
        return observations

    def _bptt(self, roots, step_data) -> TrajectoryGradientResult:
        def objective(parameter_values, data):
            observations = self._observation_rollout(parameter_values, data)
            return _require_scalar_loss(self.loss(observations, data))

        loss, root_gradients = jax.value_and_grad(objective, argnums=0)(roots, step_data)
        return TrajectoryGradientResult(loss, dict(zip(self.parameter_names, root_gradients)))

    def _rtrl_two_pass(self, roots, step_data) -> TrajectoryGradientResult:
        observations = jax.lax.stop_gradient(self._observation_rollout(roots, step_data))
        loss, learning_signals = jax.value_and_grad(lambda values: _require_scalar_loss(self.loss(values, step_data)))(
            observations
        )
        values, tangents = self._initial_full_carry(roots)
        gradient = jnp.zeros((self._parameter_coordinates.size,), dtype=_coordinate_dtype(roots))

        def scan_step(carry, inputs):
            current_values, current_tangents, current_gradient = carry
            item, learning_signal = inputs

            def transition(state_values):
                return self._functional_step.call(state_values, item)

            (next_values, _observation), linear_map = jax.linearize(transition, current_values)
            next_tangents, observation_tangents = jax.vmap(linear_map)(current_tangents)
            contribution = _contract_observation_tangents(observation_tangents, learning_signal)
            return (next_values, next_tangents, current_gradient + contribution), None

        (_, _, gradient), _ = jax.lax.scan(
            scan_step,
            (values, tangents, gradient),
            (step_data, learning_signals),
        )
        return TrajectoryGradientResult(loss, self._parameter_coordinates.unflatten(gradient))


def build_rollout_value_and_grad(
    target,
    *,
    step: Callable[[object], object],
    method: str = "rtrl",
    initializer: Callable[[], None] | None = None,
    parameters: Mapping[str, brainstate.ParamState] | None = None,
) -> RolloutGradientEngine:
    """Build an experimental rollout gradient engine.

    The returned engine prepares lazily from the first supplied ``step_data``.
    Call :meth:`RolloutGradientEngine.prepare` explicitly before placing it
    inside another JAX transform.
    """
    return RolloutGradientEngine(
        target,
        step=step,
        method=method,
        initializer=initializer,
        parameters=parameters,
    )


def build_trajectory_value_and_grad(
    target,
    *,
    step: Callable[[object], object],
    loss: Callable[[object, object], object],
    method: str = "rtrl",
    initializer: Callable[[], None] | None = None,
    parameters: Mapping[str, brainstate.ParamState] | None = None,
) -> TrajectoryGradientEngine:
    """Build an opt-in full-trace BPTT or two-pass exact-RTRL engine."""
    return TrajectoryGradientEngine(
        target,
        step=step,
        loss=loss,
        method=method,
        initializer=initializer,
        parameters=parameters,
    )


def _validate_parameters(parameters) -> dict[str, brainstate.ParamState]:
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a non-empty stable-name mapping.")
    checked = {}
    for name, state in parameters.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Parameter names must be non-empty strings.")
        if not isinstance(state, brainstate.ParamState):
            raise TypeError(f"Parameter {name!r} must be a brainstate.ParamState.")
        checked[name] = state
    return checked


def _replace_parameter_values(values, indices, roots):
    values = list(values)
    if len(indices) != len(roots):
        raise ValueError("Parameter values do not match the traced parameter roots.")
    for index, root in zip(indices, roots):
        values[index] = root
    return tuple(values)


def _align_state_values(source_step, target_step, source_values):
    source_indices = {id(state): index for index, state in enumerate(source_step.state_trace.states)}
    target_current = target_step.state_values()
    return tuple(
        source_values[source_indices[id(state)]] if id(state) in source_indices else current
        for state, current in zip(target_step.state_trace.states, target_current)
    )


def _time_axis_length(step_data) -> int:
    leaves = jax.tree.leaves(step_data)
    if not leaves:
        raise ValueError("step_data must contain at least one array leaf.")
    lengths = set()
    for leaf in leaves:
        shape = getattr(leaf, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError("Every step_data leaf must have a leading time axis.")
        lengths.add(int(shape[0]))
    if len(lengths) != 1:
        raise ValueError(f"step_data leaves must share one time-axis length, got {sorted(lengths)!r}.")
    return lengths.pop()


def _validate_sample_indices(at, *, length: int) -> tuple[int, ...]:
    if isinstance(at, (str, bytes)):
        raise TypeError("at must be a sequence of integer step indices.")
    indices = tuple(int(index) for index in at)
    if not indices:
        raise ValueError("at must contain at least one step index.")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("at must contain sorted unique step indices.")
    if indices[0] < 0 or indices[-1] >= length:
        raise IndexError(f"Diagnostic step indices must be within [0, {length}), got {indices!r}.")
    return indices


def _coordinate_dtype(roots):
    return jnp.result_type(*[jax.tree.leaves(root)[0].dtype for root in roots])


def _require_scalar_loss(value):
    if tuple(getattr(value, "shape", ())) != ():
        raise ValueError(f"Trajectory loss must return a scalar, got shape {getattr(value, 'shape', ())!r}.")
    return value


def _contract_observation_tangents(observation_tangents, learning_signal):
    tangent_leaves, tangent_tree = jax.tree.flatten(observation_tangents)
    signal_leaves, signal_tree = jax.tree.flatten(learning_signal)
    if tangent_tree != signal_tree:
        raise ValueError("Observation tangent and learning-signal trees differ.")
    direction_count = int(tangent_leaves[0].shape[0])
    total = None
    for tangent, signal in zip(tangent_leaves, signal_leaves):
        if tangent.dtype == jax.dtypes.float0 or signal.dtype == jax.dtypes.float0:
            continue
        contribution = jnp.sum((tangent * signal).reshape((direction_count, -1)), axis=1)
        total = contribution if total is None else total + contribution
    if total is None:
        return jnp.zeros((direction_count,), dtype=tangent_leaves[0].dtype)
    return total


def _diagnostic_step(functional_step, coordinates, carry, step_data):
    current_values, current_tangents, current_gradient, current_loss = carry
    next_values, next_tangents, local_loss, local_gradient = forward_sensitivity_step(
        functional_step,
        current_values,
        current_tangents,
        step_data,
    )
    learning_signal = _local_loss_gradient(functional_step, current_values, step_data)
    direct = _direct_parameter_gradient(coordinates, learning_signal)
    eligibility = _contract_sensitivity_learning_signal(
        current_tangents,
        learning_signal,
        excluded_indices=set(coordinates.state_indices),
    )
    prefix = current_gradient + local_gradient
    sample = (
        local_loss,
        current_tangents,
        learning_signal,
        direct,
        eligibility,
        local_gradient,
        prefix,
        local_gradient - eligibility - direct,
    )
    next_carry = (
        next_values,
        next_tangents,
        prefix,
        current_loss + local_loss,
    )
    return next_carry, sample


def _build_diagnostic_result(coordinates, *, at, samples, final_carry):
    _values, _tangents, final_gradient, final_loss = final_carry
    return FullRTRLDiagnostic(
        at=jnp.asarray(at, dtype=jnp.int32),
        losses=samples[0],
        sensitivity=samples[1],
        learning_signal=samples[2],
        direct_gradients=samples[3],
        eligibility_gradients=samples[4],
        local_gradients=samples[5],
        prefix_gradients=samples[6],
        decomposition_residual=samples[7],
        loss=final_loss,
        gradients=coordinates.unflatten(final_gradient),
    )


def _local_loss_gradient(functional_step, state_values, step_data):
    def loss_only(values):
        _next_values, local_loss = functional_step.call(values, step_data)
        return local_loss

    return jax.grad(loss_only, allow_int=True)(state_values)


def _direct_parameter_gradient(coordinates, learning_signal):
    leaves = []
    for state_index, shape in zip(coordinates.state_indices, coordinates.shapes):
        state_leaves = jax.tree.leaves(learning_signal[state_index])
        if len(state_leaves) != 1:
            raise ValueError("Diagnostic parameter gradients require one array leaf per root.")
        leaves.append(jnp.asarray(state_leaves[0]).reshape(shape).reshape(-1))
    return jnp.concatenate(leaves)


def _contract_sensitivity_learning_signal(tangents, learning_signal, *, excluded_indices):
    direction_count = _direction_count(tangents)
    total = None
    for state_index, (tangent_value, signal_value) in enumerate(zip(tangents, learning_signal)):
        if state_index in excluded_indices:
            continue
        tangent_leaves, tangent_tree = jax.tree.flatten(tangent_value)
        signal_leaves, signal_tree = jax.tree.flatten(signal_value)
        if tangent_tree != signal_tree:
            raise ValueError("Sensitivity and learning-signal trees differ.")
        for tangent, signal in zip(tangent_leaves, signal_leaves):
            if tangent.dtype == jax.dtypes.float0 or signal.dtype == jax.dtypes.float0:
                continue
            product = tangent * signal
            contribution = jnp.sum(product.reshape((direction_count, -1)), axis=1)
            total = contribution if total is None else total + contribution
    if total is None:
        first = jax.tree.leaves(tangents)[0]
        return jnp.zeros((direction_count,), dtype=first.dtype)
    return total


def _direction_count(tangents) -> int:
    counts = {int(leaf.shape[0]) for leaf in jax.tree.leaves(tangents)}
    if len(counts) != 1:
        raise ValueError("Every sensitivity leaf must have the same parameter-direction axis.")
    return counts.pop()
