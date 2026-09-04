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

"""Small reference reduction models exercising the public runtime contract."""

from __future__ import annotations

from abc import abstractmethod

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.mech import ScalarEventInput
from braincell.reduction.core import ReductionContext, ReductionInputs, ReductionModel, ReductionOutput

__all__ = [
    "EventAccumulatorReduction",
    "PayloadAccumulatorReduction",
    "SynapticKernelAccumulatorReduction",
]


class _PopulationParameters:
    """Store scalar defaults plus sparse declaration-time population overrides."""

    def __init__(self, **defaults) -> None:
        self.defaults = dict(defaults)
        self.overrides = {name: {} for name in defaults}

    def set(self, population_indices, **parameters) -> None:
        unknown = set(parameters).difference(self.defaults)
        if unknown:
            raise KeyError(f"Unknown reduction parameters: {tuple(sorted(unknown))!r}.")
        for name, value in parameters.items():
            for index, item in zip(population_indices, _selected_values(value, len(population_indices), name=name)):
                self.overrides[name][int(index)] = item

    def get(self, field: str, population_indices):
        if field not in self.defaults:
            raise KeyError(f"Unknown reduction parameter {field!r}.")
        values = [self.overrides[field].get(int(index), self.defaults[field]) for index in population_indices]
        default = self.defaults[field]
        if isinstance(default, u.Quantity):
            unit = u.get_unit(default)
            return u.Quantity(np.asarray([value.to_decimal(unit) for value in values]), unit)
        return np.asarray(values)

    def materialize(self, field: str, population_size: int):
        values = self.get(field, tuple(range(population_size)))
        if isinstance(values, u.Quantity):
            return u.Quantity(jnp.asarray(u.get_mantissa(values)), u.get_unit(values))
        return jnp.asarray(values)


class _AccumulatorReduction(ReductionModel):
    """Share decay, threshold, reset, and population-parameter behavior."""

    def __init__(self, *, alpha: float, threshold) -> None:
        self._validate_alpha(alpha)
        self._validate_threshold(threshold)
        self._parameters = _PopulationParameters(alpha=float(alpha), threshold=threshold)
        self._state = None
        self._context = None

    def init_state(self, context: ReductionContext, batch_size=None) -> ReductionOutput:
        self._prepare_context(context)
        self._context = context
        shape = _runtime_shape(context.pop_size, batch_size)
        self._state = brainstate.ShortTermState(self._initial_value(shape))
        self._materialize_parameters()
        return self._zero_output()

    def update(self, inputs: ReductionInputs) -> ReductionOutput:
        self._require_initialized()
        candidate = self._alpha * self._state.value + self._drive(inputs)
        event = candidate > self._threshold
        self._state.value = u.math.where(event, u.math.zeros_like(candidate), candidate)
        return ReductionOutput(values={"value": candidate}, event=event.astype(jnp.int32))

    def reset_state(self, batch_size=None) -> ReductionOutput:
        self._require_initialized()
        shape = _runtime_shape(self._context.pop_size, batch_size)
        self._state.value = self._initial_value(shape)
        self._materialize_parameters()
        return self._zero_output()

    def reset(self) -> None:
        self._state = None
        self._context = None
        self._clear_context()
        for name in ("_alpha", "_threshold"):
            if hasattr(self, name):
                delattr(self, name)

    def get(self, field: str, population_indices: tuple[int, ...]):
        return self._parameters.get(field, population_indices)

    def set(self, population_indices: tuple[int, ...], **parameters) -> None:
        if "alpha" in parameters:
            for value in _selected_values(parameters["alpha"], len(population_indices), name="alpha"):
                self._validate_alpha(value)
        if "threshold" in parameters:
            for value in _selected_values(parameters["threshold"], len(population_indices), name="threshold"):
                self._validate_threshold(value)
        self._parameters.set(population_indices, **parameters)

    def _materialize_parameters(self) -> None:
        population_size = self._context.population_size
        self._alpha = self._parameters.materialize("alpha", population_size)
        self._threshold = self._parameters.materialize("threshold", population_size)

    def _zero_output(self) -> ReductionOutput:
        zero = u.math.zeros_like(self._state.value)
        event = jnp.zeros_like(u.get_mantissa(zero), dtype=jnp.int32)
        return ReductionOutput(values={"value": zero}, event=event)

    def _require_initialized(self) -> None:
        if self._state is None or self._context is None:
            raise RuntimeError(f"{type(self).__name__} requires init_state() first.")

    def _prepare_context(self, context: ReductionContext) -> None:
        del context

    def _clear_context(self) -> None:
        pass

    @abstractmethod
    def _initial_value(self, shape: tuple[int, ...]):
        raise NotImplementedError

    @abstractmethod
    def _drive(self, inputs: ReductionInputs):
        raise NotImplementedError

    def _validate_alpha(self, value) -> None:
        _validate_alpha(value, owner=type(self).__name__)

    @abstractmethod
    def _validate_threshold(self, value) -> None:
        raise NotImplementedError


class EventAccumulatorReduction(_AccumulatorReduction):
    """Accumulate nonzero synapse slots with decay and threshold-reset events."""

    def __init__(self, *, alpha: float = 0.0, threshold: float = 1.0) -> None:
        super().__init__(alpha=alpha, threshold=threshold)

    def _initial_value(self, shape: tuple[int, ...]):
        return jnp.zeros(shape, dtype=float)

    def _drive(self, inputs: ReductionInputs):
        return _active_synapse_count(inputs, self._state.value)

    def _validate_threshold(self, value) -> None:
        _validate_numeric_threshold(value, owner=type(self).__name__)


class PayloadAccumulatorReduction(_AccumulatorReduction):
    """Accumulate scalar synapse payloads while preserving their physical unit."""

    def __init__(self, *, alpha: float = 0.0, threshold=1.0 * u.uS) -> None:
        super().__init__(alpha=alpha, threshold=threshold)

    def _prepare_context(self, context: ReductionContext) -> None:
        threshold_unit = u.get_unit(self._parameters.defaults["threshold"])
        for group in context.input_groups:
            event_input = group.event_input
            if not isinstance(event_input, ScalarEventInput):
                raise TypeError(
                    f"{type(self).__name__} requires ScalarEventInput groups; "
                    f"synapse type {group.synapse_type!r} declares {type(event_input).__name__}."
                )
            if u.get_dim(event_input.unit) != u.get_dim(threshold_unit):
                raise ValueError(
                    f"{type(self).__name__} threshold units are incompatible with "
                    f"synapse type {group.synapse_type!r} payload units {event_input.unit}."
                )

    def _initial_value(self, shape: tuple[int, ...]):
        threshold = self._parameters.defaults["threshold"]
        return u.Quantity(jnp.zeros(shape, dtype=float), u.get_unit(threshold))

    def _drive(self, inputs: ReductionInputs):
        unit = u.get_unit(self._state.value)
        return _sum_rows_by_population(
            inputs,
            template=u.get_mantissa(self._state.value),
            row_values=lambda group: group.payload.to_decimal(unit),
            unit=unit,
        )

    def _validate_threshold(self, value) -> None:
        _validate_quantity_threshold(
            value,
            owner=type(self).__name__,
            expected_unit=u.uS,
        )


class SynapticKernelAccumulatorReduction(_AccumulatorReduction):
    """Accumulate type-aware analytic synaptic-kernel areas without synapse state."""

    def __init__(self, *, alpha: float = 0.0, threshold=1.0 * u.uS * u.ms) -> None:
        super().__init__(alpha=alpha, threshold=threshold)
        self._kernel_area_ms = {}

    def _prepare_context(self, context: ReductionContext) -> None:
        threshold_unit = u.get_unit(self._parameters.defaults["threshold"])
        synapses_by_id = {synapse.id: synapse for synapse in context.synapses}
        kernel_area_ms = {}
        for group in context.input_groups:
            event_input = group.event_input
            if not isinstance(event_input, ScalarEventInput):
                raise TypeError(
                    f"{type(self).__name__} requires ScalarEventInput groups; "
                    f"synapse type {group.synapse_type!r} declares {type(event_input).__name__}."
                )
            if u.get_dim(event_input.unit * u.ms) != u.get_dim(threshold_unit):
                raise ValueError(
                    f"{type(self).__name__} threshold units are incompatible with "
                    f"synapse type {group.synapse_type!r} kernel-area units {event_input.unit * u.ms}."
                )
            kernel_area_ms[group.layout_id] = jnp.asarray(
                [_synaptic_kernel_area_ms(synapses_by_id[int(synapse_id)]) for synapse_id in group.synapse_id],
                dtype=float,
            )
        self._kernel_area_ms = kernel_area_ms

    def _clear_context(self) -> None:
        self._kernel_area_ms = {}

    def _initial_value(self, shape: tuple[int, ...]):
        threshold = self._parameters.defaults["threshold"]
        return u.Quantity(jnp.zeros(shape, dtype=float), u.get_unit(threshold))

    def _drive(self, inputs: ReductionInputs):
        unit = u.get_unit(self._state.value)

        def kernel_area_payload(group):
            area = self._kernel_area_ms[group.schema.layout_id] * u.ms
            return (group.payload * area).to_decimal(unit)

        return _sum_rows_by_population(
            inputs,
            template=u.get_mantissa(self._state.value),
            row_values=kernel_area_payload,
            unit=unit,
        )

    def _validate_threshold(self, value) -> None:
        _validate_quantity_threshold(
            value,
            owner=type(self).__name__,
            expected_unit=u.uS * u.ms,
        )


def _active_synapse_count(inputs: ReductionInputs, template):
    result = jnp.zeros_like(template)
    for group in inputs:
        payload = u.get_magnitude(group.payload)
        active = jnp.asarray(payload != 0, dtype=result.dtype)
        target_shape = tuple(result.shape[:-1]) + (group.schema.size,)
        active = jnp.broadcast_to(active, target_shape)
        result = result.at[..., jnp.asarray(group.schema.population_index)].add(active)
    return result


def _sum_rows_by_population(inputs: ReductionInputs, *, template, row_values, unit):
    result = jnp.zeros_like(template)
    for group in inputs:
        values = jnp.asarray(row_values(group), dtype=result.dtype)
        target_shape = tuple(result.shape[:-1]) + (group.schema.size,)
        values = jnp.broadcast_to(values, target_shape)
        result = result.at[..., jnp.asarray(group.schema.population_index)].add(values)
    return u.Quantity(result, unit)


def _synaptic_kernel_area_ms(synapse) -> float:
    if synapse.synapse_type == "ExpSyn":
        return _time_parameter_ms(synapse, "tau")
    if synapse.synapse_type == "Exp2Syn":
        tau1 = _time_parameter_ms(synapse, "tau1")
        tau2 = _time_parameter_ms(synapse, "tau2")
        t_peak = tau1 * tau2 / (tau2 - tau1) * np.log(tau2 / tau1)
        factor = 1.0 / (np.exp(-t_peak / tau2) - np.exp(-t_peak / tau1))
        return float(factor * (tau2 - tau1))
    raise TypeError(
        "SynapticKernelAccumulatorReduction supports only ExpSyn and Exp2Syn; "
        f"got synapse type {synapse.synapse_type!r}."
    )


def _time_parameter_ms(synapse, name: str) -> float:
    try:
        value = synapse.parameters[name]
    except KeyError as exc:
        raise ValueError(f"Synapse type {synapse.synapse_type!r} is missing required parameter {name!r}.") from exc
    try:
        result = float(value.to_decimal(u.ms))
    except Exception as exc:
        raise ValueError(f"Synapse type {synapse.synapse_type!r} parameter {name!r} must have time units.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"Synapse type {synapse.synapse_type!r} parameter {name!r} must be finite and > 0.")
    return result


def _runtime_shape(pop_size: tuple[int, ...], batch_size) -> tuple[int, ...]:
    if len(pop_size) > 1:
        raise NotImplementedError("Reference reduction models currently support scalar or one-dimensional pop_size.")
    population_shape = pop_size if pop_size else (1,)
    return ((int(batch_size),) if batch_size is not None else ()) + population_shape


def _selected_values(value, count: int, *, name: str):
    if isinstance(value, u.Quantity):
        shape = tuple(value.shape)
        if shape == ():
            return tuple(value for _ in range(count))
        if shape != (count,):
            raise ValueError(f"Reduction parameter {name!r} must be scalar or shape {(count,)!r}, got {shape!r}.")
        return tuple(value[index] for index in range(count))
    array = np.asarray(value)
    if array.shape == ():
        return tuple(array.item() for _ in range(count))
    if array.shape != (count,):
        raise ValueError(f"Reduction parameter {name!r} must be scalar or shape {(count,)!r}, got {array.shape!r}.")
    return tuple(array.tolist())


def _validate_alpha(value, *, owner: str) -> None:
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
        raise TypeError(f"{owner}.alpha must be a finite scalar.")
    if float(value) < 0.0 or float(value) > 1.0:
        raise ValueError(f"{owner}.alpha must be between 0 and 1 inclusive.")


def _validate_numeric_threshold(value, *, owner: str) -> None:
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
        raise TypeError(f"{owner}.threshold must be a finite scalar.")
    if float(value) < 0.0:
        raise ValueError(f"{owner}.threshold must be >= 0.")


def _validate_quantity_threshold(value, *, owner: str, expected_unit=None) -> None:
    if not isinstance(value, u.Quantity) or tuple(value.shape) != ():
        raise TypeError(f"{owner}.threshold must be a scalar quantity.")
    if expected_unit is not None and u.get_dim(value) != u.get_dim(expected_unit):
        raise ValueError(f"{owner}.threshold must have units compatible with {expected_unit}.")
    magnitude = np.asarray(u.get_mantissa(value))
    if not np.isfinite(magnitude).item():
        raise TypeError(f"{owner}.threshold must be finite.")
    if float(magnitude) < 0.0:
        raise ValueError(f"{owner}.threshold must be >= 0.")
