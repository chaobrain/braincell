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

"""Signature-derived channel metadata and compact density parameter storage."""

from __future__ import annotations

from collections.abc import Mapping
import inspect

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._parameter_schema import ParameterSpec, RuntimeParameterState
from braincell.mech import Density, get_registry

__all__ = [
    "RuntimeParameterState",
    "density_parameter_names",
    "density_parameter_spec",
    "density_parameter_value",
    "make_runtime_parameter_state",
    "parameter_state_value",
    "set_parameter_row",
    "set_runtime_parameter_value",
]


def density_parameter_schema(mechanism: Density) -> Mapping[str, ParameterSpec]:
    """Return signature metadata for channels and existing schemas for other densities."""
    runtime_cls = get_registry().get(mechanism.category, mechanism.class_name)
    if mechanism.category == "channel":
        return {
            name: _SignatureParameterSpec(
                mechanism.params.get(name, parameter.default)
                if parameter.default is inspect.Parameter.empty or parameter.default is None
                else parameter.default
            )
            for name, parameter in _channel_signature(runtime_cls).items()
        }
    schema = getattr(runtime_cls, "parameters", {})
    return schema if isinstance(schema, Mapping) else {}


def _channel_signature(runtime_cls) -> dict[str, inspect.Parameter]:
    # BrainState's metaclass exposes (*args, **kwargs) on the class itself.
    parameters = tuple(inspect.signature(runtime_cls.__init__).parameters.values())[1:]
    result = {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        parent = next((cls for cls in runtime_cls.__mro__[1:] if "__init__" in vars(cls)), None)
        if parent is not None and parent is not object:
            result.update(_channel_signature(parent))
    result.update(
        (p.name, p)
        for p in parameters
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    return result


class _SignatureParameterSpec(ParameterSpec):
    def validate(self, value: object, name: str) -> None:
        default = self.default
        if default is inspect.Parameter.empty or default is None or callable(default):
            default = value
        ParameterSpec(default).validate(value, name)


def _is_buffer_value(value: object) -> bool:
    """Distinguish numeric storage from ordinary constructor configuration."""
    if callable(value) and value is not inspect.Parameter.empty:
        return True
    try:
        return np.asarray(u.get_mantissa(value)).dtype.kind in "biufc"
    except (TypeError, ValueError):
        return False


def density_parameter_names(mechanism: Density) -> tuple[str, ...]:
    """Return fields needing numeric runtime storage, not a trainability whitelist."""
    schema = density_parameter_schema(mechanism)
    if mechanism.category == "channel":
        names = dict.fromkeys((*schema, *mechanism.params))
        return tuple(
            name
            for name in names
            if _is_buffer_value(mechanism.params.get(name, schema[name].default if name in schema else None))
        )
    return tuple(schema) if schema else tuple(mechanism.params)


def density_parameter_spec(mechanism: Density, name: str) -> ParameterSpec | None:
    """Return one declared parameter specification when available."""
    return density_parameter_schema(mechanism).get(name)


def density_parameter_value(mechanism: Density, name: str) -> object:
    """Resolve an explicit declaration value or its schema default."""
    if name in mechanism.params:
        return mechanism.params[name]
    spec = density_parameter_spec(mechanism, name)
    if spec is None or spec.default is inspect.Parameter.empty:
        raise KeyError(f"Mechanism has no parameter {name!r}.")
    return spec.default


def make_runtime_parameter_state(
    value: object,
    *,
    full_shape: tuple[int, ...],
    spec: ParameterSpec,
    name: str,
    point_mask: object | None = None,
) -> RuntimeParameterState:
    """Validate and normalize one schema parameter into a compact runtime state."""
    spec.validate(value, name)
    pop_shape, point_size = _rectangular_shape(full_shape)
    population_size = int(np.prod(pop_shape, dtype=int))
    state_kwargs = {
        "full_shape": full_shape,
        "point_mask": point_mask,
        "zero_inactive": name in {"g_max", "g", "gbar", "conductance"},
    }
    shape = tuple(getattr(value, "shape", ()))
    if shape == ():
        return RuntimeParameterState(value, axis="uniform", **state_kwargs)
    if shape == pop_shape + (1,):
        return RuntimeParameterState(value, axis="population", **state_kwargs)
    if shape == (1, point_size):
        return RuntimeParameterState(value[0], axis="cv", **state_kwargs)
    if shape == (point_size,) and point_size != population_size:
        return RuntimeParameterState(value, axis="cv", **state_kwargs)
    if shape == pop_shape and point_size != population_size:
        return RuntimeParameterState(value[..., None], axis="population", **state_kwargs)
    if shape == full_shape:
        return RuntimeParameterState(value, axis="row", **state_kwargs)
    if shape == (population_size,) and population_size == point_size:
        raise ValueError(
            f"Parameter {name!r} shape {shape!r} is ambiguous; use ({population_size}, 1) for population "
            f"or (1, {point_size}) for CV sharing."
        )
    try:
        broadcast = u.math.broadcast_to(value, full_shape)
    except Exception as exc:
        raise ValueError(f"Parameter {name!r} cannot broadcast to runtime shape {full_shape!r}.") from exc
    return RuntimeParameterState(broadcast, axis="row", **state_kwargs)


def parameter_state_value(value: object) -> object:
    """Unwrap a runtime parameter state and pass other values through."""
    return value.dense_value() if isinstance(value, RuntimeParameterState) else value


def set_runtime_parameter_value(state: RuntimeParameterState, value: object) -> None:
    """Set a complete runtime value while preserving compact scalar storage."""
    full_shape = tuple(state.full_shape)
    shape = tuple(getattr(value, "shape", ()))
    current = state.value
    current_unit = current.unit if isinstance(current, u.Quantity) else None
    if current_unit is not None:
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Runtime parameter requires a Quantity compatible with {current_unit}.")
        value = value.to(current_unit)
    elif isinstance(value, u.Quantity):
        raise TypeError("Dimensionless runtime parameter cannot be assigned a Quantity.")
    if shape == ():
        state.value = value
        state.axis = "uniform"
        return
    if shape != full_shape:
        raise ValueError(f"State assignment shape mismatch: expected {full_shape!r}, got {shape!r}.")
    state.value = value
    state.axis = "row"


def set_parameter_row(
    state: RuntimeParameterState,
    *,
    population_index: int,
    point_id: int,
    population_size: int,
    point_size: int,
    value: object,
) -> None:
    """Write one logical row, promoting the state to row storage when needed."""
    current = state.dense_value()
    unit = current.unit if isinstance(current, u.Quantity) else None
    if unit is not None:
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Density parameter requires a Quantity compatible with {unit}.")
        replacement = value.to_decimal(unit)
        dtype = jnp.result_type(current.mantissa, replacement)
        mantissa = jnp.asarray(current.to_decimal(unit), dtype=dtype).at[population_index, point_id].set(replacement)
        state.value = u.Quantity(mantissa, unit)
    else:
        if isinstance(value, u.Quantity):
            raise TypeError("Dimensionless density parameter cannot be assigned a Quantity.")
        state.value = (
            jnp.asarray(current, dtype=jnp.result_type(current, value)).at[population_index, point_id].set(value)
        )
    state.axis = "row"


def _as_full_value(value: object, *, axis: str, population_size: int, point_size: int) -> object:
    full_shape = (population_size, point_size)
    if axis == "uniform":
        return u.math.broadcast_to(value, full_shape)
    if axis == "population":
        return u.math.broadcast_to(value, full_shape)
    if axis == "cv":
        return u.math.broadcast_to(value, full_shape)
    if axis == "row":
        return value
    raise ValueError(f"Unknown density parameter axis {axis!r}.")


def _rectangular_shape(shape: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    if len(shape) < 2:
        raise ValueError(f"Schema density parameters require population and CV axes, got {shape!r}.")
    return tuple(int(size) for size in shape[:-1]), int(shape[-1])
