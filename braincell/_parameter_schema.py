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

"""Shared field declarations for runtime mechanisms."""

from __future__ import annotations

import inspect

import brainstate
import brainunit as u

from braincell.mech._synapse_schema import ParameterSpec, StateSpec, positive

__all__ = ["ParameterSpec", "RuntimeParameterState", "StateSpec", "positive"]


def constructor_parameters(runtime_cls):
    """Inspect explicit constructor fields, including forwarded inheritance."""
    parameters = tuple(inspect.signature(runtime_cls.__init__).parameters.values())[1:]
    result = {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        parent = next((cls for cls in runtime_cls.__mro__[1:] if "__init__" in vars(cls)), None)
        if parent is not None and parent is not object:
            result.update(constructor_parameters(parent))
    result.update(
        (p.name, p)
        for p in parameters
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    return result


class SignatureParameterSpec(ParameterSpec):
    """Infer a missing unit prototype from the supplied constructor value."""

    def validate(self, value: object, name: str) -> None:
        default = self.default
        if default is inspect.Parameter.empty or default is None or callable(default):
            default = value
        ParameterSpec(default).validate(value, name)


class RuntimeParameterState(brainstate.LongTermState):
    """Store a materialized physical parameter outside the optimizer tree."""

    def __init__(
        self,
        value: object,
        *,
        axis: str = "uniform",
        full_shape: tuple[int, ...] | None = None,
        point_mask: object | None = None,
        zero_inactive: bool = False,
    ) -> None:
        super().__init__(
            value,
            axis=axis,
            full_shape=full_shape,
            point_mask=point_mask,
            zero_inactive=zero_inactive,
        )

    def dense_value(self, *, masked: bool = False):
        """Return the broadcast runtime rectangle without expanding stored state.

        ``point_mask`` is retained as serialized metadata compatibility; for
        density runtime parameters it contains the active CV mask.
        """
        value = self.value
        if self.full_shape is not None and tuple(getattr(value, "shape", ())) != tuple(self.full_shape):
            value = u.math.broadcast_to(value, self.full_shape)
        if masked and self.zero_inactive and self.point_mask is not None:
            mask = u.math.asarray(self.point_mask, dtype=bool)
            if isinstance(value, u.Quantity):
                value = u.Quantity(u.math.where(mask, value.mantissa, 0.0), value.unit)
            else:
                value = u.math.where(mask, value, 0.0)
        return value

    @property
    def shape(self) -> tuple[int, ...]:
        """Expose the logical dense shape for compatibility inspection."""
        return tuple(getattr(self.dense_value(), "shape", ()))

    @property
    def unit(self):
        """Expose the physical unit of a Quantity-backed parameter."""
        value = self.dense_value()
        if not isinstance(value, u.Quantity):
            raise AttributeError("Dimensionless runtime parameter has no unit.")
        return value.unit

    @property
    def mantissa(self):
        """Expose a dense mantissa for compatibility inspection."""
        value = self.dense_value()
        return value.mantissa if isinstance(value, u.Quantity) else value

    def __getitem__(self, index):
        return self.dense_value()[index]

    def to_decimal(self, unit):
        """Return the current physical mantissa in the requested unit.

        Parameters
        ----------
        unit : brainunit.Unit
            Unit compatible with the stored physical quantity.

        Returns
        -------
        array
            Converted mantissa without copying the optimizer parameter.
        """
        return self.dense_value().to_decimal(unit)
