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

from dataclasses import dataclass
from typing import Callable

import brainstate
import brainunit as u
import numpy as np

__all__ = ["DerivedSpec", "ParameterSpec", "StateSpec", "positive"]

Validator = Callable[[object, str], None]
_OWNER_MANAGED = object()


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


@dataclass(frozen=True)
class ParameterSpec:
    """Declare one broadcastable physical runtime parameter."""

    default: object
    validator: Validator | None = None

    def validate(self, value: object, name: str) -> None:
        """Validate one resolved parameter value against this declaration."""
        _validate_like_default(value, self.default, name=name)
        if self.validator is not None:
            self.validator(value, name)


@dataclass(frozen=True)
class StateSpec:
    """Declare one differential state and its optional generic initial value."""

    initial: object = _OWNER_MANAGED

    @property
    def owner_managed(self) -> bool:
        """Whether the mechanism implements initialization itself."""
        return self.initial is _OWNER_MANAGED

    def validate(self, value: object, name: str) -> None:
        """Validate a state value when this declaration owns initialization."""
        if self.owner_managed:
            return
        _validate_like_default(value, self.initial, name=name)


@dataclass(frozen=True)
class DerivedSpec:
    """Mark one public, read-only value computed by a runtime model."""


def positive(value: object, name: str) -> None:
    """Require every canonical value to be finite and strictly positive."""
    decimal = _decimal(value)
    if np.any(decimal <= 0.0):
        raise ValueError(f"Mechanism parameter {name!r} must be > 0.")


def _validate_like_default(value: object, default: object, *, name: str) -> None:
    if isinstance(default, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Mechanism field {name!r} requires a quantity compatible with {default.unit}.")
        try:
            decimal = np.asarray(value.to_decimal(default.unit))
        except Exception as exc:
            raise ValueError(f"Mechanism field {name!r} has units incompatible with {default.unit}.") from exc
    else:
        if isinstance(value, u.Quantity):
            raise TypeError(f"Mechanism field {name!r} must be dimensionless.")
        decimal = np.asarray(value)
    if decimal.size == 0:
        raise ValueError(f"Mechanism field {name!r} cannot be empty.")
    try:
        finite = np.isfinite(decimal)
    except TypeError as exc:
        raise TypeError(f"Mechanism field {name!r} must be numeric.") from exc
    if not np.all(finite):
        raise ValueError(f"Mechanism field {name!r} must contain only finite values.")


def _decimal(value: object) -> np.ndarray:
    if isinstance(value, u.Quantity):
        return np.asarray(value.to_decimal(value.unit))
    return np.asarray(value)
