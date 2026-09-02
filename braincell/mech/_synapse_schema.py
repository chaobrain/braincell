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

"""Field declarations for vectorized runtime synapse models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import brainunit as u
import numpy as np

__all__ = ["ParameterSpec", "StateSpec", "positive"]

Validator = Callable[[object, str], None]


@dataclass(frozen=True)
class ParameterSpec:
    """Declare one broadcastable physical runtime parameter."""

    default: object
    validator: Validator | None = None

    def validate(self, value: object, name: str) -> None:
        _validate_like_default(value, self.default, name=name)
        if self.validator is not None:
            self.validator(value, name)


@dataclass(frozen=True)
class StateSpec:
    """Declare one vectorized differential state and its reset value."""

    initial: object

    def validate(self, value: object, name: str) -> None:
        _validate_like_default(value, self.initial, name=name)


def positive(value: object, name: str) -> None:
    """Require every canonical value to be finite and strictly positive."""
    if np.any(np.asarray(u.get_mantissa(value)) <= 0.0):
        raise ValueError(f"Synapse parameter {name!r} must be > 0.")


def _validate_like_default(value: object, default: object, *, name: str) -> None:
    if isinstance(default, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Synapse field {name!r} requires a quantity compatible with {default.unit}.")
        try:
            decimal = np.asarray(value.to_decimal(default.unit))
        except Exception as exc:
            raise ValueError(f"Synapse field {name!r} has units incompatible with {default.unit}.") from exc
    else:
        if isinstance(value, u.Quantity):
            raise TypeError(f"Synapse field {name!r} must be dimensionless.")
        decimal = np.asarray(value)
    if decimal.size == 0:
        raise ValueError(f"Synapse field {name!r} cannot be empty.")
    try:
        finite = np.isfinite(decimal)
    except TypeError as exc:
        raise TypeError(f"Synapse field {name!r} must be numeric.") from exc
    if not np.all(finite):
        raise ValueError(f"Synapse field {name!r} must contain only finite values.")
