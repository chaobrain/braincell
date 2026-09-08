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

"""Optimizer-facing live parameter collection."""

from __future__ import annotations

from collections.abc import Mapping

import brainstate
import brainunit as u
import numpy as np

__all__ = ["ParameterSet"]


class ParameterSet:
    """Expose stable names for the original trainable roots."""

    __slots__ = ("_roots",)

    def __init__(self, roots: Mapping[str, brainstate.nn.Param]) -> None:
        self._roots = roots

    def states(self) -> dict[str, brainstate.ParamState]:
        """Return the original fitted ParamState objects by stable name."""
        return {name: root.val for name, root in self._roots.items() if isinstance(root.val, brainstate.ParamState)}

    def physical_values(self) -> dict[str, object]:
        """Return transformed physical root values."""
        return {name: root.value() for name, root in self._roots.items()}

    def optimizer_values(self) -> dict[str, object]:
        """Return optimizer-space root values."""
        return {
            name: root.val.value if isinstance(root.val, brainstate.State) else root.val
            for name, root in self._roots.items()
        }

    def set_physical_values(self, values: Mapping[str, object]) -> None:
        """Atomically assign a complete physical-value mapping."""
        checked = self._validate_tree(values, self.physical_values())
        raw_values = {}
        optimizer_values = self.optimizer_values()
        for name, value in checked.items():
            raw = self._roots[name].t.inverse(value)
            _validate_value(raw, optimizer_values[name], name=name)
            raw_values[name] = raw
        for name, raw in raw_values.items():
            state = self._roots[name].val
            if not isinstance(state, brainstate.State):
                raise TypeError(f"Root {name!r} is fixed and has no optimizer state.")
            state.value = raw

    def set_optimizer_values(self, values: Mapping[str, object]) -> None:
        """Atomically assign a complete optimizer-space mapping."""
        checked = self._validate_tree(values, self.optimizer_values())
        for name, value in checked.items():
            state = self._roots[name].val
            if not isinstance(state, brainstate.State):
                raise TypeError(f"Root {name!r} is fixed and has no optimizer state.")
            state.value = value

    def _validate_tree(self, values: Mapping[str, object], current: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(values, Mapping):
            raise TypeError("Parameter values must be a mapping keyed by stable root name.")
        if set(values) != set(current):
            missing = tuple(sorted(set(current).difference(values)))
            extra = tuple(sorted(set(values).difference(current)))
            raise KeyError(f"Parameter tree keys differ (missing={missing!r}, extra={extra!r}).")
        checked = {}
        for name, value in values.items():
            reference = current[name]
            _validate_value(value, reference, name=name)
            checked[name] = value
        return checked


def _validate_value(value: object, reference: object, *, name: str) -> None:
    if isinstance(reference, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Root {name!r} requires a Quantity compatible with {reference.unit}.")
        try:
            decimal = np.asarray(value.to_decimal(reference.unit))
        except Exception as exc:
            raise ValueError(f"Root {name!r} has incompatible units.") from exc
    else:
        if isinstance(value, u.Quantity):
            raise TypeError(f"Root {name!r} is dimensionless.")
        decimal = np.asarray(value)
    if tuple(decimal.shape) != tuple(getattr(reference, "shape", ())):
        raise ValueError(
            f"Root {name!r} shape mismatch: expected {getattr(reference, 'shape', ())!r}, got {decimal.shape!r}."
        )
    if not np.all(np.isfinite(decimal)):
        raise ValueError(f"Root {name!r} must contain only finite values.")
