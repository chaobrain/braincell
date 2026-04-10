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

from dataclasses import dataclass
from typing import Any, Callable, Union

import brainunit as u


@dataclass(frozen=True)
class SynapseMechanism:
    synapse_type: str
    params: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class GapJunctionMechanism:
    params: tuple[tuple[str, Any], ...] = ()


class CurrentClamp:
    """Piecewise-constant point current clamp.

    Preferred form:
    - ``CurrentClamp(start=..., durations=(...), amplitudes=(...))``

    Backward-compatible form:
    - ``CurrentClamp(amplitude=..., delay=..., duration=...)``
    """

    __slots__ = ("start", "durations", "amplitudes")

    def __init__(
        self,
        *,
        start: u.Quantity[u.ms] | None = None,
        durations: object | None = None,
        amplitudes: object | None = None,
        amplitude: u.Quantity[u.nA] | None = None,
        delay: u.Quantity[u.ms] | None = None,
        duration: u.Quantity[u.ms] | None = None,
    ) -> None:
        if amplitudes is None and durations is None and amplitude is not None and duration is not None:
            start = 0.0 * u.ms if start is None else start
            delay = 0.0 * u.ms if delay is None else delay
            start = start + delay
            durations = (duration,)
            amplitudes = (amplitude,)
        elif any(value is not None for value in (amplitude, delay, duration)):
            raise TypeError(
                "CurrentClamp(...) accepts either (amplitude, delay, duration) "
                "or (start, durations, amplitudes), not a mixed form."
            )

        if start is None:
            start = 0.0 * u.ms
        durations_tuple = _normalize_quantity_sequence(
            values=durations,
            unit=u.ms,
            field_name="durations",
        )
        amplitudes_tuple = _normalize_quantity_sequence(
            values=amplitudes,
            unit=u.nA,
            field_name="amplitudes",
        )
        if len(durations_tuple) == 0:
            raise ValueError("CurrentClamp.durations must be non-empty.")
        if len(durations_tuple) != len(amplitudes_tuple):
            raise ValueError("CurrentClamp.durations and amplitudes must have the same length.")
        for item in durations_tuple:
            if float(item.to_decimal(u.ms)) <= 0.0:
                raise ValueError(f"CurrentClamp durations must be > 0, got {item!r}.")

        object.__setattr__(self, "start", start.in_unit(u.ms))
        object.__setattr__(self, "durations", durations_tuple)
        object.__setattr__(self, "amplitudes", amplitudes_tuple)

    @property
    def delay(self) -> u.Quantity[u.ms]:
        return self.start

    @property
    def duration(self) -> u.Quantity[u.ms]:
        if len(self.durations) != 1:
            raise AttributeError("CurrentClamp.duration is only defined for single-segment clamps.")
        return self.durations[0]

    @property
    def amplitude(self) -> u.Quantity[u.nA]:
        if len(self.amplitudes) != 1:
            raise AttributeError("CurrentClamp.amplitude is only defined for single-segment clamps.")
        return self.amplitudes[0]

    def __repr__(self) -> str:
        return (
            "CurrentClamp("
            f"start={self.start!r}, durations={self.durations!r}, amplitudes={self.amplitudes!r})"
        )


@dataclass(frozen=True)
class SineClamp:
    amplitude: u.Quantity[u.nA]
    frequency: u.Quantity[u.Hz]
    phase: float = 0.0
    offset: u.Quantity[u.nA] = 0.0 * u.nA
    start: u.Quantity[u.ms] = 0.0 * u.ms
    duration: u.Quantity[u.ms] = 1.0 * u.ms


@dataclass(frozen=True)
class FunctionClamp:
    fn: Callable[[object], object]
    start: u.Quantity[u.ms] = 0.0 * u.ms
    duration: u.Quantity[u.ms] = 1.0 * u.ms


@dataclass(frozen=True)
class ProbeMechanism:
    variable: str
    target: str | None = None


PointMechanism = Union[
    SynapseMechanism,
    GapJunctionMechanism,
    CurrentClamp,
    SineClamp,
    FunctionClamp,
    ProbeMechanism,
]


def _normalize_quantity_sequence(*, values: object | None, unit: object, field_name: str) -> tuple[object, ...]:
    if values is None:
        raise TypeError(f"CurrentClamp.{field_name} is required.")
    shape = getattr(values, "shape", ())
    if hasattr(values, "to_decimal") and shape != ():
        decimals = values.to_decimal(unit)
        return tuple(item * unit for item in decimals.reshape((-1,)).tolist())
    if isinstance(values, (list, tuple)):
        normalized: list[object] = []
        for item in values:
            if not hasattr(item, "to_decimal"):
                raise TypeError(f"CurrentClamp.{field_name} entries must be quantities, got {item!r}.")
            normalized.append(item.in_unit(unit))
        return tuple(normalized)
    if hasattr(values, "to_decimal"):
        return (values.in_unit(unit),)
    raise TypeError(f"CurrentClamp.{field_name} must be a quantity or sequence of quantities.")
