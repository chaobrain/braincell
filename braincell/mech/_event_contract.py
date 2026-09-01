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

"""Event-input contracts declared by target mechanisms.

A mechanism advertises what kind of discrete event it can consume by
setting a class-level ``event_input`` to one of the contracts here. The
contracts are pure declarations: they carry no runtime state and import
nothing from :mod:`braincell`, which keeps :mod:`braincell.mech` a leaf
that the channel/synapse base classes and the runtime compiler can both
depend on.

The event *sources* that produce the events described by these contracts
live in :mod:`braincell.network.event`.
"""

from dataclasses import dataclass

import brainunit as u

__all__ = [
    "EventInput",
    "NoEventInput",
    "ScalarEventInput",
    "TriggerEventInput",
]


@dataclass(frozen=True)
class EventInput:
    """Base event-input contract declared by a target mechanism."""

    payload_kind: str
    aggregation: str


@dataclass(frozen=True, init=False)
class NoEventInput(EventInput):
    """Declare that a mechanism cannot consume discrete events."""

    def __init__(self) -> None:
        super().__init__(payload_kind="none", aggregation="none")


@dataclass(frozen=True, init=False)
class TriggerEventInput(EventInput):
    """Declare a payload-free event input."""

    def __init__(self, *, aggregation: str = "count") -> None:
        if aggregation not in {"count", "any", "ordered"}:
            raise ValueError(f"Unsupported trigger-event aggregation {aggregation!r}.")
        super().__init__(payload_kind="trigger", aggregation=aggregation)


@dataclass(frozen=True, init=False)
class ScalarEventInput(EventInput):
    """Declare a scalar physical event payload."""

    unit: object

    def __init__(self, unit, *, aggregation: str = "sum") -> None:
        if aggregation not in {"sum", "ordered"}:
            raise ValueError(f"Unsupported scalar-event aggregation {aggregation!r}.")
        super().__init__(payload_kind="scalar", aggregation=aggregation)
        object.__setattr__(self, "unit", unit)

    def validate_payload(self, payload):
        """Validate and return a payload without forcing host materialization."""
        if not isinstance(payload, u.Quantity):
            raise TypeError(f"Scalar event payload must be a quantity compatible with {self.unit}.")
        try:
            payload.to_decimal(self.unit)
        except Exception as exc:
            raise ValueError(f"Scalar event payload has units incompatible with {self.unit}.") from exc
        return payload
