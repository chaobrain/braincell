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

#: The aggregations a :class:`TriggerEventInput` may declare. Each one has a
#: runtime behaviour test; add a value here only together with its test.
TRIGGER_AGGREGATIONS = frozenset({"count"})
#: The aggregations a :class:`ScalarEventInput` may declare, under the same rule.
SCALAR_AGGREGATIONS = frozenset({"sum", "sum_count"})

#: Values accepted before braincell 0.3 that the runtime never honoured
#: (CELL-40): it summed every payload and counted every trigger.
_REMOVED_TRIGGER_AGGREGATIONS = ("any", "ordered")
_REMOVED_SCALAR_AGGREGATIONS = ("ordered",)


def _check_aggregation(
    contract: str, aggregation: str, allowed: frozenset, removed: tuple, runtime: str, replacement: str
) -> None:
    if aggregation in allowed:
        return
    supported = ", ".join(repr(value) for value in sorted(allowed))
    if aggregation in removed:
        raise ValueError(
            f"{contract}(aggregation={aggregation!r}) was removed in braincell 0.3: the runtime always "
            f"{runtime} the events that reach one synapse row on one boundary, so {aggregation!r} never took "
            f"effect. Use aggregation={replacement!r}."
        )
    kind = "trigger" if contract == "TriggerEventInput" else "scalar"
    raise ValueError(f"Unsupported {kind}-event aggregation {aggregation!r}. Supported: {supported}.")


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
    """Declare a payload-free event input.

    Parameters
    ----------
    aggregation : {"count"}, optional
        How the events that reach one synapse row on one boundary combine.
        ``"count"``, the only value, passes their number to
        ``apply_events``.

    Raises
    ------
    ValueError
        For any other value. ``"any"`` and ``"ordered"`` were accepted
        before braincell 0.3, but the runtime counted the events anyway;
        they are now rejected with a pointer to ``"count"``.

    Examples
    --------
    >>> from braincell.mech import TriggerEventInput
    >>> TriggerEventInput().aggregation
    'count'
    """

    def __init__(self, *, aggregation: str = "count") -> None:
        _check_aggregation(
            "TriggerEventInput", aggregation, TRIGGER_AGGREGATIONS, _REMOVED_TRIGGER_AGGREGATIONS, "counted", "count"
        )
        super().__init__(payload_kind="trigger", aggregation=aggregation)


@dataclass(frozen=True, init=False)
class ScalarEventInput(EventInput):
    """Declare a scalar physical event payload.

    Parameters
    ----------
    unit : brainunit.Unit
        Unit the payload must be compatible with, such as ``u.uS`` for a
        conductance increment.
    aggregation : {"sum", "sum_count"}, optional
        How the payloads of the events that reach one synapse row on one
        boundary combine. ``"sum"`` (the default) passes their sum
        ``sum(w * n)`` to ``apply_events(payload, V_post=None)``.
        ``"sum_count"`` passes the same sum and, as the keyword argument
        ``count``, the number of those events ``sum(n)``:
        ``apply_events(payload, V_post=None, *, count)``. ``count`` is a
        dimensionless float array shaped like ``payload`` that holds whole
        numbers, zero on rows without events. Every delivery path counts:
        scheduled sources, the Network's delay queues and zero-delay
        delivery, a standalone Cell's live connections, and rows with a
        plasticity rule.

    Raises
    ------
    ValueError
        For any other value. ``"ordered"`` was accepted before braincell
        0.3, but the runtime summed the payloads anyway; it is now rejected
        with a pointer to ``"sum"``.

    Notes
    -----
    The events that reach one synapse row on one boundary form an unordered
    multiset, so a ``"sum_count"`` model may depend on ``(sum(w * n),
    sum(n))`` only, not on each event's payload or their order. A
    ``"sum_count"`` synapse class needs an ``apply_events`` that accepts
    ``count``; :func:`~braincell.mech.register_synapse` rejects it otherwise.
    A reduction model receives payload sums without counts and rejects a
    ``"sum_count"`` synapse at ``init_state``. See
    :class:`braincell.synapse.ExpSynSTD` and
    :class:`braincell.synapse.ExpSynSTP` for models that use the count.

    Examples
    --------
    >>> import brainunit as u
    >>> from braincell.mech import ScalarEventInput
    >>> ScalarEventInput(u.uS).aggregation
    'sum'
    >>> ScalarEventInput(u.uS, aggregation="sum_count").aggregation
    'sum_count'
    """

    unit: object

    def __init__(self, unit, *, aggregation: str = "sum") -> None:
        _check_aggregation(
            "ScalarEventInput", aggregation, SCALAR_AGGREGATIONS, _REMOVED_SCALAR_AGGREGATIONS, "summed", "sum"
        )
        super().__init__(payload_kind="scalar", aggregation=aggregation)
        object.__setattr__(self, "unit", unit)

    def validate_payload(self, payload):
        """Validate and return a payload without forcing host materialization.

        The check is purely dimensional, so it compares dimension
        metadata rather than converting the payload. The previous
        implementation called ``payload.to_decimal(self.unit)`` and threw
        the result away, which emitted a real array multiply on every
        call -- dead work that only XLA's DCE removed, and only on the
        jitted run paths.
        """
        if not isinstance(payload, u.Quantity):
            raise TypeError(f"Scalar event payload must be a quantity compatible with {self.unit}.")
        try:
            compatible = payload.dim == u.get_dim(self.unit)
        except Exception as exc:
            raise ValueError(f"Scalar event payload has units incompatible with {self.unit}.") from exc
        if not compatible:
            raise ValueError(f"Scalar event payload has units incompatible with {self.unit}.")
        return payload
