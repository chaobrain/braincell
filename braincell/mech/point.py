from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from .._units import u


@dataclass(frozen=True)
class SynapseMechanism:
    synapse_type: str
    params: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class GapJunctionMechanism:
    params: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class CurrentClamp:
    amplitude: u.Quantity[u.nA]
    delay: u.Quantity[u.ms]
    duration: u.Quantity[u.ms]


@dataclass(frozen=True)
class ProbeMechanism:
    variable: str
    target: str | None = None


PointMechanism = Union[SynapseMechanism, GapJunctionMechanism, CurrentClamp, ProbeMechanism]
