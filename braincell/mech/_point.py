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

"""Point-located mechanism declarations.

A *point mechanism* is a declaration that gets attached to one specific
location on a cell (a compartment midpoint in the current
implementation) rather than distributed over a region. All point
mechanisms share :class:`Point` as a common base class so that
downstream consumers can dispatch on ``isinstance(x, Point)``.

Concrete point mechanisms defined here:

- :class:`CurrentClamp` — piecewise-constant current injection.
- :class:`SineClamp` — sinusoidal current injection.
- :class:`FunctionClamp` — arbitrary ``t → I`` callable.
- :class:`StateProbe` — probe for cell-owned state such as ``v``.
- :class:`MechanismProbe` — probe for runtime state on a named mechanism.
- :class:`CurrentProbe` — probe for mechanism or total ion current.
- :class:`ProbeMechanism` — legacy recorder for a named variable.
- :class:`Synapse` — registry-keyed synapse model.

The :class:`~braincell.mech.Junction` gap-junction declaration also
inherits from :class:`Point` but lives in its own module
:mod:`braincell.mech._junction`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import brainunit as u
import numpy as np

from ._base import Mechanism
from ._params import Params, quantity_hashable

__all__ = [
    "Point",
    "CurrentClamp",
    "FunctionClamp",
    "NetStim",
    "StateProbe",
    "MechanismProbe",
    "CurrentProbe",
    "ProbeMechanism",
    "SineClamp",
    "Synapse",
]


class Point(Mechanism):
    """Marker base class for point-located mechanism declarations.

    All concrete subclasses are frozen :func:`~dataclasses.dataclass`
    types — this base exists solely so that consumers can write
    ``isinstance(x, Point)`` instead of maintaining a parallel tuple
    of concrete types.

    :class:`Point` defines no abstract methods. Runtime evaluation of
    clamp-like mechanisms happens in :mod:`braincell._compute.runtime`,
    which inspects concrete subclasses directly.
    """

    __slots__ = ()


# ---------------------------------------------------------------------------
# Current injection
# ---------------------------------------------------------------------------


@quantity_hashable
@dataclass(frozen=True)
class CurrentClamp(Point):
    """Piecewise-constant current clamp.

    Parameters
    ----------
    delay : Quantity[ms]
        Absolute simulation time at which the first segment begins. May be a
        scalar or broadcastable to the placed target shape.
    durations : Quantity[ms] or sequence of Quantity[ms]
        Single-segment duration or multi-segment durations.
    amplitudes : Quantity[nA] or sequence of Quantity[nA]
        Single-segment amplitude or multi-segment amplitudes.
    target_index : array-like of int or None, optional
        Reserved sparse target indices over the flattened placement target
        axis. ``None`` keeps dense/broadcast semantics.

    Raises
    ------
    TypeError
        If ``delay``, ``durations`` or ``amplitudes`` are not quantities.
    ValueError
        If any duration is non-positive, or if ``target_index`` is not a
        one-dimensional non-negative integer array.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.mech import CurrentClamp
        >>> cc = CurrentClamp(
        ...     delay=10 * u.ms,
        ...     durations=50 * u.ms,
        ...     amplitudes=0.2 * u.nA,
        ... )
    """

    delay: Any = field(default_factory=lambda: 0.0 * u.ms)
    durations: Any = field(default_factory=lambda: 1.0 * u.ms)
    amplitudes: Any = field(default_factory=lambda: 0.0 * u.nA)
    target_index: Any = None

    def __post_init__(self) -> None:
        delay = _coerce_quantity(
            self.delay, unit=u.ms, field_name="CurrentClamp.delay"
        )
        durations = _coerce_quantity(
            self.durations, unit=u.ms, field_name="CurrentClamp.durations"
        )
        amplitudes = _coerce_quantity(
            self.amplitudes, unit=u.nA, field_name="CurrentClamp.amplitudes"
        )
        _raise_if_nonpositive_duration(durations)
        target_index = _normalize_target_index(self.target_index)

        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "target_index", target_index)


@quantity_hashable
@dataclass(frozen=True)
class SineClamp(Point):
    """Sinusoidal current clamp.

    Parameters
    ----------
    amplitude : Quantity[nA]
        Peak amplitude.
    frequency : Quantity[Hz]
        Oscillation frequency.
    phase : float
        Phase offset in radians.
    offset : Quantity[nA]
        Constant offset added to the sine.
    delay : Quantity[ms]
        Absolute start time.
    duration : Quantity[ms]
        Length of the active window. The clamp returns zero before
        ``delay`` and after ``delay + duration``.
    """

    amplitude: Any
    frequency: Any
    phase: float = 0.0
    offset: Any = field(default_factory=lambda: 0.0 * u.nA)
    delay: Any = field(default_factory=lambda: 0.0 * u.ms)
    duration: Any = field(default_factory=lambda: 1.0 * u.ms)

    def __post_init__(self) -> None:
        frequency = _coerce_scalar_quantity(
            self.frequency, unit=u.Hz, field_name="SineClamp.frequency",
        )
        if float(frequency.to_decimal(u.Hz)) <= 0.0:
            raise ValueError(
                f"SineClamp.frequency must be > 0, got {self.frequency!r}."
            )
        duration = _coerce_scalar_quantity(
            self.duration, unit=u.ms, field_name="SineClamp.duration",
        )
        if float(duration.to_decimal(u.ms)) <= 0.0:
            raise ValueError(
                f"SineClamp.duration must be > 0, got {self.duration!r}."
            )
        if not isinstance(self.phase, (int, float)) or isinstance(self.phase, bool):
            raise TypeError(
                f"SineClamp.phase must be a real number, "
                f"got {type(self.phase).__name__!r}."
            )
        amplitude = _coerce_scalar_quantity(
            self.amplitude, unit=u.nA, field_name="SineClamp.amplitude",
        )
        offset = _coerce_scalar_quantity(
            self.offset, unit=u.nA, field_name="SineClamp.offset",
        )
        delay = _coerce_scalar_quantity(
            self.delay, unit=u.ms, field_name="SineClamp.delay",
        )
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "duration", duration)


@quantity_hashable
@dataclass(frozen=True)
class FunctionClamp(Point):
    """Arbitrary-callable current clamp.

    Parameters
    ----------
    fn : Callable
        A function ``f(t) -> Quantity[nA]`` called each step with the
        absolute simulation time. Use explicit conditions inside ``fn``
        for windowed current injection.

    Notes
    -----
    Equality and hashing follow identity on ``fn`` (frozen dataclass
    auto-generated dunder methods compare lambdas by ``==``, which
    falls back to identity). Two :class:`FunctionClamp` instances built
    from two separate ``lambda`` definitions with identical bodies are
    considered distinct.

    The runtime layer fingerprints ``fn`` by bytecode + closure cells so
    structurally identical lambdas can merge into one layout. Closure
    cells holding opaque, non-hashable objects fall back to ``id(value)``
    and therefore defeat dedup. Such lambdas trigger a one-shot
    :class:`RuntimeWarning` — hoist to module level with a named function
    to recover dedup.
    """

    fn: Callable

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(
                f"FunctionClamp.fn must be callable, "
                f"got {type(self.fn).__name__!r}."
            )


# ---------------------------------------------------------------------------
# Point spike sources, observers & synapses
# ---------------------------------------------------------------------------


@quantity_hashable
@dataclass(frozen=True)
class NetStim(Point):
    """Deterministic point spike source aligned with NEURON's `NetStim`.

    Parameters
    ----------
    start : Quantity[ms]
        Absolute time of the first spike.
    number : int
        Number of spikes to emit.
    interval : Quantity[ms]
        Spacing between spikes.
    noise : float, optional
        Randomness level. Only ``0.0`` is supported in v1.
    weight : float, optional
        Event amplitude written into downstream ``pre_spike`` buffers.
    name : str or None, optional
        Optional instance label.
    """

    start: Any
    number: int
    interval: Any
    noise: float = 0.0
    weight: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        start = _coerce_scalar_quantity(
            self.start, unit=u.ms, field_name="NetStim.start",
        )
        interval = _coerce_scalar_quantity(
            self.interval, unit=u.ms, field_name="NetStim.interval",
        )
        if float(interval.to_decimal(u.ms)) <= 0.0:
            raise ValueError(
                f"NetStim.interval must be > 0, got {self.interval!r}."
            )
        if not isinstance(self.number, int) or isinstance(self.number, bool):
            raise TypeError(f"NetStim.number must be int, got {self.number!r}.")
        if self.number < 0:
            raise ValueError(f"NetStim.number must be >= 0, got {self.number!r}.")
        if not isinstance(self.noise, (int, float)) or isinstance(self.noise, bool):
            raise TypeError(f"NetStim.noise must be a real number, got {self.noise!r}.")
        if float(self.noise) != 0.0:
            raise ValueError(
                f"NetStim.noise={self.noise!r} is not supported yet; use noise=0.0."
            )
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError(f"NetStim.weight must be a real number, got {self.weight!r}.")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"NetStim.name must be a non-empty string or None, got {self.name!r}.")

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "interval", interval)

    @property
    def instance_name(self) -> str:
        """Display label for runtime layout and probe/debug views."""
        return self.name if self.name is not None else "NetStim"


@dataclass(frozen=True)
class StateProbe(Point):
    """Probe for cell-owned state at one placed location."""

    name: str | None = None
    field: str = "v"

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"StateProbe.name must be a non-empty string or None, got {self.name!r}.")
        if not isinstance(self.field, str) or not self.field:
            raise ValueError(f"StateProbe.field must be a non-empty string, got {self.field!r}.")
        if self.field != "v":
            raise ValueError(f"Unsupported StateProbe field {self.field!r}; only 'v' is supported.")


@dataclass(frozen=True)
class MechanismProbe(Point):
    """Probe for runtime state on a named mechanism."""

    mechanism: str
    field: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"MechanismProbe.name must be a non-empty string or None, got {self.name!r}.")
        for field_name in ("mechanism", "field"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"MechanismProbe.{field_name} must be a non-empty string, got {value!r}."
                )


@dataclass(frozen=True)
class CurrentProbe(Point):
    """Probe for current at a placed location."""

    ion: str | None = None
    mechanism: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"CurrentProbe.name must be a non-empty string or None, got {self.name!r}.")
        if self.ion is not None and (not isinstance(self.ion, str) or not self.ion):
            raise ValueError(f"CurrentProbe.ion must be a non-empty string or None, got {self.ion!r}.")
        if self.mechanism is not None and (not isinstance(self.mechanism, str) or not self.mechanism):
            raise ValueError(
                f"CurrentProbe.mechanism must be a non-empty string or None, got {self.mechanism!r}."
            )
        if self.ion is None and self.mechanism is None:
            raise ValueError("CurrentProbe requires at least one of 'ion' or 'mechanism'.")


@dataclass(frozen=True)
class ProbeMechanism(Point):
    """Observer that records a named variable at a point location.

    Parameters
    ----------
    variable : str
        Name of the variable to record (e.g. ``"v"``, ``"ina"``).
    target : str or None
        Optional sub-target label (e.g. the owning mechanism's
        instance name) used to disambiguate probes of the same
        variable on different mechanisms.
    """

    variable: str
    target: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError(
                f"ProbeMechanism.variable must be a non-empty str, "
                f"got {self.variable!r}."
            )
        if self.target is not None and not isinstance(self.target, str):
            raise TypeError(
                f"ProbeMechanism.target must be str or None, "
                f"got {type(self.target).__name__!r}."
            )


class Synapse(Point):
    """Registry-keyed synapse declaration.

    Parameters
    ----------
    synapse_type : str
        Registry key for the target synapse class (e.g. ``"AMPA"``,
        ``"NMDA"``).
    name : str or None
        Optional instance label.
    **params
        Synapse parameters.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.mech import Synapse
        >>> syn = Synapse("AMPA")
        >>> syn.synapse_type
        'AMPA'
    """

    __slots__ = ("synapse_type", "params", "name")

    def __init__(
        self,
        synapse_type: str,
        /,
        *,
        name: str | None = None,
        **params: Any,
    ) -> None:
        if not isinstance(synapse_type, str) or not synapse_type:
            raise ValueError(
                f"Synapse.synapse_type must be a non-empty string, "
                f"got {synapse_type!r}."
            )
        if "params" in params:
            raise TypeError(
                "Synapse parameters must be passed as keyword arguments, "
                "not as params={...}."
            )
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"Synapse.name must be a string or None, "
                f"got {type(name).__name__!r}."
            )
        object.__setattr__(self, "synapse_type", synapse_type)
        object.__setattr__(self, "params", Params(params) if params else Params())
        object.__setattr__(self, "name", name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot set attribute {name!r}."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot delete attribute {name!r}."
        )

    @property
    def instance_name(self) -> str:
        """Display label — ``self.name`` if set, else ``synapse_type``."""
        return self.name if self.name is not None else self.synapse_type

    @property
    def identity(self) -> tuple[str, str]:
        """Return ``(instance_name, synapse_type)`` for table views."""
        return (self.instance_name, self.synapse_type)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.synapse_type == other.synapse_type
            and self.params == other.params
            and self.name == other.name
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.synapse_type, self.params, self.name))

    def __repr__(self) -> str:
        return (
            f"Synapse(synapse_type={self.synapse_type!r}, "
            f"params={self.params!r}, name={self.name!r})"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coerce_scalar_quantity(value: Any, *, unit: Any, field_name: str) -> Any:
    if not hasattr(value, "to_decimal"):
        raise TypeError(
            f"{field_name} must be a Quantity, got {value!r}."
        )
    return value.in_unit(unit)


def _coerce_quantity(value: Any, *, unit: Any, field_name: str) -> Any:
    if value is None:
        raise TypeError(f"{field_name} must not be None.")
    if hasattr(value, "to_decimal"):
        return value.in_unit(unit)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{field_name} must be non-empty.")
        if not all(hasattr(item, "to_decimal") for item in value):
            raise TypeError(f"{field_name} entries must be Quantities.")
        decimals = [item.to_decimal(unit) for item in value]
        return u.Quantity(np.stack(decimals, axis=-1), unit)

    raise TypeError(
        f"{field_name} must be a Quantity or sequence of Quantities, "
        f"got {type(value).__name__!r}."
    )


def _raise_if_nonpositive_duration(value: Any) -> None:
    decimals = np.asarray(value.to_decimal(u.ms), dtype=float)
    if decimals.size == 0:
        raise ValueError("CurrentClamp.durations must be non-empty.")
    if np.any(decimals <= 0.0):
        raise ValueError(f"CurrentClamp.durations entries must be > 0, got {value!r}.")


def _normalize_target_index(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(
            f"CurrentClamp.target_index must be one-dimensional, got shape {arr.shape!r}."
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("CurrentClamp.target_index entries must be integers.")
    if np.any(arr < 0):
        raise ValueError("CurrentClamp.target_index entries must be non-negative.")
    return tuple(int(item) for item in arr.tolist())
