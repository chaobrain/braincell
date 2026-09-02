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
- :class:`Synapse` — registry-keyed synapse declaration.

The :class:`~braincell.mech.Junction` gap-junction declaration also
inherits from :class:`Point` but lives in its own module
:mod:`braincell.mech._junction`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping
import warnings

import brainunit as u
import numpy as np

from ._base import Mechanism
from ._params import Params, quantity_hashable
from ._validate import require_str

__all__ = [
    "Point",
    "CurrentClamp",
    "FunctionClamp",
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
    clamp-like mechanisms happens in :mod:`braincell._compute.layouts`,
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

    Raises
    ------
    TypeError
        If ``delay``, ``durations`` or ``amplitudes`` are not quantities.
    ValueError
        If any duration is non-positive.

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

    def __post_init__(self) -> None:
        for name, unit in (("delay", u.ms), ("durations", u.ms), ("amplitudes", u.nA)):
            coerced = _coerce_quantity(
                getattr(self, name),
                unit=unit,
                field_name=f"CurrentClamp.{name}",
            )
            object.__setattr__(self, name, coerced)
        _raise_if_nonpositive_duration(self.durations)


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

    #: Quantity field -> unit. ``phase`` is dimensionless and handled
    #: separately.
    _UNITS: ClassVar[Mapping[str, Any]] = {
        "amplitude": u.nA,
        "frequency": u.Hz,
        "offset": u.nA,
        "delay": u.ms,
        "duration": u.ms,
    }
    #: Fields that must additionally be strictly positive.
    _POSITIVE: ClassVar[tuple[str, ...]] = ("frequency", "duration")

    def __post_init__(self) -> None:
        if not isinstance(self.phase, (int, float)) or isinstance(self.phase, bool):
            raise TypeError(f"SineClamp.phase must be a real number, got {type(self.phase).__name__!r}.")
        for name, unit in self._UNITS.items():
            original = getattr(self, name)
            coerced = _coerce_quantity(
                original,
                unit=unit,
                field_name=f"SineClamp.{name}",
                allow_sequence=False,
            )
            if name in self._POSITIVE and float(coerced.to_decimal(unit)) <= 0.0:
                raise ValueError(f"SineClamp.{name} must be > 0, got {original!r}.")
            object.__setattr__(self, name, coerced)


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
            raise TypeError(f"FunctionClamp.fn must be callable, got {type(self.fn).__name__!r}.")


# ---------------------------------------------------------------------------
# Point spike sources, observers & synapses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LegacyProbe(Point):
    """Shared preamble for the deprecated probe declarations.

    The four probe classes each opened ``__post_init__`` with the same
    deprecation warning -- passing their own class name as a string
    literal, which a rename would silently desync -- followed by the same
    optional-``name`` check. Both now happen once, here, and each
    subclass implements only :meth:`_validate` for its own fields.
    """

    def __post_init__(self) -> None:
        warnings.warn(
            f"{type(self).__name__} is deprecated; use Cell.record(..., braincell.observe.*) for new code.",
            DeprecationWarning,
            stacklevel=3,
        )
        require_str(getattr(self, "name", None), type(self).__name__, "name", optional=True)
        self._validate()

    def _validate(self) -> None:
        """Validate subclass-specific fields. Overridden as needed."""


@dataclass(frozen=True)
class StateProbe(_LegacyProbe):
    """Probe for cell-owned state at one placed location."""

    name: str | None = None
    field: str = "v"

    def _validate(self) -> None:
        require_str(self.field, "StateProbe", "field")
        if self.field != "v":
            raise ValueError(f"Unsupported StateProbe field {self.field!r}; only 'v' is supported.")


@dataclass(frozen=True)
class MechanismProbe(_LegacyProbe):
    """Probe for runtime state on a named mechanism."""

    mechanism: str
    field: str
    name: str | None = None

    def _validate(self) -> None:
        require_str(self.mechanism, "MechanismProbe", "mechanism")
        require_str(self.field, "MechanismProbe", "field")


@dataclass(frozen=True)
class CurrentProbe(_LegacyProbe):
    """Probe for current at a placed location."""

    ion: str | None = None
    mechanism: str | None = None
    name: str | None = None

    def _validate(self) -> None:
        require_str(self.ion, "CurrentProbe", "ion", optional=True)
        require_str(self.mechanism, "CurrentProbe", "mechanism", optional=True)
        if self.ion is None and self.mechanism is None:
            raise ValueError("CurrentProbe requires at least one of 'ion' or 'mechanism'.")


@dataclass(frozen=True)
class ProbeMechanism(_LegacyProbe):
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

    def _validate(self) -> None:
        require_str(self.variable, "ProbeMechanism", "variable")
        if self.target is not None and not isinstance(self.target, str):
            raise TypeError(f"ProbeMechanism.target must be str or None, got {type(self.target).__name__!r}.")


class Synapse(Point):
    """Registry-keyed synapse declaration.

    Parameters
    ----------
    synapse_type : str
        Registry key for the target synapse class (currently ``"ExpSyn"`` or
        ``"Exp2Syn"``).
    name : str or None
        Optional instance label.
    **params
        Synapse parameters.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.mech import Synapse
        >>> syn = Synapse("ExpSyn")
        >>> syn.synapse_type
        'ExpSyn'
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
        require_str(synapse_type, "Synapse", "synapse_type")
        if "params" in params:
            raise TypeError("Synapse parameters must be passed as keyword arguments, not as params={...}.")
        require_str(name, "Synapse", "name", optional=True)
        object.__setattr__(self, "synapse_type", synapse_type)
        object.__setattr__(self, "params", Params(params) if params else Params())
        object.__setattr__(self, "name", name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot delete attribute {name!r}.")

    @property
    def instance_name(self) -> str:
        """Display label — ``self.name`` if set, else ``synapse_type``."""
        return self.name if self.name is not None else self.synapse_type

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.synapse_type == other.synapse_type and self.params == other.params and self.name == other.name

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.synapse_type, self.params, self.name))

    def __repr__(self) -> str:
        return f"Synapse(synapse_type={self.synapse_type!r}, params={self.params!r}, name={self.name!r})"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coerce_quantity(value: Any, *, unit: Any, field_name: str, allow_sequence: bool = True) -> Any:
    """Coerce ``value`` to a Quantity in ``unit``.

    With ``allow_sequence=False`` this is the former
    ``_coerce_scalar_quantity``, which was a strict subset of this
    function -- and which, despite its name, never checked scalar-ness.
    """
    if value is None:
        raise TypeError(f"{field_name} must not be None.")
    if hasattr(value, "to_decimal"):
        return value.in_unit(unit)
    if allow_sequence and isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{field_name} must be non-empty.")
        if not all(hasattr(item, "to_decimal") for item in value):
            raise TypeError(f"{field_name} entries must be Quantities.")
        decimals = [item.to_decimal(unit) for item in value]
        return u.Quantity(np.stack(decimals, axis=-1), unit)
    if not allow_sequence:
        raise TypeError(f"{field_name} must be a Quantity, got {value!r}.")
    raise TypeError(f"{field_name} must be a Quantity or sequence of Quantities, got {type(value).__name__!r}.")


def _raise_if_nonpositive_duration(value: Any) -> None:
    decimals = np.asarray(value.to_decimal(u.ms), dtype=float)
    if decimals.size == 0:
        raise ValueError("CurrentClamp.durations must be non-empty.")
    if np.any(decimals <= 0.0):
        raise ValueError(f"CurrentClamp.durations entries must be > 0, got {value!r}.")
