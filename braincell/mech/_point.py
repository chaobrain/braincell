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
- :class:`Synapse` — registry-keyed synapse declaration.

The :class:`~braincell.mech.Junction` gap-junction declaration also
inherits from :class:`Point` but lives in its own module
:mod:`braincell.mech._junction`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping

import brainunit as u
import numpy as np

from braincell.mech._base import Mechanism
from braincell.mech._density import _resolve_class_name
from braincell.mech._params import Params, quantity_hashable
from braincell.mech._units import declared_parameters_are_valid, warn_invalid_declared_parameters
from braincell.mech._validate import require_str

__all__ = [
    "Point",
    "CurrentClamp",
    "FunctionClamp",
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


class Synapse(Point):
    """Synapse declaration, by registry name or class.

    Parameters
    ----------
    synapse_type : str or type
        Registry name of the synapse class (such as ``"ExpSyn"``), or the
        class itself. An unregistered class is kept by identity: it is
        stored under the key :meth:`MechanismRegistry.key_for` returns.
    name : str or None
        Optional instance label.
    **params
        Synapse parameters.

    Warns
    -----
    FutureWarning
        When the synapse class is known at construction (a class object, or
        a registered name) and a literal parameter has the wrong unit or a
        parameter name is unknown (``TypeError`` at ``init_state``). The
        warning points at the declaring line and quotes the ``init_state``
        error, e.g. ``Synapse 'ExpSyn' parameter 'tau' requires a quantity
        compatible with ms; got 5.0 (no unit). This will raise TypeError at
        declaration from braincell 0.4.``. From braincell 0.4 the declaration
        raises instead.

    Notes
    -----
    Units are checked in ``Cell.init_state()`` against the runtime class's
    prototypes (signature defaults, or its ``parameter_units``): a bare
    number for a dimensioned parameter raises ``TypeError``, incompatible
    units raise ``ValueError``. The message names the synapse, the parameter,
    the expected unit, the value, the locset and the ``place`` line.

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
        synapse_type: str | type,
        /,
        *,
        name: str | None = None,
        **params: Any,
    ) -> None:
        synapse_type = _resolve_class_name("synapse", synapse_type)
        if "params" in params:
            raise TypeError("Synapse parameters must be passed as keyword arguments, not as params={...}.")
        require_str(name, "Synapse", "name", optional=True)
        object.__setattr__(self, "synapse_type", synapse_type)
        object.__setattr__(self, "params", Params(params) if params else Params())
        object.__setattr__(self, "name", name)
        warn_invalid_declared_parameters("synapse", synapse_type, self.instance_name, self.params)

    def _with_params(self, params: Mapping[str, Any], *, checked: bool) -> "Synapse":
        """This declaration with other parameter values, as ``Synapse(self.synapse_type, name=self.name, **params)``.

        ``checked=True`` skips the declaration check: the caller passes
        parts of values this declaration holds, and
        :meth:`_values_are_checked` is ``True`` (CELL-51, one declaration
        split per member).
        """
        if not checked:
            return type(self)(self.synapse_type, name=self.name, **params)
        split = object.__new__(type(self))
        object.__setattr__(split, "synapse_type", self.synapse_type)
        object.__setattr__(split, "params", Params(params) if params else Params())
        object.__setattr__(split, "name", self.name)
        return split

    def _values_are_checked(self) -> bool:
        """Whether every parameter is a literal value that passes its declaration check."""
        return declared_parameters_are_valid("synapse", self.synapse_type, self.instance_name, self.params)

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
