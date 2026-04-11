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
mechanisms share :class:`PointMechanism` as a common base class so that
downstream consumers can dispatch on
``isinstance(x, PointMechanism)``.

Concrete point mechanisms:

- :class:`CurrentClamp` — piecewise-constant current injection.
- :class:`SineClamp` — sinusoidal current injection.
- :class:`FunctionClamp` — arbitrary ``t → I`` callable.
- :class:`ProbeMechanism` — recorder for a named variable.
- :class:`SynapseMechanism` — registry-keyed synapse model.
- :class:`GapJunctionMechanism` — gap-junction coupling placeholder.

The ergonomic factory :func:`Synapse` builds a
:class:`SynapseMechanism` the same way :func:`braincell.mech.Channel`
builds a :class:`DensityMechanism`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import brainunit as u

from ._params import Params

__all__ = [
    "CurrentClamp",
    "FunctionClamp",
    "GapJunctionMechanism",
    "PointMechanism",
    "ProbeMechanism",
    "SineClamp",
    "Synapse",
    "SynapseMechanism",
]


class PointMechanism:
    """Marker base class for point-located mechanism declarations.

    All concrete subclasses are frozen :func:`~dataclasses.dataclass`
    types — this base exists solely so that consumers can write
    ``isinstance(x, PointMechanism)`` instead of maintaining a parallel
    tuple of concrete types.

    :class:`PointMechanism` defines no abstract methods. Runtime
    evaluation of clamp-like mechanisms happens in
    :mod:`braincell.compute._runtime`, which inspects concrete
    subclasses directly.
    """

    __slots__ = ()


# ---------------------------------------------------------------------------
# Current injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurrentClamp(PointMechanism):
    """Piecewise-constant current clamp.

    The canonical form is a multi-segment step protocol. Users with a
    simple single-step clamp can use :meth:`step` for a cleaner
    constructor.

    Parameters
    ----------
    start : Quantity[ms]
        Absolute simulation time at which the first segment begins.
    durations : tuple of Quantity[ms]
        Duration of each segment in order. Must be non-empty and all
        entries must be strictly positive.
    amplitudes : tuple of Quantity[nA]
        Amplitude held during each segment. Must have the same length
        as ``durations``.

    Raises
    ------
    TypeError
        If ``durations`` or ``amplitudes`` are not tuples of
        quantities.
    ValueError
        If the two sequences have different lengths, if either is
        empty, or if any duration is non-positive.

    See Also
    --------
    CurrentClamp.step : Single-step convenience constructor.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.mech import CurrentClamp
        >>> # Single step, starting at t = 10 ms, 50 ms long
        >>> cc = CurrentClamp.step(0.2 * u.nA, 50 * u.ms, delay=10 * u.ms)

        >>> # Multi-step protocol: ramp up then hold
        >>> cc = CurrentClamp(
        ...     start=0.0 * u.ms,
        ...     durations=(20 * u.ms, 30 * u.ms, 50 * u.ms),
        ...     amplitudes=(0.0 * u.nA, 0.1 * u.nA, 0.3 * u.nA),
        ... )
    """

    start: Any
    durations: tuple
    amplitudes: tuple

    def __post_init__(self) -> None:
        durations = _normalize_quantity_tuple(
            self.durations, unit=u.ms, field_name="durations"
        )
        amplitudes = _normalize_quantity_tuple(
            self.amplitudes, unit=u.nA, field_name="amplitudes"
        )
        if len(durations) == 0:
            raise ValueError("CurrentClamp.durations must be non-empty.")
        if len(durations) != len(amplitudes):
            raise ValueError(
                f"CurrentClamp.durations and amplitudes must have the "
                f"same length; got {len(durations)} and {len(amplitudes)}."
            )
        for item in durations:
            if float(item.to_decimal(u.ms)) <= 0.0:
                raise ValueError(
                    f"CurrentClamp durations must be > 0, got {item!r}."
                )

        start = _coerce_scalar_quantity(
            self.start, unit=u.ms, field_name="start"
        )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "amplitudes", amplitudes)

    @classmethod
    def step(
        cls,
        amplitude: Any,
        duration: Any,
        *,
        delay: Any = 0.0 * u.ms,
    ) -> "CurrentClamp":
        """Build a single-step clamp.

        Parameters
        ----------
        amplitude : Quantity[nA]
            Current held during the step.
        duration : Quantity[ms]
            Step duration.
        delay : Quantity[ms]
            Absolute start time; aliased to
            :attr:`CurrentClamp.start`.

        Returns
        -------
        CurrentClamp
            A one-segment clamp.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell.mech import CurrentClamp
            >>> cc = CurrentClamp.step(0.2 * u.nA, 50 * u.ms, delay=10 * u.ms)
            >>> cc.start.to_decimal(u.ms)
            10.0
        """
        return cls(
            start=delay,
            durations=(duration,),
            amplitudes=(amplitude,),
        )


@dataclass(frozen=True)
class SineClamp(PointMechanism):
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
    start : Quantity[ms]
        Absolute start time.
    duration : Quantity[ms]
        Length of the active window. The clamp returns zero before
        ``start`` and after ``start + duration``.
    """

    amplitude: Any
    frequency: Any
    phase: float = 0.0
    offset: Any = 0.0 * u.nA
    start: Any = 0.0 * u.ms
    duration: Any = 1.0 * u.ms


@dataclass(frozen=True)
class FunctionClamp(PointMechanism):
    """Arbitrary-callable current clamp.

    Parameters
    ----------
    fn : Callable
        A function ``f(local_t) -> Quantity[nA]`` called each step with
        the simulation time relative to ``start``.
    start : Quantity[ms]
        Absolute start time.
    duration : Quantity[ms]
        Length of the active window; outside this range ``current`` is
        zero.

    Notes
    -----
    Equality and hashing follow identity on ``fn`` (frozen dataclass
    auto-generated dunder methods compare lambdas by ``==``, which
    falls back to identity). Two :class:`FunctionClamp` instances built
    from two separate ``lambda`` definitions with identical bodies are
    considered distinct.
    """

    fn: Callable
    start: Any = 0.0 * u.ms
    duration: Any = 1.0 * u.ms


# ---------------------------------------------------------------------------
# Observers & couplings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeMechanism(PointMechanism):
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


@dataclass(frozen=True)
class SynapseMechanism(PointMechanism):
    """Registry-keyed synapse declaration.

    Parameters
    ----------
    synapse_type : str
        Registry key for the target synapse class (e.g. ``"AMPA"``,
        ``"NMDA"``).
    params : Params or Mapping
        Synapse parameters.
    name : str or None
        Optional instance label.

    See Also
    --------
    Synapse : Ergonomic factory.
    """

    synapse_type: str
    params: Params = field(default_factory=Params)
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.synapse_type, str) or not self.synapse_type:
            raise ValueError(
                f"SynapseMechanism.synapse_type must be a non-empty "
                f"string, got {self.synapse_type!r}."
            )
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(
                f"SynapseMechanism.name must be a string or None, "
                f"got {type(self.name).__name__!r}."
            )
        object.__setattr__(self, "params", Params.coerce(self.params))

    @property
    def instance_name(self) -> str:
        """Display label — ``self.name`` if set, else ``synapse_type``."""
        return self.name if self.name is not None else self.synapse_type

    @property
    def identity(self) -> tuple[str, str]:
        """Return ``(instance_name, synapse_type)`` for table views."""
        return (self.instance_name, self.synapse_type)


@dataclass(frozen=True)
class GapJunctionMechanism(PointMechanism):
    """Gap-junction coupling declaration (placeholder).

    The current implementation records only a parameter bundle. A
    future revision should add a ``partner`` locset/cell handle so
    multi-cell gap junctions can be expressed end-to-end.
    """

    params: Params = field(default_factory=Params)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", Params.coerce(self.params))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def Synapse(
    class_name: str,
    /,
    *,
    name: str | None = None,
    **params: Any,
) -> SynapseMechanism:
    """Build a :class:`SynapseMechanism` with keyword parameters.

    Parameters
    ----------
    class_name : str
        Registry key for the target synapse class (e.g. ``"AMPA"``).
    name : str or None
        Optional instance label.
    **params
        Synapse parameters.

    Returns
    -------
    SynapseMechanism

    Examples
    --------

    .. code-block:: python

        >>> from braincell.mech import Synapse
        >>> syn = Synapse("AMPA", tau_rise=0.5, tau_decay=5.0)
        >>> syn.synapse_type
        'AMPA'
    """
    return SynapseMechanism(
        synapse_type=class_name,
        params=Params(params),
        name=name,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coerce_scalar_quantity(value: Any, *, unit: Any, field_name: str) -> Any:
    if not hasattr(value, "to_decimal"):
        raise TypeError(
            f"CurrentClamp.{field_name} must be a Quantity, got {value!r}."
        )
    return value.in_unit(unit)


def _normalize_quantity_tuple(
    values: Any, *, unit: Any, field_name: str
) -> tuple:
    """Coerce ``values`` into a tuple of same-unit quantities.

    Accepts either a single quantity (wrapped into a 1-tuple), a
    tuple/list of quantities, or a vectorized quantity with a
    non-scalar shape.
    """
    if values is None:
        raise TypeError(f"CurrentClamp.{field_name} must not be None.")

    # Already a tuple/list of quantities.
    if isinstance(values, (list, tuple)):
        normalized: list = []
        for item in values:
            if not hasattr(item, "to_decimal"):
                raise TypeError(
                    f"CurrentClamp.{field_name} entries must be "
                    f"Quantities, got {item!r}."
                )
            normalized.append(item.in_unit(unit))
        return tuple(normalized)

    # Vectorized Quantity (shape != ()) → unpack to scalar quantities.
    shape = getattr(values, "shape", ())
    if hasattr(values, "to_decimal") and shape != ():
        decimals = values.to_decimal(unit)
        return tuple(item * unit for item in decimals.reshape((-1,)).tolist())

    # Single scalar quantity → 1-tuple.
    if hasattr(values, "to_decimal"):
        return (values.in_unit(unit),)

    raise TypeError(
        f"CurrentClamp.{field_name} must be a Quantity or sequence of "
        f"Quantities, got {type(values).__name__!r}."
    )
