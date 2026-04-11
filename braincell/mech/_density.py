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

"""Distributed (density) mechanism declarations.

This module defines the declaration-layer hierarchy for distributed
mechanisms — mechanisms that are painted over a region of a cell rather
than attached to a single location:

- :class:`Density` is the abstract base that carries the structural
  fields shared by both flavours (``class_name``, ``params``, ``name``,
  ``coverage_area_fraction``) plus equality / hashing / immutability.
- :class:`Channel` is the concrete subclass for ion-channel
  declarations (``category == "channel"``).
- :class:`Ion` is the concrete subclass for ion-species declarations
  (``category == "ion"``).

Both :class:`Channel` and :class:`Ion` accept either a registry key
string (``"IL"``, ``"SodiumFixed"``) or a concrete class object (e.g.
``braincell.ion.PotassiumFixed``). When a class object is passed, its
canonical registry name is resolved via :mod:`braincell.mech._registry`
at construction time — so whatever form the user passes, the stored
``class_name`` is always a plain string.

This module is deliberately decoupled from any runtime state. Runtime
lookup of the concrete class happens later, during ``Cell`` compile,
via :func:`braincell.mech.get_registry().get(category, class_name)
<braincell.mech.MechanismRegistry.get>`.
"""

from typing import Any, ClassVar

from ._base import Mechanism
from ._params import Params

__all__ = [
    "Channel",
    "Density",
    "Ion",
]


_CHANNEL = "channel"
_ION = "ion"


class Density(Mechanism):
    """Base class for distributed mechanism declarations.

    :class:`Density` is the shared skeleton for :class:`Channel` and
    :class:`Ion`. It stores the declaration fields (``class_name``,
    ``params``, ``name``, ``coverage_area_fraction``) and implements
    equality / hashing / immutability, but it is abstract in the sense
    that ``category`` is only set by the concrete subclasses. Attempting
    to construct ``Density`` directly raises ``TypeError``.

    Parameters
    ----------
    class_name : str or type
        Registry key for the concrete runtime class, or a class object
        whose canonical name the registry already knows (e.g.
        ``braincell.ion.PotassiumFixed``). A type argument is resolved
        through :func:`braincell.mech.get_registry` at construction
        time; the stored :attr:`class_name` is always a string.
    params : Mapping or None
        Parameter mapping. ``None`` (the default) produces an empty
        :class:`Params`.
    name : str or None
        Optional instance label. When ``None``, :attr:`class_name` is
        used as the display label (see :attr:`instance_name`).
    coverage_area_fraction : float
        Fraction in ``[0, 1]`` of the target control volume's lateral
        area covered by this declaration. Set by
        :func:`braincell.cv._mech._scale_density_for_coverage` when a
        paint region only partially overlaps a CV. Defaults to ``1.0``.

    Raises
    ------
    TypeError
        If ``Density`` itself is instantiated directly (the concrete
        subclass determines ``category``), or if ``class_name`` is
        neither a string nor a class, or if ``name`` is not a string or
        ``None``.
    ValueError
        If ``class_name`` resolves to an empty string, or if
        ``coverage_area_fraction`` is outside ``[0, 1]``.

    See Also
    --------
    Channel : Concrete subclass for ion channels.
    Ion : Concrete subclass for ion species.
    """

    __slots__ = ("class_name", "params", "name", "coverage_area_fraction")

    #: Category discriminator, set by concrete subclasses to ``"channel"``
    #: or ``"ion"``. Instances of the abstract base have an empty string.
    category: ClassVar[str] = ""

    def __init__(
        self,
        class_name: Any,
        /,
        *,
        params: Any = None,
        name: str | None = None,
        coverage_area_fraction: float = 1.0,
    ) -> None:
        cls = type(self)
        if cls is Density or not cls.category:
            raise TypeError(
                f"{cls.__name__} is an abstract base; instantiate "
                f"Channel or Ion instead."
            )
        resolved = _resolve_class_name(cls.category, class_name)
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"{cls.__name__}.name must be a string or None, "
                f"got {type(name).__name__!r}."
            )
        coverage = float(coverage_area_fraction)
        if not (0.0 <= coverage <= 1.0):
            raise ValueError(
                f"{cls.__name__}.coverage_area_fraction must lie in "
                f"[0, 1], got {coverage!r}."
            )
        object.__setattr__(self, "class_name", resolved)
        object.__setattr__(self, "params", Params.coerce(params))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coverage_area_fraction", coverage)

    # ------------------------------------------------------------------
    # immutability
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot set "
            f"attribute {name!r}."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot delete "
            f"attribute {name!r}."
        )

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def instance_name(self) -> str:
        """Display label for this declaration.

        Returns ``self.name`` when set, otherwise ``self.class_name``.
        """
        return self.name if self.name is not None else self.class_name

    @property
    def identity(self) -> tuple[str, str]:
        """Return ``(instance_name, class_name)`` for table views."""
        return (self.instance_name, self.class_name)

    # ------------------------------------------------------------------
    # equality / hashing: structural, type-exact
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.class_name == other.class_name
            and self.params == other.params
            and self.name == other.name
            and self.coverage_area_fraction == other.coverage_area_fraction
        )

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(
            (
                type(self).__name__,
                self.class_name,
                self.params,
                self.name,
                self.coverage_area_fraction,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"class_name={self.class_name!r}, "
            f"params={self.params!r}, "
            f"name={self.name!r}, "
            f"coverage_area_fraction={self.coverage_area_fraction!r})"
        )

    # ------------------------------------------------------------------
    # non-mutating updates
    # ------------------------------------------------------------------

    def with_params(self, **updates: Any) -> "Density":
        """Return a copy with ``updates`` merged into ``params``.

        Parameters
        ----------
        **updates
            Parameters to add or replace.

        Returns
        -------
        Density
            A new instance of the same concrete subclass. ``self`` is
            unchanged.
        """
        return self._replace(params=self.params.with_updates(**updates))

    def with_coverage(self, fraction: float) -> "Density":
        """Return a copy with a new ``coverage_area_fraction``.

        Parameters
        ----------
        fraction : float
            New fraction, must lie in ``[0, 1]``.

        Returns
        -------
        Density
            A new instance of the same concrete subclass. ``self`` is
            unchanged.
        """
        return self._replace(coverage_area_fraction=float(fraction))

    def with_name(self, name: str | None) -> "Density":
        """Return a copy with a new instance label."""
        return self._replace(name=name)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _replace(self, **updates: Any) -> "Density":
        """Return a copy with selected fields overridden.

        Constructs a new instance of the exact concrete subclass via
        :meth:`object.__new__`, bypassing ``__init__`` so callers do
        not have to re-validate unchanged fields.
        """
        new = object.__new__(type(self))
        object.__setattr__(new, "class_name", updates.get("class_name", self.class_name))
        object.__setattr__(
            new, "params", Params.coerce(updates.get("params", self.params))
        )
        object.__setattr__(new, "name", updates.get("name", self.name))
        object.__setattr__(
            new,
            "coverage_area_fraction",
            float(
                updates.get("coverage_area_fraction", self.coverage_area_fraction)
            ),
        )
        return new


class Channel(Density):
    """Distributed ion-channel declaration.

    Parameters
    ----------
    class_name : str or type
        Registry key for the target channel class (e.g. ``"IL"``,
        ``"INa_HH1952"``, or ``"leaky"`` via an alias), or a class
        object such as ``braincell.channel.IL``.
    name : str or None
        Optional instance label. See
        :attr:`Density.instance_name`.
    coverage_area_fraction : float
        Fraction in ``[0, 1]`` of the target CV's lateral area this
        declaration covers. Callers rarely set this directly — it is
        typically computed by the paint lowering pass.
    **params
        Channel parameters, passed as keyword arguments with
        ``brainunit`` quantity values (e.g. ``g_max=0.1 * u.mS /
        u.cm ** 2``, ``E=-70 * u.mV``).

    See Also
    --------
    Ion : Ion-species counterpart.
    braincell.mech.register_channel : Registration decorator for
        channel classes.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import braincell
        >>> from braincell.mech import Channel
        >>> Channel("IL", g_max=0.1 * u.mS / u.cm ** 2, E=-70 * u.mV).category
        'channel'

        >>> # Passing a class object works too, as long as the class is
        >>> # already registered (i.e. ``braincell.channel`` is imported).
        >>> spec = Channel(braincell.channel.IL, g_max=0.1 * u.mS / u.cm ** 2)
        >>> spec.class_name
        'IL'
    """

    __slots__ = ()
    category: ClassVar[str] = _CHANNEL

    def __init__(
        self,
        class_name: Any,
        /,
        *,
        name: str | None = None,
        coverage_area_fraction: float = 1.0,
        **params: Any,
    ) -> None:
        super().__init__(
            class_name,
            params=Params(params) if params else None,
            name=name,
            coverage_area_fraction=coverage_area_fraction,
        )


class Ion(Density):
    """Distributed ion-species declaration.

    Parameters
    ----------
    class_name : str or type
        Registry key for the target ion class (e.g. ``"SodiumFixed"``,
        ``"CalciumDetailed"``), or a class object such as
        ``braincell.ion.PotassiumFixed``.
    name : str or None
        Optional instance label.
    coverage_area_fraction : float
        Fraction in ``[0, 1]`` of the target CV's lateral area this
        declaration covers.
    **params
        Ion parameters, passed as keyword arguments.

    See Also
    --------
    Channel : Channel counterpart.
    braincell.mech.register_ion : Registration decorator for ion
        classes.

    Examples
    --------

    .. code-block:: python

        >>> import braincell
        >>> from braincell.mech import Ion
        >>> Ion("SodiumFixed", c0=12.0).category
        'ion'

        >>> # Class-object form
        >>> spec = Ion(braincell.ion.PotassiumFixed)
        >>> spec.class_name
        'PotassiumFixed'
    """

    __slots__ = ()
    category: ClassVar[str] = _ION

    def __init__(
        self,
        class_name: Any,
        /,
        *,
        name: str | None = None,
        coverage_area_fraction: float = 1.0,
        **params: Any,
    ) -> None:
        super().__init__(
            class_name,
            params=Params(params) if params else None,
            name=name,
            coverage_area_fraction=coverage_area_fraction,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_class_name(category: str, value: Any) -> str:
    """Normalise a ``class_name`` argument to a string.

    Accepts either a plain string (returned after a non-empty check) or
    a class object. For class objects, the canonical registry name is
    preferred when the class is already registered; otherwise we fall
    back to ``cls.__name__``. The string form is what gets stored on
    the declaration and later consumed by
    :meth:`braincell.mech.MechanismRegistry.get`.
    """
    if isinstance(value, str):
        if not value:
            raise ValueError(
                "class_name must be a non-empty string."
            )
        return value
    if isinstance(value, type):
        # Local import to avoid a hard import cycle between _density
        # and _registry at module load time.
        from ._registry import get_registry

        reg = get_registry()
        for entry_name, entry_cls in reg.items(category):
            if entry_cls is value:
                return entry_name
        return value.__name__
    raise TypeError(
        f"class_name must be a string or class, got "
        f"{type(value).__name__!r}."
    )
