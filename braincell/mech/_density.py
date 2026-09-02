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

import operator
from typing import Any, Callable, ClassVar, Mapping

from ._base import Mechanism
from ._params import Params
from ._registry import _CATEGORY_CHANNEL, _CATEGORY_ION, get_registry
from ._validate import require_fraction, require_str

__all__ = [
    "Density",
    "Channel",
    "Ion",
]


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
        Parameter mapping, supplied by the concrete subclass from its own
        ``**params`` capture. This is an internal contract between
        :class:`Density` and its subclasses -- **callers pass parameters
        as keyword arguments**, and ``Channel(..., params={...})`` is
        rejected with :exc:`TypeError` rather than silently creating a
        parameter named ``"params"``.
    name : str or None
        Optional instance label. When ``None``, :attr:`class_name` is
        used as the display label (see :attr:`instance_name`).
    coverage_area_fraction : float
        Fraction in ``[0, 1]`` of the target control volume's lateral
        area covered by this declaration. Set by the paint lowering pass
        in :mod:`braincell._discretization` when a paint region only
        partially overlaps a CV. Defaults to ``1.0``.
    solver : str or Callable or None
        Optional declaration-local integrator override. It is applied only
        to Markov channels and kinetic ions and must be supplied together
        with ``substeps``. ``None`` inherits the enclosing Cell schedule.
    substeps : int or None
        Optional declaration-local substep count, paired with ``solver``.

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

    __slots__ = (
        "class_name",
        "params",
        "name",
        "coverage_area_fraction",
        "solver",
        "substeps",
    )

    #: Category discriminator, set by concrete subclasses to ``"channel"``
    #: or ``"ion"``. Instances of the abstract base have an empty string.
    category: ClassVar[str] = ""

    #: Field names in declaration order, derived once per subclass from
    #: ``__slots__`` along the MRO. See :meth:`__init_subclass__`.
    _FIELDS: ClassVar[tuple[str, ...]] = ()

    #: Per-field coercions applied to every *incoming* value, whether it
    #: arrives through ``__init__`` or through :meth:`_replace`. Stored
    #: values are already normalized, so they are never re-coerced. Each
    #: entry is called as ``normalizer(value, owner_class_name)``.
    _NORMALIZERS: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        "params": lambda value, owner: Params.coerce(value),
        "coverage_area_fraction": lambda value, owner: require_fraction(value, owner, "coverage_area_fraction"),
    }

    #: Per-field ``repr`` overrides for fields whose stored form differs
    #: from the form the constructor accepts.
    _REPRS: ClassVar[Mapping[str, Callable[[Any], str]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        names: list[str] = []
        for klass in reversed(cls.__mro__):
            for slot in getattr(klass, "__slots__", ()):
                if slot not in names:
                    names.append(slot)
        cls._FIELDS = tuple(names)

    def __init__(
        self,
        class_name: Any,
        /,
        *,
        params: Any = None,
        name: str | None = None,
        coverage_area_fraction: float = 1.0,
        solver: str | Callable | None = None,
        substeps: int | None = None,
    ) -> None:
        cls = type(self)
        if not cls.category:
            raise TypeError(f"{cls.__name__} is an abstract base; instantiate Channel or Ion instead.")
        if isinstance(params, Mapping) and "params" in params:
            raise TypeError(
                f"{cls.__name__} parameters must be passed as keyword arguments, not as params={{...}}. "
                f"Write {cls.__name__}(..., g_max=value) rather than "
                f"{cls.__name__}(..., params={{'g_max': value}})."
            )
        resolved = _resolve_class_name(cls.category, class_name)
        require_str(name, cls.__name__, "name", optional=True)
        solver, substeps = _normalize_integration_override(
            solver,
            substeps,
            owner=cls.__name__,
        )
        object.__setattr__(self, "class_name", resolved)
        object.__setattr__(self, "params", Params.coerce(params))
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "coverage_area_fraction",
            require_fraction(coverage_area_fraction, cls.__name__, "coverage_area_fraction"),
        )
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "substeps", substeps)

    # ------------------------------------------------------------------
    # immutability
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot delete attribute {name!r}.")

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def instance_name(self) -> str:
        """Display label for this declaration.

        Returns ``self.name`` when set, otherwise ``self.class_name``.
        """
        return self.name if self.name is not None else self.class_name

    # ------------------------------------------------------------------
    # equality / hashing: structural, type-exact
    #
    # All three dunders below, and ``_replace``, walk ``_FIELDS`` rather
    # than naming the fields. A subclass therefore declares ``__slots__``
    # and nothing else -- it does not re-implement any of them.
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        # ``params`` is the only field whose comparison walks contents
        # (and may hold arrays), so let the cheap scalars reject first.
        for field in type(self)._FIELDS:
            if field != "params" and getattr(self, field) != getattr(other, field):
                return False
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self).__name__, *(getattr(self, field) for field in type(self)._FIELDS)))

    def __repr__(self) -> str:
        cls = type(self)
        fields = ", ".join(f"{field}={cls._REPRS.get(field, repr)(getattr(self, field))}" for field in cls._FIELDS)
        return f"{cls.__name__}({fields})"

    # ------------------------------------------------------------------
    # non-mutating updates
    # ------------------------------------------------------------------

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

        Raises
        ------
        ValueError
            If ``fraction`` lies outside ``[0, 1]`` -- the same check
            :meth:`__init__` applies, reached through the shared
            normalization in :meth:`_replace`.
        """
        return self._replace(coverage_area_fraction=fraction)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _replace(self, **updates: Any) -> "Density":
        """Return a copy with selected fields overridden.

        Constructs a new instance of the exact concrete subclass via
        :meth:`object.__new__`, bypassing ``__init__`` so unchanged
        fields are not re-validated. Fields that *are* being replaced go
        through the same :attr:`_NORMALIZERS` entry ``__init__`` uses, so
        a copy cannot hold a value the constructor would have rejected.
        """
        cls = type(self)
        new = object.__new__(cls)
        for field in cls._FIELDS:
            if field in updates:
                normalizer = cls._NORMALIZERS.get(field)
                value = updates[field]
                value = normalizer(value, cls.__name__) if normalizer is not None else value
            else:
                value = getattr(self, field)
            object.__setattr__(new, field, value)
        return new


class Channel(Density):
    """Distributed ion-channel declaration.

    Parameters
    ----------
    class_name : str or type
        Registry key for the target channel class (e.g. ``"IL"``,
        ``"Na_HH1952"``, or ``"leaky"`` via an alias), or a class
        object such as ``braincell.channel.IL``.
    name : str or None
        Optional instance label. See
        :attr:`Density.instance_name`.
    coverage_area_fraction : float
        Fraction in ``[0, 1]`` of the target CV's lateral area this
        declaration covers. Callers rarely set this directly — it is
        typically computed by the paint lowering pass.
    solver : str or Callable or None
        Optional Markov integration override. Must be paired with
        ``substeps``; ``None`` inherits the enclosing Cell schedule.
    substeps : int or None
        Optional Markov substep count, paired with ``solver``.
    **params
        Channel parameters, passed as keyword arguments with
        ``brainunit`` quantity values (e.g. ``g_max=0.1 * u.mS /
        u.cm ** 2``, ``E=-70 * u.mV``). A parameter may also be a
        callable accepting one :class:`braincell.mech.CVContext`; it is
        resolved once per active CV during ``Cell.init_state()``.

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

    __slots__ = ("ion_name", "ion_names")
    category: ClassVar[str] = _CATEGORY_CHANNEL
    _NORMALIZERS: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **Density._NORMALIZERS,
        "ion_names": lambda value, owner: _normalize_ion_names(value),
    }
    #: ``ion_names`` is stored as a sorted tuple of pairs but accepted as
    #: a mapping, so echo the constructor's form.
    _REPRS: ClassVar[Mapping[str, Callable[[Any], str]]] = {
        "ion_names": lambda value: repr(dict(value) if value is not None else None),
    }

    def __init__(
        self,
        class_name: Any,
        /,
        *,
        name: str | None = None,
        coverage_area_fraction: float = 1.0,
        ion_name: str | None = None,
        ion_names: Mapping[str, str] | None = None,
        solver: str | Callable | None = None,
        substeps: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            class_name,
            params=Params(params) if params else None,
            name=name,
            coverage_area_fraction=coverage_area_fraction,
            solver=solver,
            substeps=substeps,
        )
        require_str(ion_name, "Channel", "ion_name", optional=True)
        normalized_ion_names = _normalize_ion_names(ion_names)
        if ion_name is not None and normalized_ion_names is not None:
            raise ValueError("Channel cannot define both ion_name and ion_names.")
        object.__setattr__(self, "ion_name", ion_name)
        object.__setattr__(self, "ion_names", normalized_ion_names)


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
    solver : str or Callable or None
        Optional kinetic-ion integration override. Must be paired with
        ``substeps``; ``None`` inherits the enclosing Cell schedule.
    substeps : int or None
        Optional kinetic-ion substep count, paired with ``solver``.
    **params
        Ion parameters, passed as keyword arguments. A parameter may also
        be a callable accepting one :class:`braincell.mech.CVContext`; it is
        resolved once per active CV during ``Cell.init_state()``.

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
        >>> import brainunit as u
        >>> Ion("SodiumFixed", Ci=12.0 * u.mM).category
        'ion'

        >>> # Class-object form
        >>> spec = Ion(braincell.ion.PotassiumFixed)
        >>> spec.class_name
        'PotassiumFixed'
    """

    __slots__ = ()
    category: ClassVar[str] = _CATEGORY_ION

    def __init__(
        self,
        class_name: Any,
        /,
        *,
        name: str | None = None,
        coverage_area_fraction: float = 1.0,
        solver: str | Callable | None = None,
        substeps: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            class_name,
            params=Params(params) if params else None,
            name=name,
            coverage_area_fraction=coverage_area_fraction,
            solver=solver,
            substeps=substeps,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _normalize_integration_override(
    solver: str | Callable | None,
    substeps: int | None,
    *,
    owner: str,
) -> tuple[str | Callable | None, int | None]:
    if (solver is None) != (substeps is None):
        raise ValueError(f"{owner}.solver and {owner}.substeps must be provided together or both be None.")
    if solver is None:
        return None, None
    if not isinstance(solver, str) and not callable(solver):
        raise TypeError(f"{owner}.solver must be a non-empty string, callable, or None, got {type(solver).__name__!r}.")
    if isinstance(solver, str) and not solver:
        raise ValueError(f"{owner}.solver must be a non-empty string.")
    try:
        hash(solver)
    except TypeError as exc:
        raise TypeError(f"{owner}.solver callable must be hashable.") from exc
    if isinstance(substeps, bool):
        raise TypeError(f"{owner}.substeps must be an integer, got bool.")
    try:
        normalized_substeps = operator.index(substeps)
    except TypeError as exc:
        raise TypeError(f"{owner}.substeps must be an integer, got {type(substeps).__name__!r}.") from exc
    if normalized_substeps < 1:
        raise ValueError(f"{owner}.substeps must be at least 1, got {normalized_substeps!r}.")
    return solver, normalized_substeps


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
            raise ValueError("class_name must be a non-empty string.")
        return value
    if isinstance(value, type):
        reg = get_registry()
        for entry_name, entry_cls in reg.items(category):
            if entry_cls is value:
                return entry_name
        return value.__name__
    raise TypeError(f"class_name must be a string or class, got {type(value).__name__!r}.")


def _normalize_ion_names(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"Channel.ion_names must be a mapping or None, got {type(value).__name__!r}.")
    normalized: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"Channel.ion_names keys must be non-empty strings, got {key!r}.")
        if not isinstance(item, str) or not item:
            raise TypeError(f"Channel.ion_names values must be non-empty strings, got {item!r}.")
        normalized.append((key, item))
    return tuple(sorted(normalized))
