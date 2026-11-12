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

"""Mechanism registry for :mod:`braincell.mech`.

This module implements the single lookup table that maps a short string
name like ``"IL"`` or ``"Na_HH1952"`` to a concrete runtime class in
:mod:`braincell.channel`, :mod:`braincell.ion`, :mod:`braincell.synapse`,
or :mod:`braincell.plasticity`.

Registration happens at class definition time via decorators such as
:func:`register_channel`, :func:`register_ion`, :func:`register_synapse`,
and :func:`register_connection_rule`, which simply insert the class into the
module-level :data:`_REGISTRY` singleton as a side effect of importing
the module that defines the class. After
``import braincell.channel`` has been executed, the registry is fully
populated and callers can look up classes by name through
:func:`get_registry`.

The registry is intentionally small and synchronous. It has no lazy
loading, no plugin entry points, no YAML manifests — registration is a
Python decorator side effect. This keeps provenance grep-able and keeps
import-time behavior predictable.

Examples
--------

.. code-block:: python

    >>> from braincell.mech import get_registry, register_channel
    >>> from braincell import channel
    >>> reg = get_registry()
    >>> reg.get("channel", "IL") is channel.IL
    True
    >>> reg.get("channel", "leaky") is channel.IL  # alias
    True
    >>> "Na_HH1952" in reg.names("channel")
    True
"""

import difflib
from dataclasses import dataclass, field
from typing import Callable, TypeVar

__all__ = [
    "MechanismEntry",
    "MechanismRegistry",
    "get_registry",
    "register_channel",
    "register_connection_rule",
    "register_ion",
    "register_synapse",
]


_CATEGORY_CHANNEL = "channel"
_CATEGORY_ION = "ion"
_CATEGORY_SYNAPSE = "synapse"
_CATEGORY_CONNECTION_RULE = "connection_rule"

_VALID_CATEGORIES = frozenset({_CATEGORY_CHANNEL, _CATEGORY_ION, _CATEGORY_SYNAPSE, _CATEGORY_CONNECTION_RULE})


_T = TypeVar("_T", bound=type)


@dataclass(frozen=True)
class MechanismEntry:
    """One entry in the :class:`MechanismRegistry`.

    Parameters
    ----------
    category : str
        One of ``"channel"``, ``"ion"``, ``"synapse"``, or
        ``"connection_rule"``.
    name : str
        The canonical registry key. Must be unique within a category.
    cls : type
        The concrete runtime class this entry resolves to.
    aliases : tuple of str
        Alternate names that also resolve to ``cls``. Aliases must not
        collide with any other canonical name or alias in the same
        category.
    """

    category: str
    name: str
    cls: type
    aliases: tuple[str, ...] = field(default_factory=tuple)


class MechanismRegistry:
    """Name → class lookup for mechanism classes.

    A single module-level instance is exposed through
    :func:`get_registry`. User code should usually not instantiate
    :class:`MechanismRegistry` directly — it is public only so
    downstream libraries can build their own isolated registries for
    testing.

    Entries are keyed by ``(category, name)``. Each category has its
    own flat namespace, so ``("channel", "IL")`` and
    ``("ion", "IL")`` would be two independent entries (in practice
    ``IL`` only lives in the channel namespace).

    The registry is **not thread-safe**. Register classes at module
    import time only.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], MechanismEntry] = {}
        self._aliases: dict[tuple[str, str], str] = {}
        # Unregistered classes used in declarations, by identity: key -> class
        # and class -> key. Kept apart from the named entries, so ``names`` and
        # ``items`` list registered mechanisms only.
        self._anonymous: dict[tuple[str, str], type] = {}
        self._anonymous_keys: dict[tuple[str, type], str] = {}

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------

    def register(self, entry: MechanismEntry) -> None:
        """Register a new :class:`MechanismEntry`.

        Parameters
        ----------
        entry : MechanismEntry
            The entry to install.

        Raises
        ------
        ValueError
            If ``entry.category`` is not a valid category, if the
            canonical name is already taken in that category, or if any
            alias collides with an existing canonical name or alias in
            the same category.
        """
        _check_category(entry.category)
        key = (entry.category, entry.name)
        for name in (entry.name, *entry.aliases):
            used = self._anonymous.get((entry.category, name))
            if used is not None and used is not entry.cls:
                raise ValueError(
                    f"Cannot register {entry.category!r}/{name!r}: an unregistered class "
                    f"{used!r} has already been declared under that name."
                )
        if key in self._entries:
            raise ValueError(
                f"Mechanism {entry.category!r}/{entry.name!r} is already "
                f"registered (existing class: {self._entries[key].cls!r})."
            )
        if key in self._aliases:
            raise ValueError(
                f"Cannot register {entry.category!r}/{entry.name!r}: name "
                f"is already used as an alias for "
                f"{self._aliases[key]!r}."
            )

        # Install the entry first so partial failures in the alias pass
        # can roll back cleanly.
        self._entries[key] = entry
        installed_aliases: list[tuple[str, str]] = []
        try:
            for alias in entry.aliases:
                alias_key = (entry.category, alias)
                if alias_key in self._entries:
                    raise ValueError(f"Alias {entry.category!r}/{alias!r} collides with existing canonical name.")
                if alias_key in self._aliases:
                    raise ValueError(
                        f"Alias {entry.category!r}/{alias!r} is already registered for {self._aliases[alias_key]!r}."
                    )
                self._aliases[alias_key] = entry.name
                installed_aliases.append(alias_key)
        except Exception:
            for alias_key in installed_aliases:
                self._aliases.pop(alias_key, None)
            self._entries.pop(key, None)
            raise
        # A class declared before it was registered keeps resolving: its
        # identity key, if any, stays as a lookup for the same class.

    def unregister(self, category: str, name: str) -> None:
        """Remove a canonical entry and all of its aliases.

        Parameters
        ----------
        category : str
            Category of the entry to remove.
        name : str
            Canonical name (not an alias) of the entry to remove.

        Raises
        ------
        KeyError
            If no canonical entry exists for ``(category, name)``.
        """
        _check_category(category)
        key = (category, name)
        if key not in self._entries:
            raise KeyError(f"No canonical mechanism registered at {category!r}/{name!r}.")
        entry = self._entries.pop(key)
        for alias in entry.aliases:
            self._aliases.pop((category, alias), None)

    def clear(self) -> None:
        """Remove every registered entry and alias.

        This method exists primarily for tests. Production code should
        never need it — the registry is populated once at import time.
        """
        self._entries.clear()
        self._aliases.clear()
        self._anonymous.clear()
        self._anonymous_keys.clear()

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def contains(self, category: str, name: str) -> bool:
        """Return whether ``name`` (canonical, alias or identity key) is known."""
        _check_category(category)
        key = (category, name)
        return key in self._entries or key in self._aliases or key in self._anonymous

    def key_for(self, category: str, cls: type) -> str:
        """Return the name under which declarations refer to ``cls``.

        A registered class resolves to its canonical name. An unregistered
        class gets an identity key the first time it is seen: its
        ``__name__`` when no other class uses that name in ``category``,
        else ``module.qualname``, with ``#n`` appended if that is taken
        too. :meth:`get` resolves the key back to this very class, so two
        unregistered classes that share a ``__name__`` stay distinct.

        Parameters
        ----------
        category : str
            Mechanism category.
        cls : type
            The class to refer to.

        Returns
        -------
        str
            The canonical name or identity key.
        """
        _check_category(category)
        for entry_name, entry_cls in self.items(category):
            if entry_cls is cls:
                return entry_name
        known = self._anonymous_keys.get((category, cls))
        if known is not None:
            return known
        base = f"{cls.__module__}.{cls.__qualname__}"
        candidates = (cls.__name__, base)
        key = next((name for name in candidates if not self.contains(category, name)), None)
        suffix = 2
        while key is None:
            if not self.contains(category, f"{base}#{suffix}"):
                key = f"{base}#{suffix}"
            suffix += 1
        self._anonymous[(category, key)] = cls
        self._anonymous_keys[(category, cls)] = key
        return key

    def get(self, category: str, name: str) -> type:
        """Return the concrete class registered as ``(category, name)``.

        Parameters
        ----------
        category : str
            One of ``"channel"``, ``"ion"``, ``"synapse"``, or
            ``"connection_rule"``.
        name : str
            Canonical name, alias, or identity key (see :meth:`key_for`).

        Returns
        -------
        type
            The registered class.

        Raises
        ------
        KeyError
            If no entry is registered. The error message includes a
            difflib-based "did you mean ...?" suggestion.
        """
        return self.entry(category, name).cls

    def entry(self, category: str, name: str) -> MechanismEntry:
        """Return the :class:`MechanismEntry` for ``(category, name)``.

        Parameters
        ----------
        category : str
            Mechanism category.
        name : str
            Canonical name, alias, or identity key (see :meth:`key_for`).

        Returns
        -------
        MechanismEntry
            The resolved entry.

        Raises
        ------
        KeyError
            If no entry is registered. The error message includes a
            difflib-based "did you mean ...?" suggestion.
        """
        _check_category(category)
        key = (category, name)
        if key in self._entries:
            return self._entries[key]
        if key in self._aliases:
            return self._entries[(category, self._aliases[key])]
        if key in self._anonymous:
            return MechanismEntry(category=category, name=name, cls=self._anonymous[key])
        raise KeyError(_missing_mechanism_message(self, category, name))

    def names(
        self,
        category: str | None = None,
        *,
        include_aliases: bool = False,
    ) -> tuple[str, ...]:
        """Return all registered canonical names in a category.

        Parameters
        ----------
        category : str or None
            Restrict to one category, or ``None`` for all categories.
        include_aliases : bool
            If ``True``, alias names are appended after canonical names.

        Returns
        -------
        tuple of str
            Sorted tuple of names. Canonical names come first when
            ``include_aliases`` is ``True``.
        """
        if category is not None:
            _check_category(category)
        canonical = sorted(
            entry.name for (cat, _), entry in self._entries.items() if category is None or cat == category
        )
        if not include_aliases:
            return tuple(canonical)
        aliases = sorted(alias for (cat, alias) in self._aliases if category is None or cat == category)
        return tuple(canonical + aliases)

    def items(self, category: str | None = None) -> tuple[tuple[str, type], ...]:
        """Return ``(name, cls)`` pairs for all entries in a category.

        Parameters
        ----------
        category : str or None
            Restrict to one category, or ``None`` for all entries.

        Returns
        -------
        tuple of (str, type)
            Sorted by name. Aliases are not included — one pair per
            canonical entry.
        """
        if category is not None:
            _check_category(category)
        pairs = [
            (entry.name, entry.cls) for (cat, _), entry in self._entries.items() if category is None or cat == category
        ]
        pairs.sort(key=lambda item: item[0])
        return tuple(pairs)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        total = len(self._entries)
        counts = {cat: sum(1 for (c, _) in self._entries if c == cat) for cat in sorted(_VALID_CATEGORIES)}
        summary = ", ".join(f"{cat}={count}" for cat, count in counts.items())
        return f"MechanismRegistry(total={total}, {summary})"


# Runtime base classes ----------------------------------------------------
#
# ``braincell.mech`` may not import the runtime packages (it is a leaf), so the
# modules that define the runtime bases report them here. Declarations use
# them to reject a class outside a category's hierarchy at construction.
_RUNTIME_BASES: dict[str, tuple[type, tuple[type, ...]]] = {}


def _register_runtime_base(category: str, base: type, *, excluded: tuple[type, ...] = ()) -> None:
    """Record the runtime base class of ``category`` and subclasses it excludes."""
    _check_category(category)
    _RUNTIME_BASES[category] = (base, tuple(excluded))


def _runtime_base(category: str) -> tuple[type, tuple[type, ...]]:
    """Return ``(base, excluded)`` for ``category``.

    ``import braincell`` imports the base modules that report them, and
    ``braincell.mech`` cannot be imported without ``braincell``, so every
    category is reported before a declaration is built.
    """
    return _RUNTIME_BASES[category]


# Module-level singleton --------------------------------------------------

_REGISTRY = MechanismRegistry()


def get_registry() -> MechanismRegistry:
    """Return the module-level :class:`MechanismRegistry` singleton."""
    return _REGISTRY


# Decorators --------------------------------------------------------------


def register_channel(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Callable[[_T], _T]:
    """Class decorator that registers an ion-channel class by name.

    Parameters
    ----------
    name : str
        Canonical registry name for the decorated class.
    aliases : tuple of str
        Additional alias names that also resolve to this class.

    Returns
    -------
    callable
        A decorator that returns its argument unchanged after
        registering it.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.mech import register_channel
        >>> @register_channel("IL", aliases=("leaky",))
        ... class IL(...):
        ...     ...
    """
    return _make_decorator(_CATEGORY_CHANNEL, name, aliases)


def register_ion(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Callable[[_T], _T]:
    """Class decorator that registers an ion-species class by name.

    See :func:`register_channel` for the parameter semantics.
    """
    return _make_decorator(_CATEGORY_ION, name, aliases)


def register_synapse(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Callable[[_T], _T]:
    """Class decorator that registers a synapse class by name.

    See :func:`register_channel` for the parameter semantics.
    """
    return _make_decorator(_CATEGORY_SYNAPSE, name, aliases)


def register_connection_rule(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Callable[[_T], _T]:
    """Class decorator that registers a Connection plasticity rule by name.

    The decorated class subclasses the runtime base
    :class:`braincell.ConnectionRule`; a
    :class:`~braincell.mech.ConnectionRule` declaration then names it, and
    ``connect(..., plasticity=...)`` looks the name up. See
    :func:`register_channel` for the parameter semantics.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import braincell
        >>> @braincell.mech.register_connection_rule("ConstantDrift")
        ... class ConstantDrift(braincell.ConnectionRule):
        ...     def __init__(self, size, weight_unit, *, rate):
        ...         super().__init__(size, weight_unit)
        ...         self.rate = rate
        ...     def update(self, state, *, pre, dt):
        ...         return {"w": state["w"] + self.rate * dt}
        >>> braincell.mech.get_registry().get("connection_rule", "ConstantDrift") is ConstantDrift
        True
        >>> braincell.mech.get_registry().unregister("connection_rule", "ConstantDrift")
    """
    return _make_decorator(_CATEGORY_CONNECTION_RULE, name, aliases)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _make_decorator(
    category: str,
    name: str,
    aliases: tuple[str, ...],
) -> Callable[[_T], _T]:
    def decorator(cls: _T) -> _T:
        _REGISTRY.register(
            MechanismEntry(
                category=category,
                name=name,
                cls=cls,
                aliases=tuple(aliases),
            )
        )
        return cls

    return decorator


def _check_category(category: str) -> None:
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(_VALID_CATEGORIES)!r}, got {category!r}.")


#: ``(category, old name) -> (new name, release)`` for mechanisms that a release
#: renamed without keeping an alias (MECH-42), so that a lookup of the old name
#: says what it became instead of guessing.
_RENAMED = {
    ("channel", "CaHT_HM1992"): ("CaT_HM1992_Shifted25", "0.3"),
}


def _missing_mechanism_message(
    registry: MechanismRegistry,
    category: str,
    name: str,
) -> str:
    base = f"No {category!r} mechanism registered as {name!r}."
    renamed = _RENAMED.get((category, name))
    if renamed is not None:
        new_name, release = renamed
        return f"{base} braincell {release} renamed it to {new_name!r} and kept no alias."
    candidates = registry.names(category, include_aliases=True)
    suggestions = difflib.get_close_matches(name, candidates, n=3)
    if suggestions:
        base += f" Did you mean {suggestions!r}?"
    elif candidates:
        preview = ", ".join(repr(n) for n in candidates[:8])
        base += f" Registered {category} names: {preview}..."
    return base
