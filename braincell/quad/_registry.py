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

"""Integrator registry for ``braincell.quad``.

This module provides the central, decentralized mechanism that lets numerical
integration methods (the ``*_step`` functions) declare a stable, user-facing
name independent of their Python identifier. The registry replaces the
previous ``locals()``-scanning approach in :mod:`braincell.quad.__init__`,
which forced every new integrator to be re-imported into ``__init__.py`` and
silently coupled the user-facing name to the function symbol.

The registry is intentionally small. Its responsibilities are:

* Map a stable canonical name (e.g. ``"staggered"``) to a step function.
* Track aliases (e.g. ``"stagger"``, ``"explicit"``) without conflating them
  with canonical names.
* Carry lightweight metadata (category, convergence order, description,
  deprecation flag) for documentation and introspection.
* Detect collisions and provide actionable error messages on misses.

Typical usage::

    from braincell.quad import register_integrator

    @register_integrator(
        "staggered",
        aliases=("stagger",),
        category="staggered",
        description="Staggered voltage / ion-channel splitting.",
    )
    def staggered_step(target, *args):
        ...
"""



import difflib
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Mapping

__all__ = [
    'IntegratorEntry',
    'IntegratorRegistry',
    'get_registry',
    'register_integrator',
]


@dataclass(frozen=True)
class IntegratorEntry:
    """A single registered integrator and its metadata.

    Parameters
    ----------
    name : str
        Canonical, user-facing name of the integrator (e.g. ``"euler"``).
    func : Callable
        The underlying ``*_step`` function.
    aliases : tuple of str
        Alternative names that resolve to this entry.
    category : str
        Loose grouping for introspection and documentation. Common values
        used in-tree are ``"explicit"``, ``"implicit"``, ``"exponential"``,
        ``"staggered"``, ``"diffrax"``, ``"voltage"``, and ``"general"``.
    order : int or None
        Convergence order of the method, when applicable.
    description : str
        One-line human-readable description shown in error messages and
        documentation.
    deprecated : bool
        If ``True``, looking up this entry by name emits a
        :class:`DeprecationWarning`.
    module : str
        Fully-qualified module name where ``func`` is defined. Populated
        automatically by :meth:`IntegratorRegistry.register`.
    """

    name: str
    func: Callable
    aliases: tuple[str, ...] = ()
    category: str = "general"
    order: int | None = None
    description: str = ""
    deprecated: bool = False
    module: str = ""


class IntegratorRegistry:
    """Mutable, observable registry of integrator step functions.

    The registry maintains two indexes:

    * ``_entries``: canonical name → :class:`IntegratorEntry`
    * ``_aliases``: alias name → canonical name

    Lookups via :meth:`__getitem__` and :meth:`get` consult both, so callers
    do not need to know whether a string is a canonical name or an alias.

    Most code interacts with the global singleton returned by
    :func:`get_registry`. Constructing a fresh ``IntegratorRegistry()`` is
    useful in tests that want to avoid touching global state.
    """

    def __init__(self) -> None:
        self._entries: dict[str, IntegratorEntry] = {}
        self._aliases: dict[str, str] = {}
        self._deprecation_warned: set[str] = set()

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #
    def register(
        self,
        name: str,
        func: Callable,
        *,
        aliases: Iterable[str] = (),
        category: str = "general",
        order: int | None = None,
        description: str = "",
        deprecated: bool = False,
        override: bool = False,
    ) -> IntegratorEntry:
        """Register an integrator under ``name`` (and any ``aliases``).

        Parameters
        ----------
        name : str
            Canonical name to register. Must not already be present unless
            ``override=True``.
        func : Callable
            The integrator step function to associate with ``name``.
        aliases : iterable of str
            Optional alternative names. Each alias must be unique across the
            full registry (canonical names *and* other aliases).
        category, order, description, deprecated
            Forwarded to the resulting :class:`IntegratorEntry`.
        override : bool
            If ``True``, replace an existing entry with the same canonical
            name and emit a :class:`RuntimeWarning` identifying the previous
            owner. If ``False`` (the default), a colliding canonical name
            raises :class:`ValueError`.

        Returns
        -------
        IntegratorEntry
            The entry that was inserted.

        Raises
        ------
        TypeError
            If ``func`` is not callable, or ``name`` is not a non-empty string.
        ValueError
            If ``name`` collides with an existing canonical name or alias and
            ``override`` is False, or if any alias collides with an existing
            canonical name or alias owned by a different entry.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(f"Integrator name must be a non-empty string, got {name!r}.")
        if not callable(func):
            raise TypeError(f"Integrator {name!r} must be a callable, got {type(func).__name__}.")

        alias_tuple = tuple(aliases)
        for alias in alias_tuple:
            if not isinstance(alias, str) or not alias:
                raise TypeError(
                    f"Aliases for integrator {name!r} must be non-empty strings, got {alias!r}."
                )

        # Canonical-name collision check.
        if name in self._entries and not override:
            existing = self._entries[name]
            raise ValueError(
                f"Integrator {name!r} is already registered by {existing.module!r}. "
                f"Pass override=True to replace it."
            )
        if name in self._aliases and self._aliases[name] != name:
            owner = self._aliases[name]
            raise ValueError(
                f"Cannot register integrator {name!r}: the name is already an alias "
                f"for {owner!r}."
            )

        # Alias collision check.
        for alias in alias_tuple:
            if alias in self._entries and alias != name:
                raise ValueError(
                    f"Cannot register alias {alias!r} for {name!r}: a canonical "
                    f"integrator with that name already exists."
                )
            existing_owner = self._aliases.get(alias)
            if existing_owner is not None and existing_owner != name:
                raise ValueError(
                    f"Cannot register alias {alias!r} for {name!r}: it is already "
                    f"an alias for {existing_owner!r}."
                )

        if override and name in self._entries:
            previous = self._entries[name]
            warnings.warn(
                f"Overriding integrator {name!r} previously registered by "
                f"{previous.module!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Strip the previous aliases owned by this canonical name.
            for alias, target in list(self._aliases.items()):
                if target == name:
                    del self._aliases[alias]

        entry = IntegratorEntry(
            name=name,
            func=func,
            aliases=alias_tuple,
            category=category,
            order=order,
            description=description,
            deprecated=deprecated,
            module=getattr(func, "__module__", "") or "",
        )
        self._entries[name] = entry
        for alias in alias_tuple:
            self._aliases[alias] = name
        # Reset deprecation warning state for this name so re-registration
        # under the same name resurfaces the warning if appropriate.
        self._deprecation_warned.discard(name)
        return entry

    def unregister(self, name: str) -> None:
        """Remove the integrator registered as ``name``.

        ``name`` must be a canonical name; passing an alias raises
        :class:`KeyError`. All aliases owned by the entry are removed too.
        """
        if name not in self._entries:
            raise KeyError(f"No integrator registered as {name!r}.")
        for alias, target in list(self._aliases.items()):
            if target == name:
                del self._aliases[alias]
        del self._entries[name]
        self._deprecation_warned.discard(name)

    # ------------------------------------------------------------------ #
    # lookup
    # ------------------------------------------------------------------ #
    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name in self._entries or name in self._aliases

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, name: str) -> Callable:
        return self.entry(name).func

    def get(self, name: str, default: Callable | None = None) -> Callable | None:
        """Return the step function for ``name``, or ``default`` if missing."""
        try:
            return self[name]
        except KeyError:
            return default

    def entry(self, name: str) -> IntegratorEntry:
        """Return the :class:`IntegratorEntry` for ``name``.

        ``name`` may be a canonical name or an alias. Looking up a deprecated
        entry emits a :class:`DeprecationWarning` once per name per registry
        instance.
        """
        canonical = self._aliases.get(name, name)
        try:
            entry = self._entries[canonical]
        except KeyError as exc:
            raise KeyError(name) from exc
        if entry.deprecated and name not in self._deprecation_warned:
            self._deprecation_warned.add(name)
            warnings.warn(
                f"Integrator {name!r} is deprecated.",
                DeprecationWarning,
                stacklevel=3,
            )
        return entry

    def resolve(self, name: str) -> str:
        """Return the canonical name for ``name`` (which may be an alias)."""
        return self.entry(name).name

    def names(self, *, include_aliases: bool = False) -> list[str]:
        """Return the sorted list of registered names.

        If ``include_aliases`` is ``True``, alias names are appended.
        """
        names = sorted(self._entries.keys())
        if include_aliases:
            names = sorted(set(names) | set(self._aliases.keys()))
        return names

    def by_category(self, category: str) -> list[IntegratorEntry]:
        """Return all entries belonging to ``category``, sorted by name."""
        return sorted(
            (e for e in self._entries.values() if e.category == category),
            key=lambda e: e.name,
        )

    def items(self) -> Iterator[tuple[str, Callable]]:
        """Iterate over ``(name, func)`` pairs for canonical entries only."""
        for name, entry in self._entries.items():
            yield name, entry.func

    def entries(self) -> Iterator[IntegratorEntry]:
        """Iterate over all canonical :class:`IntegratorEntry` instances."""
        return iter(self._entries.values())

    def as_dict(self, *, include_aliases: bool = True) -> dict[str, Callable]:
        """Return a flat ``{name: func}`` snapshot.

        By default the snapshot includes alias keys, matching the behavior of
        the legacy ``all_integrators`` mapping.
        """
        out: dict[str, Callable] = {name: entry.func for name, entry in self._entries.items()}
        if include_aliases:
            for alias, canonical in self._aliases.items():
                out[alias] = self._entries[canonical].func
        return out

    def suggest(self, name: str, *, n: int = 1, cutoff: float = 0.6) -> list[str]:
        """Return up to ``n`` close-match suggestions for an unknown ``name``."""
        candidates = self.names(include_aliases=True)
        return difflib.get_close_matches(name, candidates, n=n, cutoff=cutoff)


# ---------------------------------------------------------------------- #
# global singleton + decorator
# ---------------------------------------------------------------------- #
_REGISTRY = IntegratorRegistry()


def get_registry() -> IntegratorRegistry:
    """Return the process-wide :class:`IntegratorRegistry` singleton."""
    return _REGISTRY


def register_integrator(
    name: str,
    *,
    aliases: Iterable[str] = (),
    category: str = "general",
    order: int | None = None,
    description: str = "",
    deprecated: bool = False,
    override: bool = False,
) -> Callable[[Callable], Callable]:
    """Decorator: register the wrapped function in the global registry.

    The decorated function is returned unchanged so it remains directly
    callable. Place this decorator **outside** any other decorators (such as
    :func:`braincell._misc.set_module_as`) so that the registry stores the
    final exported function.

    Parameters
    ----------
    name : str
        Canonical name under which to register the integrator.
    aliases : iterable of str
        Alternative lookup names. See :meth:`IntegratorRegistry.register`.
    category, order, description, deprecated, override
        Forwarded to :meth:`IntegratorRegistry.register`.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.quad import register_integrator
        >>> @register_integrator("euler", aliases=("explicit",),
        ...                      category="explicit", order=1)
        ... def euler_step(target, *args):
        ...     ...
    """

    def _wrap(func: Callable) -> Callable:
        _REGISTRY.register(
            name,
            func,
            aliases=aliases,
            category=category,
            order=order,
            description=description,
            deprecated=deprecated,
            override=override,
        )
        return func

    return _wrap
