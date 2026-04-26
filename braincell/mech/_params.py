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

"""Frozen parameter mapping for :mod:`braincell.mech`.

:class:`Params` is an immutable, hashable, dict-like mapping used to
store the declared parameters of a mechanism (e.g. ``g_max``, ``E`` for
a leak channel). Its two load-bearing properties are:

1. **Order-preserving iteration** so reprs, runtime constructor kwargs,
   and layout-signature tuples are stable across runs.
2. **Order-insensitive equality and hashing** so two declarations that
   differ only in keyword order (``Channel("IL", g_max=g, E=e)`` vs
   ``Channel("IL", E=e, g_max=g)``) deduplicate correctly in paint
   grouping.

:class:`Params` implements :class:`collections.abc.Mapping`, so
``dict(params)`` and ``**params`` unpacking both work.
"""

from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["Params"]


class Params(Mapping[str, Any]):
    """Frozen, hashable, order-preserving mechanism parameter mapping.

    Parameters
    ----------
    data : Mapping, iterable of (name, value), or None
        Initial key-value pairs. When ``None`` (the default), the
        mapping starts empty.
    **kwargs
        Additional parameters provided as keyword arguments. These
        override entries of the same name in ``data``.

    Notes
    -----
    - Iteration order matches declaration order: entries in ``data``
      come first (in their original order), followed by entries in
      ``kwargs`` (in the order they were passed).
    - Equality is dict-like — two instances with the same key-value
      pairs compare equal regardless of iteration order.
    - Hashing is consistent with equality: instances that compare
      equal share the same hash, so ``Params`` is safe to use as a
      dict key or set element.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.mech import Params
        >>> p = Params(g_max=0.1 * u.mS / u.cm ** 2, E=-70 * u.mV)
        >>> p["g_max"]
        0.1 * mS / cm ** 2
        >>> list(p)
        ['g_max', 'E']
        >>> p == Params(E=-70 * u.mV, g_max=0.1 * u.mS / u.cm ** 2)
        True
    """

    __slots__ = ("_items",)

    def __init__(self, data: object = None, /, **kwargs: Any) -> None:
        merged: dict[str, Any] = {}
        if data is None:
            pass
        elif isinstance(data, Params):
            merged.update(data._items)
        elif isinstance(data, Mapping):
            for key, value in data.items():
                merged[str(key)] = value
        else:
            # Interpret as an iterable of (name, value) pairs.
            try:
                iterator = iter(data)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError(
                    f"Params() expected a Mapping or iterable of "
                    f"(name, value) pairs, got {type(data).__name__!r}."
                ) from exc
            for pair in iterator:
                try:
                    key, value = pair
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"Params() entries must be (name, value) pairs, "
                        f"got {pair!r}."
                    ) from exc
                merged[str(key)] = value
        for key, value in kwargs.items():
            merged[str(key)] = value
        for key, value in merged.items():
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError(
                    f"Params value for {key!r} must be hashable "
                    f"(arrays are rejected); got {type(value).__name__!r}."
                ) from exc
        object.__setattr__(self, "_items", merged)

    # ------------------------------------------------------------------
    # Mapping protocol
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def keys(self):  # type: ignore[override]
        return self._items.keys()

    def values(self):  # type: ignore[override]
        return self._items.values()

    def items(self):  # type: ignore[override]
        return self._items.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._items.get(key, default)

    # ------------------------------------------------------------------
    # immutability: block accidental mutation
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"Params is immutable; cannot set attribute {name!r}."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"Params is immutable; cannot delete attribute {name!r}."
        )

    # ------------------------------------------------------------------
    # equality / hashing: order-insensitive (dict-like)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Params):
            return self._items == other._items
        if isinstance(other, Mapping):
            return dict(self._items) == dict(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        try:
            return hash(frozenset(self._items.items()))
        except TypeError as exc:
            raise TypeError(
                f"Params values must be hashable to hash the mapping; "
                f"unhashable entry encountered ({exc})."
            ) from exc

    # ------------------------------------------------------------------
    # repr: stable declared order
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._items:
            return "Params()"
        parts = ", ".join(f"{k}={v!r}" for k, v in self._items.items())
        return f"Params({parts})"

    # ------------------------------------------------------------------
    # structural updates (non-mutating)
    # ------------------------------------------------------------------

    def with_updates(self, **kwargs: Any) -> "Params":
        """Return a copy with ``kwargs`` merged in.

        Parameters
        ----------
        **kwargs
            Parameters to add or replace. Keys already present in
            ``self`` keep their original position in iteration order;
            new keys are appended at the end.

        Returns
        -------
        Params
            A new :class:`Params` instance. ``self`` is unchanged.
        """
        merged = dict(self._items)
        for key, value in kwargs.items():
            merged[str(key)] = value
        return Params(merged)

    def without(self, *keys: str) -> "Params":
        """Return a copy with the given keys removed.

        Unknown keys are silently ignored.

        Parameters
        ----------
        *keys : str
            Keys to drop.

        Returns
        -------
        Params
            A new :class:`Params` instance.
        """
        if not keys:
            return self
        to_drop = {str(k) for k in keys}
        remaining = {
            key: value
            for key, value in self._items.items()
            if key not in to_drop
        }
        return Params(remaining)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def coerce(cls, value: object) -> "Params":
        """Convert ``value`` to a :class:`Params` instance.

        Parameters
        ----------
        value : Params, Mapping, iterable of (name, value), or None
            Source data.

        Returns
        -------
        Params
            If ``value`` is already a :class:`Params`, it is returned
            unchanged. Otherwise, a new instance is constructed.
        """
        if isinstance(value, Params):
            return value
        if value is None:
            return cls()
        return cls(value)
