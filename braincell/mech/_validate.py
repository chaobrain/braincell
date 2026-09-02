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

"""Shared field validation for :mod:`braincell.mech` declarations.

Every declaration class in this package validates the same handful of
field shapes. Before this module existed the "must be a non-empty
string" check alone was written eleven times across :mod:`._density`
and :mod:`._point`, and — because each copy chose its own exception —
the same condition raised ``TypeError`` in some classes and
``ValueError`` in others. Callers had no way to write a correct
``except`` clause.

The rule these helpers apply is the Python convention the package was
already half-following:

* ``TypeError`` when the value is of the wrong type.
* ``ValueError`` when the type is right but the value is not.

This module deliberately depends on nothing outside the standard
library, so it does not disturb the leaf-package invariant documented in
:mod:`braincell.mech` — ``braincell.mech`` imports nothing from
``braincell``.
"""

from typing import Any

__all__ = ["require_str", "require_fraction"]


def require_str(value: Any, owner: str, field: str, *, optional: bool = False) -> str | None:
    """Validate that ``value`` is a non-empty string.

    Parameters
    ----------
    value : object
        The value to check.
    owner : str
        Owning class name, used to build the error message.
    field : str
        Field name, used to build the error message.
    optional : bool
        When ``True``, ``None`` is accepted and returned unchanged.

    Returns
    -------
    str or None
        ``value`` unchanged, once validated.

    Raises
    ------
    TypeError
        If ``value`` is not a :class:`str` (or ``None`` when *optional*).
    ValueError
        If ``value`` is an empty string.
    """
    if optional and value is None:
        return None
    suffix = " or None" if optional else ""
    if not isinstance(value, str):
        raise TypeError(f"{owner}.{field} must be a non-empty string{suffix}, got {type(value).__name__!r}.")
    if not value:
        raise ValueError(f"{owner}.{field} must be a non-empty string{suffix}, got {value!r}.")
    return value


def require_fraction(value: Any, owner: str, field: str) -> float:
    """Validate that ``value`` is a real number in ``[0, 1]``.

    Parameters
    ----------
    value : object
        The value to check; converted with :class:`float`.
    owner : str
        Owning class name, used to build the error message.
    field : str
        Field name, used to build the error message.

    Returns
    -------
    float
        ``value`` as a float, once validated.

    Raises
    ------
    TypeError
        If ``value`` cannot be converted to :class:`float`.
    ValueError
        If the converted value lies outside ``[0, 1]``.
    """
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{owner}.{field} must be a real number, got {type(value).__name__!r}.") from exc
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"{owner}.{field} must lie in [0, 1], got {fraction!r}.")
    return fraction
