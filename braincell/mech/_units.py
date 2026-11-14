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

"""Declaration-time parameter checks and the user call site they report.

``bc.mech.Channel``, ``Ion`` and ``Synapse`` check their literal parameters
against the runtime class's ``parameter_info()`` when the class can be
resolved at construction time. In 0.3 a problem is a ``FutureWarning``
pointing at the line that wrote the declaration; ``Cell.init_state()``
still raises the error the warning quotes.
"""

from __future__ import annotations

import difflib
import numbers
import weakref
from typing import Any, Mapping

import brainunit as u
import jax
import numpy as np

# ``user_frame``, ``warn_at_user_frame`` and ``_PACKAGE_DIR`` moved to
# ``braincell._deprecation``; they stay importable from here.
from braincell._deprecation import _PACKAGE_DIR as _PACKAGE_DIR, Deprecation, user_frame, warn_at_user_frame
from braincell.mech._registry import get_registry
from braincell.mech._synapse_schema import describe_mechanism

__all__ = [
    "user_frame",
    "warn_at_user_frame",
    "warn_invalid_constructed_parameters",
    "warn_invalid_declared_parameters",
]

#: Exception ``Cell.init_state()`` raises for a parameter name the runtime
#: class does not accept.
_UNKNOWN_NAME_ERRORS = {"channel": TypeError, "ion": ValueError, "synapse": TypeError}

#: Constructor arguments the Cell supplies itself.
_CELL_SUPPLIED = frozenset({"size", "name"})

#: MECH-26: a declaration whose literal parameters ``init_state()`` rejects.
_INVALID_DECLARATION = Deprecation(
    "Constructing an invalid mechanism declaration",
    since="0.3",
    removed_in="0.4",
    category=FutureWarning,
    replacement=None,
    note="Fix the parameter the warning names; the declaration itself will raise.",
)
#: MECH-28: a directly constructed Channel or Ion whose parameters have the wrong units.
_INVALID_CONSTRUCTION = Deprecation(
    "Directly constructing a Channel or Ion with wrongly united parameters",
    since="0.3",
    removed_in="0.4",
    category=FutureWarning,
    replacement=None,
    note="Fix the parameter the warning names; the construction itself will raise.",
)
_CHECKS = {"declaration": _INVALID_DECLARATION, "construction": _INVALID_CONSTRUCTION}

_UNRESOLVABLE = object()
_PARAMETER_INFO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def warn_invalid_declared_parameters(
    category: str,
    class_name: str,
    instance_name: str | None,
    params: Mapping[str, Any],
) -> None:
    """Warn about declaration parameters ``Cell.init_state()`` will reject.

    Checks only what is known at construction: the class must be resolvable
    (a class object, or a name already registered); unknown names are those
    missing from ``parameter_info()``; values are checked only when literal
    (quantity, number or array), with traced values compared by unit alone.
    Each problem gives one ``FutureWarning`` at the user's line, quoting the
    error ``init_state`` raises and the exception type it will raise at the
    declaration from braincell 0.4.

    Parameters
    ----------
    category : str
        ``"channel"``, ``"ion"`` or ``"synapse"``.
    class_name : str
        The registry key the declaration stores.
    instance_name : str or None
        The declaration's logical name.
    params : Mapping
        The declared parameters.
    """
    if not params:
        return
    info = _parameter_info(category, class_name)
    if info is _UNRESOLVABLE:
        return
    owner = describe_mechanism(category, class_name, instance_name)
    accepted = [name for name in info if name not in _CELL_SUPPLIED]
    for name, value in params.items():
        if name not in accepted:
            error = _UNKNOWN_NAME_ERRORS[category]
            close = difflib.get_close_matches(name, accepted, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            _warn(f"{owner} got an unexpected parameter {name!r}.{hint}", error)
        elif _is_literal(value):
            try:
                info[name].validate(value, name, owner=owner)
            except (TypeError, ValueError) as exc:
                _warn(str(exc), type(exc))


def declared_parameters_are_valid(
    category: str,
    class_name: str,
    instance_name: str | None,
    params: Mapping[str, Any],
) -> bool:
    """Whether :func:`warn_invalid_declared_parameters` would warn about nothing, every value checked.

    ``True`` only when the class resolves and every parameter is a known
    name with a literal value that passes its check, so that any part of
    a value (a row of an array) passes too (CELL-51). Warns about nothing.
    """
    info = _parameter_info(category, class_name)
    if info is _UNRESOLVABLE:
        return False
    owner = describe_mechanism(category, class_name, instance_name)
    accepted = [name for name in info if name not in _CELL_SUPPLIED]
    for name, value in params.items():
        if name not in accepted or not _is_literal(value):
            return False
        try:
            info[name].validate(value, name, owner=owner)
        except (TypeError, ValueError):
            return False
    return True


def warn_invalid_constructed_parameters(node: Any, category: str) -> None:
    """Warn about the units of the parameters a directly built Channel or Ion stores.

    ``node`` is a ``category`` (``"channel"`` or ``"ion"``) instance whose
    constructor has returned. Each ``parameter_info()`` name is read back and
    compared by unit only (no finite check, no array conversion), so values
    traced under ``jax.jit`` or ``jax.grad`` are checked too. ``size``,
    ``name``, ``*_initializer``, missing attributes and non-literal values
    are skipped. The warning goes to the constructing line, past brainstate's
    metaclass ``__call__`` (see :func:`user_frame`).
    """
    info = _class_parameter_info(type(node))
    if info is _UNRESOLVABLE:
        return
    owner = None
    for name, spec in info.items():
        if name in _CELL_SUPPLIED or name.endswith("_initializer"):
            continue
        value = getattr(node, name, None)
        if _is_literal(value):
            owner = owner or describe_mechanism(category, type(node).__name__, getattr(node, "name", None))
            try:
                spec.validate(value, name, owner=owner, units_only=True)
            except (TypeError, ValueError) as exc:
                _warn(str(exc), type(exc), "construction")


def _warn(message: str, error: type[Exception], where: str = "declaration") -> None:
    check = _CHECKS[where]
    check.warn(f"{message} This will raise {error.__name__} at {where} from braincell {check.removed_in}.", at="user")


def _parameter_info(category: str, class_name: str):
    """Return the runtime class's ``parameter_info()``, cached per class.

    Returns ``_UNRESOLVABLE`` when the name is not registered yet or the
    class's metadata cannot be built; ``init_state`` then reports the problem.
    """
    registry = get_registry()
    if not registry.contains(category, class_name):
        return _UNRESOLVABLE
    return _class_parameter_info(registry.get(category, class_name))


def _class_parameter_info(cls: type):
    """Return ``cls.parameter_info()`` as a dict, cached per class; ``_UNRESOLVABLE`` if it raises."""
    info = _PARAMETER_INFO.get(cls)
    if info is None:
        try:
            info = dict(cls.parameter_info())
        except (AttributeError, TypeError, ValueError):
            info = _UNRESOLVABLE
        _PARAMETER_INFO[cls] = info
    return info


def _is_literal(value: object) -> bool:
    return isinstance(value, (u.Quantity, numbers.Number, np.ndarray, np.generic, jax.Array))
