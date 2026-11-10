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
import os
import sys
import warnings
import weakref
from typing import Any, Mapping

import brainunit as u
import jax
import numpy as np

from braincell.mech._registry import get_registry
from braincell.mech._synapse_schema import describe_mechanism

__all__ = ["user_frame", "warn_at_user_frame", "warn_invalid_declared_parameters"]

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

#: Exception ``Cell.init_state()`` raises for a parameter name the runtime
#: class does not accept.
_UNKNOWN_NAME_ERRORS = {"channel": TypeError, "ion": ValueError, "synapse": TypeError}

#: Constructor arguments the Cell supplies itself.
_CELL_SUPPLIED = frozenset({"size", "name"})

_UNRESOLVABLE = object()
_PARAMETER_INFO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def user_frame(frame):
    """Return the nearest frame outside braincell's own code, and its distance.

    Frames in the braincell package are skipped, except its ``*_test.py``
    files, which call the API as users do.

    Parameters
    ----------
    frame : frame or None
        Frame to start from.

    Returns
    -------
    tuple
        ``(frame, hops)``: the first user frame (``None`` if there is none)
        and the number of frames skipped to reach it.
    """
    hops = 0
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(_PACKAGE_DIR) or filename.endswith("_test.py"):
            return frame, hops
        frame = frame.f_back
        hops += 1
    return None, hops


def warn_at_user_frame(message: str, category: type[Warning] = FutureWarning) -> None:
    """Issue ``message`` attributed to the nearest user frame (see :func:`user_frame`).

    The caller's own frame counts, so a call from a braincell ``*_test.py``
    is attributed to that line. Where frames are unavailable (``sys._getframe``
    is CPython-specific) the warning points at the caller.
    """
    stacklevel = 2
    getframe = getattr(sys, "_getframe", None)
    if getframe is not None:
        frame, hops = user_frame(getframe(1))
        if frame is not None:
            stacklevel += hops
    warnings.warn(message, category, stacklevel=stacklevel)


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


def _warn(message: str, error: type[Exception]) -> None:
    warn_at_user_frame(f"{message} This will raise {error.__name__} at declaration from braincell 0.4.")


def _parameter_info(category: str, class_name: str):
    """Return the runtime class's ``parameter_info()``, cached per class.

    Returns ``_UNRESOLVABLE`` when the name is not registered yet or the
    class's metadata cannot be built; ``init_state`` then reports the problem.
    """
    registry = get_registry()
    if not registry.contains(category, class_name):
        return _UNRESOLVABLE
    cls = registry.get(category, class_name)
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
