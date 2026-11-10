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

"""Field declarations shared by vectorized runtime mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import brainunit as u
import numpy as np

__all__ = ["ParameterSpec", "StateSpec", "positive"]

Validator = Callable[[object, str], None]
_OWNER_MANAGED = object()
_NO_VALUE = object()


@dataclass(frozen=True)
class ParameterSpec:
    """Declare one broadcastable physical runtime parameter."""

    default: object
    validator: Validator | None = None

    def validate(self, value: object, name: str, *, owner: str | None = None, source: str | None = None) -> None:
        """Validate one resolved parameter value against this declaration.

        Parameters
        ----------
        value : object
            Resolved parameter value.
        name : str
            Parameter name used in error messages.
        owner : str, optional
            Mechanism description from :func:`describe_mechanism`, such as
            ``"Channel 'leak' (IL)"``. With ``owner`` and ``source`` both
            ``None`` the messages keep their short ``Mechanism field`` form.
        source : str, optional
            Sentence appended to unit errors that says where the value came
            from, such as the ``paint`` call that declared it.

        Raises
        ------
        TypeError
            If a bare number is given for a dimensioned parameter, or a
            quantity for a dimensionless one.
        ValueError
            If the units are incompatible, or the value is empty or not finite.
        """
        _validate_like_default(value, self.default, name=name, owner=owner, source=source)
        if self.validator is not None:
            self.validator(value, name)


@dataclass(frozen=True)
class StateSpec:
    """Declare one differential state and its optional generic initial value."""

    initial: object = _OWNER_MANAGED

    @property
    def owner_managed(self) -> bool:
        """Whether the mechanism implements initialization itself."""
        return self.initial is _OWNER_MANAGED

    def validate(self, value: object, name: str) -> None:
        """Validate a state value when this declaration owns initialization."""
        if self.owner_managed:
            return
        _validate_like_default(value, self.initial, name=name)


def positive(value: object, name: str) -> None:
    """Require every canonical value to be finite and strictly positive."""
    decimal = _decimal(value)
    if np.any(decimal <= 0.0):
        raise ValueError(f"Mechanism parameter {name!r} must be > 0.")


def describe_mechanism(category: str, class_name: str, instance_name: str | None = None) -> str:
    """Name a mechanism for an error message, e.g. ``Channel 'leak' (IL)``.

    Parameters
    ----------
    category : str
        ``"channel"``, ``"ion"`` or ``"synapse"``.
    class_name : str
        Registry name or class name of the runtime mechanism.
    instance_name : str, optional
        Logical instance name; omitted when ``None`` or equal to ``class_name``.

    Returns
    -------
    str
        ``"<Category> '<instance>' (<class>)"`` or ``"<Category> '<class>'"``.
    """
    label = str(category).title()
    if instance_name is None or str(instance_name) == str(class_name):
        return f"{label} {str(class_name)!r}"
    return f"{label} {str(instance_name)!r} ({class_name})"


def describe_value(value: object) -> str:
    """Describe a parameter value and its unit, e.g. ``0.1 mV`` or ``0.1 (no unit)``."""
    is_quantity = isinstance(value, u.Quantity)
    mantissa = value.mantissa if is_quantity else value
    unit = f" {value.unit}" if is_quantity else " (no unit)"
    shape = tuple(getattr(mantissa, "shape", ()))
    if int(np.prod(shape, dtype=int)) != 1:
        return f"an array of shape {shape!r}" + (f" in{unit}" if is_quantity else unit)
    try:
        mantissa = np.asarray(mantissa).reshape(())[()]
    except Exception:  # traced values have no concrete scalar
        pass
    return f"{mantissa}{unit}"


def unit_error_message(
    name: str,
    problem: str,
    *,
    owner: str | None = None,
    source: str | None = None,
    value: object = _NO_VALUE,
) -> str:
    """Build the message shared by every parameter-unit error.

    Without ``owner`` and ``source`` this is ``Mechanism field '<name>'
    <problem>.``; otherwise ``<owner> parameter '<name>' <problem>; got
    <value>. <Source>``, with the first letter of ``source`` capitalized.
    """
    if owner is None and source is None:
        return f"Mechanism field {name!r} {problem}."
    text = f"{owner or 'Mechanism'} parameter {name!r} {problem}"
    if value is not _NO_VALUE:
        text += f"; got {describe_value(value)}"
    text += "."
    return f"{text} {source[:1].upper()}{source[1:]}" if source else text


def _validate_like_default(
    value: object,
    default: object,
    *,
    name: str,
    owner: str | None = None,
    source: str | None = None,
    declared_in: str | None = None,
) -> None:
    def message(problem: str) -> str:
        return unit_error_message(name, problem, owner=owner, source=source, value=value)

    if isinstance(default, u.Quantity):
        expected = f"{default.unit}" + (f" (declared in {declared_in})" if declared_in else "")
        if not isinstance(value, u.Quantity):
            raise TypeError(message(f"requires a quantity compatible with {expected}"))
        try:
            decimal = np.asarray(value.to_decimal(default.unit))
        except Exception as exc:
            raise ValueError(message(f"has units incompatible with {expected}")) from exc
    else:
        if isinstance(value, u.Quantity):
            raise TypeError(message("must be dimensionless"))
        decimal = np.asarray(value)
    if decimal.size == 0:
        raise ValueError(message("cannot be empty"))
    try:
        finite = np.isfinite(decimal)
    except TypeError as exc:
        raise TypeError(message("must be numeric")) from exc
    if not np.all(finite):
        raise ValueError(message("must contain only finite values"))


def _decimal(value: object) -> np.ndarray:
    if isinstance(value, u.Quantity):
        return np.asarray(value.to_decimal(value.unit))
    return np.asarray(value)
