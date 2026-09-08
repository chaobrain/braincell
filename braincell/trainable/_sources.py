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

"""Immutable declarations for trainable parameter mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import brainstate

__all__ = ["ParameterSource", "parameter", "parameterized", "scale"]

_GROUPS = frozenset({"row", "population", "cv", "all"})


@dataclass(frozen=True)
class DirectSource:
    initial: object | None
    group_by: str
    transform: object
    name: str | None


@dataclass(frozen=True)
class ScaleSource:
    parameter: brainstate.nn.Param | None
    group_by: str
    transform: object | None
    name: str | None


@dataclass(frozen=True)
class ParameterizedSource:
    function: Callable
    arguments: tuple[tuple[str, object], ...]


ParameterSource = DirectSource | ScaleSource | ParameterizedSource


def parameter(
    initial: object | None = None,
    *,
    group_by: str = "row",
    transform=None,
    name: str | None = None,
) -> DirectSource:
    """Create a direct physical trainable-parameter source."""
    return DirectSource(initial, _group(group_by), _transform(transform), _name(name))


def scale(
    parameter: brainstate.nn.Param | None = None,
    *,
    group_by: str = "all",
    transform=None,
    name: str | None = None,
) -> ScaleSource:
    """Create a frozen-baseline, dimensionless scale source."""
    if parameter is not None and not isinstance(parameter, brainstate.nn.Param):
        raise TypeError("scale(parameter=...) expects brainstate.nn.Param or None.")
    if parameter is not None and transform is not None:
        raise ValueError("An existing nn.Param already owns its transform; do not pass transform again.")
    return ScaleSource(parameter, _group(group_by), transform, _name(name))


def parameterized(function: Callable, /, **arguments: object) -> ParameterizedSource:
    """Create a source that maps CV context and explicit arguments to a field."""
    if not callable(function):
        raise TypeError("parameterized() first argument must be callable.")
    return ParameterizedSource(function, tuple(arguments.items()))


def _group(value: str) -> str:
    if value not in _GROUPS:
        raise ValueError(f"group_by must be one of {tuple(sorted(_GROUPS))!r}, got {value!r}.")
    return value


def _transform(value):
    return brainstate.nn.IdentityT() if value is None else value


def _name(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("Trainable root name must be a non-empty string or None.")
    return value
