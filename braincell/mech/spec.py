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


from dataclasses import dataclass
from typing import Any

from .density import DensityMechanism

__all__ = [
    "MechanismSpec",
    "Channel",
    "Ion",
    "Synapse",
    "density_class_name",
    "density_identity",
    "density_instance_name",
    "density_name",
    "density_params",
    "density_replace_params",
    "density_signature",
    "is_density_mechanism",
]


@dataclass(frozen=True)
class MechanismSpec:
    """Declarative mechanism spec used by the cell frontend.

    This object is intentionally lightweight: it stores the user declaration
    needed to later instantiate a real runtime mechanism once layout/size is
    known.
    """

    category: str
    name: str
    class_name: str
    params: tuple[tuple[str, Any], ...] = ()


def Channel(class_name: str, /, *, name: str | None = None, **params: Any) -> MechanismSpec:
    resolved_name = str(class_name) if name is None else str(name)
    return MechanismSpec(
        category="channel",
        name=resolved_name,
        class_name=str(class_name),
        params=tuple(params.items()),
    )


def Ion(class_name: str, /, *, name: str | None = None, **params: Any) -> MechanismSpec:
    resolved_name = str(class_name) if name is None else str(name)
    return MechanismSpec(
        category="ion",
        name=resolved_name,
        class_name=str(class_name),
        params=tuple(params.items()),
    )


def Synapse(class_name: str, /, *, name: str | None = None, **params: Any) -> MechanismSpec:
    resolved_name = str(class_name) if name is None else str(name)
    return MechanismSpec(
        category="synapse",
        name=resolved_name,
        class_name=str(class_name),
        params=tuple(params.items()),
    )


def is_density_mechanism(mechanism: object) -> bool:
    if isinstance(mechanism, DensityMechanism):
        return True
    return isinstance(mechanism, MechanismSpec) and mechanism.category in {"channel", "ion"}


def density_name(mechanism: object) -> tuple[str, str]:
    return density_class_name(mechanism)


def density_class_name(mechanism: object) -> tuple[str, str]:
    if isinstance(mechanism, DensityMechanism):
        if mechanism.channel_type is not None:
            return ("channel", mechanism.channel_type)
        if mechanism.ion_type is not None:
            return ("ion", mechanism.ion_type)
        return ("density", "anonymous")
    if isinstance(mechanism, MechanismSpec):
        return (mechanism.category, mechanism.class_name)
    raise TypeError(f"Unsupported density mechanism type {type(mechanism).__name__!s}.")


def density_instance_name(mechanism: object) -> str:
    if isinstance(mechanism, DensityMechanism):
        _, class_name = density_class_name(mechanism)
        return class_name
    if isinstance(mechanism, MechanismSpec):
        return mechanism.name
    raise TypeError(f"Unsupported density mechanism type {type(mechanism).__name__!s}.")


def density_identity(mechanism: object) -> tuple[str, str]:
    return (density_instance_name(mechanism), density_class_name(mechanism)[1])


def density_params(mechanism: object) -> tuple[tuple[str, Any], ...]:
    if isinstance(mechanism, DensityMechanism):
        return mechanism.params
    if isinstance(mechanism, MechanismSpec):
        return mechanism.params
    raise TypeError(f"Unsupported density mechanism type {type(mechanism).__name__!s}.")


def density_signature(mechanism: object) -> tuple[object, ...]:
    category, class_name = density_class_name(mechanism)
    return (category, density_instance_name(mechanism), class_name, tuple(density_params(mechanism)))


def density_replace_params(
    mechanism: object,
    *,
    params: tuple[tuple[str, Any], ...],
) -> object:
    if isinstance(mechanism, DensityMechanism):
        category, class_name = density_class_name(mechanism)
        if category == "channel":
            return DensityMechanism(channel_type=class_name, params=params)
        if category == "ion":
            return DensityMechanism(ion_type=class_name, params=params)
        return DensityMechanism(params=params)
    if isinstance(mechanism, MechanismSpec):
        return MechanismSpec(
            category=mechanism.category,
            name=mechanism.name,
            class_name=mechanism.class_name,
            params=params,
        )
    raise TypeError(f"Unsupported density mechanism type {type(mechanism).__name__!s}.")
