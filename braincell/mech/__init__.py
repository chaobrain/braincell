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

"""Declarative mechanism specs for :class:`braincell.Cell`.

The :mod:`braincell.mech` package is purely declarative: it describes
*what* to install on a cell without touching runtime state, JAX, or
``brainstate``. The two main flavors are:

- **Density mechanisms** — distributed over a region of a cell. See
  :class:`DensityMechanism` and the ergonomic factories
  :func:`Channel` / :func:`Ion`.
- **Point mechanisms** — attached to a single location. See
  :class:`PointMechanism` and its concrete subclasses
  :class:`CurrentClamp`, :class:`SineClamp`, :class:`FunctionClamp`,
  :class:`ProbeMechanism`, :class:`SynapseMechanism`, and
  :class:`GapJunctionMechanism`. Use :func:`Synapse` as the keyword
  factory for synapse declarations.

Passive cable properties are recorded via :class:`CableProperty`.

Class lookup by name is handled by :class:`MechanismRegistry`
(accessed through :func:`get_registry`). Concrete channel / ion /
synapse classes register themselves with the global registry via the
:func:`register_channel` / :func:`register_ion` /
:func:`register_synapse` decorators, which run as a side effect of
importing ``braincell.channel`` / ``braincell.ion`` /
``braincell.synapse``.

See Also
--------
braincell.channel : Concrete ion-channel implementations.
braincell.ion : Concrete ion-species implementations.
braincell.synapse : Concrete synapse implementations.
"""

from ._cable import CableProperty
from ._density import Channel, DensityMechanism, Ion
from ._params import Params
from ._point import (
    CurrentClamp,
    FunctionClamp,
    GapJunctionMechanism,
    PointMechanism,
    ProbeMechanism,
    SineClamp,
    Synapse,
    SynapseMechanism,
)
from ._registry import (
    MechanismEntry,
    MechanismRegistry,
    get_registry,
    register_channel,
    register_ion,
    register_synapse,
)

__all__ = [
    # Cable properties
    "CableProperty",
    # Density mechanisms
    "Channel",
    "DensityMechanism",
    "Ion",
    # Shared parameter container
    "Params",
    # Point mechanisms
    "CurrentClamp",
    "FunctionClamp",
    "GapJunctionMechanism",
    "PointMechanism",
    "ProbeMechanism",
    "SineClamp",
    "Synapse",
    "SynapseMechanism",
    # Registry
    "MechanismEntry",
    "MechanismRegistry",
    "get_registry",
    "register_channel",
    "register_ion",
    "register_synapse",
]
