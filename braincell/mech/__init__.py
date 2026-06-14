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
``brainstate``. Every declaration type inherits from the
:class:`Mechanism` marker base class.

- **Density mechanisms** — distributed over a region of a cell. See
  :class:`Density` and its concrete subclasses :class:`Channel` (for
  ion channels) and :class:`Ion` (for ion species). Both accept a
  registry key string *or* a class object for their first positional
  argument.
- **Point mechanisms** — attached to a single location. See
  :class:`Point` and its concrete subclasses :class:`CurrentClamp`,
  :class:`SineClamp`, :class:`FunctionClamp`, :class:`StateProbe`,
  :class:`MechanismProbe`, :class:`CurrentProbe`, :class:`ProbeMechanism`, and
  :class:`Synapse`. :class:`Junction` is the gap-junction point
  declaration and lives in :mod:`braincell.mech._junction`.

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

from ._base import Mechanism
from ._cable import CableProperty
from ._density import Channel, Density, Ion
from ._junction import Junction
from ._params import Params
from ._point import (
    CurrentProbe,
    CurrentClamp,
    FunctionClamp,
    MechanismProbe,
    NetStim,
    Point,
    ProbeMechanism,
    SineClamp,
    StateProbe,
    Synapse,
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
    # Base
    "Mechanism",
    # Cable properties
    "CableProperty",
    # Density mechanisms
    "Channel",
    "Density",
    "Ion",
    # Shared parameter container
    "Params",
    # Point mechanisms
    "CurrentProbe",
    "CurrentClamp",
    "FunctionClamp",
    "Junction",
    "MechanismProbe",
    "NetStim",
    "Point",
    "ProbeMechanism",
    "SineClamp",
    "StateProbe",
    "Synapse",
    # Registry
    "MechanismEntry",
    "MechanismRegistry",
    "get_registry",
    "register_channel",
    "register_ion",
    "register_synapse",
]
