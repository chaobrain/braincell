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
  :class:`SineClamp`, :class:`FunctionClamp` and :class:`Synapse`. :class:`Junction` is the gap-junction point
  declaration and lives in :mod:`braincell.mech._junction`.

Passive cable properties are recorded via :class:`CableProperty`.

Alongside the declarations, this package owns the two *contract*
vocabularies a mechanism author writes against: the event-input
contracts (:class:`EventInput`, :class:`NoEventInput`,
:class:`TriggerEventInput`, :class:`ScalarEventInput`) that say what
discrete events a mechanism can consume, and the field schema
(:class:`ParameterSpec` and :class:`StateSpec`) that
vectorized runtime synapse models use to declare their parameters and
states. Both are pure declarations, which keeps this package a leaf:
:mod:`braincell.mech` imports nothing from :mod:`braincell`. The event
*sources* that emit these events live in :mod:`braincell.network.event`.

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

from braincell._deprecation import removed_attributes as _removed_attributes
from braincell.mech._base import Mechanism
from braincell.mech._cable import CableProperty
from braincell.mech._context import CVContext
from braincell.mech._density import Channel, Density, Ion
from braincell.mech._event_contract import EventInput, NoEventInput, ScalarEventInput, TriggerEventInput
from braincell.mech._junction import Junction
from braincell.mech._params import Params
from braincell.mech._synapse_schema import ParameterSpec, StateSpec, positive
from braincell.mech._point import (
    CurrentClamp,
    FunctionClamp,
    Point,
    SineClamp,
    Synapse,
)
from braincell.mech._registry import (
    MechanismEntry,
    MechanismRegistry,
    get_registry,
    register_channel,
    register_ion,
    register_synapse,
)

# Grouped by kind rather than sorted: the headings below are the map of
# what this package declares. ``__init___test`` checks membership and
# uniqueness but deliberately does not require ASCII order.
__all__ = [
    # Base
    "Mechanism",
    # Cable properties
    "CableProperty",
    "CVContext",
    # Density mechanisms
    "Channel",
    "Density",
    "Ion",
    # Shared parameter container
    "Params",
    # Event-input contracts
    "EventInput",
    "NoEventInput",
    "ScalarEventInput",
    "TriggerEventInput",
    # Runtime synapse field schema
    "ParameterSpec",
    "StateSpec",
    "positive",
    # Point mechanisms
    "CurrentClamp",
    "FunctionClamp",
    "Junction",
    "Point",
    "SineClamp",
    "Synapse",
    # Registry
    "MechanismEntry",
    "MechanismRegistry",
    "get_registry",
    "register_channel",
    "register_ion",
    "register_synapse",
]


def _removed_probe(name: str, example: str) -> str:
    return (
        f"braincell.mech.{name} was removed in braincell 0.3. Declare a recording on the Cell instead of placing "
        f"a probe: {example}; Cell.run returns it in RunResult.samples (or traces) under that name."
    )


#: The legacy probe declarations braincell 0.3 removed without a deprecation
#: period (API-15), with what to write instead; raised as ``AttributeError``
#: (``ImportError`` on a ``from braincell.mech import ...`` line).
__getattr__ = _removed_attributes(
    __name__,
    {
        "StateProbe": _removed_probe(
            "StateProbe", 'cell.loc(locset).record("V", braincell.observe.state("V")) for StateProbe()'
        ),
        "MechanismProbe": _removed_probe(
            "MechanismProbe",
            'cell.loc(locset).record("m", braincell.observe.channel(name="Na").state("m")) for '
            'MechanismProbe(mechanism="Na", field="m"); observe.ion(name=...) and observe.synapse(name=...) '
            "select ions and synapses",
        ),
        "CurrentProbe": _removed_probe(
            "CurrentProbe",
            'cell.loc(locset).record("i", braincell.observe.channel(name="Na").current()) for '
            'CurrentProbe(mechanism="Na"), and braincell.observe.ion(name="na").current() for CurrentProbe(ion="na")',
        ),
        "ProbeMechanism": (
            "braincell.mech.ProbeMechanism was removed in braincell 0.3; no run ever sampled it. Declare a "
            'recording on the Cell instead: cell.loc(locset).record("V", braincell.observe.state("V")) records the '
            "membrane voltage, and observe.channel/ion/synapse(name=...).state(field) a mechanism state."
        ),
    },
)
