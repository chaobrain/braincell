# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

__version__ = "0.0.8"
__version_info__ = tuple(map(int, __version__.split(".")))

from . import mech
from . import neuron
from . import quad
from .mech import channel, ion, synapse
from ._base import (
    Channel,
    HHTypedNeuron,
    Ion,
    IonInfo,
    IonChannel,
    MixIons,
    mix_ions,
)
from ._single_compartment import SingleCompartment
from .cell import CVPolicy, Cell
from .filter import LocsetExpr, RegionExpr, SelectionCache
from .io import AscReader, NeuroMlReader, SwcIssue, SwcReadOptions, SwcReader, SwcReport
from .mech import CableProperties, CurrentClamp, DensityMechanism, PointMechanism, ProbeMechanism
from .morpho import Branch, BranchConnection, Morpho, MorphoBranch
from .quad import *

_neuron_deprecations = {
    'SingleCompartment': (
        f"braincell.neuron.SingleCompartment has been moved "
        f"into braincell.SingleCompartment",
        SingleCompartment
    ),
}

from braincell._misc import deprecation_getattr

neuron.__getattr__ = deprecation_getattr(__name__, _neuron_deprecations)
del deprecation_getattr


__all__ = [
    "AscReader",
    "Branch",
    "BranchConnection",
    "CableProperties",
    "Cell",
    "Channel",
    "CurrentClamp",
    "DensityMechanism",
    "CVPolicy",
    "HHTypedNeuron",
    "Ion",
    "IonChannel",
    "IonInfo",
    "LocsetExpr",
    "MixIons",
    "Morpho",
    "MorphoBranch",
    "NeuroMlReader",
    "PointMechanism",
    "ProbeMechanism",
    "RegionExpr",
    "SelectionCache",
    "SingleCompartment",
    "SwcIssue",
    "SwcReadOptions",
    "SwcReader",
    "SwcReport",
    "channel",
    "ion",
    "mix_ions",
    "mech",
    "neuron",
    "quad",
    "synapse",
]
