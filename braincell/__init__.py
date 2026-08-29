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


from braincell._single_compartment import (
    SingleCompartment,
)
from braincell.quad.protocol import (
    DiffEqGroupState,
    DiffEqModule,
    DiffEqSingleState,
    DiffEqState,
    IndependentIntegration,
    state,
    hidden_state,
    state_grouping,
)
from . import quad, mech, channel, synapse, ion, filter, morph, trainable
from ._base_channel import (
    Channel,
    IonInfo,
    IonChannel,
    Synapse,
)
from ._base_ion import (
    Ion,
    MixIons,
    mix_ions,
)
from ._base_neuron import HHTypedNeuron
from ._discretization import (
    CompositeByTypePolicy,
    CV,
    CVTree,
    CVPerBranch,
    CVPerBranchList,
    CVPolicy,
    CVPolicyByTypeRule,
    DLambda,
    MaxCVLen,
    Node,
    NodeTree,
    PointPlacement,
)
from ._multi_compartment import (
    Cell,
    CellView,
    ChannelView,
    IonView,
    MultiCompartment,
    RunResult,
    SynapseView,
)
from .network import (
    Network,
    NetworkResult,
)
from .network.connection import ConnectionView, NetworkConnections, connect
from .network.event import (
    EventSequence,
    EventSource,
    EventSourceView,
    EventTable,
    NetStim,
    VoltageCrossingSource,
)
from .network.recording import EventSeries, RecordingSchema, RecordingSpec, SampleBlock, observe
from ._version import (
    __version__,
    __version_info__,
)
from .mech import (
    CableProperty,
    CurrentClamp,
    FunctionClamp,
    SineClamp,
)
from .morph.branch import (
    ApicalDendrite,
    Axon,
    BasalDendrite,
    Branch,
    CustomBranch,
    Dendrite,
    Soma,
)
from .morph.morphology import (
    Morphology,
)
from . import io, network, vis

# ASCII-sorted, so a new export lands in exactly one place and the guard in
# ``__init___test.py`` can say so. Every public domain package is listed:
# ``filter`` and ``morph`` were reachable only as a side effect of some other
# module importing them, and ``braincell.io`` did not resolve at all.
__all__ = [
    "ApicalDendrite",
    "Axon",
    "BasalDendrite",
    "Branch",
    "CV",
    "CVPerBranch",
    "CVPerBranchList",
    "CVPolicy",
    "CVPolicyByTypeRule",
    "CVTree",
    "CableProperty",
    "Cell",
    "CellView",
    "Channel",
    "ChannelView",
    "CompositeByTypePolicy",
    "ConnectionView",
    "CurrentClamp",
    "CustomBranch",
    "DLambda",
    "Dendrite",
    "DiffEqGroupState",
    "DiffEqModule",
    "DiffEqSingleState",
    "DiffEqState",
    "EventSequence",
    "EventSeries",
    "EventSource",
    "EventSourceView",
    "EventTable",
    "FunctionClamp",
    "HHTypedNeuron",
    "IndependentIntegration",
    "Ion",
    "IonChannel",
    "IonInfo",
    "IonView",
    "MaxCVLen",
    "MixIons",
    "Morphology",
    "MultiCompartment",
    "NetStim",
    "Network",
    "NetworkConnections",
    "NetworkResult",
    "Node",
    "NodeTree",
    "PointPlacement",
    "RecordingSchema",
    "RecordingSpec",
    "RunResult",
    "SampleBlock",
    "SineClamp",
    "SingleCompartment",
    "Soma",
    "Synapse",
    "SynapseView",
    "VoltageCrossingSource",
    "__version__",
    "__version_info__",
    "channel",
    "connect",
    "filter",
    "hidden_state",
    "io",
    "ion",
    "mech",
    "mix_ions",
    "morph",
    "network",
    "observe",
    "quad",
    "state",
    "state_grouping",
    "synapse",
    "trainable",
    "vis",
]
