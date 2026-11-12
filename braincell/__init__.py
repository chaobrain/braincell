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
from braincell import quad, mech, channel, synapse, ion, filter, morph, reduction, trainable
from braincell._base_channel import Channel, IonInfo, IonChannel, Synapse
from braincell._base_rule import ConnectionRule
from braincell._base_ion import Ion, MixIons, mix_ions
from braincell._base_neuron import HHTypedNeuron
from braincell._discretization import (
    CompositeByTypePolicy,
    CV,
    CVTree,
    CVPerBranch,
    CVPerBranchList,
    CVPolicy,
    CVPolicyByTypeRule,
    DLambda,
    Lumped,
    MaxCVLen,
    Node,
    NodeTree,
    PointPlacement,
)
from braincell._multi_compartment import (
    Cell,
    CellView,
    ChannelView,
    ClampView,
    IonView,
    RunResult,
    SynapseView,
)
from braincell.network import Network, NetworkResult
from braincell.network.connection import ConnectionView, NetworkConnections, connect
from braincell.network.event import (
    EventSequence,
    EventSource,
    EventSourceView,
    EventTable,
    NetStim,
    VoltageCrossingSource,
)
from braincell.network.recording import EventSeries, RecordingSchema, RecordingSpec, SampleBlock, observe
from braincell.reduction import (
    EventAccumulatorReduction,
    PayloadAccumulatorReduction,
    ReductionContext,
    ReductionInputGroup,
    ReductionInputGroupSchema,
    ReductionInputs,
    ReductionModel,
    ReductionOutput,
    ReductionSynapse,
    ReductionView,
    ReductionViewCollection,
    SynapticKernelAccumulatorReduction,
)
from braincell._version import __version__, __version_info__
from braincell.mech import CableProperty, CurrentClamp, FunctionClamp, SineClamp
from braincell.morph.branch import ApicalDendrite, Axon, BasalDendrite, Branch, CustomBranch, Dendrite, Soma
from braincell.morph.morphology import Morphology
from braincell import io, network, plasticity
from braincell._deprecation import removed_attributes as _removed_attributes

#: Names that braincell 0.3 removed without a deprecation period (CELL-28),
#: with what to use instead; raised as ``AttributeError`` (``ImportError`` on a
#: ``from braincell import ...`` line).
_REMOVED_ATTRIBUTES = {
    "SingleCompartment": (
        "braincell.SingleCompartment was removed in braincell 0.3. Build a point neuron with "
        "braincell.Cell.point(pop_size, length=..., radius=...) and declare its ions and channels "
        "with paint(); the cells guide (\"Coming from SingleCompartment\") translates a model line by line."
    ),
    "MultiCompartment": (
        "braincell.MultiCompartment was removed in braincell 0.3. Use braincell.Cell, the class it was an alias of."
    ),
}
__getattr__ = _removed_attributes(__name__, _REMOVED_ATTRIBUTES)

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
    "ClampView",
    "CompositeByTypePolicy",
    "ConnectionRule",
    "ConnectionView",
    "CurrentClamp",
    "CustomBranch",
    "DLambda",
    "Dendrite",
    "DiffEqGroupState",
    "DiffEqModule",
    "DiffEqSingleState",
    "DiffEqState",
    "EventAccumulatorReduction",
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
    "Lumped",
    "MaxCVLen",
    "MixIons",
    "Morphology",
    "NetStim",
    "Network",
    "NetworkConnections",
    "NetworkResult",
    "Node",
    "NodeTree",
    "PayloadAccumulatorReduction",
    "PointPlacement",
    "RecordingSchema",
    "RecordingSpec",
    "ReductionContext",
    "ReductionInputGroup",
    "ReductionInputGroupSchema",
    "ReductionInputs",
    "ReductionModel",
    "ReductionOutput",
    "ReductionSynapse",
    "ReductionView",
    "ReductionViewCollection",
    "RunResult",
    "SampleBlock",
    "SineClamp",
    "Soma",
    "Synapse",
    "SynapseView",
    "SynapticKernelAccumulatorReduction",
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
    "plasticity",
    "quad",
    "reduction",
    "state",
    "state_grouping",
    "synapse",
    "trainable",
]
