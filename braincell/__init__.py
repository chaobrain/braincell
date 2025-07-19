# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


__version__ = "0.0.4"

from . import channel
from . import ion
from . import neuron
from ._base import (
    HHTypedNeuron,
    IonChannel,
    Ion,
    Channel,
    MixIons,
    mix_ions,
    IonInfo,
)
from ._integrator import *
from ._morphology import (
    Section,
    Morphology,
)
from ._multi_compartment import (
    MultiCompartment,
)
from ._protocol import (
    DiffEqState,
    DiffEqModule,
)
from ._single_compartment import (
    SingleCompartment,
)

from braincell._misc import deprecation_getattr
_deprecations = {
    'SingleCompartment': (
        f"braincell.neuron.SingleCompartment has been moved "
        f"into braincell.SingleCompartment",
        SingleCompartment
    ),
    'MultiCompartment': (
        f"braincell.neuron.MultiCompartment has been moved "
        f"into braincell.MultiCompartment",
        MultiCompartment
    ),
}

neuron.__getattr__ = deprecation_getattr(__name__, _deprecations)
del deprecation_getattr
