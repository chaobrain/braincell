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

"""First-class population, connection, and runtime APIs.

This package is imported from the middle of
:mod:`braincell._multi_compartment`'s own initialization:
:mod:`braincell._multi_compartment.cell` imports
:mod:`braincell.network.event` and :mod:`braincell.network.recording` at
module scope, and Python runs a package's ``__init__`` before any of its
submodules. So when this file runs, ``braincell._multi_compartment`` is
present in :data:`sys.modules` but only partially executed.

Importing :mod:`.connection`, :mod:`.engine`, and :mod:`.pairing` here is
still safe, because none of them import a *name* from that partially
executed package root -- they import its submodules
(:mod:`~braincell._multi_compartment.probes`,
:mod:`~braincell._multi_compartment.run`,
:mod:`~braincell._multi_compartment.synapses`), which Python resolves
against a partial parent without complaint.

That is the invariant this package depends on, and
``braincell/network/__init___test.py`` checks it directly rather than
maintaining a list of modules presumed dangerous.
"""

from .connection import NetworkConnections
from .core import NetworkResult, Population
from .engine import Network

__all__ = [
    "Network",
    "NetworkConnections",
    "NetworkResult",
    "Population",
]
