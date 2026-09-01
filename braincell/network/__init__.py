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

This module is deliberately cheap to import. :mod:`braincell.network.event`
and :mod:`braincell.network.recording` are leaves that lower layers
(:mod:`braincell._multi_compartment`) import directly, and Python runs a
package's ``__init__`` before any of its submodules. Importing
:mod:`.connection`, :mod:`.engine`, or :mod:`.lowering` here would
therefore drag :mod:`braincell._multi_compartment` into the middle of its
own initialization.

So only the leaf :mod:`.core` is imported eagerly; the three heavyweight
public names resolve on first attribute access via :pep:`562`
``__getattr__``. ``braincell.network.Network`` behaves exactly as before —
the laziness is invisible from the outside, and
``braincell/network/__init___test.py`` guards the invariant.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .core import NetworkResult, NetworkRunResult, Population

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .connection import NetworkConnections
    from .engine import Network
    from .lowering import ConnectionBlock

_LAZY_ATTRS = {
    "ConnectionBlock": ".lowering",
    "Network": ".engine",
    "NetworkConnections": ".connection",
}

__all__ = [
    "ConnectionBlock",
    "Network",
    "NetworkConnections",
    "NetworkResult",
    "NetworkRunResult",
    "Population",
]


def __getattr__(name: str):
    """Resolve the heavyweight public names on first access."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ATTRS})
