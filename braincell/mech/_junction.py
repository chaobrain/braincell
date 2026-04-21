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

"""Gap-junction declarations.

:class:`Junction` is the declaration-layer record for a gap-junction
coupling. It is a point mechanism — it attaches to a specific location
on a cell rather than being distributed over a region — but it differs
enough from the stimulus / probe / synapse declarations in
:mod:`braincell.mech._point` that it lives in its own module.

The current implementation is a placeholder: it only records a
parameter bundle. A future revision should add a ``partner`` locset /
cell handle so multi-cell gap junctions can be expressed end-to-end.
"""

from dataclasses import dataclass, field
from typing import Any

from ._params import Params
from ._point import Point

__all__ = ["Junction"]


@dataclass(frozen=True)
class Junction(Point):
    """Gap-junction coupling declaration (placeholder).

    Parameters
    ----------
    params : Params or Mapping
        Parameter mapping for the junction (e.g. conductance). An
        empty mapping is allowed while the full ``partner`` wiring is
        still being designed.

    Notes
    -----
    Only ``params`` is currently stored. A future revision should add
    a ``partner`` field (locset / cell handle) so two-ended gap
    junctions can be expressed in a single declaration.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.mech import Junction, Params
        >>> gap = Junction(params=Params(g=1.0))
        >>> gap.params["g"]
        1.0
    """

    params: Params = field(default_factory=Params)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", Params.coerce(self.params))
