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

"""Control-volume (CV) layer: geometry, mechanism rules, and policies.

This package freezes a :class:`braincell.morph.Morphology` together with a
:class:`CVPolicy` into an immutable list of :class:`CV` records. It has three
internal layers:

- :mod:`braincell.cv._geo` lowers morphology + policy into geometric CV data
- :mod:`braincell.cv._mech` maps ``paint`` / ``place`` declaration rules onto
  the CVs produced by ``_geo``
- :mod:`braincell.cv._cv` freezes both lower layers into the user-facing
  :class:`CV` objects exposed by ``Cell.cvs``

The :class:`CVPolicy` hierarchy in :mod:`braincell.cv._policy` is the knob
callers use to pick how a morphology is discretized.
"""

from ._cv import CV
from ._policy import (
    CompositeByTypePolicy,
    CVPerBranch,
    CVPolicy,
    CVPolicyByTypeRule,
    DLambda,
    MaxCVLen,
)

__all__ = [
    "CV",
    "CompositeByTypePolicy",
    "CVPerBranch",
    "CVPolicy",
    "CVPolicyByTypeRule",
    "DLambda",
    "MaxCVLen",
]
