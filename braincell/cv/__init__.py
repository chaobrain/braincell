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

The package resolves a :class:`braincell.morph.Morphology` together with a
:class:`CVPolicy` and paint / place rules into an immutable
``tuple[braincell.cv.CV, ...]`` via the internal ``build_cvs`` entry point.

- :mod:`braincell.cv._policy` — CV discretization policies.
- :mod:`braincell.cv._lower` — pure-functional lowering pipeline
  (``lower``, rule dataclasses, merge helpers).
- :mod:`braincell.cv._cv` — ``CV`` dataclass and the ``build_cvs`` entry.
- :mod:`braincell.cv._debug` — helpers for reconstructing geometry from a CV.
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
