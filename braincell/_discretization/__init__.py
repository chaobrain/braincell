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

"""Static declaration-time discretization layer.

This package holds the immutable, pre-runtime representation of a
multi-compartment cell declaration:

- policy resolution decides how many control volumes each branch gets
- geometry assembly computes the physical CV facts
- mechanism lowering attaches normalized declarations to those CVs
- node construction derives the point-space structural view

The top-level import surface intentionally stays small. Public callers
should usually only need the frozen record types plus
``build_discretization(...)`` and the policy classes.
"""

from .base import (
    CV,
    CVEdge,
    CVPointMechanism,
    CVTree,
    Discretization,
    Node,
    NodeEdge,
    NodeEdgeRole,
    NodeRole,
    NodeTree,
    build_discretization,
)
from .policy import (
    CompositeByTypePolicy,
    CVPerBranch,
    CVPolicy,
    CVPolicyByTypeRule,
    DLambda,
    MaxCVLen,
)

__all__ = [
    "CV",
    "CVEdge",
    "CVPointMechanism",
    "CVTree",
    "CompositeByTypePolicy",
    "CVPerBranch",
    "CVPolicy",
    "CVPolicyByTypeRule",
    "DLambda",
    "Discretization",
    "MaxCVLen",
    "Node",
    "NodeEdge",
    "NodeEdgeRole",
    "NodeRole",
    "NodeTree",
    "build_discretization",
]
