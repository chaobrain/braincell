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

"""Shared fixtures for the :mod:`braincell._discretization` test modules.

The leading underscore keeps pytest from collecting this file as a test
module. Everything here is a builder used by more than one sibling
``*_test.py``; a fixture needed by exactly one test module stays in that
module.
"""

import brainunit as u
import numpy as np

from braincell._discretization.geometry import build_cv_geometry

# Re-exported: ``CableProperty`` is a ``braincell.mech`` type, so its
# builder lives with the package that owns it.
from braincell.mech._testing import make_cable  # noqa: F401
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology

__all__ = [
    "build_geo",
    "make_branch",
    "make_cable",
    "make_single_branch_morpho",
    "make_two_branch_morpho",
]


def make_branch(lengths: list[float], radii: list[float], type: str = "dendrite") -> Branch:
    """Build a single branch from ``um`` lengths and ``um`` radii."""
    return Branch.from_lengths(
        lengths=np.asarray(lengths) * u.um,
        radii=np.asarray(radii) * u.um,
        type=type,
    )


def make_single_branch_morpho(type: str = "soma") -> Morphology:
    """A one-branch morphology: 10 um long, constant 2 um radius."""
    return Morphology.from_root(
        make_branch([10.0], [2.0, 2.0], type=type),
        name=type,
    )


def make_two_branch_morpho() -> Morphology:
    """A soma with one tapering basal dendrite attached."""
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def build_geo(morpho, bounds):
    """Run :func:`build_cv_geometry` and unpack it to ``(geos, branch_to_cv_ids)``."""
    geometry = build_cv_geometry(morpho, bounds)
    return geometry.geos, geometry.branch_to_cv_ids
