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

"""Shared fixture builders for ``braincell.filter`` tests.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. Helpers here are consumed by the co-located
``*_test.py`` modules; nothing in this file is part of the public API.

The per-type branch builders are re-exported rather than redefined:
``braincell/morph/_testing.py`` owns them, because the objects they build
are morphology objects with no selection content. This is the same
re-export pattern ``braincell/vis/_testing.py`` uses for
``braincell/io/_testing.py``'s fixture paths.
"""

import brainunit as u

from braincell import Branch, Morphology

# Re-exported so the filter tests keep a single import site for fixtures.
from braincell.morph._testing import (  # noqa: F401
    make_apical,
    make_axon,
    make_basal,
    make_dendrite,
    make_soma,
    make_soma_dend_tree,
)

__all__ = [
    "make_apical",
    "make_axon",
    "make_basal",
    "make_dendrite",
    "make_soma",
    "make_single_branch_tree",
    "make_soma_dend_tree",
]


def make_single_branch_tree(*, length: float = 10.0, radius: float = 1.0) -> Morphology:
    """A one-branch morphology whose single segment has a uniform radius."""
    root = Branch.from_lengths(lengths=[length] * u.um, radii=[radius, radius] * u.um, type="soma")
    return Morphology.from_root(root, name="soma")
