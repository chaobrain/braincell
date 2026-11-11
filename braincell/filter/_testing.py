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
are morphology objects with no selection content.

The second half is the selection-cache layer of the morphology edit table
(``braincell.morph._testing``): :data:`SELECTION_BATTERY` and
:func:`assert_selection_fresh`.
"""

import warnings
from typing import Callable, Mapping

import brainunit as u

from braincell import Branch, Morphology
from braincell.filter.cache import SelectionCache
from braincell.filter.locset import BranchPoints, LocsetMask, RootLocation, Terminals, UniformSamples, at
from braincell.filter.region import AllRegion, EmptyRegion, SubtreeRegion, branch_in
from braincell.io._testing import FIXTURE_DIR
from braincell.morph._testing import EditCase, assert_layer_fresh, make_tree_t

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
    "SELECTION_BATTERY",
    "assert_selection_fresh",
    "fixture_morphologies",
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


def fixture_morphologies() -> list[tuple[str, Morphology]]:
    """Every morphology under ``FIXTURE_DIR``: the SWC files, then the ASC files."""
    paths = sorted(FIXTURE_DIR.glob("*.swc")) + sorted(FIXTURE_DIR.glob("*.asc"))
    with warnings.catch_warnings():
        # Some fixtures repeat a sample; the reader keeps it as a zero-length segment and says so.
        warnings.filterwarnings("ignore", r"from_points\(\) produced \d+ zero-length segment", UserWarning)
        return [
            (path.name, Morphology.from_swc(path) if path.suffix == ".swc" else Morphology.from_asc(path))
            for path in paths
        ]


# --------------------------------------------------------------------------- #
# Selection-cache layer of the edit table
# --------------------------------------------------------------------------- #

#: Expressions a ``SelectionCache`` serves after an edit, one per thing a mask
#: can depend on: segment lengths, topology, branch type, the name -> index
#: mapping, and subtree membership by index and by name. Each is a composite whose operands are all hashable, because
#: the cache memoizes operands only; a top-level leaf would never be cached
#: and could not go stale.
SELECTION_BATTERY: tuple[tuple[str, object], ...] = (
    ("lengths", UniformSamples(AllRegion(), 5) + RootLocation(0.5)),
    ("topology", Terminals() + BranchPoints()),
    ("types", (branch_in("type", "basal_dendrite") | branch_in("type", "axon")) & AllRegion()),
    ("names", at("soma", 0.5) + at("axon", 1.0)),
    # Two entries, not one union: "reattach d2 under the axon" moves d2 from
    # one subtree to the other, and a union of both would not change.
    ("subtree by index", SubtreeRegion(1) | EmptyRegion()),
    ("subtree by name", SubtreeRegion("axon") | EmptyRegion()),
)


def _selection_fingerprint(morpho: Morphology, cache: SelectionCache | None) -> dict[str, object]:
    """Evaluate every battery expression on ``morpho`` through ``cache``.

    A locset reads as its ``(branch_id, branch_x)`` rows and display names, a
    region as its ``(branch_id, prox, dist)`` intervals; both are plain
    Python numbers, so the comparison is exact.
    """
    fingerprint: dict[str, object] = {}
    for label, expr in SELECTION_BATTERY:
        mask = morpho.select(expr, cache=cache)
        if isinstance(mask, LocsetMask):
            fingerprint[label] = (mask.points, mask.display_names)
        else:
            fingerprint[label] = tuple((int(b), float(p), float(d)) for b, p, d in mask.intervals)
    return fingerprint


def assert_selection_fresh(
    testcase,
    case: EditCase,
    build: Callable[[], Morphology] = make_tree_t,
) -> None:
    """Selection-cache layer: after ``case.edit``, a warm cache answers like no cache on a cold rebuild.

    Evaluates :data:`SELECTION_BATTERY` once through one
    :class:`~braincell.filter.SelectionCache` (filling it with the operand
    masks), applies the edit, evaluates the battery again through the same
    cache and compares the result with the battery evaluated without a cache
    on ``cold_rebuild`` of the edited tree.

    Parameters
    ----------
    testcase : unittest.TestCase
        Reports the failure.
    case : EditCase
        Row of the edit table to check.
    build : callable
        Builds the tree to edit.

    Raises
    ------
    AssertionError
        Naming every battery expression whose result differs.
    """

    def warm(morpho: Morphology) -> tuple[Morphology, SelectionCache]:
        return morpho, SelectionCache()

    def observe(consumer: tuple[Morphology, SelectionCache]) -> Mapping[str, object]:
        return _selection_fingerprint(*consumer)

    def reference(cold: Morphology) -> Mapping[str, object]:
        return _selection_fingerprint(cold, None)

    assert_layer_fresh(
        testcase,
        case,
        layer="selection cache",
        warm=warm,
        observe=observe,
        reference=reference,
        build=build,
    )
