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

"""Shared fixture builders for ``braincell.morph`` tests.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. Helpers here are consumed by the co-located
``*_test.py`` modules; nothing in this file is part of the public API.

Each ``make_*`` branch builder returns the canonical single-segment branch
of one type, with the dimensions the morph tests have always used. They
exist because those five literals were spelled out 41 times across
``morphology_test.py`` alone, which made the dimensions look meaningful
when they are merely arbitrary — a test that genuinely depends on a
specific radius should pass it explicitly.
"""

import brainunit as u

from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology

__all__ = [
    "make_apical",
    "make_axon",
    "make_basal",
    "make_deep_chain_tree",
    "make_dendrite",
    "make_soma",
    "make_soma_dend_tree",
]


def make_soma(*, length: float = 20.0, radius: float = 10.0) -> Branch:
    """A cylindrical soma branch of uniform radius."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=[radius, radius] * u.um, type="soma")


def make_dendrite(*, length: float = 60.0, radii: tuple[float, float] = (2.0, 1.0)) -> Branch:
    """A tapering branch of the untyped ``"dendrite"`` class."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="dendrite")


def make_basal(*, length: float = 60.0, radii: tuple[float, float] = (2.0, 1.0)) -> Branch:
    """A tapering ``"basal_dendrite"`` branch."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="basal_dendrite")


def make_apical(*, length: float = 30.0, radii: tuple[float, float] = (1.0, 0.6)) -> Branch:
    """A tapering ``"apical_dendrite"`` branch."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="apical_dendrite")


def make_axon(*, length: float = 40.0, radii: tuple[float, float] = (0.8, 0.4)) -> Branch:
    """A tapering ``"axon"`` branch."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="axon")


def make_soma_dend_tree(
    *,
    soma_length: float = 20.0,
    dend_type: str = "basal_dendrite",
    dend_length: float = 30.0,
) -> Morphology:
    """A soma with one tapering dendrite attached at its distal end.

    The smallest tree with a parent/child relationship, which is what most
    structural assertions need. Three packages built this independently --
    ``filter`` parameterized, ``vis`` and ``_discretization`` with
    byte-identical hard-coded bodies -- and all three produced the same
    two-branch tree. It lives here because the object it builds is a
    morphology with no selection, rendering, or discretization content.

    Parameters
    ----------
    soma_length : float
        Soma length in micrometres. The radius is :func:`make_soma`'s
        default 10 um.
    dend_type : str
        Branch type for the child, e.g. ``"basal_dendrite"`` or
        ``"apical_dendrite"``.
    dend_length : float
        Dendrite length in micrometres. The radii taper 2 um -> 1 um
        regardless of length.

    Returns
    -------
    Morphology
        A two-branch tree named ``soma`` -> ``dend``.
    """
    tree = Morphology.from_root(make_soma(length=soma_length), name="soma")
    child = Branch.from_lengths(
        lengths=[dend_length] * u.um,
        radii=[2.0, 1.0] * u.um,
        type=dend_type,
    )
    tree.attach(parent="soma", child_branch=child, child_name="dend", parent_x=1.0)
    return tree


def make_deep_chain_tree(n_branches: int = 1200) -> Morphology:
    """An unbranched chain of ``n_branches`` branches hanging off a soma.

    Reconstructions of long, thin neurites routinely produce chains far
    deeper than CPython's default recursion limit, so this fixture exists
    to pin the tree walks in :mod:`braincell.morph` and :mod:`braincell.vis`
    as iterative. The default depth is comfortably past the ~400-branch
    point where a recursive walk raises ``RecursionError``, while staying
    small enough to build and lay out in well under a second.

    Parameters
    ----------
    n_branches : int
        Total branch count, soma included. Must be at least 1.

    Returns
    -------
    Morphology
        A chain ``soma -> seg_0 -> seg_1 -> ...``, every dendrite branch
        carrying two 15 µm / 10 µm segments.
    """
    if n_branches < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches!r}.")
    tree = Morphology.from_root(make_soma(), name="soma")
    parent = "soma"
    for index in range(n_branches - 1):
        child = Branch.from_lengths(
            lengths=[15.0, 10.0] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="apical_dendrite",
        )
        name = f"seg_{index}"
        tree.attach(parent=parent, child_branch=child, child_name=name, parent_x=1.0)
        parent = name
    return tree
