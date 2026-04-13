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

"""Shared fixture builders for ``braincell.vis`` tests.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. Helpers here are consumed by the co-located
``*_test.py`` modules; nothing in this file is part of the public API.

All helpers return ``braincell.morph.Morphology`` objects built from
canned parameters so assertions can reason about exact segment lengths,
radii, and positions without re-computing them per test.
"""

import brainunit as u

from braincell import Branch, Morphology


def make_point_tree() -> Morphology:
    """A single-branch soma with explicit 3D points.

    Useful whenever a test needs a morphology that satisfies
    ``has_full_point_geometry`` but is otherwise trivial.
    """
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


def make_length_only_tree(*, child_name: str = "dend") -> Morphology:
    """A soma plus a two-segment apical dendrite, length-only (no 3D points).

    The resulting tree has 2 branches:
      * ``soma`` — one segment, length 20 µm, radius 10 µm.
      * ``<child_name>`` — two segments (lengths 8 µm and 12 µm; radii
        2 / 1.5 / 1 µm).

    This layout is the reference fixture for testing stem/balloon layouts
    and frustum polygon generation because its geometry is small enough
    to verify by hand.
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[8.0, 12.0] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name=child_name, parent_x=1.0)
    return tree


def make_root_split_tree() -> Morphology:
    """Soma with one apical dendrite and one axon, both length-only.

    Used by the ``root_layout="type_split"`` tests that check that the
    axon and dendrite end up on opposite half-planes.
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[25.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    axon = Branch.from_lengths(
        lengths=[18.0] * u.um,
        radii=[1.0, 0.8] * u.um,
        type="axon",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


def make_two_dendrite_tree() -> Morphology:
    """Soma with two dendrite children of equal length.

    Shared between legacy-angle, balloon, and radial_360 layout tests
    that need a symmetric two-child tree.
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend_a = Branch.from_lengths(
        lengths=[15.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    dend_b = Branch.from_lengths(
        lengths=[15.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="basal_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
    tree.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)
    return tree


def make_projected_point_tree() -> Morphology:
    """Soma + apical dendrite with explicit 3D points.

    The dendrite has three points (so the projected 2D centerline has
    two segments) and a known 2D footprint for overlay resolution
    assertions under ``projection_plane='xy'``.
    """
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_points(
        points=[[20.0, 0.0, 0.0], [20.0, 40.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


def make_fan_root_partition_tree() -> Morphology:
    """Soma with children attached at left / middle / right root positions.

    The chosen ``parent_x`` values exercise the intended default fan
    root binning with the values the current morphology API allows:
      * ``0.0`` -> left sector
      * ``0.5`` -> middle sector
      * ``1.0`` -> right sector
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(lengths=[12.0] * u.um, radii=[1.8, 1.1] * u.um, type="apical_dendrite"),
        child_name="left_dend",
        parent_x=0.0,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(lengths=[12.0] * u.um, radii=[1.6, 1.0] * u.um, type="apical_dendrite"),
        child_name="mid_dend",
        parent_x=0.5,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(lengths=[12.0] * u.um, radii=[1.0, 0.7] * u.um, type="axon"),
        child_name="mid_axon",
        parent_x=0.5,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(lengths=[12.0] * u.um, radii=[1.5, 0.9] * u.um, type="basal_dendrite"),
        child_name="right_near",
        parent_x=1.0,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(lengths=[12.0] * u.um, radii=[1.5, 0.9] * u.um, type="apical_dendrite"),
        child_name="right_far",
        parent_x=1.0,
    )
    return tree
