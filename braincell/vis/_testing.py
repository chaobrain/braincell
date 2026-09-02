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

Most helpers return ``braincell.morph.Morphology`` objects built from
canned parameters so assertions can reason about exact segment lengths,
radii, and positions without re-computing them per test. The tail of the
module holds render-backend doubles used by the plot and backend tests.
"""

import importlib.util

import brainunit as u
import pytest

from braincell import Branch, Morphology

# Re-exported so the vis tests that render real reconstructions keep importing
# everything they need from this module. braincell/io/_testing.py owns them,
# since the fixtures are SWC/ASC files the io readers parse.
from braincell.io._testing import (  # noqa: F401
    ALLOWED_TYPES,
    FIXTURE_DIR,
    VALID_SWC_FIXTURES,
)

# Re-exported for the same reason: the deep chain pins tree walks as iterative
# in both packages, and braincell/morph/_testing.py owns it because the tree it
# builds is a morph object with no visualization content.
from braincell.morph._testing import make_deep_chain_tree  # noqa: F401


def make_node_tree() -> Morphology:
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


def make_soma_dend_tree() -> Morphology:
    """A soma plus one tapering *basal* dendrite, length-only.

    The tree the cell-topology tests draw. Deliberately distinct from
    :func:`make_length_only_tree`: one dendrite segment instead of two,
    and ``basal_dendrite`` instead of ``apical_dendrite``. The single
    segment is what keeps the CV count small enough to assert coverage
    fractions by hand, and the tapering radii give the frustum renderer
    something to narrow.
    """
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def make_four_type_tree() -> Morphology:
    """Soma with an apical dendrite, a basal dendrite, and an axon.

    Length-only geometry. The four distinct branch types exercise every
    entry of the default palette at once, so a colour-mapping regression
    shows up on a single render. Used by the matplotlib layout/colorbar
    tests and by ``compare2d_test.py``.
    """
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    apical = Branch.from_lengths(lengths=[50.0, 40.0] * u.um, radii=[3.0, 2.0, 1.5] * u.um, type="apical_dendrite")
    basal = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite")
    axon = Branch.from_lengths(lengths=[40.0] * u.um, radii=[1.0, 0.6] * u.um, type="axon")
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=apical, child_name="apical", parent_x=1.0)
    tree.attach(parent="soma", child_branch=basal, child_name="basal", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
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


def make_projected_node_tree() -> Morphology:
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


# =============================================================================
# Backend doubles
# =============================================================================


class FakeBackend:
    """Scene-agnostic test double that records the last :class:`RenderRequest`.

    Advertises ``supported_scene_kinds = frozenset({"2d", "3d"})`` so
    :func:`validate_backend_for_scene` accepts it for either dispatch
    direction, and returns the request unchanged so tests can inspect
    the scene that would be rendered.
    """

    name = "fake"
    supported_scene_kinds = frozenset({"2d", "3d"})

    def __init__(self) -> None:
        self.last_request = None

    def available(self) -> bool:
        return True

    def render(self, request):
        self.last_request = request
        return request


# =============================================================================
# TestCase mixins
# =============================================================================


class VisDefaultsResetMixin:
    """Restore ``braincell.vis`` global defaults around every test.

    ``plot2d``/``plot3d`` read module-level defaults that several tests
    mutate. Mixing this in ahead of :class:`unittest.TestCase` gives a clean
    slate on entry and on exit, including when the test raises.
    """

    def setUp(self) -> None:  # noqa: N802 - unittest naming
        super().setUp()
        from braincell import vis as _vis

        _vis.reset_defaults()
        self.addCleanup(_vis.reset_defaults)


# =============================================================================
# Optional plugin probes
# =============================================================================

PYTEST_BENCHMARK_AVAILABLE = importlib.util.find_spec("pytest_benchmark") is not None
"""True when the optional ``pytest-benchmark`` plugin is importable."""

needs_benchmark = pytest.mark.skipif(
    not PYTEST_BENCHMARK_AVAILABLE,
    reason="pytest-benchmark is not installed",
)
"""Skip mark for the performance baselines in ``braincell.vis``.

Apply it per function rather than as a module-level ``pytestmark``: every file
carrying benchmarks also carries ordinary tests that must keep running when the
plugin is absent.

Benchmarks must be plain pytest functions, never :class:`unittest.TestCase`
methods — pytest cannot inject the function-scoped ``benchmark`` fixture into a
``TestCase``. They pair with the ``clean_layout_cache`` fixture from
``braincell/vis/conftest.py``, which pytest supplies by name.
"""
