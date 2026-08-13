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

"""Performance baselines for :mod:`braincell.vis` via ``pytest-benchmark``.

The module is skipped entirely when ``pytest-benchmark`` is not
installed, so the base test suite never pays for it. When the plugin
*is* present, ``pytest --benchmark-only`` measures three hot paths on
small / medium / large synthetic morphologies:

1. :func:`build_layout_branches_2d` — the 2D layout engine.
2. :func:`build_render_scene_2d` — scene construction (uses the cache).
3. :func:`plot2d` — end-to-end matplotlib render.

The exact numbers are machine-dependent; CI should compare against a
stored baseline via ``pytest-benchmark``'s ``compare`` subcommand. The
point of committing the test is to have a reproducible trigger.

These are written as plain pytest functions rather than
``unittest.TestCase`` methods on purpose. ``benchmark`` is a
function-scoped pytest fixture, and pytest cannot inject fixtures into
``TestCase`` methods — doing so raises ``TypeError: ... missing 1
required positional argument: 'benchmark'``. As ``TestCase`` classes
these benchmarks failed whenever the plugin was installed and merely
skipped when it was not, so they had never actually run.
"""

import importlib.util

import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from braincell import Branch, Morphology
from braincell.vis import plot2d
from braincell.vis.layout import build_layout_branches_2d, get_default_layout_cache
from braincell.vis.scene2d import build_render_scene_2d

_pytest_benchmark_available = importlib.util.find_spec("pytest_benchmark") is not None

pytestmark = pytest.mark.skipif(
    not _pytest_benchmark_available,
    reason="pytest-benchmark is not installed",
)


@pytest.fixture(autouse=True)
def _clear_layout_cache():
    """Clear the shared layout cache before each benchmark, close figures after."""
    get_default_layout_cache().clear()
    yield
    plt.close("all")


def _synthetic_tree(n_branches: int) -> Morphology:
    """Build a length-only morphology with ``n_branches`` total branches.

    Every branch after the soma is attached to the previous branch's
    distal end, so the whole thing is one long chain. Chain morphologies
    are representative enough for scaling benchmarks and build
    deterministically in O(n_branches).
    """
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
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


# ---------------------------------------------------------------------------
# Layout engine — the 2D layout build on three morphology sizes.
# ---------------------------------------------------------------------------


# The medium / large sizes are chains deeper than CPython's recursion
# limit. They used to be marked xfail(RecursionError) because every tree
# walk in the layout engine recursed once per branch; the walks are
# iterative now (see braincell/vis/_traversal.py), so these must simply
# pass. braincell/vis/layout/_dispatch_test.py pins that behaviour for
# runs where pytest-benchmark is absent.
@pytest.mark.parametrize(
    "n_branches",
    [50, 500, 2000],
    ids=["small", "medium", "large"],
)
def test_layout_build(benchmark, n_branches: int) -> None:
    tree = _synthetic_tree(n_branches)
    benchmark(lambda: build_layout_branches_2d(tree, mode="tree", use_cache=False))


# ---------------------------------------------------------------------------
# Scene assembly — layout cache plus scene construction.
# ---------------------------------------------------------------------------


def test_scene_small_no_values(benchmark) -> None:
    tree = _synthetic_tree(50)
    benchmark(lambda: build_render_scene_2d(tree, layout="stem", shape="line"))


def test_scene_medium_with_values(benchmark) -> None:
    from braincell.vis.scene import OverlaySpec, ValueSpec

    tree = _synthetic_tree(500)
    n = len(tree.branches)
    values = np.linspace(0.0, 1.0, n)
    overlay = OverlaySpec(values=ValueSpec(values=values))
    benchmark(lambda: build_render_scene_2d(tree, layout="stem", shape="line", overlay=overlay))


# ---------------------------------------------------------------------------
# End-to-end plot2d render through the matplotlib backend.
# ---------------------------------------------------------------------------


def test_plot2d_small(benchmark) -> None:
    tree = _synthetic_tree(50)

    def _render():
        fig, ax = plt.subplots()
        plot2d(tree, layout="stem", shape="line", ax=ax)
        plt.close(fig)

    benchmark(_render)


def test_plot2d_medium_values(benchmark) -> None:
    tree = _synthetic_tree(500)
    n = len(tree.branches)
    values = np.linspace(0.0, 1.0, n)

    def _render():
        fig, ax = plt.subplots()
        plot2d(tree, layout="stem", shape="line", values=values, ax=ax, show_colorbar=False)
        plt.close(fig)

    benchmark(_render)
