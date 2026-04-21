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

"""Image-based regression tests for :mod:`braincell.vis` (pytest-mpl).

These tests build a set of canonical figures and compare them against
baseline PNGs committed under ``braincell/vis/_baseline_images``.
They are skipped in any environment where :mod:`pytest_mpl` is not
installed, so the base test suite does not gain an extra runtime
dependency.

To (re)generate baselines:

.. code-block:: bash

    pytest braincell/vis/visual_regression_test.py \\
        --mpl-generate-path=braincell/vis/_baseline_images

The ``--mpl`` flag turns on comparison mode; without it, each test
still runs its figure-construction code (so we still catch syntax and
build errors even without the plugin).
"""

import importlib.util
import unittest

import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

_pytest_mpl_available = importlib.util.find_spec("pytest_mpl") is not None

if _pytest_mpl_available:
    import pytest
else:  # pragma: no cover - skip path
    pytest = None  # type: ignore[assignment]


def _image_comparison(filename: str, *, tolerance: float = 25.0):
    """Decorator that adds ``@pytest.mark.mpl_image_compare`` when available."""
    if not _pytest_mpl_available:
        def _decorator(func):
            return func

        return _decorator
    return pytest.mark.mpl_image_compare(
        filename=filename,
        baseline_dir="_baseline_images",
        tolerance=tolerance,
    )


def _length_only_tree():
    from braincell import Branch, Morphology

    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    apical = Branch.from_lengths(lengths=[50.0, 40.0] * u.um, radii=[3.0, 2.0, 1.5] * u.um, type="apical_dendrite")
    basal = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite")
    axon = Branch.from_lengths(lengths=[40.0] * u.um, radii=[1.0, 0.6] * u.um, type="axon")
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=apical, child_name="apical", parent_x=1.0)
    tree.attach(parent="soma", child_branch=basal, child_name="basal", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


@unittest.skipUnless(_pytest_mpl_available, "pytest-mpl is not installed")
class VisualRegressionTest(unittest.TestCase):
    """Baseline-image regression tests for the vis public API.

    Each test builds a figure and returns it. The ``@_image_comparison``
    decorator takes over when pytest-mpl is installed. Without the
    plugin the tests still exercise the figure-construction code but
    skip the pixel comparison.
    """

    def tearDown(self) -> None:
        plt.close("all")

    @_image_comparison("stem_frustum_baseline.png")
    def test_stem_frustum_baseline(self):
        from braincell.vis import plot2d

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="stem", shape="frustum", ax=ax)
        return fig

    @_image_comparison("stem_line_baseline.png")
    def test_stem_line_baseline(self):
        from braincell.vis import plot2d

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="stem", shape="line", ax=ax)
        return fig

    @_image_comparison("balloon_line_baseline.png")
    def test_balloon_line_baseline(self):
        from braincell.vis import plot2d

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="balloon", shape="line", ax=ax)
        return fig

    @_image_comparison("radial_line_baseline.png")
    def test_radial_line_baseline(self):
        from braincell.vis import plot2d

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="radial_360", shape="line", ax=ax)
        return fig

    @_image_comparison("values_per_branch_baseline.png")
    def test_values_per_branch_baseline(self):
        from braincell.vis import plot2d

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(5, 4))
        plot2d(
            tree,
            layout="stem",
            shape="line",
            values=np.linspace(0.0, 1.0, len(tree.branches)),
            cmap="plasma",
            value_label="V_m",
            ax=ax,
        )
        return fig

    @_image_comparison("values_frustum_baseline.png")
    def test_values_frustum_baseline(self):
        from braincell.vis import plot2d

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(5, 4))
        plot2d(
            tree,
            layout="stem",
            shape="frustum",
            values=np.linspace(-65.0, -50.0, len(tree.branches)),
            cmap="viridis",
            vmin=-70.0,
            vmax=-40.0,
            value_label="V_m",
            ax=ax,
        )
        return fig

    @_image_comparison("dendrogram_baseline.png")
    def test_dendrogram_baseline(self):
        from braincell.vis import plot_dendrogram

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_dendrogram(tree, ax=ax)
        return fig

    @_image_comparison("topology_baseline.png")
    def test_topology_baseline(self):
        from braincell.vis import plot_topology

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_topology(tree, ax=ax)
        return fig

    @_image_comparison("sholl_baseline.png")
    def test_sholl_baseline(self):
        from braincell.vis import plot_sholl

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_sholl(tree, ax=ax, step_um=10.0)
        return fig

    @_image_comparison("branch_order_histogram_baseline.png")
    def test_branch_order_histogram_baseline(self):
        from braincell.vis import plot_branch_order_histogram

        tree = _length_only_tree()
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_branch_order_histogram(tree, ax=ax)
        return fig

    @_image_comparison("compare_layouts_baseline.png")
    def test_compare_layouts_baseline(self):
        from braincell.vis.compare2d import compare_layouts_2d

        tree = _length_only_tree()
        fig, _ = compare_layouts_2d(tree, figsize=(10, 4))
        return fig

    @_image_comparison("projected_scene_baseline.png")
    def test_projected_scene_baseline(self):
        from braincell import Branch, Morphology
        from braincell.vis import plot2d

        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.from_points(
            points=[[20.0, 0.0, 0.0], [20.0, 50.0, 0.0], [20.0, 100.0, 0.0]] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="projected", shape="line", ax=ax)
        return fig


if __name__ == "__main__":
    unittest.main()
