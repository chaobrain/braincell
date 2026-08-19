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

"""Tests for :mod:`braincell.vis.compare2d`."""

import unittest

import matplotlib.axes
import matplotlib.pyplot as plt

from braincell.vis._testing import (
    PYTEST_MPL_AVAILABLE,
    VisDefaultsResetMixin,
    image_comparison,
    make_four_type_tree,
    make_length_only_tree,
)
from braincell.vis.backend import BackendChooser
from braincell.vis.backend_matplotlib import MatplotlibBackend
from braincell.vis.compare2d import compare_layouts_2d


class CompareLayouts2dTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_compare_layouts_2d_renders_side_by_side_matplotlib_figure(self) -> None:
        tree = make_length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        fig, axes = compare_layouts_2d(tree, chooser=chooser)

        self.assertEqual(len(axes), 4)
        self.assertEqual([ax.get_title() for ax in axes], ["Fan", "Stem", "Balloon", "Radial 360"])
        self.assertTrue(all(isinstance(ax, matplotlib.axes.Axes) for ax in axes))
        self.assertGreaterEqual(sum(len(ax.lines) for ax in axes), 3)
        plt.close(fig)


@unittest.skipUnless(PYTEST_MPL_AVAILABLE, "pytest-mpl is not installed")
class CompareLayouts2dBaselineImageTest(unittest.TestCase):
    """Baseline-image regression figures for :func:`braincell.vis.compare2d.compare_layouts_2d`.

    Each test builds a figure and returns it; ``image_comparison`` hands it
    to pytest-mpl when the plugin is installed. Skipped wholesale otherwise —
    without the plugin the marker is inert and the figure would never be
    compared, so there is nothing left to assert.
    """

    def tearDown(self) -> None:
        plt.close("all")

    @image_comparison("compare_layouts_baseline.png")
    def test_compare_layouts_baseline(self):
        tree = make_four_type_tree()
        fig, _ = compare_layouts_2d(tree, figsize=(10, 4))
        return fig


if __name__ == "__main__":
    unittest.main()
