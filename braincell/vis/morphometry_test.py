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


import unittest

from braincell import vis as morpho_vis
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braincell import Branch, Morphology
from braincell.vis._testing import (
    make_length_only_tree,
    make_projected_point_tree,
    make_root_split_tree,
    make_two_dendrite_tree,
)
from braincell.vis.morphometry import (
    ShollProfile,
    compute_sholl_profile,
    plot_branch_order_histogram,
    plot_dendrogram,
    plot_sholl,
    plot_topology,
)


class PlotDendrogramTest(unittest.TestCase):
    def setUp(self) -> None:
        morpho_vis.reset_defaults()
        self.addCleanup(morpho_vis.reset_defaults)

    def tearDown(self) -> None:
        plt.close("all")

    def test_dendrogram_draws_one_line_per_branch(self) -> None:
        tree = make_two_dendrite_tree()
        ax = plot_dendrogram(tree)
        # 3 branches → 3 horizontal strokes + 1 vertical connector.
        self.assertGreaterEqual(len(ax.lines), len(tree.branches))

    def test_dendrogram_x_axis_matches_total_length(self) -> None:
        tree = make_length_only_tree()
        ax = plot_dendrogram(tree)
        xs = np.concatenate([line.get_xdata() for line in ax.lines])
        total_length_um = float(tree.total_length.to_decimal(u.um))
        self.assertAlmostEqual(float(np.max(xs)), total_length_um, places=4)

    def test_dendrogram_rejects_non_morphology(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects Morphology"):
            plot_dendrogram("not-a-morphology")  # type: ignore[arg-type]

    def test_dendrogram_uses_shared_branch_type_palette(self) -> None:
        tree = make_two_dendrite_tree()
        morpho_vis.configure_defaults(branch_type_colors={"apical_dendrite": "#445566"})

        ax = plot_dendrogram(tree)

        line_colors = [line.get_color() for line in ax.lines]
        self.assertIn((68 / 255.0, 85 / 255.0, 102 / 255.0), line_colors)


class PlotTopologyTest(unittest.TestCase):
    def setUp(self) -> None:
        morpho_vis.reset_defaults()
        self.addCleanup(morpho_vis.reset_defaults)

    def tearDown(self) -> None:
        plt.close("all")

    def test_topology_uses_branch_order_for_x_axis(self) -> None:
        tree = make_root_split_tree()
        ax = plot_topology(tree)
        # Root at x=0..1, children at x=1..2.
        x_bounds = np.concatenate([line.get_xdata() for line in ax.lines])
        self.assertEqual(float(np.min(x_bounds)), 0.0)
        self.assertAlmostEqual(float(np.max(x_bounds)), 2.0)

    def test_topology_uses_shared_branch_type_palette(self) -> None:
        tree = make_root_split_tree()
        morpho_vis.configure_defaults(branch_type_colors={"axon": "#334488"})

        ax = plot_topology(tree)

        line_colors = [line.get_color() for line in ax.lines]
        self.assertIn((51 / 255.0, 68 / 255.0, 136 / 255.0), line_colors)


class ShollAnalysisTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_sholl_profile_on_length_only_tree(self) -> None:
        tree = make_length_only_tree()
        profile = compute_sholl_profile(tree, step_um=10.0)
        self.assertIsInstance(profile, ShollProfile)
        self.assertEqual(profile.radii_um.shape, profile.intersections.shape)
        self.assertTrue(np.all(profile.intersections >= 0))

    def test_sholl_profile_on_point_tree(self) -> None:
        tree = make_projected_point_tree()
        profile = compute_sholl_profile(tree, step_um=20.0)
        self.assertGreaterEqual(profile.radii_um.size, 1)

    def test_sholl_profile_rejects_non_positive_step(self) -> None:
        tree = make_length_only_tree()
        with self.assertRaisesRegex(ValueError, "step_um must be > 0"):
            compute_sholl_profile(tree, step_um=0.0)

    def test_plot_sholl_returns_ax_with_one_line(self) -> None:
        tree = make_length_only_tree()
        ax = plot_sholl(tree, step_um=5.0)
        self.assertGreaterEqual(len(ax.lines), 1)


class BranchOrderHistogramTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_histogram_root_only_tree_has_one_bar(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")

        ax = plot_branch_order_histogram(tree)
        # One bar for order 0 (the root).
        from matplotlib.patches import Rectangle

        bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
        self.assertGreaterEqual(len(bars), 1)

    def test_histogram_counts_per_order(self) -> None:
        tree = make_two_dendrite_tree()
        ax = plot_branch_order_histogram(tree)
        from matplotlib.patches import Rectangle

        bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
        # 2 bars: order 0 (soma) and order 1 (two dendrites).
        self.assertEqual(len(bars), 2)
        # Order-1 bar should be twice the height of order-0.
        heights = sorted(bar.get_height() for bar in bars)
        self.assertEqual(heights, [1, 2])


if __name__ == "__main__":
    unittest.main()
