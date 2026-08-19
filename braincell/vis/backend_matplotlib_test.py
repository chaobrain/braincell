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

"""Tests for :mod:`braincell.vis.backend_matplotlib`."""

import unittest

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from braincell.filter import AllRegion, BranchPoints, Terminals
from braincell.vis import plot2d
from braincell.vis._testing import (
    VisDefaultsResetMixin,
    make_four_type_tree,
    make_length_only_tree,
    make_node_tree,
    make_projected_node_tree,
)
from braincell.vis.backend import BackendChooser
from braincell.vis.backend_matplotlib import MatplotlibBackend


class MatplotlibBackendRenderTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_matplotlib_backend_renders_projected_scene(self) -> None:
        # make_node_tree() is one soma branch running (0,0,0) -> (10,0,1), so
        # the default xy projection must drop z and leave a single 2D segment.
        tree = make_node_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(tree, layout="projected", shape="line", backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)
        self.assertEqual(len(axes.lines), len(tree.branches))
        np.testing.assert_allclose(axes.lines[0].get_xdata(), [0.0, 10.0])
        np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.0, 0.0])
        # Morphology must not be stretched by the axes box.
        self.assertEqual(axes.get_aspect(), 1.0)

    def test_matplotlib_backend_can_render_into_existing_axes(self) -> None:
        tree = make_length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))
        fig, ax = plt.subplots(figsize=(8, 4))

        rendered_ax = plot2d(tree, layout="stem", shape="line", backend="matplotlib", chooser=chooser, ax=ax)

        self.assertIs(rendered_ax, ax)
        self.assertEqual(rendered_ax.figure, fig)
        self.assertGreaterEqual(len(rendered_ax.lines), 2)
        plt.close(fig)

    def test_matplotlib_backend_renders_frustum_scene(self) -> None:
        tree = make_length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(tree, layout="stem", shape="frustum", backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)
        self.assertGreaterEqual(len(axes.patches), 1)
        self.assertGreater(float(np.diff(axes.get_xlim())[0]), 20.0)
        self.assertGreater(float(np.diff(axes.get_ylim())[0]), 10.0)

    def test_matplotlib_backend_renders_frustum_value_edges(self) -> None:
        tree = make_length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(
            tree,
            layout="stem",
            shape="frustum",
            values=np.array([0.1, 0.9]),
            show_colorbar=False,
            branch_type_edge_colors_2d={"apical_dendrite": "#112233"},
            frustum_edge_linewidth_2d=1.25,
            backend="matplotlib",
            chooser=chooser,
        )

        self.assertGreaterEqual(len(axes.collections), 1)
        matches = []
        for collection in axes.collections:
            edgecolors = collection.get_edgecolors()
            linewidths = collection.get_linewidths()
            if edgecolors.size == 0 or linewidths.size == 0:
                continue
            matches.append(
                np.allclose(edgecolors[0][:3], np.array([17, 34, 51]) / 255.0)
                and np.allclose(linewidths, np.array([1.25]))
            )

        self.assertIn(True, matches)

    def test_matplotlib_renders_region_highlight_strokes_on_top(self) -> None:
        tree = make_length_only_tree()
        region = AllRegion().evaluate(tree)
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        ax = plot2d(
            tree,
            layout="stem",
            shape="line",
            region=region,
            backend="matplotlib",
            chooser=chooser,
        )

        # Base polylines + overlay strokes are both rendered as `ax.lines`
        # entries. The overlay strokes get a high zorder so they render
        # above the base; count them instead of asserting pixel output.
        overlay_lines = [line for line in ax.lines if line.get_zorder() >= 10_000]
        self.assertEqual(len(overlay_lines), len(tree.branches))
        plt.close(ax.figure)

    def test_matplotlib_renders_locset_markers_as_scatter(self) -> None:
        tree = make_length_only_tree()
        locset = Terminals().evaluate(tree)
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        ax = plot2d(
            tree,
            layout="stem",
            shape="line",
            locset=locset,
            backend="matplotlib",
            chooser=chooser,
        )

        # Each Marker2D becomes one `PathCollection` from `ax.scatter`.
        self.assertGreaterEqual(len(ax.collections), len(locset.points))
        plt.close(ax.figure)


class MatplotlibLayoutAndColorbarTest(VisDefaultsResetMixin, unittest.TestCase):
    """End-to-end renders that no other matplotlib test reaches.

    ``MatplotlibBackendRenderTest`` above covers the projected, stem-line
    and stem-frustum paths. These cover the two remaining layout families
    and the colourbar, which is the one part of the value pipeline that
    only appears when ``show_colorbar`` is left at its default.
    """

    def tearDown(self) -> None:
        plt.close("all")

    def _render(self, **kwargs):
        chooser = BackendChooser(backends=(MatplotlibBackend(),))
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(make_four_type_tree(), backend="matplotlib", chooser=chooser, ax=ax, **kwargs)
        return fig, ax

    def test_balloon_layout_renders_finite_lines(self) -> None:
        fig, ax = self._render(layout="balloon", shape="line")

        # 4 branches → at least one polyline each, all at finite coordinates.
        self.assertGreaterEqual(len(ax.lines), 4)
        for line in ax.lines:
            self.assertTrue(np.all(np.isfinite(line.get_xydata())))
        self.assertEqual(len(fig.axes), 1)

    def test_radial_360_layout_renders_finite_lines(self) -> None:
        fig, ax = self._render(layout="radial_360", shape="line")

        self.assertGreaterEqual(len(ax.lines), 4)
        for line in ax.lines:
            self.assertTrue(np.all(np.isfinite(line.get_xydata())))
        self.assertEqual(len(fig.axes), 1)

    def test_values_draw_a_labelled_colorbar_in_the_requested_cmap(self) -> None:
        fig, _ = self._render(
            layout="stem",
            shape="line",
            values=np.linspace(0.0, 1.0, 4),
            cmap="plasma",
            value_label="V_m",
        )

        # fig.colorbar() appends its own axes, so the colourbar is observable
        # as a second axes carrying the label.
        self.assertEqual(len(fig.axes), 2)
        colorbar_ax = fig.axes[1]
        self.assertEqual(colorbar_ax.get_ylabel(), "V_m")
        self.assertEqual(fig.axes[0].collections[0].cmap.name, "plasma")

    def test_explicit_bounds_reach_the_collection_norm(self) -> None:
        # Values span -65..-50 but the caller pins a wider scale; the norm
        # must honour the request rather than the data range.
        _, ax = self._render(
            layout="stem",
            shape="frustum",
            values=np.linspace(-65.0, -50.0, 4),
            vmin=-70.0,
            vmax=-40.0,
            show_colorbar=False,
        )

        norms = [collection.norm for collection in ax.collections if collection.norm is not None]
        self.assertTrue(norms)
        self.assertAlmostEqual(norms[0].vmin, -70.0)
        self.assertAlmostEqual(norms[0].vmax, -40.0)


if __name__ == "__main__":
    unittest.main()
