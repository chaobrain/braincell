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

"""Tests for :mod:`braincell.vis.plot2d`."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from braincell.vis import plot2d
from braincell.vis._testing import (
    BaselineMixin,
    FakeBackend,
    VisDefaultsResetMixin,
    make_baseline_tree,
    make_deep_chain_tree,
    make_length_only_tree,
    make_node_tree,
    needs_benchmark,
)
from braincell.vis.backend import BackendChooser


class Plot2dDispatchTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_plot2d_defaults_to_fan_frustum(self) -> None:
        tree = make_node_tree()
        backend = FakeBackend()

        request = plot2d(tree, chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.dimensionality, "2d")
        self.assertEqual(request.layout, "fan")
        self.assertEqual(request.shape, "frustum")
        self.assertEqual(request.scene.layout, "fan")
        self.assertEqual(request.scene.shape, "frustum")
        self.assertEqual(request.scene.projection_plane, None)
        self.assertTrue(all(polygon.edge_color_rgb is not None for polygon in request.scene.polygons))
        self.assertTrue(all(polygon.edge_color_rgb != polygon.color_rgb for polygon in request.scene.polygons))
        self.assertTrue(all(abs(polygon.edge_linewidth - 0.9) < 1e-9 for polygon in request.scene.polygons))

    def test_plot2d_line_shape_accepts_length_only_morphology(self) -> None:
        tree = make_length_only_tree()
        backend = FakeBackend()

        request = plot2d(tree, layout="stem", shape="line", chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.layout, "stem")
        self.assertEqual(request.shape, "line")
        self.assertEqual(request.scene.layout, "stem")
        self.assertEqual(request.scene.shape, "line")
        self.assertEqual(len(request.scene.polylines), 3)
        self.assertEqual(len(request.scene.polygons), 0)

    def test_plot2d_frustum_shape_accepts_length_only_morphology(self) -> None:
        tree = make_length_only_tree()
        backend = FakeBackend()

        request = plot2d(tree, layout="stem", shape="frustum", chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.layout, "stem")
        self.assertEqual(request.shape, "frustum")
        self.assertEqual(request.scene.layout, "stem")
        self.assertEqual(request.scene.shape, "frustum")
        self.assertEqual(len(request.scene.polygons), 3)
        self.assertEqual(len(request.scene.polylines), 0)

    def test_plot2d_projected_layout_requires_points(self) -> None:
        tree = make_length_only_tree()

        with self.assertRaisesRegex(ValueError, "layout='stem'.*shape='line'.*shape='frustum'"):
            plot2d(tree, layout="projected", shape="line", chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot2d_rejects_unknown_shape(self) -> None:
        tree = make_node_tree()

        with self.assertRaisesRegex(ValueError, "Unsupported 2D shape"):
            plot2d(tree, layout="stem", shape="layout", chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot2d_projected_layout_rejects_frustum_shape(self) -> None:
        tree = make_node_tree()

        with self.assertRaisesRegex(ValueError, "layout='projected' only supports shape='line'"):
            plot2d(tree, layout="projected", shape="frustum", chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot2d_per_call_2d_style_overrides_do_not_leak(self) -> None:
        backend = FakeBackend()

        inside = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="frustum",
            branch_type_colors={"apical_dendrite": "#445566"},
            branch_type_edge_colors_2d={"apical_dendrite": "#112233"},
            frustum_edge_linewidth_2d=1.5,
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(inside.scene.polygons[1].color_rgb, (68, 85, 102))
        self.assertEqual(inside.scene.polygons[1].edge_color_rgb, (17, 34, 51))
        self.assertAlmostEqual(inside.scene.polygons[1].edge_linewidth, 1.5)

        after = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="frustum",
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(after.scene.polygons[1].color_rgb, (214, 173, 98))
        self.assertEqual(after.scene.polygons[1].edge_color_rgb, (154, 125, 71))
        self.assertAlmostEqual(after.scene.polygons[1].edge_linewidth, 0.9)

    def test_plot2d_frustum_values_keep_edge_style_overrides(self) -> None:
        backend = FakeBackend()

        rendered = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="frustum",
            values=np.array([0.1, 0.9]),
            branch_type_edge_colors_2d={"apical_dendrite": "#112233"},
            frustum_edge_linewidth_2d=1.25,
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(len(rendered.scene.polygon_value_batches), len(make_length_only_tree().branches))
        self.assertEqual(rendered.scene.polygon_value_batches[1].edge_color_rgb, (17, 34, 51))
        self.assertAlmostEqual(rendered.scene.polygon_value_batches[1].edge_linewidth, 1.25)


# ---------------------------------------------------------------------------
# Performance baselines (pytest-benchmark). See ``needs_benchmark`` in
# braincell/vis/_testing.py for why these are plain functions; the
# ``clean_layout_cache`` fixture comes from braincell/vis/conftest.py.
# ---------------------------------------------------------------------------


@needs_benchmark
def test_plot2d_small(benchmark, clean_layout_cache) -> None:
    tree = make_deep_chain_tree(50)

    def _render():
        fig, ax = plt.subplots()
        plot2d(tree, layout="stem", shape="line", ax=ax)
        plt.close(fig)

    benchmark(_render)


@needs_benchmark
def test_plot2d_medium_values(benchmark, clean_layout_cache) -> None:
    tree = make_deep_chain_tree(500)
    values = np.linspace(0.0, 1.0, len(tree.branches))

    def _render():
        fig, ax = plt.subplots()
        plot2d(tree, layout="stem", shape="line", values=values, ax=ax, show_colorbar=False)
        plt.close(fig)

    benchmark(_render)


class Plot2dPixelBaselineTest(BaselineMixin, unittest.TestCase):
    """Pixel baselines for ``plot2d``; regenerate with ``BRAINCELL_UPDATE_BASELINES=1``."""

    def _draw(self, tree=None, **kwargs):
        fig, ax = self.new_axes()
        plot2d(tree if tree is not None else make_baseline_tree(), backend="matplotlib", ax=ax, **kwargs)
        return fig

    def test_projected_layout(self) -> None:
        self.assert_baseline(self._draw(layout="projected", shape="line"), "plot2d_projected")

    def test_stem_layout(self) -> None:
        self.assert_baseline(self._draw(layout="stem", shape="frustum"), "plot2d_stem")

    def test_balloon_layout(self) -> None:
        self.assert_baseline(self._draw(layout="balloon", shape="frustum"), "plot2d_balloon")

    def test_radial_layout(self) -> None:
        self.assert_baseline(self._draw(layout="radial_360", shape="frustum"), "plot2d_radial")

    def test_values_with_colorbar(self) -> None:
        values = np.linspace(-70.0, -40.0, 11)
        fig = self._draw(layout="stem", shape="frustum", values=values, value_label="V (mV)")
        self.assert_baseline(fig, "plot2d_values_colorbar")

    def test_baseline_detects_one_percent_branch_length_change(self) -> None:
        # Guard on the comparison itself: a trunk 1 % longer must not pass for the stem baseline.
        fig = self._draw(make_baseline_tree(apical_scale=1.01), layout="stem", shape="frustum")
        self.assert_baseline_differs(fig, "plot2d_stem")

    def test_baseline_detects_a_different_colormap(self) -> None:
        values = np.linspace(-70.0, -40.0, 11)
        fig = self._draw(layout="stem", shape="frustum", values=values, value_label="V (mV)", cmap="plasma")
        self.assert_baseline_differs(fig, "plot2d_values_colorbar")


if __name__ == "__main__":
    unittest.main()
