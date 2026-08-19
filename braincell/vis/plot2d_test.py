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

import pytest

import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braincell import Branch, Morphology
from braincell.vis import plot2d
from braincell.vis._testing import (
    PYTEST_BENCHMARK_AVAILABLE,
    PYTEST_MPL_AVAILABLE,
    FakeBackend,
    VisDefaultsResetMixin,
    image_comparison,
    make_deep_chain_tree,
    make_four_type_tree,
    make_length_only_tree,
    make_node_tree,
    make_projected_node_tree,
)
from braincell.vis.backend import BackendChooser
from braincell.vis.layout import get_default_layout_cache


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


@unittest.skipUnless(PYTEST_MPL_AVAILABLE, "pytest-mpl is not installed")
class Plot2dBaselineImageTest(unittest.TestCase):
    """Baseline-image regression figures for :func:`braincell.vis.plot2d`.

    Each test builds a figure and returns it; ``image_comparison`` hands it
    to pytest-mpl when the plugin is installed. Skipped wholesale otherwise —
    without the plugin the marker is inert and the figure would never be
    compared, so there is nothing left to assert.
    """

    def tearDown(self) -> None:
        plt.close("all")

    @image_comparison("stem_frustum_baseline.png")
    def test_stem_frustum_baseline(self):
        tree = make_four_type_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="stem", shape="frustum", ax=ax)
        return fig

    @image_comparison("stem_line_baseline.png")
    def test_stem_line_baseline(self):
        tree = make_four_type_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="stem", shape="line", ax=ax)
        return fig

    @image_comparison("balloon_line_baseline.png")
    def test_balloon_line_baseline(self):
        tree = make_four_type_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="balloon", shape="line", ax=ax)
        return fig

    @image_comparison("radial_line_baseline.png")
    def test_radial_line_baseline(self):
        tree = make_four_type_tree()
        fig, ax = plt.subplots(figsize=(4, 4))
        plot2d(tree, layout="radial_360", shape="line", ax=ax)
        return fig

    @image_comparison("values_per_branch_baseline.png")
    def test_values_per_branch_baseline(self):
        tree = make_four_type_tree()
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

    @image_comparison("values_frustum_baseline.png")
    def test_values_frustum_baseline(self):
        tree = make_four_type_tree()
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

    @image_comparison("projected_scene_baseline.png")
    def test_projected_scene_baseline(self):
        from braincell import Branch, Morphology

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


# ---------------------------------------------------------------------------
# Performance baselines (pytest-benchmark)
#
# Plain pytest functions, not TestCase methods: pytest cannot inject the
# function-scoped ``benchmark`` fixture into a ``unittest.TestCase``. The
# skipif is per-function rather than a module-level ``pytestmark`` because
# the rest of this file must keep running without the plugin.
# ---------------------------------------------------------------------------

_needs_benchmark = pytest.mark.skipif(
    not PYTEST_BENCHMARK_AVAILABLE,
    reason="pytest-benchmark is not installed",
)


@pytest.fixture
def clean_layout_cache():
    """Clear the shared layout cache before a benchmark, close figures after."""
    get_default_layout_cache().clear()
    yield
    plt.close("all")


@_needs_benchmark
def test_plot2d_small(benchmark, clean_layout_cache) -> None:
    tree = make_deep_chain_tree(50)

    def _render():
        fig, ax = plt.subplots()
        plot2d(tree, layout="stem", shape="line", ax=ax)
        plt.close(fig)

    benchmark(_render)


@_needs_benchmark
def test_plot2d_medium_values(benchmark, clean_layout_cache) -> None:
    tree = make_deep_chain_tree(500)
    values = np.linspace(0.0, 1.0, len(tree.branches))

    def _render():
        fig, ax = plt.subplots()
        plot2d(tree, layout="stem", shape="line", values=values, ax=ax, show_colorbar=False)
        plt.close(fig)

    benchmark(_render)


if __name__ == "__main__":
    unittest.main()
