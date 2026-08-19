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

import matplotlib.pyplot as plt
import numpy as np

from braincell.vis import plot2d
from braincell.vis._testing import (
    PYTEST_BENCHMARK_AVAILABLE,
    FakeBackend,
    VisDefaultsResetMixin,
    make_deep_chain_tree,
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
