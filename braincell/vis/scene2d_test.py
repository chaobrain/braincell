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

"""Tests for :mod:`braincell.vis.scene2d`."""

import unittest

import matplotlib.pyplot as plt
import pytest

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.filter import AllRegion, Terminals
from braincell.vis import plot2d
from braincell.vis._testing import (
    PYTEST_BENCHMARK_AVAILABLE,
    FakeBackend,
    VisDefaultsResetMixin,
    make_deep_chain_tree,
    make_length_only_tree,
)
from braincell.vis.backend import BackendChooser
from braincell.vis.layout import build_layout_branches_2d, get_default_layout_cache
from braincell.vis.scene import OverlaySpec, ValueSpec
from braincell.vis.scene2d import build_render_scene_2d


def _point_geometry_tree_with_same_lengths() -> Morphology:
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [0.0, 6.4, 4.8], [12.0, 6.4, 4.8]] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


class BuildRenderScene2dTest(unittest.TestCase):
    def test_frustum_scene_builds_polygon_per_segment(self) -> None:
        tree = make_length_only_tree()

        scene = build_render_scene_2d(tree, layout="stem", shape="frustum")

        self.assertEqual(scene.layout, "stem")
        self.assertEqual(scene.shape, "frustum")
        self.assertEqual(len(scene.polygons), 3)
        root_polygon = scene.polygons[0]
        self.assertEqual(root_polygon.points_um.shape, (4, 2))

        child_polygons = [polygon for polygon in scene.polygons if polygon.branch_name == "dend"]
        first_child = child_polygons[0]
        start_midpoint = 0.5 * (first_child.points_um[0] + first_child.points_um[3])
        end_midpoint = 0.5 * (first_child.points_um[1] + first_child.points_um[2])
        proximal_width = np.linalg.norm(first_child.points_um[0] - first_child.points_um[3])
        distal_width = np.linalg.norm(first_child.points_um[1] - first_child.points_um[2])

        self.assertAlmostEqual(float(np.linalg.norm(end_midpoint - start_midpoint)), 8.0)
        self.assertAlmostEqual(float(proximal_width), 4.0)
        self.assertAlmostEqual(float(distal_width), 3.0)

    def test_tree_and_frustum_ignore_real_points_geometry(self) -> None:
        length_tree = make_length_only_tree()
        point_geometry_tree = _point_geometry_tree_with_same_lengths()

        length_layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(length_tree, mode="tree")}
        point_layouts = {
            layout.branch_name: layout for layout in build_layout_branches_2d(point_geometry_tree, mode="tree")
        }

        self.assertTrue(np.allclose(length_layouts["dend"].segment_points_um, point_layouts["dend"].segment_points_um))

        length_scene = build_render_scene_2d(length_tree, layout="stem", shape="frustum")
        point_scene = build_render_scene_2d(
            point_geometry_tree,
            layout="stem",
            shape="frustum",
        )

        self.assertEqual(len(length_scene.polygons), len(point_scene.polygons))
        for length_polygon, point_polygon in zip(length_scene.polygons, point_scene.polygons):
            self.assertTrue(np.allclose(length_polygon.points_um, point_polygon.points_um))


class Scene2dOverlayTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_plot2d_with_region_overlay_emits_highlight_strokes(self) -> None:
        tree = make_length_only_tree()
        region = AllRegion().evaluate(tree)
        backend = FakeBackend()

        rendered = plot2d(
            tree,
            layout="stem",
            shape="line",
            region=region,
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertIs(rendered.overlay.region, region)
        self.assertEqual(len(rendered.scene.highlight_strokes), len(tree.branches))
        for stroke, branch in zip(rendered.scene.highlight_strokes, tree.branches):
            self.assertEqual(stroke.branch_index, branch.index)
            self.assertGreaterEqual(stroke.points_um.shape[0], 2)
            self.assertGreater(stroke.linewidth, 0.0)

    def test_plot2d_with_locset_overlay_emits_markers(self) -> None:
        tree = make_length_only_tree()
        locset = Terminals().evaluate(tree)
        backend = FakeBackend()

        rendered = plot2d(
            tree,
            layout="stem",
            shape="line",
            locset=locset,
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertIs(rendered.overlay.locset, locset)
        self.assertEqual(len(rendered.scene.markers), len(locset.points))
        for marker in rendered.scene.markers:
            self.assertEqual(marker.position_um.shape, (2,))

    def test_plot2d_per_branch_values_emit_value_primitives(self) -> None:
        tree = make_length_only_tree()
        backend = FakeBackend()

        rendered = plot2d(
            tree,
            layout="stem",
            shape="line",
            values=np.array([0.1, 0.9]),
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertIsNotNone(rendered.scene.value_spec)
        self.assertEqual(len(rendered.scene.polyline_values), len(tree.branches))
        self.assertEqual(len(rendered.scene.polylines), 0)

    def test_plot2d_frustum_values_emit_polygon_value_batches(self) -> None:
        tree = make_length_only_tree()
        backend = FakeBackend()

        rendered = plot2d(
            tree,
            layout="stem",
            shape="frustum",
            values=np.array([0.1, 0.9]),
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(len(rendered.scene.polygon_value_batches), len(tree.branches))
        self.assertEqual(len(rendered.scene.polygons), 0)


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
def test_scene_small_no_values(benchmark, clean_layout_cache) -> None:
    tree = make_deep_chain_tree(50)
    benchmark(lambda: build_render_scene_2d(tree, layout="stem", shape="line"))


@_needs_benchmark
def test_scene_medium_with_values(benchmark, clean_layout_cache) -> None:
    tree = make_deep_chain_tree(500)
    values = np.linspace(0.0, 1.0, len(tree.branches))
    overlay = OverlaySpec(values=ValueSpec(values=values))
    benchmark(lambda: build_render_scene_2d(tree, layout="stem", shape="line", overlay=overlay))


if __name__ == "__main__":
    unittest.main()
