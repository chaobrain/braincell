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
from unittest import mock

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from braincell import vis as morpho_vis
from braincell.filter import AllRegion, BranchPoints, Terminals
from braincell.vis import plot2d, plot3d
from braincell.vis.backend import BackendChooser
from braincell.vis.backend_matplotlib import MatplotlibBackend
from braincell.vis.backend_pyvista import PyVistaBackend
from braincell.vis.compare2d import compare_layouts_2d
from braincell.vis._test_helper import FakeBackend
from braincell.vis._testing import (
    make_length_only_tree,
    make_point_tree,
    make_projected_point_tree,
    make_root_split_tree,
)


class VisPlotTest(unittest.TestCase):
    def setUp(self) -> None:
        morpho_vis.reset_defaults()
        self.addCleanup(morpho_vis.reset_defaults)

    def test_plot2d_defaults_to_fan_frustum(self) -> None:
        tree = make_point_tree()
        backend = FakeBackend()

        request = plot2d(tree, chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.dimensionality, "2d")
        self.assertEqual(request.layout, "fan")
        self.assertEqual(request.shape, "frustum")
        self.assertEqual(request.scene.layout, "fan")
        self.assertEqual(request.scene.shape, "frustum")
        self.assertEqual(request.scene.projection_plane, None)

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

    def test_plot3d_requires_points_and_suggests_2d_fallbacks(self) -> None:
        tree = make_length_only_tree()

        with self.assertRaisesRegex(ValueError, r"vis2d\(layout='stem', shape='line'\).+vis2d\(layout='stem', shape='frustum'\)"):
            plot3d(tree, chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot2d_rejects_unknown_shape(self) -> None:
        tree = make_point_tree()

        with self.assertRaisesRegex(ValueError, "Unsupported 2D shape"):
            plot2d(tree, layout="stem", shape="layout", chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot2d_projected_layout_rejects_frustum_shape(self) -> None:
        tree = make_point_tree()

        with self.assertRaisesRegex(ValueError, "layout='projected' only supports shape='line'"):
            plot2d(tree, layout="projected", shape="frustum", chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot3d_rejects_unknown_mode(self) -> None:
        tree = make_point_tree()

        with self.assertRaisesRegex(ValueError, "Unsupported 3D mode"):
            plot3d(tree, mode="projected")

    def test_plot3d_accepts_skeleton_mode(self) -> None:
        tree = make_point_tree()
        backend = FakeBackend()

        request = plot3d(tree, mode="skeleton", chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.mode, "skeleton")
        self.assertEqual(request.scene.mode, "skeleton")

    def test_plot2d_rejects_pyvista_backend(self) -> None:
        tree = make_point_tree()

        # Force PyVista to report as available so the dispatch reaches the
        # scene-kind validation step even when pyvista isn't installed.
        with mock.patch.object(PyVistaBackend, "available", return_value=True):
            with self.assertRaisesRegex(ValueError, "only supports 3D scenes"):
                plot2d(tree, backend="pyvista")

    def test_plot3d_rejects_matplotlib_backend(self) -> None:
        tree = make_point_tree()

        with self.assertRaisesRegex(ValueError, "only supports 2D scenes"):
            plot3d(tree, backend="matplotlib")

    def test_matplotlib_backend_renders_projected_scene(self) -> None:
        tree = make_point_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(tree, layout="projected", shape="line", backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)

    def test_global_vis_defaults_change_layout_shape_and_style(self) -> None:
        morpho_vis.configure_defaults(
            layout_2d_default="stem",
            shape_2d_default="line",
            branch_type_colors={"soma": "#000000"},
            branch_type_colors_2d={"soma": "#123456"},
            alpha_2d=0.25,
            alpha_3d_tube=0.4,
        )
        backend = FakeBackend()

        request_2d = plot2d(make_point_tree(), chooser=BackendChooser(backends=(backend,)))
        request_3d = plot3d(make_point_tree(), chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request_2d.layout, "stem")
        self.assertEqual(request_2d.shape, "line")
        self.assertTrue(all(polyline.color_rgb == (18, 52, 86) for polyline in request_2d.scene.polylines))
        self.assertTrue(all(abs(polyline.alpha - 0.25) < 1e-9 for polyline in request_2d.scene.polylines))
        self.assertEqual(request_3d.scene.batches[0].color_rgb, (0, 0, 0))
        self.assertAlmostEqual(request_3d.scene.batches[0].opacity, 0.4)

    def test_theme_context_manager_restores_defaults_on_exit(self) -> None:
        backend = FakeBackend()

        with morpho_vis.theme(branch_type_colors_2d={"soma": "#ff0000"}, alpha_2d=0.1):
            inside = plot2d(
                make_point_tree(),
                shape="line",
                chooser=BackendChooser(backends=(backend,)),
            )
            self.assertEqual(inside.scene.polylines[0].color_rgb, (255, 0, 0))
            self.assertAlmostEqual(inside.scene.polylines[0].alpha, 0.1)

        after = plot2d(
            make_point_tree(),
            shape="line",
            chooser=BackendChooser(backends=(backend,)),
        )
        # Soma default is black (0, 0, 0); alpha_2d_line defaults to 1.0.
        self.assertEqual(after.scene.polylines[0].color_rgb, (0, 0, 0))
        self.assertAlmostEqual(after.scene.polylines[0].alpha, 0.8)

    def test_global_2d_style_also_applies_to_frustum(self) -> None:
        morpho_vis.configure_defaults(
            branch_type_colors_2d={"apical_dendrite": "#445566"},
            alpha_2d=0.6,
        )
        backend = FakeBackend()

        request = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="frustum",
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertTrue(all(polygon.color_rgb == (68, 85, 102) for polygon in request.scene.polygons[1:]))
        self.assertTrue(all(abs(polygon.alpha - 0.6) < 1e-9 for polygon in request.scene.polygons))

    def test_shape_specific_2d_alpha_overrides_shared_alpha(self) -> None:
        morpho_vis.configure_defaults(
            alpha_2d=0.6,
            alpha_2d_line=0.2,
            alpha_2d_poly=0.9,
        )
        backend = FakeBackend()

        line_request = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="line",
            chooser=BackendChooser(backends=(backend,)),
        )
        poly_request = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="frustum",
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertTrue(all(abs(polyline.alpha - 0.2) < 1e-9 for polyline in line_request.scene.polylines))
        self.assertTrue(all(abs(polygon.alpha - 0.9) < 1e-9 for polygon in poly_request.scene.polygons))

    def test_generic_branch_type_colors_remain_2d_fallback(self) -> None:
        morpho_vis.configure_defaults(branch_type_colors={"soma": "#abcdef"})
        backend = FakeBackend()

        request = plot2d(make_point_tree(), shape="line", chooser=BackendChooser(backends=(backend,)))
        self.assertEqual(request.scene.polylines[0].color_rgb, (171, 205, 239))

    def test_theme_context_manager_restores_on_exception(self) -> None:
        original = morpho_vis.get_defaults().branch_type_colors["soma"]

        with self.assertRaises(RuntimeError):
            with morpho_vis.theme(branch_type_colors={"soma": "#abcdef"}):
                raise RuntimeError("boom")

        self.assertEqual(morpho_vis.get_defaults().branch_type_colors["soma"], original)

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

    def test_compare_layouts_2d_renders_side_by_side_matplotlib_figure(self) -> None:
        tree = make_length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        fig, axes = compare_layouts_2d(tree, chooser=chooser)

        self.assertEqual(len(axes), 4)
        self.assertEqual([ax.get_title() for ax in axes], ["Fan", "Stem", "Balloon", "Radial 360"])
        self.assertTrue(all(isinstance(ax, matplotlib.axes.Axes) for ax in axes))
        self.assertGreaterEqual(sum(len(ax.lines) for ax in axes), 3)
        plt.close(fig)


class VisOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        morpho_vis.reset_defaults()
        self.addCleanup(morpho_vis.reset_defaults)

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

    def test_plot3d_with_region_overlay_emits_highlight_strokes(self) -> None:
        tree = make_projected_point_tree()
        region = AllRegion().evaluate(tree)
        backend = FakeBackend()

        rendered = plot3d(tree, region=region, chooser=BackendChooser(backends=(backend,)))

        self.assertIs(rendered.overlay.region, region)
        self.assertEqual(len(rendered.scene.highlight_strokes), len(tree.branches))
        for stroke in rendered.scene.highlight_strokes:
            self.assertEqual(stroke.points_um.shape[1], 3)

    def test_plot3d_with_locset_overlay_emits_markers(self) -> None:
        tree = make_projected_point_tree()
        locset = (BranchPoints() | Terminals()).evaluate(tree)
        backend = FakeBackend()

        rendered = plot3d(tree, locset=locset, chooser=BackendChooser(backends=(backend,)))

        self.assertIs(rendered.overlay.locset, locset)
        self.assertEqual(len(rendered.scene.markers), len(locset.points))
        for marker in rendered.scene.markers:
            self.assertEqual(marker.position_um.shape, (3,))

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

    def test_plot3d_per_branch_values_emit_value_batches(self) -> None:
        tree = make_projected_point_tree()
        backend = FakeBackend()

        rendered = plot3d(
            tree,
            values=np.array([0.25, 0.75]),
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertGreaterEqual(len(rendered.scene.value_batches), 1)
        self.assertIsNotNone(rendered.scene.value_spec)

    def test_plot2d_values_length_mismatch_raises(self) -> None:
        tree = make_length_only_tree()

        with self.assertRaisesRegex(ValueError, "ValueSpec.values has length"):
            plot2d(
                tree,
                layout="stem",
                shape="line",
                values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                chooser=BackendChooser(backends=(FakeBackend(),)),
            )

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


if __name__ == "__main__":
    unittest.main()
