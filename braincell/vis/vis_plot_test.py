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

import brainunit as u
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from braincell import Branch, Morpho
from braincell.morpho import vis as morpho_vis
from braincell.vis import BackendChooser, MatplotlibBackend, compare_layouts_2d, plot2d, plot3d
from braincell.vis._test_helper import FakeBackend


def _point_tree() -> Morpho:
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    return Morpho.from_root(soma, name="soma")


def _length_only_tree() -> Morpho:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[8.0, 12.0] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morpho.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dendrite", parent_x=1.0)
    return tree


class VisPlotTest(unittest.TestCase):
    def setUp(self) -> None:
        morpho_vis.reset_defaults()
        self.addCleanup(morpho_vis.reset_defaults)

    def test_plot2d_defaults_to_frustum_mode(self) -> None:
        tree = _point_tree()
        backend = FakeBackend()

        request = plot2d(tree, chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.dimensionality, "2d")
        self.assertEqual(request.mode, "frustum")
        self.assertEqual(request.scene.mode, "frustum")
        self.assertEqual(request.scene.projection_plane, None)

    def test_plot2d_tree_mode_accepts_length_only_morphology(self) -> None:
        tree = _length_only_tree()
        backend = FakeBackend()

        request = plot2d(tree, mode="tree", chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.mode, "tree")
        self.assertEqual(request.scene.mode, "tree")
        self.assertEqual(len(request.scene.polylines), 3)
        self.assertEqual(len(request.scene.polygons), 0)

    def test_plot2d_frustum_mode_accepts_length_only_morphology(self) -> None:
        tree = _length_only_tree()
        backend = FakeBackend()

        request = plot2d(tree, mode="frustum", chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.mode, "frustum")
        self.assertEqual(request.scene.mode, "frustum")
        self.assertEqual(len(request.scene.polygons), 3)
        self.assertEqual(len(request.scene.polylines), 0)

    def test_plot2d_projected_mode_requires_points(self) -> None:
        tree = _length_only_tree()

        with self.assertRaisesRegex(ValueError, "mode='tree'.*mode='frustum'"):
            plot2d(tree, mode="projected", chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot3d_requires_points_and_suggests_2d_fallbacks(self) -> None:
        tree = _length_only_tree()

        with self.assertRaisesRegex(ValueError, r"vis2d\(mode='tree'\).+vis2d\(mode='frustum'\)"):
            plot3d(tree, chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot2d_rejects_unknown_mode(self) -> None:
        tree = _point_tree()

        with self.assertRaisesRegex(ValueError, "Unsupported 2D mode"):
            plot2d(tree, mode="layout", chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot3d_rejects_unknown_mode(self) -> None:
        tree = _point_tree()

        with self.assertRaisesRegex(ValueError, "Unsupported 3D mode"):
            plot3d(tree, mode="projected")

    def test_plot2d_rejects_pyvista_backend(self) -> None:
        tree = _point_tree()

        with self.assertRaisesRegex(ValueError, "only supports 3D scenes"):
            plot2d(tree, backend="pyvista")

    def test_plot3d_rejects_matplotlib_backend(self) -> None:
        tree = _point_tree()

        with self.assertRaisesRegex(ValueError, "only supports 2D scenes"):
            plot3d(tree, backend="matplotlib")

    def test_matplotlib_backend_renders_projected_scene(self) -> None:
        tree = _point_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(tree, mode="projected", backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)

    def test_global_vis_defaults_change_mode_and_style(self) -> None:
        morpho_vis.configure(
            mode_2d_default="tree",
            branch_type_colors={"soma": "#123456"},
            alpha_2d_line=0.25,
            alpha_3d_tube=0.4,
        )
        backend = FakeBackend()

        request_2d = plot2d(_point_tree(), chooser=BackendChooser(backends=(backend,)))
        request_3d = plot3d(_point_tree(), chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request_2d.mode, "tree")
        self.assertTrue(all(polyline.color_rgb == (18, 52, 86) for polyline in request_2d.scene.polylines))
        self.assertTrue(all(abs(polyline.alpha - 0.25) < 1e-9 for polyline in request_2d.scene.polylines))
        self.assertEqual(request_3d.scene.batches[0].color_rgb, (18, 52, 86))
        self.assertAlmostEqual(request_3d.scene.batches[0].opacity, 0.4)

    def test_matplotlib_backend_can_render_into_existing_axes(self) -> None:
        tree = _length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))
        fig, ax = plt.subplots(figsize=(8, 4))

        rendered_ax = plot2d(tree, mode="tree", backend="matplotlib", chooser=chooser, ax=ax)

        self.assertIs(rendered_ax, ax)
        self.assertEqual(rendered_ax.figure, fig)
        self.assertGreaterEqual(len(rendered_ax.lines), 2)
        plt.close(fig)

    def test_matplotlib_backend_renders_frustum_scene(self) -> None:
        tree = _length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(tree, mode="frustum", backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)
        self.assertGreaterEqual(len(axes.patches), 1)
        self.assertGreater(float(np.diff(axes.get_xlim())[0]), 20.0)
        self.assertGreater(float(np.diff(axes.get_ylim())[0]), 10.0)

    def test_compare_layouts_2d_renders_side_by_side_matplotlib_figure(self) -> None:
        tree = _length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        fig, axes = compare_layouts_2d(tree, chooser=chooser)

        self.assertEqual(len(axes), 3)
        self.assertEqual([ax.get_title() for ax in axes], ["Stem", "Balloon", "Radial 360"])
        self.assertTrue(all(isinstance(ax, matplotlib.axes.Axes) for ax in axes))
        self.assertGreaterEqual(sum(len(ax.lines) for ax in axes), 3)
        plt.close(fig)
