from __future__ import annotations

import unittest

import matplotlib.axes
import numpy as np

from braincell._test_support import FakeBackend, u

from braincell import Branch, Morpho
from braincell.vis import BackendChooser, MatplotlibBackend, plot2d, plot3d


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
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


class VisPlotTest(unittest.TestCase):
    def test_plot2d_defaults_to_projected_mode(self) -> None:
        tree = _point_tree()
        backend = FakeBackend()

        request = plot2d(tree, chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.dimensionality, "2d")
        self.assertEqual(request.mode, "projected")
        self.assertEqual(request.scene.mode, "projected")
        self.assertEqual(request.scene.projection_plane, "xy")

    def test_plot2d_tree_mode_accepts_length_only_morphology(self) -> None:
        tree = _length_only_tree()
        backend = FakeBackend()

        request = plot2d(tree, mode="tree", chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.mode, "tree")
        self.assertEqual(request.scene.mode, "tree")
        self.assertEqual(len(request.scene.polylines), 2)
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

        with self.assertRaisesRegex(ValueError, "point geometry"):
            plot2d(tree, mode="projected", chooser=BackendChooser(backends=(FakeBackend(),)))

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

        axes = plot2d(tree, backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)

    def test_matplotlib_backend_renders_frustum_scene(self) -> None:
        tree = _length_only_tree()
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(tree, mode="frustum", backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)
        self.assertGreaterEqual(len(axes.patches), 1)
        self.assertTrue(np.allclose(axes.get_xlim(), (-2.0, 42.0)))
        self.assertTrue(np.allclose(axes.get_ylim(), (-11.0, 11.0)))
