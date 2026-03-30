from __future__ import annotations

import unittest

import matplotlib.axes

from braincell._test_support import FakeBackend, u

from braincell import Branch, Morpho
from braincell.vis import BackendChooser, MatplotlibBackend, plot2d, plot3d


class VisPlotTest(unittest.TestCase):
    def test_plot2d_defaults_to_projected_mode(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")
        backend = FakeBackend()

        request = plot2d(tree, chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.dimensionality, "2d")
        self.assertEqual(request.mode, "projected")
        self.assertEqual(request.scene.mode, "projected")
        self.assertEqual(request.scene.projection_plane, "xy")

    def test_plot3d_rejects_unknown_mode(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaisesRegex(ValueError, "Unsupported 3D mode"):
            plot3d(tree, mode="projected")

    def test_plot2d_rejects_pyvista_backend(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaisesRegex(ValueError, "only supports 3D scenes"):
            plot2d(tree, backend="pyvista")

    def test_plot3d_rejects_matplotlib_backend(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaisesRegex(ValueError, "only supports 2D scenes"):
            plot3d(tree, backend="matplotlib")

    def test_matplotlib_backend_renders_projected_scene(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")
        chooser = BackendChooser(backends=(MatplotlibBackend(),))

        axes = plot2d(tree, backend="matplotlib", chooser=chooser)

        self.assertIsInstance(axes, matplotlib.axes.Axes)

    def test_plot2d_layout_mode_is_reserved_but_explicit(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaisesRegex(NotImplementedError, "layout mode"):
            plot2d(tree, mode="layout", chooser=BackendChooser(backends=(FakeBackend(),)))
