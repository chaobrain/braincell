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


import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt

from braincell.vis import plot2d
from braincell.vis._testing import make_length_only_tree
from braincell.vis.export import save_figure


class SaveFigureMatplotlibTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = make_length_only_tree()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self) -> None:
        plt.close("all")

    def test_save_axes_produces_file(self) -> None:
        fig, ax = plt.subplots()
        plot2d(self.tree, layout="stem", shape="line", ax=ax)
        out = Path(self.tmp.name) / "figure.png"
        result = save_figure(ax, out, dpi=100)
        self.assertTrue(result.exists())
        self.assertEqual(result, out)
        self.assertGreater(out.stat().st_size, 0)

    def test_save_figure_instance_produces_file(self) -> None:
        fig, ax = plt.subplots()
        plot2d(self.tree, layout="stem", shape="line", ax=ax)
        out = Path(self.tmp.name) / "figure.pdf"
        save_figure(fig, out)
        self.assertTrue(out.exists())

    def test_save_with_transparent(self) -> None:
        fig, ax = plt.subplots()
        plot2d(self.tree, layout="stem", shape="line", ax=ax)
        out = Path(self.tmp.name) / "transp.png"
        save_figure(ax, out, transparent=True)
        self.assertTrue(out.exists())

    def test_unknown_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "does not know how to save"):
            save_figure(object(), Path(self.tmp.name) / "nope.png")


class SaveFigurePlotlyDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_fake_plotly_html(self) -> None:
        import importlib.util

        if importlib.util.find_spec("plotly") is None:
            self.skipTest("plotly not installed")
        import plotly.graph_objects as go

        fig = go.Figure()
        out = Path(self.tmp.name) / "fig.html"
        save_figure(fig, out)
        self.assertTrue(out.exists())


class SaveFigurePyVistaDispatchTest(unittest.TestCase):
    def test_fake_plotter_dispatches_to_screenshot(self) -> None:
        import importlib.util

        if importlib.util.find_spec("pyvista") is None:
            self.skipTest("pyvista not installed")

        class _FakePlotter:
            __module__ = "pyvista.plotting.plotter"

            def __init__(self) -> None:
                self.calls: list[tuple[str, dict]] = []

            def screenshot(self, path, **kwargs):
                self.calls.append((path, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            plotter = _FakePlotter()
            out = Path(tmp) / "tree.png"
            with mock.patch(
                "braincell.vis.export._is_pyvista_plotter",
                lambda obj: isinstance(obj, _FakePlotter),
            ):
                save_figure(plotter, out, transparent=True, dpi=200)
            self.assertEqual(len(plotter.calls), 1)
            self.assertTrue(plotter.calls[0][1]["transparent_background"])

    def test_pyvista_vector_without_save_graphic_raises(self) -> None:
        class _FakePlotter:
            __module__ = "pyvista.plotting.plotter"

        with tempfile.TemporaryDirectory() as tmp:
            plotter = _FakePlotter()
            out = Path(tmp) / "tree.pdf"
            with mock.patch(
                "braincell.vis.export._is_pyvista_plotter",
                lambda obj: isinstance(obj, _FakePlotter),
            ):
                with self.assertRaisesRegex(ValueError, "cannot save vector format"):
                    save_figure(plotter, out)


if __name__ == "__main__":
    unittest.main()
