from __future__ import annotations

import unittest
from pathlib import Path

import braincell._test_support  # noqa: F401

try:
    import pyvista as pv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pv = None

from braincell import Morpho
from braincell.vis import BackendChooser, PyVistaBackend


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "io" / "morpho_files"
VALID_SWC_FIXTURES = ("grc.swc", "io.swc")


class RealFilePyVistaVisTest(unittest.TestCase):
    def test_valid_real_swc_fixtures_render_with_pyvista_backend(self) -> None:
        if pv is None:
            self.skipTest("pyvista is not installed.")
        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                tree = Morpho.from_swc(FIXTURE_DIR / fixture_name)
                backend = PyVistaBackend(plotter_kwargs={"off_screen": True}, show_axes=False)
                chooser = BackendChooser(backends=(backend,))
                plotter = tree.vis3d(backend="pyvista", chooser=chooser)

                self.assertIsInstance(plotter, pv.Plotter)
                self.assertIsNotNone(plotter.renderer)
                self.assertGreater(len(plotter.renderer.actors), 0)
                plotter.close()
