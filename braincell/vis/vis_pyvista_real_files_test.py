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
from pathlib import Path

try:
    import pyvista as pv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pv = None

from braincell import Morpho
from braincell.vis import BackendChooser, PyVistaBackend


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "develop_doc" / "morpho_files"
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
