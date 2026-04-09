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
import textwrap
import unittest
from pathlib import Path

from braincell import Morpho
from braincell.io.asc import AscReader

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "develop_doc" / "morpho_files"
ALLOWED_TYPES = {
    "soma",
    "axon",
    "dendrite",
    "basal_dendrite",
    "apical_dendrite",
    "custom",
}


class AscReaderTest(unittest.TestCase):
    def _write_asc(self, body: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "sample.asc"
        path.write_text(textwrap.dedent(body).strip() + "\n")
        return path

    def test_reader_imports_simple_neurolucida_tree_and_metadata(self) -> None:
        path = self._write_asc(
            '''
            ; example asc
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )

            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
              (
                (0 10 0 1)
                (0 15 0 1)
                (Spine "s1")
                Normal
              |
                (5 5 0 1)
                (10 5 0 1)
                (Marker "m1")
                (FilledCircle 1 2 3)
                Normal
              )
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertIsInstance(tree, Morpho)
        self.assertFalse(report.has_errors)
        self.assertEqual(tree.root.type, "soma")
        self.assertEqual(len(tree.branches), 4)
        self.assertEqual(tree.root.n_children, 1)
        self.assertEqual(tree.branch(index=1).type, "dendrite")
        self.assertEqual(tree.branch(index=1).parent_x, 0.5)
        self.assertEqual(tree.branch(index=1).n_children, 2)
        self.assertEqual(tree.branch(index=2).parent_x, 1.0)
        self.assertEqual(tree.branch(index=3).parent_x, 1.0)
        self.assertEqual(len(report.metadata.spines), 1)
        self.assertEqual(len(report.metadata.markers), 1)
        self.assertEqual(len(report.metadata.filled_circles), 1)
        self.assertGreaterEqual(len(report.metadata.comments), 1)
        self.assertIn("Cell Body", report.metadata.source_labels)

    def test_morpho_from_asc_supports_return_report(self) -> None:
        path = self._write_asc(
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ( (Axon)
              (0 0 0 1)
              (10 0 0 1)
            )
            '''
        )

        tree, report = Morpho.from_asc(path, return_report=True)

        self.assertIsInstance(tree, Morpho)
        self.assertFalse(report.has_errors)
        self.assertEqual(tree.root.type, "soma")
        self.assertEqual(tree.branch(index=1).type, "axon")


class AscRealFileSmokeTest(unittest.TestCase):
    def test_valid_real_asc_fixtures_pass_smoke_checks(self) -> None:
        reader = AscReader()
        for fixture_name in ("goc.asc", "pc.asc"):
            with self.subTest(fixture=fixture_name):
                tree, report = reader.read(FIXTURE_DIR / fixture_name, return_report=True)

                self.assertIsInstance(tree, Morpho)
                self.assertFalse(report.has_errors)
                self.assertGreater(len(tree.branches), 0)
                self.assertEqual(tree.root.type, "soma")
                self.assertTrue(tree.topo())
                self.assertTrue(all(branch.type in ALLOWED_TYPES for branch in tree.branches))

    def test_valid_real_asc_fixtures_support_morpho_from_asc(self) -> None:
        for fixture_name in ("goc.asc", "pc.asc"):
            with self.subTest(fixture=fixture_name):
                path = FIXTURE_DIR / fixture_name
                tree = Morpho.from_asc(path)
                tree_with_report, report = Morpho.from_asc(path, return_report=True)

                self.assertIsInstance(tree, Morpho)
                self.assertIsInstance(tree_with_report, Morpho)
                self.assertFalse(report.has_errors)
                self.assertGreater(len(tree.branches), 0)
                self.assertEqual(tree.root.type, "soma")
                self.assertTrue(tree.topo())
