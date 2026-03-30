from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from braincell import AscReader, Morpho


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
        self.assertEqual(tree.branch(index=1).type, "dend")
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
