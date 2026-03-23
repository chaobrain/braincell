from __future__ import annotations

import unittest
from pathlib import Path

from . import _support  # noqa: F401

from braincell import AscReader, Morpho, SwcReader


FIXTURE_DIR = Path(__file__).resolve().parent / "morpho_files"
ALLOWED_TYPES = {
    "soma",
    "axon",
    "dend",
    "basal_dend",
    "basal_dendrite",
    "apical_dend",
    "apical_dendrite",
    "custom",
}


class SwcRealFileSmokeTest(unittest.TestCase):
    def test_valid_real_swc_fixtures_pass_smoke_checks(self) -> None:
        for fixture_name in ("grc.swc", "io.swc"):
            with self.subTest(fixture=fixture_name):
                path = FIXTURE_DIR / fixture_name
                report = SwcReader().check(path)
                self.assertFalse(report.has_errors)
                tree = SwcReader().read(path)
                self.assertGreater(len(tree.branches), 0)
                self.assertEqual(len(tree.connections), len(tree.branches) - 1)
                self.assertTrue(tree.topo())
                self.assertTrue(all(branch.type in ALLOWED_TYPES for branch in tree.branches))

    def test_valid_real_swc_fixtures_support_morpho_from_swc(self) -> None:
        for fixture_name in ("grc.swc", "io.swc"):
            with self.subTest(fixture=fixture_name):
                path = FIXTURE_DIR / fixture_name
                tree = Morpho.from_swc(path)
                tree_with_report, report = Morpho.from_swc(path, return_report=True)

                self.assertIsInstance(tree, Morpho)
                self.assertIsInstance(tree_with_report, Morpho)
                self.assertFalse(report.has_errors)
                self.assertGreater(len(tree.branches), 0)
                self.assertTrue(tree.root.name)
                self.assertTrue(tree.topo())
                self.assertTrue(all(branch.type in ALLOWED_TYPES for branch in tree.branches))

    def test_problematic_real_swc_fixture_surfaces_check_errors(self) -> None:
        path = FIXTURE_DIR / "bc.swc"

        report = SwcReader().check(path)

        self.assertTrue(report.has_errors)
        self.assertTrue(any(issue.code == "geometry.zero_length_segment" for issue in report.issues))
        with self.assertRaises(ValueError):
            SwcReader().read(path)

    def test_problematic_real_swc_fixture_fails_strict_morpho_import(self) -> None:
        path = FIXTURE_DIR / "bc.swc"

        with self.assertRaises(ValueError):
            Morpho.from_swc(path)


class AscRealFilePlaceholderTest(unittest.TestCase):
    def test_real_asc_fixtures_are_explicitly_not_implemented_yet(self) -> None:
        reader = AscReader()
        for fixture_name in ("goc.asc", "pc.asc"):
            with self.subTest(fixture=fixture_name):
                with self.assertRaises(NotImplementedError):
                    reader.read(FIXTURE_DIR / fixture_name)
