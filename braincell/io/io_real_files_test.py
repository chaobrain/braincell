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

from __future__ import annotations

import unittest
from pathlib import Path

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
                self.assertEqual(len(tree.edges), len(tree.branches) - 1)
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

    def test_problematic_real_swc_fixture_surfaces_check_warnings(self) -> None:
        path = FIXTURE_DIR / "bc.swc"

        report = SwcReader().check(path)

        self.assertFalse(report.has_errors)
        self.assertTrue(report.has_warnings)
        self.assertTrue(any(issue.code == "geometry.duplicate_xyzr_node" for issue in report.issues))
        self.assertTrue(any(issue.code == "semantics.unknown_type" for issue in report.issues))

        tree = SwcReader().read(path)

        self.assertIsInstance(tree, Morpho)
        self.assertGreater(len(tree.branches), 0)
        self.assertTrue(tree.topo())

    def test_problematic_real_swc_fixture_supports_morpho_import_with_warnings(self) -> None:
        path = FIXTURE_DIR / "bc.swc"

        tree, report = Morpho.from_swc(path, return_report=True)

        self.assertIsInstance(tree, Morpho)
        self.assertFalse(report.has_errors)
        self.assertTrue(report.has_warnings)
        self.assertGreater(len(tree.branches), 0)
        self.assertTrue(tree.topo())


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
