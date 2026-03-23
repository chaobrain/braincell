from __future__ import annotations

import unittest
from pathlib import Path

from . import _support  # noqa: F401

from braincell import Morpho
from braincell.vis import build_render_geometry_3d


FIXTURE_DIR = Path(__file__).resolve().parent / "morpho_files"
VALID_SWC_FIXTURES = ("grc.swc", "io.swc")
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


class RealFileVisTest(unittest.TestCase):
    def test_valid_real_swc_fixtures_build_render_geometry(self) -> None:
        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                tree = Morpho.from_swc(FIXTURE_DIR / fixture_name)
                geometry = build_render_geometry_3d(tree)

                self.assertEqual(len(geometry.branches), len(tree.branches))
                self.assertGreaterEqual(len(geometry.batches), 1)
                for branch in geometry.branches:
                    self.assertTrue(branch.branch_name)
                    self.assertIn(branch.branch_type, ALLOWED_TYPES)
                    self.assertEqual(branch.points_um.ndim, 2)
                    self.assertEqual(branch.points_um.shape[1], 3)
                    self.assertGreaterEqual(len(branch.points_um), 2)
                    self.assertEqual(len(branch.radii_um), len(branch.points_um))
