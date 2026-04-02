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

import braincell._test_support  # noqa: F401

from braincell import Morpho
from braincell.vis import build_render_scene_3d


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "io" / "morpho_files"
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
    def test_valid_real_swc_fixtures_build_render_scene_3d(self) -> None:
        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                tree = Morpho.from_swc(FIXTURE_DIR / fixture_name)
                scene = build_render_scene_3d(tree)

                self.assertEqual(len(scene.branches), len(tree.branches))
                self.assertGreaterEqual(len(scene.batches), 1)
                for branch in scene.branches:
                    self.assertTrue(branch.branch_name)
                    self.assertIn(branch.branch_type, ALLOWED_TYPES)
                    self.assertEqual(branch.points_um.ndim, 2)
                    self.assertEqual(branch.points_um.shape[1], 3)
                    self.assertGreaterEqual(len(branch.points_um), 2)
                    self.assertEqual(len(branch.radii_um), len(branch.points_um))
