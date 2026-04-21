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
import warnings

import numpy as np

from braincell.vis._testing import make_two_dendrite_tree
from braincell.vis.layout._common import _build_layout_specs
from braincell.vis.layout._legacy import _build_layout_branches_legacy


class BuildLegacyLayoutTest(unittest.TestCase):
    def test_legacy_builds_layouts_for_each_branch(self) -> None:
        tree = make_two_dendrite_tree()
        specs = _build_layout_specs(tree)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            layouts = _build_layout_branches_legacy(
                tree,
                layout_specs=specs,
                min_branch_angle_deg=25.0,
            )
        self.assertEqual(len(layouts), len(tree.branches))
        for layout in layouts:
            self.assertTrue(np.all(np.isfinite(layout.segment_points_um)))


if __name__ == "__main__":
    unittest.main()
