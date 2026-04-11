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


import math
import unittest

import numpy as np

from braincell.vis._testing import make_two_dendrite_tree
from braincell.vis.layout._balloon import (
    _allocate_balloon_group_angles,
    _build_layout_branches_balloon,
)
from braincell.vis.layout._common import (
    _build_layout_specs,
    _leaf_counts_by_branch,
)


class BuildBalloonTest(unittest.TestCase):
    def test_balloon_builds_layouts_for_each_branch(self) -> None:
        tree = make_two_dendrite_tree()
        specs = _build_layout_specs(tree)
        layouts = _build_layout_branches_balloon(
            tree,
            layout_specs=specs,
            min_branch_angle_deg=25.0,
            root_layout="type_split",
        )
        self.assertEqual(len(layouts), len(tree.branches))
        for layout in layouts:
            self.assertTrue(np.all(np.isfinite(layout.segment_points_um)))

    def test_balloon_places_two_dendrites_at_distinct_angles(self) -> None:
        tree = make_two_dendrite_tree()
        specs = _build_layout_specs(tree)
        layouts = {
            layout.branch_name: layout
            for layout in _build_layout_branches_balloon(
                tree,
                layout_specs=specs,
                min_branch_angle_deg=25.0,
                root_layout="type_split",
            )
        }

        dend_a = layouts["dend_a"].end_direction_um
        dend_b = layouts["dend_b"].end_direction_um
        self.assertFalse(np.allclose(dend_a, dend_b))


class AllocateBalloonGroupAnglesTest(unittest.TestCase):
    def test_empty_returns_empty(self) -> None:
        angles = _allocate_balloon_group_angles(
            (),
            leaf_counts={},
            interval=(-1.0, 1.0),
            min_branch_angle_rad=0.0,
        )
        self.assertEqual(angles, {})

    def test_two_equal_weight_children_get_symmetric_angles(self) -> None:
        tree = make_two_dendrite_tree()
        leaf_counts = _leaf_counts_by_branch(tree.root)
        angles = _allocate_balloon_group_angles(
            tree.root.children,
            leaf_counts=leaf_counts,
            interval=(-math.pi / 2.0, math.pi / 2.0),
            min_branch_angle_rad=0.0,
        )
        values = sorted(angles.values())
        self.assertEqual(len(values), 2)
        # Symmetric around 0.0.
        self.assertAlmostEqual(values[0] + values[1], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
