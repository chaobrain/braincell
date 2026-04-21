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

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.vis._testing import make_fan_root_partition_tree, make_two_dendrite_tree
from braincell.vis.layout import point_on_layout_branch
from braincell.vis.layout._fan import _build_layout_branches_fan
from braincell.vis.layout._common import _build_layout_specs


class BuildFanLayoutTest(unittest.TestCase):
    def test_fan_builds_layouts_for_each_branch(self) -> None:
        tree = make_two_dendrite_tree()
        layouts = _build_layout_branches_fan(
            tree,
            layout_specs=_build_layout_specs(tree),
            min_branch_angle_deg=25.0,
        )
        self.assertEqual(len(layouts), len(tree.branches))

    def test_fan_root_partition_places_children_into_expected_half_planes(self) -> None:
        tree = make_fan_root_partition_tree()
        layouts = {
            layout.branch_name: layout
            for layout in _build_layout_branches_fan(
                tree,
                layout_specs=_build_layout_specs(tree),
                min_branch_angle_deg=25.0,
            )
            if layout.branch_name != "soma"
        }
        root_layout = next(
            layout for layout in _build_layout_branches_fan(
                tree,
                layout_specs=_build_layout_specs(tree),
                min_branch_angle_deg=25.0,
            )
            if layout.branch_name == "soma"
        )
        self.assertLess(point_on_layout_branch(root_layout, 0.0)[0], 0.0)
        self.assertAlmostEqual(float(point_on_layout_branch(root_layout, 0.5)[0]), 0.0, places=6)
        self.assertGreater(point_on_layout_branch(root_layout, 1.0)[0], 0.0)
        self.assertLess(layouts["left_dend"].end_direction_um[0], 0.0)
        self.assertGreater(layouts["mid_dend"].end_direction_um[1], 0.0)
        self.assertLess(layouts["mid_axon"].end_direction_um[1], 0.0)
        self.assertGreater(layouts["right_near"].end_direction_um[0], 0.0)

    def test_fan_branches_are_straight(self) -> None:
        tree = make_fan_root_partition_tree()
        layouts = [
            layout
            for layout in _build_layout_branches_fan(
                tree,
                layout_specs=_build_layout_specs(tree),
                min_branch_angle_deg=25.0,
            )
            if layout.branch_name != "soma"
        ]
        for layout in layouts:
            self.assertTrue(np.allclose(layout.segment_directions_um[0], layout.segment_directions_um[-1]))

    def test_fan_odd_middle_group_puts_extra_branch_in_upper_sector(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        for index in range(3):
            tree.attach(
                parent="soma",
                child_branch=Branch.from_lengths(
                    lengths=[12.0] * u.um,
                    radii=[1.6, 1.0] * u.um,
                    type="apical_dendrite",
                ),
                child_name=f"mid_{index}",
                parent_x=0.5,
            )
        layouts = {
            layout.branch_name: layout
            for layout in _build_layout_branches_fan(
                tree,
                layout_specs=_build_layout_specs(tree),
                min_branch_angle_deg=25.0,
            )
            if layout.branch_name.startswith("mid_")
        }
        n_upper = sum(layout.end_direction_um[1] > 0.0 for layout in layouts.values())
        n_lower = sum(layout.end_direction_um[1] < 0.0 for layout in layouts.values())
        self.assertEqual(n_upper, 2)
        self.assertEqual(n_lower, 1)

    def test_fan_separates_two_root_siblings_by_min_angle(self) -> None:
        tree = make_two_dendrite_tree()
        layouts = {
            layout.branch_name: layout
            for layout in _build_layout_branches_fan(
                tree,
                layout_specs=_build_layout_specs(tree),
                min_branch_angle_deg=35.0,
            )
            if layout.branch_name.startswith("dend")
        }
        cos_theta = float(
            np.clip(
                np.dot(layouts["dend_a"].end_direction_um, layouts["dend_b"].end_direction_um)
                / (
                    np.linalg.norm(layouts["dend_a"].end_direction_um)
                    * np.linalg.norm(layouts["dend_b"].end_direction_um)
                ),
                -1.0,
                1.0,
            )
        )
        self.assertGreaterEqual(math.degrees(math.acos(cos_theta)), 34.0)


if __name__ == "__main__":
    unittest.main()
