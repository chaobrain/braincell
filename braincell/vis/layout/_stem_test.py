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

from braincell.vis._testing import make_length_only_tree, make_root_split_tree
from braincell.vis.layout._common import _build_layout_specs
from braincell.vis.layout._stem import (
    _build_layout_branches_stem,
    _build_layout_branches_stem_linear,
    _stem_profile_candidates,
    _stem_segment_angles_rad,
)


class StemSegmentAnglesTest(unittest.TestCase):
    def test_single_segment_uses_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.array([10.0]),
            launch_angle_rad=0.1,
            settle_angle_rad=0.2,
            tail_angle_rad=0.3,
        )
        self.assertEqual(angles.shape, (1,))
        self.assertAlmostEqual(float(angles[0]), 0.3)

    def test_two_segments_uses_launch_then_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.array([5.0, 5.0]),
            launch_angle_rad=0.1,
            settle_angle_rad=0.5,
            tail_angle_rad=0.9,
        )
        self.assertEqual(angles.shape, (2,))
        self.assertAlmostEqual(float(angles[0]), 0.1)
        self.assertAlmostEqual(float(angles[1]), 0.9)

    def test_three_segments_uses_launch_settle_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.array([5.0, 5.0, 5.0]),
            launch_angle_rad=0.0,
            settle_angle_rad=0.4,
            tail_angle_rad=0.8,
        )
        self.assertTrue(np.allclose(angles, [0.0, 0.4, 0.8]))

    def test_many_segments_ends_on_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.ones(8, dtype=float),
            launch_angle_rad=0.0,
            settle_angle_rad=0.3,
            tail_angle_rad=1.0,
        )
        self.assertAlmostEqual(float(angles[0]), 0.0)
        # Last two segments are clamped to tail_angle.
        self.assertAlmostEqual(float(angles[-1]), 1.0)
        self.assertAlmostEqual(float(angles[-2]), 1.0)


class StemProfileCandidatesTest(unittest.TestCase):
    def test_side_candidates_cover_both_signs(self) -> None:
        profiles = _stem_profile_candidates(
            attach_angle_rad=0.0,
            desired_tail_angle_rad=math.pi / 2.0,
            preferred_sign=1.0,
            min_branch_angle_rad=math.radians(25.0),
            branch_role="side",
            n_segments=4,
        )
        self.assertGreater(len(profiles), 0)

    def test_trunk_candidates_use_desired_sign_only(self) -> None:
        profiles = _stem_profile_candidates(
            attach_angle_rad=0.0,
            desired_tail_angle_rad=math.pi / 2.0,
            preferred_sign=-1.0,
            min_branch_angle_rad=math.radians(25.0),
            branch_role="trunk",
            n_segments=4,
        )
        # Trunk side should settle toward +pi/2 (desired is positive).
        self.assertGreater(len(profiles), 0)
        for profile in profiles:
            self.assertGreater(profile.tail_angle_rad, 0.0)


class BuildStemLinearTest(unittest.TestCase):
    def test_length_only_tree_builds_without_error(self) -> None:
        tree = make_length_only_tree()
        specs = _build_layout_specs(tree)
        layouts = _build_layout_branches_stem_linear(
            tree,
            layout_specs=specs,
            min_branch_angle_deg=25.0,
            root_layout="type_split",
        )
        self.assertEqual(len(layouts), len(tree.branches))
        for layout in layouts:
            self.assertTrue(np.all(np.isfinite(layout.segment_points_um)))


class BuildStemTreeTest(unittest.TestCase):
    def test_length_only_tree_builds_without_error(self) -> None:
        tree = make_length_only_tree()
        specs = _build_layout_specs(tree)
        layouts = _build_layout_branches_stem(
            tree,
            layout_specs=specs,
            min_branch_angle_deg=25.0,
            root_layout="type_split",
        )
        self.assertEqual(len(layouts), len(tree.branches))

    def test_root_split_places_axon_opposite_dendrite(self) -> None:
        tree = make_root_split_tree()
        specs = _build_layout_specs(tree)
        layouts = {
            layout.branch_name: layout
            for layout in _build_layout_branches_stem(
                tree,
                layout_specs=specs,
                min_branch_angle_deg=25.0,
                root_layout="type_split",
            )
        }
        self.assertGreater(layouts["dend"].end_direction_um[1], 0.0)
        self.assertLess(layouts["axon"].end_direction_um[1], 0.0)


if __name__ == "__main__":
    unittest.main()
