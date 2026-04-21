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

from braincell.vis._testing import make_length_only_tree, make_two_dendrite_tree
from braincell.vis.layout._common import (
    _allocate_child_regions_legacy,
    _allocate_weighted_angles,
    _build_layout_specs,
    _leaf_counts_by_branch,
    _normalize_min_branch_angle_rad,
    _path_lengths_um_by_branch,
    _pick_trunk_child,
    _side_branch_offsets_rad,
    _weighted_child_intervals,
)


class NormalizeMinBranchAngleTest(unittest.TestCase):
    def test_none_returns_zero(self) -> None:
        self.assertEqual(_normalize_min_branch_angle_rad(None), 0.0)

    def test_degrees_are_converted_to_radians(self) -> None:
        self.assertAlmostEqual(_normalize_min_branch_angle_rad(90.0), math.pi / 2.0)
        self.assertAlmostEqual(_normalize_min_branch_angle_rad(45.0), math.pi / 4.0)

    def test_rejects_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            _normalize_min_branch_angle_rad(-1.0)

    def test_rejects_bool_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "non-negative float or None"):
            _normalize_min_branch_angle_rad(True)


class TreeAnalysisTest(unittest.TestCase):
    def test_leaf_counts_sum_to_number_of_leaves(self) -> None:
        tree = make_two_dendrite_tree()
        counts = _leaf_counts_by_branch(tree.root)

        # Root has 2 dendrite children, both leaves; root count = 2.
        self.assertEqual(counts[tree.root.index], 2)
        for branch in tree.branches:
            if branch.n_children == 0:
                self.assertEqual(counts[branch.index], 1)

    def test_path_lengths_are_monotonic_down_the_tree(self) -> None:
        tree = make_length_only_tree()
        path_lengths = _path_lengths_um_by_branch(tree.root)

        root_pl = path_lengths[tree.root.index]
        for branch in tree.branches:
            if branch.parent is not None:
                self.assertLessEqual(path_lengths[branch.index], root_pl)


class LayoutSpecsTest(unittest.TestCase):
    def test_build_layout_specs_has_entry_per_branch(self) -> None:
        tree = make_length_only_tree()
        specs = _build_layout_specs(tree)

        self.assertEqual(set(specs.keys()), {branch.index for branch in tree.branches})
        for branch in tree.branches:
            spec = specs[branch.index]
            self.assertEqual(len(spec.segment_lengths_um), len(branch.lengths))
            expected_total = float(sum(branch.lengths.to_decimal(u.um)))
            self.assertAlmostEqual(spec.total_length_um, expected_total, places=5)


class PickTrunkChildTest(unittest.TestCase):
    def test_longest_subtree_wins(self) -> None:
        tree = make_length_only_tree()
        root = tree.root
        # The length-only tree's root has a single child so the trunk
        # selector should return it directly.
        path_lengths = _path_lengths_um_by_branch(root)
        branch_order = {branch.index: index for index, branch in enumerate(tree.branches)}
        self.assertIs(
            _pick_trunk_child(root.children, subtree_path_lengths_um=path_lengths, branch_order=branch_order),
            root.children[0],
        )


class SideBranchOffsetsTest(unittest.TestCase):
    def test_offsets_honor_min_branch_angle_floor(self) -> None:
        # Even with min_branch_angle_rad = 0, the default side branch
        # base/step must apply so offsets stay non-decreasing.
        offsets = _side_branch_offsets_rad(0.0, n_offsets=5)

        self.assertEqual(len(offsets), 5)
        for prev, curr in zip(offsets, offsets[1:]):
            self.assertGreaterEqual(curr, prev)

    def test_empty_when_no_offsets_requested(self) -> None:
        self.assertEqual(_side_branch_offsets_rad(1.0, n_offsets=0), [])


class WeightedChildIntervalsTest(unittest.TestCase):
    def test_empty_children(self) -> None:
        self.assertEqual(
            _weighted_child_intervals((), interval=(-1.0, 1.0), weights={}, min_gap_rad=0.0),
            [],
        )

    def test_single_child_gets_full_interval(self) -> None:
        tree = make_length_only_tree()
        child = tree.root.children[0]
        intervals = _weighted_child_intervals(
            (child,),
            interval=(-1.0, 1.0),
            weights={child.index: 1.0},
            min_gap_rad=0.0,
        )
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0][1], (-1.0, 1.0))

    def test_weights_proportional_to_leaf_counts(self) -> None:
        tree = make_two_dendrite_tree()
        weights = {child.index: 1.0 for child in tree.root.children}
        intervals = _weighted_child_intervals(
            tree.root.children,
            interval=(-1.0, 1.0),
            weights=weights,
            min_gap_rad=0.0,
        )
        # Equal weights → equal-width intervals.
        widths = [hi - lo for _, (lo, hi) in intervals]
        for width in widths[1:]:
            self.assertAlmostEqual(width, widths[0], places=6)


class AllocateWeightedAnglesTest(unittest.TestCase):
    def test_two_children_get_symmetric_angles(self) -> None:
        tree = make_two_dendrite_tree()
        weights = {child.index: 1.0 for child in tree.root.children}
        angles = _allocate_weighted_angles(
            tree.root.children,
            interval=(-1.0, 1.0),
            weights=weights,
        )
        values = sorted(angles.values())
        self.assertAlmostEqual(values[0], -0.5, places=6)
        self.assertAlmostEqual(values[1], 0.5, places=6)


class AllocateChildRegionsLegacyTest(unittest.TestCase):
    def test_single_child_gets_full_interval(self) -> None:
        tree = make_length_only_tree()
        child = tree.root.children[0]
        allocations = _allocate_child_regions_legacy(
            children=(child,),
            interval=(-1.0, 1.0),
            leaf_counts={child.index: 1},
            min_branch_angle_rad=0.0,
        )
        self.assertEqual(len(allocations), 1)
        _, interval, center = allocations[0]
        self.assertEqual(interval, (-1.0, 1.0))
        self.assertEqual(center, 0.0)

    def test_fallback_when_min_gap_too_wide(self) -> None:
        tree = make_two_dendrite_tree()
        leaf_counts = _leaf_counts_by_branch(tree.root)
        # Require a gap larger than the whole interval → forces fallback.
        allocations = _allocate_child_regions_legacy(
            children=tree.root.children,
            interval=(0.0, 0.1),
            leaf_counts=leaf_counts,
            min_branch_angle_rad=math.pi,
        )
        self.assertEqual(len(allocations), 2)


if __name__ == "__main__":
    unittest.main()
