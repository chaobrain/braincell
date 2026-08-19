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

"""Tests for :mod:`braincell.filter.helper`.

The four set operations are the arithmetic every :class:`RegionSetOp` in
``region.py`` delegates to, so a sign error here surfaces as a subtly wrong
selection rather than an exception. ``region_test.py`` drives them through the
expression tree; these tests pin the arithmetic itself.

All of it is pure functions over ``tuple[int, float, float]`` triples, so no
``Morphology`` is needed — the intervals are written out literally, which is
what makes the expected results checkable by eye.
"""

import unittest

from braincell.filter import helper as helper_mod
from braincell.filter.helper import (
    EPSILON,
    complement_region_intervals,
    difference_region_intervals,
    intersect_region_intervals,
    normalize_region_intervals,
    union_region_intervals,
)


class HelperModuleAllTest(unittest.TestCase):
    def test_every_declared_export_actually_exists(self) -> None:
        missing = [name for name in helper_mod.__all__ if not hasattr(helper_mod, name)]
        self.assertEqual(missing, [])

    def test_all_covers_the_public_interval_and_locset_surface(self) -> None:
        # The names region.py and locset.py import from this module. Kept as an
        # explicit list so deleting an export fails here rather than at the
        # import site of whichever expression happened to use it.
        self.assertEqual(
            set(helper_mod.__all__),
            {
                "EPSILON",
                "branch_slice_intervals",
                "branch_in_intervals",
                "branch_range_intervals",
                "normalize_region_intervals",
                "union_region_intervals",
                "intersect_region_intervals",
                "difference_region_intervals",
                "complement_region_intervals",
                "branch_points_locations",
                "terminal_locations",
                "uniform_samples_from_region",
                "random_samples_from_region",
                "normalize_locset_points",
                "union_locset_points",
                "intersect_locset_points",
                "difference_locset_points",
            },
        )


class NormalizeRegionIntervalsTest(unittest.TestCase):
    def test_overlapping_ranges_on_one_branch_merge(self) -> None:
        result = normalize_region_intervals([(0, 0.0, 0.5), (0, 0.3, 0.8)])
        self.assertEqual(result, ((0, 0.0, 0.8),))

    def test_exactly_touching_ranges_merge(self) -> None:
        # start <= current_end + epsilon, so abutting ranges become one.
        result = normalize_region_intervals([(0, 0.0, 0.5), (0, 0.5, 1.0)])
        self.assertEqual(result, ((0, 0.0, 1.0),))

    def test_disjoint_ranges_on_one_branch_stay_separate(self) -> None:
        result = normalize_region_intervals([(0, 0.6, 0.9), (0, 0.0, 0.4)])
        self.assertEqual(result, ((0, 0.0, 0.4), (0, 0.6, 0.9)))

    def test_a_range_fully_inside_another_is_absorbed(self) -> None:
        result = normalize_region_intervals([(0, 0.0, 1.0), (0, 0.4, 0.6)])
        self.assertEqual(result, ((0, 0.0, 1.0),))

    def test_output_is_sorted_by_branch_then_start(self) -> None:
        result = normalize_region_intervals([(2, 0.0, 0.5), (0, 0.5, 1.0), (1, 0.0, 1.0)])
        self.assertEqual(result, ((0, 0.5, 1.0), (1, 0.0, 1.0), (2, 0.0, 0.5)))

    def test_zero_length_ranges_are_dropped(self) -> None:
        self.assertEqual(normalize_region_intervals([(0, 0.5, 0.5)]), ())

    def test_ranges_thinner_than_epsilon_are_dropped(self) -> None:
        self.assertEqual(normalize_region_intervals([(0, 0.5, 0.5 + EPSILON / 2)]), ())

    def test_a_hair_outside_the_unit_interval_is_clipped_not_rejected(self) -> None:
        # _clip_norm_x absorbs float error from upstream length arithmetic.
        result = normalize_region_intervals([(0, -EPSILON / 2, 1.0 + EPSILON / 2)])
        self.assertEqual(result, ((0, 0.0, 1.0),))

    def test_clearly_out_of_range_coordinates_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            normalize_region_intervals([(0, 0.0, 1.5)])

    def test_non_integer_branch_index_raises_type_error(self) -> None:
        with self.assertRaisesRegex(TypeError, "Branch index must be int"):
            normalize_region_intervals([(0.0, 0.0, 1.0)])

    def test_empty_input_yields_empty_output(self) -> None:
        self.assertEqual(normalize_region_intervals([]), ())


class UnionRegionIntervalsTest(unittest.TestCase):
    def test_union_merges_across_the_two_operands(self) -> None:
        result = union_region_intervals([(0, 0.0, 0.5)], [(0, 0.4, 1.0)])
        self.assertEqual(result, ((0, 0.0, 1.0),))

    def test_union_keeps_branches_present_on_only_one_side(self) -> None:
        result = union_region_intervals([(0, 0.0, 1.0)], [(1, 0.0, 1.0)])
        self.assertEqual(result, ((0, 0.0, 1.0), (1, 0.0, 1.0)))

    def test_union_is_order_insensitive(self) -> None:
        left, right = [(1, 0.2, 0.4)], [(0, 0.0, 0.3)]
        self.assertEqual(
            union_region_intervals(left, right),
            union_region_intervals(right, left),
        )


class IntersectRegionIntervalsTest(unittest.TestCase):
    def test_intersection_is_the_overlap(self) -> None:
        result = intersect_region_intervals([(0, 0.0, 0.6)], [(0, 0.4, 1.0)])
        self.assertEqual(result, ((0, 0.4, 0.6),))

    def test_a_branch_present_on_only_one_side_drops_out(self) -> None:
        self.assertEqual(intersect_region_intervals([(0, 0.0, 1.0)], [(1, 0.0, 1.0)]), ())

    def test_disjoint_ranges_on_the_same_branch_yield_nothing(self) -> None:
        self.assertEqual(intersect_region_intervals([(0, 0.0, 0.3)], [(0, 0.7, 1.0)]), ())

    def test_touching_at_a_single_point_is_not_an_overlap(self) -> None:
        # The shared endpoint has zero length, so it is not a region.
        self.assertEqual(intersect_region_intervals([(0, 0.0, 0.5)], [(0, 0.5, 1.0)]), ())

    def test_one_range_intersecting_two_yields_both_pieces(self) -> None:
        result = intersect_region_intervals(
            [(0, 0.0, 1.0)],
            [(0, 0.1, 0.2), (0, 0.6, 0.8)],
        )
        self.assertEqual(result, ((0, 0.1, 0.2), (0, 0.6, 0.8)))


class DifferenceRegionIntervalsTest(unittest.TestCase):
    def test_removing_the_tail_shortens_the_range(self) -> None:
        result = difference_region_intervals([(0, 0.0, 1.0)], [(0, 0.5, 1.0)])
        self.assertEqual(result, ((0, 0.0, 0.5),))

    def test_removing_the_middle_splits_the_range_in_two(self) -> None:
        result = difference_region_intervals([(0, 0.0, 1.0)], [(0, 0.4, 0.6)])
        self.assertEqual(result, ((0, 0.0, 0.4), (0, 0.6, 1.0)))

    def test_a_branch_absent_from_the_right_passes_through_whole(self) -> None:
        result = difference_region_intervals([(0, 0.0, 1.0), (1, 0.2, 0.8)], [(0, 0.0, 1.0)])
        self.assertEqual(result, ((1, 0.2, 0.8),))

    def test_subtracting_a_superset_leaves_nothing(self) -> None:
        self.assertEqual(difference_region_intervals([(0, 0.2, 0.8)], [(0, 0.0, 1.0)]), ())

    def test_difference_is_not_symmetric(self) -> None:
        left, right = [(0, 0.0, 0.6)], [(0, 0.4, 1.0)]
        self.assertEqual(difference_region_intervals(left, right), ((0, 0.0, 0.4),))
        self.assertEqual(difference_region_intervals(right, left), ((0, 0.6, 1.0),))


class ComplementRegionIntervalsTest(unittest.TestCase):
    def test_complement_spans_the_branches_the_input_never_mentions(self) -> None:
        result = complement_region_intervals([(0, 0.0, 1.0)], n_branches=3)
        self.assertEqual(result, ((1, 0.0, 1.0), (2, 0.0, 1.0)))

    def test_complement_of_a_partial_range_keeps_the_rest_of_that_branch(self) -> None:
        result = complement_region_intervals([(0, 0.25, 0.75)], n_branches=1)
        self.assertEqual(result, ((0, 0.0, 0.25), (0, 0.75, 1.0)))

    def test_complement_of_everything_is_empty(self) -> None:
        universe = [(idx, 0.0, 1.0) for idx in range(3)]
        self.assertEqual(complement_region_intervals(universe, n_branches=3), ())

    def test_complement_of_nothing_is_every_branch(self) -> None:
        result = complement_region_intervals([], n_branches=2)
        self.assertEqual(result, ((0, 0.0, 1.0), (1, 0.0, 1.0)))

    def test_intervals_beyond_n_branches_do_not_leak_into_the_result(self) -> None:
        # The universe is built from n_branches alone, so a stale index on the
        # right cannot conjure a branch the caller never asked about.
        result = complement_region_intervals([(5, 0.0, 1.0)], n_branches=2)
        self.assertEqual(result, ((0, 0.0, 1.0), (1, 0.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
