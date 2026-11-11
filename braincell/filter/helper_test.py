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

import brainunit as u
import numpy as np

from braincell.filter import helper as helper_mod
from braincell.filter.helper import (
    EPSILON,
    _length_scalar_um,
    _linear_piece_select,
    _quadratic_sublevel,
    _segment_distance_sublevel,
    _value_in_range,
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
                "coerce_positive_count",
                "branch_in_intervals",
                "branch_range_intervals",
                "normalize_region_intervals",
                "union_region_intervals",
                "intersect_region_intervals",
                "difference_region_intervals",
                "complement_region_intervals",
                "fork_points_locations",
                "terminal_locations",
                "uniform_samples_from_region",
                "random_samples_from_region",
                "normalize_locset_points",
                "concat_locset_points",
                "unique_locset_points",
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


class LengthScalarUmTest(unittest.TestCase):
    """The scalar length check shared by StepSamples and the range regions."""

    def test_a_scalar_length_becomes_micrometres(self) -> None:
        self.assertEqual(_length_scalar_um(30 * u.um, name="step"), 30.0)
        self.assertEqual(_length_scalar_um(0.03 * u.mmeter, name="step"), 30.0)
        self.assertEqual(_length_scalar_um(np.array([5.0]) * u.um, name="step", positive=True), 5.0)
        self.assertEqual(_length_scalar_um(0 * u.um, name="minimum"), 0.0)
        self.assertIsNone(_length_scalar_um(None, name="maximum", allow_none=True))

    def test_bad_inputs_raise_the_same_errors_as_step_samples(self) -> None:
        for value, error, message in (
            (5.0, TypeError, "step must satisfy unit um"),
            (5 * u.ms, TypeError, "step must satisfy unit um"),
            (None, TypeError, "step cannot be None"),
            (0 * u.um, ValueError, "step must be > 0"),
            (-5 * u.um, ValueError, "step must be > 0"),
            (np.inf * u.um, ValueError, "step must be finite"),
            (np.nan * u.um, ValueError, "step must be finite"),
            (np.array([5.0, 6.0]) * u.um, ValueError, "step"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(error, message):
                _length_scalar_um(value, name="step", positive=True)

    def test_a_non_positive_check_still_rejects_negative_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum must be >= 0"):
            _length_scalar_um(-1 * u.um, name="minimum")
        with self.assertRaisesRegex(ValueError, "minimum must be finite; use None for no bound"):
            _length_scalar_um(np.inf * u.um, name="minimum", allow_none=True)


class LinearPieceSelectTest(unittest.TestCase):
    """The crossing solve behind the range regions: f linear from f0 at x0 to f1 at x1."""

    def test_a_rising_piece_is_cut_where_it_crosses_each_bound(self) -> None:
        self.assertEqual(_linear_piece_select(0.0, 1.0, 0.0, 4.0, 1.0, 3.0, "both"), (0.25, 0.75))
        self.assertEqual(_linear_piece_select(0.0, 1.0, 0.0, 4.0, 1.0, None, "both"), (0.25, 1.0))
        self.assertEqual(_linear_piece_select(0.0, 1.0, 0.0, 4.0, None, 3.0, "both"), (0.0, 0.75))
        self.assertEqual(_linear_piece_select(0.2, 0.6, 0.0, 4.0, None, None, "both"), (0.2, 0.6))

    def test_a_falling_piece_is_cut_from_the_other_side(self) -> None:
        self.assertEqual(_linear_piece_select(0.0, 1.0, 4.0, 0.0, 1.0, 3.0, "both"), (0.25, 0.75))
        self.assertEqual(_linear_piece_select(0.5, 1.0, 4.0, 0.0, 3.0, None, "both"), (0.5, 0.625))

    def test_closed_does_not_matter_where_f_varies(self) -> None:
        for closed in ("both", "left", "right", "neither"):
            with self.subTest(closed=closed):
                self.assertEqual(_linear_piece_select(0.0, 1.0, 0.0, 4.0, 1.0, 3.0, closed), (0.25, 0.75))

    def test_a_piece_outside_or_touching_at_one_point_selects_nothing(self) -> None:
        self.assertIsNone(_linear_piece_select(0.0, 1.0, 0.0, 1.0, 2.0, 3.0, "both"))
        self.assertIsNone(_linear_piece_select(0.0, 1.0, 0.0, 1.0, 1.0, 3.0, "both"))
        self.assertIsNone(_linear_piece_select(0.0, 1.0, 2.0, 4.0, 1.0, 2.0, "both"))

    def test_a_plateau_is_all_in_or_all_out_by_closed(self) -> None:
        cases = (
            ("both", (0.0, 1.0), (0.0, 1.0)),
            ("left", (0.0, 1.0), None),
            ("right", None, (0.0, 1.0)),
            ("neither", None, None),
        )
        for closed, at_lower, at_upper in cases:
            with self.subTest(closed=closed):
                self.assertEqual(_linear_piece_select(0.0, 1.0, 1.0, 1.0, 1.0, 3.0, closed), at_lower)
                self.assertEqual(_linear_piece_select(0.0, 1.0, 3.0, 3.0, 1.0, 3.0, closed), at_upper)
        self.assertEqual(_linear_piece_select(0.0, 1.0, 2.0, 2.0, 1.0, 3.0, "neither"), (0.0, 1.0))
        self.assertIsNone(_linear_piece_select(0.0, 1.0, 5.0, 5.0, 1.0, 3.0, "both"))

    def test_value_in_range_treats_none_as_unbounded(self) -> None:
        self.assertTrue(_value_in_range(1e9, None, None, "neither"))
        self.assertTrue(_value_in_range(0.0, 0.0, None, "both"))
        self.assertFalse(_value_in_range(0.0, 0.0, None, "neither"))
        self.assertFalse(_value_in_range(4.0, None, 3.0, "both"))


def _sublevel(a, b, c, t_lo=-10.0, t_hi=10.0, *, strict=False, scale=1.0):
    lo, hi = _quadratic_sublevel(
        np.array([a], dtype=float),
        np.array([b], dtype=float),
        np.array([c], dtype=float),
        t_lo,
        t_hi,
        strict=strict,
        scale=np.array([scale], dtype=float),
    )
    return float(lo[0]), float(hi[0])


class QuadraticSublevelTest(unittest.TestCase):
    """The solve behind EuclideanDistanceRegion: where a t^2 + b t + c <= 0 on [t_lo, t_hi]."""

    def test_two_roots_bound_the_set_and_the_domain_clips_it(self) -> None:
        self.assertEqual(_sublevel(1.0, 0.0, -1.0), (-1.0, 1.0))
        self.assertEqual(_sublevel(1.0, 0.0, -1.0, 0.0, 0.5), (0.0, 0.5))
        self.assertEqual(_sublevel(4.0, -4.0, 0.75), (0.25, 0.75))

    def test_no_real_root_or_a_single_touching_point_is_empty(self) -> None:
        lo, hi = _sublevel(1.0, 0.0, 1.0)
        self.assertGreater(lo, hi)
        lo, hi = _sublevel(1.0, 0.0, 0.0)  # touches 0 at t = 0 only
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_the_small_root_keeps_its_precision(self) -> None:
        # Roots 1e-8 and 1e8: the textbook formula loses the small one to cancellation.
        lo, hi = _sublevel(1.0, -(1e8 + 1e-8), 1.0, 0.0, 1.0)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo / 1e-8, 1.0, places=14)

    def test_a_negligible_leading_coefficient_is_solved_as_a_line(self) -> None:
        self.assertEqual(_sublevel(1e-14, 2.0, -1.0, 0.0, 1.0), (0.0, 0.5))
        self.assertEqual(_sublevel(0.0, -2.0, 1.0, 0.0, 1.0), (0.5, 1.0))
        # The threshold is relative: the same coefficient is a real quadratic at a small scale.
        lo, hi = _sublevel(1e-14, 0.0, -1e-14, scale=1e-6)
        self.assertEqual((lo, hi), (-1.0, 1.0))

    def test_a_constant_is_all_in_or_all_out_and_strict_decides_zero(self) -> None:
        self.assertEqual(_sublevel(0.0, 0.0, -1.0, 0.0, 1.0), (0.0, 1.0))
        self.assertEqual(_sublevel(0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 1.0))
        lo, hi = _sublevel(0.0, 0.0, 0.0, 0.0, 1.0, strict=True)
        self.assertGreater(lo, hi)
        lo, hi = _sublevel(0.0, 1e-13, 1.0, 0.0, 1.0)
        self.assertGreater(lo, hi)


def _segment_point_distance(points, a, b):
    w = b - a
    ww = float(w @ w)
    t = np.zeros(len(points)) if ww == 0.0 else np.clip((points - a) @ w / ww, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * w), axis=1)


class SegmentDistanceSublevelTest(unittest.TestCase):
    """Where each segment p(t) = start + t * step is within a radius of a reference segment [a, b]."""

    def assertMatchesSampling(self, starts, steps, a, b, radius) -> None:
        ts = np.linspace(0.0, 1.0, 2001)
        for strict in (False, True):
            lo, hi = _segment_distance_sublevel(starts, steps, a, b, radius, strict=strict)
            for i in range(len(starts)):
                f = _segment_point_distance(starts[i] + ts[:, None] * steps[i], a, b)
                want = (f < radius) if strict else (f <= radius)
                got = (ts >= lo[i]) & (ts <= hi[i])
                # Samples next to a crossing may fall either way.
                near = (np.abs(f - radius) <= 1e-9) | (np.minimum(np.abs(ts - lo[i]), np.abs(ts - hi[i])) <= 1e-9)
                with self.subTest(i=i, strict=strict):
                    self.assertFalse(np.any((want != got) & ~near), (lo[i], hi[i]))

    def test_random_segments_match_sampling(self) -> None:
        rng = np.random.default_rng(0)
        starts = rng.uniform(-20.0, 20.0, (300, 3))
        steps = rng.uniform(-30.0, 30.0, (300, 3))
        for a, b in (
            (np.zeros(3), np.array([10.0, 0.0, 0.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([-4.0, 8.0, 1.0])),
        ):
            self.assertMatchesSampling(starts, steps, a, b, 15.0)

    def test_a_zero_length_reference_is_a_point(self) -> None:
        a = np.array([1.0, 1.0, 0.0])
        lo, hi = _segment_distance_sublevel(
            np.array([[-4.0, 4.0, 0.0]]), np.array([[10.0, 0.0, 0.0]]), a, a, 5.0, strict=False
        )
        np.testing.assert_allclose((lo[0], hi[0]), (0.1, 0.9), rtol=0.0, atol=1e-15)
        rng = np.random.default_rng(1)
        self.assertMatchesSampling(rng.uniform(-10.0, 10.0, (50, 3)), rng.uniform(-20.0, 20.0, (50, 3)), a, a, 6.0)

    def test_a_segment_parallel_or_perpendicular_to_the_reference(self) -> None:
        a, b = np.zeros(3), np.array([10.0, 0.0, 0.0])
        # Perpendicular (the nearest point does not move) beside, before and past the reference.
        starts = np.array([[5.0, -3.0, 4.0], [-3.0, 0.0, -10.0], [14.0, 0.0, -10.0], [5.0, 6.0, 0.0]])
        steps = np.array([[0.0, 0.0, -8.0], [0.0, 0.0, 20.0], [0.0, 0.0, 20.0], [0.0, 0.0, 1.0]])
        lo, hi = _segment_distance_sublevel(starts, steps, a, b, 5.0, strict=False)
        np.testing.assert_allclose(lo[:3], (0.0, 0.5 - 0.2, 0.5 - 0.15), rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(hi[:3], (1.0, 0.5 + 0.2, 0.5 + 0.15), rtol=0.0, atol=1e-15)
        self.assertGreater(lo[3], hi[3])  # 6 um away all along
        # Parallel at 4 um, running past both ends: within 5 um from x = -3 to 13.
        lo, hi = _segment_distance_sublevel(
            np.array([[-10.0, 4.0, 0.0]]), np.array([[30.0, 0.0, 0.0]]), a, b, 5.0, strict=False
        )
        np.testing.assert_allclose((lo[0], hi[0]), (7 / 30, 23 / 30), rtol=0.0, atol=1e-15)
        self.assertMatchesSampling(starts, steps, a, b, 5.0)

    def test_strict_drops_a_tangency_and_keeps_the_rest(self) -> None:
        a, b = np.zeros(3), np.array([10.0, 0.0, 0.0])
        starts, steps = np.array([[0.0, 5.0, 0.0]]), np.array([[10.0, 0.0, 0.0]])  # 5 um all along
        self.assertEqual(
            tuple(map(float, np.ravel(_segment_distance_sublevel(starts, steps, a, b, 5.0, strict=False)))), (0.0, 1.0)
        )
        lo, hi = _segment_distance_sublevel(starts, steps, a, b, 5.0, strict=True)
        self.assertGreater(lo[0], hi[0])


if __name__ == "__main__":
    unittest.main()
