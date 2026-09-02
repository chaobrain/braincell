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

"""Tests for :mod:`braincell.filter.cache`.

Three things are live and tested here:

* the *plumbing* — every composite region and locset expression threads
  the caller's cache down to the leaves;
* *sub-expression memoization* — an operand that appears more than once in
  one tree is evaluated once;
* *invalidation* — a memoized mask is dropped when the cache meets a
  different morphology, or the same morphology after a structural change.

The reserved surface is pinned too, so implementing one of the five
without wiring the cache up fails loudly.
"""

import unittest

import brainunit as u

from braincell import Branch, Morphology
from braincell.filter._testing import make_soma_dend_tree
from braincell.filter import AllRegion, SelectionCache
from braincell.filter.locset import (
    LocsetExpr,
    LocsetMask,
    LocsetSetOp,
    StepSamples,
    UniformSamples,
)
from braincell.filter.region import (
    EuclideanDistanceRegion,
    RadiusRangeRegion,
    RegionExpr,
    RegionMask,
    RegionSetOp,
    SubtreeRegion,
    TreeDistanceRegion,
)


def _soma_dend_tree() -> Morphology:
    """A two-branch tree: soma plus one apical dendrite."""
    return make_soma_dend_tree(dend_type="apical_dendrite")


class _SpyRegion(RegionExpr):
    """Leaf region that records the cache each ``evaluate`` call receives."""

    def __init__(self, intervals=((0, 0.0, 1.0),)) -> None:
        self.seen: list[SelectionCache | None] = []
        self._intervals = tuple(intervals)

    def _evaluate(self, morpho, cache=None) -> RegionMask:
        self.seen.append(cache)
        return RegionMask(self._intervals)


class _SpyLocset(LocsetExpr):
    """Leaf locset that records the cache each ``evaluate`` call receives."""

    def __init__(self, points=((0, 0.5),)) -> None:
        self.seen: list[SelectionCache | None] = []
        self._points = tuple(points)

    def _evaluate(self, morpho, cache=None) -> LocsetMask:
        self.seen.append(cache)
        return LocsetMask(points=self._points, display_names=("spy",) * len(self._points))


class RegionCacheThreadingTest(unittest.TestCase):
    """A cache handed to a composite must reach every leaf beneath it."""

    def setUp(self) -> None:
        self.tree = _soma_dend_tree()
        self.cache = SelectionCache()

    def test_union_forwards_the_same_cache_to_both_operands(self) -> None:
        left, right = _SpyRegion(), _SpyRegion()

        (left | right).evaluate(self.tree, self.cache)

        # assertIs, not assertEqual: SelectionCache is a plain dataclass, so
        # two empty caches compare equal and == would not catch a leaf being
        # handed a fresh instance.
        self.assertEqual(len(left.seen), 1)
        self.assertIs(left.seen[0], self.cache)
        self.assertEqual(len(right.seen), 1)
        self.assertIs(right.seen[0], self.cache)

    def test_intersection_forwards_the_cache(self) -> None:
        left, right = _SpyRegion(), _SpyRegion()

        (left & right).evaluate(self.tree, self.cache)

        self.assertIs(left.seen[0], self.cache)
        self.assertIs(right.seen[0], self.cache)

    def test_difference_forwards_the_cache(self) -> None:
        left, right = _SpyRegion(), _SpyRegion()

        (left - right).evaluate(self.tree, self.cache)

        self.assertIs(left.seen[0], self.cache)
        self.assertIs(right.seen[0], self.cache)

    def test_complement_forwards_the_cache(self) -> None:
        operand = _SpyRegion()

        operand.complement().evaluate(self.tree, self.cache)

        self.assertIs(operand.seen[0], self.cache)

    def test_nested_composites_reach_the_deepest_leaf(self) -> None:
        deep = _SpyRegion()

        ((deep | _SpyRegion()) & AllRegion()).evaluate(self.tree, self.cache)

        self.assertIs(deep.seen[0], self.cache)

    def test_absent_cache_arrives_as_none_rather_than_a_fresh_instance(self) -> None:
        # Callers that pass nothing must not silently get a per-leaf cache;
        # that would look like memoization while sharing nothing.
        left, right = _SpyRegion(), _SpyRegion()

        (left | right).evaluate(self.tree)

        self.assertEqual(left.seen, [None])
        self.assertEqual(right.seen, [None])

    def test_set_op_rejects_an_unknown_operation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported region set operation"):
            RegionSetOp("nope", (_SpyRegion(), _SpyRegion())).evaluate(self.tree, self.cache)


class LocsetCacheThreadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _soma_dend_tree()
        self.cache = SelectionCache()

    def test_uniform_samples_forwards_the_cache_to_its_region(self) -> None:
        region = _SpyRegion()

        UniformSamples(region=region, count=3).evaluate(self.tree, self.cache)

        self.assertIs(region.seen[0], self.cache)

    def test_locset_set_op_forwards_the_cache_to_both_operands(self) -> None:
        left, right = _SpyLocset(), _SpyLocset(points=((1, 0.5),))

        LocsetSetOp("union", (left, right)).evaluate(self.tree, self.cache)

        self.assertIs(left.seen[0], self.cache)
        self.assertIs(right.seen[0], self.cache)


class SubExpressionMemoizationTest(unittest.TestCase):
    """A repeated operand costs one traversal, not one per occurrence."""

    def setUp(self) -> None:
        self.tree = _soma_dend_tree()
        self.cache = SelectionCache()

    def test_a_shared_region_operand_is_evaluated_once(self) -> None:
        shared = _SpyRegion()
        left, right = _SpyRegion(), _SpyRegion()

        ((shared | left) & (shared | right)).evaluate(self.tree, self.cache)

        self.assertEqual(len(shared.seen), 1)
        self.assertEqual(len(left.seen), 1)
        self.assertEqual(len(right.seen), 1)

    def test_equal_but_distinct_expressions_share_one_result(self) -> None:
        # Region expressions are frozen dataclasses, so the memo key is
        # structural equality rather than object identity: two separately
        # constructed AllRegion() operands hit the same entry.
        spy = _SpyRegion()
        calls: list[Morphology] = []

        class _CountingAll(AllRegion):
            def _evaluate(self, morpho, cache=None):
                calls.append(morpho)
                return super()._evaluate(morpho, cache)

        ((_CountingAll() | spy) & (_CountingAll() | spy)).evaluate(self.tree, self.cache)

        self.assertEqual(len(calls), 1)

    def test_a_shared_locset_operand_is_evaluated_once(self) -> None:
        shared = _SpyLocset()

        LocsetSetOp("union", (shared, shared)).evaluate(self.tree, self.cache)

        self.assertEqual(len(shared.seen), 1)

    def test_without_a_cache_every_occurrence_is_recomputed(self) -> None:
        shared = _SpyRegion()

        ((shared | _SpyRegion()) & (shared | _SpyRegion())).evaluate(self.tree)

        self.assertEqual(shared.seen, [None, None])

    def test_a_second_morphology_does_not_reuse_the_first_result(self) -> None:
        shared = _SpyRegion()
        expr = shared | _SpyRegion()

        expr.evaluate(self.tree, self.cache)
        expr.evaluate(_soma_dend_tree(), self.cache)

        self.assertEqual(len(shared.seen), 2)

    def test_attaching_a_branch_invalidates_memoized_masks(self) -> None:
        # Morphology is mutable. A cache reused across an attach must not
        # answer from a mask computed for the smaller tree.
        shared = _SpyRegion()
        expr = shared | _SpyRegion()
        expr.evaluate(self.tree, self.cache)

        self.tree.attach(
            parent="dend",
            child_branch=Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon"),
            child_name="axon",
            parent_x=1.0,
        )
        expr.evaluate(self.tree, self.cache)

        self.assertEqual(len(shared.seen), 2)

    def test_an_unhashable_expression_falls_through_instead_of_raising(self) -> None:
        class _UnhashableSpy(_SpyRegion):
            __hash__ = None  # type: ignore[assignment]

        unhashable = _UnhashableSpy()

        mask = (unhashable | _SpyRegion()).evaluate(self.tree, self.cache)

        self.assertEqual(len(unhashable.seen), 1)
        self.assertIsInstance(mask, RegionMask)


class ReservedCacheConsumersTest(unittest.TestCase):
    """The five expressions that would populate the cache are unimplemented.

    When any of them grows a real implementation it must also start filling
    the matching ``SelectionCache`` dict — these assertions turn that into a
    deliberate, visible edit rather than a silent omission.
    """

    def setUp(self) -> None:
        self.tree = _soma_dend_tree()

    def test_radius_range_region_is_reserved(self) -> None:
        with self.assertRaises(NotImplementedError):
            RadiusRangeRegion(minimum=1.0 * u.um, maximum=2.0 * u.um).evaluate(self.tree)

    def test_tree_distance_region_is_reserved(self) -> None:
        with self.assertRaises(NotImplementedError):
            TreeDistanceRegion(minimum=0.0 * u.um, maximum=10.0 * u.um).evaluate(self.tree)

    def test_euclidean_distance_region_is_reserved(self) -> None:
        with self.assertRaises(NotImplementedError):
            EuclideanDistanceRegion(minimum=0.0 * u.um, maximum=10.0 * u.um).evaluate(self.tree)

    def test_subtree_region_is_reserved(self) -> None:
        with self.assertRaises(NotImplementedError):
            SubtreeRegion(root_branch_index=0).evaluate(self.tree)

    def test_step_samples_locset_is_reserved(self) -> None:
        with self.assertRaises(NotImplementedError):
            StepSamples(region=AllRegion(), step=5.0 * u.um).evaluate(self.tree)


if __name__ == "__main__":
    unittest.main()
