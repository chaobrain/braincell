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

``SelectionCache``'s three dictionaries are **reserved**: nothing in the
tree reads or writes them today. They exist for the four region types and
one locset type that would need memoized per-branch distances and radii —
:class:`RadiusRangeRegion`, :class:`TreeDistanceRegion`,
:class:`EuclideanDistanceRegion`, :class:`SubtreeRegion`, and
:class:`StepSamples` — every one of which currently raises
``NotImplementedError``.

So there is no memoization to test. What is real, load-bearing, and
otherwise untested is the *plumbing*: every composite region and locset
expression threads the caller's cache down to the leaves. These tests pin
that, and pin the reserved surface so it fails loudly when someone
implements one of the five without wiring the cache up.
"""

import unittest

import brainunit as u

from braincell import Branch, Morphology
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
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.0] * u.um, type="apical_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


class _SpyRegion(RegionExpr):
    """Leaf region that records the cache each ``evaluate`` call receives."""

    def __init__(self, intervals=((0, 0.0, 1.0),)) -> None:
        self.seen: list[SelectionCache | None] = []
        self._intervals = tuple(intervals)

    def evaluate(self, morpho, cache=None) -> RegionMask:
        self.seen.append(cache)
        return RegionMask(self._intervals)


class _SpyLocset(LocsetExpr):
    """Leaf locset that records the cache each ``evaluate`` call receives."""

    def __init__(self, points=((0, 0.5),)) -> None:
        self.seen: list[SelectionCache | None] = []
        self._points = tuple(points)

    def evaluate(self, morpho, cache=None) -> LocsetMask:
        self.seen.append(cache)
        return LocsetMask(points=self._points, display_names=("spy",) * len(self._points))


class SelectionCacheFieldsTest(unittest.TestCase):
    def test_fields_default_to_empty_dicts(self) -> None:
        cache = SelectionCache()

        self.assertEqual(cache.tree_distance_to_root, {})
        self.assertEqual(cache.euclidean_distance_to_root, {})
        self.assertEqual(cache.branch_radius_summary, {})

    def test_instances_do_not_share_their_dicts(self) -> None:
        # All three fields use default_factory; a plain `{}` default would
        # make every SelectionCache in the process alias one set of dicts.
        first = SelectionCache()
        second = SelectionCache()

        first.tree_distance_to_root[0] = 10.0 * u.um
        first.euclidean_distance_to_root[0] = 8.0 * u.um
        first.branch_radius_summary[0] = (1.0 * u.um, 2.0 * u.um)

        self.assertEqual(second.tree_distance_to_root, {})
        self.assertEqual(second.euclidean_distance_to_root, {})
        self.assertEqual(second.branch_radius_summary, {})

    def test_cache_is_mutable_so_callers_can_populate_it(self) -> None:
        # Not frozen, by design: the reserved consumers must be able to fill
        # these in during evaluation.
        cache = SelectionCache()

        cache.tree_distance_to_root = {1: 5.0 * u.um}

        self.assertEqual(set(cache.tree_distance_to_root), {1})


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
