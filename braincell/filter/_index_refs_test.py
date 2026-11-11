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

"""Tests for :mod:`braincell.filter._index_refs`."""

import dataclasses
import inspect
import unittest

import brainunit as u
import numpy as np

import braincell.filter  # noqa: F401 -- registers every expression class
from braincell.filter import (
    AllRegion,
    BranchSlice,
    LocsetExpr,
    RegionAnchors,
    RegionExpr,
    RootLocation,
    SubtreeRegion,
    Terminals,
    UniformSamples,
    at,
    branch_in,
    branch_range,
)
from braincell.filter._index_refs import INDEX_LEAVES, held_branch_indices
from braincell.morph._testing import make_tree_u

#: Fields of the other expression classes that never hold a branch index:
#: sub-expressions, coordinates, counts, seeds, quantities and flags.
_NOT_AN_INDEX = frozenset(
    {
        "region",
        "operand",
        "operands",
        "op",
        "x",
        "count",
        "number",
        "seed",
        "measure",
        "density",
        "u_resolution",
        "step",
        "minimum",
        "maximum",
        "closed",
    }
)


def _expression_classes() -> list[type]:
    """Every concrete region or locset expression class braincell defines (tests aside)."""
    found, stack = [], [RegionExpr, LocsetExpr]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if inspect.isabstract(cls) or not cls.__module__.startswith("braincell."):
            continue
        if cls.__module__.endswith("_test"):
            continue
        found.append(cls)
    return found


class HeldBranchIndicesTest(unittest.TestCase):
    def test_a_leaf_naming_a_branch_by_index_holds_it(self):
        cases = (
            (BranchSlice(1, 0.0, 1.0), (1,)),
            (BranchSlice([2, 1, 2], 0.0, 1.0), (2, 1)),
            (BranchSlice(np.asarray([3, 1]), 0.0, 0.5), (3, 1)),
            (SubtreeRegion(1), (1,)),
            (SubtreeRegion(np.int64(2)), (2,)),
            (at(1, 0.5), (1,)),
            (at(-1, 0.5), (-1,)),
            (branch_in("index", [1, 3]), (1, 3)),
            (branch_in("branch_id", 2), (2,)),
            (branch_in("parent_id", [-1, 1]), (1,)),
            (branch_range("index", (1, None)), (1,)),
            (branch_range("index", (0.5, 2.5)), (0.5, 2.5)),
        )
        for expression, indices in cases:
            with self.subTest(expression):
                self.assertEqual(held_branch_indices(expression), indices)

    def test_names_types_and_geometry_hold_no_index(self):
        cases = (
            AllRegion(),
            SubtreeRegion("dend"),
            at("axon", 0.5),
            RootLocation(0.5),
            Terminals(),
            branch_in("name", ["dend"]),
            branch_in("type", "axon"),
            branch_in("parent_id", -1),
            branch_range("length", (10.0 * u.um, None)),
            UniformSamples(branch_in("type", "axon"), 3),
        )
        for expression in cases:
            with self.subTest(expression):
                self.assertEqual(held_branch_indices(expression), ())

    def test_composite_expressions_are_walked_in_order_without_repeats(self):
        region = (branch_in("type", "soma") | BranchSlice(2, 0.0, 0.5)) - SubtreeRegion(1)
        self.assertEqual(held_branch_indices(region), (2, 1))
        self.assertEqual(held_branch_indices(region.complement()), (2, 1))
        locset = (RegionAnchors(region, 0.5) + at(3, 0.5) + at(2, 1.0)).unique()
        self.assertEqual(held_branch_indices(locset), (2, 1, 3))
        self.assertEqual(held_branch_indices(UniformSamples(SubtreeRegion(3), 4) | at("axon", 0.5)), (3,))

    def test_resolved_masks_and_other_values_hold_none(self):
        # A mask's indices are checked as a mask (MORPH-16), not as an expression.
        morpho = make_tree_u()
        for value in (at(1, 0.5).evaluate(morpho), BranchSlice(1, 0.0, 1.0).evaluate(morpho), None, "dend", 1):
            with self.subTest(value):
                self.assertEqual(held_branch_indices(value), ())

    def test_every_expression_class_is_classified(self):
        # A new leaf that holds a branch index must be added to INDEX_LEAVES,
        # or a Cell declaration with it would silently move to other branches
        # after a removal; a field that never holds one goes in _NOT_AN_INDEX.
        classes = _expression_classes()
        self.assertGreaterEqual(len(classes), 20, [cls.__name__ for cls in classes])
        for cls in classes:
            with self.subTest(cls.__name__):
                self.assertTrue(dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass")
                if cls in INDEX_LEAVES:
                    continue
                unknown = {item.name for item in dataclasses.fields(cls)} - _NOT_AN_INDEX
                self.assertEqual(unknown, set(), f"{cls.__name__} has unclassified fields {sorted(unknown)}")


if __name__ == "__main__":
    unittest.main()
