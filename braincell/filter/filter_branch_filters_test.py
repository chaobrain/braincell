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

from braincell._test_support import u

from braincell import Branch, Morpho
from braincell.filter import BranchInFilter, BranchRangeFilter, branch_in, branch_range


def _build_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    axon = Branch.from_lengths(lengths=[120.0] * u.um, radii=[0.8, 0.5] * u.um, type="axon")
    tuft = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 0.6] * u.um, type="apical_dendrite")

    tree = Morpho.from_root(soma, name="soma")
    tree.soma.dend = dend
    tree.soma.axon = axon
    tree.soma.dend.tuft = tuft
    return tree


class BranchFilterTest(unittest.TestCase):
    def test_branch_in_filter_supports_type_and_name(self) -> None:
        tree = _build_tree()

        region_type = BranchInFilter(property="type", values="axon").evaluate(tree)
        region_name = BranchInFilter(property="name", values={"soma", "tuft"}).evaluate(tree)

        self.assertEqual(region_type.intervals, ((2, 0.0, 1.0),))
        self.assertEqual(region_name.intervals, ((0, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_branch_in_filter_supports_topology_properties(self) -> None:
        tree = _build_tree()

        by_order = BranchInFilter(property="branch_order", values=[1]).evaluate(tree)
        by_parent = BranchInFilter(property="parent_id", values={1}).evaluate(tree)
        by_children = BranchInFilter(property="n_children", values=0).evaluate(tree)

        self.assertEqual(by_order.intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0)))
        self.assertEqual(by_parent.intervals, ((3, 0.0, 1.0),))
        self.assertEqual(by_children.intervals, ((2, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_branch_range_filter_supports_closed_semantics(self) -> None:
        tree = _build_tree()

        neither = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="neither").evaluate(tree)
        left = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="left").evaluate(tree)
        right = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="right").evaluate(tree)
        both = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="both").evaluate(tree)

        self.assertEqual(neither.intervals, ())
        self.assertEqual(left.intervals, ((1, 0.0, 1.0),))
        self.assertEqual(right.intervals, ((2, 0.0, 1.0),))
        self.assertEqual(both.intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0)))

    def test_branch_range_filter_supports_quantity_bounds(self) -> None:
        tree = _build_tree()

        shorter = BranchRangeFilter(
            property="length",
            bounds=(None, 80.0 * u.um),
            closed="right",
        ).evaluate(tree)
        longer = BranchRangeFilter(
            property="length",
            bounds=(80.0 * u.um, None),
            closed="left",
        ).evaluate(tree)

        self.assertEqual(shorter.intervals, ((0, 0.0, 1.0), (1, 0.0, 1.0), (3, 0.0, 1.0)))
        self.assertEqual(longer.intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0)))

    def test_branch_range_filter_supports_vector_quantity_bounds(self) -> None:
        tree = _build_tree()

        in_length_window = BranchRangeFilter(
            property="length",
            bounds=(0, 80) * u.um,
            closed="right",
        ).evaluate(tree)

        self.assertEqual(in_length_window.intervals, ((0, 0.0, 1.0), (1, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_helper_constructors_match_class_behavior(self) -> None:
        tree = _build_tree()

        region_a = branch_in("type", {"soma", "axon"}).evaluate(tree)
        region_b = branch_range("branch_order", (1, None), closed="left").evaluate(tree)

        self.assertEqual(region_a.intervals, ((0, 0.0, 1.0), (2, 0.0, 1.0)))
        self.assertEqual(region_b.intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_invalid_conditions_raise_clear_errors(self) -> None:
        tree = _build_tree()

        with self.assertRaises(ValueError):
            BranchInFilter(property="missing_property", values=1).evaluate(tree)
        with self.assertRaises(TypeError):
            BranchInFilter(property="children", values=0).evaluate(tree)
        with self.assertRaises(ValueError):
            BranchRangeFilter(property="length", bounds=(0, 100), closed="both").evaluate(tree)
        with self.assertRaises(ValueError):
            BranchRangeFilter(property="n_children", bounds=(0 * u.um, 1 * u.um), closed="both").evaluate(tree)
        with self.assertRaises(ValueError):
            BranchRangeFilter(property="branch_id", bounds=(2, 1), closed="both").evaluate(tree)
        with self.assertRaises(ValueError):
            BranchRangeFilter(property="volume", bounds=(None, 100 * (u.um**3)), closed="right").evaluate(tree)
        with self.assertRaises(TypeError):
            BranchRangeFilter(property="branch_id", bounds=(0, 1), unit=u.um, closed="both")
