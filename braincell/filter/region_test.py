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

"""Tests for :mod:`braincell.filter.region`."""

import unittest

import brainunit as u

from braincell import Branch, Morphology
from braincell.filter._testing import make_apical, make_soma
from braincell.filter import (
    BranchInFilter,
    BranchRangeFilter,
    BranchSlice,
    RegionSetOp,
    branch_in,
    branch_range,
)
from braincell.filter import region as region_mod


def _soma_dend_axon_tree(*, with_tuft: bool = False) -> Morphology:
    """Soma with one basal dendrite and one axon, optionally plus an apical tuft.

    These dimensions are load-bearing for the range assertions below (the
    80 / 120 um lengths straddle several of the tested bounds), so they stay
    spelled out here rather than moving to the shared builders.
    """
    dend = Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    axon = Branch.from_lengths(lengths=[120.0] * u.um, radii=[0.8, 0.5] * u.um, type="axon")

    tree = Morphology.from_root(make_soma(), name="soma")
    tree.soma.dend = dend
    tree.soma.axon = axon
    if with_tuft:
        tree.soma.dend.tuft = make_apical()
    return tree


class BranchFilterTest(unittest.TestCase):
    def test_branch_in_filter_supports_type_and_name(self) -> None:
        tree = _soma_dend_axon_tree(with_tuft=True)

        region_type = BranchInFilter(property="type", values="axon").evaluate(tree)
        region_name = BranchInFilter(property="name", values={"soma", "tuft"}).evaluate(tree)

        self.assertEqual(region_type.intervals, ((2, 0.0, 1.0),))
        self.assertEqual(region_name.intervals, ((0, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_branch_in_filter_supports_topology_properties(self) -> None:
        tree = _soma_dend_axon_tree(with_tuft=True)

        by_order = BranchInFilter(property="branch_order", values=[1]).evaluate(tree)
        by_parent = BranchInFilter(property="parent_id", values={1}).evaluate(tree)
        by_children = BranchInFilter(property="n_children", values=0).evaluate(tree)

        self.assertEqual(by_order.intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0)))
        self.assertEqual(by_parent.intervals, ((3, 0.0, 1.0),))
        self.assertEqual(by_children.intervals, ((2, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_branch_range_filter_supports_closed_semantics(self) -> None:
        tree = _soma_dend_axon_tree(with_tuft=True)

        neither = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="neither").evaluate(tree)
        left = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="left").evaluate(tree)
        right = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="right").evaluate(tree)
        both = BranchRangeFilter(property="branch_id", bounds=(1, 2), closed="both").evaluate(tree)

        self.assertEqual(neither.intervals, ())
        self.assertEqual(left.intervals, ((1, 0.0, 1.0),))
        self.assertEqual(right.intervals, ((2, 0.0, 1.0),))
        self.assertEqual(both.intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0)))

    def test_branch_range_filter_supports_quantity_bounds(self) -> None:
        tree = _soma_dend_axon_tree(with_tuft=True)

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
        tree = _soma_dend_axon_tree(with_tuft=True)

        in_length_window = BranchRangeFilter(
            property="length",
            bounds=(0, 80) * u.um,
            closed="right",
        ).evaluate(tree)

        self.assertEqual(in_length_window.intervals, ((0, 0.0, 1.0), (1, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_branch_filters_support_geometry_metric_properties(self) -> None:
        tree = _soma_dend_axon_tree(with_tuft=True)

        by_mean_radius = branch_range(
            "mean_radius",
            (0.75 * u.um, 1.5 * u.um),
            closed="both",
        ).evaluate(tree)
        by_area = branch_range(
            "area",
            (500.0 * (u.um**2), 800.0 * (u.um**2)),
            closed="both",
        ).evaluate(tree)
        by_volume = branch_range(
            "volume",
            (None, 200.0 * (u.um**3)),
            closed="right",
        ).evaluate(tree)
        by_exact_mean_radius = branch_in(
            "mean_radius",
            [10.0 * u.um, 1.5 * u.um],
        ).evaluate(tree)
        by_diam_arc_mean = branch_range(
            "diam_arc_mean",
            (1.5 * u.um, 3.0 * u.um),
            closed="both",
        ).evaluate(tree)

        self.assertEqual(by_mean_radius.intervals, ((1, 0.0, 1.0), (3, 0.0, 1.0)))
        self.assertEqual(by_area.intervals, ((1, 0.0, 1.0),))
        self.assertEqual(by_volume.intervals, ((2, 0.0, 1.0), (3, 0.0, 1.0)))
        self.assertEqual(by_exact_mean_radius.intervals, ((0, 0.0, 1.0), (1, 0.0, 1.0)))
        self.assertEqual(by_diam_arc_mean.intervals, ((1, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_helper_constructors_match_class_behavior(self) -> None:
        tree = _soma_dend_axon_tree(with_tuft=True)

        region_a = branch_in("type", {"soma", "axon"}).evaluate(tree)
        region_b = branch_range("branch_order", (1, None), closed="left").evaluate(tree)

        self.assertEqual(region_a.intervals, ((0, 0.0, 1.0), (2, 0.0, 1.0)))
        self.assertEqual(region_b.intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_invalid_conditions_raise_clear_errors(self) -> None:
        tree = _soma_dend_axon_tree(with_tuft=True)
        multi_segment_tree = Morphology.from_root(
            Branch.from_lengths(
                lengths=[10.0, 20.0] * u.um,
                radii=[2.0, 1.5, 1.0] * u.um,
                type="dendrite",
            ),
            name="dend",
        )

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
            BranchRangeFilter(property="max_radius", bounds=(None, 2 * u.um), closed="right").evaluate(tree)
        with self.assertRaises(TypeError):
            BranchRangeFilter(
                property="areas",
                bounds=(None, 100 * (u.um**2)),
                closed="right",
            ).evaluate(multi_segment_tree)
        with self.assertRaises(TypeError):
            BranchRangeFilter(property="branch_id", bounds=(0, 1), unit=u.um, closed="both")


class BranchSliceRegionTest(unittest.TestCase):
    def test_branch_slice_single_interval_remains_compatible(self) -> None:
        tree = _soma_dend_axon_tree()

        region = BranchSlice(branch_index=1, prox=0.2, dist=0.8).evaluate(tree)

        self.assertEqual(region.intervals, ((1, 0.2, 0.8),))

    def test_branch_slice_accepts_vector_inputs(self) -> None:
        tree = _soma_dend_axon_tree()

        region = BranchSlice(
            branch_index=[0, 2],
            prox=[0.0, 0.3],
            dist=[1.0, 0.9],
        ).evaluate(tree)

        self.assertEqual(region.intervals, ((0, 0.0, 1.0), (2, 0.3, 0.9)))

    def test_branch_slice_broadcasts_shared_prox_dist(self) -> None:
        tree = _soma_dend_axon_tree()

        region = BranchSlice(branch_index=[0, 1, 2], prox=0.0, dist=1.0).evaluate(tree)

        self.assertEqual(
            region.intervals,
            ((0, 0.0, 1.0), (1, 0.0, 1.0), (2, 0.0, 1.0)),
        )

    def test_branch_slice_broadcasts_single_branch_index(self) -> None:
        tree = _soma_dend_axon_tree()

        region = BranchSlice(branch_index=1, prox=[0.0, 0.2], dist=[0.1, 0.9]).evaluate(tree)

        self.assertEqual(region.intervals, ((1, 0.0, 0.1), (1, 0.2, 0.9)))

    def test_branch_slice_rejects_unbroadcastable_lengths(self) -> None:
        tree = _soma_dend_axon_tree()

        with self.assertRaises(ValueError):
            BranchSlice(
                branch_index=[0, 1],
                prox=[0.0, 0.2, 0.4],
                dist=[1.0, 0.9],
            ).evaluate(tree)

    def test_branch_slice_rejects_invalid_interval_bounds(self) -> None:
        tree = _soma_dend_axon_tree()

        invalid_intervals = [(-0.1, 0.3), (0.5, 0.5), (0.2, 1.1)]
        for prox, dist in invalid_intervals:
            with self.subTest(prox=prox, dist=dist):
                with self.assertRaises(ValueError):
                    BranchSlice(branch_index=0, prox=prox, dist=dist).evaluate(tree)

    def test_branch_slice_rejects_non_integer_branch_indices(self) -> None:
        tree = _soma_dend_axon_tree()

        for branch_index in (1.2, True, "1"):
            with self.subTest(branch_index=branch_index):
                with self.assertRaises(TypeError):
                    BranchSlice(branch_index=branch_index, prox=0.1, dist=0.9).evaluate(tree)

    def test_branch_slice_rejects_out_of_range_branch_indices(self) -> None:
        tree = _soma_dend_axon_tree()

        for branch_index in (-1, 3):
            with self.subTest(branch_index=branch_index):
                with self.assertRaises(IndexError):
                    BranchSlice(branch_index=branch_index, prox=0.1, dist=0.9).evaluate(tree)


class RegionSetOpTest(unittest.TestCase):
    def _assert_intervals_close(
        self,
        got: tuple[tuple[int, float, float], ...],
        expected: tuple[tuple[int, float, float], ...],
        *,
        places: int = 12,
    ) -> None:
        self.assertEqual(len(got), len(expected))
        for actual, target in zip(got, expected):
            self.assertEqual(actual[0], target[0])
            self.assertAlmostEqual(actual[1], target[1], places=places)
            self.assertAlmostEqual(actual[2], target[2], places=places)

    def test_union_intersection_difference_work_on_same_branch(self) -> None:
        tree = _soma_dend_axon_tree()
        left = BranchSlice(branch_index=0, prox=0.0, dist=0.6)
        right = BranchSlice(branch_index=0, prox=0.5, dist=1.0)

        union_region = (left | right).evaluate(tree)
        inter_region = (left & right).evaluate(tree)
        diff_region = (left - right).evaluate(tree)

        self._assert_intervals_close(union_region.intervals, ((0, 0.0, 1.0),))
        self._assert_intervals_close(inter_region.intervals, ((0, 0.5, 0.6),))
        self._assert_intervals_close(diff_region.intervals, ((0, 0.0, 0.5),))

    def test_complement_and_double_complement(self) -> None:
        tree = _soma_dend_axon_tree()
        expr = BranchSlice(branch_index=[0, 2], prox=[0.2, 0.1], dist=[0.8, 0.3])

        complement = expr.complement().evaluate(tree)
        double = expr.complement().complement().evaluate(tree)
        direct = expr.evaluate(tree)

        self._assert_intervals_close(
            complement.intervals,
            (
                (0, 0.0, 0.2),
                (0, 0.8, 1.0),
                (1, 0.0, 1.0),
                (2, 0.0, 0.1),
                (2, 0.3, 1.0),
            ),
        )
        self._assert_intervals_close(double.intervals, direct.intervals)

    def test_cross_branch_operations_do_not_interfere(self) -> None:
        tree = _soma_dend_axon_tree()
        base = BranchSlice(branch_index=0, prox=0.1, dist=0.9)
        other = BranchSlice(branch_index=1, prox=0.2, dist=0.8)

        diff = (other - base).evaluate(tree)
        union = (base | other).evaluate(tree)

        self._assert_intervals_close(diff.intervals, ((1, 0.2, 0.8),))
        self._assert_intervals_close(union.intervals, ((0, 0.1, 0.9), (1, 0.2, 0.8)))

    def test_touching_intervals_merge_with_epsilon(self) -> None:
        tree = _soma_dend_axon_tree()
        left = BranchSlice(branch_index=0, prox=0.0, dist=0.5)
        right = BranchSlice(branch_index=0, prox=0.5 + 1e-13, dist=1.0)

        region = (left | right).evaluate(tree)

        self._assert_intervals_close(region.intervals, ((0, 0.0, 1.0),))

    def test_region_setop_rejects_invalid_operator_and_arity(self) -> None:
        tree = _soma_dend_axon_tree()
        left = BranchSlice(branch_index=0, prox=0.1, dist=0.9)
        right = BranchSlice(branch_index=1, prox=0.2, dist=0.8)

        with self.assertRaises(ValueError):
            RegionSetOp("invalid", (left, right)).evaluate(tree)
        with self.assertRaises(ValueError):
            RegionSetOp("complement", (left, right)).evaluate(tree)
        with self.assertRaises(ValueError):
            RegionSetOp("union", (left,)).evaluate(tree)


class RegionExprOperatorsRejectNonRegionTest(unittest.TestCase):
    """MED-05: RegionExpr | int must raise TypeError, not build a broken op."""

    def test_or_with_int_raises(self) -> None:
        region = BranchSlice(branch_index=0, prox=0.0, dist=1.0)
        with self.assertRaises(TypeError):
            _ = region | 5

    def test_and_with_str_raises(self) -> None:
        region = BranchSlice(branch_index=0, prox=0.0, dist=1.0)
        with self.assertRaises(TypeError):
            _ = region & "foo"

    def test_sub_with_list_raises(self) -> None:
        region = BranchSlice(branch_index=0, prox=0.0, dist=1.0)
        with self.assertRaises(TypeError):
            _ = region - [1, 2, 3]


class RegionModuleAllTest(unittest.TestCase):
    def test_region_module_declares_all(self) -> None:
        self.assertIn("RegionSetOp", region_mod.__all__)
        self.assertIn("BranchSlice", region_mod.__all__)
        self.assertIn("branch_range", region_mod.__all__)


if __name__ == "__main__":
    unittest.main()
