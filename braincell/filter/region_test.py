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

import copy
import itertools
import pickle
import unittest

import brainunit as u
import numpy as np

import braincell
from braincell import Branch, Morphology
from braincell.filter._testing import (
    brute_force_mismatches,
    crossing_errors,
    fixture_morphologies,
    make_apical,
    make_soma,
    make_tree_r,
    radius_um,
)
from braincell.filter import (
    AllRegion,
    BranchInFilter,
    BranchRangeFilter,
    BranchSlice,
    RadiusRangeRegion,
    RegionMask,
    RegionSetOp,
    SubtreeRegion,
    TreeDistanceRegion,
    branch_in,
    branch_range,
)
from braincell.filter import region as region_mod
from braincell.morph._testing import make_tree_t, make_tree_u


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


def _from_points(points, radii, type_name: str) -> Branch:
    return Branch.from_points(
        points=np.asarray(points, dtype=float) * u.um,
        radii=np.asarray(radii, dtype=float) * u.um,
        type=type_name,
    )


def _tree_b() -> Morphology:
    """Seven branches with a fork, a reversed child and children on both of its ends.

    ``d2`` hangs on ``dend``'s distal end by its x=1 end (``child_x=1``), so
    its x=0 end is the tip; ``d2tip`` attaches at that tip and ``d2side`` at
    the end facing the parent. Default order: soma, dend, d1, d2, d2tip,
    axon, d2side (indices 0-6).
    """
    tree = Morphology.from_root(_from_points([[0, 0, 0], [20, 0, 0]], [5, 5], "soma"), name="soma")
    tree.attach(
        parent="soma", child_name="dend", child_branch=_from_points([[20, 0, 0], [80, 0, 0]], [2, 1], "basal_dendrite")
    )
    tree.attach(
        parent="dend",
        child_name="d1",
        child_branch=_from_points([[80, 0, 0], [120, 30, 0]], [1, 0.5], "basal_dendrite"),
    )
    tree.attach(
        parent="dend",
        child_name="d2",
        child_x=1.0,
        child_branch=_from_points([[120, -30, 0], [80, 0, 0]], [0.5, 1], "basal_dendrite"),
    )
    tree.attach(
        parent="d2",
        child_name="d2tip",
        parent_x=0.0,
        child_branch=_from_points([[120, -30, 0], [150, -30, 0]], [0.5, 0.4], "basal_dendrite"),
    )
    tree.attach(
        parent="soma",
        child_name="axon",
        parent_x=0.0,
        child_branch=_from_points([[0, 0, 0], [-30, 0, 0]], [0.5, 0.5], "axon"),
    )
    tree.attach(
        parent="d2",
        child_name="d2side",
        parent_x=1.0,
        child_branch=_from_points([[80, 0, 0], [80, 20, 0]], [0.5, 0.5], "basal_dendrite"),
    )
    return tree


def _leak(name: str = "IL"):
    return braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV, name=name)


class SubtreeRegionTest(unittest.TestCase):
    """MORPH-9a: a branch and all its topological descendants, each whole."""

    def setUp(self) -> None:
        self.tree = _tree_b()

    def test_tree_b_layout_is_as_documented(self) -> None:
        self.assertEqual(
            [
                (b.name, None if b.parent is None else (b.parent.name, b.parent_x, b.child_x))
                for b in self.tree.branches
            ],
            [
                ("soma", None),
                ("dend", ("soma", 1.0, 0.0)),
                ("d1", ("dend", 1.0, 0.0)),
                ("d2", ("dend", 1.0, 1.0)),
                ("d2tip", ("d2", 0.0, 0.0)),
                ("axon", ("soma", 0.0, 0.0)),
                ("d2side", ("d2", 1.0, 0.0)),
            ],
        )

    def test_selects_the_root_and_every_descendant_whatever_child_x(self) -> None:
        select = lambda expr: self.tree.select(expr).intervals  # noqa: E731
        self.assertEqual(
            select(SubtreeRegion("dend")),
            ((1, 0.0, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0), (6, 0.0, 1.0)),
        )
        self.assertEqual(select(SubtreeRegion("d2")), ((3, 0.0, 1.0), (4, 0.0, 1.0), (6, 0.0, 1.0)))
        self.assertEqual(self.tree.select(SubtreeRegion(3)), self.tree.select(SubtreeRegion("d2")))
        self.assertEqual(select(SubtreeRegion("d1")), ((2, 0.0, 1.0),))
        self.assertEqual(select(SubtreeRegion("axon")), ((5, 0.0, 1.0),))
        self.assertEqual(select(SubtreeRegion(np.int64(2))), ((2, 0.0, 1.0),))
        self.assertEqual(self.tree.select(SubtreeRegion("soma")), self.tree.select(AllRegion()))

    def test_a_partial_root_is_written_with_region_algebra(self) -> None:
        self.assertEqual(
            self.tree.select(SubtreeRegion("dend") - BranchSlice(1, 0.0, 0.5)).intervals,
            ((1, 0.5, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0), (6, 0.0, 1.0)),
        )

    def test_selects_the_branches_morphology_subtree_copies(self) -> None:
        fixtures = fixture_morphologies()
        self.assertGreaterEqual(len(fixtures), 15)
        for name, tree in fixtures:
            with self.subTest(name):
                for branch in tree.branches:
                    selected = {tree.branches[i].name for i, _, _ in tree.select(SubtreeRegion(branch.name)).intervals}
                    copied = {b.name for b in tree.subtree(branch.name).branches}
                    self.assertEqual(selected, copied, branch.name)

    def test_editing_the_copy_leaves_the_selection_alone(self) -> None:
        copy = self.tree.subtree("d2")
        self.assertEqual([b.name for b in copy.branches], ["d2", "d2tip", "d2side"])
        before = self.tree.select(SubtreeRegion("d2"))
        copy.remove("d2tip")
        self.assertEqual(self.tree.select(SubtreeRegion("d2")), before)

    def test_construction_rejects_roots_that_are_not_a_name_or_an_index(self) -> None:
        for root in (True, 1.0, None):
            with self.subTest(root=root), self.assertRaisesRegex(TypeError, "branch name .str. or index"):
                SubtreeRegion(root)
        with self.assertRaisesRegex(TypeError, r"SubtreeRegion\('dend'\)"):
            SubtreeRegion(self.tree.dend)

    def test_unknown_roots_raise_at_select_and_at_paint(self) -> None:
        cases = (
            ("nope", KeyError, "nope"),
            (99, IndexError, r"Branch index 99 is out of range \[0, 7\)"),
            (-1, IndexError, r"Branch index -1 is out of range \[0, 7\)"),
        )
        for root, error, message in cases:
            expr = SubtreeRegion(root)
            with self.subTest(root=root, step="select"), self.assertRaisesRegex(error, message):
                _tree_b().select(expr)
            with self.subTest(root=root, step="paint"), self.assertRaisesRegex(error, message):
                braincell.Cell(_tree_b(), cv_policy=braincell.CVPerBranch()).paint(expr, _leak())

    def test_a_cell_paints_exactly_the_subtree(self) -> None:
        def painted(region):
            cell = braincell.Cell(make_tree_t(), cv_policy=braincell.CVPerBranch())
            cell.paint(region, braincell.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-70.0 * u.mV, name="leak"))
            cell.init_state()
            names = ("soma", "dend", "d1", "d2", "axon")
            return len(cell.channels), {n: cell.on(branch_in("name", [n])).channels.names for n in names}

        count, per_branch = painted(SubtreeRegion("dend"))
        self.assertEqual(count, 3)
        self.assertEqual(
            per_branch,
            {"soma": (), "dend": ("leak",), "d1": ("leak",), "d2": ("leak",), "axon": ()},
        )
        self.assertEqual((count, per_branch), painted(branch_in("name", ["dend", "d1", "d2"])))


def _um(value):
    return None if value is None else value * u.um


class RadiusRangeRegionTest(unittest.TestCase):
    """MORPH-8a: the cable whose local radius lies in a range, cut where a taper crosses a bound."""

    def assertIntervalsClose(self, actual, expected) -> None:
        self.assertEqual([b for b, _, _ in actual], [b for b, _, _ in expected], actual)
        np.testing.assert_allclose(
            [(p, d) for _, p, d in actual], [(p, d) for _, p, d in expected], rtol=0.0, atol=1e-12
        )

    def test_intervals_on_tree_r(self) -> None:
        tree = make_tree_r()
        select = lambda *args, **kwargs: tree.select(RadiusRangeRegion(*args, **kwargs)).intervals  # noqa: E731
        # dend: 1.5 um at 25 of 90 um; both sides of the jump (1 and 0.6 um) are in. d2 is thickest
        # at x = 1. The axon's middle third is 0.8 um.
        self.assertIntervalsClose(
            select(0.55 * u.um, 1.5 * u.um),
            ((1, 5 / 18, 1.0), (2, 0.0, 1 / 6), (3, 5 / 6, 1.0), (4, 1 / 3, 2 / 3)),
        )
        self.assertIntervalsClose(select(minimum=0.6 * u.um), ((0, 0.0, 1.0), (1, 0.0, 1.0), (4, 1 / 3, 2 / 3)))
        # Open bounds drop dend's 0.6 um cylinder, which starts at the jump (x = 5/9).
        self.assertIntervalsClose(
            select(minimum=0.6 * u.um, closed="neither"), ((0, 0.0, 1.0), (1, 0.0, 5 / 9), (4, 1 / 3, 2 / 3))
        )
        self.assertIntervalsClose(select(0.6 * u.um, 0.6 * u.um), ((1, 5 / 9, 1.0),))
        self.assertEqual(select(1 * u.um, 1 * u.um), ())
        self.assertEqual(tree.select(RadiusRangeRegion()), tree.select(AllRegion()))

    def test_matches_brute_force_on_trees_r_and_r2(self) -> None:
        bounds = (
            (None, 0.8),
            (0.55, 1.5),
            (0.6, 0.6),
            (0.6, None),
            (1.0, 1.0),
            (5.0, None),
            (0.31, 0.59),
            (0.45, 0.65),
            (0.8, 0.8),
            (None, None),
        )
        samples = 200001
        xs = np.linspace(0.0, 1.0, samples)
        for second_soma in (False, True):
            tree = make_tree_r(second_soma)
            radii = {index: radius_um(tree, index, xs) for index in range(len(tree.branches))}
            crossings = 0
            for (lo, hi), closed in itertools.product(bounds, ("both", "left", "right", "neither")):
                with self.subTest(second_soma=second_soma, lo=lo, hi=hi, closed=closed):
                    mask = tree.select(RadiusRangeRegion(_um(lo), _um(hi), closed))
                    self.assertEqual(
                        brute_force_mismatches(
                            tree, mask, radius_um, lo, hi, closed, samples=samples, metric_values=radii
                        ),
                        [],
                    )
                    checked, problems = crossing_errors(tree, mask, radius_um, lo, hi)
                    self.assertEqual(problems, [])
                    crossings += checked
            self.assertGreater(crossings, 0)

    def test_matches_brute_force_on_fixture_morphologies(self) -> None:
        rng = np.random.default_rng(0)
        xs = np.linspace(0.0, 1.0, 1001)
        for name, tree in fixture_morphologies():
            radii = {index: radius_um(tree, index, xs) for index in range(len(tree.branches))}
            values = np.concatenate(list(radii.values()))
            for _ in range(4):
                lo, hi = sorted(rng.choice(values[values > 0], 2, replace=False))
                with self.subTest(name, lo=lo, hi=hi):
                    mask = tree.select(RadiusRangeRegion(lo * u.um, hi * u.um))
                    self.assertEqual(
                        brute_force_mismatches(
                            tree, mask, radius_um, lo, hi, "both", samples=1001, metric_values=radii
                        ),
                        [],
                    )

    def test_the_brute_force_checks_catch_a_wrong_mask(self) -> None:
        # Without this, a helper that never reports would make the two tests above vacuous.
        tree = make_tree_r()
        right = tree.select(RadiusRangeRegion(0.55 * u.um, 1.5 * u.um))
        wrong = RegionMask(((1, 0.2, 1.0),) + right.intervals[1:])
        self.assertEqual(brute_force_mismatches(tree, right, radius_um, 0.55, 1.5, "both", samples=1001), [])
        problems = brute_force_mismatches(tree, wrong, radius_um, 0.55, 1.5, "both", samples=1001)
        self.assertEqual(len(problems), 1)
        self.assertIn("branch 1 (dend)", problems[0])
        checked, problems = crossing_errors(tree, wrong, radius_um, 0.55, 1.5)
        self.assertEqual(checked, 3)  # dend 0.2, d1 1/6, d2 5/6; the axon ends sit on its jumps
        self.assertEqual(len(problems), 1)
        self.assertIn("end 0.2 has f =", problems[0])
        # An end at a bound but displaced along a steep taper fails the bisection check.
        nudged = RegionMask(((1, 5 / 18 + 1e-11, 1.0),) + right.intervals[1:])
        checked, problems = crossing_errors(tree, nudged, radius_um, 0.55, 1.5)
        self.assertEqual(len(problems), 1)
        self.assertIn("from the root", problems[0])

    def test_a_constant_value_is_judged_by_closed(self) -> None:
        for closed, at_minimum, at_maximum in (
            ("both", True, True),
            ("left", True, False),
            ("right", False, True),
            ("neither", False, False),
        ):
            region = RadiusRangeRegion(1 * u.um, 2 * u.um, closed)
            with self.subTest(closed=closed):
                self.assertEqual(region._accepts(1.0), at_minimum)
                self.assertEqual(region._accepts(2.0), at_maximum)
                self.assertTrue(region._accepts(1.5))
                self.assertFalse(region._accepts(2.5))

    def test_needs_no_3d_points(self) -> None:
        tree = make_tree_u()
        self.assertFalse(tree.has_full_point_geometry)
        region = RadiusRangeRegion(maximum=3 * u.um)
        self.assertEqual(tree.select(region).intervals, ((1, 0.0, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0)))
        cell = braincell.Cell(tree, cv_policy=braincell.CVPerBranch())
        cell.paint(region, braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70 * u.mV))
        cell.init_state()
        self.assertEqual(len(cell.channels), 3)

    def test_construction_checks_the_bounds(self) -> None:
        for args, message in (
            ((1.0, 2.0), "RadiusRangeRegion.minimum must satisfy unit um"),
            ((None, 2.0), "RadiusRangeRegion.maximum must satisfy unit um"),
            ((1 * u.ms, 2 * u.ms), "must satisfy unit um"),
        ):
            with self.subTest(args=args), self.assertRaisesRegex(TypeError, message):
                RadiusRangeRegion(*args)
        for kwargs, message in (
            ({"minimum": 2 * u.um, "maximum": 1 * u.um}, "minimum must be <= maximum"),
            ({"minimum": -1 * u.um, "maximum": 2 * u.um}, "minimum must be >= 0"),
            ({"minimum": np.nan * u.um, "maximum": 2 * u.um}, "minimum must be finite"),
            ({"maximum": np.inf * u.um}, "maximum must be finite; use None for no bound"),
            ({"minimum": np.array([1.0, 2.0]) * u.um}, "RadiusRangeRegion.minimum"),
            ({"closed": "open"}, r"closed must be one of \['both', 'left', 'neither', 'right'\], got 'open'"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                RadiusRangeRegion(**kwargs)
        # Either bound alone, by keyword.
        self.assertIsNone(RadiusRangeRegion(maximum=2 * u.um).minimum)
        self.assertIsNone(RadiusRangeRegion(minimum=2 * u.um).maximum)

    def test_value_semantics(self) -> None:
        region = RadiusRangeRegion(maximum=1.5 * u.um)
        self.assertEqual(region, RadiusRangeRegion(maximum=0.0015 * u.mmeter))
        self.assertEqual(hash(region), hash(RadiusRangeRegion(maximum=0.0015 * u.mmeter)))
        self.assertNotEqual(region, RadiusRangeRegion(maximum=1.5 * u.um, closed="neither"))
        self.assertNotEqual(region, TreeDistanceRegion(maximum=1.5 * u.um))
        self.assertEqual(pickle.loads(pickle.dumps(region)), region)
        self.assertEqual(copy.deepcopy(region), region)
        self.assertIn('Quantity(1.5, "um")', repr(region))

    def test_rigid_transforms_keep_the_region_and_scaling_changes_it(self) -> None:
        region = RadiusRangeRegion(0.55 * u.um, 1.5 * u.um)
        before = make_tree_r().select(region).intervals
        for label, edit in (
            ("translate", lambda tree: tree.translate(np.array([5.0, -3.0, 2.0]) * u.um)),
            ("rotate", lambda tree: tree.rotate(40 * u.degree, axis=np.array([1.0, 2.0, 3.0]))),
        ):
            with self.subTest(label):
                tree = make_tree_r()
                edit(tree)
                self.assertIntervalsClose(tree.select(region).intervals, before)
        tree = make_tree_r()
        tree.scale(2.0)
        self.assertNotEqual(tree.select(region).intervals, before)


class RegionModuleAllTest(unittest.TestCase):
    def test_region_module_declares_all(self) -> None:
        self.assertIn("RegionSetOp", region_mod.__all__)
        self.assertIn("BranchSlice", region_mod.__all__)
        self.assertIn("branch_range", region_mod.__all__)


if __name__ == "__main__":
    unittest.main()
