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


import copy
import dataclasses
import pickle
import unittest

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell import Branch, Morphology
from braincell._random_test import default_key_data, drawn_seeds
from braincell.filter._testing import make_soma, make_soma_dend_tree
from braincell.morph._testing import make_tree_t
from braincell.filter import (
    AllRegion,
    AtLocation,
    BranchPoints,
    BranchSlice,
    EmptyRegion,
    ForkPoints,
    LocsetConcatOp,
    LocsetBatch,
    LocsetMask,
    LocsetSetOp,
    LocsetUniqueOp,
    RandomSamples,
    RegionAnchors,
    RootLocation,
    SelectionCache,
    StepSamples,
    Terminals,
    UniformSamples,
    at,
    branch_in,
    sample,
)
from braincell.filter import helper
from braincell.filter import locset as locset_mod


def _build_branchpoint_morphology() -> Morphology:
    soma = make_soma()
    basal = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    axon = Branch.from_lengths(lengths=[10.0] * u.um, radii=[0.8, 0.5] * u.um, type="axon")
    apical = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.5, 1.0] * u.um, type="apical_dendrite")
    tuft = Branch.from_lengths(lengths=[15.0] * u.um, radii=[0.9, 0.7] * u.um, type="apical_dendrite")

    tree = Morphology.from_root(soma, name="soma")
    tree.soma.attach(basal, name="basal", parent_x=1.0)
    tree.soma.attach(axon, name="axon", parent_x=0.5)
    tree.soma.attach(apical, name="apical", parent_x=0.5)
    tree.soma.basal.tuft = tuft
    return tree


def _build_sampling_tree() -> Morphology:
    return make_soma_dend_tree(soma_length=10.0)


def _branch(*, branch_type: str = "dendrite") -> Branch:
    return Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type=branch_type)


def _point_is_in_region(
    point: tuple[int, float],
    intervals: tuple[tuple[int, float, float], ...],
    *,
    epsilon: float = 1e-12,
) -> bool:
    branch, x = point
    for interval_branch, prox, dist in intervals:
        if interval_branch != branch:
            continue
        if (x + epsilon) >= prox and (x - epsilon) <= dist:
            return True
    return False


class BasicLocsetTest(unittest.TestCase):
    def test_at_location_accepts_branch_index_and_name(self) -> None:
        tree = _build_branchpoint_morphology()

        by_index = at(0, 0.5).evaluate(tree)
        by_name = at("basal", 0.25).evaluate(tree)
        explicit = AtLocation(branch="axon", x=1.0).evaluate(tree)

        self.assertEqual(by_index.points, ((0, 0.5),))
        self.assertEqual(by_index.display_names, ("soma(0.5)",))
        self.assertEqual(by_name.points, ((1, 0.25),))
        self.assertEqual(by_name.display_names, ("basal(0.25)",))
        self.assertEqual(explicit.points, ((2, 1.0),))
        self.assertEqual(explicit.display_names, ("axon(1)",))

    def test_at_location_rejects_invalid_branch_reference_and_x(self) -> None:
        tree = _build_branchpoint_morphology()

        with self.assertRaises(KeyError):
            at("missing", 0.5).evaluate(tree)
        with self.assertRaises(IndexError):
            at(99, 0.5).evaluate(tree)
        with self.assertRaises(TypeError):
            at(1.2, 0.5).evaluate(tree)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            at(0, -0.1).evaluate(tree)
        with self.assertRaises(ValueError):
            at(0, 1.1).evaluate(tree)

    def test_fork_points_returns_topology_junctions_with_three_branches(self) -> None:
        tree = _build_branchpoint_morphology()

        locset = ForkPoints().evaluate(tree)

        self.assertEqual(locset.points, ((0, 0.5),))
        self.assertEqual(locset.display_names, ("soma(0.5)",))
        self.assertIs(BranchPoints, ForkPoints)

    def test_terminals_returns_leaf_branch_distal_points(self) -> None:
        tree = _build_branchpoint_morphology()

        locset = Terminals().evaluate(tree)

        self.assertEqual(locset.points, ((2, 1.0), (3, 1.0), (4, 1.0)))

    def test_locset_boolean_ops_union_intersection_difference(self) -> None:
        tree = _build_branchpoint_morphology()

        root_mid = RootLocation(x=0.5)
        terminals = Terminals()
        branch_points = BranchPoints()

        union_expr = root_mid | terminals
        inter_expr = branch_points & root_mid
        diff_expr = union_expr - root_mid

        union_locset = union_expr.evaluate(tree)
        inter_locset = inter_expr.evaluate(tree)
        diff_locset = diff_expr.evaluate(tree)

        self.assertEqual(union_locset.points, ((0, 0.5), (2, 1.0), (3, 1.0), (4, 1.0)))
        self.assertEqual(inter_locset.points, ((0, 0.5),))
        self.assertEqual(diff_locset.points, ((2, 1.0), (3, 1.0), (4, 1.0)))

    def test_concat_preserves_order_and_duplicate_rows(self) -> None:
        tree = _build_sampling_tree()
        expr = at("dend", 0.25) + at("soma", 0.5) + at("dend", 0.25)

        locset = expr.evaluate(tree)

        self.assertEqual(locset.points, ((1, 0.25), (0, 0.5), (1, 0.25)))
        self.assertEqual(locset.display_names, ("dend(0.25)", "soma(0.5)", "dend(0.25)"))

    def test_set_ops_are_stable_and_unique(self) -> None:
        tree = _build_sampling_tree()
        left = at("dend", 0.8) + at("soma", 0.5) + at("dend", 0.8)
        right = at("soma", 0.5) + at("dend", 0.2) + at("soma", 0.5)

        self.assertEqual((left | right).evaluate(tree).points, ((1, 0.8), (0, 0.5), (1, 0.2)))
        self.assertEqual((left & right).evaluate(tree).points, ((0, 0.5),))
        self.assertEqual((left - right).evaluate(tree).points, ((1, 0.8),))
        self.assertEqual(left.unique().evaluate(tree).points, ((1, 0.8), (0, 0.5)))


class LocsetMaskTest(unittest.TestCase):
    def test_column_constructor_copies_and_freezes_storage(self) -> None:
        branch_id = np.asarray([2, 1, 2], dtype=np.int32)
        branch_x = np.asarray([0.3, 0.2, 0.3], dtype=np.float32)

        mask = LocsetMask.from_columns(branch_id, branch_x, ("a", "b", "c"))
        branch_id[0] = 99
        branch_x[0] = 0.9

        np.testing.assert_array_equal(mask.branch_id, np.asarray([2, 1, 2]))
        np.testing.assert_allclose(mask.branch_x, np.asarray([0.3, 0.2, 0.3]), atol=1e-7)
        self.assertFalse(mask.branch_id.flags.writeable)
        self.assertFalse(mask.branch_x.flags.writeable)
        with self.assertRaises(ValueError):
            mask.branch_id[0] = 3

    def test_compatibility_rows_optional_names_and_immediate_unique(self) -> None:
        mask = LocsetMask(points=((2, 0.3), (1, 0.2), (2, 0.3)))
        named = LocsetMask(
            points=((2, 0.3), (1, 0.2), (2, 0.3)),
            display_names=("first", "middle", "last"),
        )

        self.assertEqual(mask.points, ((2, 0.3), (1, 0.2), (2, 0.3)))
        self.assertIsNone(mask.display_names)
        self.assertEqual(named.unique().points, ((2, 0.3), (1, 0.2)))
        self.assertEqual(named.unique().display_names, ("first", "middle"))
        self.assertEqual(named, LocsetMask(named.points, named.display_names))
        self.assertEqual(hash(named), hash(LocsetMask(named.points, named.display_names)))
        self.assertEqual(pickle.loads(pickle.dumps(named)), named)

    def test_column_constructor_validates_shape_dtype_coordinate_and_names(self) -> None:
        with self.assertRaises(ValueError):
            LocsetMask.from_columns([[0]], [0.5])
        with self.assertRaises(ValueError):
            LocsetMask.from_columns([0, 1], [0.5])
        with self.assertRaises(TypeError):
            LocsetMask.from_columns([0.0], [0.5])
        with self.assertRaises(ValueError):
            LocsetMask.from_columns([-1], [0.5])
        with self.assertRaises(TypeError):
            LocsetMask.from_columns([0], ["0.5"])
        with self.assertRaises(ValueError):
            LocsetMask.from_columns([0], [np.nan])
        with self.assertRaises(ValueError):
            LocsetMask.from_columns([0], [1.1])
        with self.assertRaises(ValueError):
            LocsetMask(points=((0, 0.5),), display_names=())
        with self.assertRaises(TypeError):
            LocsetMask(points=((0, True),))
        with self.assertRaises(ValueError):
            LocsetMask(points=((-1, 0.5),))

    def test_one_dimensional_indexing_returns_locset_mask(self) -> None:
        mask = LocsetMask(
            points=((0, 0.1), (1, 0.2), (2, 0.3)),
            display_names=("a", "b", "c"),
        )

        self.assertEqual(mask[1].points, ((1, 0.2),))
        self.assertEqual(mask[[2, 0, 2]].points, ((2, 0.3), (0, 0.1), (2, 0.3)))
        self.assertEqual(mask[np.asarray([True, False, True])].display_names, ("a", "c"))

    def test_two_dimensional_indexing_returns_locset_batch(self) -> None:
        mask = LocsetMask(
            points=((0, 0.1), (1, 0.2), (2, 0.3)),
            display_names=("a", "b", "c"),
        )

        batch = mask[np.asarray([[0, 1], [2, 0], [1, 1]])]

        self.assertIsInstance(batch, LocsetBatch)
        self.assertEqual(batch.shape, (3, 2))
        self.assertEqual(batch[0].points, ((0, 0.1), (1, 0.2)))
        self.assertEqual(batch[1:].shape, (2, 2))
        self.assertEqual(batch.display_names, (("a", "b"), ("c", "a"), ("b", "b")))


class LocsetBatchTest(unittest.TestCase):
    def test_column_constructor_copies_freezes_and_validates_storage(self) -> None:
        branch_id = np.asarray([[0, 1], [2, 0]], dtype=np.int32)
        branch_x = np.asarray([[0.1, 0.2], [0.3, 0.1]], dtype=np.float32)

        batch = LocsetBatch.from_columns(branch_id, branch_x, (("a", "b"), ("c", "a")))
        branch_id[0, 0] = 99
        branch_x[0, 0] = 0.9

        np.testing.assert_array_equal(batch.branch_id, np.asarray([[0, 1], [2, 0]]))
        np.testing.assert_allclose(batch.branch_x, np.asarray([[0.1, 0.2], [0.3, 0.1]]), atol=1e-7)
        self.assertFalse(batch.branch_id.flags.writeable)
        self.assertFalse(batch.branch_x.flags.writeable)
        with self.assertRaises(ValueError):
            LocsetBatch.from_columns([0, 1], [0.1, 0.2])
        with self.assertRaises(ValueError):
            LocsetBatch.from_columns([[0, 1]], [[0.1]])
        with self.assertRaises(ValueError):
            LocsetBatch.from_columns([[0]], [[1.1]])


class ForkPointsTest(unittest.TestCase):
    def test_multiple_forks_keep_first_attachment_order(self) -> None:
        tree = Morphology.from_root(_branch(branch_type="soma"), name="soma")
        tree.soma.attach(_branch(), name="middle_a", parent_x=0.5)
        tree.soma.attach(_branch(), name="distal_a", parent_x=1.0)
        tree.soma.attach(_branch(), name="middle_b", parent_x=0.5)
        tree.soma.attach(_branch(), name="distal_b", parent_x=1.0)

        self.assertEqual(ForkPoints().evaluate(tree).points, ((0, 0.5), (0, 1.0)))

    def test_transitive_attachment_equivalence_forms_one_fork(self) -> None:
        tree = Morphology.from_root(_branch(branch_type="soma"), name="soma")
        tree.soma.attach(_branch(), name="a", parent_x=1.0, child_x=0.0)
        tree.soma.a.attach(_branch(), name="b", parent_x=0.0, child_x=0.0)

        self.assertEqual(ForkPoints().evaluate(tree).points, ((0, 1.0),))

    def test_single_soma_midpoint_child_is_not_a_fork(self) -> None:
        tree = Morphology.from_root(_branch(branch_type="soma"), name="soma")
        tree.soma.attach(_branch(), name="dend", parent_x=0.5)

        self.assertEqual(ForkPoints().evaluate(tree).points, ())

    def test_coincident_geometry_does_not_create_topology_fork(self) -> None:
        soma = Branch.from_points(
            points=np.asarray([[0, 0, 0], [10, 0, 0], [0, 0, 0]]) * u.um,
            radii=[2.0, 2.0, 2.0] * u.um,
            type="soma",
        )
        child = Branch.from_points(
            points=np.asarray([[0, 0, 0], [0, 10, 0]]) * u.um,
            radii=[1.0, 1.0] * u.um,
            type="dendrite",
        )
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.attach(child, name="left", parent_x=0.0)
        tree.soma.attach(child, name="right", parent_x=1.0)

        self.assertEqual(ForkPoints().evaluate(tree).points, ())


class RegionSamplingLocsetTest(unittest.TestCase):
    def test_uniform_samples_are_globally_length_uniform(self) -> None:
        tree = _build_sampling_tree()
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)

        locset = UniformSamples(region=region, count=4).evaluate(tree)

        self.assertEqual(len(locset.points), 4)
        self.assertEqual(locset.points[0], (0, 0.5))
        self.assertAlmostEqual(locset.points[1][1], 1.0 / 6.0, places=12)
        self.assertAlmostEqual(locset.points[2][1], 0.5, places=12)
        self.assertAlmostEqual(locset.points[3][1], 5.0 / 6.0, places=12)

    def test_random_samples_use_seed_and_stay_in_region(self) -> None:
        tree = _build_sampling_tree()
        region = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        intervals = region.evaluate(tree).intervals

        first = RandomSamples(region=region, count=30, seed=123).evaluate(tree)
        second = RandomSamples(region=region, count=30, seed=123).evaluate(tree)
        third = RandomSamples(region=region, count=30, seed=124).evaluate(tree)

        self.assertEqual(first.points, second.points)
        self.assertNotEqual(first.points, third.points)
        for point in first.points:
            self.assertTrue(_point_is_in_region(point, intervals))

    def test_sampling_rejects_invalid_parameters(self) -> None:
        tree = _build_sampling_tree()
        region = BranchSlice(branch_index=0, prox=0.0, dist=1.0)

        with self.assertRaises(ValueError):
            UniformSamples(region=region, count=0).evaluate(tree)
        with self.assertRaises(TypeError):
            UniformSamples(region=RootLocation(x=0.5), count=1).evaluate(tree)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RandomSamples(region=region, count=1, seed="bad").evaluate(tree)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            UniformSamples(region=EmptyRegion(), count=1).evaluate(tree)
        with self.assertRaises(ValueError):
            RandomSamples(region=EmptyRegion(), count=1, seed=1).evaluate(tree)


def _tree_d() -> Morphology:
    """Soma (20 um, index 0) and dend (40 + 60 um, index 1) on the soma's distal end; T = 120 um."""
    soma = Branch.from_lengths(lengths=np.array([20.0]) * u.um, radii=np.array([5.0, 5.0]) * u.um, type="soma")
    dend = Branch.from_lengths(
        lengths=np.array([40.0, 60.0]) * u.um, radii=np.array([2.0, 1.5, 1.0]) * u.um, type="basal_dendrite"
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend")
    return tree


class RegionAnchorsTest(unittest.TestCase):
    """MORPH-10a: one location per normalized interval, at a relative position inside it."""

    def setUp(self) -> None:
        self.tree = _tree_d()

    def select(self, expr):
        mask = self.tree.select(expr)
        return mask.points, mask.display_names

    def test_one_location_per_interval_at_the_relative_position(self) -> None:
        for x, label in ((0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (1.0, "1")):
            with self.subTest(x=x):
                self.assertEqual(
                    self.select(RegionAnchors(AllRegion(), x)),
                    (((0, x), (1, x)), (f"soma({label})", f"dend({label})")),
                )

    def test_rows_follow_the_normalized_intervals(self) -> None:
        expected = (((0, 0.75), (1, 0.4)), ("soma(0.75)", "dend(0.4)"))
        self.assertEqual(self.select(RegionAnchors(BranchSlice(1, 0.2, 0.6) | BranchSlice(0, 0.5, 1.0), 0.5)), expected)
        # A multi-branch leaf slice keeps its input order until normalized.
        leaf = BranchSlice([1, 0], [0.2, 0.5], [0.6, 1.0])
        self.assertEqual(self.tree.select(leaf).intervals, ((1, 0.2, 0.6), (0, 0.5, 1.0)))
        self.assertEqual(self.select(RegionAnchors(leaf, 0.5)), expected)
        # Touching intervals on one branch merge into one, so one location.
        self.assertEqual(
            self.select(RegionAnchors(BranchSlice(1, 0.0, 0.5) | BranchSlice(1, 0.5, 1.0), 0.5)),
            (((1, 0.5),), ("dend(0.5)",)),
        )

    def test_concatenation_keeps_order_and_junction_rows(self) -> None:
        self.assertEqual(
            self.tree.select(RegionAnchors(AllRegion(), 1.0) + RegionAnchors(AllRegion(), 0.0)).points,
            ((0, 1.0), (1, 1.0), (0, 0.0), (1, 0.0)),
        )
        # (soma, 1) and (dend, 0) are one physical point; unique() keeps both rows.
        both_ends = self.tree.select((RegionAnchors(AllRegion(), 1.0) + RegionAnchors(AllRegion(), 0.0)).unique())
        self.assertEqual(both_ends.points, ((0, 1.0), (1, 1.0), (0, 0.0), (1, 0.0)))

    def test_one_branch_region_equals_at(self) -> None:
        self.assertEqual(
            self.tree.select(RegionAnchors(branch_in("name", "dend"), 0.3)),
            self.tree.select(at("dend", 0.3)),
        )

    def test_x_within_the_tolerance_is_clipped(self) -> None:
        self.assertEqual(self.tree.select(RegionAnchors(AllRegion(), 1 + 1e-13)).points, ((0, 1.0), (1, 1.0)))
        self.assertEqual(self.tree.select(RegionAnchors(AllRegion(), -1e-13)).points, ((0, 0.0), (1, 0.0)))
        self.assertEqual(self.tree.select(RegionAnchors(AllRegion(), np.float64(0.5))).points, ((0, 0.5), (1, 0.5)))
        self.assertEqual(self.tree.select(RegionAnchors(AllRegion(), 1)).points, ((0, 1.0), (1, 1.0)))

    def test_construction_rejects_bad_arguments(self) -> None:
        for x in (1.5, -0.1, float("nan"), float("inf")):
            with self.subTest(x=x), self.assertRaisesRegex(ValueError, r"RegionAnchors\.x must be within \[0, 1\]"):
                RegionAnchors(AllRegion(), x)
        for x, message in (
            (True, "real number"),
            (None, "real number"),
            ("0.5", "real number"),
            (0.5 * u.um, "not a length"),
        ):
            with self.subTest(x=x), self.assertRaisesRegex(TypeError, message):
                RegionAnchors(AllRegion(), x)
        with self.assertRaisesRegex(TypeError, "RegionAnchors.region expects RegionExpr, got str"):
            RegionAnchors("dend", 0.5)

    def test_an_empty_region_raises_at_select_and_at_place(self) -> None:
        synapse = braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV)
        for region in (EmptyRegion(), AllRegion() - AllRegion(), BranchSlice(1, 0.3, 0.3 + 1e-13)):
            expr = RegionAnchors(region, 0.5)
            with self.subTest(region=region, step="select"), self.assertRaisesRegex(ValueError, "empty region"):
                self.tree.select(expr)
            cell = braincell.Cell(_tree_d(), cv_policy=braincell.CVPerBranch())
            with self.subTest(region=region, step="place"), self.assertRaisesRegex(ValueError, "empty region"):
                cell.place(expr, synapse)

    def test_value_semantics(self) -> None:
        expr = RegionAnchors(branch_in("type", "axon"), 0.5)
        self.assertEqual(expr, RegionAnchors(branch_in("type", "axon"), 0.5))
        self.assertEqual(hash(expr), hash(RegionAnchors(branch_in("type", "axon"), 0.5)))
        self.assertNotEqual(expr, RegionAnchors(branch_in("type", "axon"), 0.25))
        self.assertEqual(pickle.loads(pickle.dumps(expr)), expr)


class StepSamplesTest(unittest.TestCase):
    """MORPH-10b: locations a fixed physical step apart, half a step in, continuing across branches."""

    def setUp(self) -> None:
        self.tree = _tree_d()

    def points(self, expr):
        return self.tree.select(expr).points

    def test_locations_on_tree_d(self) -> None:
        self.assertEqual(
            self.points(StepSamples(AllRegion(), 25 * u.um)),
            ((0, 0.625), (1, 0.175), (1, 0.425), (1, 0.675)),
        )
        self.assertEqual(
            self.tree.select(StepSamples(AllRegion(), 25 * u.um)).display_names,
            ("soma(0.625)", "dend(0.175)", "dend(0.425)", "dend(0.675)"),
        )
        thirty = ((0, 0.75), (1, 0.25), (1, 0.55), (1, 0.85))
        self.assertEqual(self.points(StepSamples(AllRegion(), 30 * u.um)), thirty)
        self.assertEqual(self.points(StepSamples(AllRegion(), 0.03 * u.mmeter)), thirty)
        self.assertEqual(self.points(UniformSamples(AllRegion(), 4)), thirty)
        self.assertEqual(
            self.points(StepSamples(BranchSlice(1, 0.5, 1.0), 10 * u.um)),
            ((1, 0.55), (1, 0.65), (1, 0.75), (1, 0.85), (1, 0.95)),
        )
        self.assertEqual(self.points(StepSamples(AllRegion(), 120 * u.um)), ((1, 0.4),))
        self.assertEqual(self.points(StepSamples(AllRegion(), 70 * u.um)), ((1, 0.15),))
        five = self.points(StepSamples(AllRegion(), np.array([5.0]) * u.um))
        self.assertEqual(len(five), 24)
        self.assertEqual(five[:2], ((0, 0.125), (0, 0.375)))
        self.assertEqual(five[-1], (1, 0.975))

    def test_a_step_that_divides_the_length_matches_uniform_samples(self) -> None:
        tree = make_tree_t()
        regions = (
            AllRegion(),
            branch_in("name", ["dend", "d2"]),
            BranchSlice([1, 3, 4], [0.1, 0.0, 0.3], [0.7, 0.4, 1.0]),
        )
        for region in regions:
            entries = helper._sample_entries(tree, tree.select(region).intervals, epsilon=helper.EPSILON)
            total = sum(entry[3] for entry in entries)
            for n in range(1, 201):
                with self.subTest(region=region, n=n):
                    stepped = tree.select(StepSamples(region, (total / n) * u.um))
                    uniform = tree.select(UniformSamples(region, n))
                    self.assertEqual(len(stepped.points), n)
                    np.testing.assert_array_equal(stepped.branch_id, uniform.branch_id)
                    # Locations are stored rounded to 12 decimals; the two may land one unit apart.
                    units = np.abs(np.rint(stepped.branch_x * 1e12) - np.rint(uniform.branch_x * 1e12))
                    self.assertLessEqual(float(units.max()), 1.0)

    def test_a_step_that_divides_the_length_up_to_rounding_keeps_its_last_location(self) -> None:
        soma = Branch.from_lengths(lengths=np.array([0.3]) * u.um, radii=np.array([1.0, 1.0]) * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        self.assertEqual(0.3 / 0.1, 2.9999999999999996)
        self.assertEqual(
            tree.select(StepSamples(AllRegion(), 0.1 * u.um)).points,
            ((0, 0.166666666667), (0, 0.5), (0, 0.833333333333)),
        )

    def test_value_semantics(self) -> None:
        thirty = StepSamples(AllRegion(), 30 * u.um)
        self.assertEqual(thirty, StepSamples(AllRegion(), 0.03 * u.mmeter))
        self.assertEqual(hash(thirty), hash(StepSamples(AllRegion(), 0.03 * u.mmeter)))
        self.assertNotEqual(thirty, StepSamples(AllRegion(), 25 * u.um))
        self.assertNotEqual(thirty, StepSamples(branch_in("name", "dend"), 30 * u.um))
        self.assertNotEqual(thirty, UniformSamples(AllRegion(), 4))
        self.assertEqual(repr(thirty), 'StepSamples(region=AllRegion(), step=Quantity(30, "um"))')
        self.assertEqual(pickle.loads(pickle.dumps(thirty)), thirty)
        self.assertEqual(copy.deepcopy(thirty), thirty)
        # Equal steps in different units are one cache key.
        cache = SelectionCache()
        both = StepSamples(AllRegion(), 25 * u.um) + StepSamples(AllRegion(), 0.025 * u.mmeter)
        self.assertEqual(len(self.tree.select(both, cache=cache).points), 8)
        self.assertEqual(len(cache._masks), 2)

    def test_construction_rejects_bad_steps_and_regions(self) -> None:
        for step, message in (
            (5.0, "must satisfy unit um"),
            (5 * u.ms, "must satisfy unit um"),
            (None, "cannot be None"),
        ):
            with self.subTest(step=step), self.assertRaisesRegex(TypeError, message):
                StepSamples(AllRegion(), step)
        with self.assertRaisesRegex(TypeError, "StepSamples.region expects RegionExpr, got str"):
            StepSamples("dend", 5 * u.um)
        for step, message in (
            (0 * u.um, "must be > 0"),
            (-5 * u.um, "must be > 0"),
            (np.inf * u.um, "must be finite"),
            (np.nan * u.um, "must be finite"),
            (np.array([5.0, 6.0]) * u.um, "StepSamples.step"),
        ):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, message):
                StepSamples(AllRegion(), step)

    def test_empty_or_too_short_regions_raise_at_select_and_at_place(self) -> None:
        synapse = braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV)
        cases = (
            (StepSamples(EmptyRegion(), 5 * u.um), r"StepSamples cannot sample from an empty or zero-length region"),
            (StepSamples(BranchSlice(1, 0.3, 0.3 + 1e-13), 5 * u.um), "empty or zero-length region"),
            (
                StepSamples(AllRegion(), 1000 * u.um),
                r"1000\.0 um exceeds the region's total length 120\.0 um.*UniformSamples\(region, 1\)",
            ),
        )
        for expr, message in cases:
            with self.subTest(expr=expr, step="select"), self.assertRaisesRegex(ValueError, message):
                self.tree.select(expr)
            cell = braincell.Cell(_tree_d(), cv_policy=braincell.CVPerBranch())
            with self.subTest(expr=expr, step="place"), self.assertRaisesRegex(ValueError, message):
                cell.place(expr, synapse)


class LocsetSetOpValidationTest(unittest.TestCase):
    def test_invalid_operator_and_arity_raise(self) -> None:
        tree = _build_sampling_tree()
        left = RootLocation(x=0.2)
        right = RootLocation(x=0.8)

        with self.assertRaises(ValueError):
            LocsetSetOp("invalid", (left, right)).evaluate(tree)
        with self.assertRaises(ValueError):
            LocsetSetOp("union", (left,)).evaluate(tree)
        with self.assertRaises(ValueError):
            LocsetConcatOp((left,)).evaluate(tree)

    def test_unique_expression_type_is_public(self) -> None:
        self.assertIsInstance(RootLocation(x=0.5).unique(), LocsetUniqueOp)


#: The two random-location expressions, built without a seed and with one.
_SAMPLERS = (
    ("sample", lambda region, seed=None: sample(region, number=3, seed=seed)),
    ("RandomSamples", lambda region, seed=None: RandomSamples(region, 3, seed)),
)


class ContextSeedSamplingTest(unittest.TestCase):
    """CELL-11a: ``sample`` and ``RandomSamples`` without a seed draw one when built."""

    def test_the_seed_is_drawn_at_construction_by_the_definition(self) -> None:
        # Acceptance 1 and 3. On main ``sample`` without a seed raised at the
        # call, ``RandomSamples`` without one at construction, and both with
        # ``seed=None`` at evaluate.
        tree = _build_sampling_tree()
        with brainstate.random.seed_context(42):
            built = [make(AllRegion()) for _, make in _SAMPLERS]
            direct = locset_mod.SampleLocations(AllRegion(), 3, None)
        self.assertEqual([expr.seed for expr in (*built, direct)], drawn_seeds(42, 3))
        for (name, make), expr in zip(_SAMPLERS, built):
            with self.subTest(sampler=name):
                self.assertIs(type(expr.seed), int)
                points = expr.evaluate(tree).points
                self.assertEqual(expr.evaluate(tree).points, points)
                self.assertEqual(expr.evaluate(tree, SelectionCache()).points, points)
                self.assertEqual(make(AllRegion(), expr.seed).evaluate(tree).points, points)

    def test_a_context_seed_reproduces_and_changes_the_locations(self) -> None:
        tree = _build_sampling_tree()
        for name, make in _SAMPLERS:
            with self.subTest(sampler=name):
                runs = []
                for seed in (42, 42, 43):
                    with brainstate.random.seed_context(seed):
                        runs.append(make(AllRegion()).evaluate(tree).points)
                self.assertEqual(runs[0], runs[1])
                self.assertNotEqual(runs[0], runs[2])

    def test_copies_keep_the_drawn_seed_without_drawing_again(self) -> None:
        with brainstate.random.seed_context(7):
            built = [make(AllRegion()) for _, make in _SAMPLERS]
            before = default_key_data()
            copies = [
                (copy.copy(expr), copy.deepcopy(expr), pickle.loads(pickle.dumps(expr)), dataclasses.replace(expr))
                for expr in built
            ]
            np.testing.assert_array_equal(default_key_data(), before)
        for expr, made in zip(built, copies):
            for duplicate in made:
                self.assertEqual(duplicate.seed, expr.seed)

    def test_explicit_seeds_ignore_the_context(self) -> None:
        # Acceptance 7.
        tree = _build_sampling_tree()
        for name, make, seed in (("sample", _SAMPLERS[0][1], 3), ("RandomSamples", _SAMPLERS[1][1], 123)):
            with self.subTest(sampler=name):
                with brainstate.random.seed_context(1):
                    one = make(AllRegion(), seed)
                with brainstate.random.seed_context(2):
                    two = make(AllRegion(), seed)
                self.assertEqual(one.seed, seed)
                self.assertEqual(two.evaluate(tree).points, one.evaluate(tree).points)

    def test_a_bool_seed_still_raises_at_evaluate(self) -> None:
        # Acceptance 6; main raised the same TypeError.
        tree = _build_sampling_tree()
        for name, make in _SAMPLERS:
            with self.subTest(sampler=name), self.assertRaisesRegex(TypeError, "seed must be an integer"):
                make(AllRegion(), True).evaluate(tree)

    def test_seed_none_under_a_transformation_raises_and_leaves_the_default_stream(self) -> None:
        # Acceptance 6.
        for name, make in _SAMPLERS:
            with self.subTest(sampler=name):
                brainstate.random.seed(1)
                expected = float(brainstate.random.uniform())
                brainstate.random.seed(1)
                before = default_key_data()
                with self.assertRaisesRegex(TypeError, rf"{name}\(seed=None\).*traced"):
                    brainstate.transform.jit(lambda: make(AllRegion()).seed)()
                np.testing.assert_array_equal(default_key_data(), before)
                self.assertEqual(float(brainstate.random.uniform()), expected)


class LocsetModuleAllTest(unittest.TestCase):
    def test_locset_module_declares_all(self) -> None:
        self.assertIn("LocsetMask", locset_mod.__all__)
        self.assertIn("LocsetExpr", locset_mod.__all__)
        self.assertIn("AtLocation", locset_mod.__all__)
        self.assertIn("RootLocation", locset_mod.__all__)
        self.assertIn("RandomSamples", locset_mod.__all__)
        self.assertIn("LocsetSetOp", locset_mod.__all__)
        self.assertIn("StepSamples", locset_mod.__all__)
        self.assertIn("at", locset_mod.__all__)
