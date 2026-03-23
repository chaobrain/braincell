from __future__ import annotations

import unittest

from braincell import Branch, Morpho
from braincell.filter import BranchSlice, RegionSetOp
from braincell.filter import helper as helper_mod
from braincell.filter import locset as locset_mod
from braincell.filter import region as region_mod


def _build_tree() -> Morpho:
    soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
    dend = Branch.lengths_shared(lengths=[80.0], radii=[2.0, 1.0], type="basal_dendrite")
    axon = Branch.lengths_shared(lengths=[120.0], radii=[0.8, 0.5], type="axon")

    tree = Morpho.from_root(soma, name="soma")
    tree.soma.dend = dend
    tree.soma.axon = axon
    return tree


class BranchSliceRegionTest(unittest.TestCase):
    def test_branch_slice_single_interval_remains_compatible(self) -> None:
        tree = _build_tree()

        region = BranchSlice(branch_index=1, prox=0.2, dist=0.8).evaluate(tree)

        self.assertEqual(region.intervals, ((1, 0.2, 0.8),))

    def test_branch_slice_accepts_vector_inputs(self) -> None:
        tree = _build_tree()

        region = BranchSlice(
            branch_index=[0, 2],
            prox=[0.0, 0.3],
            dist=[1.0, 0.9],
        ).evaluate(tree)

        self.assertEqual(region.intervals, ((0, 0.0, 1.0), (2, 0.3, 0.9)))

    def test_branch_slice_broadcasts_shared_prox_dist(self) -> None:
        tree = _build_tree()

        region = BranchSlice(branch_index=[0, 1, 2], prox=0.0, dist=1.0).evaluate(tree)

        self.assertEqual(
            region.intervals,
            ((0, 0.0, 1.0), (1, 0.0, 1.0), (2, 0.0, 1.0)),
        )

    def test_branch_slice_broadcasts_single_branch_index(self) -> None:
        tree = _build_tree()

        region = BranchSlice(branch_index=1, prox=[0.0, 0.2], dist=[0.1, 0.9]).evaluate(tree)

        self.assertEqual(region.intervals, ((1, 0.0, 0.1), (1, 0.2, 0.9)))

    def test_branch_slice_rejects_unbroadcastable_lengths(self) -> None:
        tree = _build_tree()

        with self.assertRaises(ValueError):
            BranchSlice(
                branch_index=[0, 1],
                prox=[0.0, 0.2, 0.4],
                dist=[1.0, 0.9],
            ).evaluate(tree)

    def test_branch_slice_rejects_invalid_interval_bounds(self) -> None:
        tree = _build_tree()

        invalid_intervals = [(-0.1, 0.3), (0.5, 0.5), (0.2, 1.1)]
        for prox, dist in invalid_intervals:
            with self.subTest(prox=prox, dist=dist):
                with self.assertRaises(ValueError):
                    BranchSlice(branch_index=0, prox=prox, dist=dist).evaluate(tree)

    def test_branch_slice_rejects_non_integer_branch_indices(self) -> None:
        tree = _build_tree()

        for branch_index in (1.2, True, "1"):
            with self.subTest(branch_index=branch_index):
                with self.assertRaises(TypeError):
                    BranchSlice(branch_index=branch_index, prox=0.1, dist=0.9).evaluate(tree)

    def test_branch_slice_rejects_out_of_range_branch_indices(self) -> None:
        tree = _build_tree()

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
        tree = _build_tree()
        left = BranchSlice(branch_index=0, prox=0.0, dist=0.6)
        right = BranchSlice(branch_index=0, prox=0.5, dist=1.0)

        union_region = (left | right).evaluate(tree)
        inter_region = (left & right).evaluate(tree)
        diff_region = (left - right).evaluate(tree)

        self._assert_intervals_close(union_region.intervals, ((0, 0.0, 1.0),))
        self._assert_intervals_close(inter_region.intervals, ((0, 0.5, 0.6),))
        self._assert_intervals_close(diff_region.intervals, ((0, 0.0, 0.5),))

    def test_complement_and_double_complement(self) -> None:
        tree = _build_tree()
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
        tree = _build_tree()
        base = BranchSlice(branch_index=0, prox=0.1, dist=0.9)
        other = BranchSlice(branch_index=1, prox=0.2, dist=0.8)

        diff = (other - base).evaluate(tree)
        union = (base | other).evaluate(tree)

        self._assert_intervals_close(diff.intervals, ((1, 0.2, 0.8),))
        self._assert_intervals_close(union.intervals, ((0, 0.1, 0.9), (1, 0.2, 0.8)))

    def test_touching_intervals_merge_with_epsilon(self) -> None:
        tree = _build_tree()
        left = BranchSlice(branch_index=0, prox=0.0, dist=0.5)
        right = BranchSlice(branch_index=0, prox=0.5 + 1e-13, dist=1.0)

        region = (left | right).evaluate(tree)

        self._assert_intervals_close(region.intervals, ((0, 0.0, 1.0),))

    def test_region_setop_rejects_invalid_operator_and_arity(self) -> None:
        tree = _build_tree()
        left = BranchSlice(branch_index=0, prox=0.1, dist=0.9)
        right = BranchSlice(branch_index=1, prox=0.2, dist=0.8)

        with self.assertRaises(ValueError):
            RegionSetOp("invalid", (left, right)).evaluate(tree)
        with self.assertRaises(ValueError):
            RegionSetOp("complement", (left, right)).evaluate(tree)
        with self.assertRaises(ValueError):
            RegionSetOp("union", (left,)).evaluate(tree)


class FilterModuleAllTest(unittest.TestCase):
    def test_region_module_declares_all(self) -> None:
        self.assertIn("RegionSetOp", region_mod.__all__)
        self.assertIn("BranchSlice", region_mod.__all__)
        self.assertIn("branch_range", region_mod.__all__)

    def test_locset_module_declares_all(self) -> None:
        self.assertIn("LocsetMask", locset_mod.__all__)
        self.assertIn("LocsetExpr", locset_mod.__all__)
        self.assertIn("RootLocation", locset_mod.__all__)
        self.assertIn("RandomSamples", locset_mod.__all__)
        self.assertIn("LocsetSetOp", locset_mod.__all__)
        self.assertIn("StepSamples", locset_mod.__all__)

    def test_helper_module_declares_all(self) -> None:
        self.assertIn("branch_slice_intervals", helper_mod.__all__)
        self.assertIn("union_region_intervals", helper_mod.__all__)
        self.assertIn("complement_region_intervals", helper_mod.__all__)
