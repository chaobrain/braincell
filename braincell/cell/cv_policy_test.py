from __future__ import annotations

import unittest

from braincell._test_support import u

from braincell import Branch, CVPolicy, Cell, Morpho


def _branch_cv_counts(cell: Cell) -> dict[int, int]:
    counts: dict[int, int] = {}
    for cv in cell.cvs:
        counts[cv.branch_id] = counts.get(cv.branch_id, 0) + 1
    return counts


def _build_three_branch_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    dend_a = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite")
    dend_b = Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.5, 1.0] * u.um, type="apical_dendrite")
    tree = Morpho.from_root(soma, name="soma")
    tree.soma.a = dend_a
    tree.soma.b = dend_b
    return tree


def _build_two_branch_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[100.0] * u.um, radii=[10.0, 8.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[45.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morpho.from_root(soma, name="soma")
    tree.soma.d = dend
    return tree


class CVPolicyTest(unittest.TestCase):
    def test_cv_per_branch_counts_cvs_on_each_branch(self) -> None:
        tree = _build_three_branch_tree()
        cell = Cell(tree, cv_policy=CVPolicy(cv_per_branch=3))
        self.assertEqual(cell.n_cv, 9)
        self.assertEqual(_branch_cv_counts(cell), {0: 3, 1: 3, 2: 3})

    def test_max_cv_len_uses_ceil_per_branch(self) -> None:
        tree = _build_two_branch_tree()
        cell = Cell(
            tree,
            cv_policy=CVPolicy(mode="max_cv_len", max_cv_len=20.0 * u.um),
        )
        self.assertEqual(cell.n_cv, 8)
        self.assertEqual(_branch_cv_counts(cell), {0: 5, 1: 3})

    def test_max_cv_len_bounds_each_cv_length(self) -> None:
        tree = _build_two_branch_tree()
        max_len = 12.5 * u.um
        cell = Cell(
            tree,
            cv_policy=CVPolicy(mode="max_cv_len", max_cv_len=max_len),
        )

        max_len_um = float(max_len.to_decimal(u.um))
        for cv in cell.cvs:
            self.assertLessEqual(float(cv.length.to_decimal(u.um)), max_len_um + 1e-9)

    def test_max_cv_len_preserves_cross_branch_topology(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.d = dend
        cell = Cell(tree, cv_policy=CVPolicy(mode="max_cv_len", max_cv_len=5.0 * u.um))

        self.assertEqual(cell.n_cv, 4)
        self.assertEqual(cell.cv(2).parent_cv, 1)
        self.assertIn(2, cell.cv(1).children_cv)
        self.assertEqual(cell.cv(3).parent_cv, 2)

    def test_policy_validation_errors(self) -> None:
        tree = _build_two_branch_tree()

        with self.assertRaises(ValueError):
            Cell(tree, cv_policy=CVPolicy(mode="unsupported"))

        with self.assertRaises(TypeError):
            Cell(tree, cv_policy=CVPolicy(mode="cv_per_branch", cv_per_branch=True))
        with self.assertRaises(ValueError):
            Cell(tree, cv_policy=CVPolicy(mode="cv_per_branch", cv_per_branch=0))

        with self.assertRaises(TypeError):
            Cell(tree, cv_policy=CVPolicy(mode="max_cv_len"))
        with self.assertRaises(TypeError):
            Cell(tree, cv_policy=CVPolicy(mode="max_cv_len", max_cv_len=20.0))
        with self.assertRaises(ValueError):
            Cell(tree, cv_policy=CVPolicy(mode="max_cv_len", max_cv_len=0.0 * u.um))

