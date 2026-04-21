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

import brainunit as u

from braincell import (
    Branch,
    CompositeByTypePolicy,
    CVPerBranch,
    CVPolicy,
    CVPolicyByTypeRule,
    CableProperty,
    Cell,
    DLambda,
    MaxCVLen,
    Morphology,
)
from braincell.filter import BranchSlice


def _branch_cv_counts(cell: Cell) -> dict[int, int]:
    counts: dict[int, int] = {}
    for cv in cell.cvs:
        counts[cv.branch_id] = counts.get(cv.branch_id, 0) + 1
    return counts


def _build_three_branch_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    dend_a = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite")
    dend_b = Branch.from_lengths(lengths=[40.0] * u.um, radii=[2.5, 1.0] * u.um, type="apical_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.a = dend_a
    tree.soma.b = dend_b
    return tree


def _build_two_branch_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[100.0] * u.um, radii=[10.0, 8.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[45.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.d = dend
    return tree


def _build_mixed_type_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    axon = Branch.from_lengths(lengths=[80.0] * u.um, radii=[1.0, 0.8] * u.um, type="axon")
    basal = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite")
    apical = Branch.from_lengths(lengths=[45.0] * u.um, radii=[2.5, 1.2] * u.um, type="apical_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.ax = axon
    tree.soma.ba = basal
    tree.soma.ap = apical
    return tree


class CVPolicyTest(unittest.TestCase):
    def test_base_policy_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            CVPolicy()

    def test_cv_per_branch_resolve_bounds_on_policy(self) -> None:
        tree = _build_three_branch_tree()
        policy = CVPerBranch(cv_per_branch=2)
        self.assertEqual(
            policy.resolve_cv_bounds(tree),
            (
                ((0.0, 0.5), (0.5, 1.0)),
                ((0.0, 0.5), (0.5, 1.0)),
                ((0.0, 0.5), (0.5, 1.0)),
            ),
        )

    def test_cv_per_branch_counts_cvs_on_each_branch(self) -> None:
        tree = _build_three_branch_tree()
        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=3))
        self.assertEqual(cell.n_cv, 9)
        self.assertEqual(_branch_cv_counts(cell), {0: 3, 1: 3, 2: 3})

    def test_max_cv_len_resolve_bounds_on_policy(self) -> None:
        tree = _build_two_branch_tree()
        policy = MaxCVLen(max_cv_len=20.0 * u.um)
        self.assertEqual(
            policy.resolve_cv_bounds(tree),
            (
                ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)),
                ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0)),
            ),
        )
        cell = Cell(
            tree,
            cv_policy=policy,
        )
        self.assertEqual(cell.n_cv, 8)
        self.assertEqual(_branch_cv_counts(cell), {0: 5, 1: 3})

    def test_max_cv_len_promotes_even_count_to_odd_by_default(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        policy = MaxCVLen(max_cv_len=5.0 * u.um)
        self.assertEqual(
            policy.resolve_cv_bounds(tree),
            (((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0)),),
        )
        cell = Cell(tree, cv_policy=policy)
        self.assertEqual(cell.n_cv, 3)

    def test_max_cv_len_can_disable_keep_odd(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        policy = MaxCVLen(max_cv_len=5.0 * u.um, keep_odd=False)
        self.assertEqual(
            policy.resolve_cv_bounds(tree),
            (((0.0, 0.5), (0.5, 1.0)),),
        )
        cell = Cell(tree, cv_policy=policy)
        self.assertEqual(cell.n_cv, 2)

    def test_max_cv_len_bounds_each_cv_length(self) -> None:
        tree = _build_two_branch_tree()
        max_len = 12.5 * u.um
        cell = Cell(tree, cv_policy=MaxCVLen(max_cv_len=max_len, keep_odd=False))

        max_len_um = float(max_len.to_decimal(u.um))
        for cv in cell.cvs:
            self.assertLessEqual(float(cv.length.to_decimal(u.um)), max_len_um + 1e-9)

    def test_max_cv_len_preserves_cross_branch_topology(self) -> None:
        soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[3.0, 2.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[10.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.d = dend
        cell = Cell(tree, cv_policy=MaxCVLen(max_cv_len=5.0 * u.um))

        self.assertEqual(cell.n_cv, 6)
        self.assertEqual(cell.cvs[3].parent_cv, 2)
        self.assertIn(3, cell.cvs[2].children_cv)
        self.assertEqual(cell.cvs[5].parent_cv, 4)

    def test_policy_validation_errors(self) -> None:
        tree = _build_two_branch_tree()

        with self.assertRaises(TypeError):
            Cell(tree, cv_policy=CVPerBranch(cv_per_branch=True))
        with self.assertRaises(ValueError):
            Cell(tree, cv_policy=CVPerBranch(cv_per_branch=0))

        with self.assertRaises(TypeError):
            Cell(tree, cv_policy=object())
        with self.assertRaises(TypeError):
            Cell(tree, cv_policy=MaxCVLen(max_cv_len=20.0))
        with self.assertRaises(ValueError):
            Cell(tree, cv_policy=MaxCVLen(max_cv_len=0.0 * u.um))

    def test_d_lambda_is_placeholder(self) -> None:
        policy = DLambda(d_lambda=0.1)
        self.assertTrue(policy.keep_odd)
        self.assertAlmostEqual(float(policy.frequency.to_decimal(u.Hz)), 100.0, places=12)
        self.assertFalse(DLambda(d_lambda=0.1, keep_odd=False).keep_odd)

    def test_d_lambda_uses_branch_specific_ra_cm_from_paint(self) -> None:
        tree = _build_two_branch_tree()
        cell = Cell(tree, cv_policy=DLambda(d_lambda=0.1, keep_odd=False))
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=1.0),
            CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
                axial_resistivity=200.0 * (u.ohm * u.cm),
            ),
        )
        counts = _branch_cv_counts(cell)
        self.assertGreater(counts[1], counts[0])

    def test_d_lambda_promotes_even_count_to_odd_by_default(self) -> None:
        soma = Branch.from_lengths(lengths=[78.0] * u.um, radii=[1.0, 1.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        cell_odd = Cell(tree, cv_policy=DLambda(d_lambda=0.1))
        cell_even = Cell(tree, cv_policy=DLambda(d_lambda=0.1, keep_odd=False))
        self.assertEqual(cell_even.n_cv, 2)
        self.assertEqual(cell_odd.n_cv, 3)

    def test_d_lambda_rejects_branch_internal_ra_cm_conflict(self) -> None:
        tree = _build_two_branch_tree()
        cell = Cell(tree, cv_policy=DLambda(d_lambda=0.1))
        with self.assertRaisesRegex(ValueError, "branch-wise uniform cable properties"):
            cell.paint(
                BranchSlice(branch_index=1, prox=0.0, dist=0.5),
                CableProperty(
                    resting_potential=-65.0 * u.mV,
                    membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
                    axial_resistivity=200.0 * (u.ohm * u.cm),
                ),
            ).cvs

    def test_d_lambda_ignores_resting_potential_and_temperature_conflicts(self) -> None:
        tree = _build_two_branch_tree()
        cell = Cell(tree, cv_policy=DLambda(d_lambda=0.1))
        cell.paint(
            BranchSlice(branch_index=1, prox=0.0, dist=0.5),
            CableProperty(
                resting_potential=-55.0 * u.mV,
                membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
                axial_resistivity=100.0 * (u.ohm * u.cm),
                temperature=u.celsius2kelvin(30.0),
            ),
        )
        self.assertGreater(cell.n_cv, 0)

    def test_composite_by_type_policy_selects_policies_per_branch_type(self) -> None:
        tree = _build_mixed_type_tree()
        policy = CompositeByTypePolicy(
            rules=(
                CVPolicyByTypeRule(branch_types=("soma",), policy=CVPerBranch(cv_per_branch=1)),
                CVPolicyByTypeRule(
                    branch_types=("basal_dendrite", "apical_dendrite"),
                    policy=MaxCVLen(max_cv_len=20.0 * u.um, keep_odd=False),
                ),
            ),
            default_policy=CVPerBranch(cv_per_branch=2),
        )
        cell = Cell(tree, cv_policy=policy)
        self.assertEqual(_branch_cv_counts(cell), {0: 1, 1: 2, 2: 2, 3: 3})

    def test_composite_by_type_policy_uses_last_matching_rule(self) -> None:
        tree = _build_mixed_type_tree()
        policy = CompositeByTypePolicy(
            rules=(
                CVPolicyByTypeRule(branch_types=("axon",), policy=CVPerBranch(cv_per_branch=1)),
                CVPolicyByTypeRule(branch_types=("axon",), policy=CVPerBranch(cv_per_branch=3)),
            ),
            default_policy=CVPerBranch(cv_per_branch=2),
        )
        cell = Cell(tree, cv_policy=policy)
        self.assertEqual(_branch_cv_counts(cell)[1], 3)

    def test_composite_by_type_policy_requires_default_policy(self) -> None:
        with self.assertRaises(TypeError):
            CompositeByTypePolicy(rules=(), default_policy=object())  # type: ignore[arg-type]

    def test_composite_by_type_rule_validates_branch_types(self) -> None:
        with self.assertRaises(ValueError):
            CVPolicyByTypeRule(branch_types=(), policy=CVPerBranch())
        with self.assertRaises(ValueError):
            CVPolicyByTypeRule(branch_types=("not_a_real_type",), policy=CVPerBranch())

    def test_composite_by_type_policy_allows_d_lambda_subpolicy(self) -> None:
        tree = _build_mixed_type_tree()
        policy = CompositeByTypePolicy(
            rules=(
                CVPolicyByTypeRule(branch_types=("axon",), policy=DLambda(d_lambda=0.1, keep_odd=False)),
            ),
            default_policy=CVPerBranch(cv_per_branch=1),
        )
        cell = Cell(tree, cv_policy=policy)
        counts = _branch_cv_counts(cell)
        self.assertGreater(counts[1], counts[0])

    def test_public_base_class_is_still_exported(self) -> None:
        self.assertTrue(issubclass(CVPerBranch, CVPolicy))
        self.assertTrue(issubclass(MaxCVLen, CVPolicy))
        self.assertTrue(issubclass(DLambda, CVPolicy))
        self.assertTrue(issubclass(CompositeByTypePolicy, CVPolicy))
