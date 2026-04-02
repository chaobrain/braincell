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
from braincell.filter import (
    BranchPoints,
    BranchSlice,
    LocsetSetOp,
    RandomSamples,
    RootLocation,
    Terminals,
    UniformSamples,
)


def _build_branchpoint_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    basal = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    axon = Branch.from_lengths(lengths=[10.0] * u.um, radii=[0.8, 0.5] * u.um, type="axon")
    apical = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.5, 1.0] * u.um, type="apical_dendrite")
    tuft = Branch.from_lengths(lengths=[15.0] * u.um, radii=[0.9, 0.7] * u.um, type="apical_dendrite")

    tree = Morpho.from_root(soma, name="soma")
    tree.soma.attach(basal, name="basal", parent_x=0.25)
    tree.soma.attach(axon, name="axon", parent_x=0.5)
    tree.soma.attach(apical, name="apical", parent_x=0.5)
    tree.soma.basal.tuft = tuft
    return tree


def _build_sampling_tree() -> Morpho:
    soma = Branch.from_lengths(lengths=[10.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")

    tree = Morpho.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


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
    def test_branch_points_returns_parent_side_points_for_multifurcations(self) -> None:
        tree = _build_branchpoint_tree()

        locset = BranchPoints().evaluate(tree)

        self.assertEqual(locset.points, ((0, 0.25), (0, 0.5)))

    def test_terminals_returns_leaf_branch_distal_points(self) -> None:
        tree = _build_branchpoint_tree()

        locset = Terminals().evaluate(tree)

        self.assertEqual(locset.points, ((2, 1.0), (3, 1.0), (4, 1.0)))

    def test_locset_boolean_ops_union_intersection_difference(self) -> None:
        tree = _build_branchpoint_tree()

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


class LocsetSetOpValidationTest(unittest.TestCase):
    def test_invalid_operator_and_arity_raise(self) -> None:
        tree = _build_sampling_tree()
        left = RootLocation(x=0.2)
        right = RootLocation(x=0.8)

        with self.assertRaises(ValueError):
            LocsetSetOp("invalid", (left, right)).evaluate(tree)
        with self.assertRaises(ValueError):
            LocsetSetOp("union", (left,)).evaluate(tree)
