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

"""Tests for :mod:`braincell._compute.topology`."""

import unittest

import brainunit as u
import numpy as np

from braincell._compute.topology import (
    _EPS_PARAM,
    _compute_peel_levels,
    _locate_branch_cv_by_x,
    build_point_scheduling,
    build_point_tree,
)
from braincell._cv import CVPerBranch
from braincell._cv.base import build_cvs
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology


def _two_branch_morpho() -> Morphology:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"
    )
    dend = Branch.from_lengths(
        lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite"
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


class BuildPointTreeEdgeHalves(unittest.TestCase):

    def test_intra_branch_edges_carry_both_halves(self) -> None:
        morpho = _two_branch_morpho()
        cvs = build_cvs(morpho, policy=CVPerBranch())
        tree = build_point_tree(morpho, cvs=cvs)

        dend_cv_ids = [cv.id for cv in cvs if cv.branch_id == 1]
        self.assertGreater(len(dend_cv_ids), 0)

        halves_seen: set[str] = set()
        for edge in tree.edges:
            for cv_edge in edge.cv_edges:
                if cv_edge.cv_id in dend_cv_ids:
                    halves_seen.add(cv_edge.half)
        self.assertEqual(halves_seen, {"prox", "dist"})

    def test_every_cv_has_exactly_one_prox_and_one_dist_role(self) -> None:
        """Guard against a regression that collapses both halves to a single tag."""
        morpho = _two_branch_morpho()
        cvs = build_cvs(morpho, policy=CVPerBranch())
        tree = build_point_tree(morpho, cvs=cvs)

        per_cv_halves: dict[int, list[str]] = {cv.id: [] for cv in cvs}
        for edge in tree.edges:
            for cv_edge in edge.cv_edges:
                per_cv_halves[cv_edge.cv_id].append(cv_edge.half)

        for cv_id, halves in per_cv_halves.items():
            self.assertEqual(
                sorted(halves),
                ["dist", "prox"],
                f"CV {cv_id} must record exactly one prox and one dist edge role; got {halves!r}.",
            )


class ComputePeelLevels(unittest.TestCase):

    def test_peel_levels_correct_when_child_id_less_than_parent_id(self) -> None:
        # Graph with root id > child ids: root=3, children=(0, 1); 0 -> 2.
        point_parent = np.asarray([3, 3, 0, -1], dtype=np.int32)
        point_children = (
            (2,),
            (),
            (),
            (0, 1),
        )
        levels = _compute_peel_levels(
            point_parent=point_parent,
            point_children=point_children,
        )
        # Leaves (1, 2) have peel 0; node 0 has peel 1; root 3 has peel 2.
        self.assertEqual(levels.tolist(), [1, 0, 0, 2])

    def test_peel_levels_handle_isolated_leaves(self) -> None:
        point_parent = np.asarray([-1, -1], dtype=np.int32)
        point_children = ((), ())
        levels = _compute_peel_levels(
            point_parent=point_parent,
            point_children=point_children,
        )
        self.assertEqual(levels.tolist(), [0, 0])

    def test_peel_levels_raise_on_cycle(self) -> None:
        # Contrived cycle 0 -> 1 -> 0. Not producible by build_point_tree but
        # peel computation must terminate with a clear error.
        point_parent = np.asarray([1, 0], dtype=np.int32)
        point_children = ((1,), (0,))
        with self.assertRaises(ValueError) as ctx:
            _compute_peel_levels(
                point_parent=point_parent,
                point_children=point_children,
            )
        self.assertIn("cycle", str(ctx.exception).lower())


class _FakeCV:

    def __init__(self, id_: int, prox: float, dist: float) -> None:
        self.id = id_
        self.prox = prox
        self.dist = dist


class LocateBranchCVByX(unittest.TestCase):

    def _cvs(self, tiles):
        return tuple(_FakeCV(i, p, d) for i, (p, d) in enumerate(tiles))

    def test_interior_x_lands_in_matching_cv(self) -> None:
        cvs = self._cvs([(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)])
        ids = (0, 1, 2)
        got = _locate_branch_cv_by_x(ids, cvs, x=0.5, epsilon=_EPS_PARAM)
        self.assertEqual(got, 1)

    def test_x_near_one_returns_last_cv(self) -> None:
        cvs = self._cvs([(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)])
        ids = (0, 1, 2)
        got = _locate_branch_cv_by_x(ids, cvs, x=0.999, epsilon=_EPS_PARAM)
        self.assertEqual(got, 2)

    def test_x_in_gap_between_tiles_raises(self) -> None:
        cvs = self._cvs([(0.0, 0.4), (0.6, 1.0)])
        ids = (0, 1)
        with self.assertRaises(ValueError) as ctx:
            _locate_branch_cv_by_x(ids, cvs, x=0.5, epsilon=_EPS_PARAM)
        self.assertIn("0.5", str(ctx.exception))


class VocabularyLock(unittest.TestCase):

    def test_cvpoint_positions_are_three_letter_codes(self) -> None:
        morpho = _two_branch_morpho()
        cvs = build_cvs(morpho, policy=CVPerBranch())
        tree = build_point_tree(morpho, cvs=cvs)
        seen = {cvp.position for point in tree.points for cvp in point.cv_points}
        self.assertTrue(seen.issubset({"prox", "mid", "dist"}))
        self.assertIn("mid", seen)


class BuildPointSchedulingGroups(unittest.TestCase):
    """Groups are level-partitioned: each row appears exactly once.

    Regression for a bug where ``_build_groups`` sliced a flat ``order``
    array without capping at the current level end, so group[i] contained
    every row from level_starts[i] through the end of the array. DHS edge
    building then emitted the same parent/child pairs many times per step
    and the implicit-Euler voltage solve blew up (rest -65 mV drifted to
    ~0 mV within a single step; no HH spike could form).
    """

    def _ctx(self):
        morpho = _two_branch_morpho()
        cvs = build_cvs(morpho, policy=CVPerBranch(cv_per_branch=2))
        tree = build_point_tree(morpho, cvs=cvs)
        return tree, build_point_scheduling(tree)

    def test_group_sizes_sum_to_n_point(self) -> None:
        tree, sched = self._ctx()
        self.assertEqual(sum(len(g) for g in sched.groups), len(tree.points))

    def test_each_row_appears_in_exactly_one_group(self) -> None:
        tree, sched = self._ctx()
        all_rows = [int(row) for g in sched.groups for row in g.tolist()]
        self.assertEqual(sorted(all_rows), list(range(len(tree.points))))

    def test_group_sizes_match_level_size(self) -> None:
        _tree, sched = self._ctx()
        self.assertEqual(
            [len(g) for g in sched.groups],
            list(sched.level_size.tolist()),
        )

    def test_edges_count_equals_tree_edges(self) -> None:
        tree, sched = self._ctx()
        expected = sum(1 for p in tree.points if int(tree.point_parent[p.id]) >= 0)
        self.assertEqual(sched.edges.shape, (expected, 2))

    def test_dhs_edges_are_unique(self) -> None:
        _tree, sched = self._ctx()
        seen = {(int(r), int(p)) for r, p in sched.edges.tolist()}
        self.assertEqual(len(seen), sched.edges.shape[0])


if __name__ == "__main__":
    unittest.main()
