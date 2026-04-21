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
    _compute_peel_levels,
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


if __name__ == "__main__":
    unittest.main()
