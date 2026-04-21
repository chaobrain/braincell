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

from braincell._compute.topology import build_point_tree
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


if __name__ == "__main__":
    unittest.main()
