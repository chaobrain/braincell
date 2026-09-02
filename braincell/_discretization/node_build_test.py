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

"""Tests for :mod:`braincell._discretization.node_build`."""

import unittest

from braincell._discretization import CVPerBranch
from braincell._discretization._testing import make_two_branch_morpho
from braincell._discretization.base import build_discretization
from braincell._discretization.node_build import build_node_tree_from_cvs as build_node_tree


class BuildNodeTreeEdgeHalves(unittest.TestCase):
    def test_intra_branch_edges_carry_both_halves(self) -> None:
        morpho = make_two_branch_morpho()
        cvs = build_discretization(morpho, policy=CVPerBranch()).cvs
        tree = build_node_tree(morpho, cvs=cvs)

        dend_cv_ids = [cv.id for cv in cvs if cv.branch_id == 1]
        self.assertGreater(len(dend_cv_ids), 0)

        halves_seen: set[str] = set()
        for edge in tree.edges:
            for cv_edge in edge.roles:
                if cv_edge.cv_id in dend_cv_ids:
                    halves_seen.add(cv_edge.half)
        self.assertEqual(halves_seen, {"prox", "dist"})

    def test_every_cv_has_exactly_one_prox_and_one_dist_role(self) -> None:
        """Guard against a regression that collapses both halves to a single tag."""
        morpho = make_two_branch_morpho()
        cvs = build_discretization(morpho, policy=CVPerBranch()).cvs
        tree = build_node_tree(morpho, cvs=cvs)

        per_cv_halves: dict[int, list[str]] = {cv.id: [] for cv in cvs}
        for edge in tree.edges:
            for cv_edge in edge.roles:
                per_cv_halves[cv_edge.cv_id].append(cv_edge.half)

        for cv_id, halves in per_cv_halves.items():
            self.assertEqual(
                sorted(halves),
                ["dist", "prox"],
                f"CV {cv_id} must record exactly one prox and one dist edge role; got {halves!r}.",
            )


class VocabularyLock(unittest.TestCase):
    def test_cvpoint_positions_are_three_letter_codes(self) -> None:
        morpho = make_two_branch_morpho()
        cvs = build_discretization(morpho, policy=CVPerBranch()).cvs
        tree = build_node_tree(morpho, cvs=cvs)
        seen = {role.position for node in tree.nodes for role in node.roles}
        self.assertTrue(seen.issubset({"prox", "mid", "dist"}))
        self.assertIn("mid", seen)


if __name__ == "__main__":
    unittest.main()
