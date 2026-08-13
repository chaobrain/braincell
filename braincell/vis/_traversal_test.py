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

import sys
import unittest

from braincell.morph import MorphoBranch
from braincell.vis._testing import (
    make_deep_chain_tree,
    make_fan_root_partition_tree,
    make_length_only_tree,
    make_two_dendrite_tree,
)
from braincell.vis._traversal import iter_bottom_up, iter_depth_first


def _recursive_depth_first(node: MorphoBranch) -> list[MorphoBranch]:
    """The recursion ``iter_depth_first`` replaces, kept as an oracle."""
    order = [node]
    for child in node.children:
        order.extend(_recursive_depth_first(child))
    return order


class IterDepthFirstTest(unittest.TestCase):
    def test_matches_the_recursive_order(self) -> None:
        for tree in (make_length_only_tree(), make_two_dendrite_tree(), make_fan_root_partition_tree()):
            with self.subTest(n_branches=len(tree.branches)):
                self.assertEqual(
                    [branch.index for branch in iter_depth_first(tree.root)],
                    [branch.index for branch in _recursive_depth_first(tree.root)],
                )

    def test_visits_every_branch_exactly_once(self) -> None:
        tree = make_fan_root_partition_tree()
        indices = [branch.index for branch in iter_depth_first(tree.root)]

        self.assertEqual(len(indices), len(tree.branches))
        self.assertEqual(set(indices), {branch.index for branch in tree.branches})

    def test_parents_precede_their_children(self) -> None:
        tree = make_fan_root_partition_tree()
        position = {branch.index: order for order, branch in enumerate(iter_depth_first(tree.root))}

        for branch in tree.branches:
            if branch.parent is not None:
                self.assertLess(position[branch.parent.index], position[branch.index])

    def test_single_branch_tree(self) -> None:
        tree = make_length_only_tree()
        leaf = tree.root.children[0]
        self.assertEqual(iter_depth_first(leaf), [leaf])

    def test_subtree_root_only_yields_that_subtree(self) -> None:
        tree = make_fan_root_partition_tree()
        child = tree.root.children[0]
        indices = {branch.index for branch in iter_depth_first(child)}

        self.assertIn(child.index, indices)
        self.assertNotIn(tree.root.index, indices)


class IterBottomUpTest(unittest.TestCase):
    def test_children_precede_their_parents(self) -> None:
        tree = make_fan_root_partition_tree()
        position = {branch.index: order for order, branch in enumerate(iter_bottom_up(tree.root))}

        for branch in tree.branches:
            if branch.parent is not None:
                self.assertLess(position[branch.index], position[branch.parent.index])

    def test_root_is_last(self) -> None:
        tree = make_two_dendrite_tree()
        self.assertIs(iter_bottom_up(tree.root)[-1], tree.root)

    def test_is_the_reverse_of_depth_first(self) -> None:
        tree = make_fan_root_partition_tree()
        self.assertEqual(
            [branch.index for branch in iter_bottom_up(tree.root)],
            [branch.index for branch in reversed(iter_depth_first(tree.root))],
        )


class DeepChainTest(unittest.TestCase):
    """Chains deeper than the interpreter's frame budget must still walk."""

    def test_depth_first_survives_a_chain_deeper_than_the_recursion_limit(self) -> None:
        n_branches = sys.getrecursionlimit() * 2
        tree = make_deep_chain_tree(n_branches)

        order = iter_depth_first(tree.root)

        self.assertEqual(len(order), n_branches)
        self.assertIs(order[0], tree.root)

    def test_bottom_up_survives_the_same_chain(self) -> None:
        n_branches = sys.getrecursionlimit() * 2
        tree = make_deep_chain_tree(n_branches)

        order = iter_bottom_up(tree.root)

        self.assertEqual(len(order), n_branches)
        self.assertIs(order[-1], tree.root)
        self.assertEqual(order[0].n_children, 0)


if __name__ == "__main__":
    unittest.main()
