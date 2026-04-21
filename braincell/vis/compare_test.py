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

import matplotlib.pyplot as plt
import numpy as np

from braincell.vis._testing import make_length_only_tree, make_two_dendrite_tree
from braincell.vis.compare import compare_morphologies, compare_values


class CompareMorphologiesTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_two_morphologies_yield_two_axes(self) -> None:
        tree_a = make_length_only_tree()
        tree_b = make_two_dendrite_tree()
        fig, axes = compare_morphologies([tree_a, tree_b], layout="stem", shape="line")
        self.assertEqual(len(axes), 2)
        self.assertIs(axes[0].figure, fig)

    def test_empty_list_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one morphology"):
            compare_morphologies([])

    def test_custom_titles_are_honoured(self) -> None:
        tree_a = make_length_only_tree()
        fig, axes = compare_morphologies(
            [tree_a, tree_a],
            titles=["control", "treated"],
            layout="stem",
            shape="line",
            align=None,
        )
        self.assertEqual(axes[0].get_title(), "control")
        self.assertEqual(axes[1].get_title(), "treated")

    def test_align_suffix_is_added_to_titles(self) -> None:
        tree_a = make_length_only_tree()
        fig, axes = compare_morphologies([tree_a], layout="stem", shape="line", align="soma")
        self.assertIn("(soma)", axes[0].get_title())

    def test_title_length_mismatch_raises(self) -> None:
        tree_a = make_length_only_tree()
        with self.assertRaisesRegex(ValueError, "titles"):
            compare_morphologies([tree_a, tree_a], titles=["only-one"])


class CompareValuesTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_two_panels_share_morphology(self) -> None:
        tree = make_length_only_tree()
        v_a = np.zeros(len(tree.branches))
        v_b = np.ones(len(tree.branches))
        fig, axes = compare_values(
            tree,
            [v_a, v_b],
            layout="stem",
            shape="line",
            cmap="viridis",
            value_label="V_m",
        )
        self.assertEqual(len(axes), 2)

    def test_titles_default_to_panel_number(self) -> None:
        tree = make_length_only_tree()
        v = np.zeros(len(tree.branches))
        fig, axes = compare_values(tree, [v, v], layout="stem", shape="line")
        self.assertEqual(axes[0].get_title(), "panel 0")
        self.assertEqual(axes[1].get_title(), "panel 1")

    def test_empty_values_raise(self) -> None:
        tree = make_length_only_tree()
        with self.assertRaisesRegex(ValueError, "at least one value array"):
            compare_values(tree, [])

    def test_title_length_mismatch_raises(self) -> None:
        tree = make_length_only_tree()
        v = np.zeros(len(tree.branches))
        with self.assertRaisesRegex(ValueError, "titles"):
            compare_values(tree, [v, v], titles=["only-one"], layout="stem", shape="line")


if __name__ == "__main__":
    unittest.main()
