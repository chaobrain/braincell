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
import numpy as np

from braincell import Branch, Morphology
from braincell.vis.layout._common import _build_layout_specs
from braincell.vis.layout._radial import _build_layout_branches_radial_360


def _four_stem_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[12.0] * u.um, radii=[6.0, 6.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
    for index in range(4):
        tree.attach(
            parent="soma",
            child_branch=Branch.from_lengths(
                lengths=[10.0, 6.0] * u.um,
                radii=[1.5, 1.0, 0.8] * u.um,
                type="basal_dendrite",
            ),
            child_name=f"d{index}",
            parent_x=1.0,
        )
    return tree


class BuildRadial360Test(unittest.TestCase):
    def test_radial_builds_layouts_for_each_branch(self) -> None:
        tree = _four_stem_tree()
        specs = _build_layout_specs(tree)
        layouts = _build_layout_branches_radial_360(
            tree,
            layout_specs=specs,
            min_branch_angle_deg=10.0,
        )
        self.assertEqual(len(layouts), len(tree.branches))
        for layout in layouts:
            self.assertTrue(np.all(np.isfinite(layout.segment_points_um)))

    def test_root_stems_span_multiple_quadrants(self) -> None:
        tree = _four_stem_tree()
        specs = _build_layout_specs(tree)
        layouts = {
            layout.branch_name: layout
            for layout in _build_layout_branches_radial_360(
                tree,
                layout_specs=specs,
                min_branch_angle_deg=10.0,
            )
            if layout.branch_name.startswith("d")
        }
        quadrants = {
            (np.sign(layout.end_direction_um[0]), np.sign(layout.end_direction_um[1]))
            for layout in layouts.values()
        }
        self.assertGreaterEqual(len(quadrants), 3)


if __name__ == "__main__":
    unittest.main()
