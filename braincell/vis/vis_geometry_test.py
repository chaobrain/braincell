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

from __future__ import annotations

import unittest

import numpy as np

from braincell._test_support import u

from braincell import Branch, Morpho
from braincell.vis import build_layout_branches_2d, build_render_scene_2d, build_render_scene_3d


def _length_only_tree() -> Morpho:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[8.0, 12.0] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morpho.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


class VisGeometryTest(unittest.TestCase):
    def test_build_render_scene_3d_groups_branches_by_type(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        axon = Branch.from_points(
            points=[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]] * u.um,
            radii=[1.0, 0.8, 0.6] * u.um,
            type="axon",
        )
        dend = Branch.from_points(
            points=[[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=axon, child_name="axon_slot", parent_x=1.0)
        tree.attach(parent="soma", child_branch=dend, child_name="dend_slot", parent_x=1.0)

        scene = build_render_scene_3d(tree)

        self.assertEqual(len(scene.branches), 3)
        self.assertEqual({batch.branch_type for batch in scene.batches}, {"soma", "axon", "apical_dendrite"})
        soma_branch = scene.branches[0]
        self.assertEqual(soma_branch.branch_name, "soma")
        self.assertEqual(soma_branch.points_um.shape, (2, 3))
        self.assertTrue(np.allclose(soma_branch.radii_um, np.array([5.0, 5.0])))

    def test_build_render_scene_3d_requires_point_geometry(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaises(ValueError):
            build_render_scene_3d(tree)

    def test_tree_layout_uses_uniform_branch_length_and_mean_radius(self) -> None:
        tree = _length_only_tree()

        layouts = build_layout_branches_2d(tree, mode="tree")

        self.assertEqual(len(layouts), 2)
        self.assertTrue(np.allclose(layouts[0].segment_points_um[0], np.array([0.0, 0.0])))
        self.assertAlmostEqual(layouts[0].total_length_um, layouts[1].total_length_um)
        expected_radius_um = float(tree.branch(name="dend").mean_radius.to_decimal(u.um))
        self.assertAlmostEqual(float(layouts[1].radii_proximal_um[0]), expected_radius_um)

    def test_frustum_scene_builds_polygon_per_segment(self) -> None:
        tree = _length_only_tree()

        scene = build_render_scene_2d(tree, mode="frustum")

        self.assertEqual(scene.mode, "frustum")
        self.assertEqual(len(scene.polygons), 3)
        root_polygon = scene.polygons[0]
        self.assertEqual(root_polygon.points_um.shape, (4, 2))

        child_polygons = [polygon for polygon in scene.polygons if polygon.branch_name == "dend"]
        first_child = child_polygons[0]
        start_midpoint = 0.5 * (first_child.points_um[0] + first_child.points_um[3])
        end_midpoint = 0.5 * (first_child.points_um[1] + first_child.points_um[2])
        proximal_width = np.linalg.norm(first_child.points_um[0] - first_child.points_um[3])
        distal_width = np.linalg.norm(first_child.points_um[1] - first_child.points_um[2])

        self.assertAlmostEqual(float(np.linalg.norm(end_midpoint - start_midpoint)), 8.0)
        self.assertAlmostEqual(float(proximal_width), 4.0)
        self.assertAlmostEqual(float(distal_width), 3.0)
