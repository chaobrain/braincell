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



import math
import unittest

import brainunit as u
import numpy as np

from braincell import Branch, Morpho
from braincell.vis import build_layout_branches_2d, build_render_scene_2d, build_render_scene_3d
from braincell.vis.layout2d import tangent_on_layout_branch


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


def _point_tree_with_same_lengths() -> Morpho:
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [0.0, 6.4, 4.8], [12.0, 6.4, 4.8]] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morpho.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


def _root_split_tree() -> Morpho:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[25.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    axon = Branch.from_lengths(
        lengths=[18.0] * u.um,
        radii=[1.0, 0.8] * u.um,
        type="axon",
    )
    tree = Morpho.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


def _legacy_angle_tree() -> Morpho:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend_a = Branch.from_lengths(
        lengths=[15.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    dend_b = Branch.from_lengths(
        lengths=[15.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="basal_dendrite",
    )
    tree = Morpho.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
    tree.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)
    return tree


def _stem_tree() -> Morpho:
    soma = Branch.from_lengths(
        lengths=[12.0] * u.um,
        radii=[8.0, 8.0] * u.um,
        type="soma",
    )
    trunk = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    side = Branch.from_lengths(
        lengths=[10.0] * u.um,
        radii=[1.5, 1.0] * u.um,
        type="basal_dendrite",
    )
    trunk_child = Branch.from_lengths(
        lengths=[25.0] * u.um,
        radii=[1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morpho.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
    tree.attach(parent="soma", child_branch=side, child_name="side", parent_x=1.0)
    tree.attach(parent="trunk", child_branch=trunk_child, child_name="trunk_child", parent_x=1.0)
    return tree


def _overlap_tree() -> Morpho:
    soma = Branch.from_lengths(
        lengths=[10.0] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    axon_0 = Branch.from_lengths(
        lengths=[10.0] * u.um,
        radii=[1.0, 1.0] * u.um,
        type="axon",
    )
    axon_1 = Branch.from_lengths(
        lengths=[8.0] * u.um,
        radii=[1.0, 0.8] * u.um,
        type="axon",
    )
    tree = Morpho.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=axon_0, child_name="axon_0", parent_x=0.5)
    tree.attach(parent="axon_0", child_branch=axon_1, child_name="axon_1", parent_x=0.0, child_x=0.0)
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

    def test_tree_layout_uses_actual_branch_segment_lengths_and_radii(self) -> None:
        tree = _length_only_tree()

        layouts = build_layout_branches_2d(tree, mode="tree")

        self.assertEqual(len(layouts), 2)
        self.assertTrue(np.allclose(layouts[0].segment_points_um[0], np.array([0.0, 0.0])))
        self.assertAlmostEqual(layouts[0].total_length_um, 20.0)
        self.assertAlmostEqual(layouts[1].total_length_um, 20.0)
        self.assertEqual(layouts[1].segment_points_um.shape, (3, 2))
        self.assertTrue(np.allclose(layouts[1].cumulative_lengths_um, np.array([0.0, 8.0, 20.0])))
        self.assertTrue(np.allclose(layouts[1].radii_proximal_um, np.array([2.0, 1.5])))
        self.assertTrue(np.allclose(layouts[1].radii_distal_um, np.array([1.5, 1.0])))

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

    def test_root_type_split_places_axon_and_dendrite_in_opposite_half_planes(self) -> None:
        tree = _root_split_tree()

        layouts = build_layout_branches_2d(tree, mode="tree")
        dend = next(layout for layout in layouts if layout.branch_name == "dend")
        axon = next(layout for layout in layouts if layout.branch_name == "axon")

        self.assertGreater(dend.end_direction_um[1], 0.0)
        self.assertLess(axon.end_direction_um[1], 0.0)

    def test_min_branch_angle_deg_is_applied_in_legacy_layout(self) -> None:
        tree = _legacy_angle_tree()

        layouts = build_layout_branches_2d(
            tree,
            mode="tree",
            min_branch_angle_deg=90.0,
            root_layout="legacy",
        )
        child_angles = sorted(
            math.degrees(math.atan2(layout.end_direction_um[1], layout.end_direction_um[0]))
            for layout in layouts
            if layout.branch_name in {"dend_a", "dend_b"}
        )

        self.assertGreaterEqual(child_angles[1] - child_angles[0], 90.0 - 1e-6)

    def test_stem_keeps_longest_subtree_on_parent_direction(self) -> None:
        tree = _stem_tree()

        layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(tree, mode="tree")}

        self.assertTrue(np.allclose(layouts["trunk_child"].end_direction_um, layouts["trunk"].end_direction_um))
        self.assertFalse(np.allclose(layouts["side"].end_direction_um, layouts["trunk"].end_direction_um))

    def test_stem_breaks_start_to_start_overlap(self) -> None:
        tree = _overlap_tree()

        layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(tree, mode="tree")}

        self.assertLess(abs(layouts["axon_1"].end_direction_um[1]), 1.0)
        self.assertNotAlmostEqual(float(layouts["axon_1"].end_direction_um[1]), 0.0)

    def test_tree_stem_can_launch_away_from_parent_local_tangent(self) -> None:
        soma = Branch.from_lengths(lengths=[12.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
        trunk = Branch.from_lengths(lengths=[8.0, 12.0] * u.um, radii=[2.0, 1.5, 1.0] * u.um, type="apical_dendrite")
        side = Branch.from_lengths(lengths=[3.0, 3.0] * u.um, radii=[1.0, 1.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
        tree.attach(parent="trunk", child_branch=side, child_name="side", parent_x=0.0)

        layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(tree, mode="tree")}

        trunk_tangent_um = tangent_on_layout_branch(layouts["trunk"], 0.0)
        self.assertFalse(np.allclose(layouts["side"].start_direction_um, trunk_tangent_um))

    def test_frustum_stem_keeps_parent_local_tangent_at_attach_point(self) -> None:
        soma = Branch.from_lengths(lengths=[12.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
        trunk = Branch.from_lengths(lengths=[8.0, 12.0] * u.um, radii=[2.0, 1.5, 1.0] * u.um, type="apical_dendrite")
        side = Branch.from_lengths(lengths=[3.0, 3.0] * u.um, radii=[1.0, 1.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
        tree.attach(parent="trunk", child_branch=side, child_name="side", parent_x=0.0)

        layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(tree, mode="frustum")}

        trunk_tangent_um = tangent_on_layout_branch(layouts["trunk"], 0.0)
        self.assertTrue(np.allclose(layouts["side"].start_direction_um, trunk_tangent_um))

    def test_tree_stem_stabilizes_tail_direction_on_late_segments(self) -> None:
        soma = Branch.from_lengths(lengths=[12.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
        trunk = Branch.from_lengths(lengths=[16.0] * u.um, radii=[2.0, 1.8] * u.um, type="apical_dendrite")
        side = Branch.from_lengths(
            lengths=[3.0, 3.0, 3.0, 3.0] * u.um,
            radii=[1.0, 1.0, 1.0, 1.0, 1.0] * u.um,
            type="basal_dendrite",
        )
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
        tree.attach(parent="trunk", child_branch=side, child_name="side", parent_x=0.0)

        layout = next(layout for layout in build_layout_branches_2d(tree, mode="tree") if layout.branch_name == "side")

        self.assertFalse(np.allclose(layout.segment_directions_um[0], layout.segment_directions_um[-1]))
        self.assertTrue(np.allclose(layout.segment_directions_um[-2], layout.segment_directions_um[-1]))

    def test_tree_and_frustum_ignore_real_points_geometry(self) -> None:
        length_tree = _length_only_tree()
        point_tree = _point_tree_with_same_lengths()

        length_layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(length_tree, mode="tree")}
        point_layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(point_tree, mode="tree")}

        self.assertTrue(np.allclose(length_layouts["dend"].segment_points_um, point_layouts["dend"].segment_points_um))

        length_scene = build_render_scene_2d(length_tree, mode="frustum")
        point_scene = build_render_scene_2d(point_tree, mode="frustum")

        self.assertEqual(len(length_scene.polygons), len(point_scene.polygons))
        for length_polygon, point_polygon in zip(length_scene.polygons, point_scene.polygons):
            self.assertTrue(np.allclose(length_polygon.points_um, point_polygon.points_um))

    def test_balloon_layout_assigns_distinct_child_angles(self) -> None:
        tree = _legacy_angle_tree()

        layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(tree, mode="tree", layout_family="balloon")}
        dend_a_angle = math.degrees(math.atan2(layouts["dend_a"].end_direction_um[1], layouts["dend_a"].end_direction_um[0]))
        dend_b_angle = math.degrees(math.atan2(layouts["dend_b"].end_direction_um[1], layouts["dend_b"].end_direction_um[0]))

        self.assertNotAlmostEqual(dend_a_angle, dend_b_angle)

    def test_balloon_layout_straightens_after_initial_bend(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend_a = Branch.from_lengths(lengths=[5.0, 5.0, 10.0] * u.um, radii=[2.0, 1.8, 1.4, 1.0] * u.um, type="apical_dendrite")
        dend_b = Branch.from_lengths(lengths=[8.0] * u.um, radii=[1.5, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
        tree.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)

        layout = next(
            layout for layout in build_layout_branches_2d(tree, mode="tree", layout_family="balloon")
            if layout.branch_name == "dend_a"
        )

        self.assertFalse(np.allclose(layout.segment_directions_um[0], layout.segment_directions_um[-1]))
        self.assertTrue(np.allclose(layout.segment_directions_um[-2], layout.segment_directions_um[-1]))
        self.assertTrue(np.allclose(layout.segment_directions_um[-1], layout.end_direction_um))

    def test_radial_360_spreads_root_stems_across_multiple_quadrants(self) -> None:
        soma = Branch.from_lengths(lengths=[12.0] * u.um, radii=[6.0, 6.0] * u.um, type="soma")
        tree = Morpho.from_root(soma, name="soma")
        for index in range(4):
            tree.attach(
                parent="soma",
                child_branch=Branch.from_lengths(lengths=[10.0, 6.0] * u.um, radii=[1.5, 1.0, 0.8] * u.um, type="basal_dendrite"),
                child_name=f"d{index}",
                parent_x=1.0,
            )

        layouts = {
            layout.branch_name: layout
            for layout in build_layout_branches_2d(tree, mode="tree", layout_family="radial_360")
            if layout.branch_name.startswith("d")
        }

        quadrants = {
            (np.sign(layout.end_direction_um[0]), np.sign(layout.end_direction_um[1]))
            for layout in layouts.values()
        }
        self.assertGreaterEqual(len(quadrants), 3)
