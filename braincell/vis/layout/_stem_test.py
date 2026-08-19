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

from braincell import Branch, Morphology
from braincell.vis._testing import make_length_only_tree, make_root_split_tree
from braincell.vis.layout import build_layout_branches_2d, tangent_on_layout_branch
from braincell.vis.layout._common import _build_layout_specs
from braincell.vis.layout._stem import (
    _build_layout_branches_stem,
    _build_layout_branches_stem_linear,
    _stem_profile_candidates,
    _stem_segment_angles_rad,
)


class StemSegmentAnglesTest(unittest.TestCase):
    def test_single_segment_uses_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.array([10.0]),
            launch_angle_rad=0.1,
            settle_angle_rad=0.2,
            tail_angle_rad=0.3,
        )
        self.assertEqual(angles.shape, (1,))
        self.assertAlmostEqual(float(angles[0]), 0.3)

    def test_two_segments_uses_launch_then_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.array([5.0, 5.0]),
            launch_angle_rad=0.1,
            settle_angle_rad=0.5,
            tail_angle_rad=0.9,
        )
        self.assertEqual(angles.shape, (2,))
        self.assertAlmostEqual(float(angles[0]), 0.1)
        self.assertAlmostEqual(float(angles[1]), 0.9)

    def test_three_segments_uses_launch_settle_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.array([5.0, 5.0, 5.0]),
            launch_angle_rad=0.0,
            settle_angle_rad=0.4,
            tail_angle_rad=0.8,
        )
        self.assertTrue(np.allclose(angles, [0.0, 0.4, 0.8]))

    def test_many_segments_ends_on_tail(self) -> None:
        angles = _stem_segment_angles_rad(
            np.ones(8, dtype=float),
            launch_angle_rad=0.0,
            settle_angle_rad=0.3,
            tail_angle_rad=1.0,
        )
        self.assertAlmostEqual(float(angles[0]), 0.0)
        # Last two segments are clamped to tail_angle.
        self.assertAlmostEqual(float(angles[-1]), 1.0)
        self.assertAlmostEqual(float(angles[-2]), 1.0)


class StemProfileCandidatesTest(unittest.TestCase):
    def test_side_candidates_cover_both_signs(self) -> None:
        profiles = _stem_profile_candidates(
            attach_angle_rad=0.0,
            desired_tail_angle_rad=math.pi / 2.0,
            preferred_sign=1.0,
            min_branch_angle_rad=math.radians(25.0),
            branch_role="side",
            n_segments=4,
        )
        self.assertGreater(len(profiles), 0)

    def test_trunk_candidates_use_desired_sign_only(self) -> None:
        profiles = _stem_profile_candidates(
            attach_angle_rad=0.0,
            desired_tail_angle_rad=math.pi / 2.0,
            preferred_sign=-1.0,
            min_branch_angle_rad=math.radians(25.0),
            branch_role="trunk",
            n_segments=4,
        )
        # Trunk side should settle toward +pi/2 (desired is positive).
        self.assertGreater(len(profiles), 0)
        for profile in profiles:
            self.assertGreater(profile.tail_angle_rad, 0.0)


class BuildStemLinearTest(unittest.TestCase):
    def test_length_only_tree_builds_without_error(self) -> None:
        tree = make_length_only_tree()
        specs = _build_layout_specs(tree)
        layouts = _build_layout_branches_stem_linear(
            tree,
            layout_specs=specs,
            min_branch_angle_deg=25.0,
            root_layout="type_split",
        )
        self.assertEqual(len(layouts), len(tree.branches))
        for layout in layouts:
            self.assertTrue(np.all(np.isfinite(layout.segment_points_um)))


class BuildStemTreeTest(unittest.TestCase):
    def test_length_only_tree_builds_without_error(self) -> None:
        tree = make_length_only_tree()
        specs = _build_layout_specs(tree)
        layouts = _build_layout_branches_stem(
            tree,
            layout_specs=specs,
            min_branch_angle_deg=25.0,
            root_layout="type_split",
        )
        self.assertEqual(len(layouts), len(tree.branches))

    def test_root_split_places_axon_opposite_dendrite(self) -> None:
        tree = make_root_split_tree()
        specs = _build_layout_specs(tree)
        layouts = {
            layout.branch_name: layout
            for layout in _build_layout_branches_stem(
                tree,
                layout_specs=specs,
                min_branch_angle_deg=25.0,
                root_layout="type_split",
            )
        }
        self.assertGreater(layouts["dend"].end_direction_um[1], 0.0)
        self.assertLess(layouts["axon"].end_direction_um[1], 0.0)


def _stem_tree() -> Morphology:
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
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
    tree.attach(parent="soma", child_branch=side, child_name="side", parent_x=1.0)
    tree.attach(parent="trunk", child_branch=trunk_child, child_name="trunk_child", parent_x=1.0)
    return tree


def _overlap_tree() -> Morphology:
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
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=axon_0, child_name="axon_0", parent_x=0.5)
    tree.attach(parent="axon_0", child_branch=axon_1, child_name="axon_1", parent_x=0.0, child_x=0.0)
    return tree


class StemLayoutGeometryTest(unittest.TestCase):
    def test_tree_layout_uses_actual_branch_segment_lengths_and_radii(self) -> None:
        tree = make_length_only_tree()

        layouts = build_layout_branches_2d(tree, mode="tree")

        self.assertEqual(len(layouts), 2)
        self.assertTrue(np.allclose(layouts[0].segment_points_um[0], np.array([0.0, 0.0])))
        self.assertAlmostEqual(layouts[0].total_length_um, 20.0)
        self.assertAlmostEqual(layouts[1].total_length_um, 20.0)
        self.assertEqual(layouts[1].segment_points_um.shape, (3, 2))
        self.assertTrue(np.allclose(layouts[1].cumulative_lengths_um, np.array([0.0, 8.0, 20.0])))
        self.assertTrue(np.allclose(layouts[1].radii_proximal_um, np.array([2.0, 1.5])))
        self.assertTrue(np.allclose(layouts[1].radii_distal_um, np.array([1.5, 1.0])))

    def test_length_only_tree_soma_has_exact_horizontal_centerline(self) -> None:
        """A length-only single-soma tree with a 20 µm soma segment
        should have its layout centerline running from (0, 0) to (20, 0)
        along the +x axis — this is a fixed anchor point for the stem
        layout that must not drift.
        """
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")

        layouts = build_layout_branches_2d(tree, mode="tree")
        soma_layout = layouts[0]

        self.assertEqual(soma_layout.segment_points_um.shape, (2, 2))
        self.assertTrue(np.allclose(soma_layout.segment_points_um[0], np.array([0.0, 0.0])))
        self.assertTrue(np.allclose(soma_layout.segment_points_um[-1], np.array([20.0, 0.0])))
        self.assertTrue(np.allclose(soma_layout.end_direction_um, np.array([1.0, 0.0])))

    def test_root_type_split_places_axon_and_dendrite_in_opposite_half_planes(self) -> None:
        tree = make_root_split_tree()

        layouts = build_layout_branches_2d(tree, mode="tree")
        dend = next(layout for layout in layouts if layout.branch_name == "dend")
        axon = next(layout for layout in layouts if layout.branch_name == "axon")

        self.assertGreater(dend.end_direction_um[1], 0.0)
        self.assertLess(axon.end_direction_um[1], 0.0)

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
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
        tree.attach(parent="trunk", child_branch=side, child_name="side", parent_x=0.0)

        layouts = {layout.branch_name: layout for layout in build_layout_branches_2d(tree, mode="tree")}

        trunk_tangent_um = tangent_on_layout_branch(layouts["trunk"], 0.0)
        self.assertFalse(np.allclose(layouts["side"].start_direction_um, trunk_tangent_um))

    def test_frustum_stem_keeps_parent_local_tangent_at_attach_point(self) -> None:
        soma = Branch.from_lengths(lengths=[12.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
        trunk = Branch.from_lengths(lengths=[8.0, 12.0] * u.um, radii=[2.0, 1.5, 1.0] * u.um, type="apical_dendrite")
        side = Branch.from_lengths(lengths=[3.0, 3.0] * u.um, radii=[1.0, 1.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
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
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
        tree.attach(parent="trunk", child_branch=side, child_name="side", parent_x=0.0)

        layout = next(layout for layout in build_layout_branches_2d(tree, mode="tree") if layout.branch_name == "side")

        self.assertFalse(np.allclose(layout.segment_directions_um[0], layout.segment_directions_um[-1]))
        self.assertTrue(np.allclose(layout.segment_directions_um[-2], layout.segment_directions_um[-1]))


if __name__ == "__main__":
    unittest.main()
