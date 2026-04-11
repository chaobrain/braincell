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


import dataclasses
import math
import unittest

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.vis._testing import (
    make_length_only_tree,
    make_root_split_tree,
    make_two_dendrite_tree,
)
from braincell.vis.layout import (
    DEFAULT_LAYOUT_CONFIG,
    LayoutConfig,
    build_layout_branches_2d,
)


def _two_axon_tree() -> Morphology:
    """Root with two axon children + two dendrite children.

    Useful for exercising balloon/stem type-split spans, because the
    span actually matters only when a group holds 2+ children (a
    single-child group is always placed at the group center).
    """
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
    for index in range(2):
        tree.attach(
            parent="soma",
            child_branch=Branch.from_lengths(
                lengths=[15.0] * u.um,
                radii=[2.0, 1.0] * u.um,
                type="axon",
            ),
            child_name=f"axon_{index}",
            parent_x=1.0,
        )
    for index in range(2):
        tree.attach(
            parent="soma",
            child_branch=Branch.from_lengths(
                lengths=[15.0] * u.um,
                radii=[2.0, 1.0] * u.um,
                type="apical_dendrite",
            ),
            child_name=f"dend_{index}",
            parent_x=1.0,
        )
    return tree


class LayoutConfigDataclassTest(unittest.TestCase):
    def test_is_frozen(self) -> None:
        config = LayoutConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.collision_margin_um = 99.0  # type: ignore[misc]

    def test_default_singleton_is_default_instance(self) -> None:
        self.assertEqual(DEFAULT_LAYOUT_CONFIG, LayoutConfig())

    def test_replace_produces_modified_copy(self) -> None:
        config = dataclasses.replace(DEFAULT_LAYOUT_CONFIG, collision_margin_um=10.0)
        self.assertEqual(config.collision_margin_um, 10.0)
        self.assertEqual(DEFAULT_LAYOUT_CONFIG.collision_margin_um, 2.0)

    def test_every_documented_field_is_present(self) -> None:
        # Defensive: these names are referenced in docstrings and in
        # the scene2d plumbing. Renaming one without updating callers
        # would break silently otherwise.
        expected = {
            "collision_margin_um",
            "collision_retry_limit",
            "stem_collision_window",
            "collision_cell_size_um",
            "default_bend_fraction",
            "balloon_bend_fraction",
            "radial_bend_fraction",
            "stem_root_full_span_rad",
            "stem_root_group_span_rad",
            "balloon_root_span_rad",
            "balloon_child_span_rad",
            "balloon_type_split_span_rad",
            "radial_root_span_rad",
            "radial_child_span_rad",
            "legacy_root_child_span_rad",
            "stem_collision_weight",
            "stem_tail_delta_weight",
            "stem_settle_delta_weight",
            "stem_overturn_weight",
            "stem_trunk_tail_delta_weight",
            "stem_side_opening_weight",
        }
        actual = {field.name for field in dataclasses.fields(LayoutConfig)}
        self.assertEqual(expected, actual)


class LayoutConfigThreadingTest(unittest.TestCase):
    def test_none_recovers_default_behaviour(self) -> None:
        tree = make_length_only_tree()
        without = build_layout_branches_2d(tree, mode="tree", layout_config=None)
        with_default = build_layout_branches_2d(tree, mode="tree", layout_config=DEFAULT_LAYOUT_CONFIG)
        self.assertEqual(len(without), len(with_default))
        for a, b in zip(without, with_default):
            self.assertTrue(np.allclose(a.segment_points_um, b.segment_points_um))

    def test_balloon_span_change_shifts_multi_child_group(self) -> None:
        tree = _two_axon_tree()
        tight = LayoutConfig(balloon_type_split_span_rad=math.radians(20.0))
        wide = LayoutConfig(balloon_type_split_span_rad=math.radians(160.0))

        tight_layouts = {
            layout.branch_name: layout
            for layout in build_layout_branches_2d(
                tree,
                mode="tree",
                layout_family="balloon",
                layout_config=tight,
            )
        }
        wide_layouts = {
            layout.branch_name: layout
            for layout in build_layout_branches_2d(
                tree,
                mode="tree",
                layout_family="balloon",
                layout_config=wide,
            )
        }

        # With 2 dendrites per group, the wider span should push them
        # further apart. Compare the angle between dend_0 and dend_1.
        def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
            cos_t = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
            return math.acos(cos_t)

        tight_gap = _angle_between(
            tight_layouts["dend_0"].end_direction_um,
            tight_layouts["dend_1"].end_direction_um,
        )
        wide_gap = _angle_between(
            wide_layouts["dend_0"].end_direction_um,
            wide_layouts["dend_1"].end_direction_um,
        )
        self.assertGreater(wide_gap, tight_gap + math.radians(5.0))

    def test_balloon_bend_fraction_changes_curve_shape(self) -> None:
        # A multi-segment branch going off-axis has a visible curve,
        # and that curve depends on the bend fraction. Use a
        # dendrite with 3 segments.
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend_a = Branch.from_lengths(
            lengths=[5.0, 5.0, 5.0] * u.um,
            radii=[2.0, 1.8, 1.5, 1.0] * u.um,
            type="apical_dendrite",
        )
        dend_b = Branch.from_lengths(
            lengths=[5.0, 5.0, 5.0] * u.um,
            radii=[2.0, 1.8, 1.5, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
        tree.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)

        sharp = LayoutConfig(balloon_bend_fraction=0.05)
        smooth = LayoutConfig(balloon_bend_fraction=0.95)
        sharp_layouts = {
            layout.branch_name: layout
            for layout in build_layout_branches_2d(
                tree,
                mode="tree",
                layout_family="balloon",
                layout_config=sharp,
            )
        }
        smooth_layouts = {
            layout.branch_name: layout
            for layout in build_layout_branches_2d(
                tree,
                mode="tree",
                layout_family="balloon",
                layout_config=smooth,
            )
        }
        self.assertFalse(
            np.allclose(
                sharp_layouts["dend_a"].segment_points_um,
                smooth_layouts["dend_a"].segment_points_um,
            )
        )


if __name__ == "__main__":
    unittest.main()
