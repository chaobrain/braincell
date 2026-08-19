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

"""Tests for :mod:`braincell.vis.scene`.

The module is mostly frozen dataclasses that other suites exercise
indirectly. What is tested here is the part that carries behaviour: the
``segment_values`` midpoint rule, ``OverlaySpec`` normalization, the
config delegators, and the container defaults that every scene builder
relies on.
"""

import dataclasses
import unittest

import numpy as np

from braincell import vis as morpho_vis
from braincell.vis._testing import VisDefaultsResetMixin, make_node_tree
from braincell.vis.scene import (
    BranchValues,
    OverlaySpec,
    RenderRequest,
    RenderScene2D,
    RenderScene3D,
    ValueSpec,
    alpha_for_2d,
    alpha_for_2d_line,
    alpha_for_2d_poly,
    alpha_for_3d_tube,
    color_for_2d_branch_type,
    color_for_branch_type,
    edge_color_for_2d_branch_type,
    frustum_edge_linewidth_2d,
)


class BranchValuesTest(unittest.TestCase):
    def test_segment_values_are_endpoint_midpoints(self) -> None:
        values = BranchValues(branch_index=0, point_values=np.array([0.0, 1.0, 3.0]))

        np.testing.assert_allclose(values.segment_values, [0.5, 2.0])

    def test_two_points_collapse_to_one_segment(self) -> None:
        values = BranchValues(branch_index=0, point_values=np.array([-65.0, -55.0]))

        np.testing.assert_allclose(values.segment_values, [-60.0])

    def test_single_point_is_returned_unchanged(self) -> None:
        values = BranchValues(branch_index=0, point_values=np.array([7.0]))

        np.testing.assert_allclose(values.segment_values, [7.0])

    def test_single_point_result_is_a_copy(self) -> None:
        # The <= 1 branch returns .copy() precisely so a caller mutating the
        # result cannot reach back into the branch's own array.
        point_values = np.array([7.0])
        values = BranchValues(branch_index=0, point_values=point_values)

        segments = values.segment_values
        segments[0] = 999.0

        self.assertEqual(point_values[0], 7.0)

    def test_empty_point_values_produce_empty_segments(self) -> None:
        values = BranchValues(branch_index=3, point_values=np.array([]))

        self.assertEqual(values.segment_values.size, 0)


class OverlaySpecTest(unittest.TestCase):
    def test_defaults_are_all_absent(self) -> None:
        overlay = OverlaySpec()

        self.assertIsNone(overlay.region)
        self.assertIsNone(overlay.locset)
        self.assertIsNone(overlay.values)
        self.assertIsNone(overlay.values_spec())

    def test_an_existing_value_spec_is_passed_through_by_identity(self) -> None:
        spec = ValueSpec(values=np.array([0.1, 0.9]), cmap="plasma", label="V_m")

        self.assertIs(OverlaySpec(values=spec).values_spec(), spec)

    def test_a_bare_array_is_wrapped_with_default_styling(self) -> None:
        raw = np.array([0.1, 0.9])

        spec = OverlaySpec(values=raw).values_spec()

        self.assertIsInstance(spec, ValueSpec)
        np.testing.assert_allclose(spec.values, raw)
        self.assertEqual(spec.cmap, "viridis")
        self.assertIsNone(spec.vmin)
        self.assertIsNone(spec.vmax)
        self.assertIsNone(spec.norm)
        self.assertIsNone(spec.label)
        self.assertIsNone(spec.unit_label)
        self.assertTrue(spec.show_colorbar)


class SceneConfigDelegationTest(VisDefaultsResetMixin, unittest.TestCase):
    """The eight accessors must resolve against config at call time.

    ``scene.py`` imports them from ``config`` at module import. Binding the
    *values* there instead of the functions would make every one of these go
    stale the moment a caller reconfigures, and no scene builder would notice
    because they all read through this module.
    """

    def test_branch_type_color_tracks_configure(self) -> None:
        morpho_vis.configure_defaults(branch_type_colors={"soma": "#123456"})

        self.assertEqual(color_for_branch_type("soma"), (18, 52, 86))
        self.assertEqual(color_for_2d_branch_type("soma"), (18, 52, 86))

    def test_unknown_branch_type_falls_back_to_custom(self) -> None:
        morpho_vis.configure_defaults(branch_type_colors={"custom": "#010203"})

        self.assertEqual(color_for_branch_type("not-a-real-type"), (1, 2, 3))

    def test_edge_color_tracks_the_2d_edge_palette(self) -> None:
        morpho_vis.configure_defaults(branch_type_edge_colors_2d={"axon": "#112233"})

        self.assertEqual(edge_color_for_2d_branch_type("axon"), (17, 34, 51))

    def test_shared_alpha_applies_to_both_line_and_poly(self) -> None:
        morpho_vis.configure_defaults(alpha_2d=0.25)

        self.assertAlmostEqual(alpha_for_2d(), 0.25)
        # Neither shape-specific override is set, so both fall back.
        self.assertAlmostEqual(alpha_for_2d_line(), 0.25)
        self.assertAlmostEqual(alpha_for_2d_poly(), 0.25)

    def test_shape_specific_alpha_overrides_the_shared_one(self) -> None:
        morpho_vis.configure_defaults(alpha_2d=0.25, alpha_2d_line=0.4, alpha_2d_poly=0.9)

        self.assertAlmostEqual(alpha_for_2d(), 0.25)
        self.assertAlmostEqual(alpha_for_2d_line(), 0.4)
        self.assertAlmostEqual(alpha_for_2d_poly(), 0.9)

    def test_linewidth_and_3d_alpha_track_configure(self) -> None:
        morpho_vis.configure_defaults(frustum_edge_linewidth_2d=1.75, alpha_3d_tube=0.4)

        self.assertAlmostEqual(frustum_edge_linewidth_2d(), 1.75)
        self.assertAlmostEqual(alpha_for_3d_tube(), 0.4)

    def test_theme_context_manager_is_visible_and_then_restored(self) -> None:
        before = color_for_branch_type("soma")

        with morpho_vis.theme(branch_type_colors={"soma": "#ff0000"}, alpha_2d=0.1):
            self.assertEqual(color_for_branch_type("soma"), (255, 0, 0))
            self.assertAlmostEqual(alpha_for_2d(), 0.1)

        self.assertEqual(color_for_branch_type("soma"), before)


class RenderSceneDefaultsTest(unittest.TestCase):
    def test_render_scene_2d_starts_empty(self) -> None:
        scene = RenderScene2D()

        for name in (
            "polylines",
            "polygons",
            "circles",
            "labels",
            "highlight_strokes",
            "markers",
            "polyline_values",
            "polygon_value_batches",
            "draw_order",
        ):
            self.assertEqual(getattr(scene, name), (), msg=name)
        self.assertIsNone(scene.value_spec)
        self.assertIsNone(scene.projection_plane)
        self.assertEqual(scene.layout, "projected")
        self.assertEqual(scene.shape, "line")

    def test_render_scene_3d_requires_branches_and_batches(self) -> None:
        scene = RenderScene3D(branches=(), batches=())

        self.assertEqual(scene.highlight_strokes, ())
        self.assertEqual(scene.markers, ())
        self.assertEqual(scene.value_batches, ())
        self.assertIsNone(scene.value_spec)
        self.assertEqual(scene.mode, "geometry")

    def test_scene_containers_are_frozen(self) -> None:
        scene = RenderScene2D()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            scene.layout = "stem"  # type: ignore[misc]


class RenderRequestTest(unittest.TestCase):
    def test_defaults(self) -> None:
        request = RenderRequest(morpho=make_node_tree())

        self.assertIsNone(request.scene)
        self.assertIsInstance(request.overlay, OverlaySpec)
        self.assertEqual(request.dimensionality, "3d")
        self.assertIsNone(request.mode)
        self.assertIsNone(request.layout)
        self.assertIsNone(request.shape)
        self.assertEqual(dict(request.backend_options), {})

    def test_two_requests_do_not_share_mutable_defaults(self) -> None:
        # Both fields use default_factory; a plain mutable default would make
        # every request in the process alias one dict and one overlay.
        morpho = make_node_tree()
        first = RenderRequest(morpho=morpho)
        second = RenderRequest(morpho=morpho)

        self.assertIsNot(first.overlay, second.overlay)
        self.assertIsNot(first.backend_options, second.backend_options)

        first.backend_options["ax"] = object()
        self.assertEqual(dict(second.backend_options), {})


if __name__ == "__main__":
    unittest.main()
