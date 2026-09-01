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

import braincell
import matplotlib as mpl

from braincell import vis as morpho_vis
from braincell.vis import plot2d, plot3d
from braincell.vis._testing import (
    FakeBackend,
    VisDefaultsResetMixin,
    make_length_only_tree,
    make_node_tree,
)
from braincell.vis.backend import BackendChooser
from braincell.vis.config import (
    DISPATCHED_2D_LAYOUTS,
    LAYOUT_2D_ALIASES,
    LAYOUT_2D_DISPATCHED_NAMES,
    LAYOUT_2D_FAMILIES,
    PUBLICATION_BRANCH_TYPE_COLORS,
    PUBLICATION_RC_PARAMS,
    PublicationTheme,
    SUPPORTED_2D_LAYOUTS,
    configure as configure_defaults,
    edge_color_for_2d_branch_type,
    frustum_edge_linewidth_2d,
    get_defaults,
    layout_2d_title,
    publication_theme,
    resolve_default_2d_layout,
    reset_defaults,
    rgb_to_float,
    rgb_to_hex,
)


class PublicationThemeDataclassTest(unittest.TestCase):
    def test_default_fields_are_copies(self) -> None:
        theme_a = PublicationTheme()
        theme_b = PublicationTheme()
        # Mutating one must not touch the other — the dataclass uses
        # ``default_factory`` so each instance owns its own dict.
        theme_a.branch_type_colors["soma"] = (255, 255, 255)
        self.assertEqual(
            theme_b.branch_type_colors["soma"],
            PUBLICATION_BRANCH_TYPE_COLORS["soma"],
        )

    def test_preset_rc_params_default_keys_match(self) -> None:
        self.assertEqual(set(PublicationTheme().rc_params), set(PUBLICATION_RC_PARAMS))


class PublicationThemeContextManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_defaults()
        self.addCleanup(reset_defaults)
        self._rc_before = dict(mpl.rcParams)

    def tearDown(self) -> None:
        mpl.rcParams.update(self._rc_before)

    def test_enter_applies_vis_defaults(self) -> None:
        baseline_alpha = get_defaults().alpha_2d
        with publication_theme():
            inside = get_defaults()
            self.assertEqual(inside.alpha_2d, 0.7)
            self.assertIsNone(inside.alpha_2d_line)
            self.assertIsNone(inside.alpha_2d_poly)
        self.assertEqual(get_defaults().alpha_2d, baseline_alpha)

    def test_enter_applies_rc_params(self) -> None:
        original_lw = mpl.rcParams["lines.linewidth"]
        with publication_theme():
            self.assertEqual(mpl.rcParams["lines.linewidth"], 1.6)
            self.assertEqual(mpl.rcParams["axes.grid"], False)
        self.assertEqual(mpl.rcParams["lines.linewidth"], original_lw)

    def test_rc_overrides_merge_on_top_of_preset(self) -> None:
        with publication_theme(rc_overrides={"lines.linewidth": 4.0}):
            self.assertEqual(mpl.rcParams["lines.linewidth"], 4.0)

    def test_unknown_rc_keys_are_dropped_silently(self) -> None:
        # Exotic override keys should not crash the context manager.
        with publication_theme(rc_overrides={"this.key.does.not.exist": 1.0}):
            pass  # No exception is the assertion.

    def test_restores_on_exception(self) -> None:
        baseline = dict(mpl.rcParams)
        try:
            with publication_theme():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # The specific keys the theme touches should be back.
        for key in PUBLICATION_RC_PARAMS:
            if key in baseline:
                self.assertEqual(mpl.rcParams[key], baseline[key])


class DefaultsTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_defaults()
        self.addCleanup(reset_defaults)

    def test_default_2d_layout_is_fan(self) -> None:
        self.assertEqual(get_defaults().layout_2d_default, "fan")
        self.assertEqual(resolve_default_2d_layout(None), "fan")

    def test_braincell_top_level_exposes_vis_namespace(self) -> None:
        self.assertTrue(hasattr(braincell, "vis"))
        self.assertTrue(hasattr(braincell.vis, "theme"))

    def test_default_2d_branch_palette_is_publication_friendly(self) -> None:
        defaults = get_defaults()

        self.assertEqual(defaults.branch_type_colors["soma"], (47, 49, 54))
        self.assertEqual(defaults.branch_type_colors["apical_dendrite"], (214, 173, 98))
        self.assertEqual(edge_color_for_2d_branch_type("soma"), (34, 35, 39))
        self.assertAlmostEqual(frustum_edge_linewidth_2d(), 0.9)

    def test_configure_merges_2d_edge_colours(self) -> None:
        original = edge_color_for_2d_branch_type("soma")

        configure_defaults(
            branch_type_edge_colors_2d={"soma": "#123456"},
            frustum_edge_linewidth_2d=1.75,
        )

        self.assertEqual(edge_color_for_2d_branch_type("soma"), (18, 52, 86))
        self.assertEqual(edge_color_for_2d_branch_type("axon"), (78, 102, 125))
        self.assertAlmostEqual(frustum_edge_linewidth_2d(), 1.75)
        self.assertNotEqual(edge_color_for_2d_branch_type("soma"), original)


class VisDefaultsThroughPlotTest(VisDefaultsResetMixin, unittest.TestCase):
    """Global defaults and the theme context manager, observed through plot2d."""

    def test_global_vis_defaults_change_layout_shape_and_style(self) -> None:
        morpho_vis.configure_defaults(
            layout_2d_default="stem",
            shape_2d_default="line",
            branch_type_colors={"soma": "#123456"},
            alpha_2d=0.25,
            alpha_3d_tube=0.4,
        )
        backend = FakeBackend()

        request_2d = plot2d(make_node_tree(), chooser=BackendChooser(backends=(backend,)))
        request_3d = plot3d(make_node_tree(), chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request_2d.layout, "stem")
        self.assertEqual(request_2d.shape, "line")
        self.assertTrue(all(polyline.color_rgb == (18, 52, 86) for polyline in request_2d.scene.polylines))
        self.assertTrue(all(abs(polyline.alpha - 0.25) < 1e-9 for polyline in request_2d.scene.polylines))
        self.assertEqual(request_3d.scene.batches[0].color_rgb, (18, 52, 86))
        self.assertAlmostEqual(request_3d.scene.batches[0].opacity, 0.4)

    def test_theme_context_manager_restores_defaults_on_exit(self) -> None:
        backend = FakeBackend()

        with morpho_vis.theme(branch_type_colors={"soma": "#ff0000"}, alpha_2d=0.1):
            inside = plot2d(
                make_node_tree(),
                shape="line",
                chooser=BackendChooser(backends=(backend,)),
            )
            self.assertEqual(inside.scene.polylines[0].color_rgb, (255, 0, 0))
            self.assertAlmostEqual(inside.scene.polylines[0].alpha, 0.1)

        after = plot2d(
            make_node_tree(),
            shape="line",
            chooser=BackendChooser(backends=(backend,)),
        )
        self.assertEqual(after.scene.polylines[0].color_rgb, (47, 49, 54))
        self.assertAlmostEqual(after.scene.polylines[0].alpha, 0.8)

    def test_global_2d_style_also_applies_to_frustum(self) -> None:
        morpho_vis.configure_defaults(
            branch_type_colors={"apical_dendrite": "#445566"},
            branch_type_edge_colors_2d={"apical_dendrite": "#112233"},
            frustum_edge_linewidth_2d=1.75,
            alpha_2d=0.6,
        )
        backend = FakeBackend()

        request = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="frustum",
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertTrue(all(polygon.color_rgb == (68, 85, 102) for polygon in request.scene.polygons[1:]))
        self.assertTrue(all(polygon.edge_color_rgb == (17, 34, 51) for polygon in request.scene.polygons[1:]))
        self.assertTrue(all(abs(polygon.edge_linewidth - 1.75) < 1e-9 for polygon in request.scene.polygons))
        self.assertTrue(all(abs(polygon.alpha - 0.6) < 1e-9 for polygon in request.scene.polygons))

    def test_shape_specific_2d_alpha_overrides_shared_alpha(self) -> None:
        morpho_vis.configure_defaults(
            alpha_2d=0.6,
            alpha_2d_line=0.2,
            alpha_2d_poly=0.9,
        )
        backend = FakeBackend()

        line_request = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="line",
            chooser=BackendChooser(backends=(backend,)),
        )
        poly_request = plot2d(
            make_length_only_tree(),
            layout="stem",
            shape="frustum",
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertTrue(all(abs(polyline.alpha - 0.2) < 1e-9 for polyline in line_request.scene.polylines))
        self.assertTrue(all(abs(polygon.alpha - 0.9) < 1e-9 for polygon in poly_request.scene.polygons))

    def test_generic_branch_type_colors_also_drive_2d_palette(self) -> None:
        morpho_vis.configure_defaults(branch_type_colors={"soma": "#abcdef"})
        backend = FakeBackend()

        request = plot2d(make_node_tree(), shape="line", chooser=BackendChooser(backends=(backend,)))
        self.assertEqual(request.scene.polylines[0].color_rgb, (171, 205, 239))

    def test_theme_context_manager_restores_on_exception(self) -> None:
        original = morpho_vis.get_defaults().branch_type_colors["soma"]

        with self.assertRaises(RuntimeError):
            with morpho_vis.theme(branch_type_colors={"soma": "#abcdef"}):
                raise RuntimeError("boom")

        self.assertEqual(morpho_vis.get_defaults().branch_type_colors["soma"], original)


class Layout2DFamilyRegistryTest(VisDefaultsResetMixin, unittest.TestCase):
    """The registry is the single source of truth for 2D layout names.

    Before it existed, ``config.SUPPORTED_2D_LAYOUTS`` and the layout
    dispatcher's private set disagreed: ``configure(layout_2d_default=
    'trunk_first')`` raised while ``plot2d(layout='trunk_first')``
    worked.
    """

    def _plot2d_accepts(self, layout: str) -> bool:
        try:
            plot2d(
                make_node_tree(),
                layout=layout,
                shape="line",
                chooser=BackendChooser(backends=(FakeBackend(),)),
            )
        except ValueError as exc:
            # Only a *name* rejection counts; 'projected' can still fail
            # later on a morphology without 3D points.
            return "layout family" not in str(exc)
        return True

    def _configure_accepts(self, layout: str) -> bool:
        try:
            configure_defaults(layout_2d_default=layout)
        except ValueError:
            return False
        return True

    def test_both_entry_points_accept_exactly_the_registry_names(self) -> None:
        for layout in sorted(SUPPORTED_2D_LAYOUTS):
            with self.subTest(layout=layout):
                self.assertTrue(self._plot2d_accepts(layout), f"plot2d rejected {layout!r}")
                self.assertTrue(self._configure_accepts(layout), f"configure rejected {layout!r}")

    def test_both_entry_points_reject_the_same_unknown_name(self) -> None:
        self.assertFalse(self._plot2d_accepts("bogus"))
        self.assertFalse(self._configure_accepts("bogus"))

    def test_trunk_first_alias_is_accepted_by_both_entry_points(self) -> None:
        # The regression that motivated the registry.
        self.assertIn("trunk_first", SUPPORTED_2D_LAYOUTS)
        self.assertTrue(self._plot2d_accepts("trunk_first"))
        self.assertTrue(self._configure_accepts("trunk_first"))

    def test_supported_set_is_dispatched_names_plus_projected(self) -> None:
        self.assertEqual(SUPPORTED_2D_LAYOUTS - DISPATCHED_2D_LAYOUTS, {"projected"})
        self.assertEqual(LAYOUT_2D_DISPATCHED_NAMES, ("fan", "stem", "balloon", "radial_360"))

    def test_alias_map_covers_every_registered_alias(self) -> None:
        expected = {alias: family.name for family in LAYOUT_2D_FAMILIES for alias in family.aliases}
        self.assertEqual(LAYOUT_2D_ALIASES, expected)
        self.assertEqual(LAYOUT_2D_ALIASES["trunk_first"], "stem")

    def test_titles_resolve_for_names_aliases_and_unknowns(self) -> None:
        self.assertEqual(layout_2d_title("radial_360"), "Radial 360")
        self.assertEqual(layout_2d_title("stem"), "Stem")
        self.assertEqual(layout_2d_title("trunk_first"), "Stem")
        self.assertEqual(layout_2d_title("my_custom"), "My Custom")


class PaletteDecodeTest(unittest.TestCase):
    def test_rgb_to_float_scales_channels(self) -> None:
        self.assertEqual(rgb_to_float((255, 128, 0)), (1.0, 128.0 / 255.0, 0.0))

    def test_rgb_to_hex_is_lowercase_and_zero_padded(self) -> None:
        self.assertEqual(rgb_to_hex((255, 128, 0)), "#ff8000")
        self.assertEqual(rgb_to_hex((0, 0, 0)), "#000000")


if __name__ == "__main__":
    unittest.main()
