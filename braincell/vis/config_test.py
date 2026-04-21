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

import matplotlib as mpl

from braincell.vis.config import (
    PUBLICATION_BRANCH_TYPE_COLORS,
    PUBLICATION_RC_PARAMS,
    PublicationTheme,
    get_defaults,
    publication_theme,
    resolve_default_2d_layout,
    reset_defaults,
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


if __name__ == "__main__":
    unittest.main()
