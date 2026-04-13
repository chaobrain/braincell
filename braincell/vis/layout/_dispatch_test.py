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
import warnings

from braincell.vis._testing import make_length_only_tree
from braincell.vis.layout._dispatch import (
    _LAYOUT_FAMILY_ALIASES,
    _VALID_LAYOUT_FAMILIES,
    _VALID_ROOT_LAYOUTS,
    build_layout_branches_2d,
)


class ValidationTest(unittest.TestCase):
    def test_rejects_wrong_morphology_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects Morpho"):
            build_layout_branches_2d("not a morpho", mode="tree")  # type: ignore[arg-type]

    def test_rejects_unknown_mode(self) -> None:
        tree = make_length_only_tree()
        with self.assertRaisesRegex(ValueError, "Unsupported layout mode"):
            build_layout_branches_2d(tree, mode="bogus")

    def test_rejects_unknown_root_layout(self) -> None:
        tree = make_length_only_tree()
        with self.assertRaisesRegex(ValueError, "Unsupported root layout"):
            build_layout_branches_2d(tree, mode="tree", root_layout="bogus")

    def test_rejects_unknown_layout_family(self) -> None:
        tree = make_length_only_tree()
        with self.assertRaisesRegex(ValueError, "Unsupported 2D layout family"):
            build_layout_branches_2d(tree, mode="tree", layout_family="bogus")

    def test_valid_families_and_root_layouts_constants(self) -> None:
        # Defensive sanity check so renaming a family doesn't silently
        # break the dispatcher.
        self.assertIn("fan", _VALID_LAYOUT_FAMILIES)
        self.assertIn("stem", _VALID_LAYOUT_FAMILIES)
        self.assertIn("balloon", _VALID_LAYOUT_FAMILIES)
        self.assertIn("radial_360", _VALID_LAYOUT_FAMILIES)
        self.assertIn("trunk_first", _VALID_LAYOUT_FAMILIES)
        self.assertIn("type_split", _VALID_ROOT_LAYOUTS)
        self.assertIn("legacy", _VALID_ROOT_LAYOUTS)
        self.assertEqual(_LAYOUT_FAMILY_ALIASES["trunk_first"], "stem")


class DispatchSmokeTest(unittest.TestCase):
    def test_dispatches_to_default_layout_by_default(self) -> None:
        tree = make_length_only_tree()
        layouts = build_layout_branches_2d(tree, mode="tree")
        self.assertEqual(len(layouts), len(tree.branches))

    def test_dispatches_to_fan(self) -> None:
        tree = make_length_only_tree()
        layouts = build_layout_branches_2d(tree, mode="tree", layout_family="fan")
        self.assertEqual(len(layouts), len(tree.branches))

    def test_trunk_first_is_alias_for_stem(self) -> None:
        tree = make_length_only_tree()
        stem_layouts = build_layout_branches_2d(tree, mode="tree", layout_family="stem")
        alias_layouts = build_layout_branches_2d(tree, mode="tree", layout_family="trunk_first")
        self.assertEqual(len(stem_layouts), len(alias_layouts))
        for a, b in zip(stem_layouts, alias_layouts):
            self.assertEqual(a.branch_name, b.branch_name)

    def test_dispatches_to_balloon(self) -> None:
        tree = make_length_only_tree()
        layouts = build_layout_branches_2d(tree, mode="tree", layout_family="balloon")
        self.assertEqual(len(layouts), len(tree.branches))

    def test_dispatches_to_radial_360(self) -> None:
        tree = make_length_only_tree()
        layouts = build_layout_branches_2d(tree, mode="tree", layout_family="radial_360")
        self.assertEqual(len(layouts), len(tree.branches))

    def test_dispatches_to_legacy(self) -> None:
        tree = make_length_only_tree()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            layouts = build_layout_branches_2d(tree, mode="tree", root_layout="legacy")
        self.assertEqual(len(layouts), len(tree.branches))

    def test_legacy_root_layout_emits_deprecation_warning(self) -> None:
        tree = make_length_only_tree()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_layout_branches_2d(tree, mode="tree", root_layout="legacy")
        deprecations = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        self.assertIn("legacy", str(deprecations[0].message))
        self.assertIn("v0.1.0", str(deprecations[0].message))

    def test_type_split_root_layout_does_not_warn(self) -> None:
        tree = make_length_only_tree()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_layout_branches_2d(tree, mode="tree", root_layout="type_split")
        deprecations = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
        self.assertEqual(deprecations, [])

    def test_frustum_mode_uses_linear_stem(self) -> None:
        tree = make_length_only_tree()
        layouts = build_layout_branches_2d(tree, mode="frustum")
        self.assertEqual(len(layouts), len(tree.branches))


if __name__ == "__main__":
    unittest.main()
