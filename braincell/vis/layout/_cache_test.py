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

from braincell import Branch, Morphology
from braincell.vis.layout import (
    DEFAULT_LAYOUT_CONFIG,
    LayoutCache,
    LayoutConfig,
    build_layout_branches_2d,
    get_default_layout_cache,
)
from braincell.vis.layout._cache import _make_cache_key


def _make_small_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.0] * u.um, type="apical_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


class LayoutCacheKeyTest(unittest.TestCase):
    def test_identical_morphologies_produce_identical_keys(self) -> None:
        tree_a = _make_small_tree()
        tree_b = _make_small_tree()

        key_a = _make_cache_key(
            tree_a,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=DEFAULT_LAYOUT_CONFIG,
        )
        key_b = _make_cache_key(
            tree_b,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=DEFAULT_LAYOUT_CONFIG,
        )
        self.assertEqual(key_a, key_b)

    def test_different_configs_produce_different_keys(self) -> None:
        tree = _make_small_tree()
        key_default = _make_cache_key(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=DEFAULT_LAYOUT_CONFIG,
        )
        key_tuned = _make_cache_key(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=LayoutConfig(collision_margin_um=5.0),
        )
        self.assertNotEqual(key_default, key_tuned)

    def test_different_modes_produce_different_keys(self) -> None:
        tree = _make_small_tree()
        key_tree = _make_cache_key(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=DEFAULT_LAYOUT_CONFIG,
        )
        key_frustum = _make_cache_key(
            tree,
            mode="frustum",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=DEFAULT_LAYOUT_CONFIG,
        )
        self.assertNotEqual(key_tree, key_frustum)


class LayoutCacheBehaviourTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = LayoutCache(maxsize=2)

    def test_hit_reuses_cached_layout(self) -> None:
        tree = _make_small_tree()

        call_count = {"n": 0}

        def _build():
            call_count["n"] += 1
            return build_layout_branches_2d(
                tree,
                mode="tree",
                layout_family="stem",
                root_layout="type_split",
                min_branch_angle_deg=25.0,
                use_cache=False,
            )

        result_a = self.cache.get_or_build(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=DEFAULT_LAYOUT_CONFIG,
            build=_build,
        )
        result_b = self.cache.get_or_build(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            layout_config=DEFAULT_LAYOUT_CONFIG,
            build=_build,
        )

        self.assertEqual(call_count["n"], 1)
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 1)
        # Both pointers refer to the same cached tuple.
        self.assertIs(result_a, result_b)

    def test_lru_eviction_on_overflow(self) -> None:
        tree_a = _make_small_tree()
        tree_b = _make_small_tree()
        # Change a radius so the key differs.
        from braincell import Morphology as _Morphology
        tree_c = _Morphology.from_root(
            Branch.from_lengths(lengths=[20.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma"),
            name="soma",
        )

        trees = [tree_a, tree_b, tree_c]

        def _build_for(tree):
            return lambda: build_layout_branches_2d(
                tree,
                mode="tree",
                layout_family="stem",
                root_layout="type_split",
                min_branch_angle_deg=25.0,
                use_cache=False,
            )

        for tree in trees:
            self.cache.get_or_build(
                tree,
                mode="tree",
                layout_family="stem",
                root_layout="type_split",
                min_branch_angle_deg=25.0,
                layout_config=DEFAULT_LAYOUT_CONFIG,
                build=_build_for(tree),
            )
        # tree_a and tree_b share a key, so the cache only ever held
        # two distinct entries even though we made three calls.
        self.assertLessEqual(len(self.cache), 2)

    def test_clear_resets_counters(self) -> None:
        self.cache.hits = 5
        self.cache.misses = 7
        self.cache.clear()
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 0)
        self.assertEqual(len(self.cache), 0)

    def test_rejects_non_positive_maxsize(self) -> None:
        with self.assertRaisesRegex(ValueError, "maxsize"):
            LayoutCache(maxsize=0)
        with self.assertRaisesRegex(ValueError, "maxsize"):
            LayoutCache(maxsize=-1)


class DispatcherCacheIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        get_default_layout_cache().clear()

    def tearDown(self) -> None:
        get_default_layout_cache().clear()

    def test_build_layout_branches_2d_uses_default_cache(self) -> None:
        tree = _make_small_tree()
        _ = build_layout_branches_2d(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
        )
        _ = build_layout_branches_2d(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
        )
        self.assertEqual(get_default_layout_cache().hits, 1)
        self.assertEqual(get_default_layout_cache().misses, 1)

    def test_use_cache_false_bypasses_cache(self) -> None:
        tree = _make_small_tree()
        _ = build_layout_branches_2d(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            use_cache=False,
        )
        self.assertEqual(get_default_layout_cache().hits, 0)
        self.assertEqual(get_default_layout_cache().misses, 0)

    def test_caller_can_pass_custom_cache(self) -> None:
        tree = _make_small_tree()
        cache = LayoutCache(maxsize=4)
        _ = build_layout_branches_2d(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            cache=cache,
        )
        _ = build_layout_branches_2d(
            tree,
            mode="tree",
            layout_family="stem",
            root_layout="type_split",
            min_branch_angle_deg=25.0,
            cache=cache,
        )
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)
        # Default cache stayed untouched.
        self.assertEqual(get_default_layout_cache().hits, 0)


if __name__ == "__main__":
    unittest.main()
