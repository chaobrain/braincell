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


import importlib.util
import sys
import unittest

import pytest
import warnings

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.vis._testing import (
    make_deep_chain_tree,
    make_length_only_tree,
    make_two_dendrite_tree,
    needs_benchmark,
)
from braincell.vis.layout import LayoutBranch2D
from braincell.vis.layout._collision import _segments_intersect
from braincell.vis.config import DISPATCHED_2D_LAYOUTS, LAYOUT_2D_ALIASES
from braincell.vis.layout._dispatch import (
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
        self.assertIn("fan", DISPATCHED_2D_LAYOUTS)
        self.assertIn("stem", DISPATCHED_2D_LAYOUTS)
        self.assertIn("balloon", DISPATCHED_2D_LAYOUTS)
        self.assertIn("radial_360", DISPATCHED_2D_LAYOUTS)
        self.assertIn("trunk_first", DISPATCHED_2D_LAYOUTS)
        self.assertIn("type_split", _VALID_ROOT_LAYOUTS)
        self.assertIn("legacy", _VALID_ROOT_LAYOUTS)
        self.assertEqual(LAYOUT_2D_ALIASES["trunk_first"], "stem")

    def test_projected_is_not_a_dispatchable_family(self) -> None:
        # ``layout='projected'`` is intercepted by build_render_scene_2d,
        # so it must stay out of the dispatcher's accepted set even
        # though ``plot2d(layout='projected')`` is valid.
        tree = make_length_only_tree()
        self.assertNotIn("projected", DISPATCHED_2D_LAYOUTS)
        with self.assertRaisesRegex(ValueError, "Unsupported 2D layout family"):
            build_layout_branches_2d(tree, mode="tree", layout_family="projected")


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


class DeepMorphologyTest(unittest.TestCase):
    """Every layout family must lay out chains deeper than the frame limit.

    Reconstructions of long neurites are routinely thousands of branches
    deep. Each family used to descend with one interpreter frame per
    branch, so anything past a few hundred branches raised
    ``RecursionError`` — see the builders in ``_stem`` / ``_balloon`` /
    ``_fan`` / ``_radial`` / ``_legacy``, which all walk iteratively now.
    """

    # One entry per distinct builder reachable through the dispatcher.
    COMBINATIONS = (
        ("tree", "stem", "type_split"),
        ("frustum", "stem", "type_split"),
        ("tree", "balloon", "type_split"),
        ("tree", "radial_360", "type_split"),
        ("tree", "fan", "type_split"),
        ("tree", "stem", "legacy"),
    )

    def test_every_family_handles_a_chain_deeper_than_the_recursion_limit(self) -> None:
        n_branches = sys.getrecursionlimit() * 2
        tree = make_deep_chain_tree(n_branches)

        for mode, layout_family, root_layout in self.COMBINATIONS:
            with self.subTest(mode=mode, layout_family=layout_family, root_layout=root_layout):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    layouts = build_layout_branches_2d(
                        tree,
                        mode=mode,
                        layout_family=layout_family,
                        root_layout=root_layout,
                        use_cache=False,
                    )
                self.assertEqual(len(layouts), n_branches)
                # A branch left unplaced would surface as a KeyError while
                # building the result tuple, so reaching here means every
                # branch in the chain was visited.
                self.assertEqual(layouts[0].branch_name, "soma")


# =============================================================================
# Hypothesis-driven invariants across every layout family
#
# Property 4 (no proper intersection between non-adjacent branches) is the
# non-trivial one: it is the reason the stem family carries a collision-aware
# scoring loop. Skipped wholesale when hypothesis is not installed.
# =============================================================================


_hypothesis_available = importlib.util.find_spec("hypothesis") is not None

if _hypothesis_available:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
else:  # pragma: no cover - skip path
    st = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]
    HealthCheck = None  # type: ignore[assignment]

    def given(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


_LAYOUT_FAMILIES = ("fan", "stem", "balloon", "radial_360")
_MODES = ("tree", "frustum")


def _build_random_tree(
    *,
    n_children: int,
    child_specs: list[tuple[int, float, float]],
) -> Morphology:
    """Build a small tree from a hypothesis sample.

    ``child_specs`` is a list of (n_segments, seg_len_um, radius_um)
    tuples — one per child dendrite. All children attach to the soma
    at ``parent_x=1.0`` so we avoid degenerate start-to-start overlaps.
    """
    soma = Branch.from_lengths(lengths=[15.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
    for child_index, (n_segments, seg_len_um, radius_um) in enumerate(child_specs[:n_children]):
        lengths_um = [float(seg_len_um)] * n_segments
        radii_um = [float(radius_um)] * (n_segments + 1)
        tree.attach(
            parent="soma",
            child_branch=Branch.from_lengths(
                lengths=lengths_um * u.um,
                radii=radii_um * u.um,
                type="basal_dendrite",
            ),
            child_name=f"dend_{child_index}",
            parent_x=1.0,
        )
    return tree


def _any_proper_intersection(layouts: tuple[LayoutBranch2D, ...]) -> bool:
    """Return True if any pair of non-adjacent branches has a proper
    segment intersection."""
    branch_indices = [layout.branch_index for layout in layouts]
    for i, layout_a in enumerate(layouts):
        for j in range(i + 1, len(layouts)):
            layout_b = layouts[j]
            # Same-index cannot happen; skip parent/child sharing an
            # endpoint is fine because _segments_intersect ignores
            # shared endpoints via strict "< 0.0" orientation product.
            pts_a = np.asarray(layout_a.segment_points_um, dtype=float)
            pts_b = np.asarray(layout_b.segment_points_um, dtype=float)
            for seg_a in range(len(pts_a) - 1):
                for seg_b in range(len(pts_b) - 1):
                    if _segments_intersect(
                        pts_a[seg_a],
                        pts_a[seg_a + 1],
                        pts_b[seg_b],
                        pts_b[seg_b + 1],
                    ):
                        return True
    _ = branch_indices  # silence unused
    return False


@unittest.skipUnless(_hypothesis_available, "hypothesis is not installed")
class LayoutFamilyPropertyTest(unittest.TestCase):
    """Run each layout family against a battery of random small trees.

    Uses a modest ``max_examples`` so the suite stays under a second
    while still catching regressions in the angle-assignment logic.
    """

    if _hypothesis_available:
        # Wider spec for invariant tests (count, length, finiteness):
        # these properties don't care about the geometric quality of
        # the layout, so we can throw a lot of variance at them.
        _wide_child_spec_strategy = st.tuples(
            st.integers(min_value=1, max_value=3),  # n_segments
            st.floats(min_value=5.0, max_value=30.0),  # seg_len_um
            st.floats(min_value=0.5, max_value=3.0),  # radius_um
        )

        # Tighter spec for intersection tests: single-segment children
        # of roughly uniform length. With 2-3 same-length children
        # radiating from a shared attach point, balloon and radial
        # must produce non-crossing layouts — anything else is a
        # real regression in the angle-assignment logic.
        _uniform_child_spec_strategy = st.tuples(
            st.just(1),  # n_segments
            st.floats(min_value=10.0, max_value=20.0),  # seg_len_um
            st.just(1.0),  # radius_um
        )

        @given(
            n_children=st.integers(min_value=2, max_value=5),
            child_specs=st.lists(_wide_child_spec_strategy, min_size=2, max_size=5),
        )
        @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
        def test_layouts_preserve_branch_count_and_length(self, n_children, child_specs):
            tree = _build_random_tree(n_children=n_children, child_specs=child_specs)
            expected_lengths = {branch.index: float(sum(branch.lengths.to_decimal(u.um))) for branch in tree.branches}
            for family in _LAYOUT_FAMILIES:
                for mode in _MODES:
                    layouts = build_layout_branches_2d(
                        tree,
                        mode=mode,
                        layout_family=family,
                    )
                    self.assertEqual(len(layouts), len(tree.branches))
                    for layout in layouts:
                        self.assertTrue(np.all(np.isfinite(layout.segment_points_um)))
                        self.assertAlmostEqual(
                            layout.total_length_um,
                            expected_lengths[layout.branch_index],
                            places=5,
                        )

        @given(
            n_children=st.integers(min_value=2, max_value=3),
            child_specs=st.lists(_uniform_child_spec_strategy, min_size=2, max_size=3),
        )
        @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
        def test_uniform_children_have_no_proper_intersections(self, n_children, child_specs):
            # With 2-3 identical-shape children of a single segment
            # each, every layout family (stem / balloon / radial_360)
            # should produce non-crossing strokes: each child is a
            # single radius and they fan out at distinct angles from
            # a shared root.
            tree = _build_random_tree(n_children=n_children, child_specs=child_specs)
            for family in _LAYOUT_FAMILIES:
                layouts = build_layout_branches_2d(
                    tree,
                    mode="tree",
                    layout_family=family,
                )
                self.assertFalse(
                    _any_proper_intersection(layouts),
                    msg=f"{family} produced a proper intersection for {child_specs!r}",
                )


class LayoutFamilyParametricTest(unittest.TestCase):
    """Shared invariants across all non-legacy layout families.

    Parametrized over (family) × (mode) so that adding a new layout family
    automatically picks up the whole test matrix by appending one name.
    """

    families = ("fan", "stem", "balloon", "radial_360")
    modes = ("tree", "frustum")

    def test_all_layouts_produce_one_entry_per_branch(self) -> None:
        tree = make_two_dendrite_tree()
        expected_count = len(tree.branches)
        for family in self.families:
            for mode in self.modes:
                with self.subTest(family=family, mode=mode):
                    layouts = build_layout_branches_2d(tree, mode=mode, layout_family=family)
                    self.assertEqual(len(layouts), expected_count)

    def test_all_layouts_produce_finite_coordinates(self) -> None:
        tree = make_length_only_tree()
        for family in self.families:
            for mode in self.modes:
                with self.subTest(family=family, mode=mode):
                    layouts = build_layout_branches_2d(tree, mode=mode, layout_family=family)
                    for layout in layouts:
                        self.assertTrue(np.all(np.isfinite(layout.segment_points_um)))
                        self.assertTrue(np.all(np.isfinite(layout.segment_directions_um)))
                        self.assertTrue(np.all(np.isfinite(layout.cumulative_lengths_um)))

    def test_all_layouts_preserve_total_length(self) -> None:
        tree = make_length_only_tree()
        expected_lengths = {
            branch.index: float(np.sum(np.asarray(branch.lengths.to_decimal(u.um), dtype=float)))
            for branch in tree.branches
        }
        for family in self.families:
            for mode in self.modes:
                with self.subTest(family=family, mode=mode):
                    layouts = build_layout_branches_2d(tree, mode=mode, layout_family=family)
                    for layout in layouts:
                        self.assertAlmostEqual(
                            layout.total_length_um,
                            expected_lengths[layout.branch_index],
                            places=6,
                        )


# ---------------------------------------------------------------------------
# Performance baselines (pytest-benchmark). See ``needs_benchmark`` in
# braincell/vis/_testing.py for why these are plain functions; the
# ``clean_layout_cache`` fixture comes from braincell/vis/conftest.py.
# ---------------------------------------------------------------------------


# The medium / large sizes are chains deeper than CPython's recursion limit.
# They used to be marked xfail(RecursionError) because every tree walk in the
# layout engine recursed once per branch; the walks are iterative now (see
# braincell/vis/_traversal.py), so these must simply pass. DeepMorphologyTest
# above pins that behaviour for runs where pytest-benchmark is absent.
@needs_benchmark
@pytest.mark.parametrize("n_branches", [50, 500, 2000], ids=["small", "medium", "large"])
def test_layout_build(benchmark, clean_layout_cache, n_branches: int) -> None:
    tree = make_deep_chain_tree(n_branches)
    benchmark(lambda: build_layout_branches_2d(tree, mode="tree", use_cache=False))


if __name__ == "__main__":
    unittest.main()
