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

"""Hypothesis-driven property tests for the 2D layout families.

These tests sample small random trees and assert invariants that
should hold for every layout family:

1. One ``LayoutBranch2D`` per morphology branch.
2. All segment coordinates are finite.
3. Layout total length matches the branch's total length in µm.
4. No two non-adjacent branches have a proper segment intersection.

Property (4) is the non-trivial one — it is the primary reason the
stem family has a collision-aware scoring loop. Adjacent branches
(parent/child) are allowed to share an endpoint, which is why the
assertion is on *proper* intersections only.

The whole module is skipped when :mod:`hypothesis` is not installed,
so the base test suite does not pick up an extra dependency.
"""

import importlib.util
import unittest

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.vis.layout import LayoutBranch2D, build_layout_branches_2d
from braincell.vis.layout._collision import _segments_intersect

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


_LAYOUT_FAMILIES = ("stem", "balloon", "radial_360")
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
            st.integers(min_value=1, max_value=3),      # n_segments
            st.floats(min_value=5.0, max_value=30.0),   # seg_len_um
            st.floats(min_value=0.5, max_value=3.0),    # radius_um
        )

        # Tighter spec for intersection tests: single-segment children
        # of roughly uniform length. With 2-3 same-length children
        # radiating from a shared attach point, balloon and radial
        # must produce non-crossing layouts — anything else is a
        # real regression in the angle-assignment logic.
        _uniform_child_spec_strategy = st.tuples(
            st.just(1),                                  # n_segments
            st.floats(min_value=10.0, max_value=20.0),   # seg_len_um
            st.just(1.0),                                # radius_um
        )

        @given(
            n_children=st.integers(min_value=2, max_value=5),
            child_specs=st.lists(_wide_child_spec_strategy, min_size=2, max_size=5),
        )
        @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
        def test_layouts_preserve_branch_count_and_length(self, n_children, child_specs):
            tree = _build_random_tree(n_children=n_children, child_specs=child_specs)
            expected_lengths = {
                branch.index: float(sum(branch.lengths.to_decimal(u.um)))
                for branch in tree.branches
            }
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


if __name__ == "__main__":
    unittest.main()
