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
import numpy as np

from braincell.vis import plot2d
from braincell.vis._testing import FakeBackend, VisDefaultsResetMixin, make_length_only_tree
from braincell.vis._values import ValueLayout, resolve_values, resolved_colorbar_label
from braincell.vis.backend import BackendChooser
from braincell.vis.scene import ValueSpec


class ResolveValuesTest(unittest.TestCase):
    def setUp(self) -> None:
        # make_length_only_tree has 2 branches:
        #   soma: 1 segment → 2 points
        #   dend: 2 segments → 3 points
        # Total segments = 3, total points = 5, n_branches = 2.
        self.tree = make_length_only_tree()

    def test_per_branch_array_broadcasts_to_points(self) -> None:
        spec = ValueSpec(values=np.array([0.1, 0.9]))
        per_branch, unit_label = resolve_values(self.tree, spec)

        self.assertIsNone(unit_label)
        self.assertEqual(len(per_branch), 2)
        soma_idx = self.tree.soma.index
        dend_idx = self.tree.soma.dend.index
        np.testing.assert_allclose(per_branch[soma_idx].point_values, [0.1, 0.1])
        np.testing.assert_allclose(per_branch[dend_idx].point_values, [0.9, 0.9, 0.9])

    def test_per_segment_array_interpolates_to_points(self) -> None:
        spec = ValueSpec(values=np.array([0.0, 1.0, 2.0]))
        per_branch, _ = resolve_values(self.tree, spec)

        # Soma has 1 segment → endpoints both = 0.0.
        soma_idx = self.tree.soma.index
        np.testing.assert_allclose(per_branch[soma_idx].point_values, [0.0, 0.0])
        # Dend has 2 segments with values (1, 2): point[0]=1, point[1]=avg=1.5, point[2]=2.
        dend_idx = self.tree.soma.dend.index
        np.testing.assert_allclose(per_branch[dend_idx].point_values, [1.0, 1.5, 2.0])

    def test_per_point_array_passes_through(self) -> None:
        spec = ValueSpec(values=np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        per_branch, _ = resolve_values(self.tree, spec)

        soma_idx = self.tree.soma.index
        dend_idx = self.tree.soma.dend.index
        np.testing.assert_allclose(per_branch[soma_idx].point_values, [10.0, 20.0])
        np.testing.assert_allclose(per_branch[dend_idx].point_values, [30.0, 40.0, 50.0])

    def test_mismatched_length_raises(self) -> None:
        spec = ValueSpec(values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
        with self.assertRaisesRegex(ValueError, "length 7"):
            resolve_values(self.tree, spec)

    def test_brainunit_array_strips_unit_label(self) -> None:
        spec = ValueSpec(values=np.array([0.1, 0.9]) * u.mV)
        per_branch, unit_label = resolve_values(self.tree, spec)
        self.assertIsNotNone(unit_label)
        self.assertIn("mV", unit_label or "")
        # Point values should be raw floats (no units).
        self.assertEqual(per_branch[self.tree.soma.index].point_values.dtype, np.float64)

    def test_non_1d_array_raises(self) -> None:
        spec = ValueSpec(values=np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "1-D"):
            resolve_values(self.tree, spec)


class ValueLayoutTest(unittest.TestCase):
    """The frame-invariant half of value resolution, reused by ``plot_movie``."""

    def setUp(self) -> None:
        # make_length_only_tree: soma 1 segment, dend 2 segments.
        self.tree = make_length_only_tree()
        self.layout = ValueLayout.from_morphology(self.tree)

    def test_shape_counts_match_the_morphology(self) -> None:
        self.assertEqual(self.layout.n_branches, 2)
        self.assertEqual(self.layout.segment_counts, (1, 2))
        self.assertEqual(self.layout.total_segments, 3)
        self.assertEqual(self.layout.total_points, 5)

    def test_branch_indices_follow_default_branch_order(self) -> None:
        self.assertEqual(self.layout.branch_indices, tuple(branch.index for branch in self.tree.branches))

    def test_expand_agrees_with_resolve_values_at_every_granularity(self) -> None:
        for values in (
            np.array([0.1, 0.9]),
            np.array([0.0, 1.0, 2.0]),
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        ):
            with self.subTest(length=len(values)):
                expected, _ = resolve_values(self.tree, ValueSpec(values=values))
                actual = self.layout.expand(values)
                self.assertEqual(set(actual), set(expected))
                for branch_index, branch_values in actual.items():
                    np.testing.assert_array_equal(
                        branch_values.point_values,
                        expected[branch_index].point_values,
                    )

    def test_one_layout_expands_many_frames(self) -> None:
        # plot_movie derives the layout once and calls expand per frame;
        # reuse must not leak state between frames.
        frames = np.array([[0.1, 0.9], [0.5, 0.25]])
        first = self.layout.expand(frames[0])
        second = self.layout.expand(frames[1])
        soma_idx = self.tree.soma.index
        np.testing.assert_allclose(first[soma_idx].point_values, [0.1, 0.1])
        np.testing.assert_allclose(second[soma_idx].point_values, [0.5, 0.5])

    def test_length_mismatch_names_all_three_accepted_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 branches, 3 segments, and 5 centerline points"):
            self.layout.expand(np.arange(7, dtype=float))

    def test_non_1d_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 1-D"):
            self.layout.expand(np.zeros((2, 5)))


class SegmentValuesPropertyTest(unittest.TestCase):
    def test_segment_values_from_per_point_are_mean_of_neighbours(self) -> None:
        from braincell.vis.scene import BranchValues

        values = BranchValues(
            branch_index=0,
            point_values=np.array([0.0, 2.0, 6.0, 8.0]),
        )
        np.testing.assert_allclose(values.segment_values, [1.0, 4.0, 7.0])

    def test_single_point_returns_copy(self) -> None:
        from braincell.vis.scene import BranchValues

        values = BranchValues(branch_index=0, point_values=np.array([3.14]))
        np.testing.assert_allclose(values.segment_values, [3.14])


class ColorbarLabelTest(unittest.TestCase):
    def test_label_and_unit_combine(self) -> None:
        spec = ValueSpec(values=np.array([0.0]), label="V_m", unit_label="mV")
        self.assertEqual(resolved_colorbar_label(spec, None), "V_m [mV]")

    def test_label_only(self) -> None:
        spec = ValueSpec(values=np.array([0.0]), label="Ca")
        self.assertEqual(resolved_colorbar_label(spec, None), "Ca")

    def test_unit_only(self) -> None:
        spec = ValueSpec(values=np.array([0.0]))
        self.assertEqual(resolved_colorbar_label(spec, "mV"), "[mV]")

    def test_neither_returns_none(self) -> None:
        spec = ValueSpec(values=np.array([0.0]))
        self.assertIsNone(resolved_colorbar_label(spec, None))


class ValueSpecLengthMismatchTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_plot2d_values_length_mismatch_raises(self) -> None:
        tree = make_length_only_tree()

        with self.assertRaisesRegex(ValueError, "ValueSpec.values has length"):
            plot2d(
                tree,
                layout="stem",
                shape="line",
                values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                chooser=BackendChooser(backends=(FakeBackend(),)),
            )


if __name__ == "__main__":
    unittest.main()
