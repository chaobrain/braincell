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
from unittest import mock

import brainunit as u
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from braincell._discretization.base import build_discretization
from braincell._discretization.node_build import build_node_tree_from_cvs as build_node_tree
from braincell._discretization import CVPerBranch
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology
from braincell.vis import plot_point_topology
from braincell.vis.point_topology import _blend_node_rgba, _resolve_coverage_intensities


def _make_node_tree(*, lengths: tuple[float, float, float] = (20.0, 80.0, 60.0)):
    soma = Branch.from_lengths(
        lengths=[lengths[0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend_a = Branch.from_lengths(
        lengths=[lengths[1]] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="dendrite",
    )
    dend_b = Branch.from_lengths(
        lengths=[lengths[2]] * u.um,
        radii=[2.5, 1.2] * u.um,
        type="dendrite",
    )
    morpho = Morphology.from_root(soma, name="soma")
    morpho.soma.a = dend_a
    morpho.soma.b = dend_b
    cvs = build_discretization(morpho, policy=CVPerBranch()).cvs
    return build_node_tree(morpho, cvs=cvs)


class PointTopologyPlotTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_point_topology_returns_axes(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_point_topology(node_tree, layout="kamada_kawai")
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_invalid_layout_scale_is_rejected(self) -> None:
        node_tree = _make_node_tree()
        for bad_value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "layout_scale must be a finite positive float"):
                    plot_point_topology(node_tree, layout="kamada_kawai", layout_scale=bad_value)

    def test_rejects_highlight_and_values_together(self) -> None:
        node_tree = _make_node_tree()
        with self.assertRaisesRegex(ValueError, "does not support highlight_point_ids together with values"):
            plot_point_topology(
                node_tree,
                highlight_point_ids=[1],
                values=np.linspace(0.0, 1.0, len(node_tree.nodes)),
            )

    def test_values_shape_must_match_n_points(self) -> None:
        node_tree = _make_node_tree()
        with self.assertRaisesRegex(ValueError, "values must have shape"):
            plot_point_topology(
                node_tree,
                layout="kamada_kawai",
                color_mode="values",
                values=np.ones(len(node_tree.nodes) - 1),
            )

    def test_graphviz_layout_falls_back_to_kamada_kawai(self) -> None:
        node_tree = _make_node_tree()
        with mock.patch(
            "braincell.vis.point_topology._graphviz_layout_positions",
            side_effect=RuntimeError("graphviz unavailable"),
        ):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                ax = plot_point_topology(node_tree, layout="twopi")
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertTrue(any("falling back to 'kamada_kawai'" in str(item.message) for item in captured))

    def test_same_topology_yields_same_layout_even_when_geometry_differs(self) -> None:
        node_tree_a = _make_node_tree(lengths=(20.0, 80.0, 60.0))
        node_tree_b = _make_node_tree(lengths=(50.0, 10.0, 200.0))

        ax_a = plot_point_topology(node_tree_a, layout="kamada_kawai")
        ax_b = plot_point_topology(node_tree_b, layout="kamada_kawai")

        coords_a = ax_a.collections[-1].get_offsets()
        coords_b = ax_b.collections[-1].get_offsets()
        np.testing.assert_allclose(coords_a, coords_b)

    def test_depth_color_mode_draws_all_points(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_point_topology(node_tree, preset="depth", layout="kamada_kawai")
        offsets = ax.collections[-1].get_offsets()
        self.assertEqual(offsets.shape[0], len(node_tree.nodes))

    def test_kamada_kawai_layout_scale_changes_point_spacing(self) -> None:
        node_tree = _make_node_tree()
        ax_small = plot_point_topology(node_tree, layout="kamada_kawai", layout_scale=0.5)
        ax_large = plot_point_topology(node_tree, layout="kamada_kawai", layout_scale=2.0)

        coords_small = np.asarray(ax_small.collections[-1].get_offsets(), dtype=float)
        coords_large = np.asarray(ax_large.collections[-1].get_offsets(), dtype=float)
        radius_small = float(np.max(np.linalg.norm(coords_small, axis=1)))
        radius_large = float(np.max(np.linalg.norm(coords_large, axis=1)))
        self.assertGreater(radius_large, radius_small)

    def test_graphviz_layout_scale_scales_returned_positions(self) -> None:
        node_tree = _make_node_tree()
        node_ids = tuple(point.id for point in node_tree.nodes)
        base_positions = {
            node_id: np.asarray((index + 1.0, (index + 1.0) * 2.0), dtype=float)
            for index, node_id in enumerate(node_ids)
        }
        with mock.patch(
            "braincell.vis.point_topology._graphviz_layout_positions",
            return_value=base_positions,
        ):
            ax = plot_point_topology(node_tree, layout="twopi", layout_scale=2.0)
        coords = np.asarray(ax.collections[-1].get_offsets(), dtype=float)
        expected = np.asarray([base_positions[node_id] * 2.0 for node_id in node_ids], dtype=float)
        np.testing.assert_allclose(coords, expected)

    def test_graphviz_fallback_preserves_layout_scale(self) -> None:
        node_tree = _make_node_tree()
        with mock.patch(
            "braincell.vis.point_topology._graphviz_layout_positions",
            side_effect=RuntimeError("graphviz unavailable"),
        ):
            with mock.patch(
                "braincell.vis.point_topology._kamada_kawai_positions",
                return_value={point.id: np.zeros(2, dtype=float) for point in node_tree.nodes},
            ) as mocked:
                plot_point_topology(node_tree, layout="twopi", layout_scale=2.5)
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.kwargs["layout_scale"], 2.5)

    def test_highlight_mode_draws_selected_overlay(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_point_topology(node_tree, highlight_point_ids=[3], layout="kamada_kawai")
        highlighted = np.asarray(ax.collections[-1].get_offsets(), dtype=float)
        self.assertEqual(highlighted.shape[0], 1)

    def test_values_mode_can_draw_colorbar(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_point_topology(
            node_tree,
            values=np.linspace(-70.0, -50.0, len(node_tree.nodes)),
            value_label="V",
            show_colorbar=True,
        )
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(len(ax.figure.axes), 2)

    def test_fraction_coverage_mode_preserves_fraction(self) -> None:
        intensities = _resolve_coverage_intensities(
            node_ids=(0, 1, 2),
            highlight_fractions={1: 0.5},
            coverage_mode="fraction",
        )
        np.testing.assert_allclose(intensities, np.asarray([0.0, 0.5, 0.0]))

    def test_any_coverage_mode_promotes_partial_overlap(self) -> None:
        intensities = _resolve_coverage_intensities(
            node_ids=(0, 1, 2),
            highlight_fractions={1: 0.2},
            coverage_mode="any",
        )
        np.testing.assert_allclose(intensities, np.asarray([0.0, 1.0, 0.0]))

    def test_all_coverage_mode_requires_full_overlap(self) -> None:
        intensities = _resolve_coverage_intensities(
            node_ids=(0, 1, 2),
            highlight_fractions={1: 0.999, 2: 1.0},
            coverage_mode="all",
        )
        np.testing.assert_allclose(intensities, np.asarray([0.0, 0.0, 1.0]))

    def test_invalid_coverage_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported coverage_mode"):
            _resolve_coverage_intensities(
                node_ids=(0, 1),
                highlight_fractions={0: 1.0},
                coverage_mode="bad",  # type: ignore[arg-type]
            )

    def test_blend_node_rgba_matches_expected_mix(self) -> None:
        base = np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=float)
        blended = _blend_node_rgba(
            base,
            highlight_color="#ff0000",
            intensities=np.asarray([0.5], dtype=float),
        )
        np.testing.assert_allclose(blended[0, :3], np.asarray([0.5, 0.0, 0.5]))


if __name__ == "__main__":
    unittest.main()
