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

from braincell import Branch, Morpho, MorphoBranch


class MorphoTest(unittest.TestCase):
    def test_tree_topology_queries_and_branch_views(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(
            lengths=np.array([100.0]) * u.um,
            radii=np.array([2.0, 1.0]) * u.um,
            type="basal_dendrite",
        )
        axon = Branch.from_lengths(lengths=[40.0] * u.um, radii=[0.8, 0.4] * u.um, type="axon")

        tree = Morpho.from_root(soma, name="soma")
        dend_view = tree.soma.attach(dend, name="dendrite", parent_x=1.0)
        axon_view = tree.attach(parent=tree.soma, child_branch=axon, child_name=None, parent_x=0.5, child_x=1.0)
        tree.soma.extra = Branch.from_lengths(
            lengths=[30.0] * u.um,
            radii=[1.0, 0.6] * u.um,
            type="apical_dendrite",
        )

        self.assertIsInstance(tree.soma, MorphoBranch)
        self.assertIsNone(tree.soma.parent)
        self.assertIsNone(tree.soma.parent_id)
        self.assertEqual(dend_view.parent.name, "soma")
        self.assertEqual(dend_view.parent_id, 0)
        self.assertEqual(dend_view.parent_x, 1.0)
        self.assertEqual(axon_view.child_x, 1.0)
        self.assertEqual(tree.branch(index=1).name, "dendrite")
        self.assertEqual(tree.branch(name="axon_0").parent.name, "soma")
        self.assertEqual(tree.soma.dendrite.name, "dendrite")
        self.assertEqual(tree.soma.type, "soma")
        self.assertEqual(tree.soma.axon_0.name, "axon_0")
        self.assertEqual(tree.soma.n_children, 3)
        self.assertEqual(tree.path_to_root(2), (0, 2))
        self.assertEqual(len(tree.branches), 4)
        self.assertEqual(len(tree.edges), 3)
        self.assertEqual(tree.edges[0].parent_x, 1.0)
        self.assertEqual(tree.edges[0].child_x, 0.0)
        self.assertEqual(tree.edges[1].child_x, 1.0)
        self.assertEqual(tree.soma.length.to_decimal(u.um), 20.0)
        self.assertEqual(tree.soma.radii_proximal[0].to_decimal(u.um), 10.0)
        self.assertEqual(tree.soma.radii_distal[-1].to_decimal(u.um), 10.0)
        self.assertEqual(
            tree.topo(),
            "\n".join(
                (
                    "soma",
                    "├── dendrite",
                    "├── axon_0",
                    "└── extra",
                )
            ),
        )

    def test_attach_by_name_and_attachment_point(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        branch = tree.soma[0.5, 1.0].attach(dend, name="dendrite")

        self.assertEqual(branch.name, "dendrite")
        self.assertEqual(branch.parent_x, 0.5)
        self.assertEqual(branch.child_x, 1.0)
        self.assertEqual(tree.edges[0].child_x, 1.0)

    def test_child_x_accepts_only_endpoint_values(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")

        child0 = tree.soma[0.0, 0].attach(dend, name="d0")
        child1 = tree.soma.attach(dend, name="d1", child_x=1.0)

        self.assertEqual(child0.parent_x, 0.0)
        self.assertEqual(child0.child_x, 0.0)
        self.assertEqual(child1.parent_x, 1.0)
        self.assertEqual(child1.child_x, 1.0)

        midpoint = tree.soma.attach(dend, name="d_mid", parent_x=0.5)
        self.assertEqual(midpoint.parent_x, 0.5)

        for invalid in (0.2, 0.3, 0.4, 0.7, 0.9, -1, 2):
            with self.subTest(parent_x=invalid):
                with self.assertRaises(ValueError):
                    tree.soma.attach(dend, parent_x=invalid)

        for invalid in (0.4, -1, 2):
            with self.subTest(child_x=invalid):
                with self.assertRaises(ValueError):
                    tree.soma.attach(dend, child_x=invalid)

        for invalid in (True, False):
            with self.subTest(parent_x=invalid):
                with self.assertRaises(TypeError):
                    tree.soma.attach(dend, parent_x=invalid)

        for invalid in (True, False):
            with self.subTest(child_x=invalid):
                with self.assertRaises(TypeError):
                    tree.soma.attach(dend, child_x=invalid)

    def test_topo_renders_nested_tree(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tuft = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 0.6] * u.um, type="apical_dendrite")
        axon = Branch.from_lengths(lengths=[40.0] * u.um, radii=[0.8, 0.4] * u.um, type="axon")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.dend.tuft = tuft
        tree.soma.axon = axon

        self.assertEqual(
            tree.topo(),
            "\n".join(
                (
                    "soma",
                    "├── dend",
                    "│   └── tuft",
                    "└── axon",
                )
            ),
        )

    def test_auto_names_apply_only_when_explicit_name_is_missing(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="dendrite")

        tree = Morpho.from_root(soma, name="soma")
        explicit = tree.soma.attach(dend, name="first")
        auto0 = tree.soma.attach(dend)
        auto1 = tree.soma.attach(Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="dendrite"))

        self.assertEqual(explicit.name, "first")
        self.assertEqual(auto0.name, "dendrite_0")
        self.assertEqual(auto1.name, "dendrite_1")

    def test_root_can_opt_into_type_based_auto_naming(self) -> None:
        axon = Branch.from_lengths(lengths=[40.0] * u.um, radii=[0.8, 0.4] * u.um, type="axon")

        tree = Morpho.from_root(axon, name=None)

        self.assertEqual(tree.root.name, "axon_0")
        self.assertEqual(tree.topo(), "axon_0")

    def test_branch_order_queries_are_available(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        axon = Branch.from_lengths(lengths=[40.0] * u.um, radii=[0.8, 0.4] * u.um, type="axon")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="dendrite")
        apical = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.2, 0.8] * u.um, type="apical_dendrite")
        custom = Branch.from_lengths(lengths=[25.0] * u.um, radii=[0.7, 0.5] * u.um, type="custom")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.d = dend
        tree.soma.a = axon
        tree.soma.t = apical
        tree.soma.c = custom

        self.assertEqual(tuple(branch.name for branch in tree.branches), ("soma", "d", "a", "t", "c"))
        self.assertEqual(tuple(branch.name for branch in tree.branch_by_order(order="default")), ("soma", "d", "a", "t", "c"))
        self.assertEqual(tuple(branch.name for branch in tree.branch_by_order(order="type")), ("soma", "a", "d", "t", "c"))
        self.assertEqual(tree.branch(index=1).name, "d")
        self.assertEqual(tree.branch(index=1, order="type").name, "a")
        self.assertEqual(tree.branch(name="a").name, "a")
        self.assertEqual(tree.soma.index, 0)
        self.assertEqual(tree.soma.index_by(order="type"), 0)

        with self.assertRaises(TypeError):
            tree.branch()
        with self.assertRaises(TypeError):
            tree.branch(name="soma", index=0)
        with self.assertRaises(TypeError):
            tree.branch(name="soma", order="type")
        with self.assertRaises(ValueError):
            tree.branch_by_order(order="unknown")

    def test_summary_exposes_compact_tree_metrics(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        summary = tree.summary()

        self.assertEqual(summary["root_name"], "soma")
        self.assertEqual(summary["root_type"], "soma")
        self.assertEqual(summary["n_branches"], tree.n_branches)
        self.assertEqual(summary["n_stems"], tree.n_stems)
        self.assertEqual(summary["n_bifurcations"], tree.n_bifurcations)
        self.assertEqual(summary["max_branch_order"], tree.max_branch_order)
        self.assertEqual(summary["total_length"], tree.total_length)
        self.assertEqual(summary["total_area"], tree.total_area)
        self.assertEqual(summary["total_volume"], tree.total_volume)
        self.assertEqual(summary["mean_radius"], tree.mean_radius)
        self.assertFalse(tree.has_full_point_geometry)
        self.assertFalse(summary["has_full_point_geometry"])
        self.assertNotIn("has_point_geometry", summary)
        self.assertNotIn("has_full_point_geometry_for_distance_metrics", summary)

    def test_summary_reports_point_geometry_capabilities(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um,
                                  type="soma")
        dend = Branch.from_points(points=[(5.0, 0.0, 0.0), (5.0, 10.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um,
                                  type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.main = dend

        summary = tree.summary()

        self.assertTrue(tree.has_full_point_geometry)
        self.assertTrue(summary["has_full_point_geometry"])

    def test_summary_reports_partial_point_geometry_as_false(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[20.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.main = dend

        summary = tree.summary()

        self.assertFalse(tree.has_full_point_geometry)
        self.assertFalse(summary["has_full_point_geometry"])

    def test_morpho_equality_compares_structure_and_geometry(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        axon = Branch.from_lengths(lengths=[40.0] * u.um, radii=[0.8, 0.4] * u.um, type="axon")

        tree0 = Morpho.from_root(soma, name="soma")
        tree0.soma.attach(dend, name="dendrite", parent_x=1.0)
        tree0.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=0.5, child_x=1.0)

        tree1 = Morpho.from_root(soma, name="soma")
        tree1.soma.dendrite = dend
        tree1.soma[0.5, 1.0].axon = axon

        self.assertEqual(tree0, tree1)

        renamed = Morpho.from_root(soma, name="soma")
        renamed.soma.attach(dend, name="d_other", parent_x=1.0)
        renamed.soma[0.5, 1.0].axon = axon
        self.assertNotEqual(tree0, renamed)

        shifted = Morpho.from_root(soma, name="soma")
        shifted.soma.attach(dend, name="dendrite", parent_x=0.0)
        shifted.soma[0.5, 1.0].axon = axon
        self.assertNotEqual(tree0, shifted)

        other_geom = Morpho.from_root(soma, name="soma")
        other_geom.soma.attach(
            Branch.from_lengths(lengths=[61.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite"),
            name="dendrite",
            parent_x=1.0,
        )
        other_geom.soma[0.5, 1.0].axon = axon
        self.assertNotEqual(tree0, other_geom)

        self.assertFalse(tree0 == object())
        with self.assertRaises(TypeError):
            hash(tree0)

    def test_parent_x_midpoint_is_soma_only(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tuft = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 0.6] * u.um, type="apical_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        with self.assertRaises(ValueError):
            tree.dend.attach(tuft, parent_x=0.5)

        with self.assertRaises(ValueError):
            tree.dend[0.5].attach(tuft, name="tuft")

    def test_metric_exposes_tree_level_metrics_with_compatible_shortcuts(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        tuft = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 0.6] * u.um, type="apical_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.dend.tuft = tuft

        self.assertEqual(tree.metric.n_branches, 3)
        self.assertEqual(tree.n_branches, 3)
        self.assertEqual(tree.metric.n_stems, 1)
        self.assertEqual(tree.n_stems, 1)
        self.assertEqual(tree.metric.n_bifurcations, 0)
        self.assertEqual(tree.n_bifurcations, 0)
        self.assertEqual(tree.metric.max_branch_order, 2)
        self.assertEqual(tree.max_branch_order, 2)
        self.assertEqual(tree.soma.branch, soma)
        self.assertEqual(tree.soma.mean_radius.to_decimal(u.um), 10.0)
        self.assertEqual(tree.soma.n_children, 1)
        self.assertEqual(tree.soma.dend.mean_radius.to_decimal(u.um), 1.5)
        self.assertEqual(tree.soma.dend.n_children, 1)
        self.assertAlmostEqual(tree.soma.dend.tuft.mean_radius.to_decimal(u.um), 0.8)
        self.assertEqual(tree.soma.dend.tuft.n_children, 0)
        self.assertEqual(tree.metric.path_to_root(2), (0, 1, 2))
        self.assertEqual(tree.path_to_root(2), (0, 1, 2))
        self.assertEqual(tree.metric.max_path_distance.to_decimal(u.um), 110.0)
        self.assertEqual(tree.max_path_distance.to_decimal(u.um), 110.0)
        self.assertAlmostEqual(tree.metric.mean_radius.to_decimal(u.um), 314.0 / 110.0)
        self.assertAlmostEqual(tree.mean_radius.to_decimal(u.um), 314.0 / 110.0)
        self.assertEqual(tree.metric.total_length.to_decimal(u.um), 110.0)
        self.assertEqual(tree.total_length.to_decimal(u.um), 110.0)
        self.assertAlmostEqual(
            tree.metric.total_area.to_decimal(u.um ** 2),
            (soma.area + dend.area + tuft.area).to_decimal(u.um ** 2),
        )
        self.assertAlmostEqual(
            tree.total_area.to_decimal(u.um ** 2),
            tree.metric.total_area.to_decimal(u.um ** 2),
        )
        self.assertAlmostEqual(
            tree.metric.total_volume.to_decimal(u.um ** 3),
            (soma.volume + dend.volume + tuft.volume).to_decimal(u.um ** 3),
        )
        self.assertAlmostEqual(
            tree.total_volume.to_decimal(u.um ** 3),
            tree.metric.total_volume.to_decimal(u.um ** 3),
        )

    def test_metric_repr_and_str_expose_compact_summary(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        summary_repr = repr(tree.metric)
        summary_str = str(tree.metric)

        self.assertIn("MorphMetrics(", summary_repr)
        self.assertIn("n_branches=2", summary_repr)
        self.assertIn("n_stems=1", summary_repr)
        self.assertIn("n_bifurcations=0", summary_repr)
        self.assertIn("total_length=", summary_repr)
        self.assertIn("mean_radius=", summary_repr)
        self.assertIn("total_area=", summary_repr)
        self.assertIn("total_volume=", summary_repr)

        self.assertIn("n_branches", summary_str)
        self.assertIn("n_stems", summary_str)
        self.assertIn("n_bifurcations", summary_str)
        self.assertIn("max_branch_order", summary_str)
        self.assertIn("total_length", summary_str)
        self.assertIn("mean_radius", summary_str)
        self.assertIn("total_area", summary_str)
        self.assertIn("total_volume", summary_str)
        self.assertIn("max_path_dist", summary_str)

    def test_metric_exposes_coordinate_ranges_for_point_geometries(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (0.0, 10.0, 0.0)] * u.um, radii=[10.0, 10.0] * u.um,
                                  type="soma")
        dend = Branch.from_points(points=[(0.0, 10.0, 0.0), (30.0, 5.0, -2.0)] * u.um, radii=[2.0, 1.0] * u.um,
                                  type="basal_dendrite")
        tuft = Branch.from_points(points=[(0.0, 10.0, 0.0), (-7.0, 4.0, 9.0)] * u.um, radii=[2.0, 1.0] * u.um,
                                  type="apical_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.tuft = tuft

        self.assertEqual(tree.metric.n_stems, 2)
        self.assertEqual(tree.n_stems, 2)
        self.assertEqual(tree.metric.n_bifurcations, 1)
        self.assertEqual(tree.n_bifurcations, 1)
        self.assertEqual(tree.metric.max_branch_order, 1)
        self.assertEqual(tree.max_branch_order, 1)
        self.assertEqual(tree.metric.x_range.to_decimal(u.um), 37.0)
        self.assertEqual(tree.x_range.to_decimal(u.um), 37.0)
        self.assertEqual(tree.metric.y_range.to_decimal(u.um), 10.0)
        self.assertEqual(tree.y_range.to_decimal(u.um), 10.0)
        self.assertEqual(tree.metric.z_range.to_decimal(u.um), 11.0)
        self.assertEqual(tree.z_range.to_decimal(u.um), 11.0)

    def test_metric_exposes_neuromorpho_style_height_width_depth(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        tall_lo = Branch.from_points(points=[(0.0, 0.0, 0.0), (0.0, -30.0, 0.0)] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")
        tall_hi = Branch.from_points(points=[(0.0, 0.0, 0.0), (0.0, 30.0, 0.0)] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")
        height_tree = Morpho.from_root(soma, name="soma")
        height_tree.soma.tall_lo = tall_lo
        height_tree.soma.tall_hi = tall_hi
        self.assertEqual(height_tree.metric.height.to_decimal(u.um), 60.0)
        self.assertEqual(height_tree.height.to_decimal(u.um), 60.0)
        self.assertEqual(height_tree.metric.width.to_decimal(u.um), 0.0)
        self.assertEqual(height_tree.metric.depth.to_decimal(u.um), 0.0)

        wide_lo = Branch.from_points(points=[(0.0, 0.0, 0.0), (-10.0, 0.0, 0.0)] * u.um, radii=[1.0, 1.0] * u.um, type="apical_dendrite")
        wide_hi = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[1.0, 1.0] * u.um, type="apical_dendrite")
        width_tree = Morpho.from_root(soma, name="soma")
        width_tree.soma.wide_lo = wide_lo
        width_tree.soma.wide_hi = wide_hi
        self.assertEqual(width_tree.metric.height.to_decimal(u.um), 20.0)
        self.assertEqual(width_tree.height.to_decimal(u.um), 20.0)
        self.assertEqual(width_tree.metric.width.to_decimal(u.um), 0.0)
        self.assertEqual(width_tree.metric.depth.to_decimal(u.um), 0.0)

        deep_lo = Branch.from_points(points=[(0.0, 0.0, 0.0), (0.0, 0.0, -4.0)] * u.um, radii=[1.0, 1.0] * u.um, type="axon")
        deep_hi = Branch.from_points(points=[(0.0, 0.0, 0.0), (0.0, 0.0, 4.0)] * u.um, radii=[1.0, 1.0] * u.um, type="axon")
        depth_tree = Morpho.from_root(soma, name="soma")
        depth_tree.soma.deep_lo = deep_lo
        depth_tree.soma.deep_hi = deep_hi
        self.assertEqual(depth_tree.metric.height.to_decimal(u.um), 8.0)
        self.assertEqual(depth_tree.height.to_decimal(u.um), 8.0)
        self.assertEqual(depth_tree.metric.width.to_decimal(u.um), 0.0)
        self.assertEqual(depth_tree.metric.depth.to_decimal(u.um), 0.0)

    def test_height_width_depth_use_lmeasure_style_length_weighted_span(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        arbor_points = [(float(index), 0.0, 0.0) for index in range(21)] + [(100.0, 0.0, 0.0)]
        arbor = Branch.from_points(points=arbor_points * u.um, radii=[1.0] * len(arbor_points) * u.um, type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.main = arbor

        self.assertAlmostEqual(tree.height.to_decimal(u.um), 94.0, places=4)
        self.assertAlmostEqual(tree.width.to_decimal(u.um), 0.0, places=5)
        self.assertAlmostEqual(tree.depth.to_decimal(u.um), 0.0, places=5)
        self.assertEqual(tree.x_range.to_decimal(u.um), 100.0)

    def test_height_width_depth_ignore_soma_in_orientation_and_span(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (200.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        arbor = Branch.from_points(points=[(100.0, -10.0, 0.0), (100.0, 0.0, 0.0), (100.0, 10.0, 0.0)] * u.um, radii=[1.0, 1.0, 1.0] * u.um, type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.main = arbor

        self.assertEqual(tree.height.to_decimal(u.um), 10.0)
        self.assertEqual(tree.width.to_decimal(u.um), 0.0)
        self.assertEqual(tree.depth.to_decimal(u.um), 0.0)

    def test_height_width_depth_return_zero_for_soma_only_tree(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma")

        tree = Morpho.from_root(soma, name="soma")

        self.assertEqual(tree.height.to_decimal(u.um), 0.0)
        self.assertEqual(tree.width.to_decimal(u.um), 0.0)
        self.assertEqual(tree.depth.to_decimal(u.um), 0.0)

    def test_metric_exposes_neuromorpho_distance_metrics(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um,
                                  type="soma")
        main = Branch.from_points(points=[(5.0, 0.0, 0.0), (5.0, 10.0, 0.0)] * u.um, radii=[2.0, 1.5] * u.um,
                                  type="basal_dendrite")
        tuft = Branch.from_points(points=[(5.0, 10.0, 0.0), (5.0, 20.0, 0.0)] * u.um, radii=[1.5, 1.0] * u.um,
                                  type="apical_dendrite")
        side = Branch.from_points(points=[(10.0, 0.0, 0.0), (12.0, 0.0, 0.0)] * u.um, radii=[1.0, 0.8] * u.um,
                                  type="axon")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.attach(main, name="main", parent_x=0.5)
        tree.main.attach(tuft, name="tuft")
        tree.soma.attach(side, name="side", parent_x=1.0)

        self.assertAlmostEqual(tree.metric.max_path_distance.to_decimal(u.um), 25.0)
        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um), 25.0)
        self.assertAlmostEqual(tree.metric.max_euclidean_distance.to_decimal(u.um), np.sqrt(425.0))
        self.assertAlmostEqual(tree.max_euclidean_distance.to_decimal(u.um), np.sqrt(425.0))
        self.assertAlmostEqual(tree.metric.max_path_distance_excluding_soma.to_decimal(u.um), 20.0)
        self.assertAlmostEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 20.0)
        self.assertAlmostEqual(tree.metric.max_euclidean_distance_excluding_soma.to_decimal(u.um), 20.0)
        self.assertAlmostEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 20.0)

    def test_excluding_soma_distances_remove_full_root_contribution_at_distal_attach(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um,
                                  type="soma")
        distal = Branch.from_points(points=[(10.0, 0.0, 0.0), (10.0, 16.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um,
                                    type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.attach(distal, name="distal", parent_x=1.0)

        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um), 26.0)
        self.assertAlmostEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 16.0)
        self.assertAlmostEqual(tree.max_euclidean_distance.to_decimal(u.um), np.sqrt(356.0))
        self.assertAlmostEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 16.0)

    def test_excluding_soma_distances_do_not_apply_global_half_soma_subtraction(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um,
                                  type="soma")
        midpoint = Branch.from_points(points=[(5.0, 0.0, 0.0), (5.0, 15.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um,
                                      type="basal_dendrite")
        distal = Branch.from_points(points=[(10.0, 0.0, 0.0), (10.0, 16.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um,
                                    type="axon")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.attach(midpoint, name="midpoint", parent_x=0.5)
        tree.soma.attach(distal, name="distal", parent_x=1.0)

        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um), 26.0)
        self.assertAlmostEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 16.0)
        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um) - tree.soma.length.to_decimal(u.um) / 2.0, 21.0)
        self.assertAlmostEqual(tree.max_euclidean_distance.to_decimal(u.um), np.sqrt(356.0))
        self.assertAlmostEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 16.0)

    def test_excluding_soma_distances_match_existing_metrics_for_non_soma_root(self) -> None:
        root = Branch.from_points(points=[(0.0, 0.0, 0.0), (12.0, 0.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um,
                                  type="axon")
        child = Branch.from_points(points=[(12.0, 0.0, 0.0), (12.0, 8.0, 0.0)] * u.um, radii=[1.0, 0.8] * u.um,
                                   type="basal_dendrite")

        tree = Morpho.from_root(root, name="axon")
        tree.axon.attach(child, name="child")

        self.assertEqual(tree.max_path_distance_excluding_soma, tree.max_path_distance)
        self.assertEqual(tree.max_euclidean_distance_excluding_soma, tree.max_euclidean_distance)

    def test_excluding_soma_distances_return_zero_for_soma_only_tree(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um,
                                  type="soma")

        tree = Morpho.from_root(soma, name="soma")

        self.assertEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 0.0)
        self.assertEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 0.0)

    def test_coordinate_range_metrics_require_full_point_geometry(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.0, 0.8] * u.um, type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        self.assertFalse(tree.has_full_point_geometry)

        with self.assertRaisesRegex(ValueError, "Coordinate range metrics require full point geometry on every branch"):
            _ = tree.metric.x_range
        with self.assertRaisesRegex(ValueError, "Coordinate range metrics require full point geometry on every branch"):
            _ = tree.x_range
        with self.assertRaisesRegex(ValueError, "Height/width/depth metrics require full point geometry on every branch"):
            _ = tree.metric.height
        with self.assertRaisesRegex(ValueError, "Height/width/depth metrics require full point geometry on every branch"):
            _ = tree.height

    def test_max_euclidean_distance_requires_full_point_geometry(self) -> None:
        soma = Branch.from_points(points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.0, 0.8] * u.um, type="basal_dendrite")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        with self.assertRaisesRegex(ValueError, "Euclidean distance metrics require full point geometry on every branch"):
            _ = tree.metric.max_euclidean_distance
        with self.assertRaisesRegex(ValueError, "Euclidean distance metrics require full point geometry on every branch"):
            _ = tree.metric.max_euclidean_distance_excluding_soma

    def test_foreign_missing_and_reserved_children_are_rejected(self) -> None:
        soma0 = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        soma1 = Branch.from_lengths(lengths=[18.0] * u.um, radii=[9.0, 9.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")

        tree0 = Morpho.from_root(soma0, name="soma")
        tree1 = Morpho.from_root(soma1, name="other")
        tree0.soma.dend = dend

        with self.assertRaises(ValueError):
            tree1.other.foreign = tree0.soma.dend
        with self.assertRaises(KeyError):
            tree0.attach(parent="missing", child_branch=dend)
        with self.assertRaises(ValueError):
            tree0.attach(parent=tree1.other, child_branch=dend, child_name="foreign")
        with self.assertRaises(ValueError):
            tree0.soma.length = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            tree0.soma.select = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            tree0.soma.metric = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            tree0.soma.total_area = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            tree0.soma.n_children = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon")
        with self.assertRaises(ValueError):
            tree0.soma.branch = Branch.from_lengths(
                lengths=[60.0] * u.um,
                radii=[2.0, 1.0] * u.um,
                type="basal_dendrite",
            )
