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


from dataclasses import is_dataclass
import unittest
import warnings

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.filter import BranchSlice, LocsetMask, RootLocation, Terminals
from braincell.vis._testing import FakeBackend
from braincell.vis.backend import BackendChooser
from braincell.morph import MorphoBranch, MorphoMetric
from braincell.morph._testing import (
    make_apical,
    make_axon,
    make_basal,
    make_deep_chain_tree,
    make_dendrite,
    make_soma,
)


class MorphoTest(unittest.TestCase):
    def test_tree_topology_queries_and_branch_views(self) -> None:
        soma = make_soma()
        dend = Branch.from_lengths(
            lengths=np.array([100.0]) * u.um,
            radii=np.array([2.0, 1.0]) * u.um,
            type="basal_dendrite",
        )
        axon = make_axon()

        tree = Morphology.from_root(soma, name="soma")
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
        soma = make_soma()
        dend = make_basal()
        tree = Morphology.from_root(soma, name="soma")
        branch = tree.soma[0.5, 1.0].attach(dend, name="dendrite")

        self.assertEqual(branch.name, "dendrite")
        self.assertEqual(branch.parent_x, 0.5)
        self.assertEqual(branch.child_x, 1.0)
        self.assertEqual(tree.edges[0].child_x, 1.0)

    def test_child_x_accepts_only_endpoint_values(self) -> None:
        soma = make_soma()
        dend = make_basal()
        tree = Morphology.from_root(soma, name="soma")

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
        soma = make_soma()
        dend = make_basal()
        tuft = make_apical()
        axon = make_axon()

        tree = Morphology.from_root(soma, name="soma")
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
        soma = make_soma()
        dend = make_dendrite()

        tree = Morphology.from_root(soma, name="soma")
        explicit = tree.soma.attach(dend, name="first")
        auto0 = tree.soma.attach(dend)
        auto1 = tree.soma.attach(make_dendrite())

        self.assertEqual(explicit.name, "first")
        self.assertEqual(auto0.name, "dendrite_0")
        self.assertEqual(auto1.name, "dendrite_1")

    def test_root_can_opt_into_type_based_auto_naming(self) -> None:
        axon = make_axon()

        tree = Morphology.from_root(axon, name=None)

        self.assertEqual(tree.root.name, "axon_0")
        self.assertEqual(tree.topo(), "axon_0")

    def test_branch_order_queries_are_available(self) -> None:
        soma = make_soma()
        axon = make_axon()
        dend = make_dendrite()
        apical = Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.2, 0.8] * u.um, type="apical_dendrite")
        custom = Branch.from_lengths(lengths=[25.0] * u.um, radii=[0.7, 0.5] * u.um, type="custom")

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.d = dend
        tree.soma.a = axon
        tree.soma.t = apical
        tree.soma.c = custom

        self.assertEqual(tuple(branch.name for branch in tree.branches), ("soma", "d", "a", "t", "c"))
        self.assertEqual(
            tuple(branch.name for branch in tree.branch_by_order(order="default")), ("soma", "d", "a", "t", "c")
        )
        self.assertEqual(
            tuple(branch.name for branch in tree.branch_by_order(order="type")), ("soma", "a", "d", "t", "c")
        )
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

    def test_morpho_equality_compares_structure_and_geometry(self) -> None:
        soma = make_soma()
        dend = make_basal()
        axon = make_axon()

        tree0 = Morphology.from_root(soma, name="soma")
        tree0.soma.attach(dend, name="dendrite", parent_x=1.0)
        tree0.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=0.5, child_x=1.0)

        tree1 = Morphology.from_root(soma, name="soma")
        tree1.soma.dendrite = dend
        tree1.soma[0.5, 1.0].axon = axon

        self.assertEqual(tree0, tree1)

        renamed = Morphology.from_root(soma, name="soma")
        renamed.soma.attach(dend, name="d_other", parent_x=1.0)
        renamed.soma[0.5, 1.0].axon = axon
        self.assertNotEqual(tree0, renamed)

        shifted = Morphology.from_root(soma, name="soma")
        shifted.soma.attach(dend, name="dendrite", parent_x=0.0)
        shifted.soma[0.5, 1.0].axon = axon
        self.assertNotEqual(tree0, shifted)

        other_geom = Morphology.from_root(soma, name="soma")
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
        soma = make_soma()
        dend = make_basal()
        tuft = make_apical()

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        with self.assertRaises(ValueError):
            tree.dend.attach(tuft, parent_x=0.5)

        with self.assertRaises(ValueError):
            tree.dend[0.5].attach(tuft, name="tuft")

    def test_metric_exposes_tree_level_metrics_with_compatible_shortcuts(self) -> None:
        soma = make_soma()
        dend = make_basal()
        tuft = make_apical()

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.dend.tuft = tuft

        self.assertEqual(tree.n_branches, 3)
        self.assertEqual(tree.n_stems, 1)
        self.assertEqual(tree.n_bifurcations, 0)
        self.assertEqual(tree.max_branch_order, 2)
        self.assertEqual(tree.soma.branch, soma)
        self.assertEqual(tree.soma.mean_radius.to_decimal(u.um), 10.0)
        self.assertEqual(tree.soma.n_children, 1)
        self.assertEqual(tree.soma.dend.mean_radius.to_decimal(u.um), 1.5)
        self.assertEqual(tree.soma.dend.n_children, 1)
        self.assertAlmostEqual(tree.soma.dend.tuft.mean_radius.to_decimal(u.um), 0.8)
        self.assertEqual(tree.soma.dend.tuft.n_children, 0)
        self.assertEqual(tree.path_to_root(2), (0, 1, 2))
        self.assertEqual(tree.max_path_distance.to_decimal(u.um), 110.0)
        self.assertAlmostEqual(tree.mean_radius.to_decimal(u.um), 314.0 / 110.0)
        self.assertEqual(tree.total_length.to_decimal(u.um), 110.0)
        self.assertAlmostEqual(
            tree.total_area.to_decimal(u.um**2),
            (soma.area + dend.area + tuft.area).to_decimal(u.um**2),
        )
        self.assertAlmostEqual(
            tree.total_volume.to_decimal(u.um**3),
            (soma.volume + dend.volume + tuft.volume).to_decimal(u.um**3),
        )

    def test_metric_returns_dataclass_snapshot_with_compact_str(self) -> None:
        soma = make_soma()
        dend = make_basal()

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        metric = tree.metric
        summary_str = str(metric)

        self.assertIsInstance(metric, MorphoMetric)
        self.assertTrue(is_dataclass(metric))
        self.assertEqual(metric.n_branches, tree.n_branches)
        self.assertEqual(metric.n_stems, tree.n_stems)
        self.assertEqual(metric.n_bifurcations, tree.n_bifurcations)
        self.assertEqual(metric.max_branch_order, tree.max_branch_order)
        self.assertEqual(metric.total_length, tree.total_length)
        self.assertEqual(metric.total_area, tree.total_area)
        self.assertEqual(metric.total_volume, tree.total_volume)
        self.assertEqual(metric.mean_radius, tree.mean_radius)
        self.assertEqual(metric.max_path_distance, tree.max_path_distance)
        self.assertFalse(metric.has_full_point_geometry)

        self.assertIn("n_branches", summary_str)
        self.assertIn("n_stems", summary_str)
        self.assertIn("n_bifurcations", summary_str)
        self.assertIn("max_branch_order", summary_str)
        self.assertIn("total_length", summary_str)
        self.assertIn("mean_radius", summary_str)
        self.assertIn("total_area", summary_str)
        self.assertIn("total_volume", summary_str)
        self.assertIn("max_path_dist", summary_str)

    def test_metric_uses_optional_fields_for_unavailable_point_metrics(self) -> None:
        soma = make_soma()
        dend = make_basal()

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        metric = tree.metric

        self.assertEqual(metric.n_branches, tree.n_branches)
        self.assertFalse(metric.has_full_point_geometry)
        self.assertIsNone(metric.max_euclidean_distance)
        self.assertIsNone(metric.max_euclidean_distance_excluding_soma)
        self.assertIsNone(metric.x_range)
        self.assertIsNone(metric.y_range)
        self.assertIsNone(metric.z_range)

    def test_metric_exposes_coordinate_ranges_for_point_geometries(self) -> None:
        soma = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (0.0, 10.0, 0.0)] * u.um, radii=[10.0, 10.0] * u.um, type="soma"
        )
        dend = Branch.from_points(
            points=[(0.0, 10.0, 0.0), (30.0, 5.0, -2.0)] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite"
        )
        tuft = Branch.from_points(
            points=[(0.0, 10.0, 0.0), (-7.0, 4.0, 9.0)] * u.um, radii=[2.0, 1.0] * u.um, type="apical_dendrite"
        )

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.tuft = tuft

        metric = tree.metric

        self.assertEqual(tree.n_stems, 2)
        self.assertEqual(tree.n_bifurcations, 1)
        self.assertEqual(tree.max_branch_order, 1)
        self.assertEqual(tree.x_range.to_decimal(u.um), 37.0)
        self.assertEqual(tree.y_range.to_decimal(u.um), 10.0)
        self.assertEqual(tree.z_range.to_decimal(u.um), 11.0)
        self.assertTrue(metric.has_full_point_geometry)
        self.assertEqual(metric.x_range, tree.x_range)
        self.assertEqual(metric.y_range, tree.y_range)
        self.assertEqual(metric.z_range, tree.z_range)

    def test_metric_exposes_neuromorpho_distance_metrics(self) -> None:
        soma = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma"
        )
        main = Branch.from_points(
            points=[(5.0, 0.0, 0.0), (5.0, 10.0, 0.0)] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite"
        )
        tuft = Branch.from_points(
            points=[(5.0, 10.0, 0.0), (5.0, 20.0, 0.0)] * u.um, radii=[1.5, 1.0] * u.um, type="apical_dendrite"
        )
        side = Branch.from_points(
            points=[(10.0, 0.0, 0.0), (12.0, 0.0, 0.0)] * u.um, radii=[1.0, 0.8] * u.um, type="axon"
        )

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.attach(main, name="main", parent_x=0.5)
        tree.main.attach(tuft, name="tuft")
        tree.soma.attach(side, name="side", parent_x=1.0)

        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um), 25.0)
        self.assertAlmostEqual(tree.max_euclidean_distance.to_decimal(u.um), np.sqrt(425.0))
        self.assertAlmostEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 20.0)
        self.assertAlmostEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 20.0)

    def test_excluding_soma_distances_remove_full_root_contribution_at_distal_attach(self) -> None:
        soma = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma"
        )
        distal = Branch.from_points(
            points=[(10.0, 0.0, 0.0), (10.0, 16.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite"
        )

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.attach(distal, name="distal", parent_x=1.0)

        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um), 26.0)
        self.assertAlmostEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 16.0)
        self.assertAlmostEqual(tree.max_euclidean_distance.to_decimal(u.um), np.sqrt(356.0))
        self.assertAlmostEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 16.0)

    def test_excluding_soma_distances_do_not_apply_global_half_soma_subtraction(self) -> None:
        soma = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma"
        )
        midpoint = Branch.from_points(
            points=[(5.0, 0.0, 0.0), (5.0, 15.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite"
        )
        distal = Branch.from_points(
            points=[(10.0, 0.0, 0.0), (10.0, 16.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um, type="axon"
        )

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.attach(midpoint, name="midpoint", parent_x=0.5)
        tree.soma.attach(distal, name="distal", parent_x=1.0)

        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um), 26.0)
        self.assertAlmostEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 16.0)
        self.assertAlmostEqual(tree.max_path_distance.to_decimal(u.um) - tree.soma.length.to_decimal(u.um) / 2.0, 21.0)
        self.assertAlmostEqual(tree.max_euclidean_distance.to_decimal(u.um), np.sqrt(356.0))
        self.assertAlmostEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 16.0)

    def test_excluding_soma_distances_match_existing_metrics_for_non_soma_root(self) -> None:
        root = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (12.0, 0.0, 0.0)] * u.um, radii=[2.0, 1.0] * u.um, type="axon"
        )
        child = Branch.from_points(
            points=[(12.0, 0.0, 0.0), (12.0, 8.0, 0.0)] * u.um, radii=[1.0, 0.8] * u.um, type="basal_dendrite"
        )

        tree = Morphology.from_root(root, name="axon")
        tree.axon.attach(child, name="child")

        self.assertEqual(tree.max_path_distance_excluding_soma, tree.max_path_distance)
        self.assertEqual(tree.max_euclidean_distance_excluding_soma, tree.max_euclidean_distance)

    def test_excluding_soma_distances_return_zero_for_soma_only_tree(self) -> None:
        soma = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma"
        )

        tree = Morphology.from_root(soma, name="soma")

        self.assertEqual(tree.max_path_distance_excluding_soma.to_decimal(u.um), 0.0)
        self.assertEqual(tree.max_euclidean_distance_excluding_soma.to_decimal(u.um), 0.0)

    def test_coordinate_range_metrics_require_full_point_geometry(self) -> None:
        soma = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma"
        )
        dend = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.0, 0.8] * u.um, type="basal_dendrite")

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        self.assertFalse(tree.has_full_point_geometry)

        with self.assertRaisesRegex(ValueError, "Coordinate range metrics require full point geometry on every branch"):
            _ = tree.x_range

    def test_max_euclidean_distance_requires_full_point_geometry(self) -> None:
        soma = Branch.from_points(
            points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * u.um, radii=[5.0, 5.0] * u.um, type="soma"
        )
        dend = Branch.from_lengths(lengths=[20.0] * u.um, radii=[1.0, 0.8] * u.um, type="basal_dendrite")

        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        with self.assertRaisesRegex(
            ValueError, "Euclidean distance metrics require full point geometry on every branch"
        ):
            _ = tree.max_euclidean_distance
        with self.assertRaisesRegex(
            ValueError, "Euclidean distance metrics require full point geometry on every branch"
        ):
            _ = tree.max_euclidean_distance_excluding_soma

    def test_foreign_missing_and_reserved_children_are_rejected(self) -> None:
        soma0 = make_soma()
        soma1 = Branch.from_lengths(lengths=[18.0] * u.um, radii=[9.0, 9.0] * u.um, type="soma")
        dend = make_basal()

        tree0 = Morphology.from_root(soma0, name="soma")
        tree1 = Morphology.from_root(soma1, name="other")
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


class MorphoDerivedCacheTest(unittest.TestCase):
    """``Morphology`` memoizes branch ordering; mutation must invalidate it.

    ``_branch_index_map`` used to rebuild an N-entry dict on every ``.index``
    read. It is cached now, and ``Morphology`` is mutable, so every one of
    these asserts that reading a derived value *before* an ``attach`` cannot
    leave a stale answer behind afterwards.
    """

    @staticmethod
    def _soma() -> Branch:
        return make_soma()

    @staticmethod
    def _dend(length: float = 30.0) -> Branch:
        return Branch.from_lengths(lengths=[length] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")

    @staticmethod
    def _dend_with_points() -> Branch:
        return Branch.from_points(
            points=np.asarray([[0.0, 0.0, 0.0], [0.0, 30.0, 0.0]]) * u.um,
            radii=np.asarray([2.0, 1.0]) * u.um,
            type="basal_dendrite",
        )

    def test_branch_index_updates_after_a_later_attach(self) -> None:
        tree = Morphology.from_root(self._soma(), name="soma")
        first = tree.soma.attach(self._dend(), name="a")

        # Read every cached view before mutating.
        self.assertEqual(first.index, 1)
        self.assertEqual(tree.n_branches, 2)
        self.assertEqual([b.name for b in tree.branches], ["soma", "a"])

        second = first.attach(self._dend(), name="b")

        self.assertEqual(first.index, 1)
        self.assertEqual(second.index, 2)
        self.assertEqual(tree.n_branches, 3)
        self.assertEqual([b.name for b in tree.branches], ["soma", "a", "b"])
        self.assertIs(tree.branch(index=2), second)

    def test_orders_other_than_default_also_invalidate(self) -> None:
        tree = Morphology.from_root(self._soma(), name="soma")
        axon = tree.soma.attach(
            Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 0.5] * u.um, type="axon"),
            name="ax",
        )

        self.assertEqual([b.name for b in tree.branch_by_order(order="type")], ["soma", "ax"])
        self.assertEqual([b.name for b in tree.branch_by_order(order="depth")], ["soma", "ax"])
        self.assertEqual(axon.index_by(order="depth"), 1)

        deep = axon.attach(self._dend(), name="deep")

        self.assertEqual({b.name for b in tree.branch_by_order(order="type")}, {"soma", "ax", "deep"})
        self.assertEqual([b.name for b in tree.branch_by_order(order="depth")], ["soma", "ax", "deep"])
        self.assertEqual(deep.index_by(order="depth"), 2)
        self.assertEqual(deep.branch_order, 2)

    def test_has_full_point_geometry_flips_when_a_geometryless_branch_arrives(self) -> None:
        tree = Morphology.from_root(
            Branch.from_points(
                points=np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]) * u.um,
                radii=np.asarray([10.0, 10.0]) * u.um,
                type="soma",
            ),
            name="soma",
        )
        tree.soma.attach(self._dend_with_points(), name="a")
        self.assertTrue(tree.has_full_point_geometry)
        self.assertTrue(tree.metric.has_full_point_geometry)

        tree.soma.attach(self._dend(), name="b")  # from_lengths: no 3-D points

        self.assertFalse(tree.has_full_point_geometry)
        self.assertFalse(tree.metric.has_full_point_geometry)

    def test_max_branch_order_and_path_to_root_follow_new_attachments(self) -> None:
        tree = Morphology.from_root(self._soma(), name="soma")
        node = tree.soma.attach(self._dend(), name="a")
        self.assertEqual(tree.max_branch_order, 1)
        self.assertEqual(tree.path_to_root(1), (0, 1))

        for name in ("b", "c"):
            node = node.attach(self._dend(), name=name)

        self.assertEqual(tree.max_branch_order, 3)
        self.assertEqual(tree.path_to_root(3), (0, 1, 2, 3))
        self.assertEqual(node.branch_order, 3)

    def test_revision_advances_on_every_attach(self) -> None:
        tree = Morphology.from_root(self._soma(), name="soma")
        before = tree._revision
        tree.soma.attach(self._dend(), name="a")
        self.assertGreater(tree._revision, before)


class MorphoBranchDerivedPropertyTest(unittest.TestCase):
    """``branch_id`` / ``branch_order`` / ``n_tapers`` published by MorphoBranch.

    These used to be re-derived inside ``braincell.filter.helper``; they are
    branch facts and belong on the branch view.
    """

    def _tree(self) -> Morphology:
        soma = make_soma()
        tree = Morphology.from_root(soma, name="soma")
        mid = tree.soma.attach(
            Branch.from_lengths(lengths=[30.0, 20.0] * u.um, radii=[2.0, 1.5, 1.0] * u.um, type="basal_dendrite"),
            name="mid",
        )
        mid.attach(
            Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 0.5] * u.um, type="basal_dendrite"),
            name="tip",
        )
        return tree

    def test_branch_id_matches_index(self) -> None:
        tree = self._tree()
        for expected, branch in enumerate(tree.branches):
            self.assertEqual(branch.branch_id, expected)
            self.assertEqual(branch.branch_id, branch.index)

    def test_branch_order_counts_parent_hops(self) -> None:
        tree = self._tree()
        self.assertEqual([b.branch_order for b in tree.branches], [0, 1, 2])
        for branch in tree.branches:
            self.assertEqual(branch.branch_order, len(tree.path_to_root(branch.index)) - 1)

    def test_n_tapers_aliases_n_segments(self) -> None:
        tree = self._tree()
        self.assertEqual([b.n_tapers for b in tree.branches], [1, 2, 1])
        for branch in tree.branches:
            self.assertEqual(branch.n_tapers, branch.n_segments)

    def test_new_property_names_are_reserved_for_branch_names(self) -> None:
        tree = self._tree()
        for name in ("branch_id", "branch_order", "n_tapers"):
            with self.assertRaisesRegex(ValueError, "reserved"):
                tree.attach(
                    parent="soma",
                    child_branch=Branch.from_lengths(lengths=[5.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon"),
                    child_name=name,
                )


class MorphoReservedNameTest(unittest.TestCase):
    """Every public ``Morphology`` / ``MorphoBranch`` attribute is name-reserved.

    The reserved set used to be a hand-typed literal that drifted behind the
    classes it guarded: ``topo`` and ``from_neuromorpho`` were both accepted
    as branch names and then permanently unreachable, because attribute
    lookup finds the class member before ``__getattr__`` ever runs.
    """

    def _tree(self) -> Morphology:
        return Morphology.from_root(make_soma(), name="soma")

    def _attach_named(self, tree: Morphology, name: str) -> None:
        tree.attach(parent="soma", child_branch=make_axon(), child_name=name, parent_x=1.0)

    def test_names_that_used_to_slip_through_are_rejected(self) -> None:
        for name in ("topo", "from_neuromorpho"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "reserved by the Morphology API"):
                    self._attach_named(self._tree(), name)

    def test_dynamic_morpho_branch_attributes_are_reserved(self) -> None:
        # These five are served by MorphoBranch.__getattr__ and so are absent
        # from dir(MorphoBranch); deriving the reserved set from dir() alone
        # would silently stop protecting them.
        for name in ("branch", "name", "parent_id", "parent_x", "child_x"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "reserved by the Morphology API"):
                    self._attach_named(self._tree(), name)

    def test_every_public_attribute_of_both_classes_is_reserved(self) -> None:
        public = {n for n in dir(Morphology) if not n.startswith("_")}
        public |= {n for n in dir(MorphoBranch) if not n.startswith("_")}
        self.assertIn("topo", public)
        for name in sorted(public):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "reserved by the Morphology API"):
                    self._attach_named(self._tree(), name)


class MorphoDeepTreeTest(unittest.TestCase):
    """Whole-tree walks stay iterative on morphologies deeper than the C stack.

    ``topo()`` formatted the tree by recursive descent and raised
    ``RecursionError`` past roughly 400 chained branches -- a depth real
    thin-neurite reconstructions reach routinely.
    """

    def test_topo_formats_a_1200_branch_chain(self) -> None:
        tree = make_deep_chain_tree(1200)
        rendered = tree.topo()
        self.assertEqual(rendered.count("\n"), 1199)
        self.assertIn("soma", rendered)
        self.assertIn("seg_1198", rendered)

    def test_deep_chain_aggregates_do_not_recurse(self) -> None:
        tree = make_deep_chain_tree(1200)
        self.assertEqual(tree.n_branches, 1200)
        self.assertEqual(tree.max_branch_order, 1199)
        self.assertEqual(len(tree.edges), 1199)
        self.assertGreater(tree.total_area.to_decimal(u.um**2), 0.0)


class MorphoNamingStateTest(unittest.TestCase):
    """``naming_state`` / ``restore_naming_state``, the public auto-name API.

    ``braincell.io.checkpoint`` used to reach into ``_type_name_counters``
    directly to round-trip these.
    """

    def _tree(self, n_dendrites: int = 3) -> Morphology:
        soma = make_soma()
        tree = Morphology.from_root(soma, name="soma")
        for _ in range(n_dendrites):
            tree.attach(
                parent="soma",
                child_branch=Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 0.5] * u.um, type="dendrite"),
            )
        return tree

    def test_reports_the_next_suffix_per_type(self) -> None:
        tree = self._tree()

        self.assertEqual(tree.naming_state(), {"dendrite": 3})

    def test_the_returned_mapping_is_a_copy(self) -> None:
        tree = self._tree()

        state = tree.naming_state()
        state["dendrite"] = 999

        self.assertEqual(tree.naming_state()["dendrite"], 3)

    def test_restoring_resumes_the_sequence(self) -> None:
        source = self._tree()
        target = Morphology.from_root(
            make_soma(),
            name="soma",
        )

        target.restore_naming_state(source.naming_state())
        view = target.attach(
            parent="soma",
            child_branch=Branch.from_lengths(lengths=[5.0] * u.um, radii=[1.0, 0.5] * u.um, type="dendrite"),
        )

        self.assertEqual(view.name, "dendrite_3")

    def test_restoring_merges_rather_than_replaces(self) -> None:
        tree = self._tree()

        tree.restore_naming_state({"axon": 7})

        self.assertEqual(tree.naming_state(), {"dendrite": 3, "axon": 7})

    def test_a_stale_counter_still_cannot_collide(self) -> None:
        # A too-small counter only costs extra probes; attach() skips every
        # suffix already taken, so no restored value can produce a duplicate.
        tree = self._tree()

        tree.restore_naming_state({"dendrite": 0})
        view = tree.attach(
            parent="soma",
            child_branch=Branch.from_lengths(lengths=[5.0] * u.um, radii=[1.0, 0.5] * u.um, type="dendrite"),
        )

        self.assertEqual(view.name, "dendrite_3")

    def test_rejects_a_non_mapping(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            self._tree().restore_naming_state([("dendrite", 3)])

    def test_rejects_a_non_integer_counter(self) -> None:
        for value in ("3", 3.5, True, None):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self._tree().restore_naming_state({"dendrite": value})

    def test_rejects_a_negative_counter(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._tree().restore_naming_state({"dendrite": -1})


class MorphoSelectAndVisTest(unittest.TestCase):
    """``Morphology.select``, ``.vis2d``, and ``.vis3d`` — all defined in morphology.py."""

    def test_morpho_select_is_region_eval_sugar(self) -> None:
        soma = make_soma()
        dend = Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 1.0] * u.um, type="apical_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        expr = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        selected = tree.select(expr)
        evaluated = expr.evaluate(tree)

        self.assertEqual(selected.intervals, evaluated.intervals)

    def test_morpho_select_accepts_locset_expr(self) -> None:
        soma = make_soma()
        dend = Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 1.0] * u.um, type="apical_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        expr = RootLocation(x=0.5) | Terminals()
        selected = tree.select(expr)
        evaluated = expr.evaluate(tree)

        self.assertIsInstance(selected, LocsetMask)
        self.assertEqual(selected.points, evaluated.points)

    def test_morpho_select_rejects_non_filter_expr(self) -> None:
        soma = make_soma()
        tree = Morphology.from_root(soma, name="soma")

        with self.assertRaises(TypeError):
            tree.select(123)  # type: ignore[arg-type]

    def test_morpho_vis3d_is_a_thin_wrapper_over_plot3d(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")
        backend = FakeBackend()

        rendered = tree.vis3d(chooser=BackendChooser(backends=(backend,)), backend="fake")

        self.assertIs(rendered, backend.last_request)
        self.assertEqual(rendered.dimensionality, "3d")
        self.assertEqual(rendered.mode, "geometry")
        self.assertEqual(len(rendered.scene.branches), 1)

    def test_morpho_vis3d_accepts_explicit_geometry_mode(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")
        backend = FakeBackend()

        rendered = tree.vis3d(mode="geometry", chooser=BackendChooser(backends=(backend,)), backend="fake")

        self.assertEqual(rendered.mode, "geometry")

    def test_morpho_vis3d_rejects_unknown_mode(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")

        with self.assertRaisesRegex(ValueError, "Unsupported 3D mode"):
            tree.vis3d(mode="layout")

    def test_morpho_vis3d_requires_full_point_geometry(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.from_lengths(
            lengths=[80.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        with self.assertRaisesRegex(ValueError, "requires complete point geometry on every branch"):
            tree.vis3d()

    def test_morpho_vis2d_routes_into_2d_plot_dispatch(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.from_points(
            points=[[20.0, 0.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend

        rendered = tree.vis2d(chooser=BackendChooser(backends=(FakeBackend(),)), backend="fake")

        self.assertEqual(rendered.dimensionality, "2d")
        self.assertEqual(rendered.layout, "fan")
        self.assertEqual(rendered.shape, "frustum")
        self.assertIsNotNone(rendered.scene)
        self.assertEqual(rendered.scene.layout, "fan")
        self.assertEqual(rendered.scene.shape, "frustum")
        self.assertEqual(rendered.scene.projection_plane, None)
        self.assertEqual(len(rendered.scene.polygons), 2)

    def test_morpho_vis2d_accepts_layout_parameters(self) -> None:
        soma = make_soma()
        dend_a = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="apical_dendrite")
        dend_b = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
        tree.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rendered = tree.vis2d(
                layout="stem",
                shape="line",
                min_branch_angle_deg=90.0,
                root_layout="legacy",
                chooser=BackendChooser(backends=(FakeBackend(),)),
                backend="fake",
            )

        self.assertEqual(rendered.layout, "stem")
        self.assertEqual(rendered.shape, "line")
        child_angles = sorted(
            np.degrees(
                np.arctan2(
                    polyline.points_um[-1, 1] - polyline.points_um[0, 1],
                    polyline.points_um[-1, 0] - polyline.points_um[0, 0],
                )
            )
            for polyline in rendered.scene.polylines
            if polyline.branch_name in {"dend_a", "dend_b"}
        )
        self.assertGreaterEqual(child_angles[1] - child_angles[0], 90.0 - 1e-6)

    def test_morpho_vis2d_accepts_style_overrides(self) -> None:
        soma = make_soma()
        dend = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="apical_dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)

        rendered = tree.vis2d(
            shape="frustum",
            branch_type_colors={"apical_dendrite": "#445566"},
            branch_type_edge_colors_2d={"apical_dendrite": "#112233"},
            frustum_edge_linewidth_2d=1.25,
            chooser=BackendChooser(backends=(FakeBackend(),)),
            backend="fake",
        )

        dend_polygons = [polygon for polygon in rendered.scene.polygons if polygon.branch_name == "dend"]
        self.assertTrue(all(polygon.color_rgb == (68, 85, 102) for polygon in dend_polygons))
        self.assertTrue(all(polygon.edge_color_rgb == (17, 34, 51) for polygon in dend_polygons))
        self.assertTrue(all(abs(polygon.edge_linewidth - 1.25) < 1e-9 for polygon in dend_polygons))
