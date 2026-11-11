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


import dataclasses
from dataclasses import is_dataclass
import pathlib
import re
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch
import warnings

import brainunit as u
import numpy as np

import braincell.morph
from braincell import Branch, Cell, MaxCVLen, Morphology, SingleCompartment
from braincell.filter import (
    AllRegion,
    BranchSlice,
    LocsetMask,
    RootLocation,
    Terminals,
    branch_range,
    density,
    sample,
)
from braincell.io._testing import ALLOWED_TYPES, FIXTURE_DIR, VALID_SWC_FIXTURES
from braincell.morph import MorphoBranch, MorphoMetric, PrincipalAxes
from braincell.morph.branch import _ALLOWED_BRANCH_TYPES
from braincell.morph.morphology import frozen_snapshot
from braincell.morph._testing import (
    EDIT_CASES,
    SABOTAGE_CASES,
    EditCase,
    assert_layer_fresh,
    assert_morpho_fresh,
    cold_rebuild,
    make_apical,
    make_axon,
    make_basal,
    make_deep_chain_tree,
    make_dendrite,
    make_projected_node_tree,
    make_root_split_tree,
    make_soma,
    make_tree_s,
    make_tree_t,
    make_tree_u,
    make_tree_w,
    morph_fingerprint,
    stretched,
)


class MorphoTest(unittest.TestCase):
    def test_to_swc_delegates_to_swc_writer(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")

        with patch("braincell.io.swc.writer.write_swc", return_value="written.swc") as writer:
            result = tree.to_swc("cell.swc")

        self.assertEqual(result, "written.swc")
        writer.assert_called_once_with(tree, "cell.swc")

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
        self.assertEqual(tree._revision, before + 1)


def _make_tree_u() -> Morphology:
    """Soma with three leaves, all length-only: dend (100 um), axon (30 um), apical (80 um).

    Default indices are soma 0, dend 1, axon 2, apical 3, so removing the
    leaf ``dend`` moves both later branches down by one.
    """
    tree = Morphology.from_root(
        Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma"), name="soma"
    )
    tree.soma.dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")
    tree.soma.axon = Branch.from_lengths(lengths=[30.0] * u.um, radii=[0.5, 0.5] * u.um, type="axon")
    tree.soma.apical = Branch.from_lengths(lengths=[80.0] * u.um, radii=[1.0, 1.0] * u.um, type="apical_dendrite")
    return tree


def _short_axon() -> Branch:
    return Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="axon")


class MorphoEditGateTest(unittest.TestCase):
    """Every edit goes through one gate: validate, write, then exactly one revision bump.

    The revision is the only staleness signal for every cache derived from a
    tree (``SelectionCache``, the Cell's discretization, ``to_vis_data``), so
    an edit that changes the tree without a bump serves stale results, and a
    rejected edit that writes anything is visible as a changed tree or a
    consumed auto-name.
    """

    def _tree(self) -> Morphology:
        tree = Morphology.from_root(
            Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma"), name="soma"
        )
        tree.soma.dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")
        return tree

    @staticmethod
    def _state(tree: Morphology) -> tuple:
        return tree.revision, tree.n_branches, tree.naming_state(), [b.name for b in tree.branches]

    def test_failed_attaches_leave_no_trace(self) -> None:
        soma = make_soma()
        failures = [
            ("duplicate name", ValueError, dict(parent="soma", child_branch=soma, child_name="dend")),
            ("unknown parent", KeyError, dict(parent="nope", child_branch=soma, child_name="x")),
            ("reserved name", ValueError, dict(parent="soma", child_branch=soma, child_name="attach")),
            (
                "explicit name, midpoint on a non-soma parent",
                ValueError,
                dict(parent="dend", child_branch=soma, child_name="y", parent_x=0.5),
            ),
            # On main this one consumed the auto-name suffix: {} -> {'axon': 1}.
            (
                "auto name, midpoint on a non-soma parent",
                ValueError,
                dict(parent="dend", child_branch=_short_axon(), parent_x=0.5),
            ),
            ("auto name, bad child_x", ValueError, dict(parent="soma", child_branch=_short_axon(), child_x=0.5)),
        ]
        for label, error, kwargs in failures:
            with self.subTest(label):
                tree = self._tree()
                before = self._state(tree)
                with self.assertRaises(error):
                    tree.attach(**kwargs)
                self.assertEqual(self._state(tree), before)

    def test_a_failed_auto_named_attach_does_not_skip_a_name(self) -> None:
        tree = self._tree()
        with self.assertRaises(ValueError):
            tree.attach(parent="dend", child_branch=_short_axon(), parent_x=0.5)

        self.assertEqual(tree.naming_state(), {})
        self.assertEqual(tree.attach(parent="soma", child_branch=_short_axon()).name, "axon_0")
        self.assertEqual(tree.naming_state(), {"axon": 1})

    def test_failed_syntax_sugar_attaches_leave_no_trace(self) -> None:
        cases = [
            ("attribute, duplicate name", lambda t: setattr(t.soma, "dend", _short_axon()), ValueError),
            ("attribute, reserved name", lambda t: setattr(t.soma, "length", _short_axon()), ValueError),
            ("attribute, not a Branch", lambda t: setattr(t.soma, "x", 1.0), TypeError),
            (
                "subscript, midpoint on a non-soma parent",
                lambda t: setattr(t.dend[0.5], "y", _short_axon()),
                ValueError,
            ),
        ]
        for label, edit, error in cases:
            with self.subTest(label):
                tree = self._tree()
                before = self._state(tree)
                with self.assertRaises(error):
                    edit(tree)
                self.assertEqual(self._state(tree), before)

    def test_each_successful_edit_advances_the_revision_by_exactly_one(self) -> None:
        edits = [
            ("Morphology.attach", lambda t: t.attach(parent="soma", child_branch=_short_axon(), child_name="a")),
            ("MorphoBranch.attach", lambda t: t.soma.attach(_short_axon(), name="a")),
            ("attribute", lambda t: setattr(t.soma, "x", _short_axon())),
            ("subscript", lambda t: setattr(t.soma[0.5], "y", _short_axon())),
            ("subscript with child_x", lambda t: setattr(t.dend[1.0, 1.0], "z", _short_axon())),
        ]
        for label, edit in edits:
            with self.subTest(label):
                tree = self._tree()
                before = tree.revision
                edit(tree)
                self.assertEqual(tree.revision, before + 1)
                self.assertEqual(tree.n_branches, 3)

    def test_reads_do_not_advance_the_revision(self) -> None:
        tree = self._tree()
        before = tree.revision
        _ = (tree.metric, tree.branches, tree.edges, tree.to_vis_data(), tree.soma.dend.index, tree.topo())
        self.assertEqual(tree.revision, before)

    def test_nested_gates_advance_the_revision_once(self) -> None:
        tree = self._tree()
        before = tree.revision
        with tree._edit():
            with tree._edit():
                pass
            self.assertEqual(tree.revision, before)
        self.assertEqual(tree.revision, before + 1)
        self.assertFalse(tree._reindexed_after(before))

    def test_a_nested_reindex_request_moves_the_watermark_at_the_outermost_exit(self) -> None:
        tree = self._tree()
        before = tree.revision
        with tree._edit():
            with tree._edit(reindex=True):
                pass
            self.assertFalse(tree._reindexed_after(before))
        self.assertEqual(tree.revision, before + 1)
        self.assertTrue(tree._reindexed_after(before))
        self.assertFalse(tree._reindexed_after(tree.revision))

    def test_the_gate_drops_every_derived_cache(self) -> None:
        tree = make_projected_node_tree()
        # Warm every cache that a revision bump must drop.
        stale_vis = tree.to_vis_data()
        self.assertEqual(tree.total_length, 100.0 * u.um)
        self.assertEqual(tree.y_range, 80.0 * u.um)
        self.assertTrue(tree.has_full_point_geometry)
        self.assertEqual([b.name for b in tree.branch_by_order(order="type")], ["soma", "dend"])
        self.assertEqual(tree.dend.index_by(order="depth"), 1)

        longer = Branch.from_points(
            points=[[20.0, 0.0, 0.0], [20.0, 100.0, 0.0], [20.0, 200.0, 0.0]] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree.replace("dend", longer)

        self.assertEqual(tree.total_length, 220.0 * u.um)
        self.assertEqual(tree.y_range, 200.0 * u.um)
        fresh_vis = tree.to_vis_data()
        self.assertIsNot(fresh_vis, stale_vis)
        self.assertEqual(u.math.sum(fresh_vis.branches[1].lengths), 200.0 * u.um)

    def test_an_error_inside_the_gate_still_invalidates(self) -> None:
        # The gate's body only does writes that cannot fail; if one does
        # anyway, consumers must not keep serving the pre-edit caches.
        tree = self._tree()
        before = tree.revision
        _ = tree.total_length
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with tree._edit(reindex=True):
                object.__setattr__(tree.dend, "_branch", make_basal(length=7.0))
                raise RuntimeError("boom")
        self.assertEqual(tree.revision, before + 1)
        self.assertTrue(tree._reindexed_after(before))
        self.assertEqual(tree.total_length, 27.0 * u.um)
        # The depth counter unwound, so the next edit bumps again.
        tree.attach(parent="soma", child_branch=_short_axon(), child_name="a")
        self.assertEqual(tree.revision, before + 2)

    def test_private_attribute_assignment_on_a_view_is_rejected(self) -> None:
        tree = self._tree()
        view = tree.dend
        _ = tree.total_length
        before = self._state(tree)
        for attr in ("_branch", "_name", "_parent_id", "_children", "_owner", "_node_id"):
            with self.subTest(attr):
                with self.assertRaisesRegex(AttributeError, "read-only"):
                    setattr(view, attr, make_basal(length=200.0))
        with self.assertRaisesRegex(AttributeError, "cannot be deleted"):
            del view._branch
        with self.assertRaisesRegex(AttributeError, "cannot be deleted"):
            del view.dend
        self.assertEqual(self._state(tree), before)
        self.assertEqual(tree.total_length, 120.0 * u.um)
        self.assertEqual(view.branch.length, 100.0 * u.um)

    def test_copies_of_views_and_trees_still_work(self) -> None:
        import copy
        import pickle

        tree = self._tree()
        for label, clone in (
            ("deepcopy", copy.deepcopy(tree)),
            ("pickle", pickle.loads(pickle.dumps(tree))),
            ("view deepcopy", copy.deepcopy(tree.dend)._owner),
        ):
            with self.subTest(label):
                self.assertEqual(clone, tree)
                clone.dend.tip = _short_axon()
                self.assertEqual(clone.n_branches, 3)
                self.assertEqual(clone.dend.tip.parent.name, "dend")
                self.assertEqual(tree.n_branches, 2)


class MorphoFrozenSnapshotTest(unittest.TestCase):
    """A frozen tree (the snapshot an initialized Cell runs on) rejects every edit."""

    def test_every_edit_entry_point_is_rejected_without_a_trace(self) -> None:
        source = _make_tree_u()
        snapshot = frozen_snapshot(source)
        before = (snapshot.revision, snapshot.n_branches, snapshot.naming_state())
        edits = [
            ("Morphology.attach", lambda t: t.attach(parent="soma", child_branch=_short_axon(), child_name="a")),
            ("auto-named attach", lambda t: t.attach(parent="soma", child_branch=_short_axon())),
            ("MorphoBranch.attach", lambda t: t.dend.attach(_short_axon(), name="a")),
            ("attribute", lambda t: setattr(t.soma, "x", _short_axon())),
            ("subscript", lambda t: setattr(t.soma[0.5], "y", _short_axon())),
            # Frozen is reported ahead of argument errors.
            ("invalid arguments", lambda t: t.attach(parent="dend", child_branch=_short_axon(), parent_x=0.5)),
            ("gate", lambda t: t._edit().__enter__()),
        ]
        for label, edit in edits:
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, r"read-only snapshot.*reset\(\)"):
                    edit(snapshot)
                self.assertEqual((snapshot.revision, snapshot.n_branches, snapshot.naming_state()), before)

    def test_the_snapshot_equals_its_source_and_leaves_it_editable(self) -> None:
        source = _make_tree_u()
        snapshot = frozen_snapshot(source)

        self.assertIsNot(snapshot, source)
        self.assertEqual(snapshot, source)
        self.assertIsNot(snapshot.dend, source.dend)
        source.soma.x = _short_axon()
        self.assertEqual(source.n_branches, 5)
        self.assertEqual(snapshot.n_branches, 4)

    def test_the_snapshot_can_still_be_read(self) -> None:
        source = _make_tree_u()
        snapshot = frozen_snapshot(source)
        self.assertEqual(snapshot.total_length, source.total_length)
        self.assertEqual(snapshot.metric, source.metric)
        self.assertEqual(snapshot.axon.index, 2)
        self.assertEqual(snapshot.axon.parent.name, "soma")

    def test_deepcopy_and_pickle_give_an_editable_tree(self) -> None:
        import copy
        import pickle

        snapshot = frozen_snapshot(_make_tree_u())
        for label, clone in (
            ("deepcopy", copy.deepcopy(snapshot)),
            ("pickle", pickle.loads(pickle.dumps(snapshot))),
        ):
            with self.subTest(label):
                self.assertEqual(clone, snapshot)
                clone.attach(parent="soma", child_branch=_short_axon(), child_name="a")
                self.assertEqual(clone.n_branches, 5)
                with self.assertRaises(RuntimeError):
                    snapshot.attach(parent="soma", child_branch=_short_axon(), child_name="a")


class MorphoDetachedViewTest(unittest.TestCase):
    """A view of a branch that ``Morphology.remove`` removed answers geometry questions, not tree questions."""

    def setUp(self) -> None:
        self.tree = _make_tree_u()
        self.dend = self.tree.dend
        self.axon = self.tree.axon
        self.apical = self.tree.apical
        self.assertEqual((self.axon.index, self.apical.index), (2, 3))
        self.revision_before = self.tree.revision
        self.tree.remove("dend")

    def test_tree_questions_raise_with_the_branch_name(self) -> None:
        questions = {
            "index": lambda v: v.index,
            "branch_id": lambda v: v.branch_id,
            "index_by": lambda v: v.index_by(order="type"),
            "branch_order": lambda v: v.branch_order,
            "parent": lambda v: v.parent,
            "parent_id": lambda v: v.parent_id,
            "parent_x": lambda v: v.parent_x,
            "child_x": lambda v: v.child_x,
            "children": lambda v: v.children,
            "n_children": lambda v: v.n_children,
            "attach": lambda v: v.attach(_short_axon(), name="tip"),
            "subscript": lambda v: v[1.0],
            "attribute attach": lambda v: setattr(v, "tip", _short_axon()),
            "child lookup": lambda v: v.tip,
            "Morphology.attach parent": lambda v: self.tree.attach(parent=v, child_branch=_short_axon()),
        }
        for label, question in questions.items():
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, r"'dend'.*detached.*by name"):
                    question(self.dend)
        self.assertEqual(self.tree.n_branches, 3)
        self.assertEqual(self.tree.naming_state(), {})

    def test_name_and_geometry_stay_readable(self) -> None:
        self.assertEqual(self.dend.name, "dend")
        self.assertEqual(self.dend.branch.length, 100.0 * u.um)
        self.assertEqual(self.dend.length, 100.0 * u.um)
        self.assertEqual(self.dend.type, "basal_dendrite")
        self.assertEqual(self.dend.n_tapers, 1)
        self.assertIn("detached", repr(self.dend))
        self.assertIn("'dend'", repr(self.dend))

    def test_views_of_remaining_branches_follow_the_new_indices(self) -> None:
        self.assertEqual(self.axon.index, 1)
        self.assertEqual(self.apical.index, 2)
        self.assertEqual(repr(self.axon), "MorphoBranch(name='axon', type='axon', index=1)")
        self.assertEqual(self.axon.parent.name, "soma")
        self.assertEqual([b.name for b in self.tree.soma.children], ["axon", "apical"])
        self.assertEqual(self.tree.total_length, 130.0 * u.um)
        with self.assertRaises(KeyError):
            self.tree.branch(name="dend")

    def test_the_removal_moves_the_reindex_watermark(self) -> None:
        tree = self.tree
        self.assertEqual(tree.revision, self.revision_before + 1)
        self.assertTrue(tree._reindexed_after(self.revision_before))
        after_remove = tree.revision
        self.assertFalse(tree._reindexed_after(after_remove))

        tree.attach(parent="soma", child_branch=_short_axon(), child_name="late")

        self.assertFalse(tree._reindexed_after(after_remove))
        self.assertTrue(tree._reindexed_after(self.revision_before))

    def test_a_rejected_removal_leaves_no_trace_and_a_branch_with_children_goes_whole(self) -> None:
        tree = _make_tree_u()
        tree.axon.tip = _short_axon()
        before = (tree.revision, tree.n_branches)
        for label, name, error in (("unknown", "nope", KeyError), ("root", "soma", ValueError)):
            with self.subTest(label):
                with self.assertRaises(error):
                    tree.remove(name)
                self.assertEqual((tree.revision, tree.n_branches), before)

        tip = tree.tip
        removed = tree.remove("axon")

        self.assertEqual([b.name for b in removed.branches], ["axon", "tip"])
        self.assertEqual([b.name for b in tree.branches], ["soma", "dend", "apical"])
        self.assertIn("detached", repr(tip))


class MorphoEditTableTest(unittest.TestCase):
    """Morph layer of the edit table: after every edit, each read of the tree equals a cold rebuild.

    A new kind of edit (MORPH-2, MORPH-3, MORPH-9) adds a row to
    ``EDIT_CASES`` and is covered here without a new test.
    """

    def test_every_edit_case_matches_a_cold_rebuild(self) -> None:
        for case in EDIT_CASES:
            with self.subTest(case.name):
                assert_morpho_fresh(self, case)

    def test_every_sabotage_case_is_reported(self) -> None:
        for case in SABOTAGE_CASES:
            with self.subTest(case.name):
                with self.assertRaisesRegex(AssertionError, r"\n  revision: "):
                    assert_morpho_fresh(self, case)

    def test_a_replacement_that_bypasses_the_gate_is_reported_key_by_key(self) -> None:
        with self.assertRaises(AssertionError) as caught:
            assert_morpho_fresh(self, SABOTAGE_CASES[0])
        message = str(caught.exception)
        for key in ("revision", "metric", "ranges", "vis/structure_key", "vis/xyz"):
            with self.subTest(key):
                self.assertRegex(message, rf"\n  {re.escape(key)}[.\[: ]")
        self.assertIn("ranges[0]: edited (180.0, 'um') != cold rebuild (250.0, 'um')", message)

    def test_replacing_dend_on_tree_s_updates_the_totals(self) -> None:
        tree = make_tree_s()
        np.testing.assert_allclose(tree.total_length.to_decimal(u.um), 150.0, rtol=1e-12)
        np.testing.assert_allclose(tree.x_range.to_decimal(u.um), 150.0, rtol=1e-12)

        tree.replace("dend", stretched(tree.dend.branch))

        np.testing.assert_allclose(tree.total_length.to_decimal(u.um), 250.0, rtol=1e-12)
        np.testing.assert_allclose(tree.x_range.to_decimal(u.um), 250.0, rtol=1e-12)
        np.testing.assert_allclose(tree.dend.points_distal.to_decimal(u.um)[-1], [220.0, 0.0, 0.0], rtol=1e-12)

    def test_the_cold_rebuild_reads_no_cache(self) -> None:
        tree = make_tree_s()
        _ = tree.total_length, tree.branches
        object.__setattr__(tree.dend, "_branch", stretched(tree.dend.branch))

        rebuilt = cold_rebuild(tree)

        np.testing.assert_allclose(rebuilt.dend.length.to_decimal(u.um), 200.0, rtol=1e-12)
        np.testing.assert_allclose(rebuilt.total_length.to_decimal(u.um), 250.0, rtol=1e-12)
        # The edited tree still serves the stale total, so the reference did not read it.
        np.testing.assert_allclose(tree.total_length.to_decimal(u.um), 150.0, rtol=1e-12)

    def test_a_structural_write_that_bypasses_the_gate_is_reported(self) -> None:
        def attach_bypassing_the_gate(tree: Morphology) -> None:
            d1 = tree._name_to_id["d1"]
            node_id = tree._register_node(name="tip", branch=make_apical(), parent_id=d1, parent_x=1.0, child_x=0.0)
            tree._nodes[d1]._children["tip"] = node_id

        with self.assertRaises(AssertionError) as caught:
            assert_morpho_fresh(self, EditCase("attach bypassing the gate", "append", attach_bypassing_the_gate))
        message = str(caught.exception)
        for key in ("revision", "metric", "names/default", "index/type", "edges", "full_point_geometry", "vis/xyz"):
            with self.subTest(key):
                self.assertRegex(message, rf"\n  {re.escape(key)}[.\[: ]")

    def test_a_wrong_reindex_declaration_is_reported(self) -> None:
        cases = [
            (
                EditCase("removal declared as keeping indices", "delete", lambda t: t.remove("d2")),
                r"watermark: .* reindex=False(.|\n)*index: default indices moved \{'axon': \(4, 3\)\}",
            ),
            (
                EditCase(
                    "append declared as reindexing",
                    "append",
                    lambda t: t.attach(parent="soma", child_branch=make_axon(), child_name="late"),
                    reindex=True,
                ),
                r"watermark: .* reindex=True",
            ),
        ]
        for case, pattern in cases:
            with self.subTest(case.name):
                with self.assertRaisesRegex(AssertionError, pattern):
                    assert_morpho_fresh(self, case)

    def test_an_edit_that_changes_nothing_is_reported(self) -> None:
        def empty_edit(tree: Morphology) -> None:
            with tree._edit():
                pass

        with self.assertRaisesRegex(AssertionError, "no-op: the cold rebuild equals the tree before the edit"):
            assert_morpho_fresh(self, EditCase("empty gate", "replace", empty_edit))

    def test_a_layer_can_supply_its_own_reference(self) -> None:
        case = EDIT_CASES[0]
        observe = lambda tree: {"n_branches": tree.n_branches}  # noqa: E731
        assert_layer_fresh(self, case, layer="toy", warm=lambda tree: tree, observe=observe, reference=observe)
        with self.assertRaisesRegex(AssertionError, r"toy: .*\n  n_branches: edited 6 != cold rebuild 0"):
            assert_layer_fresh(
                self, case, layer="toy", warm=lambda tree: tree, observe=observe, reference=lambda _: {"n_branches": 0}
            )

    def test_the_table_and_its_trees(self) -> None:
        names = [case.name for case in EDIT_CASES + SABOTAGE_CASES]
        self.assertEqual(len(names), len(set(names)))
        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            EditCase("move", "move", lambda tree: None)

        tree = make_tree_t()
        self.assertEqual(
            [(b.name, b.index) for b in tree.branches], [("soma", 0), ("dend", 1), ("d1", 2), ("d2", 3), ("axon", 4)]
        )
        self.assertTrue(tree.has_full_point_geometry)
        self.assertEqual(cold_rebuild(tree), tree)
        self.assertEqual(morph_fingerprint(cold_rebuild(tree)), morph_fingerprint(tree))

    def test_stretched_needs_3d_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "3-D points"):
            stretched(make_basal())


def _pb(points, radii, type: str) -> Branch:
    """A branch through 3-D ``points`` (um) with ``radii`` (um) at each point."""
    return Branch.from_points(points=np.asarray(points, dtype=float) * u.um, radii=list(radii) * u.um, type=type)


def _um(quantity) -> float:
    return float(quantity.to_decimal(u.um))


class _RoundTripMixin:
    """Round trips of an edited tree against ``expected``, a tree attached directly in the edited default order.

    SWC keeps neither names nor branch boundaries, so it is compared through
    the written text and the trees read back, not through ``from_swc(to_swc(m))
    == m``.
    """

    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = pathlib.Path(directory.name)

    def assert_round_trips(self, edited: Morphology, expected: Morphology, label: str = "tree") -> None:
        self.assertEqual(edited, expected)
        loaded = Morphology.load_checkpoint(edited.save_checkpoint(self.tmp / f"{label}.bcm"))
        self.assertEqual(loaded, edited)
        self.assertEqual(loaded.naming_state(), edited.naming_state())
        written = edited.to_swc(self.tmp / f"{label}-edited.swc")
        reference = expected.to_swc(self.tmp / f"{label}-expected.swc")
        self.assertEqual(written.read_text(), reference.read_text())
        self.assertEqual(Morphology.from_swc(written), Morphology.from_swc(reference))
        self.assertEqual(cold_rebuild(edited), edited)


def _build(*records) -> Morphology:
    """``records`` are (name, branch, parent, parent_x, child_x); the first is the root."""
    name, branch, *_ = records[0]
    tree = Morphology.from_root(branch, name=name)
    for name, branch, parent, parent_x, child_x in records[1:]:
        tree.attach(parent=parent, child_branch=branch, child_name=name, parent_x=parent_x, child_x=child_x)
    return tree


class MorphoTreeWTest(unittest.TestCase):
    """Tree W, the edit tests' tree that exports to SWC."""

    def test_indices_orders_and_length(self) -> None:
        tree = make_tree_w()
        self.assertEqual(
            [(b.name, b.index) for b in tree.branches], [("soma", 0), ("dend", 1), ("d1", 2), ("d2", 3), ("axon", 4)]
        )
        self.assertEqual([b.name for b in tree.branch_by_order(order="type")], ["soma", "axon", "d1", "d2", "dend"])
        np.testing.assert_allclose(_um(tree.total_length), 260.0, rtol=1e-12)
        self.assertEqual(
            [(e.parent.name, e.child.name, e.parent_x, e.child_x) for e in tree.edges],
            [
                ("soma", "dend", 1.0, 0.0),
                ("dend", "d1", 1.0, 0.0),
                ("dend", "d2", 1.0, 0.0),
                ("soma", "axon", 0.0, 0.0),
            ],
        )

    def test_exports_to_swc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            read_back = Morphology.from_swc(make_tree_w().to_swc(pathlib.Path(directory) / "w.swc"))
        self.assertEqual(read_back.n_branches, 5)
        np.testing.assert_allclose(_um(read_back.total_length), 260.0, rtol=1e-12)


class MorphoCloneTest(_RoundTripMixin, unittest.TestCase):
    """``Morphology.clone()``: an equal, independent tree that shares the Branch geometry."""

    def _tree(self) -> Morphology:
        tree = make_tree_w()
        # Auto-named, so the clone has a counter to copy; built from points so the tree still exports to SWC.
        tree.attach(parent="axon", child_branch=_pb([[-30, 0, 0], [-40, 0, 0]], (0.5, 0.5), "axon"))
        return tree

    def test_the_clone_equals_the_tree_and_shares_its_geometry(self) -> None:
        tree = self._tree()
        revision = tree.revision
        clone = tree.clone()

        self.assertIsNot(clone, tree)
        self.assertEqual(clone, tree)
        self.assertEqual(tree.naming_state(), {"axon": 1})
        self.assertEqual(clone.naming_state(), {"axon": 1})
        for name in ("soma", "dend", "d1", "d2", "axon", "axon_0"):
            with self.subTest(name):
                self.assertIs(getattr(clone, name).branch, getattr(tree, name).branch)
                self.assertIsNot(getattr(clone, name), getattr(tree, name))
        self.assertEqual(tree.revision, revision)
        self.assertFalse(clone._frozen)
        self.assert_round_trips(clone, tree)

    def test_editing_one_tree_leaves_the_other_unchanged(self) -> None:
        tree = self._tree()
        clone = tree.clone()

        clone.attach(parent="d1", child_branch=make_apical(), child_name="tip")
        self.assertEqual((tree.n_branches, clone.n_branches), (6, 7))
        tree.attach(parent="d2", child_branch=make_apical(), child_name="tip")
        self.assertEqual(tree.tip.parent.name, "d2")
        self.assertEqual(clone.tip.parent.name, "d1")
        # The copied counter continues the sequence instead of restarting it.
        self.assertEqual(clone.attach(parent="soma", child_branch=make_axon()).name, "axon_1")

    def test_a_frozen_tree_clones_into_an_editable_one(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        clone = snapshot.clone()
        self.assertEqual(clone, snapshot)
        self.assertFalse(clone._frozen)
        clone.attach(parent="d1", child_branch=make_apical(), child_name="tip")
        self.assertEqual(clone.n_branches, 6)
        self.assertEqual(snapshot.n_branches, 5)

    def test_a_deep_chain_clones_without_recursion(self) -> None:
        tree = make_deep_chain_tree(1200)
        self.assertEqual(tree.clone(), tree)

    def test_an_unpickled_tree_with_a_name_reserved_since_is_rejected(self) -> None:
        tree = make_tree_w()
        # A tree pickled before a method of that name existed: rename d1 behind the gate's back.
        node = tree._nodes[tree._name_to_id.pop("d1")]
        object.__setattr__(node, "_name", "clone")
        tree._name_to_id["clone"] = node._node_id
        with self.assertRaisesRegex(ValueError, "'clone' is reserved by the Morphology API"):
            tree.clone()


class MorphoSubtreeTest(unittest.TestCase):
    """``Morphology.subtree(target)``: a new tree rooted at ``target``."""

    def test_the_subtree_at_dend(self) -> None:
        tree = make_tree_w()
        revision = tree.revision
        part = tree.subtree("dend")

        self.assertEqual([b.name for b in part.branches], ["dend", "d1", "d2"])
        self.assertIsNone(part.root.parent)
        self.assertEqual(
            [(e.parent.name, e.child.name, e.parent_x, e.child_x) for e in part.edges],
            [("dend", "d1", 1.0, 0.0), ("dend", "d2", 1.0, 0.0)],
        )
        self.assertEqual(part.n_branches, 3)
        np.testing.assert_allclose(_um(part.total_length), 210.0, rtol=1e-12)
        self.assertEqual(part.naming_state(), {})
        self.assertIs(part.d1.branch, tree.d1.branch)
        self.assertEqual(tree.revision, revision)
        self.assertEqual(tree, make_tree_w())

    def test_a_view_or_the_root_selects_the_same_subtree(self) -> None:
        tree = make_tree_w()
        self.assertEqual(tree.subtree(tree.dend), tree.subtree("dend"))
        self.assertEqual(tree.subtree(tree.root), tree)
        self.assertEqual(tree.subtree("d2"), Morphology.from_root(tree.d2.branch, name="d2"))

    def test_the_subtree_drops_the_target_s_own_attachment(self) -> None:
        tree = make_tree_w()
        part = tree.subtree("axon")  # attached at the soma's proximal end
        self.assertEqual(part, Morphology.from_root(tree.axon.branch, name="axon"))

    def test_a_frozen_tree_gives_an_editable_subtree(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        part = snapshot.subtree(snapshot.dend)
        self.assertEqual(part, make_tree_w().subtree("dend"))
        part.attach(parent="d1", child_branch=make_apical(), child_name="tip")
        self.assertEqual(snapshot.n_branches, 5)

    def test_a_deep_chain_subtree_without_recursion(self) -> None:
        tree = make_deep_chain_tree(1200)
        part = tree.subtree("seg_0")
        self.assertEqual(part.n_branches, 1199)
        self.assertEqual(part.root.name, "seg_0")

    def test_bad_targets_are_rejected_without_a_trace(self) -> None:
        tree = make_tree_w()
        other = make_tree_w()
        for label, target, error, pattern in (
            ("unknown name", "nope", KeyError, "nope"),
            ("not a name or view", 1, TypeError, "target must be a branch name or MorphoBranch, got int"),
            ("view of another tree", other.dend, ValueError, "different Morphology"),
        ):
            with self.subTest(label):
                before = (tree.revision, morph_fingerprint(tree))
                with self.assertRaisesRegex(error, pattern):
                    tree.subtree(target)
                self.assertEqual((tree.revision, morph_fingerprint(tree)), before)
        self.assertEqual(other, make_tree_w())

    def test_a_detached_view_is_rejected(self) -> None:
        tree = make_tree_w()
        view = tree.d2
        tree.remove("d2")
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*subtree\(\).*by name"):
            tree.subtree(view)


_W_SOMA = _pb([[0, 0, 0], [20, 0, 0]], (5.0, 5.0), "soma")
_W_DEND = _pb([[20, 0, 0], [50, 40, 0], [110, 40, 0]], (2.0, 1.5, 1.0), "basal_dendrite")
_W_D1 = _pb([[110, 40, 0], [140, 80, 0]], (1.0, 0.5), "basal_dendrite")
_W_D2 = _pb([[110, 40, 0], [150, 10, 0]], (1.0, 0.5), "basal_dendrite")
_W_AXON = _pb([[0, 0, 0], [-30, 0, 0]], (0.5, 0.5), "axon")


def _edit_state(tree: Morphology) -> tuple:
    """Everything a rejected edit must leave as it was."""
    return morph_fingerprint(tree), tree.revision, tree.naming_state(), tree._reindex_revision


class MorphoRemoveTest(_RoundTripMixin, unittest.TestCase):
    """``Morphology.remove(target)`` drops a subtree, returns it, and detaches its views (tree W)."""

    def test_removing_the_leaf_d2(self) -> None:
        tree = make_tree_w()
        held_d2, held_axon = tree.d2, tree.axon
        revision = tree.revision

        removed = tree.remove("d2")

        self.assertEqual(tree.revision, revision + 1)
        self.assertTrue(tree._reindexed_after(revision))
        self.assertEqual([(b.name, b.index) for b in tree.branches], [("soma", 0), ("dend", 1), ("d1", 2), ("axon", 3)])
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*by name"):
            held_d2.index
        self.assertEqual(held_d2.name, "d2")
        np.testing.assert_allclose(_um(held_d2.branch.length), 50.0, rtol=1e-12)
        self.assertEqual(held_axon.index, 3)
        self.assertEqual(tree.n_branches, 4)
        np.testing.assert_allclose(_um(tree.total_length), 210.0, rtol=1e-12)
        self.assertEqual([b.name for b in removed.branches], ["d2"])
        self.assertIsNone(removed.root.parent)
        np.testing.assert_allclose(_um(removed.total_length), 50.0, rtol=1e-12)
        self.assert_round_trips(
            tree,
            _build(
                ("soma", _W_SOMA, None, None, None),
                ("dend", _W_DEND, "soma", 1.0, 0.0),
                ("d1", _W_D1, "dend", 1.0, 0.0),
                ("axon", _W_AXON, "soma", 0.0, 0.0),
            ),
        )

    def test_removing_the_subtree_at_dend(self) -> None:
        tree = make_tree_w()
        held = {name: getattr(tree, name) for name in ("dend", "d1", "d2")}
        expected_removed = tree.subtree("dend")

        removed = tree.remove("dend")

        self.assertEqual([(b.name, b.index) for b in tree.branches], [("soma", 0), ("axon", 1)])
        np.testing.assert_allclose(_um(tree.total_length), 50.0, rtol=1e-12)
        for name, view in held.items():
            with self.subTest(name):
                self.assertIn("detached", repr(view))
                with self.assertRaises(RuntimeError):
                    view.parent
        self.assertEqual(removed, expected_removed)
        self.assertEqual([b.name for b in removed.branches], ["dend", "d1", "d2"])
        self.assertEqual(
            [(e.parent.name, e.child.name, e.parent_x, e.child_x) for e in removed.edges],
            [("dend", "d1", 1.0, 0.0), ("dend", "d2", 1.0, 0.0)],
        )
        np.testing.assert_allclose(_um(removed.total_length), 210.0, rtol=1e-12)
        self.assert_round_trips(tree, _build(("soma", _W_SOMA, None, None, None), ("axon", _W_AXON, "soma", 0.0, 0.0)))

    def test_a_view_selects_the_branch_to_remove(self) -> None:
        tree = make_tree_w()
        removed = tree.remove(tree.d1)
        self.assertEqual(removed.root.name, "d1")
        self.assertEqual([b.name for b in tree.branches], ["soma", "dend", "d2", "axon"])

    def test_a_removed_name_can_be_used_again(self) -> None:
        tree = make_tree_w()
        held_d2 = tree.d2
        tree.remove("d2")

        new_d2 = tree.attach(parent="dend", child_branch=_W_D2, child_name="d2")

        self.assertEqual(new_d2.index, 4)
        self.assertIn("detached", repr(held_d2))
        with self.assertRaises(RuntimeError):
            held_d2.index
        # Same structure as W, but d2 now comes last in default order.
        self.assertNotEqual(tree, make_tree_w())

    def test_the_auto_naming_counter_does_not_move_back(self) -> None:
        tree = make_tree_s()
        names = [tree.attach(parent="soma", child_branch=make_basal()).name for _ in range(2)]
        self.assertEqual(names, ["basal_dendrite_0", "basal_dendrite_1"])

        tree.remove("basal_dendrite_1")

        self.assertEqual(tree.naming_state(), {"basal_dendrite": 2})
        self.assertEqual(tree.attach(parent="soma", child_branch=make_basal()).name, "basal_dendrite_2")

    def test_the_root_cannot_be_removed_and_the_hint_works(self) -> None:
        tree = make_tree_w()
        before = _edit_state(tree)
        with self.assertRaisesRegex(ValueError, r"root.*subtree\(name\)"):
            tree.remove("soma")
        self.assertEqual(_edit_state(tree), before)
        # The suggested recovery: copy out the part to keep, then edit the copy.
        part = tree.subtree("dend")
        part.remove("d2")
        self.assertEqual([b.name for b in part.branches], ["dend", "d1"])

    def test_bad_targets_are_rejected_without_a_trace(self) -> None:
        tree = make_tree_w()
        other = make_tree_w()
        for label, target, error in (
            ("unknown name", "nope", KeyError),
            ("not a name or view", 3, TypeError),
            ("view of another tree", other.d2, ValueError),
        ):
            with self.subTest(label):
                before = _edit_state(tree)
                with self.assertRaises(error):
                    tree.remove(target)
                self.assertEqual(_edit_state(tree), before)
        self.assertEqual(other, make_tree_w())
        self.assertEqual(other.revision, make_tree_w().revision)

    def test_a_detached_view_is_rejected(self) -> None:
        tree = make_tree_w()
        held = tree.d2
        tree.remove("d2")
        before = _edit_state(tree)
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*remove\(\).*by name"):
            tree.remove(held)
        self.assertEqual(_edit_state(tree), before)

    def test_a_frozen_tree_rejects_before_checking_arguments(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        before = _edit_state(snapshot)
        for target in ("d2", "nope"):
            with self.subTest(target):
                with self.assertRaisesRegex(RuntimeError, "read-only snapshot"):
                    snapshot.remove(target)
                self.assertEqual(_edit_state(snapshot), before)

    def test_del_points_to_remove_and_changes_nothing(self) -> None:
        tree = make_tree_w()
        before = _edit_state(tree)
        with self.assertRaisesRegex(
            AttributeError, r"cannot be deleted \(tried 'dend'\).*Morphology\.remove\('dend'\)"
        ):
            del tree.soma.dend
        with self.assertRaises(AttributeError) as caught:
            del tree.soma.length
        self.assertNotIn("remove", str(caught.exception))
        self.assertEqual(_edit_state(tree), before)

    def test_a_deep_chain_is_removed_without_recursion(self) -> None:
        tree = make_deep_chain_tree(1200)
        removed = tree.remove("seg_0")
        self.assertEqual(removed.n_branches, 1199)
        self.assertEqual(tree.n_branches, 1)


_W_D1_LONG = _pb([[110, 40, 0], [170, 120, 0]], (1.0, 0.5), "basal_dendrite")  # 100 um
_W_D1_APICAL = _pb([[110, 40, 0], [140, 80, 0]], (1.0, 0.5), "apical_dendrite")
_W_DEND_LONG = _pb([[20, 0, 0], [80, 80, 0], [200, 80, 0]], (2.0, 1.5, 1.0), "basal_dendrite")  # 100 + 120 um


def _w_with(**replaced: Branch) -> Morphology:
    """Tree W attached directly, with the named branches' geometry replaced."""
    geometry = {"soma": _W_SOMA, "dend": _W_DEND, "d1": _W_D1, "d2": _W_D2, "axon": _W_AXON, **replaced}
    return _build(
        ("soma", geometry["soma"], None, None, None),
        ("dend", geometry["dend"], "soma", 1.0, 0.0),
        ("d1", geometry["d1"], "dend", 1.0, 0.0),
        ("d2", geometry["d2"], "dend", 1.0, 0.0),
        ("axon", geometry["axon"], "soma", 0.0, 0.0),
    )


def _cv_length_um(cell: Cell) -> float:
    return sum(float(cv.length.to_decimal(u.um)) for cv in cell.cvs)


class MorphoReplaceTest(_RoundTripMixin, unittest.TestCase):
    """``Morphology.replace(target, branch)`` swaps one branch's geometry; names, edges and children stay."""

    @staticmethod
    def _indices(tree: Morphology) -> dict:
        return {order: [b.name for b in tree.branch_by_order(order=order)] for order in ("default", "type")}

    def test_a_longer_leaf(self) -> None:
        tree = make_tree_w()
        cell = Cell(tree)
        np.testing.assert_allclose(_cv_length_um(cell), 260.0, rtol=1e-6)
        held = tree.d1
        indices = self._indices(tree)
        revision = tree.revision

        returned = tree.replace("d1", _W_D1_LONG)

        self.assertIs(returned, held)
        self.assertIs(held.branch, _W_D1_LONG)
        self.assertEqual(self._indices(tree), indices)
        self.assertFalse(tree._reindexed_after(revision))
        self.assertEqual(tree.revision, revision + 1)
        np.testing.assert_allclose(_um(tree.total_length), 310.0, rtol=1e-12)
        np.testing.assert_allclose(_cv_length_um(cell), 310.0, rtol=1e-6)
        self.assert_round_trips(tree, _w_with(d1=_W_D1_LONG))

    def test_children_stay_on_the_same_end_of_a_longer_parent(self) -> None:
        tree = make_tree_w()
        cell = Cell(tree)
        np.testing.assert_allclose(_um(tree.max_path_distance), 180.0, rtol=1e-12)

        tree.replace("dend", _W_DEND_LONG)

        for name in ("d1", "d2"):
            with self.subTest(name):
                child = getattr(tree, name)
                self.assertEqual((child.parent.name, child.parent_x, child.child_x), ("dend", 1.0, 0.0))
        np.testing.assert_allclose(_um(tree.total_length), 370.0, rtol=1e-12)
        np.testing.assert_allclose(_um(tree.max_path_distance), 290.0, rtol=1e-12)
        np.testing.assert_allclose(_cv_length_um(cell), 370.0, rtol=1e-6)
        # The children's points stay where they were, off the new end of dend.
        with self.assertRaisesRegex(ValueError, "cannot connect branch 'd1'"):
            tree.to_swc(self.tmp / "gap.swc")
        self.assertEqual(cold_rebuild(tree), tree)

    def test_a_length_only_tree(self) -> None:
        tree = make_tree_u()
        tree.soma.dend.tip = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")
        cell = Cell(tree)
        np.testing.assert_allclose(_cv_length_um(cell), 240.0, rtol=1e-6)

        tree.replace(
            "dend",
            Branch.from_lengths(lengths=[150.0, 50.0] * u.um, radii=[1.0, 1.0, 1.0] * u.um, type="basal_dendrite"),
        )

        np.testing.assert_allclose(_cv_length_um(cell), 340.0, rtol=1e-6)
        self.assertEqual(tree.tip.parent_x, 1.0)
        np.testing.assert_allclose(_um(tree.max_path_distance), 230.0, rtol=1e-12)

    def test_a_new_type_changes_the_type_order_only(self) -> None:
        tree = make_tree_w()
        default = [b.name for b in tree.branches]

        tree.replace("d1", _W_D1_APICAL)

        self.assertEqual([b.name for b in tree.branch_by_order(order="type")], ["soma", "axon", "d2", "dend", "d1"])
        self.assertEqual([b.name for b in tree.branches], default)
        self.assert_round_trips(tree, _w_with(d1=_W_D1_APICAL))
        # d1's samples are written with SWC type 4 (apical dendrite).
        self.assertRegex((self.tmp / "tree-edited.swc").read_text(), r"(?m)^\d+ 4 140 80 0 0\.5 \d+$")

    def test_a_soma_with_midpoint_children_keeps_the_soma_type(self) -> None:
        tree = make_tree_w()
        tree.attach(
            parent="soma",
            child_branch=_pb([[10, 0, 0], [10, 30, 0]], (1.0, 1.0), "apical_dendrite"),
            child_name="ap",
            parent_x=0.5,
        )
        before = _edit_state(tree)
        with self.assertRaisesRegex(ValueError, r"\['ap'\].*parent_x=0\.5"):
            tree.replace("soma", _pb([[0, 0, 0], [20, 0, 0]], (5.0, 5.0), "custom"))
        self.assertEqual(_edit_state(tree), before)

        tree.replace("soma", _pb([[0, 0, 0], [10, 0, 0], [40, 0, 0]], (5.0, 5.0, 5.0), "soma"))

        self.assertEqual(tree.ap.parent_x, 0.5)
        np.testing.assert_allclose(_um(tree.soma.length), 40.0, rtol=1e-12)

    def test_bad_arguments_are_rejected_without_a_trace(self) -> None:
        tree = make_tree_w()
        geometry = tree.d1.branch
        for label, call, error, pattern in (
            ("not a Branch", lambda: tree.replace("d1", 1.0), TypeError, r"replace\(\) expects new_branch"),
            ("a view", lambda: tree.replace("d1", tree.d2), ValueError, r"not a MorphoBranch.*\.branch"),
            ("unknown name", lambda: tree.replace("nope", _W_D1_LONG), KeyError, "nope"),
            ("not a name or view", lambda: tree.replace(2, _W_D1_LONG), TypeError, "got int"),
            ("view of another tree", lambda: tree.replace(make_tree_w().d1, _W_D1_LONG), ValueError, "different"),
        ):
            with self.subTest(label):
                before = _edit_state(tree)
                with self.assertRaisesRegex(error, pattern):
                    call()
                self.assertEqual(_edit_state(tree), before)
                self.assertIs(tree.d1.branch, geometry)

    def test_a_frozen_tree_rejects_before_checking_arguments(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        geometry = snapshot.d1.branch
        before = _edit_state(snapshot)
        for label, call in (
            ("valid", lambda: snapshot.replace("d1", _W_D1_LONG)),
            ("unknown name", lambda: snapshot.replace("nope", _W_D1_LONG)),
            ("not a Branch", lambda: snapshot.replace("d1", 1.0)),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, "read-only snapshot"):
                    call()
                self.assertEqual(_edit_state(snapshot), before)
        self.assertIs(snapshot.d1.branch, geometry)

    def test_a_detached_view_is_rejected(self) -> None:
        tree = make_tree_w()
        held = tree.d2
        tree.remove("d2")
        before = _edit_state(tree)
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*replace\(\).*by name"):
            tree.replace(held, _W_D1_LONG)
        self.assertEqual(_edit_state(tree), before)

    def test_the_same_geometry_again_is_still_an_edit(self) -> None:
        tree = make_tree_w()
        revision = tree.revision
        tree.replace("d1", tree.d1.branch)
        self.assertEqual(tree.revision, revision + 1)
        self.assertEqual(tree, make_tree_w())


_W_TUFT = _pb([[140, 80, 0], [140, 120, 0]], (0.5, 0.4), "apical_dendrite")  # 40 um
_W_TUFT2 = _pb([[140, 120, 0], [170, 160, 0]], (0.4, 0.3), "apical_dendrite")  # 50 um


def _tuft() -> Morphology:
    """The tree E5 grafts onto W's d1: tuft (root) -> tuft2, both apical, 90 um."""
    return _build(("tuft", _W_TUFT, None, None, None), ("tuft2", _W_TUFT2, "tuft", 1.0, 0.0))


class MorphoGraftTest(_RoundTripMixin, unittest.TestCase):
    """``Morphology.graft(other, parent=...)`` attaches a copy of another tree (tree W)."""

    def test_grafting_a_two_branch_tree_onto_d1(self) -> None:
        tree = make_tree_w()
        other = _tuft()
        held = {b.name: b for b in tree.branches}
        indices = {name: view.index for name, view in held.items()}
        other_before = (other.revision, morph_fingerprint(other))
        revision = tree.revision

        returned = tree.graft(other, parent="d1")

        self.assertEqual((returned.name, returned.index, tree.tuft2.index), ("tuft", 5, 6))
        self.assertIs(returned, tree.tuft)
        for name, view in held.items():
            with self.subTest(name):
                self.assertIs(getattr(tree, name), view)
                self.assertEqual(view.index, indices[name])
        self.assertFalse(tree._reindexed_after(revision))
        self.assertEqual(tree.revision, revision + 1)
        self.assertEqual(tree.n_branches, 7)
        np.testing.assert_allclose(_um(tree.total_length), 350.0, rtol=1e-12)
        self.assertEqual(
            [(e.parent.name, e.child.name, e.parent_x, e.child_x) for e in tree.edges[-2:]],
            [("d1", "tuft", 1.0, 0.0), ("tuft", "tuft2", 1.0, 0.0)],
        )
        self.assertIs(tree.tuft.branch, other.tuft.branch)
        self.assertEqual((other.revision, morph_fingerprint(other)), other_before)
        self.assertEqual(tree.naming_state(), {})
        expected = _w_with()
        expected.attach(parent="d1", child_branch=_W_TUFT, child_name="tuft")
        expected.attach(parent="tuft", child_branch=_W_TUFT2, child_name="tuft2")
        self.assert_round_trips(tree, expected)
        # SWC keeps no branch boundary between two unforked branches of one type.
        self.assertEqual(Morphology.from_swc(self.tmp / "tree-edited.swc").n_branches, 6)

    def test_the_call_gives_the_root_s_attachment(self) -> None:
        tree = make_tree_w()
        tree.graft(_tuft(), parent="soma", parent_x=0.5, child_x=1.0)
        self.assertEqual((tree.tuft.parent.name, tree.tuft.parent_x, tree.tuft.child_x), ("soma", 0.5, 1.0))
        self.assertEqual((tree.tuft2.parent_x, tree.tuft2.child_x), (1.0, 0.0))

    def test_a_view_selects_the_parent(self) -> None:
        tree = make_tree_w()
        self.assertEqual(tree.graft(_tuft(), parent=tree.d2).parent.name, "d2")

    def test_renaming_with_a_mapping_copies_part_of_the_tree_itself(self) -> None:
        tree = make_tree_w()
        returned = tree.graft(tree.subtree("d1"), parent="d2", rename={"d1": "d1b"})
        self.assertEqual((returned.name, returned.index), ("d1b", 5))
        self.assertEqual(tree.n_branches, 6)
        np.testing.assert_allclose(_um(tree.total_length), 310.0, rtol=1e-12)
        self.assertIs(tree.d1b.branch, tree.d1.branch)

    def test_renaming_with_a_function(self) -> None:
        tree = make_tree_w()
        tree.graft(_tuft(), parent="d1", rename=lambda name: f"{name}_copy")
        self.assertEqual([b.name for b in tree.branches][-2:], ["tuft_copy", "tuft2_copy"])

    def test_a_tree_can_graft_a_copy_of_itself(self) -> None:
        tree = make_tree_w()
        returned = tree.graft(tree, parent="d1", rename=lambda name: name + "_b")
        self.assertEqual((returned.name, returned.index), ("soma_b", 5))
        self.assertEqual(tree.n_branches, 10)
        np.testing.assert_allclose(_um(tree.total_length), 520.0, rtol=1e-12)
        self.assertEqual(
            [(b.name, b.parent.name) for b in tree.branches[5:]],
            [("soma_b", "d1"), ("dend_b", "soma_b"), ("d1_b", "dend_b"), ("d2_b", "dend_b"), ("axon_b", "soma_b")],
        )
        self.assertEqual(cold_rebuild(tree), tree)

    def test_a_frozen_tree_can_be_grafted(self) -> None:
        tree = make_tree_w()
        tree.graft(frozen_snapshot(_tuft()), parent="d1")
        self.assertEqual(tree.n_branches, 7)

    def test_the_auto_naming_counters_do_not_change(self) -> None:
        tree = make_tree_s()
        other = Morphology.from_root(make_basal(), name="stem")
        self.assertEqual(other.attach(parent="stem", child_branch=make_basal()).name, "basal_dendrite_0")

        tree.graft(other, parent="dend")

        self.assertEqual(tree.naming_state(), {})
        self.assertEqual(tree.attach(parent="soma", child_branch=make_basal()).name, "basal_dendrite_1")

    def test_rejected_grafts_change_neither_tree(self) -> None:
        tree = make_tree_w()
        other = _tuft()
        clash = Morphology.from_root(_W_D1, name="d1")
        clash.attach(parent="d1", child_branch=make_apical(), child_name="tip")
        cases = [
            ("a clashing name", lambda: tree.graft(clash, parent="axon"), ValueError, r"\['d1'\].*rename="),
            (
                "the tree itself",
                lambda: tree.graft(tree, parent="d1"),
                ValueError,
                re.escape("['soma', 'dend', 'd1', 'd2', 'axon']"),
            ),
            ("unknown parent", lambda: tree.graft(other, parent="nope"), KeyError, "nope"),
            ("midpoint off a soma", lambda: tree.graft(other, parent="d1", parent_x=0.5), ValueError, "soma"),
            ("child_x=0.5", lambda: tree.graft(other, parent="d1", child_x=0.5), ValueError, "child_x"),
            ("not a Morphology", lambda: tree.graft("x", parent="d1"), TypeError, "expects other to be a Morphology"),
            ("parent of another tree", lambda: tree.graft(other, parent=other.tuft), ValueError, "different"),
            ("reserved", lambda: tree.graft(other, parent="d1", rename={"tuft": "branches"}), ValueError, "reserved"),
            (
                "not an identifier",
                lambda: tree.graft(other, parent="d1", rename={"tuft": "1x"}),
                ValueError,
                "identifier",
            ),
            ("unknown key", lambda: tree.graft(other, parent="d1", rename={"tuf": "x"}), ValueError, r"\['tuf'\]"),
            (
                "same name twice",
                lambda: tree.graft(other, parent="d1", rename=lambda n: "same"),
                ValueError,
                r"\['same'\]",
            ),
            ("non-str name", lambda: tree.graft(other, parent="d1", rename=lambda n: 1), TypeError, "str name"),
            ("non-str mapped name", lambda: tree.graft(other, parent="d1", rename={"tuft": 2}), TypeError, "str name"),
            ("rename=1", lambda: tree.graft(other, parent="d1", rename=1), TypeError, "mapping, a callable or None"),
        ]
        for label, call, error, pattern in cases:
            with self.subTest(label):
                before, other_before = _edit_state(tree), _edit_state(other)
                with self.assertRaisesRegex(error, pattern):
                    call()
                self.assertEqual(_edit_state(tree), before)
                self.assertEqual(_edit_state(other), other_before)
        self.assertEqual(clash.n_branches, 2)

    def test_a_detached_parent_is_rejected(self) -> None:
        tree = make_tree_w()
        held = tree.d2
        tree.remove("d2")
        before = _edit_state(tree)
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*by name"):
            tree.graft(_tuft(), parent=held)
        self.assertEqual(_edit_state(tree), before)

    def test_a_frozen_tree_rejects_before_checking_arguments(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        before = _edit_state(snapshot)
        with self.assertRaisesRegex(RuntimeError, "read-only snapshot"):
            snapshot.graft("x", parent="nope")
        self.assertEqual(_edit_state(snapshot), before)


class MorphoReattachTest(_RoundTripMixin, unittest.TestCase):
    """``Morphology.reattach(target, parent=...)`` moves a subtree within the tree (tree W)."""

    def test_moving_d2_to_the_proximal_end_of_d1(self) -> None:
        tree = make_tree_w()
        held_d2, held_axon = tree.d2, tree.axon
        revision = tree.revision

        moved = tree.reattach("d2", parent="d1", parent_x=0.0)

        self.assertEqual((moved.name, moved.index), ("d2", 4))
        self.assertIsNot(moved, held_d2)
        self.assertIs(moved.branch, held_d2.branch)
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*by name"):
            held_d2.index
        self.assertEqual(held_axon.index, 3)
        self.assertEqual((moved.parent.name, moved.parent_x, moved.child_x), ("d1", 0.0, 0.0))
        self.assertEqual(tree.n_branches, 5)
        np.testing.assert_allclose(_um(tree.total_length), 260.0, rtol=1e-12)
        self.assertTrue(tree._reindexed_after(revision))
        self.assertEqual(tree.revision, revision + 1)
        self.assert_round_trips(
            tree,
            _build(
                ("soma", _W_SOMA, None, None, None),
                ("dend", _W_DEND, "soma", 1.0, 0.0),
                ("d1", _W_D1, "dend", 1.0, 0.0),
                ("axon", _W_AXON, "soma", 0.0, 0.0),
                ("d2", _W_D2, "d1", 0.0, 0.0),
            ),
        )

    def test_moving_a_subtree_equals_remove_then_graft(self) -> None:
        moved = make_tree_w()
        held = {name: getattr(moved, name) for name in ("dend", "d1", "d2")}
        moved.reattach("dend", parent="soma", parent_x=0.0)

        two_steps = make_tree_w()
        two_steps.graft(two_steps.remove("dend"), parent="soma", parent_x=0.0)

        self.assertEqual(moved, two_steps)
        self.assertEqual(
            [(b.name, b.index) for b in moved.branches],
            [("soma", 0), ("axon", 1), ("dend", 2), ("d1", 3), ("d2", 4)],
        )
        for name, view in held.items():
            with self.subTest(name):
                self.assertIn("detached", repr(view))
        self.assertEqual(cold_rebuild(moved), moved)

    def test_child_x_is_kept_unless_given(self) -> None:
        tree = make_tree_s()
        tree.attach(parent="dend", child_branch=make_basal(), child_name="rev", child_x=1.0)

        self.assertEqual(tree.reattach("rev", parent="axon").child_x, 1.0)
        self.assertEqual(tree.reattach("rev", parent="soma", child_x=0.0).child_x, 0.0)

    def test_moving_back_to_the_same_place_is_still_an_edit(self) -> None:
        tree = make_tree_w()
        revision = tree.revision
        moved = tree.reattach("dend", parent="soma")
        self.assertEqual((moved.parent_x, moved.child_x), (1.0, 0.0))
        self.assertEqual([b.name for b in tree.branches], ["soma", "axon", "dend", "d1", "d2"])
        self.assertTrue(tree._reindexed_after(revision))

    def test_a_view_gives_the_target_and_the_parent(self) -> None:
        tree = make_tree_w()
        moved = tree.reattach(tree.d2, parent=tree.axon)
        self.assertEqual(moved.parent.name, "axon")

    def test_inserting_a_branch_between_two_others(self) -> None:
        tree = make_tree_w()
        tree.attach(parent="soma", child_branch=make_basal(), child_name="hub")
        tree.reattach("dend", parent="hub")
        self.assertEqual(tree.dend.parent.name, "hub")
        self.assertEqual([b.name for b in tree.dend.children], ["d1", "d2"])

    def test_a_deep_chain_is_moved_without_recursion(self) -> None:
        tree = make_deep_chain_tree(1200)
        moved = tree.reattach("seg_600", parent="soma")
        self.assertEqual(tree.n_branches, 1200)
        self.assertEqual(moved.parent.name, "soma")
        self.assertEqual(tree.max_branch_order, 600)

    def test_rejected_moves_leave_no_trace(self) -> None:
        tree = make_tree_w()
        cases = [
            ("under a descendant", lambda: tree.reattach("dend", parent="d1"), ValueError, "descendant 'd1'.*cycle"),
            ("under itself", lambda: tree.reattach("dend", parent="dend"), ValueError, "itself.*cycle"),
            ("the root", lambda: tree.reattach("soma", parent="axon"), ValueError, "root"),
            ("unknown parent", lambda: tree.reattach("d2", parent="nope"), KeyError, "nope"),
            ("unknown target", lambda: tree.reattach("nope", parent="axon"), KeyError, "nope"),
            ("midpoint off a soma", lambda: tree.reattach("d2", parent="d1", parent_x=0.5), ValueError, "soma"),
            ("child_x=0.5", lambda: tree.reattach("d2", parent="d1", child_x=0.5), ValueError, "child_x"),
            ("target of another tree", lambda: tree.reattach(make_tree_w().d2, parent="d1"), ValueError, "different"),
            ("parent of another tree", lambda: tree.reattach("d2", parent=make_tree_w().d1), ValueError, "different"),
            ("not a name or view", lambda: tree.reattach(2, parent="d1"), TypeError, "got int"),
        ]
        for label, call, error, pattern in cases:
            with self.subTest(label):
                before = _edit_state(tree)
                with self.assertRaisesRegex(error, pattern):
                    call()
                self.assertEqual(_edit_state(tree), before)

    def test_detached_views_are_rejected(self) -> None:
        tree = make_tree_w()
        held_d2, held_d1 = tree.d2, tree.d1
        tree.remove("d2")
        tree.reattach("d1", parent="axon")
        before = _edit_state(tree)
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*reattach\(\)"):
            tree.reattach(held_d2, parent="axon")
        with self.assertRaisesRegex(RuntimeError, r"'d1'.*detached"):
            tree.reattach("axon", parent=held_d1)
        self.assertEqual(_edit_state(tree), before)

    def test_a_frozen_tree_rejects_before_checking_arguments(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        before = _edit_state(snapshot)
        for label, call in (
            ("valid", lambda: snapshot.reattach("d2", parent="d1")),
            ("a cycle", lambda: snapshot.reattach("dend", parent="d1")),
            ("unknown", lambda: snapshot.reattach("nope", parent="nope")),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, "read-only snapshot"):
                    call()
                self.assertEqual(_edit_state(snapshot), before)


_T_UM = np.array([5.0, -3.0, 2.0])
_T = _T_UM * u.um
_W_NAMES = ("soma", "dend", "d1", "d2", "axon")
_RIGID_INVARIANTS = (
    "total_length",
    "total_area",
    "total_volume",
    "mean_radius",
    "max_path_distance",
    "max_path_distance_excluding_soma",
)


def _um_array(quantity) -> np.ndarray:
    return np.asarray(quantity.to_decimal(u.um), dtype=float)


def _branch_arrays(branch: Branch) -> dict[str, np.ndarray]:
    fields = ("points_proximal", "points_distal", "lengths", "radii_proximal", "radii_distal")
    return {field: _um_array(getattr(branch, field)) for field in fields}


def _orders(tree: Morphology) -> dict[str, list]:
    return {order: [(b.name, b.index_by(order=order)) for b in tree.branches] for order in ("default", "type", "depth")}


def _tip_um(tree: Morphology, name: str) -> np.ndarray:
    return _um_array(tree.branch(name=name).branch.points_distal)[-1]


def _w_with_stub() -> Morphology:
    """Tree W with a 60 um length-only ``stub`` (no 3-D points) on the soma's proximal end."""
    tree = make_tree_w()
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(
            lengths=np.array([60.0]) * u.um, radii=np.array([0.5, 0.5]) * u.um, type="axon"
        ),
        child_name="stub",
        parent_x=0.0,
    )
    return tree


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _fixture_morphologies() -> list[tuple[str, Morphology]]:
    """Every morphology under ``FIXTURE_DIR``: the SWC files, then the ASC files."""
    paths = sorted(FIXTURE_DIR.glob("*.swc")) + sorted(FIXTURE_DIR.glob("*.asc"))
    with warnings.catch_warnings():
        # CA1, bc, goc and pc repeat a sample; the reader keeps it as a zero-length segment and says so.
        warnings.filterwarnings("ignore", r"from_points\(\) produced \d+ zero-length segment", UserWarning)
        return [
            (path.name, Morphology.from_swc(path) if path.suffix == ".swc" else Morphology.from_asc(path))
            for path in paths
        ]


def _sample_points_um(tree: Morphology, limit: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """All points in default order, and at most ``limit`` evenly spaced indices into them."""
    points = np.concatenate(
        [np.vstack([_um_array(b.branch.points_proximal)[:1], _um_array(b.branch.points_distal)]) for b in tree.branches]
    )
    return points, np.unique(np.linspace(0, len(points) - 1, min(len(points), limit)).round().astype(int))


def _pairwise_um(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


class MorphoTranslateTest(unittest.TestCase):
    """``Morphology.translate(offset, subtree=...)`` moves points in place; lengths and radii stay bit for bit."""

    def test_translating_tree_w_in_place(self) -> None:
        tree = make_tree_w()
        revision = tree.revision
        held = tree.dend
        old = held.branch
        copy = tree.clone()
        before = {b.name: _branch_arrays(b.branch) for b in tree.branches}
        orders = _orders(tree)

        returned = tree.translate(_T)

        self.assertIsNone(returned)
        self.assertEqual(tree.revision, revision + 1)
        self.assertFalse(tree._reindexed_after(revision))
        self.assertEqual(_orders(tree), orders)
        for name, arrays in before.items():
            with self.subTest(name):
                now = _branch_arrays(tree.branch(name=name).branch)
                np.testing.assert_array_equal(now["points_proximal"], arrays["points_proximal"] + _T_UM)
                np.testing.assert_array_equal(now["points_distal"], arrays["points_distal"] + _T_UM)
                for field in ("lengths", "radii_proximal", "radii_distal"):
                    np.testing.assert_array_equal(now[field], arrays[field])
        self.assertIs(held, tree.dend)
        self.assertIsNot(held.branch, old)
        np.testing.assert_array_equal(_um_array(old.points_proximal), [[20, 0, 0], [50, 40, 0]])
        self.assertIs(copy.dend.branch, old)
        np.testing.assert_array_equal(_um_array(copy.soma.points_proximal), [[0, 0, 0]])

    def test_translating_a_subtree_by_view(self) -> None:
        tree = make_tree_w()
        before = {b.name: _branch_arrays(b.branch) for b in tree.branches}

        tree.translate(_T, subtree=tree.dend)

        for name in _W_NAMES:
            with self.subTest(name):
                now = _branch_arrays(tree.branch(name=name).branch)
                shift = _T_UM if name in ("dend", "d1", "d2") else 0.0
                np.testing.assert_array_equal(now["points_distal"], before[name]["points_distal"] + shift)

    def test_a_mixed_tree_moves_the_part_with_points(self) -> None:
        tree = _w_with_stub()
        before = _edit_state(tree)
        for label, call in (
            ("whole tree", lambda: tree.translate(_T)),
            ("rotate the stub", lambda: tree.rotate(90 * u.degree, "z", subtree="stub")),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, r"1 branch\(es\) in its scope have none.*'stub'"):
                    call()
                self.assertEqual(_edit_state(tree), before)

        tree.translate(_T, subtree="dend")

        np.testing.assert_array_equal(_tip_um(tree, "d1"), [145, 77, 2])

    def test_a_length_only_tree_names_its_branches(self) -> None:
        tree = make_tree_u()
        before = _edit_state(tree)
        for label, call in (
            ("translate", lambda: tree.translate(_T)),
            ("rotate", lambda: tree.rotate(90 * u.degree, "z")),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, r"4 branch\(es\).*'soma', 'dend', 'axon', 'apical'\."):
                    call()
                self.assertEqual(_edit_state(tree), before)

    def test_a_long_list_of_length_only_branches_is_cut_at_five(self) -> None:
        tree = make_tree_u()
        for index in range(3):
            tree.attach(parent="dend", child_branch=make_basal(), child_name=f"tip{index}")
        with self.assertRaisesRegex(ValueError, r"7 branch\(es\).*'apical', 'tip0' and 2 more\. Pass subtree="):
            tree.translate(_T)


class MorphoRotateTest(unittest.TestCase):
    """``Morphology.rotate(angle, axis | matrix=..., origin=..., subtree=...)`` on tree W (right-hand rule)."""

    def assert_points_close(self, tree: Morphology, expected: Morphology) -> None:
        for name in _W_NAMES:
            with self.subTest(name):
                now = _branch_arrays(tree.branch(name=name).branch)
                want = _branch_arrays(expected.branch(name=name).branch)
                for field in ("points_proximal", "points_distal"):
                    np.testing.assert_allclose(now[field], want[field], rtol=0, atol=1e-12)

    def test_quarter_turns_about_each_axis(self) -> None:
        # cos(pi/2) is 6.1e-17 in float64, so the comparisons are not exact.
        for axis, branch, expected in (
            ("z", "dend", [-40, 110, 0]),
            ("y", "soma", [0, 0, -20]),
            ("x", "dend", [110, 0, 40]),
        ):
            with self.subTest(axis):
                tree = make_tree_w()
                revision = tree.revision
                self.assertIsNone(tree.rotate(90 * u.degree, axis=axis))
                np.testing.assert_allclose(_tip_um(tree, branch), expected, rtol=0, atol=1e-12)
                self.assertEqual(tree.revision, revision + 1)
                self.assertFalse(tree._reindexed_after(revision))
        tree = make_tree_w()
        tree.rotate(90 * u.degree, axis="z")
        np.testing.assert_allclose(_tip_um(tree, "soma"), [0, 20, 0], rtol=0, atol=1e-12)

    def test_every_form_of_the_same_rotation_agrees(self) -> None:
        reference = make_tree_w()
        reference.rotate(90 * u.degree, axis="z")
        for label, kwargs in (
            ("radians", dict(angle=np.pi / 2 * u.radian, axis="z")),
            ("unnormalized axis", dict(angle=90 * u.degree, axis=np.array([0.0, 0.0, 2.0]))),
            ("matrix", dict(matrix=np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]))),
        ):
            with self.subTest(label):
                tree = make_tree_w()
                tree.rotate(**kwargs)
                self.assert_points_close(tree, reference)

    def test_a_subtree_about_its_base(self) -> None:
        tree = make_tree_w()
        before = {b.name: _branch_arrays(b.branch) for b in tree.branches}

        tree.rotate(90 * u.degree, axis="z", origin=np.array([110.0, 40.0, 0.0]) * u.um, subtree="d1")

        np.testing.assert_allclose(_tip_um(tree, "d1"), [70, 70, 0], rtol=0, atol=1e-12)
        for name in ("soma", "dend", "d2", "axon"):
            with self.subTest(name):
                now = _branch_arrays(tree.branch(name=name).branch)
                for field, array in before[name].items():
                    np.testing.assert_array_equal(now[field], array)

    def test_a_float32_matrix_is_accepted(self) -> None:
        c, s = np.float32(np.cos(np.pi / 6)), np.float32(np.sin(np.pi / 6))
        matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        self.assertGreater(np.max(np.abs(matrix.astype(float).T @ matrix.astype(float) - np.eye(3))), 1e-9)
        tree = make_tree_w()
        names = ("total_length", "total_area", "total_volume")
        totals = [float(getattr(tree, name).mantissa) for name in names]

        tree.rotate(matrix=matrix)

        self.assertEqual([float(getattr(tree, name).mantissa) for name in names], totals)


class MorphoRigidTransformFixtureTest(unittest.TestCase):
    """R1, R3, R4 on every repository morphology, in float64.

    Two rigid transforms per tree, from ``numpy.random.default_rng(20261111)``:
    a random unit quaternion's matrix about a random origin within 300 um,
    then a random offset within 500 um; and 37 degrees about (1, 2, 3), then
    a random offset. Lengths and radii arrays are kept, so the length-based
    metrics are exactly equal; distances move by float64 rounding of
    ``R p`` (about 1e-13 um for coordinates up to 1e3 um).
    """

    def test_rigid_transforms_keep_lengths_and_distances(self) -> None:
        rng = np.random.default_rng(20261111)
        trees = _fixture_morphologies()
        self.assertEqual(len(trees), 15)
        checked = 0
        for name, tree in trees:
            points, index = _sample_points_um(tree)
            distances = _pairwise_um(points[index])
            metrics = {metric: getattr(tree, metric) for metric in _RIGID_INVARIANTS}
            farthest = float(tree.max_euclidean_distance.to_decimal(u.um))
            for kind in ("quaternion", "angle-axis"):
                with self.subTest(f"{name}: {kind}"):
                    moved = tree.clone()
                    if kind == "quaternion":
                        moved.rotate(
                            matrix=_quaternion_matrix(rng.normal(size=4)), origin=rng.uniform(-300, 300, 3) * u.um
                        )
                    else:
                        moved.rotate(37 * u.degree, axis=[1.0, 2.0, 3.0])
                    moved.translate(rng.uniform(-500, 500, 3) * u.um)
                    for metric, value in metrics.items():
                        self.assertEqual(float(getattr(moved, metric).mantissa), float(value.mantissa), metric)
                        checked += 1
                    after = _pairwise_um(_sample_points_um(moved)[0][index])
                    self.assertLessEqual(float(np.max(np.abs(after - distances))), 1e-9)
                    moved_farthest = float(moved.max_euclidean_distance.to_decimal(u.um))
                    self.assertLessEqual(abs(moved_farthest - farthest) / farthest, 1e-12)
        self.assertEqual(checked, 15 * 2 * 6)


class MorphoTransformConsumerTest(unittest.TestCase):
    """A declared Cell and the filter read the transformed coordinates (tree W)."""

    def test_a_declared_cell_reads_the_moved_positions(self) -> None:
        tree = make_tree_w()
        cell = Cell(tree, cv_policy=MaxCVLen(40 * u.um))
        self.assertEqual(cell.n_cv, 11)
        before = np.array([_um_array(context.position) for context in cell.cv_contexts])

        tree.translate(_T)

        self.assertEqual(cell.n_cv, 11)
        after = np.array([_um_array(context.position) for context in cell.cv_contexts])
        np.testing.assert_allclose(after, before + _T_UM, rtol=0, atol=1e-12)

    def test_path_distances_are_exact_after_a_rigid_transform(self) -> None:
        tree = make_tree_w()
        cell = Cell(tree, cv_policy=MaxCVLen(40 * u.um))
        before = [float(context.path_distance_from_soma.to_decimal(u.um)) for context in cell.cv_contexts]

        tree.rotate(37 * u.degree, axis=[1.0, 2.0, 3.0])
        tree.translate(_T)

        after = [float(context.path_distance_from_soma.to_decimal(u.um)) for context in cell.cv_contexts]
        self.assertEqual(len(after), 11)
        self.assertEqual(after, before)

    def test_spatial_sampling_follows_the_moved_tree(self) -> None:
        centre = np.array([120.0, 40.0, 0.0])

        def draw(tree: Morphology, at: np.ndarray) -> tuple:
            gaussian = density.spatial_gaussian(at * u.um, 20 * u.um)
            return tree.select(sample(AllRegion(), number=5, seed=7, density=gaussian)).points

        tree = make_tree_w()
        before = draw(tree, centre)
        self.assertEqual([branch for branch, _ in before], [3, 3, 2, 2, 2])

        tree.translate(_T)

        self.assertEqual(draw(tree, centre + _T_UM), before)
        self.assertNotEqual(draw(tree, centre), before)


class MorphoTransformGapTest(_RoundTripMixin, unittest.TestCase):
    """``translate(gap, subtree=child)`` closes the 3-D gap an edit leaves at a joint, which ``to_swc`` reports."""

    def test_closing_the_gap_after_a_graft(self) -> None:
        tree = make_tree_w()
        other = Morphology.from_root(_pb([[0, 0, 0], [0, 40, 0]], (0.5, 0.4), "apical_dendrite"), name="tuft")
        tree.graft(other, parent="d1")
        with self.assertRaisesRegex(ValueError, "cannot connect branch 'tuft'"):
            tree.to_swc(self.tmp / "gap.swc")
        gap = tree.d1.branch.points_distal[-1] - tree.tuft.branch.points_proximal[0]
        np.testing.assert_array_equal(_um_array(gap), [140, 80, 0])

        tree.translate(gap, subtree="tuft")

        expected = make_tree_w()
        expected.attach(parent="d1", child_branch=_W_TUFT, child_name="tuft")
        self.assert_round_trips(tree, expected)

    def test_closing_the_gaps_after_a_replace(self) -> None:
        tree = make_tree_w()
        tree.replace("dend", _W_DEND_LONG)
        with self.assertRaisesRegex(ValueError, "cannot connect branch 'd1'"):
            tree.to_swc(self.tmp / "gap.swc")
        gap = tree.dend.branch.points_distal[-1] - tree.d1.branch.points_proximal[0]
        np.testing.assert_array_equal(_um_array(gap), [90, 40, 0])
        revision = tree.revision

        tree.translate(gap, subtree="d1")
        tree.translate(gap, subtree="d2")

        self.assertEqual(tree.revision, revision + 2)
        expected = _w_with(
            dend=_W_DEND_LONG,
            d1=_pb([[200, 80, 0], [230, 120, 0]], (1.0, 0.5), "basal_dendrite"),
            d2=_pb([[200, 80, 0], [240, 50, 0]], (1.0, 0.5), "basal_dendrite"),
        )
        self.assert_round_trips(tree, expected)

    def test_closing_the_gap_after_a_reattach(self) -> None:
        tree = make_tree_w()
        tree.reattach("d2", parent="soma")
        with self.assertRaisesRegex(ValueError, "cannot connect branch 'd2'"):
            tree.to_swc(self.tmp / "gap.swc")
        gap = tree.soma.branch.points_distal[-1] - tree.d2.branch.points_proximal[0]
        np.testing.assert_array_equal(_um_array(gap), [-90, -40, 0])

        tree.translate(gap, subtree="d2")

        expected = _build(
            ("soma", _W_SOMA, None, None, None),
            ("dend", _W_DEND, "soma", 1.0, 0.0),
            ("d1", _W_D1, "dend", 1.0, 0.0),
            ("axon", _W_AXON, "soma", 0.0, 0.0),
            ("d2", _pb([[20, 0, 0], [60, -30, 0]], (1.0, 0.5), "basal_dendrite"), "soma", 1.0, 0.0),
        )
        self.assert_round_trips(tree, expected)


class MorphoTransformArgumentTest(unittest.TestCase):
    """Rejected ``translate`` / ``rotate`` calls raise at the call and leave the tree as it was (tree W)."""

    def test_bad_arguments_are_rejected_without_a_trace(self) -> None:
        tree = make_tree_w()
        other = make_tree_w()
        deg = u.degree
        for label, call, error, pattern in (
            (
                "bare array offset",
                lambda: tree.translate(np.array([5.0, -3.0, 2.0])),
                TypeError,
                "offset must satisfy unit",
            ),
            ("bare list offset", lambda: tree.translate([5, -3, 2]), TypeError, "offset must satisfy unit"),
            ("offset in mV", lambda: tree.translate(_T_UM * u.mV), TypeError, "offset must satisfy unit"),
            (
                "NaN offset",
                lambda: tree.translate(np.array([np.nan, 0, 0]) * u.um),
                ValueError,
                "offset must be finite",
            ),
            (
                "inf offset",
                lambda: tree.translate(np.array([np.inf, 0, 0]) * u.um),
                ValueError,
                "offset must be finite",
            ),
            ("2-vector offset", lambda: tree.translate(np.array([1.0, 2.0]) * u.um), ValueError, r"shape \(3,\)"),
            (
                "offset that loses precision",
                lambda: tree.translate(np.array([1e308, 0, 0]) * u.um),
                ValueError,
                "branch 'soma' invalid geometry: lengths must match point-derived",
            ),
            (
                "bare origin",
                lambda: tree.rotate(90 * deg, "z", origin=np.array([1.0, 2.0, 3.0])),
                TypeError,
                "origin must satisfy unit",
            ),
            (
                "NaN origin",
                lambda: tree.rotate(90 * deg, "z", origin=np.array([np.nan, 0, 0]) * u.um),
                ValueError,
                "origin must be finite",
            ),
            ("bare angle", lambda: tree.rotate(np.pi / 2, "z"), TypeError, "angle must satisfy unit"),
            ("angle in um", lambda: tree.rotate(90 * u.um, "z"), TypeError, "angle must satisfy unit"),
            ("no axis", lambda: tree.rotate(90 * deg), TypeError, "needs both angle and axis, or matrix; got angle"),
            ("no angle", lambda: tree.rotate(axis="z"), TypeError, "got axis"),
            ("nothing", lambda: tree.rotate(), TypeError, "got neither"),
            ("both forms", lambda: tree.rotate(90 * deg, "z", matrix=np.eye(3)), TypeError, "not both"),
            ("NaN angle", lambda: tree.rotate(np.nan * deg, "z"), ValueError, "angle must be finite"),
            (
                "array angle",
                lambda: tree.rotate(np.array([90.0, 0.0]) * deg, "z"),
                ValueError,
                "angle must be a scalar",
            ),
            ("zero axis", lambda: tree.rotate(90 * deg, axis=[0, 0, 0]), ValueError, "non-zero direction"),
            ("unknown axis", lambda: tree.rotate(90 * deg, axis="w"), ValueError, "'x', 'y', 'z' or three numbers"),
            ("2-vector axis", lambda: tree.rotate(90 * deg, axis=[0.0, 1.0]), ValueError, r"got shape \(2,\)"),
            (
                "axis in um",
                lambda: tree.rotate(90 * deg, axis=np.array([0.0, 0.0, 1.0]) * u.um),
                TypeError,
                "not a quantity",
            ),
            ("text axis", lambda: tree.rotate(90 * deg, axis=["a", "b", "c"]), TypeError, "axis is a direction"),
            ("reflection", lambda: tree.rotate(matrix=np.diag([1.0, 1.0, -1.0])), ValueError, "reflection"),
            (
                "scaling matrix",
                lambda: tree.rotate(matrix=2 * np.eye(3)),
                ValueError,
                r"not a rotation: max\|R\^T R - I\| = 3",
            ),
            ("NaN matrix", lambda: tree.rotate(matrix=np.full((3, 3), np.nan)), ValueError, "matrix must be finite"),
            ("2x2 matrix", lambda: tree.rotate(matrix=np.eye(2)), ValueError, r"shape \(3, 3\)"),
            ("matrix quantity", lambda: tree.rotate(matrix=np.eye(3) * u.um), TypeError, "not a quantity"),
            ("unknown subtree", lambda: tree.translate(_T, subtree="nope"), KeyError, "nope"),
            ("subtree not a name", lambda: tree.translate(_T, subtree=1), TypeError, "subtree must be a branch name"),
            (
                "subtree of another tree",
                lambda: tree.rotate(90 * deg, "z", subtree=other.dend),
                ValueError,
                "Subtree .*different Morphology",
            ),
            # Every type is checked before any value: the bare origin wins over the NaN angle.
            (
                "types before values",
                lambda: tree.rotate(np.nan * deg, "z", origin=np.array([1.0, 2, 3])),
                TypeError,
                "origin",
            ),
        ):
            with self.subTest(label):
                before = _edit_state(tree)
                with self.assertRaisesRegex(error, pattern):
                    call()
                self.assertEqual(_edit_state(tree), before)

    def test_a_move_past_the_largest_float_is_rejected(self) -> None:
        far = Morphology.from_root(_pb([[1.5e308, 0, 0], [1.5e308, 1e150, 0]], (1.0, 1.0), "soma"), name="soma")
        before = _edit_state(far)
        with self.assertRaisesRegex(ValueError, "branch 'soma' invalid geometry: the moved points are not finite"):
            far.translate(np.array([1e308, 0, 0]) * u.um)
        self.assertEqual(_edit_state(far), before)

    def test_a_detached_view_is_rejected(self) -> None:
        tree = make_tree_w()
        held = tree.d2
        tree.remove("d2")
        before = _edit_state(tree)
        for label, call in (
            ("translate", lambda: tree.translate(_T, subtree=held)),
            ("rotate", lambda: tree.rotate(90 * u.degree, "z", subtree=held)),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, rf"'d2'.*detached.*{label}\(\).*by name"):
                    call()
                self.assertEqual(_edit_state(tree), before)

    def test_a_frozen_tree_rejects_before_checking_arguments(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        before = _edit_state(snapshot)
        for label, call in (
            ("translate", lambda: snapshot.translate(_T)),
            ("translate a string", lambda: snapshot.translate("bad")),
            ("rotate", lambda: snapshot.rotate(90 * u.degree, "z")),
            ("rotate an unknown subtree", lambda: snapshot.rotate(subtree="nope")),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(RuntimeError, "read-only snapshot"):
                    call()
                self.assertEqual(_edit_state(snapshot), before)

    def test_the_transform_rows_change_the_points(self) -> None:
        # MORPH-12: a row that left the fingerprint unchanged could not catch a stale cache.
        rows = [case for case in EDIT_CASES if case.kind == "transform"]
        self.assertGreaterEqual(len(rows), 3)
        for case in rows:
            with self.subTest(case.name):
                tree = make_tree_t()
                before = morph_fingerprint(tree)
                case.edit(tree)
                after = morph_fingerprint(tree)
                self.assertNotEqual(after["vis/xyz"], before["vis/xyz"])


_W_TOTALS = ("total_length", "total_area", "total_volume", "max_path_distance")


def _mantissas(tree: Morphology, names) -> dict[str, float]:
    return {name: float(getattr(tree, name).mantissa) for name in names}


def _length_intervals(tree: Morphology, low: float, high: float) -> tuple:
    region = tree.select(branch_range("length", np.array([low, high]) * u.um, closed="both"))
    return tuple((int(b), float(p), float(d)) for b, p, d in region.intervals)


class MorphoScaleTest(unittest.TestCase):
    """``Morphology.scale(factor, origin=..., subtree=...)``: lengths, radii and points times one factor."""

    def test_scaling_tree_w_by_2(self) -> None:
        tree = make_tree_w()
        before = {b.name: _branch_arrays(b.branch) for b in tree.branches}
        orders = _orders(tree)
        revision = tree.revision
        self.assertEqual(
            _mantissas(tree, _W_TOTALS),
            dict(zip(_W_TOTALS, (260.0, 2214.8902325628937, 2560.3980126756815, 180.0))),
        )

        self.assertIsNone(tree.scale(2.0))

        self.assertEqual(
            _mantissas(tree, _W_TOTALS),
            dict(zip(_W_TOTALS, (520.0, 8859.560930251575, 20483.18410140545, 360.0))),
        )
        for name, arrays in before.items():
            with self.subTest(name):
                now = _branch_arrays(tree.branch(name=name).branch)
                for field, array in arrays.items():
                    np.testing.assert_array_equal(now[field], 2 * array)
        self.assertEqual(tree.revision, revision + 1)
        self.assertFalse(tree._reindexed_after(revision))
        self.assertEqual(_orders(tree), orders)

    def test_a_subtree_about_its_base(self) -> None:
        tree = make_tree_w()
        before = {b.name: _branch_arrays(b.branch) for b in tree.branches}

        tree.scale(2.0, origin=np.array([110.0, 40.0, 0.0]) * u.um, subtree="d1")

        self.assertEqual(_um(tree.d1.length), 100.0)
        np.testing.assert_array_equal(_tip_um(tree, "d1"), [170, 120, 0])
        self.assertEqual(_um(tree.total_length), 310.0)
        for name in ("soma", "dend", "d2", "axon"):
            with self.subTest(name):
                now = _branch_arrays(tree.branch(name=name).branch)
                for field, array in before[name].items():
                    np.testing.assert_array_equal(now[field], array)

    def test_a_length_only_tree_and_its_declared_cell(self) -> None:
        tree = make_tree_u()
        cell = Cell(tree)
        self.assertEqual(_um(tree.total_length), 230.0)
        np.testing.assert_allclose(_cv_length_um(cell), 230.0, rtol=1e-12)

        tree.scale(2.0)

        self.assertEqual(_um(tree.total_length), 460.0)
        np.testing.assert_allclose(_cv_length_um(cell), 460.0, rtol=1e-12)

    def test_a_mixed_tree(self) -> None:
        tree = _w_with_stub()

        tree.scale(2.0)

        self.assertEqual(_um(tree.stub.length), 120.0)
        self.assertEqual(_um(tree.total_length), 640.0)
        self.assertIsNone(tree.stub.branch.points_proximal)

    def test_a_declared_cell_and_a_length_selection_follow(self) -> None:
        tree = make_tree_w()
        cell = Cell(tree, cv_policy=MaxCVLen(40 * u.um))
        self.assertEqual(cell.n_cv, 11)
        self.assertEqual(_length_intervals(tree, 45.0, 55.0), ((2, 0.0, 1.0), (3, 0.0, 1.0)))

        tree.scale(2.0)

        self.assertEqual(cell.n_cv, 17)
        self.assertEqual([sum(1 for cv in cell.cvs if cv.branch_id == index) for index in range(5)], [1, 7, 3, 3, 3])
        np.testing.assert_allclose(_cv_length_um(cell), 520.0, rtol=1e-12)
        self.assertEqual(_length_intervals(tree, 45.0, 55.0), ())
        self.assertEqual(_length_intervals(tree, 90.0, 110.0), ((2, 0.0, 1.0), (3, 0.0, 1.0)))

    def test_bad_arguments_are_rejected_without_a_trace(self) -> None:
        tree = make_tree_w()
        for label, call, error, pattern in (
            ("zero", lambda: tree.scale(0), ValueError, "finite and > 0, got 0.0"),
            ("negative", lambda: tree.scale(-2), ValueError, "finite and > 0, got -2.0"),
            ("NaN", lambda: tree.scale(np.nan), ValueError, "finite and > 0, got nan"),
            ("inf", lambda: tree.scale(np.inf), ValueError, "finite and > 0, got inf"),
            ("bool", lambda: tree.scale(True), TypeError, "plain ratio"),
            ("quantity", lambda: tree.scale(2 * u.um), TypeError, "plain ratio"),
            ("text", lambda: tree.scale("2"), TypeError, "plain ratio"),
            ("ragged", lambda: tree.scale([1.0, [2.0, 3.0]]), TypeError, "plain ratio"),
            ("per-axis factors", lambda: tree.scale(np.array([2.0, 2.0, 1.0])), ValueError, "isotropic"),
            ("overflow", lambda: tree.scale(1e308), ValueError, "branch 'soma' invalid geometry: .* not finite"),
            (
                "bare origin",
                lambda: tree.scale(2.0, origin=np.array([1.0, 2.0, 3.0])),
                TypeError,
                "origin must satisfy",
            ),
            (
                "NaN origin",
                lambda: tree.scale(2.0, origin=np.array([np.nan, 0, 0]) * u.um),
                ValueError,
                "origin must be finite",
            ),
            ("unknown subtree", lambda: tree.scale(2.0, subtree="nope"), KeyError, "nope"),
            ("subtree not a name", lambda: tree.scale(2.0, subtree=1), TypeError, "subtree must be a branch name"),
            ("subtree of another tree", lambda: tree.scale(2.0, subtree=make_tree_w().d1), ValueError, "different"),
            ("types before values", lambda: tree.scale(-1, origin=np.array([1.0, 2, 3])), TypeError, "origin"),
        ):
            with self.subTest(label):
                before = _edit_state(tree)
                with self.assertRaisesRegex(error, pattern):
                    call()
                self.assertEqual(_edit_state(tree), before)

    def test_an_overflow_without_points_is_rejected(self) -> None:
        tree = make_tree_u()
        before = _edit_state(tree)
        with self.assertRaisesRegex(ValueError, r"branch 'soma' invalid geometry: the scaled lengths are not finite"):
            tree.scale(1e308)
        self.assertEqual(_edit_state(tree), before)

    def test_a_detached_view_and_a_frozen_tree_are_rejected(self) -> None:
        tree = make_tree_w()
        held = tree.d2
        tree.remove("d2")
        before = _edit_state(tree)
        with self.assertRaisesRegex(RuntimeError, r"'d2'.*detached.*scale\(\).*by name"):
            tree.scale(2.0, subtree=held)
        self.assertEqual(_edit_state(tree), before)

        snapshot = frozen_snapshot(make_tree_w())
        before = _edit_state(snapshot)
        with self.assertRaisesRegex(RuntimeError, "read-only snapshot"):
            snapshot.scale(-1)
        self.assertEqual(_edit_state(snapshot), before)


#: S1: exactly ``factor**power`` times the value for a factor of 2 or 0.5.
_SCALED_EXACT = {"total_length": 1, "max_path_distance": 1, "total_area": 2, "total_volume": 3}
#: S2: within rtol 1e-12 for any factor.
_SCALED_RELATIVE = {**_SCALED_EXACT, "mean_radius": 1, "max_euclidean_distance": 1}


class MorphoScaleFixtureTest(unittest.TestCase):
    """S1, S2, S3 on every repository morphology, in float64, about origins from ``default_rng(20261111)``.

    Multiplying by 2 or 0.5 rounds nothing in binary, so S1 is exact; other
    factors round each product once and each sum to about 16 eps over
    4e4 segments, well inside rtol 1e-12.
    """

    def test_scaling_multiplies_lengths_areas_volumes_and_distances(self) -> None:
        rng = np.random.default_rng(20261111)
        trees = _fixture_morphologies()
        self.assertEqual(len(trees), 15)
        exact = 0
        for name, tree in trees:
            points, index = _sample_points_um(tree)
            distances = _pairwise_um(points[index])
            base = _mantissas(tree, _SCALED_RELATIVE)
            for factor in (2.0, 0.5, 1.7, 0.37, 3.3):
                with self.subTest(f"{name}: x{factor}"):
                    scaled = tree.clone()
                    scaled.scale(factor, origin=rng.uniform(-300, 300, 3) * u.um)
                    now = _mantissas(scaled, _SCALED_RELATIVE)
                    if factor in (2.0, 0.5):
                        for metric, power in _SCALED_EXACT.items():
                            self.assertEqual(now[metric], base[metric] * factor**power, metric)
                        exact += 1
                    else:
                        for metric, power in _SCALED_RELATIVE.items():
                            want = base[metric] * factor**power
                            self.assertLessEqual(abs(now[metric] - want) / want, 1e-12, metric)
                    after = _pairwise_um(_sample_points_um(scaled)[0][index])
                    self.assertLessEqual(float(np.max(np.abs(after - factor * distances))), 1e-9)
        self.assertEqual(exact, 30)


def _split_every_segment(tree: Morphology) -> Morphology:
    """A copy of ``tree`` with every segment cut in two at its midpoint: same lines, twice the samples."""

    def split(branch: Branch) -> Branch:
        a = _branch_arrays(branch)
        mid_point = 0.5 * (a["points_proximal"] + a["points_distal"])
        mid_radius = 0.5 * (a["radii_proximal"] + a["radii_distal"])
        return dataclasses.replace(
            branch,
            lengths=np.repeat(a["lengths"] / 2, 2) * u.um,
            radii_proximal=np.column_stack([a["radii_proximal"], mid_radius]).ravel() * u.um,
            radii_distal=np.column_stack([mid_radius, a["radii_distal"]]).ravel() * u.um,
            points_proximal=np.stack([a["points_proximal"], mid_point], axis=1).reshape(-1, 3) * u.um,
            points_distal=np.stack([mid_point, a["points_distal"]], axis=1).reshape(-1, 3) * u.um,
        )

    copy = tree.clone()
    for view in copy.branches:
        copy.replace(view, split(view.branch))
    return copy


def _all_points_um(tree: Morphology) -> np.ndarray:
    return np.concatenate(
        [
            np.concatenate([_um_array(b.branch.points_proximal), _um_array(b.branch.points_distal)])
            for b in tree.branches
        ]
    )


class MorphoPrincipalAxesTest(unittest.TestCase):
    """``Morphology.principal_axes()`` and ``align_principal_axes()`` on trees W, T and S."""

    def assert_axes(self, pa: PrincipalAxes, origin, axes, variances) -> None:
        np.testing.assert_allclose(_um_array(pa.origin), origin, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pa.axes, axes, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pa.variances.to_decimal(u.um**2), variances, rtol=1e-12, atol=1e-12)

    def test_tree_w(self) -> None:
        tree = make_tree_w()
        revision = tree.revision

        pa = tree.principal_axes()

        self.assertIsInstance(pa, PrincipalAxes)
        self.assertIs(PrincipalAxes, braincell.morph.PrincipalAxes)
        self.assertEqual(pa._fields, ("origin", "axes", "variances"))
        self.assert_axes(
            pa,
            [10, 0, 0],
            [[0.954113340953361, 0.29944570896043154, 0], [-0.29944570896043154, 0.954113340953361, 0], [0, 0, 1]],
            [3019.7904390693775, 206.36833805093872, 0],
        )
        self.assertEqual(pa.axes.dtype, np.float64)
        self.assertEqual(tree.revision, revision)

    def test_aligning_tree_w(self) -> None:
        tree = make_tree_w()
        metrics = _mantissas(tree, _RIGID_INVARIANTS)
        lengths = {b.name: _branch_arrays(b.branch)["lengths"] for b in tree.branches}
        revision = tree.revision

        self.assertIsNone(tree.align_principal_axes())

        self.assertEqual(tree.revision, revision + 1)
        self.assertFalse(tree._reindexed_after(revision))
        soma = _branch_arrays(tree.soma.branch)
        np.testing.assert_allclose(
            soma["points_proximal"], [[-9.541133409533609, 2.9944570896043157, 0]], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            soma["points_distal"], [[9.541133409533609, -2.9944570896043157, 0]], rtol=0, atol=1e-12
        )
        self.assertEqual(_mantissas(tree, _RIGID_INVARIANTS), metrics)
        for name, array in lengths.items():
            np.testing.assert_array_equal(_branch_arrays(tree.branch(name=name).branch)["lengths"], array)

    def test_tree_t(self) -> None:
        pa = make_tree_t().principal_axes()
        np.testing.assert_allclose(_um_array(pa.origin), [10, 0, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            pa.variances.to_decimal(u.um**2), [3078.952716992732, 116.93833031313825, 0.626922330142492], rtol=1e-12
        )

    def test_a_collinear_tree(self) -> None:
        # All of S lies on the x axis: two variances are 0 and their axes come from y and z.
        tree = make_tree_s()
        self.assert_axes(tree.principal_axes(), [10, 0, 0], np.eye(3), [1875, 0, 0])

        tree.align_principal_axes()

        np.testing.assert_allclose(_um_array(tree.soma.branch.points), [[-10, 0, 0], [10, 0, 0]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(_tip_um(tree, "dend"), [110, 0, 0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(_tip_um(tree, "axon"), [-40, 0, 0], rtol=0, atol=1e-12)

    def test_a_tree_without_a_soma_uses_its_root(self) -> None:
        tree = Morphology.from_root(_pb([[0, 0, 0], [0, 40, 0]], (1.0, 1.0), "basal_dendrite"), name="trunk")
        tree.attach(
            parent="trunk", child_branch=_pb([[0, 40, 0], [0, 100, 0]], (1.0, 1.0), "basal_dendrite"), child_name="tip"
        )
        pa = tree.principal_axes()
        np.testing.assert_allclose(_um_array(pa.origin), [0, 20, 0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(pa.axes[0], [0, 1, 0], rtol=0, atol=1e-12)

    def test_trees_without_full_points_are_rejected(self) -> None:
        for label, build, pattern in (
            ("tree U", make_tree_u, r"4 branch\(es\).*'soma', 'dend', 'axon', 'apical'\. Use the part that has points"),
            ("tree W with a stub", _w_with_stub, r"1 branch\(es\).*'stub'"),
        ):
            tree = build()
            before = _edit_state(tree)
            for method in ("principal_axes", "align_principal_axes"):
                with self.subTest(f"{label}: {method}"):
                    with self.assertRaisesRegex(ValueError, rf"{method}\(\) needs 3-D points, .*{pattern}"):
                        getattr(tree, method)()
                    self.assertEqual(_edit_state(tree), before)

    def test_a_frozen_tree_reads_but_does_not_align(self) -> None:
        snapshot = frozen_snapshot(make_tree_w())
        before = _edit_state(snapshot)
        with self.assertRaisesRegex(RuntimeError, "read-only snapshot"):
            snapshot.align_principal_axes()
        self.assertEqual(_edit_state(snapshot), before)

        pa = snapshot.principal_axes()

        expected = make_tree_w().principal_axes()
        np.testing.assert_array_equal(_um_array(pa.origin), _um_array(expected.origin))
        np.testing.assert_array_equal(pa.axes, expected.axes)


class MorphoPrincipalAxesFixtureTest(unittest.TestCase):
    """A1-A4 on every repository morphology, in float64.

    Three of the fifteen are collinear (``branched_soma_1``,
    ``single_point_soma``, ``three_points_soma``): two variances are 0 and
    the rule for equal variances picks their axes.
    """

    def test_alignment_is_exact_pose_independent_idempotent_and_resampling_free(self) -> None:
        rng = np.random.default_rng(20261111)
        trees = _fixture_morphologies()
        self.assertEqual(len(trees), 15)
        collinear = []
        for name, tree in trees:
            with self.subTest(name):
                pa = tree.principal_axes()
                variances = np.asarray(pa.variances.to_decimal(u.um**2))
                if variances[1] <= 1e-9 * variances[0]:
                    collinear.append(name)
                aligned = tree.clone()
                aligned.align_principal_axes()

                # A1: after alignment the axes are the identity at the origin.
                again = aligned.principal_axes()
                self.assertLessEqual(float(np.max(np.abs(again.axes - np.eye(3)))), 1e-12)
                self.assertLessEqual(float(np.linalg.norm(_um_array(again.origin))), 1e-9)
                after = np.asarray(again.variances.to_decimal(u.um**2))
                self.assertLessEqual(float(np.max(np.abs(after - variances))) / variances[0], 1e-12)

                # A2: any rigid pose first gives the same aligned coordinates.
                posed = tree.clone()
                posed.rotate(matrix=_quaternion_matrix(rng.normal(size=4)))
                posed.translate(rng.uniform(-500, 500, 3) * u.um)
                posed.align_principal_axes()
                self.assertLessEqual(float(np.max(np.abs(_all_points_um(posed) - _all_points_um(aligned)))), 1e-9)

                # A3: aligning twice moves nothing.
                twice = aligned.clone()
                twice.align_principal_axes()
                self.assertLessEqual(float(np.max(np.abs(_all_points_um(twice) - _all_points_um(aligned)))), 1e-9)

                # A4: splitting every segment in two changes nothing.
                split = _split_every_segment(tree).principal_axes()
                self.assertLessEqual(float(np.max(np.abs(split.axes - pa.axes))), 1e-12)
                resampled = np.asarray(split.variances.to_decimal(u.um**2))
                self.assertLessEqual(float(np.max(np.abs(resampled - variances))) / variances[0], 1e-12)
        self.assertEqual(collinear, ["branched_soma_1.swc", "single_point_soma.swc", "three_points_soma.swc"])


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

    def test_copy_and_edit_method_names_are_reserved(self) -> None:
        # Each new public method takes a name away from branches: a tree or a
        # checkpoint that used it can no longer be built (the changelog says so).
        for name in (
            "clone",
            "subtree",
            "remove",
            "replace",
            "graft",
            "reattach",
            "translate",
            "rotate",
            "scale",
            "principal_axes",
            "align_principal_axes",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "reserved by the Morphology API"):
                    self._attach_named(self._tree(), name)

    def test_branch_type_names_stay_available_as_branch_names(self) -> None:
        # A public Morphology name is a reserved branch name, so a new method
        # named after a branch type would break every tree that uses it: a
        # ``Morphology.soma`` constructor was proposed and renamed for this.
        names = sorted(_ALLOWED_BRANCH_TYPES) + ["dend", "apic"]
        for name in names:
            with self.subTest(name=name):
                tree = Morphology.from_root(make_soma(), name=name)
                self.assertEqual(getattr(tree, name).name, name)


class MorphoFromSomaTest(unittest.TestCase):
    """``Morphology.from_soma`` builds the one-branch soma a point neuron needs."""

    def test_matches_the_two_step_construction(self) -> None:
        tree = Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um)
        two_step = Morphology.from_root(
            Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"),
            name="soma",
        )
        self.assertEqual(len(tree.branches), 1)
        self.assertEqual(tree.root.name, "soma")
        self.assertEqual(tree.root.branch.type, "soma")
        self.assertEqual(tree.root.branch, two_step.root.branch)
        self.assertEqual(tree.total_area, two_step.total_area)

    def test_cell_membrane_area_is_the_lateral_cylinder_area(self) -> None:
        tree = Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um)
        area = Cell(tree).cvs[0].area.to_decimal(u.um**2)
        # Float32 by default, so compare at its precision.
        np.testing.assert_allclose(area, 2.0 * np.pi * 10.0 * 20.0, rtol=1e-6)

    def test_defaults_give_the_single_compartment_default_area(self) -> None:
        expected = SingleCompartment(1).area.to_decimal(u.um**2)
        np.testing.assert_allclose(Morphology.from_soma().total_area.to_decimal(u.um**2), expected, rtol=1e-6)

    def test_name_is_passed_to_the_root(self) -> None:
        tree = Morphology.from_soma(name="cell_body")
        self.assertEqual(tree.root.name, "cell_body")
        self.assertEqual(tree.root.branch.type, "soma")

    def test_rejects_values_that_are_not_positive_scalar_lengths(self) -> None:
        for field in ("length", "radius"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, f"{field} must satisfy unit"):
                    Morphology.from_soma(**{field: 20.0})
                with self.assertRaisesRegex(TypeError, f"{field} must satisfy unit"):
                    Morphology.from_soma(**{field: 20.0 * u.mV})
                for bad in (0.0 * u.um, -1.0 * u.um):
                    with self.assertRaisesRegex(ValueError, f"{field} must satisfy >"):
                        Morphology.from_soma(**{field: bad})
                with self.assertRaisesRegex(ValueError, "expected 0D"):
                    Morphology.from_soma(**{field: [10.0, 20.0] * u.um})


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

    def test_vis2d_forwards_every_argument_to_braintools(self) -> None:
        tree = make_projected_node_tree()
        arguments = dict(
            region="region",
            locset="locset",
            values="values",
            layout="stem",
            shape="line",
            branch_type_colors={"soma": "#000000"},
            branch_type_edge_colors_2d={"soma": "#111111"},
            frustum_edge_linewidth_2d=1.25,
            backend="matplotlib",
            chooser="chooser",
            ax="ax",
            notebook=True,
            jupyter_backend="html",
            return_plotter=True,
            projection_plane="xz",
            min_branch_angle_deg=90.0,
            root_layout="type_split",
        )
        with mock.patch("braintools.visualize.morphology.plot2d", return_value="axes") as plot2d:
            self.assertEqual(tree.vis2d(show=False, **arguments), "axes")
        plot2d.assert_called_once_with(tree, **arguments)

    def test_vis3d_forwards_every_argument_to_braintools(self) -> None:
        tree = make_projected_node_tree()
        arguments = dict(
            region="region",
            locset="locset",
            values="values",
            mode="skeleton",
            backend="plotly",
            chooser="chooser",
            notebook=True,
            jupyter_backend="html",
            return_plotter=True,
        )
        with mock.patch("braintools.visualize.morphology.plot3d", return_value="plotter") as plot3d:
            self.assertEqual(tree.vis3d(show=False, **arguments), "plotter")
        plot3d.assert_called_once_with(tree, **arguments)

    def test_show_calls_pyplot_show_once(self) -> None:
        tree = make_projected_node_tree()
        for method, target in (("vis2d", "plot2d"), ("vis3d", "plot3d")):
            with self.subTest(method=method):
                with (
                    mock.patch(f"braintools.visualize.morphology.{target}"),
                    mock.patch("matplotlib.pyplot.show") as show,
                ):
                    getattr(tree, method)()
                    show.assert_called_once_with()

    def test_vis2d_draws_with_matplotlib(self) -> None:
        import matplotlib.axes
        import matplotlib.pyplot as plt

        self.addCleanup(plt.close, "all")
        ax = make_projected_node_tree().vis2d(layout="projected", shape="line", show=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertGreater(len(ax.lines), 0)

    def test_vis_errors_come_from_braintools(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires complete point geometry on every branch"):
            make_root_split_tree().vis3d(show=False)
        with self.assertRaisesRegex(ValueError, "Unsupported 3D mode"):
            make_projected_node_tree().vis3d(mode="layout", show=False)
        with self.assertRaisesRegex(ValueError, "Unsupported root layout 'legacy'"):
            make_root_split_tree().vis2d(root_layout="legacy", show=False)


class MorphoToVisDataTest(unittest.TestCase):
    """``Morphology.to_vis_data`` copies the tree into braintools' neutral data."""

    def test_every_field_matches_the_morphology(self) -> None:
        for make in (make_root_split_tree, make_projected_node_tree):
            tree = make()
            data = tree.to_vis_data()
            with self.subTest(fixture=make.__name__):
                self.assertEqual(len(data.branches), len(tree.branches))
                self.assertEqual(data.root_index, tree.root.index)
                for index, (view, branch) in enumerate(zip(tree.branches, data.branches)):
                    self.assertEqual(branch.index, index)
                    self.assertEqual(view.index, index)
                    self.assertEqual(branch.name, view.name)
                    self.assertEqual(branch.type, view.type)
                    self.assertEqual(branch.parent_index, None if view.parent is None else view.parent.index)
                    self.assertEqual(branch.parent_x, view.parent_x)
                    self.assertEqual(branch.child_x, view.child_x)
                    for field in ("lengths", "radii_proximal", "radii_distal", "points_proximal", "points_distal"):
                        expected = getattr(view.branch, field)
                        actual = getattr(branch, field)
                        if expected is None:
                            self.assertIsNone(actual)
                        else:
                            np.testing.assert_array_equal(actual.to_decimal(u.um), expected.to_decimal(u.um))
                self.assertEqual(
                    data.children,
                    tuple(tuple(child.index for child in view.children) for view in tree.branches),
                )

    def test_the_root_has_no_attachment(self) -> None:
        root = make_root_split_tree().to_vis_data().branches[0]
        self.assertIsNone(root.parent_index)
        self.assertIsNone(root.parent_x)
        self.assertIsNone(root.child_x)

    def test_the_snapshot_is_cached_until_the_tree_is_edited(self) -> None:
        tree = make_root_split_tree()
        first = tree.to_vis_data()
        self.assertIs(tree.to_vis_data(), first)
        tree.attach(parent="dend", child_branch=make_apical(), child_name="tuft", parent_x=1.0)
        second = tree.to_vis_data()
        self.assertIsNot(second, first)
        self.assertEqual(len(second.branches), len(first.branches) + 1)
        self.assertIs(tree.to_vis_data(), second)

    def test_point_geometry_flag(self) -> None:
        self.assertFalse(make_root_split_tree().to_vis_data().has_full_point_geometry)
        self.assertTrue(make_projected_node_tree().to_vis_data().has_full_point_geometry)

    def test_the_protocol_method_is_to_vis_data(self) -> None:
        import braintools.visualize.morphology as bvm

        tree = make_projected_node_tree()
        self.assertIs(tree.__morphology_data__(), tree.to_vis_data())
        self.assertIs(bvm.as_morphology_data(tree), tree.to_vis_data())


class MorphoSwcRender3dTest(unittest.TestCase):
    """Real SWC reconstructions render in 3D (SWC reading stays in braincell)."""

    def test_swc_fixtures_build_a_3d_scene(self) -> None:
        from braintools.visualize.morphology._tree import as_tree
        from braintools.visualize.morphology.scene3d import build_render_scene_3d

        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                tree = Morphology.from_swc(FIXTURE_DIR / fixture_name)
                scene = build_render_scene_3d(as_tree(tree))

                self.assertEqual(len(scene.branches), len(tree.branches))
                self.assertGreaterEqual(len(scene.batches), 1)
                for branch in scene.branches:
                    self.assertTrue(branch.branch_name)
                    self.assertIn(branch.branch_type, ALLOWED_TYPES)
                    self.assertEqual(branch.points_um.ndim, 2)
                    self.assertEqual(branch.points_um.shape[1], 3)
                    self.assertGreaterEqual(len(branch.points_um), 2)
                    self.assertEqual(len(branch.radii_um), len(branch.points_um))

    def test_swc_fixtures_render_with_pyvista(self) -> None:
        try:
            import pyvista as pv
        except ImportError:
            self.skipTest("pyvista is not installed.")
        from braintools.visualize.morphology.backend import BackendChooser
        from braintools.visualize.morphology.backend_pyvista import PyVistaBackend

        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                tree = Morphology.from_swc(FIXTURE_DIR / fixture_name)
                backend = PyVistaBackend(plotter_kwargs={"off_screen": True}, show_axes=False)
                plotter = tree.vis3d(backend="pyvista", chooser=BackendChooser(backends=(backend,)), show=False)

                self.assertIsInstance(plotter, pv.Plotter)
                self.assertGreater(len(plotter.renderer.actors), 0)
                plotter.close()
