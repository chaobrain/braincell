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
from dataclasses import is_dataclass

import numpy as np

from braincell._test_support import FakeBackend, u

from braincell import Branch, Cell, Morpho
from braincell.filter import AllRegion, BranchSlice, RootLocation, Terminals, branch_in
from braincell.vis import BackendChooser, plot2d, plot3d


class FilterVisTest(unittest.TestCase):
    def test_filter_and_plot_accept_morpho(self) -> None:
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
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        region = AllRegion().evaluate(tree)
        backend = FakeBackend()
        rendered = plot3d(
            tree,
            region=region,
            chooser=BackendChooser(backends=(backend,)),
        )
        cell = Cell(tree)

        self.assertEqual(len(region.intervals), 2)
        self.assertEqual(len(rendered.scene.branches), 2)
        self.assertEqual(cell.n_cv, 2)
        self.assertEqual(backend.last_request.morpho.branch(name="soma").type, "soma")
        self.assertEqual(backend.last_request.dimensionality, "3d")
        self.assertEqual({batch.branch_type for batch in backend.last_request.scene.batches}, {"soma", "apical_dendrite"})

    def test_broadcast_branch_slice_region_flows_into_plot(self) -> None:
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
        axon = Branch.from_points(
            points=[[20.0, 0.0, 0.0], [100.0, 0.0, 0.0]] * u.um,
            radii=[0.8, 0.5] * u.um,
            type="axon",
        )
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.axon = axon

        region = BranchSlice(
            branch_index=[0, 2],
            prox=0.0,
            dist=[1.0, 0.7],
        ).evaluate(tree)
        backend = FakeBackend()
        rendered = plot3d(
            tree,
            region=region,
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(region.intervals, ((0, 0.0, 1.0), (2, 0.0, 0.7)))
        self.assertEqual(len(rendered.scene.branches), 3)
        self.assertIs(rendered.overlay.region, region)
        self.assertEqual(backend.last_request.dimensionality, "3d")

    def test_branch_in_filter_region_flows_into_plot(self) -> None:
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
        axon = Branch.from_points(
            points=[[20.0, 0.0, 0.0], [100.0, 0.0, 0.0]] * u.um,
            radii=[0.8, 0.5] * u.um,
            type="axon",
        )
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.axon = axon

        region = branch_in("type", "axon").evaluate(tree)
        backend = FakeBackend()
        rendered = plot3d(
            tree,
            region=region,
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(region.intervals, ((2, 0.0, 1.0),))
        self.assertIs(rendered.overlay.region, region)
        self.assertEqual(len(rendered.scene.branches), 3)

    def test_morpho_select_is_region_eval_sugar(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 1.0] * u.um, type="apical_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        expr = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        selected = tree.select(expr)
        evaluated = expr.evaluate(tree)

        self.assertEqual(selected.intervals, evaluated.intervals)

    def test_morpho_select_accepts_locset_expr(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[80.0] * u.um, radii=[2.0, 1.0] * u.um, type="apical_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        expr = RootLocation(x=0.5) | Terminals()
        selected = tree.select(expr)
        evaluated = expr.evaluate(tree)

        self.assertTrue(is_dataclass(selected))
        self.assertEqual(selected.points, evaluated.points)

    def test_morpho_select_rejects_non_filter_expr(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaises(TypeError):
            tree.select(123)  # type: ignore[arg-type]

    def test_morpho_vis3d_is_a_thin_wrapper_over_plot3d(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")
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
        tree = Morpho.from_root(soma, name="soma")
        backend = FakeBackend()

        rendered = tree.vis3d(mode="geometry", chooser=BackendChooser(backends=(backend,)), backend="fake")

        self.assertEqual(rendered.mode, "geometry")

    def test_morpho_vis3d_rejects_unknown_mode(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

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
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        with self.assertRaisesRegex(ValueError, "3D visualization requires full point geometry on every branch"):
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
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        rendered = tree.vis2d(chooser=BackendChooser(backends=(FakeBackend(),)), backend="fake")

        self.assertEqual(rendered.dimensionality, "2d")
        self.assertEqual(rendered.mode, "projected")
        self.assertIsNotNone(rendered.scene)
        self.assertEqual(rendered.scene.mode, "projected")
        self.assertEqual(rendered.scene.projection_plane, "xy")
        self.assertEqual(len(rendered.scene.polylines), 2)
        self.assertTrue(np.allclose(rendered.scene.polylines[1].points_um[:, 0], np.array([20.0, 20.0])))

    def test_morpho_vis2d_accepts_layout_parameters(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend_a = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="apical_dendrite")
        dend_b = Branch.from_lengths(lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
        tree.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)

        rendered = tree.vis2d(
            mode="tree",
            min_branch_angle_deg=90.0,
            root_layout="legacy",
            layout_family="stem",
            chooser=BackendChooser(backends=(FakeBackend(),)),
            backend="fake",
        )

        self.assertEqual(rendered.mode, "tree")
        child_angles = sorted(
            np.degrees(np.arctan2(polyline.points_um[-1, 1] - polyline.points_um[0, 1], polyline.points_um[-1, 0] - polyline.points_um[0, 0]))
            for polyline in rendered.scene.polylines
            if polyline.branch_name in {"dend_a", "dend_b"}
        )
        self.assertGreaterEqual(child_angles[1] - child_angles[0], 90.0 - 1e-6)

    def test_branch_views_are_rejected_by_downstream_entrypoints(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morpho.from_root(soma, name="soma")
        backend = FakeBackend()

        with self.assertRaises(TypeError):
            plot3d(tree.soma, chooser=BackendChooser(backends=(backend,)))
        with self.assertRaises(TypeError):
            plot2d(tree.soma, chooser=BackendChooser(backends=(backend,)))
        with self.assertRaises(TypeError):
            Cell(tree.soma)
        with self.assertRaises(TypeError):
            AllRegion().evaluate(tree.soma)
