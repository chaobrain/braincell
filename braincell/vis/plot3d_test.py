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

"""Tests for :mod:`braincell.vis.plot3d`."""

import unittest

import brainunit as u

from braincell import Branch, Cell, Morphology
from braincell.filter import AllRegion, BranchSlice, branch_in
from braincell.vis import plot3d
from braincell.vis._testing import (
    FakeBackend,
    VisDefaultsResetMixin,
    make_length_only_tree,
    make_node_tree,
)
from braincell.vis.backend import BackendChooser


class FilterRegionIntoPlot3dTest(unittest.TestCase):
    """A region built by braincell.filter must survive the trip into plot3d."""

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
        tree = Morphology.from_root(soma, name="soma")
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
        self.assertEqual(
            {batch.branch_type for batch in backend.last_request.scene.batches}, {"soma", "apical_dendrite"}
        )

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
        tree = Morphology.from_root(soma, name="soma")
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
        tree = Morphology.from_root(soma, name="soma")
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


class Plot3dDispatchTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_plot3d_requires_points_and_suggests_2d_fallbacks(self) -> None:
        tree = make_length_only_tree()

        with self.assertRaisesRegex(
            ValueError, r"vis2d\(layout='stem', shape='line'\).+vis2d\(layout='stem', shape='frustum'\)"
        ):
            plot3d(tree, chooser=BackendChooser(backends=(FakeBackend(),)))

    def test_plot3d_rejects_unknown_mode(self) -> None:
        tree = make_node_tree()

        with self.assertRaisesRegex(ValueError, "Unsupported 3D mode"):
            plot3d(tree, mode="projected")

    def test_plot3d_accepts_skeleton_mode(self) -> None:
        tree = make_node_tree()
        backend = FakeBackend()

        request = plot3d(tree, mode="skeleton", chooser=BackendChooser(backends=(backend,)))

        self.assertEqual(request.mode, "skeleton")
        self.assertEqual(request.scene.mode, "skeleton")


if __name__ == "__main__":
    unittest.main()
