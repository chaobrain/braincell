from __future__ import annotations

import unittest
from dataclasses import is_dataclass

from ._support import FakeBackend, u

from braincell import Branch, Cell, Morpho
from braincell.filter import AllRegion, BranchSlice, RootLocation, Terminals, branch_in
from braincell.vis import BackendChooser, plot


class FilterVisTest(unittest.TestCase):
    def test_filter_and_plot_accept_morpho(self) -> None:
        soma = Branch.xyz_shared(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.xyz_shared(
            points=[[20.0, 0.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        region = AllRegion().evaluate(tree)
        backend = FakeBackend()
        rendered = plot(
            tree,
            region=region,
            dimensionality="3d",
            chooser=BackendChooser(backends=(backend,)),
        )
        cell = Cell(tree)

        self.assertEqual(len(region.intervals), 2)
        self.assertEqual(len(rendered.geometry3d.branches), 2)
        self.assertEqual(cell.n_cv, 2)
        self.assertEqual(backend.last_request.morpho.branch_by_name("soma").type, "soma")
        self.assertEqual(backend.last_request.dimensionality, "3d")
        self.assertEqual({batch.branch_type for batch in backend.last_request.geometry3d.batches}, {"soma", "apical_dendrite"})

    def test_broadcast_branch_slice_region_flows_into_plot(self) -> None:
        soma = Branch.xyz_shared(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.xyz_shared(
            points=[[20.0, 0.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        axon = Branch.xyz_shared(
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
        rendered = plot(
            tree,
            region=region,
            dimensionality="3d",
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(region.intervals, ((0, 0.0, 1.0), (2, 0.0, 0.7)))
        self.assertEqual(len(rendered.geometry3d.branches), 3)
        self.assertIs(rendered.overlay.region, region)
        self.assertEqual(backend.last_request.dimensionality, "3d")

    def test_branch_in_filter_region_flows_into_plot(self) -> None:
        soma = Branch.xyz_shared(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.xyz_shared(
            points=[[20.0, 0.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        axon = Branch.xyz_shared(
            points=[[20.0, 0.0, 0.0], [100.0, 0.0, 0.0]] * u.um,
            radii=[0.8, 0.5] * u.um,
            type="axon",
        )
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.axon = axon

        region = branch_in("type", "axon").evaluate(tree)
        backend = FakeBackend()
        rendered = plot(
            tree,
            region=region,
            dimensionality="3d",
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertEqual(region.intervals, ((2, 0.0, 1.0),))
        self.assertIs(rendered.overlay.region, region)
        self.assertEqual(len(rendered.geometry3d.branches), 3)

    def test_morpho_select_is_region_eval_sugar(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        dend = Branch.lengths_shared(lengths=[80.0], radii=[2.0, 1.0], type="apical_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        expr = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        selected = tree.select(expr)
        evaluated = expr.evaluate(tree)

        self.assertEqual(selected.intervals, evaluated.intervals)

    def test_morpho_select_accepts_locset_expr(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        dend = Branch.lengths_shared(lengths=[80.0], radii=[2.0, 1.0], type="apical_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend

        expr = RootLocation(x=0.5) | Terminals()
        selected = tree.select(expr)
        evaluated = expr.evaluate(tree)

        self.assertTrue(is_dataclass(selected))
        self.assertEqual(selected.points, evaluated.points)

    def test_morpho_select_rejects_non_filter_expr(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaises(TypeError):
            tree.select(123)  # type: ignore[arg-type]

    def test_morpho_vis3d_is_a_thin_wrapper_over_plot(self) -> None:
        soma = Branch.xyz_shared(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")
        backend = FakeBackend()

        rendered = tree.vis3d(chooser=BackendChooser(backends=(backend,)), backend="fake")

        self.assertIs(rendered, backend.last_request)
        self.assertEqual(rendered.dimensionality, "3d")
        self.assertEqual(len(rendered.geometry3d.branches), 1)

    def test_morpho_vis2d_is_reserved_but_not_implemented(self) -> None:
        soma = Branch.xyz_shared(
            points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaises(NotImplementedError):
            tree.vis2d()

    def test_branch_views_are_rejected_by_downstream_entrypoints(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        tree = Morpho.from_root(soma, name="soma")
        backend = FakeBackend()

        with self.assertRaises(TypeError):
            plot(tree.soma, chooser=BackendChooser(backends=(backend,)))
        with self.assertRaises(TypeError):
            Cell(tree.soma)
        with self.assertRaises(TypeError):
            AllRegion().evaluate(tree.soma)
