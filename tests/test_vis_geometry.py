from __future__ import annotations

import unittest

import numpy as np

from ._support import u

from braincell import Branch, Morpho
from braincell.vis import build_render_geometry_3d


class VisGeometryTest(unittest.TestCase):
    def test_build_render_geometry_groups_branches_by_type(self) -> None:
        soma = Branch.xyz_shared(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        axon = Branch.xyz_shared(
            points=[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]] * u.um,
            radii=[1.0, 0.8, 0.6] * u.um,
            type="axon",
        )
        dend = Branch.xyz_shared(
            points=[[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morpho.from_root(soma, name="soma")
        tree.attach(parent="soma", child="axon_slot", branch=axon, parent_x=1.0)
        tree.attach(parent="soma", child="dend_slot", branch=dend, parent_x=1.0)

        geometry = build_render_geometry_3d(tree)

        self.assertEqual(len(geometry.branches), 3)
        self.assertEqual({batch.branch_type for batch in geometry.batches}, {"soma", "axon", "apical_dendrite"})
        soma_branch = geometry.branches[0]
        self.assertEqual(soma_branch.branch_name, "soma")
        self.assertEqual(soma_branch.points_um.shape, (2, 3))
        self.assertTrue(np.allclose(soma_branch.radii_um, np.array([5.0, 5.0])))

    def test_build_render_geometry_requires_point_geometry(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        tree = Morpho.from_root(soma, name="soma")

        with self.assertRaises(ValueError):
            build_render_geometry_3d(tree)
