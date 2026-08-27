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

from braincell import Cell
from braincell._discretization import CVPerBranch
from braincell.filter import AllRegion
from braincell.mech import CVContext, CableProperty
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology


def _tree() -> Morphology:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    tuft = Branch.from_lengths(
        lengths=[60.0] * u.um,
        radii=[1.0, 0.6] * u.um,
        type="apical_dendrite",
    )
    reverse = Branch.from_lengths(
        lengths=[40.0] * u.um,
        radii=[0.8, 0.4] * u.um,
        type="axon",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.attach(dend, name="dend", parent_x=0.5)
    tree.dend.attach(tuft, name="tuft", parent_x=1.0)
    tree.soma.attach(reverse, name="reverse", parent_x=1.0, child_x=1.0)
    return tree


def _um(values):
    return [float(value.to_decimal(u.um)) for value in values]


class CVContextTest(unittest.TestCase):
    def test_context_exposes_geometry_and_path_distances(self) -> None:
        cell = Cell(_tree(), cv_policy=CVPerBranch(cv_per_branch=2))
        contexts = cell.cv_contexts

        self.assertEqual(len(contexts), 8)
        self.assertTrue(all(isinstance(context, CVContext) for context in contexts))
        self.assertEqual([context.cv_id for context in contexts], list(range(8)))
        self.assertEqual(
            [context.branch_name for context in contexts],
            [
                "soma",
                "soma",
                "dend",
                "dend",
                "tuft",
                "tuft",
                "reverse",
                "reverse",
            ],
        )

        self.assertEqual(
            _um(context.path_distance_from_soma for context in contexts),
            [0.0, 0.0, 25.0, 75.0, 115.0, 145.0, 30.0, 10.0],
        )
        self.assertEqual(
            _um(context.path_distance_to_root for context in contexts),
            [5.0, 15.0, 35.0, 85.0, 125.0, 155.0, 50.0, 30.0],
        )
        self.assertAlmostEqual(contexts[2].length.to_decimal(u.um), 50.0)
        self.assertAlmostEqual(
            contexts[2].diam_mid.to_decimal(u.um),
            2.0 * contexts[2].radius_mid.to_decimal(u.um),
        )

        with self.assertRaises(Exception):
            contexts[0].cv_id = 3  # type: ignore[misc]

    def test_no_soma_uses_root_branch_as_reference_region(self) -> None:
        root = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="dendrite",
        )
        child = Branch.from_lengths(
            lengths=[30.0] * u.um,
            radii=[1.0, 0.5] * u.um,
            type="axon",
        )
        morpho = Morphology.from_root(root, name="trunk")
        morpho.trunk.child = child
        contexts = Cell(morpho, cv_policy=CVPerBranch(cv_per_branch=2)).cv_contexts

        self.assertEqual(_um(context.path_distance_from_soma for context in contexts), [0.0, 0.0, 7.5, 22.5])

    def test_context_position_is_cv_midpoint_when_point_geometry_exists(self) -> None:
        root = Branch.from_points(
            points=np.asarray([[0.0, 0.0, 0.0], [20.0, 4.0, 0.0]]) * u.um,
            radii=np.asarray([2.0, 1.0]) * u.um,
            type="soma",
        )
        contexts = Cell(Morphology.from_root(root, name="soma"), cv_policy=CVPerBranch(cv_per_branch=2)).cv_contexts

        np.testing.assert_allclose(contexts[0].position.to_decimal(u.um), [5.0, 1.0, 0.0])
        np.testing.assert_allclose(contexts[1].local_position.to_decimal(u.um), [15.0, 3.0, 0.0])
        self.assertIs(contexts[0].position, contexts[0].local_position)

    def test_callable_cable_property_uses_public_context(self) -> None:
        seen: list[CVContext] = []

        def capacitance(context: CVContext):
            seen.append(context)
            distance = context.path_distance_from_soma.to_decimal(u.um)
            return (1.0 + 0.001 * distance) * (u.uF / u.cm**2)

        cell = Cell(_tree(), cv_policy=CVPerBranch(cv_per_branch=2))
        cell.paint(
            AllRegion(),
            CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=capacitance,
                axial_resistivity=100.0 * (u.ohm * u.cm),
            ),
        )

        values = [float(cv.cm.to_decimal(u.uF / u.cm**2)) for cv in cell.cvs]
        expected = [
            1.0 + 0.001 * distance for distance in _um(context.path_distance_from_soma for context in cell.cv_contexts)
        ]
        self.assertEqual(values, expected)
        self.assertEqual(len(seen), cell.n_cv)
        self.assertTrue(all(isinstance(context, CVContext) for context in seen))


if __name__ == "__main__":
    unittest.main()
