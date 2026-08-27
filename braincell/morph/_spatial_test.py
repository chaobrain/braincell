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

from braincell import Branch, Morphology
from braincell.morph._spatial import MorphologySpatialGeometry, interpolate_branch


def _branch(length: float, branch_type: str = "dendrite") -> Branch:
    return Branch.from_lengths(
        lengths=[length] * u.um,
        radii=[2.0, 1.0] * u.um,
        type=branch_type,
    )


class MorphologySpatialGeometryTest(unittest.TestCase):
    def test_no_soma_uses_whole_root_branch_as_reference(self) -> None:
        morpho = Morphology.from_root(_branch(20.0), name="trunk")
        morpho.trunk.attach(_branch(30.0), name="child", parent_x=1.0)
        geometry = MorphologySpatialGeometry.build(morpho)

        np.testing.assert_array_equal(
            geometry.path_distance_from_soma(0, np.asarray([0.0, 0.4, 1.0])).to_decimal(u.um),
            [0.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(geometry.path_distance_from_soma(1, 0.5).to_decimal(u.um), 15.0)
        self.assertAlmostEqual(geometry.path_distance_to_root(1, 0.5).to_decimal(u.um), 35.0)

    def test_all_soma_branches_form_one_zero_distance_region(self) -> None:
        morpho = Morphology.from_root(_branch(10.0, "soma"), name="soma_0")
        soma_1 = morpho.soma_0.attach(_branch(8.0, "soma"), name="soma_1", parent_x=1.0)
        soma_1.attach(_branch(40.0), name="dend", parent_x=0.5, child_x=1.0)
        geometry = MorphologySpatialGeometry.build(morpho)

        self.assertEqual(geometry.path_distance_from_soma(0, 0.3).to_decimal(u.um), 0.0)
        self.assertEqual(geometry.path_distance_from_soma(1, 0.7).to_decimal(u.um), 0.0)
        self.assertAlmostEqual(geometry.path_distance_from_soma(2, 0.75).to_decimal(u.um), 10.0)
        self.assertAlmostEqual(geometry.path_distance_from_soma(2, 0.0).to_decimal(u.um), 40.0)

    def test_distance_is_shortest_path_even_when_soma_is_below_root(self) -> None:
        morpho = Morphology.from_root(_branch(20.0), name="trunk")
        morpho.trunk.attach(_branch(10.0, "soma"), name="soma", parent_x=1.0)
        geometry = MorphologySpatialGeometry.build(morpho)

        self.assertAlmostEqual(geometry.path_distance_from_soma(0, 0.0).to_decimal(u.um), 20.0)
        self.assertAlmostEqual(geometry.path_distance_from_soma(0, 1.0).to_decimal(u.um), 0.0)

    def test_interpolation_returns_continuous_radius_and_position(self) -> None:
        root = Branch.from_points(
            points=np.asarray([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0]]) * u.um,
            radii=np.asarray([3.0, 1.0]) * u.um,
            type="dendrite",
        )
        morpho = Morphology.from_root(root, name="trunk")

        radius, position = interpolate_branch(morpho, 0, 0.25)
        self.assertAlmostEqual(radius.to_decimal(u.um), 2.5)
        assert position is not None
        np.testing.assert_allclose(position.to_decimal(u.um), [2.5, 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
