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

from braincell._discretization import CV, CVPerBranch
from braincell._discretization.base import build_discretization
from braincell._discretization.mechanism import PlaceRule, default_paint_rules
from braincell.filter import RegionMask, RootLocation
from braincell.mech import CurrentClamp
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology


def _morpho() -> Morphology:
    soma = Branch.from_lengths(
        lengths=np.asarray([10.0]) * u.um,
        radii=np.asarray([2.0, 2.0]) * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


class CVShapeTest(unittest.TestCase):
    def test_cv_is_frozen(self) -> None:
        cvs = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        ).cvs
        self.assertEqual(len(cvs), 1)
        cv = cvs[0]
        self.assertIsInstance(cv, CV)
        with self.assertRaises(Exception):
            cv.id = 5  # type: ignore[misc]

    def test_cv_region_property(self) -> None:
        cvs = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(),
        ).cvs
        self.assertIsInstance(cvs[0].region, RegionMask)
        self.assertEqual(cvs[0].region.intervals, ((0, 0.0, 0.5),))
        self.assertEqual(cvs[1].region.intervals, ((0, 0.5, 1.0),))

    def test_precomputed_radii(self) -> None:
        soma = Branch.from_lengths(
            lengths=np.asarray([10.0]) * u.um,
            radii=np.asarray([2.0, 4.0]) * u.um,
            type="soma",
        )
        morpho = Morphology.from_root(soma, name="soma")
        cvs = build_discretization(
            morpho,
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        ).cvs
        cv = cvs[0]
        self.assertAlmostEqual(float(cv.radius_prox.to_decimal(u.um)), 2.0)
        self.assertAlmostEqual(float(cv.radius_mid.to_decimal(u.um)), 3.0)
        self.assertAlmostEqual(float(cv.radius_dist.to_decimal(u.um)), 4.0)

    def test_diam_mid_is_twice_radius_mid(self) -> None:
        cvs = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        ).cvs
        cv = cvs[0]
        self.assertAlmostEqual(
            float(cv.diam_mid.to_decimal(u.um)),
            2.0 * float(cv.radius_mid.to_decimal(u.um)),
        )

    def test_discretization_places_root_endpoint_on_node_tree(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms)
        tree = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(
                PlaceRule(locset=RootLocation(x=0.0), mechanisms=(clamp,)),
            ),
        ).node_tree
        node_point_mech = tuple(node.point_mech for node in tree.nodes)
        self.assertIs(node_point_mech[tree.root_node_id][0], clamp)
        midpoint_id = int(tree.cv_to_mid_node_id[0])
        self.assertEqual(node_point_mech[midpoint_id], ())

    def test_discretization_places_interior_location_on_midpoint(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms)
        tree = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(
                PlaceRule(locset=RootLocation(x=0.5), mechanisms=(clamp,)),
            ),
        ).node_tree
        node_point_mech = tuple(node.point_mech for node in tree.nodes)
        midpoint_id = int(tree.cv_to_mid_node_id[0])
        self.assertIs(node_point_mech[midpoint_id][0], clamp)

    def test_discretization_places_internal_cv_boundary_on_owning_midpoint(self) -> None:
        clamp = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms)
        disc = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(
                PlaceRule(locset=RootLocation(x=0.5), mechanisms=(clamp,)),
            ),
        )
        tree = disc.node_tree
        node_point_mech = tuple(node.point_mech for node in tree.nodes)
        left_midpoint_id = int(tree.cv_to_mid_node_id[0])
        right_midpoint_id = int(tree.cv_to_mid_node_id[1])
        self.assertEqual(node_point_mech[left_midpoint_id], ())
        self.assertIs(node_point_mech[right_midpoint_id][0], clamp)


if __name__ == "__main__":
    unittest.main()
