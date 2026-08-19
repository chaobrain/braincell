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
from hypothesis import given, settings
from hypothesis import strategies as st

from braincell._discretization import CV, CVPerBranch
from braincell._discretization._testing import (
    build_geo,
    make_branch,
    make_single_branch_morpho,
)
from braincell._discretization.base import build_discretization
from braincell._discretization.mechanism import (
    PlaceRule,
    _coverage_fraction,
    default_paint_rules,
)
from braincell._discretization.policy import CVPolicy
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


def _build_cvs(morpho, *, policy, paint_rules, place_rules):
    return build_discretization(
        morpho,
        policy=policy,
        paint_rules=paint_rules,
        place_rules=place_rules,
    ).cvs


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
        clamp = CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
        tree = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(PlaceRule(locset=RootLocation(x=0.0), mechanisms=(clamp,)),),
        ).node_tree
        node_point_mech = tuple(node.point_mech for node in tree.nodes)
        self.assertIs(node_point_mech[tree.root_node_id][0], clamp)
        midpoint_id = int(tree.cv_to_mid_node_id[0])
        self.assertEqual(node_point_mech[midpoint_id], ())

    def test_discretization_places_interior_location_on_midpoint(self) -> None:
        clamp = CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
        tree = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(PlaceRule(locset=RootLocation(x=0.5), mechanisms=(clamp,)),),
        ).node_tree
        node_point_mech = tuple(node.point_mech for node in tree.nodes)
        midpoint_id = int(tree.cv_to_mid_node_id[0])
        self.assertIs(node_point_mech[midpoint_id][0], clamp)

    def test_discretization_places_internal_cv_boundary_on_owning_midpoint(self) -> None:
        clamp = CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
        disc = build_discretization(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(PlaceRule(locset=RootLocation(x=0.5), mechanisms=(clamp,)),),
        )
        tree = disc.node_tree
        node_point_mech = tuple(node.point_mech for node in tree.nodes)
        left_midpoint_id = int(tree.cv_to_mid_node_id[0])
        right_midpoint_id = int(tree.cv_to_mid_node_id[1])
        self.assertEqual(node_point_mech[left_midpoint_id], ())
        self.assertIs(node_point_mech[right_midpoint_id][0], clamp)


class LowerSmokeTest(unittest.TestCase):
    def test_single_branch_default_cable(self) -> None:
        morpho = make_single_branch_morpho()
        cvs = _build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        self.assertEqual(len(cvs), 2)
        self.assertIsInstance(cvs[0], CV)
        self.assertEqual(cvs[0].id, 0)
        self.assertEqual(cvs[0].branch_id, 0)
        self.assertEqual(cvs[1].parent_cv, 0)
        self.assertAlmostEqual(float(cvs[0].length.to_decimal(u.um)), 5.0)

    def test_rejects_invalid_policy_bounds(self) -> None:
        morpho = make_single_branch_morpho()

        class BadPolicy(CVPolicy):
            def resolve_cv_bounds(self, morpho, *, paint_rules=None):
                return (((0.0, 0.5),),)  # missing 0.5..1.0

        with self.assertRaises(ValueError):
            _build_cvs(
                morpho,
                policy=BadPolicy(),
                paint_rules=(),
                place_rules=(),
            )

    def test_rejects_non_policy(self) -> None:
        morpho = make_single_branch_morpho()
        with self.assertRaises(TypeError):
            _build_cvs(
                morpho,
                policy="not a policy",  # type: ignore[arg-type]
                paint_rules=(),
                place_rules=(),
            )


# =============================================================================
# Property-based invariants (skipped when hypothesis missing)
# =============================================================================


class LowerPropertyTest(unittest.TestCase):
    @given(cv_count=st.integers(min_value=1, max_value=8))
    @settings(max_examples=25, deadline=None)
    def test_cv_lengths_sum_to_branch_total(self, cv_count: int) -> None:
        morpho = Morphology.from_root(
            make_branch([30.0], [3.0, 3.0], type="soma"),
            name="soma",
        )
        cvs = _build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=cv_count),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        total_um = sum(float(cv.length.to_decimal(u.um)) for cv in cvs)
        self.assertAlmostEqual(total_um, 30.0, places=4)

    @given(cv_count=st.integers(min_value=1, max_value=8))
    @settings(max_examples=25, deadline=None)
    def test_coverage_fraction_of_all_region_is_one_per_cv(
        self,
        cv_count: int,
    ) -> None:
        morpho = Morphology.from_root(
            make_branch([20.0], [2.0, 2.0], type="soma"),
            name="soma",
        )
        bounds = CVPerBranch(cv_per_branch=cv_count).resolve_cv_bounds(morpho)
        geos, _ = build_geo(morpho, bounds)
        for g in geos:
            self.assertAlmostEqual(
                _coverage_fraction(morpho, g, ((0.0, 1.0),)),
                1.0,
                places=3,
            )

    @given(cv_count=st.integers(min_value=1, max_value=6))
    @settings(max_examples=25, deadline=None)
    def test_each_cv_id_appears_once_as_child_or_root(self, cv_count: int) -> None:
        soma = make_branch([30.0], [3.0, 3.0], type="soma")
        dend = make_branch([20.0], [2.0, 1.0], type="basal_dendrite")
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.d = dend
        cvs = _build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=cv_count),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        roots = [cv.id for cv in cvs if cv.parent_cv is None]
        children = [cid for cv in cvs for cid in cv.children_cv]
        self.assertEqual(sorted(roots + children), list(range(len(cvs))))


if __name__ == "__main__":
    unittest.main()
