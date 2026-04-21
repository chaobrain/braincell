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

from braincell.cv import CV, CVPerBranch
from braincell.cv._cv import build_cvs
from braincell.cv._lower import default_paint_rules
from braincell.filter import RegionMask
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
        cvs = build_cvs(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        self.assertEqual(len(cvs), 1)
        cv = cvs[0]
        self.assertIsInstance(cv, CV)
        with self.assertRaises(Exception):
            cv.id = 5  # type: ignore[misc]

    def test_cv_region_property(self) -> None:
        cvs = build_cvs(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
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
        cvs = build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        cv = cvs[0]
        self.assertAlmostEqual(float(cv.radius_prox.to_decimal(u.um)), 2.0)
        self.assertAlmostEqual(float(cv.radius_mid.to_decimal(u.um)), 3.0)
        self.assertAlmostEqual(float(cv.radius_dist.to_decimal(u.um)), 4.0)

    def test_diam_mid_is_twice_radius_mid(self) -> None:
        cvs = build_cvs(
            _morpho(),
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        cv = cvs[0]
        self.assertAlmostEqual(
            float(cv.diam_mid.to_decimal(u.um)),
            2.0 * float(cv.radius_mid.to_decimal(u.um)),
        )


if __name__ == "__main__":
    unittest.main()
