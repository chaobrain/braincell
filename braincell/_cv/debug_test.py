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

from braincell._cv import CVPerBranch
from braincell._cv.base import build_cvs
from braincell._cv.debug import cv_to_branch
from braincell._cv.lower import default_paint_rules
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology


class CVToBranchTest(unittest.TestCase):
    def test_roundtrip_length_matches(self) -> None:
        soma = Branch.from_lengths(
            lengths=np.asarray([10.0]) * u.um,
            radii=np.asarray([2.0, 4.0]) * u.um,
            type="soma",
        )
        morpho = Morphology.from_root(soma, name="soma")
        cvs = build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=2),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        branch = cv_to_branch(cvs[0], morpho)
        self.assertEqual(branch.type, "soma")
        self.assertAlmostEqual(
            float(branch.length.to_decimal(u.um)),
            float(cvs[0].length.to_decimal(u.um)),
            places=6,
        )

    def test_branch_type_dispatch(self) -> None:
        soma = Branch.from_lengths(
            lengths=np.asarray([10.0]) * u.um,
            radii=np.asarray([2.0, 2.0]) * u.um,
            type="soma",
        )
        dend = Branch.from_lengths(
            lengths=np.asarray([10.0]) * u.um,
            radii=np.asarray([2.0, 1.0]) * u.um,
            type="basal_dendrite",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.d = dend
        cvs = build_cvs(
            morpho,
            policy=CVPerBranch(cv_per_branch=1),
            paint_rules=default_paint_rules(),
            place_rules=(),
        )
        dend_cv = next(cv for cv in cvs if cv.branch_type == "basal_dendrite")
        reconstructed = cv_to_branch(dend_cv, morpho)
        self.assertEqual(reconstructed.type, "basal_dendrite")


if __name__ == "__main__":
    unittest.main()
