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

import braincell
from braincell.filter import RootLocation, at
from braincell.network.lowering import resolve_source_cv


def _two_branch_tree() -> braincell.Morphology:
    """A soma with one section explicitly named ``dend``.

    :func:`braincell.network._testing.make_two_point_tree` builds the same
    shape but types the dendrite ``basal_dendrite``; these tests select by
    section name, so the fixture is kept local rather than shared.
    """
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dend = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="dendrite",
    )
    return morphology


class LoweringTest(unittest.TestCase):
    def test_source_location_resolves_to_canonical_cv(self) -> None:
        cell = braincell.Cell(
            _two_branch_tree(),
            cv_policy=braincell.CVPerBranch(),
            pop_size=(2,),
        )
        self.assertEqual(resolve_source_cv(cell, RootLocation(0.5)), 0)
        self.assertEqual(resolve_source_cv(cell, at("dend", 0.5)), 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_source_cv(cell, at("soma", 0.5) | at("dend", 0.5))


if __name__ == "__main__":
    unittest.main()
