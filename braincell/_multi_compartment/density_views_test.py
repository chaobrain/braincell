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

import braincell
from braincell.filter import BranchSlice


def _cell():
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = braincell.Branch.from_lengths(
        lengths=[60.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="dendrite",
    )
    morpho = braincell.Morphology.from_root(soma, name="soma")
    morpho.soma.dend_a = dend
    return braincell.Cell(morpho, cv_policy=braincell.CVPerBranchList((1, 3)), pop_size=(2,))


class DensityViewTest(unittest.TestCase):
    def test_schema_default_is_visible_and_settable_before_init(self) -> None:
        cell = _cell()
        cell.paint(BranchSlice([0, 1], 0.0, 1.0), braincell.mech.Channel("IL", name="leak"))
        leak = cell.dendrite.channels["leak"]
        expected = leak.get("g_max")
        self.assertTrue(u.math.allclose(leak.g_max, expected))
        self.assertEqual(expected.shape, (len(leak),))
        leak.set(E=-65.0 * u.mV)
        self.assertTrue(u.math.allclose(leak.E, -65.0 * u.mV))

    def test_parameter_info_uses_migrated_schema(self) -> None:
        cell = _cell()
        cell.paint(BranchSlice([0, 1], 0.0, 1.0), braincell.mech.Channel("IL", name="leak"))
        self.assertEqual(tuple(cell.channels["leak"].parameter_info()), ("g_max", "E"))

    def test_same_owner_across_disjoint_cvs_has_one_view(self) -> None:
        cell = _cell()
        cell.paint(
            BranchSlice(0, 0.0, 1.0),
            braincell.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV),
        )
        cell.paint(
            BranchSlice(1, 0.0, 1.0),
            braincell.mech.Channel("IL", name="leak", g_max=0.2 * u.mS / u.cm**2, E=-65.0 * u.mV),
        )

        self.assertEqual(cell.channels.names, ("leak",))
        self.assertEqual(len(cell.channels), 8)
        np.testing.assert_allclose(cell[0].soma.channels["leak"].get("g_max").to_decimal(u.mS / u.cm**2), [0.1])
        np.testing.assert_allclose(
            cell[0].dendrite.channels["leak"].get("g_max").to_decimal(u.mS / u.cm**2),
            [0.2, 0.2, 0.2],
        )

    def test_numeric_channel_index_is_rejected(self) -> None:
        cell = _cell()
        cell.paint(
            BranchSlice(0, 0.0, 1.0),
            braincell.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV),
        )
        with self.assertRaisesRegex(TypeError, "does not support numeric"):
            _ = cell.channels[0]

    def test_set_before_and_after_init_is_population_cv_specific(self) -> None:
        cell = _cell()
        cell.paint(
            BranchSlice([0, 1], 0.0, 1.0),
            braincell.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV),
        )
        cell[1].dendrite.channels["leak"].set(g_max=0.3 * u.mS / u.cm**2)
        cell.init_state()
        np.testing.assert_allclose(
            cell[1].dendrite.channels["leak"].get("g_max").to_decimal(u.mS / u.cm**2),
            [0.3, 0.3, 0.3],
        )
        cell[1].dendrite.channels["leak"].set(g_max=0.4 * u.mS / u.cm**2)
        np.testing.assert_allclose(
            cell[1].dendrite.channels["leak"].get("g_max").to_decimal(u.mS / u.cm**2),
            [0.4, 0.4, 0.4],
            rtol=1e-6,
        )

    def test_same_owner_overlapping_cv_is_rejected(self) -> None:
        cell = _cell()
        cell.paint(
            BranchSlice(1, 0.0, 0.6),
            braincell.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV),
        )
        cell.paint(
            BranchSlice(1, 0.6, 1.0),
            braincell.mech.Channel("IL", name="leak", g_max=0.2 * u.mS / u.cm**2, E=-65.0 * u.mV),
        )
        with self.assertRaisesRegex(ValueError, "overlap after discretization"):
            _ = cell.cvs

    def test_same_category_name_cannot_change_type(self) -> None:
        cell = _cell()
        cell.paint(BranchSlice(0, 0.0, 1.0), braincell.mech.Ion("SodiumFixed", name="main"))
        cell.paint(BranchSlice(1, 0.0, 1.0), braincell.mech.Ion("PotassiumFixed", name="main"))
        with self.assertRaisesRegex(ValueError, "cannot denote both"):
            _ = cell.cvs


if __name__ == "__main__":
    unittest.main()
