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
from braincell import Branch, Cell, Morphology
from braincell.filter import BranchSlice


def _simple_cell() -> Cell:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
    cell = Cell(tree)
    cell.paint(
        BranchSlice(branch_index=0, prox=0.0, dist=1.0),
        braincell.mech.Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
    )
    cell.init_state()
    return cell


class MechanismObjectCellAttrAccess(unittest.TestCase):
    def test_known_param_returns_value(self) -> None:
        cell = _simple_cell()
        table = cell.mech_table()
        mo = table.get(("IL", "IL"), column_id=1)
        self.assertIsNotNone(mo)
        self.assertAlmostEqual(float(mo.g_max.to_decimal(u.mS / u.cm ** 2)), 4.0, places=12)

    def test_unknown_param_raises_attribute_error_with_candidates(self) -> None:
        cell = _simple_cell()
        table = cell.mech_table()
        mo = table.get(("IL", "IL"), column_id=1)
        with self.assertRaises(AttributeError) as ctx:
            _ = mo.not_a_real_field
        msg = str(ctx.exception)
        self.assertIn("not_a_real_field", msg)
        self.assertIn("g_max", msg)
