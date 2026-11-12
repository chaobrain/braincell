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

"""Tests for the runtime base of Connection plasticity rules (CELL-12a)."""

import unittest

import brainunit as u
import numpy as np

import braincell
from braincell._base_rule import ConnectionRule, row_parameter
from braincell.mech import ConnectionRule as RuleDeclaration
from braincell.mech._registry import _REGISTRY


class _Unregistered(ConnectionRule):
    def update(self, state, *, pre, dt):
        return {"w": state["w"] + 0.001 * u.uS * pre}


class ConnectionRuleBaseTest(unittest.TestCase):
    def test_defaults(self) -> None:
        rule = ConnectionRule(3, u.uS)
        self.assertEqual(rule.size, 3)
        self.assertEqual(rule.weight_unit, u.uS)
        self.assertEqual(ConnectionRule.signals, ())
        weight = [0.1, 0.2, 0.3] * u.uS
        self.assertIs(rule.init_state(weight)["w"], weight)
        rule.check_weight(weight)  # accepts every weight

    def test_update_must_be_implemented(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "ConnectionRule must implement update"):
            ConnectionRule(1, u.uS).update({"w": [0.0] * u.uS}, pre=np.zeros(1), dt=0.1 * u.ms)

    def test_the_top_level_name_is_the_runtime_base(self) -> None:
        self.assertIs(braincell.ConnectionRule, ConnectionRule)

    def test_a_declaration_accepts_a_subclass_and_rejects_other_classes(self) -> None:
        declaration = RuleDeclaration(_Unregistered)
        self.assertIs(_REGISTRY.get("connection_rule", declaration.rule_type), _Unregistered)
        with self.assertRaisesRegex(TypeError, "ConnectionRule"):
            RuleDeclaration(int)

    def test_an_unregistered_subclass_runs_by_identity(self) -> None:
        cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
        cell = cell.place(braincell.mech.Synapse("ExpSyn", name="s"))
        source = braincell.EventSequence.from_times(([0.2, 0.5] * u.ms,))
        braincell.connect(
            "c",
            source=source,
            synapse=cell.synapses["s"],
            weight=0.01 * u.uS,
            plasticity=RuleDeclaration(_Unregistered),
        )
        cell.run(dt=0.1 * u.ms, duration=1.0 * u.ms)
        np.testing.assert_allclose(cell.connections["c"].rule_state("w").to_decimal(u.uS), [0.012], rtol=1e-6)


class RowParameterTest(unittest.TestCase):
    def test_converts_scalars_and_row_vectors(self) -> None:
        value = row_parameter(10.0 * u.nS, unit=u.uS, size=2, name="A", rule="R")
        self.assertEqual(value.unit, u.uS)
        np.testing.assert_allclose(value.mantissa, 0.01)
        value = row_parameter([1.0, 2.0] * u.ms, unit=u.ms, size=2, name="tau", rule="R")
        np.testing.assert_allclose(value.mantissa, [1.0, 2.0])

    def test_a_bare_number_is_dimensionless_only(self) -> None:
        with self.assertRaisesRegex(TypeError, "'A' requires a quantity"):
            row_parameter(0.01, unit=u.uS, size=1, name="A", rule="R")
        np.testing.assert_allclose(row_parameter(0.5, unit=u.UNITLESS, size=1, name="a", rule="R").mantissa, 0.5)

    def test_rejects_incompatible_units_shapes_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "compatible with"):
            row_parameter(1.0 * u.mV, unit=u.uS, size=1, name="A", rule="R")
        with self.assertRaisesRegex(ValueError, r"shape \(2,\)"):
            row_parameter([1.0, 2.0, 3.0] * u.uS, unit=u.uS, size=2, name="A", rule="R")
        with self.assertRaisesRegex(ValueError, "finite"):
            row_parameter(np.inf * u.uS, unit=u.uS, size=1, name="A", rule="R")


if __name__ == "__main__":
    unittest.main()
