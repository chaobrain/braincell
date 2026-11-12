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

"""Tests for the Connection plasticity rule declaration (CELL-12a)."""

import unittest

import brainunit as u

import braincell
from braincell.mech import ConnectionRule, register_connection_rule
from braincell.mech._registry import _REGISTRY


class ConnectionRuleDeclarationTest(unittest.TestCase):
    def test_name_params_and_signals(self) -> None:
        source = braincell.EventSequence.from_times(([1.0] * u.ms,))
        rule = ConnectionRule("PairSTDP", signals={"post": source}, A_plus=0.01 * u.uS)
        self.assertEqual(rule.rule_type, "PairSTDP")
        self.assertEqual(sorted(rule.params), ["A_plus"])
        self.assertIs(rule.signals["post"], source)
        with self.assertRaises(TypeError):
            rule.signals["post"] = None
        self.assertIn("signals=", repr(rule))
        self.assertNotIn("signals=", repr(ConnectionRule("PairSTDP")))

    def test_is_immutable(self) -> None:
        rule = ConnectionRule("PairSTDP")
        with self.assertRaises(AttributeError):
            rule.rule_type = "Other"
        with self.assertRaises(AttributeError):
            del rule.params

    def test_equality_and_hash(self) -> None:
        a = braincell.EventSequence.from_times(([1.0] * u.ms,))
        b = braincell.EventSequence.from_times(([1.0] * u.ms,))
        self.assertEqual(ConnectionRule("PairSTDP", A_plus=1.0 * u.uS), ConnectionRule("PairSTDP", A_plus=1.0 * u.uS))
        self.assertEqual(
            hash(ConnectionRule("PairSTDP", signals={"post": a})), hash(ConnectionRule("PairSTDP", signals={"post": b}))
        )
        self.assertEqual(
            ConnectionRule("PairSTDP", signals={"post": a}), ConnectionRule("PairSTDP", signals={"post": a})
        )
        # Sources compare by identity.
        self.assertNotEqual(
            ConnectionRule("PairSTDP", signals={"post": a}), ConnectionRule("PairSTDP", signals={"post": b})
        )
        self.assertNotEqual(ConnectionRule("PairSTDP"), ConnectionRule("PairSTDP", signals={"post": a}))
        self.assertNotEqual(ConnectionRule("PairSTDP"), "PairSTDP")

    def test_a_class_resolves_to_its_registered_name(self) -> None:
        self.assertEqual(ConnectionRule(braincell.plasticity.PairSTDP).rule_type, "PairSTDP")

    def test_rejects_bad_names_and_signals(self) -> None:
        with self.assertRaises(ValueError):
            ConnectionRule("")
        with self.assertRaises(TypeError):
            ConnectionRule(3)
        with self.assertRaisesRegex(TypeError, "mapping"):
            ConnectionRule("PairSTDP", signals=[("post", None)])
        with self.assertRaisesRegex(TypeError, "non-empty strings"):
            ConnectionRule("PairSTDP", signals={"": None})


class RegisterConnectionRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._entries = dict(_REGISTRY._entries)
        self._aliases = dict(_REGISTRY._aliases)

    def tearDown(self) -> None:
        _REGISTRY._entries = self._entries
        _REGISTRY._aliases = self._aliases

    def test_registers_under_its_category(self) -> None:
        @register_connection_rule("__test_rule__", aliases=("__test_rule_alias__",))
        class _Rule(braincell.ConnectionRule):
            pass

        self.assertIs(_REGISTRY.get("connection_rule", "__test_rule__"), _Rule)
        self.assertIs(_REGISTRY.get("connection_rule", "__test_rule_alias__"), _Rule)
        self.assertNotIn("__test_rule__", _REGISTRY.names("synapse"))

    def test_pair_stdp_is_registered(self) -> None:
        self.assertIs(_REGISTRY.get("connection_rule", "PairSTDP"), braincell.plasticity.PairSTDP)


if __name__ == "__main__":
    unittest.main()
