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

"""Event-input contracts declared by target mechanisms.

``braincell.network.connection`` dispatches on the contract type when it
validates a ``connect(...)`` payload, and ``braincell._compute.state``
dispatches on it again when it allocates the runtime event buffer. The
aggregation whitelists below are therefore load-bearing, and until these
contracts moved into :mod:`braincell.mech` they had no direct tests.
"""

import unittest

import brainstate
import brainunit as u

from braincell.mech import EventInput, NoEventInput, ScalarEventInput, TriggerEventInput


class NoEventInputTest(unittest.TestCase):
    def test_declares_the_inert_contract(self) -> None:
        contract = NoEventInput()
        self.assertIsInstance(contract, EventInput)
        self.assertEqual(contract.payload_kind, "none")
        self.assertEqual(contract.aggregation, "none")

    def test_is_frozen_and_hashable(self) -> None:
        self.assertEqual(NoEventInput(), NoEventInput())
        self.assertEqual(len({NoEventInput(), NoEventInput()}), 1)
        with self.assertRaises(Exception):
            NoEventInput().payload_kind = "scalar"


class TriggerEventInputTest(unittest.TestCase):
    def test_default_aggregation_is_count(self) -> None:
        contract = TriggerEventInput()
        self.assertEqual(contract.payload_kind, "trigger")
        self.assertEqual(contract.aggregation, "count")

    def test_every_supported_aggregation_is_accepted(self) -> None:
        for aggregation in ("count", "any", "ordered"):
            self.assertEqual(TriggerEventInput(aggregation=aggregation).aggregation, aggregation)

    def test_unsupported_aggregation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported trigger-event aggregation"):
            TriggerEventInput(aggregation="sum")

    def test_aggregation_is_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            TriggerEventInput("count")


class ScalarEventInputTest(unittest.TestCase):
    def test_default_aggregation_is_sum(self) -> None:
        contract = ScalarEventInput(u.uS)
        self.assertEqual(contract.payload_kind, "scalar")
        self.assertEqual(contract.aggregation, "sum")
        self.assertIs(contract.unit, u.uS)

    def test_every_supported_aggregation_is_accepted(self) -> None:
        for aggregation in ("sum", "ordered"):
            self.assertEqual(ScalarEventInput(u.uS, aggregation=aggregation).aggregation, aggregation)

    def test_trigger_aggregations_are_rejected(self) -> None:
        # "count" and "any" are valid for TriggerEventInput but meaningless
        # for a payload that carries a physical value.
        for aggregation in ("count", "any"):
            with self.assertRaisesRegex(ValueError, "Unsupported scalar-event aggregation"):
                ScalarEventInput(u.uS, aggregation=aggregation)

    def test_validate_payload_returns_a_compatible_quantity(self) -> None:
        contract = ScalarEventInput(u.uS)
        payload = 0.2 * u.uS
        self.assertIs(contract.validate_payload(payload), payload)
        # A different but compatible unit is accepted and returned unconverted.
        millisiemens = 0.001 * u.mS
        self.assertIs(contract.validate_payload(millisiemens), millisiemens)

    def test_validate_payload_rejects_a_bare_number(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a quantity compatible with"):
            ScalarEventInput(u.uS).validate_payload(0.2)

    def test_validate_payload_rejects_an_incompatible_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "units incompatible with"):
            ScalarEventInput(u.uS).validate_payload(0.2 * u.mV)

    def test_validate_payload_does_not_force_host_materialization(self) -> None:
        """A traced payload must survive validation for use inside jit."""
        contract = ScalarEventInput(u.uS)
        seen = {}

        @brainstate.transform.jit
        def run(weight):
            seen["validated"] = contract.validate_payload(weight)
            return weight * 2.0

        out = run(0.2 * u.uS)
        self.assertIn("validated", seen)
        self.assertEqual(u.get_unit(out), u.uS)


if __name__ == "__main__":
    unittest.main()
