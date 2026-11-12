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

Every aggregation a contract accepts has a runtime behaviour case in
``AggregationBehaviourTest`` (CELL-40): before braincell 0.3 the contracts
also accepted ``"any"`` and ``"ordered"``, which the runtime never honoured.
"""

import unittest

import brainstate
import brainunit as u
import numpy as np

import braincell
from braincell.filter import at
from braincell.mech import EventInput, NoEventInput, ScalarEventInput, TriggerEventInput, get_registry, register_synapse
from braincell.mech._event_contract import SCALAR_AGGREGATIONS, TRIGGER_AGGREGATIONS
from braincell.network import _testing
from braincell.synapse import ExpSyn


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
        self.assertEqual(TRIGGER_AGGREGATIONS, frozenset({"count"}))
        for aggregation in TRIGGER_AGGREGATIONS:
            self.assertEqual(TriggerEventInput(aggregation=aggregation).aggregation, aggregation)

    def test_unsupported_aggregation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported trigger-event aggregation 'sum'. Supported: 'count'."):
            TriggerEventInput(aggregation="sum")

    def test_unhonoured_aggregations_are_rejected_with_the_replacement(self) -> None:
        # CELL-40: accepted before braincell 0.3, but the runtime always counted.
        for aggregation in ("any", "ordered"):
            with self.subTest(aggregation=aggregation):
                with self.assertRaises(ValueError) as caught:
                    TriggerEventInput(aggregation=aggregation)
                message = str(caught.exception)
                self.assertTrue(
                    message.startswith(f"TriggerEventInput(aggregation={aggregation!r}) was removed in braincell 0.3"),
                    message,
                )
                self.assertIn("Use aggregation='count'", message)

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
        self.assertEqual(SCALAR_AGGREGATIONS, frozenset({"sum"}))
        for aggregation in SCALAR_AGGREGATIONS:
            self.assertEqual(ScalarEventInput(u.uS, aggregation=aggregation).aggregation, aggregation)

    def test_trigger_aggregations_are_rejected(self) -> None:
        # "count" and "any" are trigger aggregations (only "count" still is),
        # meaningless for a payload that carries a physical value.
        for aggregation in ("count", "any"):
            with self.assertRaisesRegex(ValueError, "Unsupported scalar-event aggregation"):
                ScalarEventInput(u.uS, aggregation=aggregation)

    def test_unhonoured_ordered_is_rejected_with_the_replacement(self) -> None:
        # CELL-40: accepted before braincell 0.3, but the runtime always summed.
        with self.assertRaises(ValueError) as caught:
            ScalarEventInput(u.uS, aggregation="ordered")
        message = str(caught.exception)
        self.assertTrue(
            message.startswith("ScalarEventInput(aggregation='ordered') was removed in braincell 0.3"), message
        )
        self.assertIn("Use aggregation='sum'", message)

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


_DT = 0.1 * u.ms
_EVENT_TIMES = [0.1] * u.ms


def _g_after_events(synapse_type: str, weights) -> np.ndarray:
    """``g`` of one synapse that ``len(weights)`` sources hit on the same boundary.

    Each source is its own connection row onto the same synapse; ``weights``
    is ``None`` for a payload-free (trigger) synapse.
    """
    cell = braincell.Cell(_testing.make_soma_tree(), cv_policy=braincell.CVPerBranch(), V_init=-65.0 * u.mV)
    cell.place(at("soma", 0.5), braincell.mech.Synapse(synapse_type, name="p", tau=2.0 * u.ms))
    cell.soma.record("g", braincell.observe.synapse(name="p").state("g"))
    rows = 2 if weights is None else len(weights)
    net = braincell.Network("aggregation")
    source = net.add_population("source", braincell.EventSequence.from_times((_EVENT_TIMES,) * rows))
    post = net.add_population("post", cell)
    net.connect("c", source=source, synapse=post.synapses["p"][[0] * rows], weight=weights)
    result = net.run(dt=_DT, duration=4 * _DT)
    return np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))[:, 0]


class _TriggerCount(ExpSyn):
    """The smallest trigger synapse: ``g`` holds the counts ``apply_events`` received."""

    event_input = TriggerEventInput(aggregation="count")

    def apply_events(self, payload, V_post=None):
        self.g.value = self.g.value + u.math.asarray(payload, dtype=float) * u.uS

    def compute_derivative(self, V_post=None):
        self.g.derivative = u.math.zeros_like(self.g.value) / u.ms


class AggregationBehaviourTest(unittest.TestCase):
    """Each aggregation a contract accepts is honoured at runtime (CELL-40).

    Before braincell 0.3 ``ScalarEventInput`` also accepted ``"ordered"`` and
    ``TriggerEventInput`` ``"any"`` and ``"ordered"``; the runtime summed them
    all, so a declared ``"ordered"`` silently meant ``"sum"``. A value added to
    either allowed set must come with a case here, or the coverage test fails.
    """

    def setUp(self) -> None:
        context = brainstate.environ.context(precision=64)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def case_scalar_sum(self) -> None:
        """Two 0.005 uS rows on one boundary add up to one 0.01 uS row."""
        two = _g_after_events("ExpSyn", [0.005, 0.005] * u.uS)
        one = _g_after_events("ExpSyn", [0.01] * u.uS)
        self.assertEqual(two[0], 0.0)
        self.assertGreater(two[-1], 0.0)
        np.testing.assert_allclose(two, one, rtol=1e-12, atol=0.0)

    def case_trigger_count(self) -> None:
        """A trigger synapse hit twice on one boundary gets a count of 2."""
        name = "_CELL40TriggerCount"
        register_synapse(name)(_TriggerCount)
        self.addCleanup(get_registry().unregister, "synapse", name)
        g = _g_after_events(name, None)
        np.testing.assert_array_equal(g[0], 0.0)
        np.testing.assert_array_equal(g[-1], 2.0)

    #: ``(contract, aggregation) -> behaviour case``; one per allowed value.
    CASES = {
        ("ScalarEventInput", "sum"): case_scalar_sum,
        ("TriggerEventInput", "count"): case_trigger_count,
    }

    def test_every_allowed_aggregation_has_a_behaviour_case(self) -> None:
        allowed = {("ScalarEventInput", value) for value in SCALAR_AGGREGATIONS}
        allowed |= {("TriggerEventInput", value) for value in TRIGGER_AGGREGATIONS}
        self.assertEqual(sorted(allowed - set(self.CASES)), [], "allowed aggregations without a behaviour case")
        self.assertEqual(sorted(set(self.CASES) - allowed), [], "behaviour cases for values no longer allowed")

    def test_scalar_sum(self) -> None:
        self.CASES[("ScalarEventInput", "sum")](self)

    def test_trigger_count(self) -> None:
        self.CASES[("TriggerEventInput", "count")](self)


if __name__ == "__main__":
    unittest.main()
