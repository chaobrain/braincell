# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.

import unittest

import brainunit as u
import numpy as np

import braincell
from braincell.mech import ScalarEventInput


class SynapseSchemaTest(unittest.TestCase):
    def test_expsyn_schema_is_the_only_parameter_source(self) -> None:
        model = braincell.synapse.ExpSyn
        self.assertEqual(tuple(model.parameters), ("tau", "e"))
        self.assertEqual(tuple(model.states), ("g",))
        self.assertEqual(model.derived, {})
        self.assertEqual(model.event_input, ScalarEventInput(u.uS, aggregation="sum"))
        self.assertNotIn("weight", model.parameters)
        self.assertFalse(hasattr(model, "event_weight_unit"))
        self.assertFalse(hasattr(model, "current_sign"))
        self.assertFalse(hasattr(model, "current_units"))

    def test_expsyn_event_and_current_are_vectorized(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=[1.0, 2.0] * u.ms)
        synapse.init_state()
        synapse.apply_events([0.01, 0.02] * u.uS)

        np.testing.assert_allclose(synapse.g.value.to_decimal(u.uS), [0.01, 0.02])
        np.testing.assert_allclose(
            synapse.current([-65.0, -50.0] * u.mV).to_decimal(u.nA),
            [0.65, 1.0],
        )

    def test_physical_validation_is_not_a_learning_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            braincell.synapse.ExpSyn(1, tau=0.0 * u.ms)
        field = braincell.synapse.ExpSyn.parameters["tau"]
        self.assertFalse(hasattr(field, "trainable"))
        self.assertFalse(hasattr(field, "transform"))

    def test_exp2syn_requires_canonical_time_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            braincell.synapse.Exp2Syn(1, tau1=2.0 * u.ms, tau2=1.0 * u.ms)

    def test_unavailable_receptor_classes_raise_targeted_error(self) -> None:
        for cls in (braincell.synapse.AMPA, braincell.synapse.GABAa, braincell.synapse.NMDA):
            with self.assertRaisesRegex(NotImplementedError, "temporarily unavailable"):
                cls(1)


if __name__ == "__main__":
    unittest.main()
