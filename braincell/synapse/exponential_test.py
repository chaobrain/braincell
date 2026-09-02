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
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.mech import ScalarEventInput


class SynapseSchemaTest(unittest.TestCase):
    def test_expsyn_schema_is_the_only_parameter_source(self) -> None:
        model = braincell.synapse.ExpSyn
        self.assertEqual(tuple(model.parameters), ("tau", "e"))
        self.assertEqual(tuple(model.states), ("g",))
        self.assertEqual(model.event_input, ScalarEventInput(u.uS, aggregation="sum"))
        self.assertNotIn("weight", model.parameters)
        self.assertFalse(hasattr(model, "event_weight_unit"))
        self.assertFalse(hasattr(model, "current_sign"))
        self.assertFalse(hasattr(model, "current_units"))

    def test_physical_validation_is_not_a_learning_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            braincell.synapse.ExpSyn(1, tau=0.0 * u.ms)
        field = braincell.synapse.ExpSyn.parameters["tau"]
        self.assertFalse(hasattr(field, "trainable"))
        self.assertFalse(hasattr(field, "transform"))

    def test_retired_receptor_names_are_gone(self) -> None:
        for name in ("AMPA", "GABAa", "NMDA"):
            self.assertFalse(hasattr(braincell.synapse, name))


class ExpSynTest(unittest.TestCase):
    def test_event_and_current_are_vectorized(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=[1.0, 2.0] * u.ms)
        synapse.init_state()
        synapse.apply_events([0.01, 0.02] * u.uS)

        np.testing.assert_allclose(synapse.g.value.to_decimal(u.uS), [0.01, 0.02])
        np.testing.assert_allclose(
            synapse.current([-65.0, -50.0] * u.mV).to_decimal(u.nA),
            [0.65, 1.0],
        )

    def test_conductance_decays_with_its_own_tau(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=[1.0, 4.0] * u.ms)
        synapse.init_state()
        synapse.apply_events([0.02, 0.02] * u.uS)
        synapse.compute_derivative()

        np.testing.assert_allclose(
            synapse.g.derivative.to_decimal(u.uS / u.ms),
            [-0.02 / 1.0, -0.02 / 4.0],
        )

    def test_reset_state_returns_g_to_the_declared_initial(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=1.0 * u.ms)
        synapse.init_state()
        synapse.apply_events([0.01, 0.02] * u.uS)
        synapse.reset_state()

        np.testing.assert_allclose(synapse.g.value.to_decimal(u.uS), [0.0, 0.0])

    def test_a_wrong_unit_payload_is_rejected(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=1.0 * u.ms)
        synapse.init_state()
        with self.assertRaisesRegex(ValueError, "units incompatible"):
            synapse.apply_events([0.01, 0.02] * u.mV)
        with self.assertRaisesRegex(TypeError, "must be a quantity"):
            synapse.apply_events([0.01, 0.02])


class Exp2SynTest(unittest.TestCase):
    TAU1 = 0.5
    TAU2 = 5.0

    def _synapse(self, size: int = 1):
        synapse = braincell.synapse.Exp2Syn(
            size,
            tau1=self.TAU1 * u.ms,
            tau2=self.TAU2 * u.ms,
        )
        synapse.init_state()
        return synapse

    def test_canonical_time_order_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            braincell.synapse.Exp2Syn(1, tau1=2.0 * u.ms, tau2=1.0 * u.ms)

    def test_time_order_compares_physical_durations_not_magnitudes(self) -> None:
        # 2000 us > 1 ms, even though 2000 > 1 is the only numeric comparison.
        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            braincell.synapse.Exp2Syn(1, tau1=2000.0 * u.us, tau2=1.0 * u.ms)
        # ...and the same pair in the right order is accepted.
        braincell.synapse.Exp2Syn(1, tau1=200.0 * u.us, tau2=1.0 * u.ms)

    def test_construction_survives_an_open_jax_trace(self) -> None:
        # Regression: ``validate_parameter_values`` used ``u.math.any``, and
        # under an open trace every jax op returns a tracer even for concrete
        # inputs, so ``bool()`` on it raised ``TracerBoolConversionError``.
        def build():
            braincell.synapse.Exp2Syn(4, tau1=self.TAU1 * u.ms, tau2=self.TAU2 * u.ms)
            return jnp.zeros(())

        jaxpr = jax.make_jaxpr(build)()
        self.assertIsNotNone(jaxpr)

    def test_conductance_is_the_difference_of_the_two_states(self) -> None:
        synapse = self._synapse(2)
        synapse.A.value = np.asarray([0.3, 0.1]) * u.uS
        synapse.B.value = np.asarray([0.9, 0.4]) * u.uS

        np.testing.assert_allclose(synapse.g.to_decimal(u.uS), [0.6, 0.3])

    def test_one_event_moves_both_states_by_the_same_amount(self) -> None:
        synapse = self._synapse(2)
        synapse.apply_events([0.01, 0.02] * u.uS)

        a = synapse.A.value.to_decimal(u.uS)
        b = synapse.B.value.to_decimal(u.uS)
        np.testing.assert_allclose(a, b)
        # g is zero at the instant of the event and rises afterwards.
        np.testing.assert_allclose(synapse.g.to_decimal(u.uS), [0.0, 0.0], atol=1e-12)

    def test_each_state_decays_with_its_own_tau_and_they_are_not_swapped(self) -> None:
        synapse = self._synapse(1)
        synapse.A.value = np.asarray([0.4]) * u.uS
        synapse.B.value = np.asarray([0.4]) * u.uS
        synapse.compute_derivative()

        np.testing.assert_allclose(synapse.A.derivative.to_decimal(u.uS / u.ms), [-0.4 / self.TAU1])
        np.testing.assert_allclose(synapse.B.derivative.to_decimal(u.uS / u.ms), [-0.4 / self.TAU2])

    def test_factor_normalizes_the_peak_conductance_to_the_payload(self) -> None:
        # The defining property of exp2syn's normalisation constant: after a
        # unit event, max_t g(t) == payload exactly. Checked twice, against
        # the closed-form peak time and against a dense sampling of g(t).
        payload = 0.02
        synapse = self._synapse(1)
        synapse.apply_events([payload] * u.uS)
        delta = float(synapse.A.value.to_decimal(u.uS)[0])

        tau1, tau2 = self.TAU1, self.TAU2
        tp = tau1 * tau2 / (tau2 - tau1) * np.log(tau2 / tau1)

        def g_of(t):
            return delta * (np.exp(-t / tau2) - np.exp(-t / tau1))

        # rtol is float32-sized: ``delta`` comes back from the default-precision
        # runtime, so the normalisation cannot be tighter than the payload's
        # own representation.
        np.testing.assert_allclose(g_of(tp), payload, rtol=1e-6)

        t = np.linspace(0.0, 10.0 * tau2, 200_001)
        np.testing.assert_allclose(float(g_of(t).max()), payload, rtol=1e-6)

    def test_current_is_inward_positive_at_the_declared_reversal(self) -> None:
        synapse = self._synapse(2)
        synapse.A.value = np.asarray([0.0, 0.0]) * u.uS
        synapse.B.value = np.asarray([0.01, 0.02]) * u.uS

        np.testing.assert_allclose(
            synapse.current([-65.0, -50.0] * u.mV).to_decimal(u.nA),
            [0.65, 1.0],
        )

    def test_reset_state_returns_both_states_to_their_declared_initials(self) -> None:
        synapse = self._synapse(2)
        synapse.apply_events([0.01, 0.02] * u.uS)
        synapse.reset_state()

        np.testing.assert_allclose(synapse.A.value.to_decimal(u.uS), [0.0, 0.0])
        np.testing.assert_allclose(synapse.B.value.to_decimal(u.uS), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
