# -*- coding: utf-8 -*-
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
import jax.numpy as jnp

from braincell._base import HHTypedNeuron
from braincell.channel._base import HH
from braincell.channel.hyperpolarization_activated import HCN_HM1992


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


class IhHM1992Test(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN_HM1992.root_type, HHTypedNeuron)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(HCN_HM1992, HH))

    def test_single_p_gate_is_declared(self) -> None:
        ch = HCN_HM1992(size=1)
        gates = ch._iter_gates()
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].name, "p")
        self.assertEqual(gates[0].power, 1)

    def test_default_parameters(self) -> None:
        ch = HCN_HM1992(size=2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                jnp.full((2,), 10.0),
            )
        )
        self.assertTrue(u.math.allclose(ch.E.to_decimal(u.mV), jnp.full((2,), 43.0)))
        self.assertTrue(
            u.math.allclose(
                ch.temp.to_decimal(u.kelvin),
                jnp.full((2,), u.celsius2kelvin(36.0).to_decimal(u.kelvin)),
            )
        )
        self.assertTrue(u.math.allclose(ch.q10, jnp.ones(2)))
        self.assertTrue(
            u.math.allclose(
                ch.temp_ref.to_decimal(u.kelvin),
                jnp.full((2,), u.celsius2kelvin(36.0).to_decimal(u.kelvin)),
            )
        )
        self.assertFalse(hasattr(ch, "phi"))

    def test_gate_phi_uses_standard_temp_interface(self) -> None:
        ch = HCN_HM1992(
            size=1,
            temp=u.celsius2kelvin(46.0),
            q10=3.0,
            temp_ref=u.celsius2kelvin(36.0),
        )
        gate = ch._iter_gates()[0]
        self.assertTrue(
            u.math.allclose(ch.gate_phi(gate), 3.0 * jnp.ones(1), atol=1e-6)
        )

    def test_reset_state_sets_p_to_steady_state(self) -> None:
        ch = HCN_HM1992(size=1)
        V = _V([-65.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_dynamics(self) -> None:
        ch = HCN_HM1992(
            size=1,
            temp=u.celsius2kelvin(46.0),
            q10=3.0,
            temp_ref=u.celsius2kelvin(36.0),
        )
        V = _V([-60.0])
        ch.init_state(V)
        ch.reset_state(V)
        ch.p.value = jnp.array([0.25])
        ch.compute_derivative(V)

        p_inf = ch.f_p_inf(V)
        tau = ch.f_p_tau(V)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (p_inf - 0.25) / tau / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_formula(self) -> None:
        ch = HCN_HM1992(size=1, g_max=5.0 * (u.mS / u.cm ** 2), E=-30.0 * u.mV)
        V = _V([-60.0])
        ch.init_state(V)
        ch.p.value = jnp.array([0.3])
        current = ch.current(V)
        expected = ch.g_max * ch.p.value * (ch.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit),
                expected.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_current_is_zero_when_p_zero(self) -> None:
        ch = HCN_HM1992(size=1)
        V = _V([-60.0])
        ch.init_state(V)
        ch.p.value = jnp.zeros(1)
        current = ch.current(V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mS / u.cm ** 2 * u.mV),
                jnp.zeros(1),
                atol=1e-9,
            )
        )

    def test_p_inf_monotone_in_hyperpolarized_regime(self) -> None:
        ch = HCN_HM1992(size=1)
        pinf_hyper = ch.f_p_inf(_V([-90.0]))
        pinf_mid = ch.f_p_inf(_V([-75.0]))
        pinf_dep = ch.f_p_inf(_V([-60.0]))
        self.assertGreater(float(pinf_hyper[0]), float(pinf_mid[0]))
        self.assertGreater(float(pinf_mid[0]), float(pinf_dep[0]))


if __name__ == "__main__":
    unittest.main()
