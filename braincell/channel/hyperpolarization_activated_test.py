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
from braincell.channel.hyperpolarization_activated import (
    Ih1_Ma2020,
    Ih2_Ma2020,
    Ih_HM1992,
)


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


class IhHM1992Test(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Ih_HM1992.root_type, HHTypedNeuron)

    def test_default_parameters(self) -> None:
        ch = Ih_HM1992(size=2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2), jnp.full((2,), 10.0)
            )
        )
        self.assertTrue(u.math.allclose(ch.E.to_decimal(u.mV), jnp.full((2,), 43.0)))
        self.assertTrue(u.math.allclose(ch.phi, jnp.ones(2)))

    def test_reset_state_sets_p_to_steady_state(self) -> None:
        ch = Ih_HM1992(size=1)
        V = _V([-65.0])
        ch.init_state(V)
        ch.reset_state(V)
        expected = 1.0 / (1.0 + jnp.exp((-65.0 + 75.0) / 5.5))
        self.assertTrue(u.math.allclose(ch.p.value, jnp.array([expected]), atol=1e-6))

    def test_compute_derivative_matches_dynamics(self) -> None:
        ch = Ih_HM1992(size=1)
        V = _V([-60.0])
        ch.init_state(V)
        ch.reset_state(V)
        ch.p.value = jnp.array([0.25])
        ch.compute_derivative(V)

        p_inf = ch.f_p_inf(V)
        tau = ch.f_p_tau(V)
        expected = ch.phi * (p_inf - 0.25) / tau / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_formula(self) -> None:
        ch = Ih_HM1992(size=1, g_max=5.0 * (u.mS / u.cm ** 2), E=-30.0 * u.mV)
        V = _V([-60.0])
        ch.init_state(V)
        ch.p.value = jnp.array([0.3])
        i = ch.current(V)
        expected = ch.g_max * ch.p.value * (ch.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(i.to_decimal(unit), expected.to_decimal(unit), atol=1e-6)
        )

    def test_current_is_zero_when_p_zero(self) -> None:
        ch = Ih_HM1992(size=1)
        V = _V([-60.0])
        ch.init_state(V)
        ch.p.value = jnp.zeros(1)
        i = ch.current(V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(u.mS / u.cm ** 2 * u.mV),
                jnp.zeros(1),
                atol=1e-9,
            )
        )

    def test_p_inf_monotone_in_hyperpolarized_regime(self) -> None:
        # Ih is hyperpolarization-activated: p_inf must decrease monotonically
        # as V becomes more depolarized.
        ch = Ih_HM1992(size=1)
        pinf_hyper = ch.f_p_inf(_V([-90.0]))
        pinf_mid = ch.f_p_inf(_V([-75.0]))
        pinf_dep = ch.f_p_inf(_V([-60.0]))
        self.assertGreater(float(pinf_hyper[0]), float(pinf_mid[0]))
        self.assertGreater(float(pinf_mid[0]), float(pinf_dep[0]))


class _MaMixin:
    CLS = None

    def test_init_state_creates_p_and_q(self) -> None:
        ch = self.CLS(size=2)
        V = _V([-60.0, -80.0])
        ch.init_state(V)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_matches_f_inf(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_current_uses_sum_of_p_and_q(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        ch.p.value = jnp.array([0.3])
        ch.q.value = jnp.array([0.4])
        i = ch.current(V)
        expected = ch.phi_g * ch.g_max * (ch.p.value + ch.q.value) * (ch.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(i.to_decimal(unit), expected.to_decimal(unit), atol=1e-6)
        )

    def test_compute_derivative_matches_first_order_ode(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        ch.p.value = jnp.array([0.1])
        ch.q.value = jnp.array([0.9])
        ch.compute_derivative(V)

        exp_dp = ch.phi_channel * (ch.f_p_inf(V) - 0.1) / ch.f_p_tau(V) / u.ms
        exp_dq = ch.phi_channel * (ch.f_q_inf(V) - 0.9) / ch.f_q_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_dp, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_dq, atol=1e-6 * u.Hz))


class Ih1_Ma2020Test(_MaMixin, unittest.TestCase):
    CLS = Ih1_Ma2020

    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Ih1_Ma2020.root_type, HHTypedNeuron)


class Ih2_Ma2020Test(_MaMixin, unittest.TestCase):
    CLS = Ih2_Ma2020

    def test_r_is_clamped_to_corridor(self) -> None:
        # Ih2_Ma2020.r returns 0 above -64.70 mV and 1 below -108.70 mV;
        # ``V`` here is the plain decimal (mV), not a quantity.
        ch = Ih2_Ma2020(size=1)
        self.assertEqual(float(ch.r(jnp.array(-50.0), ch.rA, ch.rB)), 0.0)
        self.assertEqual(float(ch.r(jnp.array(-120.0), ch.rA, ch.rB)), 1.0)
        # Inside the corridor, r is a linear function of V.
        inside = ch.r(jnp.array(-80.0), ch.rA, ch.rB)
        expected = ch.rA * -80.0 + ch.rB
        self.assertAlmostEqual(float(inside), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
