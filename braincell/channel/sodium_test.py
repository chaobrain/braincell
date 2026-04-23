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

from braincell._base import IonInfo
from braincell.channel._base import HH
from braincell.channel.sodium import Na_Ba2002, Na_HH1952, Na_TM1991
from braincell.ion import Sodium


def _na_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 140.0) * u.mM,
        E=jnp.full((size,), 50.0) * u.mV,
        valence=1,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


class _HHNaMixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_sodium(self) -> None:
        self.assertIs(self.CLS.root_type, Sodium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(self.CLS, HH))

    def test_gates_define_p3q_structure(self) -> None:
        ch = self._make(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("p", "q"))
        self.assertEqual(tuple(gate.power for gate in gates), (3, 1))

    def test_new_temperature_fields_replace_legacy_T_and_phi(self) -> None:
        ch = self._make(size=1)
        self.assertTrue(hasattr(ch, "temp"))
        self.assertTrue(hasattr(ch, "q10"))
        self.assertTrue(hasattr(ch, "temp_ref"))
        self.assertFalse(hasattr(ch, "T"))
        self.assertFalse(hasattr(ch, "phi"))

    def test_temperature_factor_is_derived_from_temp_q10_and_temp_ref(self) -> None:
        ch = self._make(
            size=1,
            temp=u.celsius2kelvin(46.0),
            q10=3.0,
            temp_ref=u.celsius2kelvin(36.0),
        )
        for gate in ch._iter_gates():
            self.assertTrue(
                u.math.allclose(ch.gate_phi(gate), 3.0 * jnp.ones(1), atol=1e-6)
            )

    def test_default_gmax_has_current_density_units(self) -> None:
        ch = self._make(size=1)
        _ = ch.g_max.to_decimal(u.mS / u.cm ** 2)

    def test_init_state_creates_gates_shaped_to_size(self) -> None:
        ch = self._make(size=4)
        V = _V([-60.0, -55.0, -50.0, -45.0])
        na = _na_info(4)

        ch.init_state(V, na)
        self.assertEqual(ch.p.value.shape, (4,))
        self.assertEqual(ch.q.value.shape, (4,))
        self.assertTrue(u.math.allclose(ch.p.value, jnp.zeros(4)))
        self.assertTrue(u.math.allclose(ch.q.value, jnp.zeros(4)))

    def test_reset_state_sets_gates_to_steady_state(self) -> None:
        ch = self._make(size=1)
        V = _V([-65.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)

        alpha_p = ch.f_p_alpha(V)
        beta_p = ch.f_p_beta(V)
        alpha_q = ch.f_q_alpha(V)
        beta_q = ch.f_q_beta(V)
        self.assertTrue(
            u.math.allclose(ch.p.value, alpha_p / (alpha_p + beta_p), atol=1e-6)
        )
        self.assertTrue(
            u.math.allclose(ch.q.value, alpha_q / (alpha_q + beta_q), atol=1e-6)
        )
        self.assertTrue(bool((ch.p.value >= 0).all() and (ch.p.value <= 1).all()))
        self.assertTrue(bool((ch.q.value >= 0).all() and (ch.q.value <= 1).all()))

    def test_compute_derivative_matches_hh_equation(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        ch.p.value = jnp.array([0.1])
        ch.q.value = jnp.array([0.9])
        ch.compute_derivative(V, na)

        alpha_p = ch.f_p_alpha(V)
        beta_p = ch.f_p_beta(V)
        alpha_q = ch.f_q_alpha(V)
        beta_q = ch.f_q_beta(V)
        gates = {gate.name: gate for gate in ch._iter_gates()}

        expected_dp = (
            ch.gate_phi(gates["p"]) * (alpha_p * (1.0 - ch.p.value) - beta_p * ch.p.value) / u.ms
        )
        expected_dq = (
            ch.gate_phi(gates["q"]) * (alpha_q * (1.0 - ch.q.value) - beta_q * ch.q.value) / u.ms
        )

        self.assertTrue(
            u.math.allclose(ch.p.derivative, expected_dp, atol=1e-6 * u.Hz)
        )
        self.assertTrue(
            u.math.allclose(ch.q.derivative, expected_dq, atol=1e-6 * u.Hz)
        )

    def test_current_matches_p3q_formula(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)

        current = ch.current(V, na)
        expected = ch.g_max * ch.p.value ** 3 * ch.q.value * (na.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit), expected.to_decimal(unit), atol=1e-6
            )
        )

    def test_current_is_zero_when_gates_closed(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.p.value = jnp.zeros(1)
        ch.q.value = jnp.zeros(1)

        current = ch.current(V, na)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mS / u.cm ** 2 * u.mV),
                jnp.zeros(1),
                atol=1e-9,
            )
        )


class Na_Ba2002Test(_HHNaMixin, unittest.TestCase):
    CLS = Na_Ba2002

    def test_V_sh_shifts_rate_functions(self) -> None:
        ch_a = Na_Ba2002(size=1, V_sh=-50.0 * u.mV)
        ch_b = Na_Ba2002(size=1, V_sh=-60.0 * u.mV)

        V = _V([-55.0])
        V_shifted = _V([-45.0])
        self.assertTrue(
            u.math.allclose(ch_a.f_p_alpha(V_shifted), ch_b.f_p_alpha(V), atol=1e-6)
        )


class Na_TM1991Test(_HHNaMixin, unittest.TestCase):
    CLS = Na_TM1991


class Na_HH1952Test(_HHNaMixin, unittest.TestCase):
    CLS = Na_HH1952


if __name__ == "__main__":
    unittest.main()
