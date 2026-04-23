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
from braincell.channel._base import HH, ghk_flux
from braincell.channel.calcium import (
    CaHT_HM1992,
    CaHT_Re1993,
    CaL_IS2008,
    CaN_IS2008,
    CaT_HM1992,
    CaT_HP1992,
    Cav1p2_MA2020,
    Cav1p3_MA2020,
    Cav3p1_MA2020,
)
from braincell.ion import Calcium


def _ca_info(size: int = 1, C: float = 1e-4, E_mV: float = 120.0) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), C) * u.mM,
        Co=jnp.full((size,), 2.0) * u.mM,
        E=jnp.full((size,), E_mV) * u.mV,
        valence=2,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


class _P2QHHMixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_calcium(self) -> None:
        self.assertIs(self.CLS.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(self.CLS, HH))

    def test_gates_define_p2_q_structure(self) -> None:
        ch = self._make(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("p", "q"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_new_temperature_fields_replace_legacy_phi_fields(self) -> None:
        ch = self._make(size=1)
        self.assertTrue(hasattr(ch, "temp"))
        self.assertTrue(hasattr(ch, "q10_p"))
        self.assertTrue(hasattr(ch, "temp_ref_p"))
        self.assertTrue(hasattr(ch, "q10_q"))
        self.assertTrue(hasattr(ch, "temp_ref_q"))
        self.assertFalse(hasattr(ch, "phi_p"))
        self.assertFalse(hasattr(ch, "phi_q"))
        self.assertFalse(hasattr(ch, "T"))

    def test_gate_phi_is_gate_specific(self) -> None:
        ch = self._make(
            size=1,
            temp=u.celsius2kelvin(34.0),
            q10_p=3.0,
            temp_ref_p=u.celsius2kelvin(24.0),
            q10_q=2.0,
            temp_ref_q=u.celsius2kelvin(24.0),
        )
        gates = {gate.name: gate for gate in ch._iter_gates()}
        self.assertTrue(
            u.math.allclose(ch.gate_phi(gates["p"]), 3.0 * jnp.ones(1), atol=1e-6)
        )
        self.assertTrue(
            u.math.allclose(ch.gate_phi(gates["q"]), 2.0 * jnp.ones(1), atol=1e-6)
        )

    def test_init_state_creates_p_and_q(self) -> None:
        ch = self._make(size=2)
        V = _V([-60.0, -70.0])
        ca = _ca_info(2)
        ch.init_state(V, ca)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_matches_gate_form(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}

        if ch._gate_form(gates["p"]) == "inf_tau":
            expected_p = ch.f_p_inf(V)
        else:
            alpha_p = ch.f_p_alpha(V)
            beta_p = ch.f_p_beta(V)
            expected_p = alpha_p / (alpha_p + beta_p)

        if ch._gate_form(gates["q"]) == "inf_tau":
            expected_q = ch.f_q_inf(V)
        else:
            alpha_q = ch.f_q_alpha(V)
            beta_q = ch.f_q_beta(V)
            expected_q = alpha_q / (alpha_q + beta_q)

        self.assertTrue(u.math.allclose(ch.p.value, expected_p, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, expected_q, atol=1e-6))

    def test_compute_derivative_matches_hh_gate_dynamics(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.6])
        ch.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}

        if ch._gate_form(gates["p"]) == "inf_tau":
            exp_dp = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V) - ch.p.value) / ch.f_p_tau(V) / u.ms
        else:
            alpha_p = ch.f_p_alpha(V)
            beta_p = ch.f_p_beta(V)
            exp_dp = ch.gate_phi(gates["p"]) * (alpha_p * (1.0 - ch.p.value) - beta_p * ch.p.value) / u.ms

        if ch._gate_form(gates["q"]) == "inf_tau":
            exp_dq = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V) - ch.q.value) / ch.f_q_tau(V) / u.ms
        else:
            alpha_q = ch.f_q_alpha(V)
            beta_q = ch.f_q_beta(V)
            exp_dq = ch.gate_phi(gates["q"]) * (alpha_q * (1.0 - ch.q.value) - beta_q * ch.q.value) / u.ms

        self.assertTrue(u.math.allclose(ch.p.derivative, exp_dp, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_dq, atol=1e-6 * u.Hz))

    def test_current_matches_p2_q_formula(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        current = ch.current(V, ca)
        expected = ch.g_max * ch.p.value ** 2 * ch.q.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_current_is_zero_when_gates_closed(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.zeros(1)
        ch.q.value = jnp.zeros(1)
        current = ch.current(V, ca)
        self.assertTrue(
            u.math.allclose(current.to_decimal(_DENSITY_UNIT), jnp.zeros(1), atol=1e-9)
        )


class CaT_HM1992Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaT_HM1992


class CaT_HP1992Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaT_HP1992


class CaHT_HM1992Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaHT_HM1992


class CaHT_Re1993Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaHT_Re1993


class CaL_IS2008Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaL_IS2008


class CaN_IS2008Test(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(CaN_IS2008.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(CaN_IS2008, HH))

    def test_single_p_gate_is_declared(self) -> None:
        ch = CaN_IS2008(size=1)
        gates = ch._iter_gates()
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].name, "p")
        self.assertEqual(gates[0].power, 1)

    def test_new_temperature_fields_replace_legacy_phi(self) -> None:
        ch = CaN_IS2008(size=1)
        self.assertTrue(hasattr(ch, "temp"))
        self.assertTrue(hasattr(ch, "q10"))
        self.assertTrue(hasattr(ch, "temp_ref"))
        self.assertFalse(hasattr(ch, "phi"))
        self.assertFalse(hasattr(ch, "T"))

    def test_reset_state_sets_p_to_logistic(self) -> None:
        ch = CaN_IS2008(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = CaN_IS2008(
            size=1,
            temp=u.celsius2kelvin(46.0),
            q10=3.0,
            temp_ref=u.celsius2kelvin(36.0),
        )
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.25])
        ch.compute_derivative(V, ca)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (ch.f_p_inf(V) - ch.p.value) / ch.f_p_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_depends_on_calcium_concentration(self) -> None:
        ch = CaN_IS2008(size=1)
        V = _V([-60.0])
        low = _ca_info(C=1e-4)
        high = _ca_info(C=10.0)
        ch.init_state(V, low)
        ch.reset_state(V, low)
        i_low = ch.current(V, low)
        i_high = ch.current(V, high)
        high_mag = float(u.math.abs(i_high).to_decimal(_DENSITY_UNIT)[0])
        low_mag = float(u.math.abs(i_low).to_decimal(_DENSITY_UNIT)[0])
        self.assertGreater(high_mag, low_mag)

    def test_current_matches_formula(self) -> None:
        ch = CaN_IS2008(size=1, E=10.0 * u.mV, g_max=1.0 * (u.mS / u.cm ** 2))
        V = _V([-60.0])
        ca = _ca_info(C=0.001)
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        current = ch.current(V, ca)
        modulation = ca.Ci / (ca.Ci + 0.2 * u.mM)
        expected = ch.g_max * modulation * ch.p.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-9,
            )
        )


class _MHNHHMixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_calcium(self) -> None:
        self.assertIs(self.CLS.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(self.CLS, HH))

    def test_gates_define_m_h_n_structure(self) -> None:
        ch = self._make(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("m", "h", "n"))
        self.assertEqual(tuple(gate.power for gate in gates), (1, 1, 1))

    def test_new_temperature_fields_replace_legacy_phi(self) -> None:
        ch = self._make(size=1)
        self.assertTrue(hasattr(ch, "temp"))
        self.assertTrue(hasattr(ch, "q10"))
        self.assertTrue(hasattr(ch, "temp_ref"))
        self.assertFalse(hasattr(ch, "phi"))
        self.assertFalse(hasattr(ch, "T"))

    def test_init_state_creates_m_h_n_gates(self) -> None:
        ch = self._make(size=2)
        V = _V([-40.0, -60.0])
        ca = _ca_info(2)
        ch.init_state(V, ca)
        self.assertEqual(ch.m.value.shape, (2,))
        self.assertEqual(ch.h.value.shape, (2,))
        self.assertEqual(ch.n.value.shape, (2,))

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = self._make(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V, ca), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = self._make(
            size=1,
            temp=u.celsius2kelvin(32.0),
            q10=3.0,
            temp_ref=u.celsius2kelvin(22.0),
        )
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.1])
        ch.h.value = jnp.array([0.2])
        ch.n.value = jnp.array([0.3])
        ch.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        exp_m = ch.gate_phi(gates["m"]) * (ch.f_m_inf(V, ca) - ch.m.value) / ch.f_m_tau(V, ca) / u.ms
        exp_h = ch.gate_phi(gates["h"]) * (ch.f_h_inf(V, ca) - ch.h.value) / ch.f_h_tau(V, ca) / u.ms
        exp_n = ch.gate_phi(gates["n"]) * (ch.f_n_inf(V, ca) - ch.n.value) / ch.f_n_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, exp_m, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.h.derivative, exp_h, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.n.derivative, exp_n, atol=1e-6 * u.Hz))

    def test_current_matches_formula(self) -> None:
        ch = self._make(size=1)
        V = _V([-50.0])
        ca = _ca_info(E_mV=140.0)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        ch.n.value = jnp.array([0.4])
        current = ch.current(V, ca)
        expected = ch.g_max * ch.m.value * ch.h.value * ch.n.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Cav1p2_MA2020Test(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p2_MA2020


class Cav1p3_MA2020Test(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p3_MA2020


class Cav3p1_MA2020Test(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p1_MA2020.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(Cav3p1_MA2020, HH))

    def test_gates_define_p2_q_structure(self) -> None:
        ch = Cav3p1_MA2020(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("p", "q"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_reset_state_matches_infinities(self) -> None:
        ch = Cav3p1_MA2020(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = Cav3p1_MA2020(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.3])
        ch.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        exp_p = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V) - ch.p.value) / ch.f_p_tau(V) / u.ms
        exp_q = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V) - ch.q.value) / ch.f_q_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_p, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_q, atol=1e-6 * u.Hz))

    def test_current_matches_ghk_formula(self) -> None:
        ch = Cav3p1_MA2020(size=1)
        V = _V([-40.0])
        ca = _ca_info(C=0.001, E_mV=140.0)
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.5])
        ch.q.value = jnp.array([0.25])
        current = ch.current(V, ca)
        expected = -ch.g_max * ch.p.value ** 2 * ch.q.value * ghk_flux(
            V=V,
            ci=ca.Ci,
            co=ca.Co,
            z=ch.z,
            T=ch.temp,
        )
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm ** 2)),
                expected.to_decimal(u.mA / (u.cm ** 2)),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
