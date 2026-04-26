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
from braincell.channel.potassium import (
    KDR_Ba2002,
    K_HH1952,
    K_Kv_test,
    K_Leak,
    K_TM1991,
    KA1_HM1992,
    KA2_HM1992,
    KK2A_HM1992,
    KK2B_HM1992,
    KNI_Ya1989,
)
from braincell.ion import Potassium


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 2.5) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
        valence=1,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


class _P4HHMixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_potassium(self) -> None:
        self.assertIs(self.CLS.root_type, Potassium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(self.CLS, HH))

    def test_single_p4_gate_is_declared(self) -> None:
        ch = self._make(size=1)
        gates = ch._iter_gates()
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].name, "p")
        self.assertEqual(gates[0].power, 4)

    def test_new_temperature_fields_replace_legacy_T_and_phi(self) -> None:
        ch = self._make(size=1)
        self.assertTrue(hasattr(ch, "temp"))
        self.assertTrue(hasattr(ch, "q10"))
        self.assertTrue(hasattr(ch, "temp_ref"))
        self.assertFalse(hasattr(ch, "T"))
        self.assertFalse(hasattr(ch, "phi"))
        self.assertFalse(hasattr(ch, "T_base"))

    def test_gate_phi_uses_standard_temp_interface(self) -> None:
        ch = self._make(
            size=1,
            temp=u.celsius2kelvin(46.0),
            q10=3.0,
            temp_ref=u.celsius2kelvin(36.0),
        )
        gate = ch._iter_gates()[0]
        self.assertTrue(
            u.math.allclose(ch.gate_phi(gate), 3.0 * jnp.ones(1), atol=1e-6)
        )

    def test_init_state_creates_p_shaped_to_size(self) -> None:
        ch = self._make(size=3)
        V = _V([-60.0, -65.0, -70.0])
        k = _k_info(3)
        ch.init_state(V, k)
        self.assertEqual(ch.p.value.shape, (3,))
        self.assertTrue(u.math.allclose(ch.p.value, jnp.zeros(3)))

    def test_reset_state_matches_steady_state(self) -> None:
        ch = self._make(size=1)
        V = _V([-65.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        alpha = ch.f_p_alpha(V)
        beta = ch.f_p_beta(V)
        self.assertTrue(
            u.math.allclose(ch.p.value, alpha / (alpha + beta), atol=1e-6)
        )

    def test_compute_derivative_matches_hh_alpha_beta_form(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.compute_derivative(V, k)
        alpha = ch.f_p_alpha(V)
        beta = ch.f_p_beta(V)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (alpha * (1.0 - ch.p.value) - beta * ch.p.value) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_is_g_max_p4_times_driving_force(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        current = ch.current(V, k)
        expected = ch.g_max * ch.p.value ** 4 * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_current_is_zero_when_p_zero(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.zeros(1)
        current = ch.current(V, k)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                jnp.zeros(1),
                atol=1e-9,
            )
        )


class KDR_Ba2002Test(_P4HHMixin, unittest.TestCase):
    CLS = KDR_Ba2002

    def test_fixed_phi_can_be_expressed_via_q10_mapping(self) -> None:
        ch = KDR_Ba2002(
            size=1,
            temp=u.celsius2kelvin(36.0),
            q10=0.25,
            temp_ref=u.celsius2kelvin(26.0),
        )
        gate = ch._iter_gates()[0]
        self.assertTrue(
            u.math.allclose(ch.gate_phi(gate), 0.25 * jnp.ones(1), atol=1e-6)
        )


class K_TM1991Test(_P4HHMixin, unittest.TestCase):
    CLS = K_TM1991


class K_HH1952Test(_P4HHMixin, unittest.TestCase):
    CLS = K_HH1952


class _P4QHHMixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_potassium(self) -> None:
        self.assertIs(self.CLS.root_type, Potassium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(self.CLS, HH))

    def test_gates_define_p4_q_structure(self) -> None:
        ch = self._make(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("p", "q"))
        self.assertEqual(tuple(gate.power for gate in gates), (4, 1))

    def test_new_temperature_fields_replace_legacy_phi_fields(self) -> None:
        ch = self._make(size=1)
        self.assertTrue(hasattr(ch, "temp"))
        self.assertTrue(hasattr(ch, "q10_p"))
        self.assertTrue(hasattr(ch, "temp_ref_p"))
        self.assertTrue(hasattr(ch, "q10_q"))
        self.assertTrue(hasattr(ch, "temp_ref_q"))
        self.assertFalse(hasattr(ch, "phi_p"))
        self.assertFalse(hasattr(ch, "phi_q"))

    def test_gate_phi_is_gate_specific(self) -> None:
        ch = self._make(
            size=1,
            temp=u.celsius2kelvin(46.0),
            q10_p=3.0,
            temp_ref_p=u.celsius2kelvin(36.0),
            q10_q=2.0,
            temp_ref_q=u.celsius2kelvin(36.0),
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
        k = _k_info(2)
        ch.init_state(V, k)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_sets_gates_to_inf(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.5])
        ch.compute_derivative(V, k)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        exp_dp = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V) - ch.p.value) / ch.f_p_tau(V) / u.ms
        exp_dq = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V) - ch.q.value) / ch.f_q_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_dp, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_dq, atol=1e-6 * u.Hz))

    def test_current_matches_p4_q_formula(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        current = ch.current(V, k)
        expected = ch.g_max * ch.p.value ** 4 * ch.q.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class KA1_HM1992Test(_P4QHHMixin, unittest.TestCase):
    CLS = KA1_HM1992


class KA2_HM1992Test(_P4QHHMixin, unittest.TestCase):
    CLS = KA2_HM1992


class _PQHHMixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_potassium(self) -> None:
        self.assertIs(self.CLS.root_type, Potassium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(self.CLS, HH))

    def test_gates_define_p_q_structure(self) -> None:
        ch = self._make(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("p", "q"))
        self.assertEqual(tuple(gate.power for gate in gates), (1, 1))

    def test_new_temperature_fields_replace_legacy_phi_fields(self) -> None:
        ch = self._make(size=1)
        self.assertTrue(hasattr(ch, "temp"))
        self.assertTrue(hasattr(ch, "q10_p"))
        self.assertTrue(hasattr(ch, "q10_q"))
        self.assertFalse(hasattr(ch, "phi_p"))
        self.assertFalse(hasattr(ch, "phi_q"))

    def test_init_state_creates_p_and_q(self) -> None:
        ch = self._make(size=2)
        V = _V([-60.0, -70.0])
        k = _k_info(2)
        ch.init_state(V, k)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_matches_infinities(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.5])
        ch.compute_derivative(V, k)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        exp_dp = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V) - ch.p.value) / ch.f_p_tau(V) / u.ms
        exp_dq = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V) - ch.q.value) / ch.f_q_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_dp, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_dq, atol=1e-6 * u.Hz))

    def test_current_matches_p_q_formula(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        current = ch.current(V, k)
        expected = ch.g_max * ch.p.value * ch.q.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class KK2A_HM1992Test(_PQHHMixin, unittest.TestCase):
    CLS = KK2A_HM1992


class KK2B_HM1992Test(_PQHHMixin, unittest.TestCase):
    CLS = KK2B_HM1992


class KNI_Ya1989Test(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(KNI_Ya1989.root_type, Potassium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(KNI_Ya1989, HH))

    def test_reset_state_matches_p_inf(self) -> None:
        ch = KNI_Ya1989(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = KNI_Ya1989(
            size=1,
            temp=u.celsius2kelvin(46.0),
            q10=3.0,
            temp_ref=u.celsius2kelvin(36.0),
        )
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.compute_derivative(V, k)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (ch.f_p_inf(V) - ch.p.value) / ch.f_p_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_formula(self) -> None:
        ch = KNI_Ya1989(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        current = ch.current(V, k)
        expected = ch.g_max * ch.p.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class IKLeakTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(K_Leak.root_type, Potassium)

    def test_current_follows_ohms_law(self) -> None:
        ch = K_Leak(size=1, g_max=0.005 * (u.mS / u.cm ** 2))
        V = _V([-60.0])
        k = _k_info()
        current = ch.current(V, k)
        expected = ch.g_max * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-9,
            )
        )

    def test_compute_derivative_is_noop(self) -> None:
        ch = K_Leak(size=1)
        self.assertIsNone(ch.compute_derivative(_V([-60.0]), _k_info()))


class K_Kv_testTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(K_Kv_test.root_type, Potassium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(K_Kv_test, HH))

    def test_single_n_gate_is_declared(self) -> None:
        ch = K_Kv_test(size=1)
        gates = ch._iter_gates()
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].name, "n")
        self.assertEqual(gates[0].power, 1)

    def test_reset_state_matches_f_n_inf(self) -> None:
        ch = K_Kv_test(size=1)
        V = _V([-20.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = K_Kv_test(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.n.value = jnp.array([0.25])
        ch.compute_derivative(V, k)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (ch.f_n_inf(V) - ch.n.value) / ch.f_n_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.n.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_linear_gating(self) -> None:
        ch = K_Kv_test(size=1, g_max=0.1 * (u.mS / u.cm ** 2))
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        current = ch.current(V, k)
        expected = ch.g_max * ch.n.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
