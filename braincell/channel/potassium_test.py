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
    KM_MA2020_GoC,
    KM_MA2020_GrC,
    KM_RI2021_SC,
    KA1_HM1992,
    KA2_HM1992,
    Kdr_ZH2019_IO,
    KK2A_HM1992,
    KK2B_HM1992,
    KNI_Ya1989,
    Kir2p3_MA2020_GrC,
    Kir2p3_MA2024_PC,
    Kir2p3_MA2025_BC,
    Kir2p3_RI2021_SC,
    Kv1p1_MA2020_GoC,
    Kv1p1_MA2020_GrC,
    Kv1p1_MA2024_PC,
    Kv1p1_MA2025_BC,
    Kv1p1_RI2021_SC,
    Kv1p5_MA2020_GrC,
    Kv1p5_MA2024_PC,
    Kv2p2_0010_MA2020_GrC,
    Kv3p3_MA2024_PC,
    Kv3p4_MA2020_GoC,
    Kv3p4_MA2020_GrC,
    Kv3p4_MA2024_PC,
    Kv3p4_MA2025_BC,
    Kv3p4_RI2021_SC,
    Kv4p3_MA2020_GoC,
    Kv4p3_MA2020_GrC,
    Kv4p3_MA2024_PC,
    Kv4p3_MA2025_BC,
    Kv4p3_RI2021_SC,
    _linoid_stable,
    fKdr_SU2015_DCN,
    sKdr_SU2015_DCN,
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
        alpha = ch.f_p_alpha(V, k)
        beta = ch.f_p_beta(V, k)
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
        alpha = ch.f_p_alpha(V, k)
        beta = ch.f_p_beta(V, k)
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
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V, k), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V, k), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.5])
        ch.compute_derivative(V, k)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        exp_dp = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V, k) - ch.p.value) / ch.f_p_tau(V, k) / u.ms
        exp_dq = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V, k) - ch.q.value) / ch.f_q_tau(V, k) / u.ms
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
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V, k), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V, k), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.5])
        ch.compute_derivative(V, k)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        exp_dp = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V, k) - ch.p.value) / ch.f_p_tau(V, k) / u.ms
        exp_dq = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V, k) - ch.q.value) / ch.f_q_tau(V, k) / u.ms
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
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V, k), atol=1e-6))

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
        expected = phi * (ch.f_p_inf(V, k) - ch.p.value) / ch.f_p_tau(V, k) / u.ms
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
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V, k), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = K_Kv_test(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.n.value = jnp.array([0.25])
        ch.compute_derivative(V, k)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (ch.f_n_inf(V, k) - ch.n.value) / ch.f_n_tau(V, k) / u.ms
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


class Kir2p3MA25BCTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(Kir2p3_MA2025_BC.root_type, Potassium)

    def test_reset_state_matches_alpha_beta_ratio(self) -> None:
        ch = Kir2p3_MA2025_BC(size=1)
        V = _V([-75.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        alpha = ch.f_d_alpha(V, k)
        beta = ch.f_d_beta(V, k)
        self.assertTrue(
            u.math.allclose(ch.d.value, alpha / (alpha + beta), atol=1e-6)
        )


class Kv1p1MA25BCTest(unittest.TestCase):
    def test_matches_template_formulas_without_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = Kv1p1_MA2025_BC(size=1, temp=temp, gateCurrent=0.0)
        V = _V([-60.0])
        k = _k_info()

        proto.init_state(V, k)
        proto.reset_state(V, k)
        alpha = proto.f_n_alpha(V, k)
        beta = proto.f_n_beta(V, k)
        self.assertTrue(u.math.allclose(proto.n.value, alpha / (alpha + beta), atol=1e-6))

        proto.compute_derivative(V, k)
        phi = proto.gate_phi(proto._iter_gates()[0])
        expected_derivative = phi * (alpha * (1.0 - proto.n.value) - beta * proto.n.value) / u.ms
        self.assertTrue(
            u.math.allclose(proto.n.derivative, expected_derivative, atol=1e-6 * u.Hz)
        )

        i_proto = proto.current(V, k)
        expected_current = proto.g_max * proto.n.value ** 4 * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_gating_current_path_matches_manual_formula(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = Kv1p1_MA2025_BC(size=1, temp=temp, gateCurrent=1.0)
        V = _V([-50.0])
        k = _k_info()

        proto.init_state(V, k)
        proto.reset_state(V, k)
        proto.n.value = jnp.array([0.35])
        alpha = proto.f_n_alpha(V, k)
        beta = proto.f_n_beta(V, k)
        phi = proto.gate_phi(proto._iter_gates()[0])
        conductive = proto.g_max * proto.n.value ** 4 * (k.E - V)
        ngate_flip = phi * (alpha * (1.0 - proto.n.value) - beta * proto.n.value) / u.ms
        nc = 1e12 * proto.g_max / proto.gunit
        igate = nc * 1e6 * proto.e0 * 4.0 * proto.zn * ngate_flip
        expected_current = conductive - igate

        i_proto = proto.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv3p4MA25BCTest(unittest.TestCase):
    def test_matches_template_state_derivative_and_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = Kv3p4_MA2025_BC(size=1, temp=temp)
        V = _V([-45.0])
        k = _k_info()

        proto.init_state(V, k)
        proto.reset_state(V, k)
        self.assertTrue(u.math.allclose(proto.m.value, proto.f_m_inf(V, k), atol=1e-6))
        self.assertTrue(u.math.allclose(proto.h.value, proto.f_h_inf(V, k), atol=1e-6))

        proto.m.value = jnp.array([0.2])
        proto.h.value = jnp.array([0.7])
        proto.compute_derivative(V, k)
        gates = {gate.name: gate for gate in proto._iter_gates()}
        expected_m = proto.gate_phi(gates["m"]) * (proto.f_m_inf(V, k) - proto.m.value) / proto.f_m_tau(V, k) / u.ms
        expected_h = proto.gate_phi(gates["h"]) * (proto.f_h_inf(V, k) - proto.h.value) / proto.f_h_tau(V, k) / u.ms
        self.assertTrue(
            u.math.allclose(proto.m.derivative, expected_m, atol=1e-6 * u.Hz)
        )
        self.assertTrue(
            u.math.allclose(proto.h.derivative, expected_h, atol=1e-6 * u.Hz)
        )

        i_proto = proto.current(V, k)
        expected_current = proto.g_max * (proto.m.value ** 3) * proto.h.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv4p3MA25BCTest(unittest.TestCase):
    def test_matches_template_state_derivative_and_current(self) -> None:
        temp = u.celsius2kelvin(30.0)
        proto = Kv4p3_MA2025_BC(size=1, temp=temp)
        V = _V([-55.0])
        k = _k_info()

        proto.init_state(V, k)
        proto.reset_state(V, k)
        self.assertTrue(u.math.allclose(proto.a.value, proto.f_a_inf(V, k), atol=1e-6))
        self.assertTrue(u.math.allclose(proto.b.value, proto.f_b_inf(V, k), atol=1e-6))

        proto.a.value = jnp.array([0.2])
        proto.b.value = jnp.array([0.7])
        proto.compute_derivative(V, k)
        gates = {gate.name: gate for gate in proto._iter_gates()}
        expected_a = proto.gate_phi(gates["a"]) * (proto.f_a_inf(V, k) - proto.a.value) / proto.f_a_tau(V, k) / u.ms
        expected_b = proto.gate_phi(gates["b"]) * (proto.f_b_inf(V, k) - proto.b.value) / proto.f_b_tau(V, k) / u.ms
        self.assertTrue(
            u.math.allclose(proto.a.derivative, expected_a, atol=1e-6 * u.Hz)
        )
        self.assertTrue(
            u.math.allclose(proto.b.derivative, expected_b, atol=1e-6 * u.Hz)
        )

        i_proto = proto.current(V, k)
        expected_current = proto.g_max * (proto.a.value ** 3) * proto.b.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_linoid_helper_uses_small_ratio_branch(self) -> None:
        x = jnp.array([1.0e-8, 1.0])
        y = jnp.array([1.0, 1.0])
        result = _linoid_stable(x, y)
        expected = jnp.array(
            [
                1.0 * (1.0 - 1.0e-8 / 2.0),
                1.0 / (jnp.exp(1.0) - 1.0),
            ]
        )
        self.assertTrue(bool(jnp.allclose(result, expected, atol=1e-6)))


class Kir2p3MA24PCTest(unittest.TestCase):
    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        bc = Kir2p3_MA2025_BC(size=1, temp=temp)
        pc = Kir2p3_MA2024_PC(size=1, temp=temp)
        V = _V([-75.0])
        k = _k_info()

        bc.init_state(V, k)
        pc.init_state(V, k)
        bc.reset_state(V, k)
        pc.reset_state(V, k)
        self.assertTrue(u.math.allclose(pc.d.value, bc.d.value, atol=1e-6))

        bc.compute_derivative(V, k)
        pc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(pc.d.derivative, bc.d.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_pc = pc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_pc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv1p1MA24PCTest(unittest.TestCase):
    def test_matches_bc_variant_without_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        bc = Kv1p1_MA2025_BC(size=1, temp=temp, gateCurrent=0.0)
        pc = Kv1p1_MA2024_PC(size=1, temp=temp, gateCurrent=0.0)
        V = _V([-60.0])
        k = _k_info()

        bc.init_state(V, k)
        pc.init_state(V, k)
        bc.reset_state(V, k)
        pc.reset_state(V, k)
        self.assertTrue(u.math.allclose(pc.n.value, bc.n.value, atol=1e-6))

        bc.compute_derivative(V, k)
        pc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(pc.n.derivative, bc.n.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_pc = pc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_pc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_bc_variant_with_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        bc = Kv1p1_MA2025_BC(size=1, temp=temp, gateCurrent=1.0)
        pc = Kv1p1_MA2024_PC(size=1, temp=temp, gateCurrent=1.0)
        V = _V([-50.0])
        k = _k_info()

        bc.init_state(V, k)
        pc.init_state(V, k)
        bc.reset_state(V, k)
        pc.reset_state(V, k)

        i_bc = bc.current(V, k)
        i_pc = pc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_pc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv1p5MA24PCTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(Kv1p5_MA2024_PC.root_type, Potassium)

    def test_declares_mod_gates_and_powers(self) -> None:
        ch = Kv1p5_MA2024_PC(size=1)
        gates = ch._iter_gates()
        self.assertEqual([(gate.name, gate.power) for gate in gates], [("m", 3), ("n", 1), ("u", 1)])

    def test_reset_state_matches_mod_steady_state(self) -> None:
        ch = Kv1p5_MA2024_PC(size=1)
        V = _V([-40.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        ch.init_state(V, k)
        ch.reset_state(V, k)

        expected_m = 1.0 / (1.0 + jnp.exp(-(v + 30.3) / 9.6))
        expected_n = 0.25 + 1.0 / (1.35 + jnp.exp((v + 7.0) / 14.0))
        expected_u = 0.1 + 1.0 / (1.1 + jnp.exp((v + 7.0) / 14.0))

        self.assertTrue(u.math.allclose(ch.m.value, expected_m, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.n.value, expected_n, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.u.value, expected_u, atol=1e-6))

    def test_tau_matches_mod_rates_and_temperature_scaling(self) -> None:
        ch = Kv1p5_MA2024_PC(size=1, temp=u.celsius2kelvin(47.0), Tauact=2.0, Tauinactf=3.0, Tauinacts=4.0)
        V = _V([-35.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        q10 = 2.2

        m_alpha = q10 * 0.65 / (jnp.exp(-(v + 10.0) / 8.5) + jnp.exp(-(v - 30.0) / 59.0))
        m_beta = q10 * 0.65 / (2.5 + jnp.exp((v + 82.0) / 17.0))
        n_alpha = q10 * 0.001 / (2.4 + 10.9 * jnp.exp(-(v + 90.0) / 78.0))
        n_beta = q10 * 0.001 * jnp.exp((v - 168.0) / 16.0)

        self.assertTrue(u.math.allclose(ch.f_m_tau(V, k), 1.0 / (m_alpha + m_beta) / 3.0 * 2.0, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.f_n_tau(V, k), 1.0 / (n_alpha + n_beta) / 3.0 * 3.0, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.f_u_tau(V, k), jnp.full((1,), 6800.0 * 4.0), atol=1e-6))

    def test_current_matches_enabled_ik_path_only(self) -> None:
        ch = Kv1p5_MA2024_PC(size=1)
        V = _V([-20.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.m.value = jnp.array([0.2])
        ch.n.value = jnp.array([0.3])
        ch.u.value = jnp.array([0.4])
        v = V.to_decimal(u.mV)

        voltage_factor = 0.1 + 1.0 / (1.0 + jnp.exp(-(v - 15.0) / 13.0))
        expected = ch.g_max * voltage_factor * (0.2 ** 3) * 0.3 * 0.4 * (k.E - V)
        current = ch.current(V, k)

        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv1p5MA20GrCTest(unittest.TestCase):
    def test_inherits_pc_default_ik_path(self) -> None:
        self.assertTrue(issubclass(Kv1p5_MA2020_GrC, Kv1p5_MA2024_PC))
        self.assertIs(Kv1p5_MA2020_GrC.root_type, Potassium)

    def test_matches_pc_variant_for_default_ik_path(self) -> None:
        temp = u.celsius2kelvin(36.0)
        pc = Kv1p5_MA2024_PC(size=1, temp=temp)
        grc = Kv1p5_MA2020_GrC(size=1, temp=temp)
        V = _V([-35.0])
        k = _k_info()

        pc.init_state(V, k)
        grc.init_state(V, k)
        pc.reset_state(V, k)
        grc.reset_state(V, k)
        self.assertTrue(u.math.allclose(grc.m.value, pc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(grc.n.value, pc.n.value, atol=1e-6))
        self.assertTrue(u.math.allclose(grc.u.value, pc.u.value, atol=1e-6))

        pc.compute_derivative(V, k)
        grc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(grc.m.derivative, pc.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(grc.n.derivative, pc.n.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(grc.u.derivative, pc.u.derivative, atol=1e-6 * u.Hz))

        i_pc = pc.current(V, k)
        i_grc = grc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_pc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv3p3MA24PCTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(Kv3p3_MA2024_PC.root_type, Potassium)

    def test_declares_hh_n4_gate(self) -> None:
        ch = Kv3p3_MA2024_PC(size=1)
        gates = ch._iter_gates()
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].name, "n")
        self.assertEqual(gates[0].power, 4)

    def test_reset_state_matches_mod_rate_ratio(self) -> None:
        ch = Kv3p3_MA2024_PC(size=1)
        V = _V([-30.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        ch.init_state(V, k)
        ch.reset_state(V, k)

        alpha = 0.22 * jnp.exp(-(v + 16.0) / -26.5)
        beta = 0.22 * jnp.exp(-(v + 16.0) / 26.5)

        self.assertTrue(u.math.allclose(ch.n.value, alpha / (alpha + beta), atol=1e-6))

    def test_derivative_uses_alpha_beta_and_temperature_scaling(self) -> None:
        ch = Kv3p3_MA2024_PC(size=1, temp=u.celsius2kelvin(32.0))
        V = _V([-20.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.n.value = jnp.array([0.35])

        ch.compute_derivative(V, k)

        alpha = ch.f_n_alpha(V, k)
        beta = ch.f_n_beta(V, k)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (alpha * (1.0 - ch.n.value) - beta * ch.n.value) / u.ms
        self.assertTrue(u.math.allclose(ch.n.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_default_ik_path(self) -> None:
        ch = Kv3p3_MA2024_PC(size=1, gateCurrent=0.0)
        V = _V([-20.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.n.value = jnp.array([0.4])

        current = ch.current(V, k)
        expected = ch.g_max * ch.n.value ** 4 * (k.E - V)

        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_gating_current_path_matches_mod_formula(self) -> None:
        ch = Kv3p3_MA2024_PC(size=1, gateCurrent=1.0)
        V = _V([-10.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.n.value = jnp.array([0.35])

        alpha = ch.f_n_alpha(V, k)
        beta = ch.f_n_beta(V, k)
        phi = ch.gate_phi(ch._iter_gates()[0])
        conductive = ch.g_max * ch.n.value ** 4 * (k.E - V)
        ngate_flip = phi * (alpha * (1.0 - ch.n.value) - beta * ch.n.value) / u.ms
        nc = 1e12 * ch.g_max / ch.gunit
        igate = nc * 1e6 * ch.e0 * 4.0 * ch.zn * ngate_flip
        expected = conductive - igate

        current = ch.current(V, k)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv3p4MA24PCTest(unittest.TestCase):
    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(22.0)
        bc = Kv3p4_MA2025_BC(size=1, temp=temp)
        pc = Kv3p4_MA2024_PC(size=1, temp=temp)
        V = _V([-45.0])
        k = _k_info()

        bc.init_state(V, k)
        pc.init_state(V, k)
        bc.reset_state(V, k)
        pc.reset_state(V, k)
        self.assertTrue(u.math.allclose(pc.m.value, bc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(pc.h.value, bc.h.value, atol=1e-6))

        bc.compute_derivative(V, k)
        pc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(pc.m.derivative, bc.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(pc.h.derivative, bc.h.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_pc = pc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_pc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv4p3MA24PCTest(unittest.TestCase):
    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        bc = Kv4p3_MA2025_BC(size=1, temp=temp)
        pc = Kv4p3_MA2024_PC(size=1, temp=temp)
        V = _V([-55.0])
        k = _k_info()

        bc.init_state(V, k)
        pc.init_state(V, k)
        bc.reset_state(V, k)
        pc.reset_state(V, k)
        self.assertTrue(u.math.allclose(pc.a.value, bc.a.value, atol=1e-6))
        self.assertTrue(u.math.allclose(pc.b.value, bc.b.value, atol=1e-6))

        bc.compute_derivative(V, k)
        pc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(pc.a.derivative, bc.a.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(pc.b.derivative, bc.b.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_pc = pc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_pc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class KMRI21SCTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(KM_RI2021_SC.root_type, Potassium)

    def test_reset_state_matches_f_n_inf(self) -> None:
        ch = KM_RI2021_SC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = KM_RI2021_SC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.n.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_template_gate_dynamics(self) -> None:
        temp = u.celsius2kelvin(30.0)
        proto = KM_RI2021_SC(size=1, temp=temp)
        V = _V([-60.0])
        k = _k_info()

        proto.init_state(V, k)
        proto.reset_state(V, k)
        self.assertTrue(u.math.allclose(proto.n.value, proto.f_n_inf(V, k), atol=1e-6))

        proto.n.value = jnp.array([0.25])
        proto.compute_derivative(V, k)
        expected = (
            proto.gate_phi(proto._iter_gates()[0])
            * (proto.f_n_inf(V, k) - proto.n.value)
            / proto.f_n_tau(V, k)
            / u.ms
        )
        self.assertTrue(
            u.math.allclose(proto.n.derivative, expected, atol=1e-6 * u.Hz)
        )


class Kir2p3RI21SCTest(unittest.TestCase):
    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        bc = Kir2p3_MA2025_BC(size=1, temp=temp)
        sc = Kir2p3_RI2021_SC(size=1, temp=temp)
        V = _V([-75.0])
        k = _k_info()

        bc.init_state(V, k)
        sc.init_state(V, k)
        bc.reset_state(V, k)
        sc.reset_state(V, k)
        self.assertTrue(u.math.allclose(sc.d.value, bc.d.value, atol=1e-6))

        bc.compute_derivative(V, k)
        sc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(sc.d.derivative, bc.d.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_sc = sc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_sc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class fKdrSU15DCNTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(fKdr_SU2015_DCN.root_type, Potassium)

    def test_reset_state_matches_f_m_inf(self) -> None:
        ch = fKdr_SU2015_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = fKdr_SU2015_DCN(size=1)
        V = _V([-40.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.m.value = jnp.array([0.5])
        i = ch.current(V, k)
        expected = ch.g_max * (ch.m.value ** 4) * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_tau_matches_mod_formula(self) -> None:
        ch = fKdr_SU2015_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        expected = 13.9 / (jnp.exp((v + 40.0) / 12.0) + jnp.exp((v + 40.0) / -13.0)) + 0.1
        self.assertTrue(u.math.allclose(ch.f_m_tau(V, k), expected, atol=1e-6))


class sKdrSU15DCNTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(sKdr_SU2015_DCN.root_type, Potassium)

    def test_reset_state_matches_f_m_inf(self) -> None:
        ch = sKdr_SU2015_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = sKdr_SU2015_DCN(size=1)
        V = _V([-40.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.m.value = jnp.array([0.5])
        i = ch.current(V, k)
        expected = ch.g_max * (ch.m.value ** 4) * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_tau_matches_mod_formula_and_differs_from_fast_kdr(self) -> None:
        slow = sKdr_SU2015_DCN(size=1)
        fast = fKdr_SU2015_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        expected = 14.95 / (jnp.exp((v + 50.0) / 21.74) + jnp.exp((v + 50.0) / -13.91)) + 0.05
        self.assertTrue(u.math.allclose(slow.f_m_tau(V, k), expected, atol=1e-6))
        self.assertFalse(bool(jnp.allclose(slow.f_m_tau(V, k), fast.f_m_tau(V, k), atol=1e-6)))


class KMMA20GoCTest(unittest.TestCase):
    def test_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        sc = KM_RI2021_SC(size=1, temp=temp)
        goc = KM_MA2020_GoC(size=1, temp=temp)
        V = _V([-60.0])
        k = _k_info()

        sc.init_state(V, k)
        goc.init_state(V, k)
        sc.reset_state(V, k)
        goc.reset_state(V, k)
        self.assertTrue(u.math.allclose(goc.n.value, sc.n.value, atol=1e-6))

        sc.compute_derivative(V, k)
        goc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(goc.n.derivative, sc.n.derivative, atol=1e-6 * u.Hz))

        i_sc = sc.current(V, k)
        i_goc = goc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv1p1MA20GoCTest(unittest.TestCase):
    def test_matches_sc_variant_without_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        sc = Kv1p1_RI2021_SC(size=1, temp=temp, gateCurrent=0.0)
        goc = Kv1p1_MA2020_GoC(size=1, temp=temp, gateCurrent=0.0)
        V = _V([-60.0])
        k = _k_info()

        sc.init_state(V, k)
        goc.init_state(V, k)
        sc.reset_state(V, k)
        goc.reset_state(V, k)
        self.assertTrue(u.math.allclose(goc.n.value, sc.n.value, atol=1e-6))

        sc.compute_derivative(V, k)
        goc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(goc.n.derivative, sc.n.derivative, atol=1e-6 * u.Hz))

        i_sc = sc.current(V, k)
        i_goc = goc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_sc_variant_with_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        sc = Kv1p1_RI2021_SC(size=1, temp=temp, gateCurrent=1.0)
        goc = Kv1p1_MA2020_GoC(size=1, temp=temp, gateCurrent=1.0)
        V = _V([-50.0])
        k = _k_info()

        sc.init_state(V, k)
        goc.init_state(V, k)
        sc.reset_state(V, k)
        goc.reset_state(V, k)

        i_sc = sc.current(V, k)
        i_goc = goc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv3p4MA20GoCTest(unittest.TestCase):
    def test_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(22.0)
        sc = Kv3p4_RI2021_SC(size=1, temp=temp)
        goc = Kv3p4_MA2020_GoC(size=1, temp=temp)
        V = _V([-45.0])
        k = _k_info()

        sc.init_state(V, k)
        goc.init_state(V, k)
        sc.reset_state(V, k)
        goc.reset_state(V, k)
        self.assertTrue(u.math.allclose(goc.m.value, sc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(goc.h.value, sc.h.value, atol=1e-6))

        sc.compute_derivative(V, k)
        goc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(goc.m.derivative, sc.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(goc.h.derivative, sc.h.derivative, atol=1e-6 * u.Hz))

        i_sc = sc.current(V, k)
        i_goc = goc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv4p3MA20GoCTest(unittest.TestCase):
    def test_matches_sc_variant_at_default_temperature(self) -> None:
        temp = u.celsius2kelvin(22.0)
        sc = Kv4p3_RI2021_SC(size=1, temp=temp)
        goc = Kv4p3_MA2020_GoC(size=1)
        V = _V([-55.0])
        k = _k_info()

        sc.init_state(V, k)
        goc.init_state(V, k)
        sc.reset_state(V, k)
        goc.reset_state(V, k)
        self.assertTrue(u.math.allclose(goc.a.value, sc.a.value, atol=1e-6))
        self.assertTrue(u.math.allclose(goc.b.value, sc.b.value, atol=1e-6))

        sc.compute_derivative(V, k)
        goc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(goc.a.derivative, sc.a.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(goc.b.derivative, sc.b.derivative, atol=1e-6 * u.Hz))

        i_sc = sc.current(V, k)
        i_goc = goc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class KMMA20GrCTest(unittest.TestCase):
    def test_matches_goc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        goc = KM_MA2020_GoC(size=1, temp=temp)
        grc = KM_MA2020_GrC(size=1, temp=temp)
        V = _V([-60.0])
        k = _k_info()

        goc.init_state(V, k)
        grc.init_state(V, k)
        goc.reset_state(V, k)
        grc.reset_state(V, k)
        self.assertTrue(u.math.allclose(grc.n.value, goc.n.value, atol=1e-6))

        goc.compute_derivative(V, k)
        grc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(grc.n.derivative, goc.n.derivative, atol=1e-6 * u.Hz))

        i_goc = goc.current(V, k)
        i_grc = grc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_goc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kir2p3MA20GrCTest(unittest.TestCase):
    def test_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        sc = Kir2p3_RI2021_SC(size=1, temp=temp)
        grc = Kir2p3_MA2020_GrC(size=1, temp=temp)
        V = _V([-75.0])
        k = _k_info()

        sc.init_state(V, k)
        grc.init_state(V, k)
        sc.reset_state(V, k)
        grc.reset_state(V, k)
        self.assertTrue(u.math.allclose(grc.d.value, sc.d.value, atol=1e-6))

        sc.compute_derivative(V, k)
        grc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(grc.d.derivative, sc.d.derivative, atol=1e-6 * u.Hz))

        i_sc = sc.current(V, k)
        i_grc = grc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv1p1MA20GrCTest(unittest.TestCase):
    def test_matches_goc_variant_without_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        goc = Kv1p1_MA2020_GoC(size=1, temp=temp, gateCurrent=0.0)
        grc = Kv1p1_MA2020_GrC(size=1, temp=temp, gateCurrent=0.0)
        V = _V([-60.0])
        k = _k_info()

        goc.init_state(V, k)
        grc.init_state(V, k)
        goc.reset_state(V, k)
        grc.reset_state(V, k)
        self.assertTrue(u.math.allclose(grc.n.value, goc.n.value, atol=1e-6))

        goc.compute_derivative(V, k)
        grc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(grc.n.derivative, goc.n.derivative, atol=1e-6 * u.Hz))

        i_goc = goc.current(V, k)
        i_grc = grc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_goc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_goc_variant_with_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        goc = Kv1p1_MA2020_GoC(size=1, temp=temp, gateCurrent=1.0)
        grc = Kv1p1_MA2020_GrC(size=1, temp=temp, gateCurrent=1.0)
        V = _V([-50.0])
        k = _k_info()

        goc.init_state(V, k)
        grc.init_state(V, k)
        goc.reset_state(V, k)
        grc.reset_state(V, k)

        i_goc = goc.current(V, k)
        i_grc = grc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_goc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv2p20010MA20GrCTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(Kv2p2_0010_MA2020_GrC.root_type, Potassium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = Kv2p2_0010_MA2020_GrC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, k), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Kv2p2_0010_MA2020_GrC(size=1)
        V = _V([-40.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        i = ch.current(V, k)
        expected = ch.g_max * ch.m.value * ch.h.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_tau_matches_mod_formula(self) -> None:
        ch = Kv2p2_0010_MA2020_GrC(size=1)
        V = _V([-60.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        expected_m_tau = 130.0 / (1.0 + jnp.exp((v + 46.56) / -44.14))
        expected_h_tau = 10000.0 / (1.0 + jnp.exp((v + 46.56) / -44.14))
        self.assertTrue(u.math.allclose(ch.f_m_tau(V, k), expected_m_tau, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.f_h_tau(V, k), expected_h_tau, atol=1e-6))


class Kv3p4MA20GrCTest(unittest.TestCase):
    def test_matches_goc_variant(self) -> None:
        temp = u.celsius2kelvin(22.0)
        goc = Kv3p4_MA2020_GoC(size=1, temp=temp)
        grc = Kv3p4_MA2020_GrC(size=1, temp=temp)
        V = _V([-45.0])
        k = _k_info()

        goc.init_state(V, k)
        grc.init_state(V, k)
        goc.reset_state(V, k)
        grc.reset_state(V, k)
        self.assertTrue(u.math.allclose(grc.m.value, goc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(grc.h.value, goc.h.value, atol=1e-6))

        goc.compute_derivative(V, k)
        grc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(grc.m.derivative, goc.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(grc.h.derivative, goc.h.derivative, atol=1e-6 * u.Hz))

        i_goc = goc.current(V, k)
        i_grc = grc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_goc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv4p3MA20GrCTest(unittest.TestCase):
    def test_matches_goc_variant_at_same_temperature(self) -> None:
        temp = u.celsius2kelvin(30.0)
        goc = Kv4p3_MA2020_GoC(size=1, temp=temp)
        grc = Kv4p3_MA2020_GrC(size=1)
        V = _V([-55.0])
        k = _k_info()

        goc.init_state(V, k)
        grc.init_state(V, k)
        goc.reset_state(V, k)
        grc.reset_state(V, k)
        self.assertTrue(u.math.allclose(grc.a.value, goc.a.value, atol=1e-6))
        self.assertTrue(u.math.allclose(grc.b.value, goc.b.value, atol=1e-6))

        goc.compute_derivative(V, k)
        grc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(grc.a.derivative, goc.a.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(grc.b.derivative, goc.b.derivative, atol=1e-6 * u.Hz))

        i_goc = goc.current(V, k)
        i_grc = grc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_goc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv1p1RI21SCTest(unittest.TestCase):
    def test_matches_bc_variant_without_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        bc = Kv1p1_MA2025_BC(size=1, temp=temp, gateCurrent=0.0)
        sc = Kv1p1_RI2021_SC(size=1, temp=temp, gateCurrent=0.0)
        V = _V([-60.0])
        k = _k_info()

        bc.init_state(V, k)
        sc.init_state(V, k)
        bc.reset_state(V, k)
        sc.reset_state(V, k)
        self.assertTrue(u.math.allclose(sc.n.value, bc.n.value, atol=1e-6))

        bc.compute_derivative(V, k)
        sc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(sc.n.derivative, bc.n.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_sc = sc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_sc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_bc_variant_with_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        bc = Kv1p1_MA2025_BC(size=1, temp=temp, gateCurrent=1.0)
        sc = Kv1p1_RI2021_SC(size=1, temp=temp, gateCurrent=1.0)
        V = _V([-50.0])
        k = _k_info()

        bc.init_state(V, k)
        sc.init_state(V, k)
        bc.reset_state(V, k)
        sc.reset_state(V, k)

        i_bc = bc.current(V, k)
        i_sc = sc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_sc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv3p4RI21SCTest(unittest.TestCase):
    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(22.0)
        bc = Kv3p4_MA2025_BC(size=1, temp=temp)
        sc = Kv3p4_RI2021_SC(size=1, temp=temp)
        V = _V([-45.0])
        k = _k_info()

        bc.init_state(V, k)
        sc.init_state(V, k)
        bc.reset_state(V, k)
        sc.reset_state(V, k)
        self.assertTrue(u.math.allclose(sc.m.value, bc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(sc.h.value, bc.h.value, atol=1e-6))

        bc.compute_derivative(V, k)
        sc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(sc.m.derivative, bc.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(sc.h.derivative, bc.h.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_sc = sc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_sc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv4p3RI21SCTest(unittest.TestCase):
    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        bc = Kv4p3_MA2025_BC(size=1, temp=temp)
        sc = Kv4p3_RI2021_SC(size=1, temp=temp)
        V = _V([-55.0])
        k = _k_info()

        bc.init_state(V, k)
        sc.init_state(V, k)
        bc.reset_state(V, k)
        sc.reset_state(V, k)
        self.assertTrue(u.math.allclose(sc.a.value, bc.a.value, atol=1e-6))
        self.assertTrue(u.math.allclose(sc.b.value, bc.b.value, atol=1e-6))

        bc.compute_derivative(V, k)
        sc.compute_derivative(V, k)
        self.assertTrue(u.math.allclose(sc.a.derivative, bc.a.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(sc.b.derivative, bc.b.derivative, atol=1e-6 * u.Hz))

        i_bc = bc.current(V, k)
        i_sc = sc.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_sc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class KdrZH19IOTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(Kdr_ZH2019_IO.root_type, Potassium)

    def test_reset_state_matches_f_n_inf(self) -> None:
        ch = Kdr_ZH2019_IO(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Kdr_ZH2019_IO(size=1)
        V = _V([-40.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.n.value = jnp.array([0.5])
        i = ch.current(V, k)
        expected = ch.g_max * (ch.n.value ** 4) * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_small_denominator_branch_is_stable(self) -> None:
        ch = Kdr_ZH2019_IO(size=1)
        self.assertTrue(u.math.allclose(ch._n_alpha(_V([-41.0])), jnp.array([10.0]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
