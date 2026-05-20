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

from braincell._base import HHTypedNeuron, IonInfo
from braincell.channel._base import HH, ghk_flux
from braincell.channel.calcium import (
    CaHVA_SU2015_DCN,
    CaLVA_SU2015_DCN,
    CaHVA_MA2020_GoC,
    CaHVA_MA2020_GrC,
    Ca_ZH2019_IO,
    CaHT_HM1992,
    CaHT_Re1993,
    CaL_IS2008,
    CaN_IS2008,
    CaT_HM1992,
    CaT_HP1992,
    Cav3p1_MA2024_PC,
    Cav2p1_MA2024_PC,
    Cav2p1_MA2025_BC,
    Cav2p1_RI2021_SC,
    Cav3p2_MA2024_PC,
    Cav3p2_MA2025_BC,
    Cav3p2_RI2021_SC,
    Cav3p3_MA2024_PC,
    Cav3p3_RI2021_SC,
    Cav2p3_MA2020_GoC,
    Cav1p2_MA2020_GoC,
    Cav1p3_MA2020_GoC,
    Cav3p1_MA2020_GoC,
)
from braincell.ion import Calcium


def _ca_info(
    size: int = 1,
    C: float = 1e-4,
    E_mV: float = 120.0,
    e_mV: float | None = None,
) -> IonInfo:
    if e_mV is not None:
        E_mV = e_mV
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
            expected_p = ch.f_p_inf(V, ca)
        else:
            alpha_p = ch.f_p_alpha(V, ca)
            beta_p = ch.f_p_beta(V, ca)
            expected_p = alpha_p / (alpha_p + beta_p)

        if ch._gate_form(gates["q"]) == "inf_tau":
            expected_q = ch.f_q_inf(V, ca)
        else:
            alpha_q = ch.f_q_alpha(V, ca)
            beta_q = ch.f_q_beta(V, ca)
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
            exp_dp = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V, ca) - ch.p.value) / ch.f_p_tau(V, ca) / u.ms
        else:
            alpha_p = ch.f_p_alpha(V, ca)
            beta_p = ch.f_p_beta(V, ca)
            exp_dp = ch.gate_phi(gates["p"]) * (alpha_p * (1.0 - ch.p.value) - beta_p * ch.p.value) / u.ms

        if ch._gate_form(gates["q"]) == "inf_tau":
            exp_dq = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V, ca) - ch.q.value) / ch.f_q_tau(V, ca) / u.ms
        else:
            alpha_q = ch.f_q_alpha(V, ca)
            beta_q = ch.f_q_beta(V, ca)
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


class CaHVA_SU2015_DCNTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(CaHVA_SU2015_DCN.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(CaHVA_SU2015_DCN, HH))

    def test_gate_definition_is_single_m_cubed(self) -> None:
        ch = CaHVA_SU2015_DCN(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("m",))
        self.assertEqual(tuple(gate.power for gate in gates), (3,))

    def test_default_parameters_are_stored(self) -> None:
        ch = CaHVA_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ch.perm, 7.5e-6 * (u.cm / u.second), atol=1e-12 * (u.cm / u.second)))
        self.assertTrue(u.math.allclose(ch.temp, u.celsius2kelvin(36.0), atol=1e-6 * u.kelvin))
        self.assertTrue(u.math.allclose(ch.qdeltat, jnp.ones(1), atol=1e-12))

    def test_reset_state_sets_m_to_minf(self) -> None:
        ch = CaHVA_SU2015_DCN(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))

    def test_compute_derivative_matches_inf_tau_form(self) -> None:
        ch = CaHVA_SU2015_DCN(size=1, qdeltat=2.0)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.2])
        ch.compute_derivative(V, ca)
        gate = ch._iter_gates()[0]
        expected = ch.gate_phi(gate) * (ch.f_m_inf(V, ca) - ch.m.value) / ch.f_m_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_mod_formula(self) -> None:
        ch = CaHVA_SU2015_DCN(size=1)
        V = _V([-60.0])
        ca = _ca_info(C=1e-4)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        current = ch.current(V, ca)
        v_mV = V.to_decimal(u.mV)
        temp = ch.temp.to_decimal(u.kelvin)
        ci = ca.Ci.to_decimal(u.mM)
        co = ca.Co.to_decimal(u.mM)
        perm = ch.perm.to_decimal(u.cm / u.second)
        A = u.math.exp(-23.20764929 * v_mV / temp)
        drive = (4.47814e6 * v_mV / temp) * ((ci / 1000.0) - (co / 1000.0) * A) / (1.0 - A)
        expected = -perm * ch.m.value ** 3 * drive
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm ** 2)),
                expected,
                atol=1e-6,
            )
        )


class CaLVA_SU2015_DCNTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(CaLVA_SU2015_DCN.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(CaLVA_SU2015_DCN, HH))

    def test_gate_definition_is_m_squared_h(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("m", "h"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_default_parameters_are_stored(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ch.perm, 1.0 * (u.cm / u.second), atol=1e-12 * (u.cm / u.second)))
        self.assertTrue(u.math.allclose(ch.temp, u.celsius2kelvin(36.0), atol=1e-6 * u.kelvin))
        self.assertTrue(u.math.allclose(ch.qdeltat, jnp.ones(1), atol=1e-12))

    def test_reset_state_sets_m_and_h_to_inf_values(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, ca), atol=1e-6))

    def test_compute_derivative_matches_inf_tau_form(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1, qdeltat=2.0)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.2])
        ch.h.value = jnp.array([0.6])
        ch.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        expected_m = ch.gate_phi(gates["m"]) * (ch.f_m_inf(V, ca) - ch.m.value) / ch.f_m_tau(V, ca) / u.ms
        expected_h = ch.gate_phi(gates["h"]) * (ch.f_h_inf(V, ca) - ch.h.value) / ch.f_h_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected_m, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.h.derivative, expected_h, atol=1e-6 * u.Hz))

    def test_h_tau_uses_low_voltage_branch_below_threshold(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        low_v = _V([-90.0])
        tau = ch.f_h_tau(low_v, _ca_info())
        expected = 0.333 * u.math.exp((-90.0 + 466.0) / 66.0)
        self.assertTrue(u.math.allclose(tau, jnp.array([expected]), atol=1e-6))

    def test_h_tau_uses_high_voltage_branch_at_threshold_and_above(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        high_v = _V([-81.0])
        tau = ch.f_h_tau(high_v, _ca_info())
        expected = 0.333 * u.math.exp((-81.0 + 21.0) / -10.5) + 9.32
        self.assertTrue(u.math.allclose(tau, jnp.array([expected]), atol=1e-6))

    def test_current_matches_mod_formula(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        V = _V([-60.0])
        ca = _ca_info(C=1e-4)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        current = ch.current(V, ca)
        v_mV = V.to_decimal(u.mV)
        temp = ch.temp.to_decimal(u.kelvin)
        ci = ca.Ci.to_decimal(u.mM)
        co = ca.Co.to_decimal(u.mM)
        perm = ch.perm.to_decimal(u.cm / u.second)
        A = u.math.exp(-23.20764929 * v_mV / temp)
        drive = (4.47814e6 * v_mV / temp) * ((ci / 1000.0) - (co / 1000.0) * A) / (1.0 - A)
        expected = -perm * ch.m.value ** 2 * ch.h.value * drive
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm ** 2)),
                expected,
                atol=1e-6,
            )
        )


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
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V, ca), atol=1e-6))

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
        expected = phi * (ch.f_p_inf(V, ca) - ch.p.value) / ch.f_p_tau(V, ca) / u.ms
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


class Cav1p2_MA2020_GoCTest(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p2_MA2020_GoC


class Cav1p3_MA2020_GoCTest(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p3_MA2020_GoC


class Cav3p1_MA2020_GoCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p1_MA2020_GoC.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(Cav3p1_MA2020_GoC, HH))

    def test_gates_define_p2_q_structure(self) -> None:
        ch = Cav3p1_MA2020_GoC(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("p", "q"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_reset_state_matches_infinities(self) -> None:
        ch = Cav3p1_MA2020_GoC(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V, ca), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = Cav3p1_MA2020_GoC(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.3])
        ch.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        exp_p = ch.gate_phi(gates["p"]) * (ch.f_p_inf(V, ca) - ch.p.value) / ch.f_p_tau(V, ca) / u.ms
        exp_q = ch.gate_phi(gates["q"]) * (ch.f_q_inf(V, ca) - ch.q.value) / ch.f_q_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_p, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_q, atol=1e-6 * u.Hz))

    def test_current_matches_ghk_formula(self) -> None:
        ch = Cav3p1_MA2020_GoC(size=1)
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
            temp=ch.temp,
        )
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm ** 2)),
                expected.to_decimal(u.mA / (u.cm ** 2)),
                atol=1e-6,
            )
        )


class Cav2p1RI21SCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav2p1_RI2021_SC.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(Cav2p1_RI2021_SC, HH))

    def test_gates_define_m3_structure(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("m",))
        self.assertEqual(tuple(gate.power for gate in gates), (3,))

    def test_reset_state_matches_template_minf(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1)
        V = _V([-30.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))

    def test_gate_phi_uses_default_ri2021_temperature_reference(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1, temp=u.celsius2kelvin(33.0))
        gate = ch._iter_gates()[0]
        self.assertTrue(
            u.math.allclose(ch.gate_phi(gate), 3.0 * jnp.ones(1), atol=1e-6)
        )

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1, temp=u.celsius2kelvin(33.0))
        V = _V([-30.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.2])
        ch.compute_derivative(V, ca)
        gate = ch._iter_gates()[0]
        expected = ch.gate_phi(gate) * (ch.f_m_inf(V, ca) - ch.m.value) / ch.f_m_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected, atol=1e-6 * u.Hz))

    def test_tau_matches_template_branches(self) -> None:
        ch = Cav2p1_RI2021_SC(size=2)
        ca = _ca_info(size=2)
        V = _V([-39.0, -41.0])
        tau = ch.f_m_tau(V, ca)
        expected = jnp.asarray(
            [
                0.2702 + 1.1622 * jnp.exp(-((-39.0 + 26.798) ** 2) / 164.19),
                0.6923 * jnp.exp(-41.0 / 1089.372),
            ]
        )
        self.assertTrue(u.math.allclose(tau, expected, atol=1e-6))

    def test_current_matches_ghk_template_formula(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1)
        V = _V([-40.0])
        ca = _ca_info(C=0.001)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        current = ch.current(V, ca)
        expected = -ch.g_max * ch.m.value ** 3 * ghk_flux(
            V=V,
            ci=ca.Ci,
            co=ca.Co,
            z=ch.z,
            temp=ch.temp,
        )
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm ** 2)),
                expected.to_decimal(u.mA / (u.cm ** 2)),
                atol=1e-6,
            )
        )

    def test_vshift_applies_to_rates_and_ghk_drive(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1, V_sh=5.0 * u.mV)
        V = _V([-35.0])
        ca = _ca_info(C=0.001)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])

        shifted = V - 5.0 * u.mV
        expected_inf = 1.0 / (1.0 + u.math.exp(-(shifted - ch.vhalfm) / ch.cvm))
        expected_current = -ch.g_max * ch.m.value ** 3 * ghk_flux(
            V=shifted,
            ci=ca.Ci,
            co=ca.Co,
            z=ch.z,
            temp=ch.temp,
        )

        self.assertTrue(u.math.allclose(ch.f_m_inf(V, ca), expected_inf, atol=1e-6))
        self.assertTrue(
            u.math.allclose(
                ch.current(V, ca).to_decimal(u.mA / (u.cm ** 2)),
                expected_current.to_decimal(u.mA / (u.cm ** 2)),
                atol=1e-6,
            )
        )


class Cav3p1MA24PCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p1_MA2024_PC.root_type, Calcium)

    def test_inherits_ma2020_variant(self) -> None:
        self.assertTrue(issubclass(Cav3p1_MA2024_PC, Cav3p1_MA2020_GoC))

    def test_matches_ma2020_variant(self) -> None:
        temp = u.celsius2kelvin(22.0)
        pc = Cav3p1_MA2024_PC(size=1, temp=temp)
        base = Cav3p1_MA2020_GoC(size=1, temp=temp)
        V = _V([-65.0])
        ca = _ca_info()

        pc.init_state(V, ca)
        base.init_state(V, ca)
        pc.reset_state(V, ca)
        base.reset_state(V, ca)

        self.assertTrue(u.math.allclose(pc.p.value, base.p.value, atol=1e-6))
        self.assertTrue(u.math.allclose(pc.q.value, base.q.value, atol=1e-6))


class Cav2p1MA25BCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav2p1_MA2025_BC.root_type, Calcium)

    def test_inherits_sc_variant(self) -> None:
        self.assertTrue(issubclass(Cav2p1_MA2025_BC, Cav2p1_RI2021_SC))

    def test_reset_state_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(23.0)
        bc = Cav2p1_MA2025_BC(size=1, temp=temp)
        sc = Cav2p1_RI2021_SC(size=1, temp=temp)
        V = _V([-30.0])
        ca = _ca_info()
        bc.init_state(V, ca)
        sc.init_state(V, ca)
        bc.reset_state(V, ca)
        sc.reset_state(V, ca)
        self.assertTrue(u.math.allclose(bc.m.value, sc.m.value, atol=1e-6))


class Cav2p1MA24PCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav2p1_MA2024_PC.root_type, Calcium)

    def test_inherits_sc_variant(self) -> None:
        self.assertTrue(issubclass(Cav2p1_MA2024_PC, Cav2p1_RI2021_SC))

    def test_reset_state_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(23.0)
        pc = Cav2p1_MA2024_PC(size=1, temp=temp)
        sc = Cav2p1_RI2021_SC(size=1, temp=temp)
        V = _V([-30.0])
        ca = _ca_info()
        pc.init_state(V, ca)
        sc.init_state(V, ca)
        pc.reset_state(V, ca)
        sc.reset_state(V, ca)
        self.assertTrue(u.math.allclose(pc.m.value, sc.m.value, atol=1e-6))


class Cav3p2RI21SCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p2_RI2021_SC.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(Cav3p2_RI2021_SC, HH))

    def test_gates_define_m2_h_structure(self) -> None:
        ch = Cav3p2_RI2021_SC(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("m", "h"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_reset_state_matches_template_inf_functions(self) -> None:
        ch = Cav3p2_RI2021_SC(size=1)
        V = _V([-65.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, ca), atol=1e-6))

    def test_gate_phi_matches_fixed_template_values(self) -> None:
        cold = Cav3p2_RI2021_SC(size=1, temp=u.celsius2kelvin(22.0))
        warm = Cav3p2_RI2021_SC(size=1, temp=u.celsius2kelvin(34.0))
        cold_gates = {gate.name: gate for gate in cold._iter_gates()}
        warm_gates = {gate.name: gate for gate in warm._iter_gates()}

        expected_m = (5.0 ** ((36.0 - 24.0) / 10.0)) * jnp.ones(1)
        expected_h = jnp.ones(1)

        self.assertTrue(u.math.allclose(cold.gate_phi(cold_gates["m"]), expected_m, atol=1e-6))
        self.assertTrue(u.math.allclose(cold.gate_phi(cold_gates["h"]), expected_h, atol=1e-6))
        self.assertTrue(u.math.allclose(warm.gate_phi(warm_gates["m"]), expected_m, atol=1e-6))
        self.assertTrue(u.math.allclose(warm.gate_phi(warm_gates["h"]), expected_h, atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = Cav3p2_RI2021_SC(size=1, temp=u.celsius2kelvin(34.0))
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.2])
        ch.h.value = jnp.array([0.3])
        ch.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        expected_m = ch.gate_phi(gates["m"]) * (ch.f_m_inf(V, ca) - ch.m.value) / ch.f_m_tau(V, ca) / u.ms
        expected_h = ch.gate_phi(gates["h"]) * (ch.f_h_inf(V, ca) - ch.h.value) / ch.f_h_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected_m, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.h.derivative, expected_h, atol=1e-6 * u.Hz))

    def test_current_matches_linear_template_formula(self) -> None:
        ch = Cav3p2_RI2021_SC(size=1)
        V = _V([-40.0])
        ca = _ca_info(E_mV=125.0)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        current = ch.current(V, ca)
        expected = ch.g_max * ch.m.value ** 2 * ch.h.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_vshift_applies_only_to_rates_with_template_sign(self) -> None:
        ch = Cav3p2_RI2021_SC(size=1, V_sh=5.0 * u.mV)
        V = _V([-60.0])
        ca = _ca_info(E_mV=120.0)
        shifted = V + 5.0 * u.mV
        expected_m_inf = 1.0 / (1.0 + u.math.exp(-(shifted.to_decimal(u.mV) + 54.8) / 7.4))
        expected_current = ch.g_max * 0.5 ** 2 * 0.25 * (ca.E - V)

        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])

        self.assertTrue(u.math.allclose(ch.f_m_inf(V, ca), expected_m_inf, atol=1e-6))
        self.assertTrue(
            u.math.allclose(
                ch.current(V, ca).to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Cav3p2MA25BCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p2_MA2025_BC.root_type, Calcium)

    def test_inherits_sc_variant(self) -> None:
        self.assertTrue(issubclass(Cav3p2_MA2025_BC, Cav3p2_RI2021_SC))

    def test_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(36.0)
        bc = Cav3p2_MA2025_BC(size=1, temp=temp)
        sc = Cav3p2_RI2021_SC(size=1, temp=temp)
        V = _V([-65.0])
        ca = _ca_info()
        bc.init_state(V, ca)
        sc.init_state(V, ca)
        bc.reset_state(V, ca)
        sc.reset_state(V, ca)
        self.assertTrue(u.math.allclose(bc.m.value, sc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(bc.h.value, sc.h.value, atol=1e-6))


class Cav3p2MA24PCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p2_MA2024_PC.root_type, Calcium)

    def test_inherits_sc_variant(self) -> None:
        self.assertTrue(issubclass(Cav3p2_MA2024_PC, Cav3p2_RI2021_SC))

    def test_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(36.0)
        pc = Cav3p2_MA2024_PC(size=1, temp=temp)
        sc = Cav3p2_RI2021_SC(size=1, temp=temp)
        V = _V([-65.0])
        ca = _ca_info()
        pc.init_state(V, ca)
        sc.init_state(V, ca)
        pc.reset_state(V, ca)
        sc.reset_state(V, ca)
        self.assertTrue(u.math.allclose(pc.m.value, sc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(pc.h.value, sc.h.value, atol=1e-6))


class Cav3p3RI21SCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p3_RI2021_SC.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(Cav3p3_RI2021_SC, HH))

    def test_gates_define_n2_l_structure(self) -> None:
        ch = Cav3p3_RI2021_SC(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("n", "l"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_reset_state_matches_template_inf_functions(self) -> None:
        ch = Cav3p3_RI2021_SC(size=1)
        V = _V([-65.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.l.value, ch.f_l_inf(V, ca), atol=1e-6))

    def test_gate_phi_uses_template_q10_from_28c(self) -> None:
        ch = Cav3p3_RI2021_SC(size=1, temp=u.celsius2kelvin(38.0))
        gates = {gate.name: gate for gate in ch._iter_gates()}
        expected = (2.3 ** ((38.0 - 28.0) / 10.0)) * jnp.ones(1)
        self.assertTrue(u.math.allclose(ch.gate_phi(gates["n"]), expected, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.gate_phi(gates["l"]), expected, atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = Cav3p3_RI2021_SC(size=1, temp=u.celsius2kelvin(36.0))
        V = _V([-65.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.n.value = jnp.array([0.2])
        ch.l.value = jnp.array([0.3])
        ch.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        expected_n = ch.gate_phi(gates["n"]) * (ch.f_n_inf(V, ca) - ch.n.value) / ch.f_n_tau(V, ca) / u.ms
        expected_l = ch.gate_phi(gates["l"]) * (ch.f_l_inf(V, ca) - ch.l.value) / ch.f_l_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.n.derivative, expected_n, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.l.derivative, expected_l, atol=1e-6 * u.Hz))

    def test_tau_matches_template_branches(self) -> None:
        ch = Cav3p3_RI2021_SC(size=2)
        ca = _ca_info(size=2)
        V = _V([-59.0, -61.0])
        tau_n = ch.f_n_tau(V, ca)
        tau_l = ch.f_l_tau(V, ca)
        expected_n = jnp.asarray([
            7.2 + 0.02 * jnp.exp(59.0 / 14.7),
            0.875 * jnp.exp((-61.0 + 120.0) / 41.0),
        ])
        expected_l = jnp.asarray([
            79.5 + 2.0 * jnp.exp(59.0 / 9.3),
            260.0,
        ])
        self.assertTrue(u.math.allclose(tau_n, expected_n, atol=1e-6))
        self.assertTrue(u.math.allclose(tau_l, expected_l, atol=1e-6))

    def test_current_matches_ghk_template_formula(self) -> None:
        ch = Cav3p3_RI2021_SC(size=1)
        V = _V([-40.0])
        ca = _ca_info(C=0.001)
        ch.init_state(V, ca)
        ch.n.value = jnp.array([0.5])
        ch.l.value = jnp.array([0.25])
        current = ch.current(V, ca)
        expected = -ch.g_scale * ch.perm * ch.n.value ** 2 * ch.l.value * ghk_flux(
            V=V,
            ci=ca.Ci,
            co=ca.Co,
            z=ch.z,
            temp=ch.temp,
        )
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(current.unit),
                expected.to_decimal(expected.unit),
                atol=1e-6,
            )
        )


class Cav3p3MA24PCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p3_MA2024_PC.root_type, Calcium)

    def test_inherits_sc_variant(self) -> None:
        self.assertTrue(issubclass(Cav3p3_MA2024_PC, Cav3p3_RI2021_SC))

    def test_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(36.0)
        pc = Cav3p3_MA2024_PC(size=1, temp=temp)
        sc = Cav3p3_RI2021_SC(size=1, temp=temp)
        V = _V([-65.0])
        ca = _ca_info()
        pc.init_state(V, ca)
        sc.init_state(V, ca)
        pc.reset_state(V, ca)
        sc.reset_state(V, ca)
        self.assertTrue(u.math.allclose(pc.n.value, sc.n.value, atol=1e-6))
        self.assertTrue(u.math.allclose(pc.l.value, sc.l.value, atol=1e-6))


class CaHVAMA20GoCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(CaHVA_MA2020_GoC.root_type, Calcium)

    def test_current_uses_ca_reversal(self) -> None:
        ch = CaHVA_MA2020_GoC(size=1)
        V = _V([-20.0])
        ca = _ca_info(e_mV=100.0)
        ch.init_state(V, ca)
        ch.s.value = jnp.array([0.5])
        ch.u.value = jnp.array([0.25])
        i = ch.current(V, ca)
        expected = ch.g_max * (ch.s.value ** 2) * ch.u.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_template_formulas_at_same_reversal(self) -> None:
        temp = u.celsius2kelvin(30.0)
        proto = CaHVA_MA2020_GoC(size=1, temp=temp)
        V = _V([-30.0])
        ca = _ca_info()

        proto.init_state(V, ca)
        proto.reset_state(V, ca)
        alpha_s = proto.f_s_alpha(V, ca)
        beta_s = proto.f_s_beta(V, ca)
        alpha_u = proto.f_u_alpha(V, ca)
        beta_u = proto.f_u_beta(V, ca)
        self.assertTrue(u.math.allclose(proto.s.value, alpha_s / (alpha_s + beta_s), atol=1e-6))
        self.assertTrue(u.math.allclose(proto.u.value, alpha_u / (alpha_u + beta_u), atol=1e-6))

        proto.s.value = jnp.array([0.2])
        proto.u.value = jnp.array([0.6])
        proto.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in proto._iter_gates()}
        expected_s = proto.gate_phi(gates["s"]) * (alpha_s * (1.0 - proto.s.value) - beta_s * proto.s.value) / u.ms
        expected_u = proto.gate_phi(gates["u"]) * (alpha_u * (1.0 - proto.u.value) - beta_u * proto.u.value) / u.ms
        self.assertTrue(u.math.allclose(proto.s.derivative, expected_s, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(proto.u.derivative, expected_u, atol=1e-6 * u.Hz))

        i_proto = proto.current(V, ca)
        expected_current = proto.g_max * (proto.s.value ** 2) * proto.u.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Cav2p3MA20GoCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav2p3_MA2020_GoC.root_type, Calcium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = Cav2p3_MA2020_GoC(size=1)
        V = _V([-60.0])
        ca = _ca_info(e_mV=140.0)
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, ca), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Cav2p3_MA2020_GoC(size=1, g_max=0.1 * (u.mS / u.cm ** 2))
        V = _V([-40.0])
        ca = _ca_info(e_mV=140.0)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        i = ch.current(V, ca)
        expected = ch.g_max * (ch.m.value ** 3) * ch.h.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_template_formulas_at_same_reversal(self) -> None:
        temp = u.celsius2kelvin(34.0)
        proto = Cav2p3_MA2020_GoC(size=1, temp=temp)
        V = _V([-40.0])
        ca = _ca_info(e_mV=140.0)

        proto.init_state(V, ca)
        proto.reset_state(V, ca)
        self.assertTrue(u.math.allclose(proto.m.value, proto.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(proto.h.value, proto.f_h_inf(V, ca), atol=1e-6))

        proto.m.value = jnp.array([0.3])
        proto.h.value = jnp.array([0.7])
        proto.compute_derivative(V, ca)
        gates = {gate.name: gate for gate in proto._iter_gates()}
        expected_m = proto.gate_phi(gates["m"]) * (proto.f_m_inf(V, ca) - proto.m.value) / proto.f_m_tau(V, ca) / u.ms
        expected_h = proto.gate_phi(gates["h"]) * (proto.f_h_inf(V, ca) - proto.h.value) / proto.f_h_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(proto.m.derivative, expected_m, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(proto.h.derivative, expected_h, atol=1e-6 * u.Hz))

        i_proto = proto.current(V, ca)
        expected_current = proto.g_max * (proto.m.value ** 3) * proto.h.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class CaHVAMA20GrCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(CaHVA_MA2020_GrC.root_type, Calcium)

    def test_matches_goc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        goc = CaHVA_MA2020_GoC(size=1, temp=temp)
        grc = CaHVA_MA2020_GrC(size=1, temp=temp)
        V = _V([-30.0])
        ca = _ca_info()

        goc.init_state(V, ca)
        grc.init_state(V, ca)
        goc.reset_state(V, ca)
        grc.reset_state(V, ca)
        self.assertTrue(u.math.allclose(grc.s.value, goc.s.value, atol=1e-6))
        self.assertTrue(u.math.allclose(grc.u.value, goc.u.value, atol=1e-6))

        goc.compute_derivative(V, ca)
        grc.compute_derivative(V, ca)
        self.assertTrue(u.math.allclose(grc.s.derivative, goc.s.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(grc.u.derivative, goc.u.derivative, atol=1e-6 * u.Hz))

        i_goc = goc.current(V, ca)
        i_grc = grc.current(V, ca)
        self.assertTrue(
            u.math.allclose(
                i_grc.to_decimal(_DENSITY_UNIT),
                i_goc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class CaZH19IOTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Ca_ZH2019_IO.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = Ca_ZH2019_IO(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_uses_instantaneous_m_and_stateful_h(self) -> None:
        ch = Ca_ZH2019_IO(size=1, E=120.0 * u.mV, mMidV=-61.0 * u.mV)
        V = _V([-50.0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])
        i = ch.current(V)
        expected = ch.g_max * ch.f_m_inf(V) * ch.h.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
