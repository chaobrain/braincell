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
from braincell.channel.hyperpolarization_activated import (
    HCN_HM1992,
    HCN1_MA2020_GoC,
    HCN1_MA2024_PC,
    HCN1_MA2025_BC,
    HCN1_RI2021_SC,
    HCN2_MA2020_GoC,
    HCN_SU2015_DCN,
    HCN_ZH2019_IO,
)


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


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


class HCN1MA25BCTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN1_MA2025_BC.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = HCN1_MA2025_BC(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN1_MA2025_BC(size=1, E=-34.4 * u.mV)
        V = _V([-65.0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])
        i = ch.current(V)
        expected = ch.g_max * ch.h.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class HCN1MA24PCTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN1_MA2024_PC.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = HCN1_MA2024_PC(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN1_MA2024_PC(size=1, E=-34.4 * u.mV)
        V = _V([-65.0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])
        i = ch.current(V)
        expected = ch.g_max * ch.h.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(23.0)
        bc = HCN1_MA2025_BC(size=1, temp=temp)
        pc = HCN1_MA2024_PC(size=1, temp=temp)
        V = _V([-70.0])

        bc.init_state(V)
        pc.init_state(V)
        bc.reset_state(V)
        pc.reset_state(V)
        self.assertTrue(u.math.allclose(pc.h.value, bc.h.value, atol=1e-6))

        i_bc = bc.current(V)
        i_pc = pc.current(V)
        self.assertTrue(
            u.math.allclose(
                i_pc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class HCN1RI21SCTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN1_RI2021_SC.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = HCN1_RI2021_SC(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN1_RI2021_SC(size=1, E=-34.4 * u.mV)
        V = _V([-65.0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])
        i = ch.current(V)
        expected = ch.g_max * ch.h.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(23.0)
        bc = HCN1_MA2025_BC(size=1, temp=temp)
        sc = HCN1_RI2021_SC(size=1, temp=temp)
        V = _V([-70.0])

        bc.init_state(V)
        sc.init_state(V)
        bc.reset_state(V)
        sc.reset_state(V)
        self.assertTrue(u.math.allclose(sc.h.value, bc.h.value, atol=1e-6))

        i_bc = bc.current(V)
        i_sc = sc.current(V)
        self.assertTrue(
            u.math.allclose(
                i_sc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class HCNSU15DCNTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN_SU2015_DCN.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_m_inf(self) -> None:
        ch = HCN_SU2015_DCN(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN_SU2015_DCN(size=1, E=-45.0 * u.mV)
        V = _V([-65.0])
        ch.init_state(V)
        ch.m.value = jnp.array([0.25])
        i = ch.current(V)
        expected = ch.g_max * (ch.m.value ** 2) * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_tau_is_constant_400_ms(self) -> None:
        V = _V([-80.0])
        self.assertTrue(
            u.math.allclose(
                HCN_SU2015_DCN(size=1).f_m_tau(V),
                jnp.array(400.0),
                atol=1e-6,
            )
        )


class HCN1MA20GoCTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN1_MA2020_GoC.root_type, HHTypedNeuron)

    def test_fast_and_slow_components_follow_template_formulas(self) -> None:
        temp = u.celsius2kelvin(22.0)
        ch = HCN1_MA2020_GoC(size=1, temp=temp)
        V = _V([-80.0])

        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.o_fast.value, ch.f_o_fast_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.o_slow.value, ch.f_o_slow_inf(V), atol=1e-6))
        self.assertTrue(
            u.math.allclose(
                ch.f_o_fast_inf(V) + ch.f_o_slow_inf(V),
                ch.o_inf(V),
                atol=1e-6,
            )
        )

        ch.o_fast.value = jnp.array([0.1])
        ch.o_slow.value = jnp.array([0.9])
        ch.compute_derivative(V)
        phi = 3.0 ** (((ch.temp - u.celsius2kelvin(23.0)) / u.kelvin) / 10.0)
        expected_fast = phi * (ch.f_o_fast_inf(V) - ch.o_fast.value) / ch.f_o_fast_tau(V) / u.ms
        expected_slow = phi * (ch.f_o_slow_inf(V) - ch.o_slow.value) / ch.f_o_slow_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.o_fast.derivative, expected_fast, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.o_slow.derivative, expected_slow, atol=1e-6 * u.Hz))

        current = ch.current(V)
        expected_current = ch._gbar_phi() * ch.g_max * (ch.o_fast.value + ch.o_slow.value) * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class HCN2MA20GoCTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN2_MA2020_GoC.root_type, HHTypedNeuron)

    def test_fast_and_slow_components_follow_template_formulas(self) -> None:
        temp = u.celsius2kelvin(22.0)
        ch = HCN2_MA2020_GoC(size=1, temp=temp)
        V = _V([-80.0])

        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.o_fast.value, ch.f_o_fast_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.o_slow.value, ch.f_o_slow_inf(V), atol=1e-6))

        ch.o_fast.value = jnp.array([0.1])
        ch.o_slow.value = jnp.array([0.9])
        ch.compute_derivative(V)
        phi = 3.0 ** (((ch.temp - u.celsius2kelvin(23.0)) / u.kelvin) / 10.0)
        expected_fast = phi * (ch.f_o_fast_inf(V) - ch.o_fast.value) / ch.f_o_fast_tau(V) / u.ms
        expected_slow = phi * (ch.f_o_slow_inf(V) - ch.o_slow.value) / ch.f_o_slow_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.o_fast.derivative, expected_fast, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.o_slow.derivative, expected_slow, atol=1e-6 * u.Hz))

        current = ch.current(V)
        expected_current = ch._gbar_phi() * ch.g_max * (ch.o_fast.value + ch.o_slow.value) * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected_current.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_r_is_clamped_to_corridor(self) -> None:
        ch = HCN2_MA2020_GoC(size=1)
        self.assertTrue(u.math.allclose(ch.r(_V([-50.0])), jnp.array([0.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.r(_V([-120.0])), jnp.array([1.0]), atol=1e-6))
        inside = ch.r(_V([-80.0]))
        expected = jnp.array([ch.rA * -80.0 + ch.rB])
        self.assertTrue(u.math.allclose(inside, expected, atol=1e-6))


class HCNZH19IOTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN_ZH2019_IO.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_q_inf(self) -> None:
        ch = HCN_ZH2019_IO(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN_ZH2019_IO(size=1, E=-43.0 * u.mV)
        V = _V([-65.0])
        ch.init_state(V)
        ch.q.value = jnp.array([0.25])
        i = ch.current(V)
        expected = ch.g_max * ch.q.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
