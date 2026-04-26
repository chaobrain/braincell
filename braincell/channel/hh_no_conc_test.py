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
from braincell.channel.hh_no_conc import (
    HCN1_MA2024_PC,
    HCN1_MA2020_GoC,
    HCN1_MA2025_BC,
    HCN2_MA2020_GoC,
    HCN1_RI2021_SC,
    HCN_ZH2019_IO,
    HCN_SU2015_DCN,
    Ca_ZH2019_IO,
    CaHVA_MA2020_GoC,
    CaHVA_MA2020_GrC,
    Cav2p3_MA2020_GoC,
    Kdr_ZH2019_IO,
    KM_RI2021_SC,
    KM_MA2020_GoC,
    KM_MA2020_GrC,
    Kir2p3_MA2025_BC,
    Kir2p3_MA2024_PC,
    Kir2p3_MA2020_GrC,
    Kir2p3_RI2021_SC,
    NaF_SU2015_DCN,
    NaP_SU2015_DCN,
    Na_ZH2019_IO,
    Kv1p1_MA2025_BC,
    Kv1p1_MA2020_GoC,
    Kv1p1_MA2020_GrC,
    Kv1p1_MA2024_PC,
    Kv1p1_RI2021_SC,
    Kv2p2_0010_MA2020_GrC,
    Kv3p4_MA2025_BC,
    Kv3p4_MA2020_GoC,
    Kv3p4_MA2020_GrC,
    Kv3p4_MA2024_PC,
    Kv3p4_RI2021_SC,
    Kv4p3_MA2025_BC,
    Kv4p3_MA2020_GoC,
    Kv4p3_MA2020_GrC,
    Kv4p3_MA2024_PC,
    Kv4p3_RI2021_SC,
    fKdr_SU2015_DCN,
    sKdr_SU2015_DCN,
    _linoid_stable,
)
from braincell.ion import Calcium, Potassium, Sodium


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 2.5) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
        valence=1,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


def _na_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 140.0) * u.mM,
        E=jnp.full((size,), 71.0) * u.mV,
        valence=1,
    )


def _ca_info(size: int = 1, e_mV: float = 129.33) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 2.0) * u.mM,
        E=jnp.full((size,), e_mV) * u.mV,
        valence=2,
    )


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


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
        self.assertTrue(u.math.allclose(HCN_SU2015_DCN(size=1).f_m_tau(V), jnp.array(400.0), atol=1e-6))


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


class NaFSU15DCNTest(unittest.TestCase):
    def test_root_type_is_sodium(self) -> None:
        self.assertIs(NaF_SU2015_DCN.root_type, Sodium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = NaF_SU2015_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, na), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, na), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = NaF_SU2015_DCN(size=1)
        V = _V([-40.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        i = ch.current(V, na)
        expected = ch.g_max * (ch.m.value ** 3) * ch.h.value * (na.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_tau_matches_mod_formula(self) -> None:
        ch = NaF_SU2015_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        v = V.to_decimal(u.mV)
        expected_m_tau = 5.83 / (jnp.exp((v - 6.4) / -9.0) + jnp.exp((v + 97.0) / 17.0)) + 0.025
        expected_h_tau = 16.67 / (jnp.exp((v - 8.3) / -29.0) + jnp.exp((v + 66.0) / 9.0)) + 0.2
        self.assertTrue(u.math.allclose(ch.f_m_tau(V, na), expected_m_tau, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.f_h_tau(V, na), expected_h_tau, atol=1e-6))


class NaPSU15DCNTest(unittest.TestCase):
    def test_root_type_is_sodium(self) -> None:
        self.assertIs(NaP_SU2015_DCN.root_type, Sodium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = NaP_SU2015_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, na), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, na), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = NaP_SU2015_DCN(size=1)
        V = _V([-50.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        i = ch.current(V, na)
        expected = ch.g_max * (ch.m.value ** 3) * ch.h.value * (na.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_tau_matches_mod_formula(self) -> None:
        ch = NaP_SU2015_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        v = V.to_decimal(u.mV)
        expected_h_tau = 1750.0 / (1.0 + jnp.exp((v + 65.0) / -8.0)) + 250.0
        self.assertTrue(u.math.allclose(ch.f_m_tau(V, na), jnp.array(50.0), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.f_h_tau(V, na), expected_h_tau, atol=1e-6))


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


class NaZH19IOTest(unittest.TestCase):
    def test_root_type_is_sodium(self) -> None:
        self.assertIs(Na_ZH2019_IO.root_type, Sodium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = Na_ZH2019_IO(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, na), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, na), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Na_ZH2019_IO(size=1)
        V = _V([-40.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        i = ch.current(V, na)
        expected = ch.g_max * (ch.m.value ** 3) * ch.h.value * (na.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_small_denominator_branches_are_stable(self) -> None:
        ch = Na_ZH2019_IO(size=1)
        self.assertTrue(u.math.allclose(ch._m_alpha(_V([-41.0])), jnp.array([1.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(ch._h_beta(_V([-50.0])), jnp.array([10.0]), atol=1e-6))


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
