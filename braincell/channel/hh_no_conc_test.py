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
    HCN1_MA24_PC,
    HCN1_MA20_GoC,
    HCN1_MA25_BC,
    HCN2_MA20_GoC,
    HCN1_RI21_SC,
    HCN_ZH19_IO,
    HCN_SU15_DCN,
    Ca_ZH19_IO,
    CaHVA_MA20_GoC,
    CaHVA_MA20_GrC,
    Cav2p3_MA20_GoC,
    Kdr_ZH19_IO,
    KM_RI21_SC,
    KM_MA20_GoC,
    KM_MA20_GrC,
    Kir2p3_MA25_BC,
    Kir2p3_MA24_PC,
    Kir2p3_MA20_GrC,
    Kir2p3_RI21_SC,
    NaF_SU15_DCN,
    NaP_SU15_DCN,
    Na_ZH19_IO,
    Kv1p1_MA25_BC,
    Kv1p1_MA20_GoC,
    Kv1p1_MA20_GrC,
    Kv1p1_MA24_PC,
    Kv1p1_RI21_SC,
    Kv2p2_0010_MA20_GrC,
    Kv3p4_MA25_BC,
    Kv3p4_MA20_GoC,
    Kv3p4_MA20_GrC,
    Kv3p4_MA24_PC,
    Kv3p4_RI21_SC,
    Kv4p3_MA25_BC,
    Kv4p3_MA20_GoC,
    Kv4p3_MA20_GrC,
    Kv4p3_MA24_PC,
    Kv4p3_RI21_SC,
    fKdr_SU15_DCN,
    sKdr_SU15_DCN,
    _linoid_stable,
)
from braincell.channel.calcium import ICaGrc_Ma2020
from braincell.channel.calcium import ICav23_Ma2020
from braincell.channel.hyperpolarization_activated import Ih1_Ma2020, Ih2_Ma2020
from braincell.channel.potassium import IKM_Grc_Ma2020, IKv11_Ak2007, IKv34_Ma2020, IKv43_Ma2020
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
        self.assertIs(HCN1_MA25_BC.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = HCN1_MA25_BC(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN1_MA25_BC(size=1, E=-34.4 * u.mV)
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
        self.assertIs(HCN1_MA24_PC.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = HCN1_MA24_PC(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN1_MA24_PC(size=1, E=-34.4 * u.mV)
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
        bc = HCN1_MA25_BC(size=1, temp=temp)
        pc = HCN1_MA24_PC(size=1, temp=temp)
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
        self.assertIs(Kir2p3_MA25_BC.root_type, Potassium)

    def test_reset_state_matches_alpha_beta_ratio(self) -> None:
        ch = Kir2p3_MA25_BC(size=1)
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
    def test_matches_legacy_without_gating_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = Kv1p1_MA25_BC(size=1, temp=temp, gateCurrent=0.0)
        legacy = IKv11_Ak2007(size=1, T=temp, gateCurrent=0.0)
        V = _V([-60.0])
        k = _k_info()

        proto.init_state(V, k)
        legacy.init_state(V, k)
        proto.reset_state(V, k)
        legacy.reset_state(V, k)

        self.assertTrue(u.math.allclose(proto.n.value, legacy.p.value, atol=1e-6))

        proto.compute_derivative(V, k)
        legacy.compute_derivative(V, k)
        self.assertTrue(
            u.math.allclose(proto.n.derivative, legacy.p.derivative, atol=1e-6 * u.Hz)
        )

        i_proto = proto.current(V, k)
        i_legacy = legacy.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_legacy_with_gating_current_enabled(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = Kv1p1_MA25_BC(size=1, temp=temp, gateCurrent=1.0)
        legacy = IKv11_Ak2007(size=1, T=temp, gateCurrent=1.0)
        V = _V([-50.0])
        k = _k_info()

        proto.init_state(V, k)
        legacy.init_state(V, k)
        proto.reset_state(V, k)
        legacy.reset_state(V, k)

        i_proto = proto.current(V, k)
        i_legacy = legacy.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv3p4MA25BCTest(unittest.TestCase):
    def test_matches_legacy_state_and_current(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = Kv3p4_MA25_BC(size=1, temp=temp)
        legacy = IKv34_Ma2020(size=1, T=temp)
        V = _V([-45.0])
        k = _k_info()

        proto.init_state(V, k)
        legacy.init_state(V, k)
        proto.reset_state(V, k)
        legacy.reset_state(V, k)

        self.assertTrue(u.math.allclose(proto.m.value, legacy.p.value, atol=1e-6))
        self.assertTrue(u.math.allclose(proto.h.value, legacy.q.value, atol=1e-6))

        proto.compute_derivative(V, k)
        legacy.compute_derivative(V, k)
        self.assertTrue(
            u.math.allclose(proto.m.derivative, legacy.p.derivative, atol=1e-6 * u.Hz)
        )
        self.assertTrue(
            u.math.allclose(proto.h.derivative, legacy.q.derivative, atol=1e-6 * u.Hz)
        )

        i_proto = proto.current(V, k)
        i_legacy = legacy.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kv4p3MA25BCTest(unittest.TestCase):
    def test_matches_legacy_state_and_current(self) -> None:
        temp = u.celsius2kelvin(30.0)
        proto = Kv4p3_MA25_BC(size=1, temp=temp)
        legacy = IKv43_Ma2020(size=1, T=temp)
        V = _V([-55.0])
        k = _k_info()

        proto.init_state(V, k)
        legacy.init_state(V, k)
        proto.reset_state(V, k)
        legacy.reset_state(V, k)

        self.assertTrue(u.math.allclose(proto.a.value, legacy.p.value, atol=1e-6))
        self.assertTrue(u.math.allclose(proto.b.value, legacy.q.value, atol=1e-6))

        proto.compute_derivative(V, k)
        legacy.compute_derivative(V, k)
        self.assertTrue(
            u.math.allclose(proto.a.derivative, legacy.p.derivative, atol=1e-6 * u.Hz)
        )
        self.assertTrue(
            u.math.allclose(proto.b.derivative, legacy.q.derivative, atol=1e-6 * u.Hz)
        )

        i_proto = proto.current(V, k)
        i_legacy = legacy.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
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
        bc = Kir2p3_MA25_BC(size=1, temp=temp)
        pc = Kir2p3_MA24_PC(size=1, temp=temp)
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
        bc = Kv1p1_MA25_BC(size=1, temp=temp, gateCurrent=0.0)
        pc = Kv1p1_MA24_PC(size=1, temp=temp, gateCurrent=0.0)
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
        bc = Kv1p1_MA25_BC(size=1, temp=temp, gateCurrent=1.0)
        pc = Kv1p1_MA24_PC(size=1, temp=temp, gateCurrent=1.0)
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
        bc = Kv3p4_MA25_BC(size=1, temp=temp)
        pc = Kv3p4_MA24_PC(size=1, temp=temp)
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
        bc = Kv4p3_MA25_BC(size=1, temp=temp)
        pc = Kv4p3_MA24_PC(size=1, temp=temp)
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
        self.assertIs(HCN1_RI21_SC.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = HCN1_RI21_SC(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN1_RI21_SC(size=1, E=-34.4 * u.mV)
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
        bc = HCN1_MA25_BC(size=1, temp=temp)
        sc = HCN1_RI21_SC(size=1, temp=temp)
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
        self.assertIs(HCN_SU15_DCN.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_m_inf(self) -> None:
        ch = HCN_SU15_DCN(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN_SU15_DCN(size=1, E=-45.0 * u.mV)
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
        self.assertTrue(u.math.allclose(HCN_SU15_DCN(size=1).f_m_tau(V), jnp.array(400.0), atol=1e-6))


class KMRI21SCTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(KM_RI21_SC.root_type, Potassium)

    def test_reset_state_matches_f_n_inf(self) -> None:
        ch = KM_RI21_SC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = KM_RI21_SC(size=1)
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

    def test_matches_legacy_gate_dynamics(self) -> None:
        temp = u.celsius2kelvin(30.0)
        proto = KM_RI21_SC(size=1, temp=temp)
        legacy = IKM_Grc_Ma2020(size=1, T=temp)
        V = _V([-60.0])
        k = _k_info()

        self.assertTrue(u.math.allclose(proto.f_n_inf(V, k), legacy.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(proto.f_n_tau(V, k), legacy.f_p_tau(V), atol=1e-6))

        proto.init_state(V, k)
        legacy.init_state(V, k)
        proto.reset_state(V, k)
        legacy.reset_state(V, k)
        self.assertTrue(u.math.allclose(proto.n.value, legacy.p.value, atol=1e-6))

        proto.compute_derivative(V, k)
        legacy.compute_derivative(V, k)
        self.assertTrue(
            u.math.allclose(proto.n.derivative, legacy.p.derivative, atol=1e-6 * u.Hz)
        )


class NaFSU15DCNTest(unittest.TestCase):
    def test_root_type_is_sodium(self) -> None:
        self.assertIs(NaF_SU15_DCN.root_type, Sodium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = NaF_SU15_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, na), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, na), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = NaF_SU15_DCN(size=1)
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
        ch = NaF_SU15_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        v = V.to_decimal(u.mV)
        expected_m_tau = 5.83 / (jnp.exp((v - 6.4) / -9.0) + jnp.exp((v + 97.0) / 17.0)) + 0.025
        expected_h_tau = 16.67 / (jnp.exp((v - 8.3) / -29.0) + jnp.exp((v + 66.0) / 9.0)) + 0.2
        self.assertTrue(u.math.allclose(ch.f_m_tau(V, na), expected_m_tau, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.f_h_tau(V, na), expected_h_tau, atol=1e-6))


class NaPSU15DCNTest(unittest.TestCase):
    def test_root_type_is_sodium(self) -> None:
        self.assertIs(NaP_SU15_DCN.root_type, Sodium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = NaP_SU15_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, na), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, na), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = NaP_SU15_DCN(size=1)
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
        ch = NaP_SU15_DCN(size=1)
        V = _V([-60.0])
        na = _na_info()
        v = V.to_decimal(u.mV)
        expected_h_tau = 1750.0 / (1.0 + jnp.exp((v + 65.0) / -8.0)) + 250.0
        self.assertTrue(u.math.allclose(ch.f_m_tau(V, na), jnp.array(50.0), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.f_h_tau(V, na), expected_h_tau, atol=1e-6))


class Kir2p3RI21SCTest(unittest.TestCase):
    def test_matches_bc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        bc = Kir2p3_MA25_BC(size=1, temp=temp)
        sc = Kir2p3_RI21_SC(size=1, temp=temp)
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
        self.assertIs(fKdr_SU15_DCN.root_type, Potassium)

    def test_reset_state_matches_f_m_inf(self) -> None:
        ch = fKdr_SU15_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = fKdr_SU15_DCN(size=1)
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
        ch = fKdr_SU15_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        expected = 13.9 / (jnp.exp((v + 40.0) / 12.0) + jnp.exp((v + 40.0) / -13.0)) + 0.1
        self.assertTrue(u.math.allclose(ch.f_m_tau(V, k), expected, atol=1e-6))


class sKdrSU15DCNTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(sKdr_SU15_DCN.root_type, Potassium)

    def test_reset_state_matches_f_m_inf(self) -> None:
        ch = sKdr_SU15_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = sKdr_SU15_DCN(size=1)
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
        slow = sKdr_SU15_DCN(size=1)
        fast = fKdr_SU15_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        v = V.to_decimal(u.mV)
        expected = 14.95 / (jnp.exp((v + 50.0) / 21.74) + jnp.exp((v + 50.0) / -13.91)) + 0.05
        self.assertTrue(u.math.allclose(slow.f_m_tau(V, k), expected, atol=1e-6))
        self.assertFalse(bool(jnp.allclose(slow.f_m_tau(V, k), fast.f_m_tau(V, k), atol=1e-6)))


class HCN1MA20GoCTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN1_MA20_GoC.root_type, HHTypedNeuron)

    def test_fast_and_slow_components_match_existing_model(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = HCN1_MA20_GoC(size=1, temp=temp)
        legacy = Ih1_Ma2020(size=1, T=temp)
        V = _V([-80.0])

        proto.init_state(V)
        legacy.init_state(V)
        proto.reset_state(V)
        legacy.reset_state(V)
        self.assertTrue(u.math.allclose(proto.o_fast.value, legacy.p.value, atol=1e-6))
        self.assertTrue(u.math.allclose(proto.o_slow.value, legacy.q.value, atol=1e-6))
        self.assertTrue(
            u.math.allclose(
                proto.f_o_fast_inf(V) + proto.f_o_slow_inf(V),
                proto.o_inf(V),
                atol=1e-6,
            )
        )

        proto.compute_derivative(V)
        legacy.compute_derivative(V)
        self.assertTrue(u.math.allclose(proto.o_fast.derivative, legacy.p.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(proto.o_slow.derivative, legacy.q.derivative, atol=1e-6 * u.Hz))

        i_proto = proto.current(V)
        i_legacy = legacy.current(V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class HCN2MA20GoCTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(HCN2_MA20_GoC.root_type, HHTypedNeuron)

    def test_matches_existing_model(self) -> None:
        temp = u.celsius2kelvin(22.0)
        proto = HCN2_MA20_GoC(size=1, temp=temp)
        legacy = Ih2_Ma2020(size=1, T=temp)
        V = _V([-80.0])

        proto.init_state(V)
        legacy.init_state(V)
        proto.reset_state(V)
        legacy.reset_state(V)
        self.assertTrue(u.math.allclose(proto.o_fast.value, legacy.p.value, atol=1e-6))
        self.assertTrue(u.math.allclose(proto.o_slow.value, legacy.q.value, atol=1e-6))

        proto.compute_derivative(V)
        legacy.compute_derivative(V)
        self.assertTrue(u.math.allclose(proto.o_fast.derivative, legacy.p.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(proto.o_slow.derivative, legacy.q.derivative, atol=1e-6 * u.Hz))

        i_proto = proto.current(V)
        i_legacy = legacy.current(V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_r_is_clamped_to_corridor(self) -> None:
        ch = HCN2_MA20_GoC(size=1)
        self.assertTrue(u.math.allclose(ch.r(_V([-50.0])), jnp.array([0.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.r(_V([-120.0])), jnp.array([1.0]), atol=1e-6))
        inside = ch.r(_V([-80.0]))
        expected = jnp.array([ch.rA * -80.0 + ch.rB])
        self.assertTrue(u.math.allclose(inside, expected, atol=1e-6))


class CaHVAMA20GoCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(CaHVA_MA20_GoC.root_type, Calcium)

    def test_current_uses_ca_reversal(self) -> None:
        ch = CaHVA_MA20_GoC(size=1)
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

    def test_matches_existing_model_at_same_reversal(self) -> None:
        temp = u.celsius2kelvin(30.0)
        proto = CaHVA_MA20_GoC(size=1, temp=temp)
        legacy = ICaGrc_Ma2020(size=1, T=temp)
        V = _V([-30.0])
        ca = _ca_info()

        proto.init_state(V, ca)
        legacy.init_state(V, ca)
        proto.reset_state(V, ca)
        legacy.reset_state(V, ca)
        self.assertTrue(u.math.allclose(proto.s.value, legacy.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(proto.u.value, legacy.h.value, atol=1e-6))

        proto.compute_derivative(V, ca)
        legacy.compute_derivative(V, ca)
        self.assertTrue(u.math.allclose(proto.s.derivative, legacy.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(proto.u.derivative, legacy.h.derivative, atol=1e-6 * u.Hz))

        i_proto = proto.current(V, ca)
        i_legacy = legacy.current(V, ca)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Cav2p3MA20GoCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav2p3_MA20_GoC.root_type, Calcium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = Cav2p3_MA20_GoC(size=1)
        V = _V([-60.0])
        ca = _ca_info(e_mV=140.0)
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, ca), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Cav2p3_MA20_GoC(size=1, g_max=0.1 * (u.mS / u.cm ** 2))
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

    def test_matches_existing_model_at_same_reversal(self) -> None:
        temp = u.celsius2kelvin(34.0)
        proto = Cav2p3_MA20_GoC(size=1, temp=temp)
        legacy = ICav23_Ma2020(size=1, T=temp)
        V = _V([-40.0])
        ca = _ca_info(e_mV=140.0)

        proto.init_state(V, ca)
        legacy.init_state(V, ca)
        proto.reset_state(V, ca)
        legacy.reset_state(V, ca)
        self.assertTrue(u.math.allclose(proto.m.value, legacy.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(proto.h.value, legacy.h.value, atol=1e-6))

        proto.compute_derivative(V, ca)
        legacy.compute_derivative(V, ca)
        self.assertTrue(u.math.allclose(proto.m.derivative, legacy.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(proto.h.derivative, legacy.h.derivative, atol=1e-6 * u.Hz))

        i_proto = proto.current(V, ca)
        i_legacy = legacy.current(V, ca)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class KMMA20GoCTest(unittest.TestCase):
    def test_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        sc = KM_RI21_SC(size=1, temp=temp)
        goc = KM_MA20_GoC(size=1, temp=temp)
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
        sc = Kv1p1_RI21_SC(size=1, temp=temp, gateCurrent=0.0)
        goc = Kv1p1_MA20_GoC(size=1, temp=temp, gateCurrent=0.0)
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
        sc = Kv1p1_RI21_SC(size=1, temp=temp, gateCurrent=1.0)
        goc = Kv1p1_MA20_GoC(size=1, temp=temp, gateCurrent=1.0)
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
        sc = Kv3p4_RI21_SC(size=1, temp=temp)
        goc = Kv3p4_MA20_GoC(size=1, temp=temp)
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
        sc = Kv4p3_RI21_SC(size=1, temp=temp)
        goc = Kv4p3_MA20_GoC(size=1)
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
        goc = KM_MA20_GoC(size=1, temp=temp)
        grc = KM_MA20_GrC(size=1, temp=temp)
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
        sc = Kir2p3_RI21_SC(size=1, temp=temp)
        grc = Kir2p3_MA20_GrC(size=1, temp=temp)
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
        goc = Kv1p1_MA20_GoC(size=1, temp=temp, gateCurrent=0.0)
        grc = Kv1p1_MA20_GrC(size=1, temp=temp, gateCurrent=0.0)
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
        goc = Kv1p1_MA20_GoC(size=1, temp=temp, gateCurrent=1.0)
        grc = Kv1p1_MA20_GrC(size=1, temp=temp, gateCurrent=1.0)
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
        self.assertIs(Kv2p2_0010_MA20_GrC.root_type, Potassium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = Kv2p2_0010_MA20_GrC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, k), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Kv2p2_0010_MA20_GrC(size=1)
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
        ch = Kv2p2_0010_MA20_GrC(size=1)
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
        goc = Kv3p4_MA20_GoC(size=1, temp=temp)
        grc = Kv3p4_MA20_GrC(size=1, temp=temp)
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
        goc = Kv4p3_MA20_GoC(size=1, temp=temp)
        grc = Kv4p3_MA20_GrC(size=1)
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
        self.assertIs(CaHVA_MA20_GrC.root_type, Calcium)

    def test_matches_goc_variant(self) -> None:
        temp = u.celsius2kelvin(30.0)
        goc = CaHVA_MA20_GoC(size=1, temp=temp)
        grc = CaHVA_MA20_GrC(size=1, temp=temp)
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
        bc = Kv1p1_MA25_BC(size=1, temp=temp, gateCurrent=0.0)
        sc = Kv1p1_RI21_SC(size=1, temp=temp, gateCurrent=0.0)
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
        bc = Kv1p1_MA25_BC(size=1, temp=temp, gateCurrent=1.0)
        sc = Kv1p1_RI21_SC(size=1, temp=temp, gateCurrent=1.0)
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
        bc = Kv3p4_MA25_BC(size=1, temp=temp)
        sc = Kv3p4_RI21_SC(size=1, temp=temp)
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
        bc = Kv4p3_MA25_BC(size=1, temp=temp)
        sc = Kv4p3_RI21_SC(size=1, temp=temp)
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
        self.assertIs(HCN_ZH19_IO.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_q_inf(self) -> None:
        ch = HCN_ZH19_IO(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = HCN_ZH19_IO(size=1, E=-43.0 * u.mV)
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
        self.assertIs(Na_ZH19_IO.root_type, Sodium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = Na_ZH19_IO(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, na), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, na), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Na_ZH19_IO(size=1)
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
        ch = Na_ZH19_IO(size=1)
        self.assertTrue(u.math.allclose(ch._m_alpha(_V([-41.0])), jnp.array([1.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(ch._h_beta(_V([-50.0])), jnp.array([10.0]), atol=1e-6))


class KdrZH19IOTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(Kdr_ZH19_IO.root_type, Potassium)

    def test_reset_state_matches_f_n_inf(self) -> None:
        ch = Kdr_ZH19_IO(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V, k), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Kdr_ZH19_IO(size=1)
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
        ch = Kdr_ZH19_IO(size=1)
        self.assertTrue(u.math.allclose(ch._n_alpha(_V([-41.0])), jnp.array([10.0]), atol=1e-6))


class CaZH19IOTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Ca_ZH19_IO.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = Ca_ZH19_IO(size=1)
        V = _V([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_uses_instantaneous_m_and_stateful_h(self) -> None:
        ch = Ca_ZH19_IO(size=1, E=120.0 * u.mV, mMidV=-61.0 * u.mV)
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
