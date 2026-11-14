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


import contextlib
import inspect
import math
import unittest
from unittest import mock

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base_neuron import HHTypedNeuron
from braincell.channel import calcium as calcium_module
from braincell.channel._base import GhkHH, HH, ghk_flux
from braincell.channel.calcium import (
    CaHVA_SU2015_DCN,
    CaL_SU2015_DCN,
    CaLVA_SU2015_DCN,
    CaHVA_MA2020_GoC,
    CaHVA_MA2020_GrC,
    Ca_ZH2019_IO,
    Ca_ZH2019_IO_Frozen,
    CaT_HM1992_Shifted25,
    CaT_HM1992_GHK,
    CaHT_Re1993,
    CaL_Li2017,
    CaN_IS2008,
    CaT_HM1992,
    CaT_HP1992,
    Cav3p1_MA2024_PC_Frozen,
    Cav3p1_MA2024_PC,
    Cav3p1Test_PC24,
    Cav3p1_MA2020_GoC_Frozen,
    Cav2p1_MA2024_PC_Frozen,
    Cav2p1_MA2025_BC_Frozen,
    Cav2p1_MA2024_PC,
    Cav2p1_MA2025_BC,
    Cav2p1_RI2021_SC,
    Cav2p1_RI2021_SC_Frozen,
    Cav3p2_MA2024_PC,
    Cav3p2_MA2025_BC,
    Cav3p2_RI2021_SC,
    Cav3p3_MA2024_PC_Frozen,
    Cav3p3_MA2024_PC,
    Cav3p3_RI2021_SC,
    Cav2p3_MA2020_GoC,
    Cav1p2_MA2020_GoC,
    Cav1p2_MA2025_BC,
    Cav1p3_MA2020_GoC,
    Cav1p3_MA2025_BC,
    Cav3p1_MA2020_GoC,
)
from braincell.ion import Calcium
from braincell.mech import get_registry
from braincell.channel._testing import (
    DENSITY_UNIT,
    assert_channels_agree,
    assert_ghk_near_zero,
    ca_info,
    voltage,
)


def _cav3p1_nmodl_ghk_flux(V, ci, co, z, temp):
    faraday = 9.6485e4 * (u.coulomb / u.mol)
    gas = 8.3145 * (u.joule / (u.kelvin * u.mol))
    zeta = (z * faraday * V) / (gas * (temp + 0.04 * u.kelvin))
    exp_term = u.math.exp(-zeta)
    numerator = ci - co * exp_term
    small_branch = (z * faraday) * numerator * (1 + zeta / 2)
    regular_branch = (z * zeta * faraday) * numerator / (1 - exp_term)
    return u.math.where(u.math.abs(1 - exp_term) < 1e-6, small_branch, regular_branch)


#: ``(V in mV, flux in C/m^3, d/dV in C/m^3 per mV)`` of the Cav3.1 and Cav3.3 NMODL GHK
#: helpers at ci = 1e-4 mM, co = 2 mM, z = 2, T = 309.15 K, from mpmath at 50 digits
#: with each .mod's F, R and Kelvin offset (MECH-43).
_CAV3P1_GHK_NEAR_ZERO = (
    (0, -385920.703, 14485.708147469784186),
    (1e-9, -385920.70298551429185, 14485.708147107371605),
    (-1e-9, -385920.70301448570815, 14485.708147832196768),
    (1e-6, -385920.68851429203374, 14485.707785057202273),
    (-1e-6, -385920.71748570832868, 14485.7085098823661),
    (2e-5, -385920.41328590953312, 14485.700899218145924),
    (-2e-5, -385920.99271423543191, 14485.715395721422449),
    (1e-3, -385906.21747305882116, 14485.34573488793908),
    (-1e-3, -385935.18888935376072, 14486.070560051629293),
)
_CAV3P3_GHK_NEAR_ZERO = (
    (0, -386060.696, 14500.48245833483132),
    (1e-9, -386060.69598549951754, 14500.482457971810781),
    (-1e-9, -386060.69601450048246, 14500.482458697851859),
    (1e-6, -386060.68149951772318, 14500.482095314292306),
    (-1e-6, -386060.71050048263985, 14500.482821355370334),
    (2e-5, -386060.40599042343741, 14500.475197924051038),
    (-2e-5, -386060.9860097217708, 14500.489718745611602),
    (1e-3, -386046.19569905193466, 14500.119437795885461),
    (-1e-3, -386075.19666396860432, 14500.84547887377718),
)


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
        self.assertTrue(u.math.allclose(ch.gate_phi(gates["p"]), 3.0 * jnp.ones(1), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.gate_phi(gates["q"]), 2.0 * jnp.ones(1), atol=1e-6))

    def test_init_state_creates_p_and_q(self) -> None:
        ch = self._make(size=2)
        V = voltage([-60.0, -70.0])
        ca = ca_info(2)
        ch.init_state(V, ca)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_matches_gate_form(self) -> None:
        ch = self._make(size=1)
        V = voltage([-60.0])
        ca = ca_info()
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
        V = voltage([-60.0])
        ca = ca_info()
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
        V = voltage([-60.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        current = ch.current(V, ca)
        expected = ch.g_max * ch.p.value**2 * ch.q.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_current_is_zero_when_gates_closed(self) -> None:
        ch = self._make(size=1)
        V = voltage([-60.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.zeros(1)
        ch.q.value = jnp.zeros(1)
        current = ch.current(V, ca)
        self.assertTrue(u.math.allclose(current.to_decimal(DENSITY_UNIT), jnp.zeros(1), atol=1e-9))


class CaT_HM1992Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaT_HM1992


class CaT_HP1992Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaT_HP1992


class CaT_HM1992_Shifted25Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaT_HM1992_Shifted25

    def test_is_cat_hm1992_with_gates_shifted_by_25_mv(self) -> None:
        """MECH-42: the renamed class is ``CaT_HM1992`` at ``V_sh = +25 mV``, element for element.

        Before braincell 0.3 the class was ``CaHT_HM1992``; the rename kept
        every rate function and default, so the gates, their derivatives and
        the current agree exactly in float64, not merely within a tolerance.
        """
        with brainstate.environ.context(precision=64):
            V = voltage(np.linspace(-120.0, 60.0, 37))
            ca = ca_info(37)
            shifted = CaT_HM1992_Shifted25(37)
            reference = CaT_HM1992(37, V_sh=25.0 * u.mV)
            for ch in (shifted, reference):
                ch.init_state(V, ca)
                ch.reset_state(V, ca)
                ch.compute_derivative(V, ca)
            self.assertEqual(shifted.p.value.dtype, jnp.float64)
            for rate in ("f_p_inf", "f_p_tau", "f_q_inf", "f_q_tau"):
                np.testing.assert_array_equal(getattr(shifted, rate)(V, ca), getattr(reference, rate)(V, ca), rate)
            for name in ("p", "q"):
                np.testing.assert_array_equal(getattr(shifted, name).value, getattr(reference, name).value, name)
                np.testing.assert_array_equal(
                    u.get_magnitude(getattr(shifted, name).derivative),
                    u.get_magnitude(getattr(reference, name).derivative),
                    name,
                )
            np.testing.assert_array_equal(
                shifted.current(V, ca).to_decimal(DENSITY_UNIT),
                reference.current(V, ca).to_decimal(DENSITY_UNIT),
            )

    def test_defaults_other_than_v_sh_are_cat_hm1992s(self) -> None:
        # ``__init__`` is wrapped by the unit check; read the signature it wraps.
        shifted = inspect.signature(inspect.unwrap(CaT_HM1992_Shifted25.__init__)).parameters
        reference = inspect.signature(inspect.unwrap(CaT_HM1992.__init__)).parameters
        self.assertEqual(list(shifted), list(reference))
        differing = [name for name in shifted if not _same_default(shifted[name].default, reference[name].default)]
        self.assertEqual(differing, ["V_sh"])
        self.assertTrue(u.math.allclose(shifted["V_sh"].default, 25.0 * u.mV))

    def test_registered_under_the_new_name_only(self) -> None:
        registry = get_registry()
        self.assertIs(registry.get("channel", "CaT_HM1992_Shifted25"), CaT_HM1992_Shifted25)
        self.assertEqual(registry.key_for("channel", CaT_HM1992_Shifted25), "CaT_HM1992_Shifted25")
        self.assertFalse(registry.contains("channel", "CaHT_HM1992"))
        with self.assertRaises(KeyError) as caught:
            registry.get("channel", "CaHT_HM1992")
        message = caught.exception.args[0]
        self.assertIn("'CaT_HM1992_Shifted25'", message)
        self.assertIn("braincell 0.3", message)


def _same_default(a, b) -> bool:
    if isinstance(a, u.Quantity) or isinstance(b, u.Quantity):
        return bool(u.math.allclose(a, b)) if u.get_unit(a) == u.get_unit(b) else False
    return a == b


class CaHT_Re1993Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaHT_Re1993


#: ``(V mV, p_inf, tau_p ms, q_inf, tau_q ms, I)`` of ``ICaL_TC.cpp`` (ModelDB 233509,
#: Li et al. 2017) evaluated from its C++ text in float64, with ``I = g p^2 q (E - V)``
#: at the steady-state gates, ``g = 1 mS/cm2`` and ``E = 120 mV``, in mS/cm2 * mV (MECH-49).
_ICAL_TC_REFERENCE = (
    (-100.0, 1.6918979223288784e-10, 0.401243268560069, 1.0, 300.1807532287766, 6.297540875077706e-18),
    (-80.0, 2.5109990926928157e-08, 0.4047163487774566, 0.99999999999886, 301.48352900408383, 1.2610232886993912e-13),
    (-60.0, 3.726639284186561e-06, 0.4178813896879442, 0.999999974890009, 312.00325047314135, 2.4998112010294158e-09),
    (-40.0, 0.0005527786369235996, 0.46724800666791844, 0.9994472213630764, 350.0, 4.886324993044687e-05),
    (-20.0, 0.07585818002124355, 0.6268189957823599, 0.07585818002124355, 312.00325047314135, 0.06111323768176895),
    (0.0, 0.9241418199787566, 0.7314168388172242, 3.726639284186561e-06, 301.48352900408383, 0.0003819230335737624),
    (20.0, 0.9994472213630764, 0.5276588220188739, 1.6918979223288784e-10, 300.1807532287766, 1.6900279492577123e-08),
    (40.0, 0.9999962733607158, 0.43476477459680163, 7.681204685202035e-15, 300.0220182748802, 6.144917948120363e-13),
)


class CaL_Li2017Test(_P2QHHMixin, unittest.TestCase):
    CLS = CaL_Li2017

    def test_matches_ical_tc_of_li_2017(self) -> None:
        """MECH-49: the class renamed from ``CaL_IS2008`` is ``ICaL_TC.cpp``, element for element."""
        V_mV, p_inf, tau_p, q_inf, tau_q, current = (np.array(column) for column in zip(*_ICAL_TC_REFERENCE))
        with brainstate.environ.context(precision=64):
            V = voltage(V_mV)
            ca = ca_info(V_mV.size, Ci=5e-5, E=120.0)
            ch = CaL_Li2017(V_mV.size)
            ch.init_state(V, ca)
            ch.reset_state(V, ca)
            self.assertEqual(ch.p.value.dtype, jnp.float64)
            for rate, expected in (("f_p_inf", p_inf), ("f_p_tau", tau_p), ("f_q_inf", q_inf), ("f_q_tau", tau_q)):
                np.testing.assert_allclose(getattr(ch, rate)(V, ca), expected, rtol=1e-12, atol=0, err_msg=rate)
            np.testing.assert_allclose(ch.current(V, ca).to_decimal(DENSITY_UNIT), current, rtol=1e-12, atol=0)
            gates = {gate.name: gate for gate in ch._iter_gates()}
            np.testing.assert_allclose(ch.gate_phi(gates["p"]), 3.55**1.2, rtol=1e-12)
            np.testing.assert_allclose(ch.gate_phi(gates["q"]), 3.0**1.2, rtol=1e-12)

    def test_signature_and_defaults_are_unchanged_by_the_rename(self) -> None:
        params = inspect.signature(inspect.unwrap(CaL_Li2017.__init__)).parameters
        self.assertEqual(
            list(params),
            ["self", "size", "g_max", "temp", "q10_p", "temp_ref_p", "q10_q", "temp_ref_q", "V_sh", "name"],
        )
        expected = {
            "g_max": 1.0 * (u.mS / u.cm**2),
            "temp": u.celsius2kelvin(36.0),
            "q10_p": 3.55,
            "temp_ref_p": u.celsius2kelvin(24.0),
            "q10_q": 3.0,
            "temp_ref_q": u.celsius2kelvin(24.0),
            "V_sh": 0.0 * u.mV,
            "name": None,
        }
        for name, value in expected.items():
            self.assertTrue(_same_default(params[name].default, value), name)

    def test_registered_under_the_new_name_only(self) -> None:
        registry = get_registry()
        self.assertIs(registry.get("channel", "CaL_Li2017"), CaL_Li2017)
        self.assertEqual(registry.key_for("channel", CaL_Li2017), "CaL_Li2017")
        self.assertFalse(registry.contains("channel", "CaL_IS2008"))
        with self.assertRaises(KeyError) as caught:
            registry.get("channel", "CaL_IS2008")
        self.assertIn("'CaL_Li2017'", caught.exception.args[0])
        self.assertIn("braincell 0.3", caught.exception.args[0])


#: ``(V mV, p_inf, tau_p ms, q_inf, tau_q ms)`` of ``ITGHK.mod`` (ModelDB 279) evaluated
#: from its text in float64 at ``shift = 2 mV``, ``actshift = 0``; tau before ``phi``. The
#: ``(v + shift) < -80`` branch of tau_h lies between -82.001 and -82 mV (MECH-49).
_ITGHK_GATES = (
    (-120.0, 5.334325475213075e-05, 2.904095260999236, 0.9999038975845005, 188.7154340014129),
    (-90.0, 0.0066928509242848554, 11.51272715070815, 0.8519528019683106, 296.0980622045103),
    (-82.001, 0.023896451284745414, 13.861867101091585, 0.43788503359106573, 333.8847240391913),
    (-82.0, 0.023900213736707834, 13.86203482789083, 0.43782349911420193, 278.5878415206564),
    (-80.0, 0.03270116507689669, 14.114244339793817, 0.320821300824607, 235.1272488898345),
    (-79.999, 0.03270626736449414, 14.114327607383062, 0.32076682951647456, 235.107523424501),
    (-60.0, 0.4597646070902748, 9.242937163960393, 0.0031726828424851893, 58.832564751815084),
    (-40.0, 0.9554051082112419, 3.7808974204835097, 2.144494842091395e-05, 32.58967641519969),
    (-20.0, 0.9981492698193938, 1.6789199931657381, 1.4449800373124837e-07, 28.683210422674946),
    (0.0, 0.9999263511981811, 0.9679077867135525, 9.736200303530205e-10, 28.10170139230423),
    (20.0, 0.9999970741877094, 0.7306141060994489, 6.560200168110743e-12, 28.015139074073435),
    (40.0, 0.9999998837754862, 0.6515270349986355, 4.420228103640978e-14, 28.002253573511712),
)
#: ``(V mV, ica mA/cm2)``: the mod file's ``pcabar m^2 h ghk(v, cai, cao)`` (outward positive)
#: at ``m = 0.6``, ``h = 0.3``, ``pcabar = 2e-4 cm/s``, ``cai = 2.4e-4 mM``, ``cao = 2 mM``,
#: 36 degC, with NEURON 9.0.2's ``FARADAY`` and ``R``.
_ITGHK_CURRENT = (
    (-60.0, -0.03797026385401351),
    (-40.0, -0.026341007284747493),
    (-20.0, -0.01610457671300051),
    (0.0, -0.008335332335530531),
    (20.0, -0.0035862868826235374),
    (40.0, -0.001304427623993546),
)
_CURRENT_DENSITY = u.mA / u.cm**2


def _naive_ghk(V, ci, co, z, temp):
    # The textbook zeta / (1 - exp(-zeta)) form without a limit at 0 mV: 0/0 there.
    zeta = (z * u.faraday_constant * V) / (u.gas_constant * temp)
    return z * u.faraday_constant * (ci - co * u.math.exp(-zeta)) * zeta / (1 - u.math.exp(-zeta))


class _NaiveGhkCaT(CaT_HM1992_GHK):
    ghk = staticmethod(_naive_ghk)


class CaT_HM1992_GHKTest(unittest.TestCase):
    """MECH-49: the T current of ``ITGHK.mod`` (ModelDB 279), GHK drive and ``q10_p = 5``."""

    def _channel(self, V, ca, cls=CaT_HM1992_GHK, **kwargs):
        ch = cls(V.shape[0], **kwargs)
        ch.init_state(V, ca)
        return ch

    def test_structure_and_defaults(self) -> None:
        self.assertIs(CaT_HM1992_GHK.root_type, Calcium)
        self.assertTrue(issubclass(CaT_HM1992_GHK, GhkHH))
        self.assertIs(get_registry().get("channel", "CaT_HM1992_GHK"), CaT_HM1992_GHK)
        ch = CaT_HM1992_GHK(1)
        gates = ch._iter_gates()
        self.assertEqual(tuple((gate.name, gate.power) for gate in gates), (("p", 2), ("q", 1)))
        params = inspect.signature(inspect.unwrap(CaT_HM1992_GHK.__init__)).parameters
        expected = {
            "perm": 2e-4 * (u.cm / u.second),
            "temp": u.celsius2kelvin(36.0),
            "q10_p": 5.0,
            "temp_ref_p": u.celsius2kelvin(24.0),
            "q10_q": 3.0,
            "temp_ref_q": u.celsius2kelvin(24.0),
            "V_sh": -2.0 * u.mV,
            "name": None,
        }
        self.assertEqual(list(params), ["self", "size", *expected])
        for name, value in expected.items():
            self.assertTrue(_same_default(params[name].default, value), name)
        self.assertFalse(hasattr(ch, "g_max"))

    def test_gates_match_itghk_mod(self) -> None:
        V_mV, p_inf, tau_p, q_inf, tau_q = (np.array(column) for column in zip(*_ITGHK_GATES))
        with brainstate.environ.context(precision=64):
            V = voltage(V_mV)
            ca = ca_info(V_mV.size, Ci=2.4e-4)
            ch = self._channel(V, ca)
            for rate, expected in (("f_p_inf", p_inf), ("f_p_tau", tau_p), ("f_q_inf", q_inf), ("f_q_tau", tau_q)):
                np.testing.assert_allclose(getattr(ch, rate)(V, ca), expected, rtol=1e-12, atol=0, err_msg=rate)
            gates = {gate.name: gate for gate in ch._iter_gates()}
            np.testing.assert_allclose(ch.gate_phi(gates["p"]), 6.898648307306074, rtol=1e-12)
            np.testing.assert_allclose(ch.gate_phi(gates["q"]), 3.7371928188465517, rtol=1e-12)
            # Counter-example: the ohmic class's q10_p of 3.55 is a different activation rate.
            old = self._channel(V, ca, q10_p=3.55)
            old_gates = {gate.name: gate for gate in old._iter_gates()}
            self.assertFalse(np.allclose(old.gate_phi(old_gates["p"]), 6.898648307306074, rtol=1e-3))

    def test_current_matches_itghk_mod(self) -> None:
        V_mV, ica = (np.array(column) for column in zip(*_ITGHK_CURRENT))
        with brainstate.environ.context(precision=64):
            V = voltage(V_mV)
            ca = ca_info(V_mV.size, Ci=2.4e-4, Co=2.0)
            ch = self._channel(V, ca)
            ch.p.value = jnp.full(V_mV.shape, 0.6)
            ch.q.value = jnp.full(V_mV.shape, 0.3)
            # BrainCell's current is inward-positive, the mod file's ica outward-positive.
            np.testing.assert_allclose(ch.current(V, ca).to_decimal(_CURRENT_DENSITY), -ica, rtol=1e-12, atol=0)

    def test_current_is_finite_and_smooth_through_0_mv(self) -> None:
        # D16: exprel, no branch. At 0 mV the flux is its limit z F (ci - co).
        points = np.array([0.0, 1e-9, -1e-9, 1e-6, -1e-6])
        ci, co = 2.4e-4, 2.0
        with brainstate.environ.context(precision=64):
            V = voltage(points)
            ca = ca_info(points.size, Ci=ci, Co=co)
            ch = self._channel(V, ca)
            ch.p.value = jnp.full(points.shape, 0.6)
            ch.q.value = jnp.full(points.shape, 0.3)
            current = ch.current(V, ca).to_decimal(_CURRENT_DENSITY)
            self.assertTrue(np.all(np.isfinite(current)))
            # -P p^2 q 2 F (ci - co): cm/s * C/mol * mM = 1e-3 mA/cm2.
            limit = -2e-4 * 0.36 * 0.3 * 2 * 96485.33212331001 * (ci - co) * 1e-3
            np.testing.assert_allclose(current[0], limit, rtol=1e-12)
            np.testing.assert_allclose(current, limit, rtol=1e-5)

            def at(v):
                return ch.current(jnp.full(points.shape, v) * u.mV, ca).to_decimal(_CURRENT_DENSITY)[0]

            slope = jax.grad(at)(0.0)
            self.assertTrue(np.isfinite(slope))
            self.assertLess(abs(slope - (at(1e-3) - at(-1e-3)) / 2e-3), 1e-6 * abs(slope))
            # Counter-example: without exprel the 0 mV point is 0/0.
            naive = self._channel(V, ca, cls=_NaiveGhkCaT)
            naive.p.value = ch.p.value
            naive.q.value = ch.q.value
            self.assertTrue(np.isnan(naive.current(V, ca).to_decimal(_CURRENT_DENSITY)[0]))


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
        V = voltage([-60.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))

    def test_compute_derivative_matches_inf_tau_form(self) -> None:
        ch = CaHVA_SU2015_DCN(size=1, qdeltat=2.0)
        V = voltage([-60.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.2])
        ch.compute_derivative(V, ca)
        gate = ch._iter_gates()[0]
        expected = ch.gate_phi(gate) * (ch.f_m_inf(V, ca) - ch.m.value) / ch.f_m_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_mod_formula(self) -> None:
        ch = CaHVA_SU2015_DCN(size=1)
        V = voltage([-60.0])
        ca = ca_info(Ci=1e-4)
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
        expected = -perm * ch.m.value**3 * drive
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm**2)),
                expected,
                atol=1e-6,
            )
        )


class CaL_SU2015_DCNTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(CaL_SU2015_DCN.root_type, HHTypedNeuron)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(CaL_SU2015_DCN, HH))

    def test_gate_definition_is_m_squared_h(self) -> None:
        ch = CaL_SU2015_DCN(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("m", "h"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_default_parameters_are_stored(self) -> None:
        ch = CaL_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ch.g_max, 0.01 * (u.mS / u.cm**2), atol=1e-12 * (u.mS / u.cm**2)))
        self.assertTrue(u.math.allclose(ch.E, 139.0 * u.mV, atol=1e-12 * u.mV))
        self.assertTrue(u.math.allclose(ch.qdeltat, jnp.ones(1), atol=1e-12))

    def test_reset_state_sets_m_and_h_to_inf_values(self) -> None:
        ch = CaL_SU2015_DCN(size=1)
        V = voltage([-60.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_rates_use_direct_formulas(self) -> None:
        ch = CaL_SU2015_DCN(size=3, qdeltat=2.0)
        V = voltage([-79.7, -60.2, -49.6])
        v_mV = V.to_decimal(u.mV)

        self.assertTrue(u.math.allclose(ch.f_m_inf(V), ch._m_inf_formula(v_mV), atol=1e-12))
        self.assertTrue(u.math.allclose(ch.f_m_tau(V), ch._m_tau_formula(v_mV) / ch.qdeltat, atol=1e-12))
        self.assertTrue(u.math.allclose(ch.f_h_inf(V), ch._h_inf_formula(v_mV), atol=1e-12))
        self.assertTrue(u.math.allclose(ch.f_h_tau(V), ch._h_tau_formula(v_mV) / ch.qdeltat, atol=1e-12))

    def test_qdeltat_scales_taus(self) -> None:
        base = CaL_SU2015_DCN(size=1, qdeltat=1.0)
        fast = CaL_SU2015_DCN(size=1, qdeltat=2.0)
        V = voltage([-60.0])
        self.assertTrue(u.math.allclose(fast.f_m_tau(V), base.f_m_tau(V) / 2.0, atol=1e-6))
        self.assertTrue(u.math.allclose(fast.f_h_tau(V), base.f_h_tau(V) / 2.0, atol=1e-6))

    def test_compute_derivative_matches_inf_tau_form(self) -> None:
        ch = CaL_SU2015_DCN(size=1, qdeltat=2.0)
        V = voltage([-60.0])
        ch.init_state(V)
        ch.m.value = jnp.array([0.2])
        ch.h.value = jnp.array([0.6])
        ch.compute_derivative(V)
        gates = {gate.name: gate for gate in ch._iter_gates()}
        expected_m = ch.gate_phi(gates["m"]) * (ch.f_m_inf(V) - ch.m.value) / ch.f_m_tau(V) / u.ms
        expected_h = ch.gate_phi(gates["h"]) * (ch.f_h_inf(V) - ch.h.value) / ch.f_h_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected_m, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.h.derivative, expected_h, atol=1e-6 * u.Hz))

    def test_current_matches_fixed_carev_mod_formula_with_braincell_sign(self) -> None:
        ch = CaL_SU2015_DCN(size=1)
        V = voltage([-60.0])
        ch.init_state(V)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        current = ch.current(V)
        expected = ch.g_max * ch.m.value**2 * ch.h.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
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
        V = voltage([-60.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, ca), atol=1e-6))

    def test_compute_derivative_matches_inf_tau_form(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1, qdeltat=2.0)
        V = voltage([-60.0])
        ca = ca_info()
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
        low_v = voltage([-90.0])
        tau = ch.f_h_tau(low_v, ca_info())
        expected = 0.333 * u.math.exp((-90.0 + 466.0) / 66.0)
        self.assertTrue(u.math.allclose(tau, jnp.array([expected]), atol=1e-6))

    def test_h_tau_uses_high_voltage_branch_at_threshold_and_above(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        high_v = voltage([-81.0])
        tau = ch.f_h_tau(high_v, ca_info())
        expected = 0.333 * u.math.exp((-81.0 + 21.0) / -10.5) + 9.32
        self.assertTrue(u.math.allclose(tau, jnp.array([expected]), atol=1e-6))

    def test_current_matches_mod_formula(self) -> None:
        ch = CaLVA_SU2015_DCN(size=1)
        V = voltage([-60.0])
        ca = ca_info(Ci=1e-4)
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
        expected = -perm * ch.m.value**2 * ch.h.value * drive
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm**2)),
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
        V = voltage([-60.0])
        ca = ca_info()
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
        V = voltage([-60.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.25])
        ch.compute_derivative(V, ca)
        phi = ch.gate_phi(ch._iter_gates()[0])
        expected = phi * (ch.f_p_inf(V, ca) - ch.p.value) / ch.f_p_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_depends_on_calcium_concentration(self) -> None:
        ch = CaN_IS2008(size=1)
        V = voltage([-60.0])
        low = ca_info(Ci=1e-4)
        high = ca_info(Ci=10.0)
        ch.init_state(V, low)
        ch.reset_state(V, low)
        i_low = ch.current(V, low)
        i_high = ch.current(V, high)
        high_mag = float(u.math.abs(i_high).to_decimal(DENSITY_UNIT)[0])
        low_mag = float(u.math.abs(i_low).to_decimal(DENSITY_UNIT)[0])
        self.assertGreater(high_mag, low_mag)

    def test_current_matches_formula(self) -> None:
        ch = CaN_IS2008(size=1, E=10.0 * u.mV, g_max=1.0 * (u.mS / u.cm**2))
        V = voltage([-60.0])
        ca = ca_info(Ci=0.001)
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        current = ch.current(V, ca)
        modulation = ca.Ci / (ca.Ci + 0.2 * u.mM)
        expected = ch.g_max * modulation * ch.p.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
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
        V = voltage([-40.0, -60.0])
        ca = ca_info(2)
        ch.init_state(V, ca)
        self.assertEqual(ch.m.value.shape, (2,))
        self.assertEqual(ch.h.value.shape, (2,))
        self.assertEqual(ch.n.value.shape, (2,))

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = self._make(size=1)
        V = voltage([-50.0])
        ca = ca_info()
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
        V = voltage([-50.0])
        ca = ca_info()
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
        V = voltage([-50.0])
        ca = ca_info(E=140.0)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        ch.n.value = jnp.array([0.4])
        current = ch.current(V, ca)
        expected = ch.g_max * ch.m.value * ch.h.value * ch.n.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Cav1p2_MA2020_GoCTest(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p2_MA2020_GoC


class Cav1p2_MA2025_BCTest(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p2_MA2025_BC

    def test_inherits_goc_implementation(self) -> None:
        self.assertTrue(issubclass(self.CLS, Cav1p2_MA2020_GoC))
        self.assertIs(get_registry().get("channel", "Cav1p2_MA2025_BC"), self.CLS)


class Cav1p3_MA2020_GoCTest(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p3_MA2020_GoC


class Cav1p3_MA2025_BCTest(_MHNHHMixin, unittest.TestCase):
    CLS = Cav1p3_MA2025_BC

    def test_inherits_goc_implementation(self) -> None:
        self.assertTrue(issubclass(self.CLS, Cav1p3_MA2020_GoC))
        self.assertIs(get_registry().get("channel", "Cav1p3_MA2025_BC"), self.CLS)


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
        V = voltage([-50.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V, ca), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = Cav3p1_MA2020_GoC(size=1)
        V = voltage([-50.0])
        ca = ca_info()
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
        V = voltage([-40.0])
        ca = ca_info(Ci=0.001, E=140.0)
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.5])
        ch.q.value = jnp.array([0.25])
        current = ch.current(V, ca)
        expected = (
            -ch.g_max
            * ch.p.value**2
            * ch.q.value
            * _cav3p1_nmodl_ghk_flux(
                V=V,
                ci=ca.Ci,
                co=ca.Co,
                z=ch.z,
                temp=ch.temp,
            )
        )
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm**2)),
                expected.to_decimal(u.mA / (u.cm**2)),
                atol=1e-6,
            )
        )

    def test_current_uses_nmodl_constants_not_generic_ghk(self) -> None:
        ch = Cav3p1_MA2020_GoC(size=1, temp=u.celsius2kelvin(36.0))
        V = voltage([-65.0])
        ca = ca_info(Ci=0.00024, E=120.0)
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.06913853271498836])
        ch.q.value = jnp.array([0.26894140923409454])

        actual = ch.current(V, ca).to_decimal(u.mA / (u.cm**2))
        generic = (
            -ch.g_max * ch.p.value**2 * ch.q.value * ghk_flux(V=V, ci=ca.Ci, co=ca.Co, z=ch.z, temp=ch.temp)
        ).to_decimal(u.mA / (u.cm**2))

        self.assertFalse(u.math.allclose(actual, generic, rtol=1e-8, atol=0.0))


class Cav3p1TestPC24Test(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav3p1Test_PC24.root_type, Calcium)

    def test_inherits_hh_template_directly(self) -> None:
        self.assertTrue(issubclass(Cav3p1Test_PC24, HH))

    def test_gates_define_p2_q_structure(self) -> None:
        ch = Cav3p1Test_PC24(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("p", "q"))
        self.assertEqual(tuple(gate.power for gate in gates), (2, 1))

    def test_reset_state_matches_infinities(self) -> None:
        ch = Cav3p1Test_PC24(size=1)
        V = voltage([-50.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V, ca), atol=1e-6))

    def test_current_matches_direct_conductance_law(self) -> None:
        ch = Cav3p1Test_PC24(size=1)
        V = voltage([-40.0])
        ca = ca_info(Ci=0.001, E=140.0)
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.5])
        ch.q.value = jnp.array([0.25])
        current = ch.current(V, ca)
        expected = -ch.g_max * ch.p.value**2 * ch.q.value * (1.0 * u.mV)
        self.assertTrue(u.math.allclose(current, expected, atol=1e-12 * expected.unit))


class _FrozenVariantTests:
    """Shared assertions for a ``freeze_drive_gradient`` variant.

    Every frozen class in this module is its unfrozen sibling plus the single
    ``freeze_drive_gradient`` declaration, so the interesting questions are the
    same for all six: does it still inherit the sibling's parameters, does the
    forward value agree, and is the gradient through the GHK drive actually
    gone?
    """

    FROZEN: type
    UNFROZEN: type
    NAME: str

    def _pair(self):
        V = voltage([-48.0])
        ca = ca_info(size=1, Ci=2.4e-4, E=137.0)
        frozen, base = self.FROZEN(size=1), self.UNFROZEN(size=1)
        for ch in (frozen, base):
            ch.init_state(V, ca)
            ch.reset_state(V, ca)
        return frozen, base, V, ca

    def test_is_registered(self) -> None:
        self.assertIs(get_registry().get("channel", self.NAME), self.FROZEN)

    def test_inherits_its_unfrozen_sibling(self) -> None:
        self.assertTrue(issubclass(self.FROZEN, self.UNFROZEN))

    def test_forward_value_agrees_with_the_unfrozen_sibling(self) -> None:
        frozen, base, V, ca = self._pair()
        unit = u.mA / (u.cm**2)
        self.assertTrue(
            u.math.allclose(
                frozen.current(V, ca).to_decimal(unit),
                base.current(V, ca).to_decimal(unit),
                atol=1e-12,
            )
        )

    def test_freezing_removes_the_gradient_through_the_drive(self) -> None:
        # The gates ignore their voltage argument, so the GHK drive is the
        # only path from V to the current: freezing it must zero the whole
        # derivative, and not freezing it must leave it non-zero.
        frozen, base, V, ca = self._pair()
        unit = u.mA / (u.cm**2)

        def sensitivity(ch):
            return float(jax.grad(lambda mV: (ch.current(mV * u.mV, ca) / unit).sum())(u.get_mantissa(V)[0]))

        self.assertNotEqual(sensitivity(base), 0.0)
        self.assertEqual(sensitivity(frozen), 0.0)


class Cav3p1FrozenMA20GoCTest(_FrozenVariantTests, unittest.TestCase):
    FROZEN = Cav3p1_MA2020_GoC_Frozen
    UNFROZEN = Cav3p1_MA2020_GoC
    NAME = "Cav3p1_MA2020_GoC_Frozen"


class Cav3p1FrozenPC24Test(_FrozenVariantTests, unittest.TestCase):
    FROZEN = Cav3p1_MA2024_PC_Frozen
    UNFROZEN = Cav3p1_MA2024_PC
    NAME = "Cav3p1_MA2024_PC_Frozen"


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
        V = voltage([-30.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))

    def test_gate_phi_uses_default_ri2021_temperature_reference(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1, temp=u.celsius2kelvin(33.0))
        gate = ch._iter_gates()[0]
        self.assertTrue(u.math.allclose(ch.gate_phi(gate), 3.0 * jnp.ones(1), atol=1e-6))

    def test_compute_derivative_matches_hh_inf_tau_form(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1, temp=u.celsius2kelvin(33.0))
        V = voltage([-30.0])
        ca = ca_info()
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.2])
        ch.compute_derivative(V, ca)
        gate = ch._iter_gates()[0]
        expected = ch.gate_phi(gate) * (ch.f_m_inf(V, ca) - ch.m.value) / ch.f_m_tau(V, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected, atol=1e-6 * u.Hz))

    def test_tau_matches_template_branches(self) -> None:
        ch = Cav2p1_RI2021_SC(size=2)
        ca = ca_info(size=2)
        V = voltage([-39.0, -41.0])
        tau = ch.f_m_tau(V, ca)
        expected = jnp.asarray(
            [
                0.2702 + 1.1622 * jnp.exp(-((-39.0 + 26.798) ** 2) / 164.19),
                0.6923 * jnp.exp(-41.0 / 1089.372),
            ]
        )
        self.assertTrue(u.math.allclose(tau, expected, atol=1e-6))

    def test_ghk_drive_uses_the_mod_file_constants(self) -> None:
        # Counter-example for MECH-6a: ``Cav2p1_*.mod`` computes the GHK drive
        # with F = 9.6485e4, R = 8.3145 and T = 273.19 + celsius. The CODATA
        # constants and exact Kelvin of the shared ``ghk_flux`` give a current
        # about 3e-4 (relative) away from this transcription, which rtol=1e-5
        # rejects.
        celsius, pcabar, m, ci, co, v_mV = 36.0, 2.2e-4, 0.7, 5e-5, 2.0, -20.0
        F, R, z = 9.6485e4, 8.3145, 2.0
        zeta = z * F * (1e-3 * v_mV) / (R * (273.19 + celsius))
        ghk = 1e-6 * z * zeta * F * (ci - co * math.exp(-zeta)) / (1.0 - math.exp(-zeta))
        neuron_ica = 1e3 * pcabar * m**3 * ghk  # mA/cm2, outward-positive
        for cls in (Cav2p1_RI2021_SC, Cav2p1_MA2024_PC, Cav2p1_MA2025_BC):
            with self.subTest(cls=cls.__name__):
                ch = cls(size=1, temp=u.celsius2kelvin(celsius))
                V = voltage([v_mV])
                ca = ca_info(Ci=ci, Co=co)
                ch.init_state(V, ca)
                ch.m.value = jnp.array([m])
                current = float(ch.current(V, ca).to_decimal(u.mA / (u.cm**2))[0])
                self.assertAlmostEqual(current / -neuron_ica, 1.0, delta=1e-5)

    def test_current_matches_ghk_template_formula(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1)
        V = voltage([-40.0])
        ca = ca_info(Ci=0.001)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        current = ch.current(V, ca)
        expected = (
            -ch.g_max
            * ch.m.value**3
            * _cav3p1_nmodl_ghk_flux(
                V=V,
                ci=ca.Ci,
                co=ca.Co,
                z=ch.z,
                temp=ch.temp,
            )
        )
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(u.mA / (u.cm**2)),
                expected.to_decimal(u.mA / (u.cm**2)),
                atol=1e-6,
            )
        )

    def test_vshift_applies_to_rates_and_ghk_drive(self) -> None:
        ch = Cav2p1_RI2021_SC(size=1, V_sh=5.0 * u.mV)
        V = voltage([-35.0])
        ca = ca_info(Ci=0.001)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])

        shifted = V - 5.0 * u.mV
        expected_inf = 1.0 / (1.0 + u.math.exp(-(shifted - ch.vhalfm) / ch.cvm))
        expected_current = (
            -ch.g_max
            * ch.m.value**3
            * _cav3p1_nmodl_ghk_flux(
                V=shifted,
                ci=ca.Ci,
                co=ca.Co,
                z=ch.z,
                temp=ch.temp,
            )
        )

        self.assertTrue(u.math.allclose(ch.f_m_inf(V, ca), expected_inf, atol=1e-6))
        self.assertTrue(
            u.math.allclose(
                ch.current(V, ca).to_decimal(u.mA / (u.cm**2)),
                expected_current.to_decimal(u.mA / (u.cm**2)),
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
        V = voltage([-65.0])
        ca = ca_info()

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
        V = voltage([-30.0])
        ca = ca_info()
        bc.init_state(V, ca)
        sc.init_state(V, ca)
        bc.reset_state(V, ca)
        sc.reset_state(V, ca)
        self.assertTrue(u.math.allclose(bc.m.value, sc.m.value, atol=1e-6))


class Cav2p1FrozenPC24Test(_FrozenVariantTests, unittest.TestCase):
    FROZEN = Cav2p1_MA2024_PC_Frozen
    UNFROZEN = Cav2p1_MA2024_PC
    NAME = "Cav2p1_MA2024_PC_Frozen"


class Cav2p1FrozenRI21SCTest(_FrozenVariantTests, unittest.TestCase):
    FROZEN = Cav2p1_RI2021_SC_Frozen
    UNFROZEN = Cav2p1_RI2021_SC
    NAME = "Cav2p1_RI2021_SC_Frozen"


class Cav2p1FrozenMA25BCTest(_FrozenVariantTests, unittest.TestCase):
    FROZEN = Cav2p1_MA2025_BC_Frozen
    UNFROZEN = Cav2p1_MA2025_BC
    NAME = "Cav2p1_MA2025_BC_Frozen"


class Cav3p3FrozenPC24Test(_FrozenVariantTests, unittest.TestCase):
    FROZEN = Cav3p3_MA2024_PC_Frozen
    UNFROZEN = Cav3p3_MA2024_PC
    NAME = "Cav3p3_MA2024_PC_Frozen"


class Cav2p1MA24PCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav2p1_MA2024_PC.root_type, Calcium)

    def test_inherits_sc_variant(self) -> None:
        self.assertTrue(issubclass(Cav2p1_MA2024_PC, Cav2p1_RI2021_SC))

    def test_reset_state_matches_sc_variant(self) -> None:
        temp = u.celsius2kelvin(23.0)
        pc = Cav2p1_MA2024_PC(size=1, temp=temp)
        sc = Cav2p1_RI2021_SC(size=1, temp=temp)
        V = voltage([-30.0])
        ca = ca_info()
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
        V = voltage([-65.0])
        ca = ca_info()
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
        V = voltage([-60.0])
        ca = ca_info()
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
        V = voltage([-40.0])
        ca = ca_info(E=125.0)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        current = ch.current(V, ca)
        expected = ch.g_max * ch.m.value**2 * ch.h.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_vshift_applies_only_to_rates_with_template_sign(self) -> None:
        ch = Cav3p2_RI2021_SC(size=1, V_sh=5.0 * u.mV)
        V = voltage([-60.0])
        ca = ca_info(E=120.0)
        shifted = V + 5.0 * u.mV
        expected_m_inf = 1.0 / (1.0 + u.math.exp(-(shifted.to_decimal(u.mV) + 54.8) / 7.4))
        expected_current = ch.g_max * 0.5**2 * 0.25 * (ca.E - V)

        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])

        self.assertTrue(u.math.allclose(ch.f_m_inf(V, ca), expected_m_inf, atol=1e-6))
        self.assertTrue(
            u.math.allclose(
                ch.current(V, ca).to_decimal(DENSITY_UNIT),
                expected_current.to_decimal(DENSITY_UNIT),
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
        V = voltage([-65.0])
        ca = ca_info()
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
        V = voltage([-65.0])
        ca = ca_info()
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
        V = voltage([-65.0])
        ca = ca_info()
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
        V = voltage([-65.0])
        ca = ca_info()
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
        ca = ca_info(size=2)
        V = voltage([-59.0, -61.0])
        tau_n = ch.f_n_tau(V, ca)
        tau_l = ch.f_l_tau(V, ca)
        expected_n = jnp.asarray(
            [
                7.2 + 0.02 * jnp.exp(59.0 / 14.7),
                0.875 * jnp.exp((-61.0 + 120.0) / 41.0),
            ]
        )
        expected_l = jnp.asarray(
            [
                79.5 + 2.0 * jnp.exp(59.0 / 9.3),
                260.0,
            ]
        )
        self.assertTrue(u.math.allclose(tau_n, expected_n, atol=1e-6))
        self.assertTrue(u.math.allclose(tau_l, expected_l, atol=1e-6))

    def test_current_matches_ghk_template_formula(self) -> None:
        ch = Cav3p3_RI2021_SC(size=1)
        V = voltage([-40.0])
        ca = ca_info(Ci=0.001)
        ch.init_state(V, ca)
        ch.n.value = jnp.array([0.5])
        ch.l.value = jnp.array([0.25])
        current = ch.current(V, ca)
        expected = (
            -ch.g_scale
            * ch.perm
            * ch.n.value**2
            * ch.l.value
            * ghk_flux(
                V=V,
                ci=ca.Ci,
                co=ca.Co,
                z=ch.z,
                temp=ch.temp,
            )
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
        V = voltage([-65.0])
        ca = ca_info()
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
        V = voltage([-20.0])
        ca = ca_info(E=100.0)
        ch.init_state(V, ca)
        ch.s.value = jnp.array([0.5])
        ch.u.value = jnp.array([0.25])
        i = ch.current(V, ca)
        expected = ch.g_max * (ch.s.value**2) * ch.u.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_template_formulas_at_same_reversal(self) -> None:
        temp = u.celsius2kelvin(30.0)
        proto = CaHVA_MA2020_GoC(size=1, temp=temp)
        V = voltage([-30.0])
        ca = ca_info()

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
        expected_current = proto.g_max * (proto.s.value**2) * proto.u.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(DENSITY_UNIT),
                expected_current.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_rates_use_continuous_voltage_formula(self) -> None:
        proto = CaHVA_MA2020_GoC(size=3)
        ca = ca_info(size=3)
        V = voltage([-120.0, -50.0, 60.0])
        v = jnp.array([-120.0, -50.0, 60.0])

        expected_alpha_s = proto.Aalpha_s * jnp.exp((v - proto.V0alpha_s) / proto.Kalpha_s)
        expected_beta_s = proto.Abeta_s * jnp.exp((v - proto.V0beta_s) / proto.Kbeta_s)
        expected_alpha_u = proto.Aalpha_u * jnp.exp((v - proto.V0alpha_u) / proto.Kalpha_u)
        expected_beta_u = proto.Abeta_u * jnp.exp((v - proto.V0beta_u) / proto.Kbeta_u)

        self.assertTrue(u.math.allclose(proto.f_s_alpha(V, ca), expected_alpha_s, atol=1e-12))
        self.assertTrue(u.math.allclose(proto.f_s_beta(V, ca), expected_beta_s, atol=1e-12))
        self.assertTrue(u.math.allclose(proto.f_u_alpha(V, ca), expected_alpha_u, atol=1e-12))
        self.assertTrue(u.math.allclose(proto.f_u_beta(V, ca), expected_beta_u, atol=1e-12))


class Cav2p3MA20GoCTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(Cav2p3_MA2020_GoC.root_type, Calcium)

    def test_reset_state_matches_inf_functions(self) -> None:
        ch = Cav2p3_MA2020_GoC(size=1)
        V = voltage([-60.0])
        ca = ca_info(E=140.0)
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, ca), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, ca), atol=1e-6))

    def test_current_matches_linear_formula(self) -> None:
        ch = Cav2p3_MA2020_GoC(size=1, g_max=0.1 * (u.mS / u.cm**2))
        V = voltage([-40.0])
        ca = ca_info(E=140.0)
        ch.init_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.25])
        i = ch.current(V, ca)
        expected = ch.g_max * (ch.m.value**3) * ch.h.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_matches_template_formulas_at_same_reversal(self) -> None:
        temp = u.celsius2kelvin(34.0)
        proto = Cav2p3_MA2020_GoC(size=1, temp=temp)
        V = voltage([-40.0])
        ca = ca_info(E=140.0)

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
        expected_current = proto.g_max * (proto.m.value**3) * proto.h.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(DENSITY_UNIT),
                expected_current.to_decimal(DENSITY_UNIT),
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
        assert_channels_agree(
            self,
            goc,
            grc,
            voltage([-30.0]),
            ca_info(),
            states=("s", "u"),
        )


class CaZH19IOTest(unittest.TestCase):
    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Ca_ZH2019_IO.root_type, HHTypedNeuron)

    def test_reset_state_matches_f_h_inf(self) -> None:
        ch = Ca_ZH2019_IO(size=1)
        V = voltage([-70.0])
        ch.init_state(V)
        ch.reset_state(V)
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V), atol=1e-6))

    def test_current_uses_instantaneous_m_and_stateful_h(self) -> None:
        ch = Ca_ZH2019_IO(size=1, E=120.0 * u.mV, mMidV=-61.0 * u.mV)
        V = voltage([-50.0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])
        i = ch.current(V)
        expected = ch.g_max * ch.f_m_inf(V) * ch.h.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(DENSITY_UNIT),
                expected.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_default_freezes_instantaneous_activation_gradient(self) -> None:
        ch = Ca_ZH2019_IO(size=1, E=120.0 * u.mV, mMidV=-61.0 * u.mV)
        V0 = -50.0
        V = voltage([V0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])

        def current_decimal(v_mV):
            return ch.current(jnp.asarray([v_mV]) * u.mV).to_decimal(DENSITY_UNIT)[0]

        actual = jax.grad(current_decimal)(V0)
        expected = -(ch.g_max * ch.f_m_inf(V) * ch.h.value).to_decimal(u.mS / u.cm**2)[0]
        self.assertTrue(u.math.allclose(actual, expected, atol=1e-6))

    def test_freeze_m_inf_can_be_disabled(self) -> None:
        ch = Ca_ZH2019_IO(
            size=1,
            E=120.0 * u.mV,
            mMidV=-61.0 * u.mV,
            freeze_m_inf=False,
        )
        V0 = -50.0
        V = voltage([V0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])

        def current_decimal(v_mV):
            return ch.current(jnp.asarray([v_mV]) * u.mV).to_decimal(DENSITY_UNIT)[0]

        def unfrozen_expected_current(v_mV):
            v = jnp.asarray([v_mV]) * u.mV
            return (ch.g_max * ch.f_m_inf(v) * ch.h.value * (ch.E - v)).to_decimal(DENSITY_UNIT)[0]

        actual = jax.grad(current_decimal)(V0)
        expected = jax.grad(unfrozen_expected_current)(V0)
        frozen_slope = -(ch.g_max * ch.f_m_inf(V) * ch.h.value).to_decimal(u.mS / u.cm**2)[0]
        self.assertTrue(u.math.allclose(actual, expected, atol=1e-6))
        self.assertFalse(bool(u.math.allclose(actual, frozen_slope, atol=1e-6)))

    def test_frozen_variant_is_registered(self) -> None:
        self.assertIs(get_registry().get("channel", "Ca_ZH2019_IO_Frozen"), Ca_ZH2019_IO_Frozen)

    def test_frozen_current_matches_unfrozen_value(self) -> None:
        V = voltage([-50.0])
        base = Ca_ZH2019_IO(size=1, E=120.0 * u.mV, mMidV=-61.0 * u.mV, freeze_m_inf=False)
        frozen = Ca_ZH2019_IO_Frozen(size=1, E=120.0 * u.mV, mMidV=-61.0 * u.mV)
        base.init_state(V)
        frozen.init_state(V)
        base.h.value = jnp.array([0.25])
        frozen.h.value = jnp.array([0.25])
        self.assertTrue(
            u.math.allclose(
                frozen.current(V).to_decimal(DENSITY_UNIT),
                base.current(V).to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_frozen_variant_keeps_only_driving_force_slope(self) -> None:
        ch = Ca_ZH2019_IO_Frozen(size=1, E=120.0 * u.mV, mMidV=-61.0 * u.mV)
        V0 = -50.0
        V = voltage([V0])
        ch.init_state(V)
        ch.h.value = jnp.array([0.25])

        def current_decimal(v_mV):
            return ch.current(jnp.asarray([v_mV]) * u.mV).to_decimal(DENSITY_UNIT)[0]

        actual = jax.grad(current_decimal)(V0)
        expected = -(ch.g_max * ch.f_m_inf(V) * ch.h.value).to_decimal(u.mS / u.cm**2)[0]
        self.assertTrue(u.math.allclose(actual, expected, atol=1e-6))


# The expressions as written before the removable 0/0 singularities were
# replaced by their analytic limits, verbatim; the singularity tests below
# compare against them away from the singular voltage V*.
def _prefix_caht_re1993_p_alpha(self, V, Ca):
    temp = (-V + self.V_sh).to_decimal(u.mV)
    delta = -27.0 + temp
    return 0.055 * delta / (u.math.exp(delta / 3.8) - 1.0)


def _prefix_cahva_su2015_dcn_m_tau(self, V, Ca):
    V = V.to_decimal(u.mV)
    activation = 31.746 / (u.math.exp((V - 5.0) / -13.89) + 1.0)
    correction = 3.97e-4 * (V + 8.9) / (u.math.exp((V + 8.9) / 5.0) - 1.0)
    return 1.0 / (activation + correction) / self.qdeltat


def _prefix_su2015_dcn_ghk_drive(V, ci, co, temp):
    v_mV = V.to_decimal(u.mV)
    temp_K = temp.to_decimal(u.kelvin)
    A = u.math.exp(-23.20764929 * v_mV / temp_K)
    return (
        (4.47814e6 * v_mV / temp_K) * ((ci.to_decimal(u.mM) / 1000.0) - (co.to_decimal(u.mM) / 1000.0) * A) / (1.0 - A)
    )


def _prefix_cav1p2_m_tau(self, V, Ca):
    V = V.to_decimal(u.mV)
    m_alpha = 39800.0 * (V + 8.124) / (u.math.exp((V + 8.124) / 9.005) - 1.0)
    m_beta = 990.0 * u.math.exp(V / 31.4)
    return 1.0 / (m_alpha + m_beta)


_PREFIX_CAHVA_SU2015_DCN = (
    (CaHVA_SU2015_DCN, "f_m_tau", _prefix_cahva_su2015_dcn_m_tau),
    (calcium_module, "_su2015_dcn_ghk_drive", _prefix_su2015_dcn_ghk_drive),
)
_PREFIX_CAV1P2 = ((Cav1p2_MA2020_GoC, "f_m_tau", _prefix_cav1p2_m_tau),)

# class, singular voltages (mV), the pre-fix expressions.
_REMOVABLE_SINGULARITIES = (
    (CaHT_Re1993, (-27.0,), ((CaHT_Re1993, "f_p_alpha", _prefix_caht_re1993_p_alpha),)),
    (CaHVA_SU2015_DCN, (-8.9, 0.0), _PREFIX_CAHVA_SU2015_DCN),
    (CaLVA_SU2015_DCN, (0.0,), _PREFIX_CAHVA_SU2015_DCN[1:]),
    (Cav1p2_MA2020_GoC, (-8.124,), _PREFIX_CAV1P2),
    (Cav1p2_MA2025_BC, (-8.124,), _PREFIX_CAV1P2),
)
_GRID_MV = np.arange(-100_000, 60_001) / 1000.0


def _gate_quantities(cls, V_mV) -> dict:
    """Steady-state gates, their derivatives at state 0 and the current at the steady state, in float64."""
    V_mV = np.asarray(V_mV, dtype=float)
    V = jnp.asarray(V_mV) * u.mV
    ca = ca_info(V_mV.size)
    ch = cls(size=V_mV.size)
    ch.init_state(V, ca)
    ch.reset_state(V, ca)
    names = [gate.name for gate in ch._iter_gates()]
    found = {f"{name}.value": getattr(ch, name).value for name in names}
    found["current"] = ch.current(V, ca).to_decimal(DENSITY_UNIT)
    for name in names:
        getattr(ch, name).value = jnp.zeros(V_mV.size)
    ch.compute_derivative(V, ca)
    for name in names:
        found[f"{name}.derivative"] = getattr(ch, name).derivative.to_decimal(u.ms**-1)
    return {
        key: np.broadcast_to(np.asarray(u.get_magnitude(value), dtype=float), V_mV.shape)
        for key, value in found.items()
    }


class RemovableSingularityTest(unittest.TestCase):
    """Rates and GHK drives that are 0/0 at one voltage V* use their analytic limit there.

    Away from V* the result is the source expression's; at V* it is the
    limit, so a Cell clamped exactly at V* no longer integrates NaN.
    """

    def _assert_finite(self, cls, V_mV):
        for key, values in _gate_quantities(cls, V_mV).items():
            self.assertTrue(
                np.all(np.isfinite(values)), f"{cls.__name__} {key} at {np.asarray(V_mV)[~np.isfinite(values)]}"
            )

    def test_finite_on_the_grid_and_equal_to_the_source_expression_away_from_V_star(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for cls, singular, prefix in _REMOVABLE_SINGULARITIES:
                with self.subTest(cls=cls.__name__):
                    fixed = _gate_quantities(cls, _GRID_MV)
                    for key, values in fixed.items():
                        self.assertTrue(np.all(np.isfinite(values)), f"{cls.__name__} {key}")
                    away = np.min(np.abs(_GRID_MV[:, None] - np.asarray(singular)[None, :]), axis=1) >= 0.01 - 1e-9
                    with contextlib.ExitStack() as stack:
                        for owner, attribute, function in prefix:
                            stack.enter_context(mock.patch.object(owner, attribute, function))
                        source = _gate_quantities(cls, _GRID_MV[away])
                    for key, values in source.items():
                        np.testing.assert_allclose(
                            fixed[key][away], values, rtol=1e-10, atol=0.0, err_msg=f"{cls.__name__} {key}"
                        )

    def test_the_value_at_V_star_is_the_limit_of_its_neighbours(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for cls, singular, _ in _REMOVABLE_SINGULARITIES:
                for v_star in singular:
                    with self.subTest(cls=cls.__name__, V=v_star):
                        at, below, above = (_gate_quantities(cls, [v_star + h]) for h in (0.0, -1e-4, 1e-4))
                        for key, value in at.items():
                            mean = 0.5 * (below[key] + above[key])
                            self.assertLessEqual(
                                float(np.abs(value - mean)[0]), 1e-6 * float(np.abs(value)[0]), f"{cls.__name__} {key}"
                            )

    def test_the_source_expressions_are_not_finite_at_V_star(self):
        # Counter-example: with the pre-fix expressions, the finiteness check fails at V*.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for cls, singular, prefix in _REMOVABLE_SINGULARITIES:
                for v_star in singular:
                    with self.subTest(cls=cls.__name__, V=v_star):
                        self._assert_finite(cls, [v_star])
                        with contextlib.ExitStack() as stack:
                            for owner, attribute, function in prefix:
                                stack.enter_context(mock.patch.object(owner, attribute, function))
                            with self.assertRaises(AssertionError):
                                self._assert_finite(cls, [v_star])


class NmodlGhkNearZeroTest(unittest.TestCase):
    def test_cav3p1_helper_matches_the_limit_near_0_mV(self) -> None:
        assert_ghk_near_zero(self, calcium_module._cav3p1_nmodl_ghk_flux, _CAV3P1_GHK_NEAR_ZERO)

    def test_cav3p3_helper_matches_the_limit_near_0_mV(self) -> None:
        assert_ghk_near_zero(self, calcium_module._cav3p3_nmodl_ghk_flux, _CAV3P3_GHK_NEAR_ZERO)


if __name__ == "__main__":
    unittest.main()
