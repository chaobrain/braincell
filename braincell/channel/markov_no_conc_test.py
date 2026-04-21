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

import brainstate
import brainunit as u
import jax.numpy as jnp

brainstate.environ.set(precision=64)

from braincell._base import IonInfo
from braincell.channel._template import Markov
from braincell.channel.markov_no_conc import NaFHF_MA20_GrC
from braincell.channel.markov_no_conc import Nav1p1_MA25_BC
from braincell.channel.markov_no_conc import Nav1p1_RI21_SC
from braincell.channel.markov_no_conc import Nav1p6_MA20_GoC
from braincell.channel.markov_no_conc import Nav1p6_MA24_PC
from braincell.channel.markov_no_conc import Nav1p6_MA25_BC
from braincell.channel.markov_no_conc import Nav1p6_RI21_SC
from braincell.channel.markov_no_conc import Nav_MA20_GrC
from braincell.channel.sodium import INa_Rsg
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


def _legacy(size: int = 1, **kwargs) -> INa_Rsg:
    return INa_Rsg(size=size, g_max=16.0 * (u.mS / u.cm ** 2), **kwargs)


def _legacy_nav1p1(size: int = 1, temp=u.celsius2kelvin(22.0), **kwargs) -> INa_Rsg:
    legacy = INa_Rsg(size=size, g_max=8.0 * (u.mS / u.cm ** 2), T=temp, **kwargs)
    legacy.phi = 2.7 ** (((temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)
    legacy.Oon = 2.3
    legacy.epsilon = 1e-12
    legacy.alfac = (legacy.Oon / legacy.Con) ** (1 / 4)
    return legacy


def _seed_states(channel) -> None:
    values = {
        "C1": jnp.array([0.18, 0.12]),
        "C2": jnp.array([0.10, 0.08]),
        "C3": jnp.array([0.05, 0.04]),
        "C4": jnp.array([0.02, 0.03]),
        "C5": jnp.array([0.01, 0.02]),
        "I1": jnp.array([0.06, 0.07]),
        "I2": jnp.array([0.05, 0.04]),
        "I3": jnp.array([0.04, 0.03]),
        "I4": jnp.array([0.03, 0.02]),
        "I5": jnp.array([0.02, 0.01]),
        "O": jnp.array([0.08, 0.09]),
        "B": jnp.array([0.01, 0.02]),
    }
    for name, value in values.items():
        getattr(channel, name).value = value


class _Nav1p6Mixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_sodium(self) -> None:
        self.assertIs(self.CLS.root_type, Sodium)

    def test_default_gmax_matches_mod_default(self) -> None:
        ch = self._make(size=1)
        expected = 16.0 * (u.mS / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                expected.to_decimal(u.mS / u.cm ** 2),
                atol=1e-12,
            )
        )

    def test_init_state_derives_visible_state_list_from_pairs(self) -> None:
        ch = self._make(size=3)
        V = _V([-60.0, -50.0, -40.0])
        na = _na_info(3)

        ch.init_state(V, na)

        self.assertEqual(
            ch.state_names,
            ("C1", "C2", "C3", "C4", "C5", "I1", "I2", "I3", "I4", "I5", "O", "B"),
        )
        self.assertEqual(ch.redundant_state, "I6")
        self.assertEqual(len(ch.state_pairs), 17)
        self.assertEqual(ch.state_pairs[-1], ("O", "I6", "fin", "bin"))
        for name in ch.state_names:
            self.assertTrue(hasattr(ch, name))
            self.assertEqual(getattr(ch, name).value.shape, (3,))
        self.assertFalse(hasattr(ch, "I6"))

    def test_state_values_reconstructs_hidden_state_from_probability_conservation(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()

        ch.init_state(V, na)
        ch.C1.value = jnp.array([0.2])
        ch.O.value = jnp.array([0.1])
        ch.B.value = jnp.array([0.05])

        states = ch.state_values()

        self.assertTrue(u.math.allclose(states["C1"], jnp.array([0.2]), atol=1e-6))
        self.assertTrue(u.math.allclose(states["O"], jnp.array([0.1]), atol=1e-6))
        self.assertTrue(u.math.allclose(states["B"], jnp.array([0.05]), atol=1e-6))
        self.assertTrue(u.math.allclose(states["I6"], jnp.array([0.65]), atol=1e-6))

    def test_temp_matches_legacy_temperature_factor(self) -> None:
        legacy = _legacy(size=1, T=u.celsius2kelvin(30.0))
        proto = self._make(size=1, temp=u.celsius2kelvin(30.0))

        self.assertTrue(u.math.allclose(proto.temp, u.celsius2kelvin(30.0), atol=1e-6 * u.kelvin))
        self.assertTrue(u.math.allclose(proto.phi, legacy.phi, atol=1e-6))

    def test_current_matches_legacy_implementation_with_matched_gmax(self) -> None:
        legacy = _legacy(size=1)
        proto = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()

        legacy.init_state(V, na)
        proto.init_state(V, na)
        legacy.reset_state(V, na)
        proto.reset_state(V, na)
        legacy.O.value = jnp.array([0.35])
        proto.O.value = jnp.array([0.35])

        i_legacy = legacy.current(V, na)
        i_proto = proto.current(V, na)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(unit),
                i_legacy.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_compute_derivative_matches_legacy_implementation(self) -> None:
        legacy = _legacy(size=2)
        proto = self._make(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        legacy.init_state(V, na)
        proto.init_state(V, na)
        _seed_states(legacy)
        _seed_states(proto)

        legacy.compute_derivative(V, na)
        proto.compute_derivative(V, na)

        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).derivative,
                    getattr(legacy, name).derivative,
                    atol=1e-6 * u.Hz,
                )
            )

    def test_compute_derivative_is_inherited_from_markov_template(self) -> None:
        proto = self._make(size=1)
        self.assertIs(proto.compute_derivative.__func__, Markov.compute_derivative)

    def test_reset_steady_state_produces_stationary_distribution(self) -> None:
        proto = self._make(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        proto.init_state(V, na)
        proto.reset_steady_state(V, na)
        states = proto.state_values()

        total = None
        for name in proto.state_names + (proto.redundant_state,):
            total = states[name] if total is None else (total + states[name])
        self.assertTrue(u.math.allclose(total, jnp.ones(2), atol=1e-6))

        proto.compute_derivative(V, na)
        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )

    def test_make_integration_matches_legacy_implementation(self) -> None:
        legacy = _legacy(size=2, solver="euler")
        proto = self._make(size=2, solver="euler")
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        legacy.init_state(V, na)
        proto.init_state(V, na)
        _seed_states(legacy)
        _seed_states(proto)

        with brainstate.environ.context(dt=0.02 * u.ms):
            legacy.make_integration(V, na)
            proto.make_integration(V, na)

        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).value,
                    getattr(legacy, name).value,
                    atol=1e-6,
                )
            )

    def test_substeps_defaults_to_legacy_refinement(self) -> None:
        proto = self._make(size=1)
        self.assertEqual(proto.substeps, 5)

    def test_substeps_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self._make(size=1, substeps=0)


class Nav1p6MA20GoCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_MA20_GoC


class Nav1p6MA24PCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_MA24_PC

    def test_matches_goc_implementation_under_same_conditions(self) -> None:
        goc = Nav1p6_MA20_GoC(size=2)
        pc = Nav1p6_MA24_PC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        goc.init_state(V, na)
        pc.init_state(V, na)
        goc.reset_steady_state(V, na)
        pc.reset_steady_state(V, na)

        goc_states = goc.state_values()
        pc_states = pc.state_values()
        for name in goc.state_names + (goc.redundant_state,):
            self.assertTrue(u.math.allclose(goc_states[name], pc_states[name], atol=1e-6))

        goc.compute_derivative(V, na)
        pc.compute_derivative(V, na)
        for name in goc.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(goc, name).derivative,
                    getattr(pc, name).derivative,
                    atol=1e-6 * u.Hz,
                )
            )

        i_goc = goc.current(V, na)
        i_pc = pc.current(V, na)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(unit),
                i_pc.to_decimal(unit),
                atol=1e-6,
            )
        )


class Nav1p6MA25BCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_MA25_BC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p6_MA25_BC(size=2)
        steady = Nav1p6_MA25_BC(size=2)

        reset.init_state(V, na)
        steady.init_state(V, na)
        reset.reset_state(V, na)
        steady.reset_steady_state(V, na)

        reset_states = reset.state_values()
        steady_states = steady.state_values()
        for name in reset.state_names + (reset.redundant_state,):
            self.assertTrue(u.math.allclose(reset_states[name], steady_states[name], atol=1e-6))

        reset.compute_derivative(V, na)
        for name in reset.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(reset, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )

    def test_matches_goc_implementation_under_same_conditions(self) -> None:
        goc = Nav1p6_MA20_GoC(size=2)
        bc = Nav1p6_MA25_BC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        goc.init_state(V, na)
        bc.init_state(V, na)
        goc.reset_steady_state(V, na)
        bc.reset_steady_state(V, na)

        goc_states = goc.state_values()
        bc_states = bc.state_values()
        for name in goc.state_names + (goc.redundant_state,):
            self.assertTrue(u.math.allclose(goc_states[name], bc_states[name], atol=1e-6))

        goc.compute_derivative(V, na)
        bc.compute_derivative(V, na)
        for name in goc.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(goc, name).derivative,
                    getattr(bc, name).derivative,
                    atol=1e-6 * u.Hz,
                )
            )

        i_goc = goc.current(V, na)
        i_bc = bc.current(V, na)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(unit),
                i_bc.to_decimal(unit),
                atol=1e-6,
            )
        )


class Nav1p6RI21SCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_RI21_SC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p6_RI21_SC(size=2)
        steady = Nav1p6_RI21_SC(size=2)

        reset.init_state(V, na)
        steady.init_state(V, na)
        reset.reset_state(V, na)
        steady.reset_steady_state(V, na)

        reset_states = reset.state_values()
        steady_states = steady.state_values()
        for name in reset.state_names + (reset.redundant_state,):
            self.assertTrue(u.math.allclose(reset_states[name], steady_states[name], atol=1e-6))

        reset.compute_derivative(V, na)
        for name in reset.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(reset, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )

    def test_matches_goc_implementation_under_same_conditions(self) -> None:
        goc = Nav1p6_MA20_GoC(size=2)
        sc = Nav1p6_RI21_SC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        goc.init_state(V, na)
        sc.init_state(V, na)
        goc.reset_steady_state(V, na)
        sc.reset_steady_state(V, na)

        goc_states = goc.state_values()
        sc_states = sc.state_values()
        for name in goc.state_names + (goc.redundant_state,):
            self.assertTrue(u.math.allclose(goc_states[name], sc_states[name], atol=1e-6))

        goc.compute_derivative(V, na)
        sc.compute_derivative(V, na)
        for name in goc.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(goc, name).derivative,
                    getattr(sc, name).derivative,
                    atol=1e-6 * u.Hz,
                )
            )

        i_goc = goc.current(V, na)
        i_sc = sc.current(V, na)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(unit),
                i_sc.to_decimal(unit),
                atol=1e-6,
            )
        )


class _Nav1p1Mixin:
    CLS = None

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_root_type_is_sodium(self) -> None:
        self.assertIs(self.CLS.root_type, Sodium)

    def test_default_gmax_matches_mod_default(self) -> None:
        ch = self._make(size=1)
        expected = 8.0 * (u.mS / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                expected.to_decimal(u.mS / u.cm ** 2),
                atol=1e-12,
            )
        )

    def test_init_state_derives_visible_state_list_from_pairs(self) -> None:
        ch = self._make(size=3)
        V = _V([-60.0, -50.0, -40.0])
        na = _na_info(3)

        ch.init_state(V, na)

        self.assertEqual(
            ch.state_names,
            ("C1", "C2", "C3", "C4", "C5", "I1", "I2", "I3", "I4", "I5", "O", "B"),
        )
        self.assertEqual(ch.redundant_state, "I6")
        self.assertEqual(len(ch.state_pairs), 17)
        self.assertEqual(ch.state_pairs[-1], ("O", "I6", "fin", "bin"))

    def test_temperature_and_parameters_match_mod_defaults(self) -> None:
        proto = self._make(size=1, temp=u.celsius2kelvin(30.0))
        expected_phi = 2.7 ** (((proto.temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)
        self.assertTrue(u.math.allclose(proto.phi, expected_phi, atol=1e-6))
        self.assertEqual(proto.Oon, 2.3)
        self.assertEqual(proto.epsilon, 1e-12)
        self.assertEqual(proto.zgate, 2.5435)

    def test_markov_dynamics_match_legacy_when_gate_current_is_off(self) -> None:
        legacy = _legacy_nav1p1(size=2)
        proto = self._make(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        legacy.init_state(V, na)
        proto.init_state(V, na)
        _seed_states(legacy)
        _seed_states(proto)

        legacy.compute_derivative(V, na)
        proto.compute_derivative(V, na)
        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).derivative,
                    getattr(legacy, name).derivative,
                    atol=1e-6 * u.Hz,
                )
            )

    def test_current_matches_legacy_when_gate_current_is_off(self) -> None:
        legacy = _legacy_nav1p1(size=1)
        proto = self._make(size=1, gateCurrent=0.0)
        V = _V([-60.0])
        na = _na_info()

        legacy.init_state(V, na)
        proto.init_state(V, na)
        legacy.reset_state(V, na)
        proto.reset_state(V, na)
        legacy.O.value = jnp.array([0.35])
        proto.O.value = jnp.array([0.35])

        i_legacy = legacy.current(V, na)
        i_proto = proto.current(V, na)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(unit),
                i_legacy.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_gate_current_path_matches_manual_formula(self) -> None:
        ch = self._make(size=1, gateCurrent=1.0)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.C1.value = jnp.array([0.12])
        ch.C2.value = jnp.array([0.08])
        ch.C3.value = jnp.array([0.05])
        ch.C4.value = jnp.array([0.03])
        ch.C5.value = jnp.array([0.02])
        ch.I1.value = jnp.array([0.07])
        ch.I2.value = jnp.array([0.06])
        ch.I3.value = jnp.array([0.04])
        ch.I4.value = jnp.array([0.03])
        ch.I5.value = jnp.array([0.02])
        ch.O.value = jnp.array([0.1])

        conductive = ch.g_max * ch.O.value * (na.E - V)
        gate_flip = (
                        ch.f01(V) * ch.C1.value
                        + (ch.f02(V) - ch.b01(V)) * ch.C2.value
                        + (ch.f03(V) - ch.b02(V)) * ch.C3.value
                        + (ch.f04(V) - ch.b03(V)) * ch.C4.value
                        - ch.b04(V) * ch.C5.value
                        + ch.f11(V) * ch.I1.value
                        + (ch.f12(V) - ch.b11(V)) * ch.I2.value
                        + (ch.f13(V) - ch.b12(V)) * ch.I3.value
                        + (ch.f14(V) - ch.b13(V)) * ch.I4.value
                        - ch.b14(V) * ch.I5.value
                    ) / u.ms
        nc = 1e12 * ch.g_max / ch.gunit
        igate = nc * 1e6 * ch.e0 * ch.zgate * gate_flip
        expected = conductive - igate
        current = ch.current(V, na)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit),
                expected.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_compute_derivative_is_inherited_from_markov_template(self) -> None:
        proto = self._make(size=1)
        self.assertIs(proto.compute_derivative.__func__, Markov.compute_derivative)

    def test_reset_steady_state_produces_stationary_distribution(self) -> None:
        proto = self._make(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        proto.init_state(V, na)
        proto.reset_steady_state(V, na)
        states = proto.state_values()
        total = None
        for name in proto.state_names + (proto.redundant_state,):
            total = states[name] if total is None else (total + states[name])
        self.assertTrue(u.math.allclose(total, jnp.ones(2), atol=1e-6))

        proto.compute_derivative(V, na)
        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )

    def test_make_integration_matches_legacy_when_gate_current_is_off(self) -> None:
        legacy = _legacy_nav1p1(size=2, solver="euler")
        proto = self._make(size=2, solver="euler", gateCurrent=0.0)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        legacy.init_state(V, na)
        proto.init_state(V, na)
        _seed_states(legacy)
        _seed_states(proto)

        with brainstate.environ.context(dt=0.02 * u.ms):
            legacy.make_integration(V, na)
            proto.make_integration(V, na)
        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).value,
                    getattr(legacy, name).value,
                    atol=1e-6,
                )
            )


class Nav1p1MA25BCTest(_Nav1p1Mixin, unittest.TestCase):
    CLS = Nav1p1_MA25_BC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p1_MA25_BC(size=2)
        steady = Nav1p1_MA25_BC(size=2)

        reset.init_state(V, na)
        steady.init_state(V, na)
        reset.reset_state(V, na)
        steady.reset_steady_state(V, na)

        reset_states = reset.state_values()
        steady_states = steady.state_values()
        for name in reset.state_names + (reset.redundant_state,):
            self.assertTrue(u.math.allclose(reset_states[name], steady_states[name], atol=1e-6))

        reset.compute_derivative(V, na)
        for name in reset.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(reset, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )


class Nav1p1RI21SCTest(_Nav1p1Mixin, unittest.TestCase):
    CLS = Nav1p1_RI21_SC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p1_RI21_SC(size=2)
        steady = Nav1p1_RI21_SC(size=2)

        reset.init_state(V, na)
        steady.init_state(V, na)
        reset.reset_state(V, na)
        steady.reset_steady_state(V, na)

        reset_states = reset.state_values()
        steady_states = steady.state_values()
        for name in reset.state_names + (reset.redundant_state,):
            self.assertTrue(u.math.allclose(reset_states[name], steady_states[name], atol=1e-6))

        reset.compute_derivative(V, na)
        for name in reset.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(reset, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )

    def test_matches_bc_implementation_under_same_conditions(self) -> None:
        bc = Nav1p1_MA25_BC(size=2)
        sc = Nav1p1_RI21_SC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        bc.init_state(V, na)
        sc.init_state(V, na)
        bc.reset_steady_state(V, na)
        sc.reset_steady_state(V, na)

        bc_states = bc.state_values()
        sc_states = sc.state_values()
        for name in bc.state_names + (bc.redundant_state,):
            self.assertTrue(u.math.allclose(bc_states[name], sc_states[name], atol=1e-6))

        bc.compute_derivative(V, na)
        sc.compute_derivative(V, na)
        for name in bc.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(bc, name).derivative,
                    getattr(sc, name).derivative,
                    atol=1e-6 * u.Hz,
                )
            )

        i_bc = bc.current(V, na)
        i_sc = sc.current(V, na)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                i_bc.to_decimal(unit),
                i_sc.to_decimal(unit),
                atol=1e-6,
            )
        )


class NavMA20GrCTest(unittest.TestCase):
    def test_root_type_and_defaults(self) -> None:
        ch = Nav_MA20_GrC(size=1)
        self.assertIs(Nav_MA20_GrC.root_type, Sodium)
        expected = 13.0 * (u.mS / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                expected.to_decimal(u.mS / u.cm ** 2),
                atol=1e-12,
            )
        )

    def test_init_state_and_hidden_state_layout(self) -> None:
        ch = Nav_MA20_GrC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        ch.init_state(V, na)

        self.assertEqual(
            ch.state_names,
            ("C1", "C2", "C3", "C4", "C5", "O", "OB", "I1", "I2", "I3", "I4", "I5"),
        )
        self.assertEqual(ch.redundant_state, "I6")
        self.assertFalse(hasattr(ch, "I6"))
        self.assertEqual(ch.state_pairs[-1], ("I5", "I6", "f1n", "b1n"))

    def test_rate_formulas_follow_mod_definition(self) -> None:
        ch = Nav_MA20_GrC(size=1)
        V = _V([-60.0])
        factor = 3 ** (((ch.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)
        alfa = factor * ch.Aalfa * u.math.exp((V / u.mV) / ch.Valfa)
        beta = factor * ch.Abeta * u.math.exp(-(V / u.mV) / ch.Vbeta)
        teta = factor * ch.Ateta * u.math.exp(-(V / u.mV) / ch.Vteta)
        self.assertTrue(u.math.allclose(ch.f01(V), ch.n1 * alfa, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.b01(V), ch.n4 * beta, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.bip(V), teta, atol=1e-6))

    def test_current_uses_open_state_only(self) -> None:
        ch = Nav_MA20_GrC(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.O.value = jnp.array([0.2])
        ch.OB.value = jnp.array([0.7])
        current = ch.current(V, na)
        expected = ch.g_max * ch.O.value * (na.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit),
                expected.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_selected_derivatives_match_manual_markov_balance(self) -> None:
        ch = Nav_MA20_GrC(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        values = {
            "C1": 0.12, "C2": 0.08, "C3": 0.07, "C4": 0.05, "C5": 0.04,
            "O": 0.1, "OB": 0.09, "I1": 0.06, "I2": 0.05, "I3": 0.04, "I4": 0.03, "I5": 0.02,
        }
        for name, value in values.items():
            getattr(ch, name).value = jnp.array([value])
        states = ch.state_values()
        ch.compute_derivative(V, na)

        expected_dO = (
                          states["C5"] * ch.f0O(V)
                          + states["OB"] * ch.bip(V)
                          + states["I6"] * ch.bin(V)
                          - states["O"] * ch.b0O(V)
                          - states["O"] * ch.fip(V)
                          - states["O"] * ch.fin(V)
                      ) / u.ms
        expected_dOB = (states["O"] * ch.fip(V) - states["OB"] * ch.bip(V)) / u.ms
        self.assertTrue(u.math.allclose(ch.O.derivative, expected_dO, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.OB.derivative, expected_dOB, atol=1e-6 * u.Hz))

    def test_reset_steady_state_produces_stationary_distribution(self) -> None:
        ch = Nav_MA20_GrC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        ch.init_state(V, na)
        ch.reset_steady_state(V, na)
        states = ch.state_values()
        total = None
        for name in ch.state_names + (ch.redundant_state,):
            total = states[name] if total is None else (total + states[name])
        self.assertTrue(u.math.allclose(total, jnp.ones(2), atol=1e-6))
        ch.compute_derivative(V, na)
        for name in ch.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(ch, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )


class NaFHFMA20GrCTest(unittest.TestCase):
    def test_root_type_and_defaults(self) -> None:
        ch = NaFHF_MA20_GrC(size=1)
        self.assertIs(NaFHF_MA20_GrC.root_type, Sodium)
        expected = 13.0 * (u.mS / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                expected.to_decimal(u.mS / u.cm ** 2),
                atol=1e-12,
            )
        )

    def test_init_state_and_hidden_state_layout(self) -> None:
        ch = NaFHF_MA20_GrC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        ch.init_state(V, na)

        self.assertEqual(
            ch.state_names,
            ("C1", "C2", "C3", "C4", "C5", "O", "OB", "I1", "I2", "I3", "I4", "I5", "L3", "L4", "L5", "L6"),
        )
        self.assertEqual(ch.redundant_state, "I6")
        self.assertFalse(hasattr(ch, "I6"))
        self.assertEqual(ch.state_pairs[-1], ("I5", "I6", "f1n", "b1n"))

    def test_extended_rate_formulas_follow_mod_definition(self) -> None:
        ch = NaFHF_MA20_GrC(size=1)
        V = _V([-60.0])
        factor = 3 ** (((ch.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)
        alfa = factor * ch.Aalfa * u.math.exp((V / u.mV) / ch.Valfa)
        self.assertTrue(u.math.allclose(ch.f33(V), ch.n3 * alfa * ch.c, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.b33(V), ch.n2 * alfa * ch.d, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.fl6(V), factor * ch.ALon * ch.c ** 2, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.bl6(V), factor * ch.ALoff * ch.d ** 2, atol=1e-6))

    def test_current_uses_open_state_only(self) -> None:
        ch = NaFHF_MA20_GrC(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.O.value = jnp.array([0.2])
        ch.OB.value = jnp.array([0.5])
        ch.L6.value = jnp.array([0.2])
        current = ch.current(V, na)
        expected = ch.g_max * ch.O.value * (na.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit),
                expected.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_selected_long_inactivation_derivatives_match_manual_balance(self) -> None:
        ch = NaFHF_MA20_GrC(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        values = {
            "C1": 0.08, "C2": 0.06, "C3": 0.07, "C4": 0.05, "C5": 0.04,
            "O": 0.08, "OB": 0.07, "I1": 0.05, "I2": 0.04, "I3": 0.04, "I4": 0.03, "I5": 0.02,
            "L3": 0.06, "L4": 0.05, "L5": 0.04, "L6": 0.03,
        }
        for name, value in values.items():
            getattr(ch, name).value = jnp.array([value])
        states = ch.state_values()
        ch.compute_derivative(V, na)

        expected_dL3 = (
                           states["C3"] * ch.fl3(V)
                           + states["L4"] * ch.b33(V)
                           - states["L3"] * ch.bl3(V)
                           - states["L3"] * ch.f33(V)
                       ) / u.ms
        expected_dL6 = (
                           states["L5"] * ch.f3n(V)
                           + states["O"] * ch.fl6(V)
                           - states["L6"] * ch.b3n(V)
                           - states["L6"] * ch.bl6(V)
                       ) / u.ms
        self.assertTrue(u.math.allclose(ch.L3.derivative, expected_dL3, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.L6.derivative, expected_dL6, atol=1e-6 * u.Hz))

    def test_reset_steady_state_produces_stationary_distribution(self) -> None:
        ch = NaFHF_MA20_GrC(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        ch.init_state(V, na)
        ch.reset_steady_state(V, na)
        states = ch.state_values()
        total = None
        for name in ch.state_names + (ch.redundant_state,):
            total = states[name] if total is None else (total + states[name])
        self.assertTrue(u.math.allclose(total, jnp.ones(2), atol=1e-6))
        ch.compute_derivative(V, na)
        for name in ch.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(ch, name).derivative,
                    jnp.zeros(2) / u.ms,
                    atol=1e-5 * u.Hz,
                )
            )


if __name__ == "__main__":
    unittest.main()
