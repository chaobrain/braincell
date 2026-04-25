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

from braincell._base import IonInfo
from braincell.channel._base import HH, Markov
from braincell.channel.sodium import (
    Na_Ba2002,
    Na_HH1952,
    Na_TM1991,
    NaF_SU2015_DCN,
    NaFHF_MA2020_GrC,
    NaP_SU2015_DCN,
    Na_ZH2019_IO,
    Nav1p1_MA2025_BC,
    Nav1p1_RI2021_SC,
    Nav1p6_MA2020_GoC,
    Nav1p6_MA2024_PC,
    Nav1p6_MA2025_BC,
    Nav1p6_RI2021_SC,
    Nav_MA2020_GrC,
)
from braincell.ion import Sodium

brainstate.environ.set(precision=64)


def _na_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 140.0) * u.mM,
        E=jnp.full((size,), 50.0) * u.mV,
        valence=1,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


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


def _manual_markov_derivatives(channel, V, *ions):
    states = channel.state_values()
    derivatives = {
        name: u.math.zeros_like(value)
        for name, value in states.items()
    }

    for src, dst, f_rate, b_rate in channel.state_pairs:
        forward = getattr(channel, f_rate)(V, *ions)
        derivatives[src] = derivatives[src] - states[src] * forward
        derivatives[dst] = derivatives[dst] + states[src] * forward

        if b_rate is not None:
            backward = getattr(channel, b_rate)(V, *ions)
            derivatives[src] = derivatives[src] + states[dst] * backward
            derivatives[dst] = derivatives[dst] - states[dst] * backward

    return {
        name: derivative / u.ms
        for name, derivative in derivatives.items()
    }


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

    def test_temp_matches_q10_formula(self) -> None:
        proto = self._make(size=1, temp=u.celsius2kelvin(30.0))
        expected_phi = 3 ** (((proto.temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)

        self.assertTrue(u.math.allclose(proto.temp, u.celsius2kelvin(30.0), atol=1e-6 * u.kelvin))
        self.assertTrue(u.math.allclose(proto.phi, expected_phi, atol=1e-6))

    def test_current_uses_open_state_only(self) -> None:
        proto = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()

        proto.init_state(V, na)
        proto.reset_state(V, na)
        proto.O.value = jnp.array([0.35])

        current = proto.current(V, na)
        expected = proto.g_max * proto.O.value * (na.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_matches_manual_markov_balance(self) -> None:
        proto = self._make(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        proto.init_state(V, na)
        _seed_states(proto)
        expected = _manual_markov_derivatives(proto, V, na)
        proto.compute_derivative(V, na)

        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).derivative,
                    expected[name],
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

    def test_make_integration_updates_states_and_keeps_them_finite(self) -> None:
        proto = self._make(size=2, solver="euler")
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        proto.init_state(V, na)
        _seed_states(proto)
        before = {
            name: getattr(proto, name).value
            for name in proto.state_names
        }

        with brainstate.environ.context(dt=0.02 * u.ms):
            proto.make_integration(V, na)

        changed = False
        for name in proto.state_names:
            value = getattr(proto, name).value
            self.assertTrue(bool(jnp.all(jnp.isfinite(u.get_magnitude(value)))))
            if not bool(u.math.allclose(value, before[name], atol=1e-9)):
                changed = True
        self.assertTrue(changed)

    def test_substeps_defaults_to_legacy_refinement(self) -> None:
        proto = self._make(size=1)
        self.assertEqual(proto.substeps, 5)

    def test_substeps_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self._make(size=1, substeps=0)


class Nav1p6MA20GoCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_MA2020_GoC


class Nav1p6MA24PCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_MA2024_PC

    def test_matches_goc_implementation_under_same_conditions(self) -> None:
        goc = Nav1p6_MA2020_GoC(size=2)
        pc = Nav1p6_MA2024_PC(size=2)
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
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_pc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Nav1p6MA25BCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_MA2025_BC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p6_MA2025_BC(size=2)
        steady = Nav1p6_MA2025_BC(size=2)

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
        goc = Nav1p6_MA2020_GoC(size=2)
        bc = Nav1p6_MA2025_BC(size=2)
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
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_bc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Nav1p6RI21SCTest(_Nav1p6Mixin, unittest.TestCase):
    CLS = Nav1p6_RI2021_SC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p6_RI2021_SC(size=2)
        steady = Nav1p6_RI2021_SC(size=2)

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
        goc = Nav1p6_MA2020_GoC(size=2)
        sc = Nav1p6_RI2021_SC(size=2)
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
        self.assertTrue(
            u.math.allclose(
                i_goc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
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

    def test_compute_derivative_matches_manual_markov_balance(self) -> None:
        proto = self._make(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        proto.init_state(V, na)
        _seed_states(proto)
        expected = _manual_markov_derivatives(proto, V, na)
        proto.compute_derivative(V, na)
        for name in proto.state_names:
            self.assertTrue(
                u.math.allclose(
                    getattr(proto, name).derivative,
                    expected[name],
                    atol=1e-6 * u.Hz,
                )
            )

    def test_current_uses_open_state_when_gate_current_is_off(self) -> None:
        proto = self._make(size=1, gateCurrent=0.0)
        V = _V([-60.0])
        na = _na_info()

        proto.init_state(V, na)
        proto.reset_state(V, na)
        proto.O.value = jnp.array([0.35])

        i_proto = proto.current(V, na)
        expected = proto.g_max * proto.O.value * (na.E - V)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
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
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
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

    def test_make_integration_updates_states_and_keeps_them_finite(self) -> None:
        proto = self._make(size=2, solver="euler", gateCurrent=0.0)
        V = _V([-60.0, -50.0])
        na = _na_info(2)

        proto.init_state(V, na)
        _seed_states(proto)
        before = {
            name: getattr(proto, name).value
            for name in proto.state_names
        }

        with brainstate.environ.context(dt=0.02 * u.ms):
            proto.make_integration(V, na)

        changed = False
        for name in proto.state_names:
            value = getattr(proto, name).value
            self.assertTrue(bool(jnp.all(jnp.isfinite(u.get_magnitude(value)))))
            if not bool(u.math.allclose(value, before[name], atol=1e-9)):
                changed = True
        self.assertTrue(changed)


class Nav1p1MA25BCTest(_Nav1p1Mixin, unittest.TestCase):
    CLS = Nav1p1_MA2025_BC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p1_MA2025_BC(size=2)
        steady = Nav1p1_MA2025_BC(size=2)

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
    CLS = Nav1p1_RI2021_SC

    def test_reset_state_uses_steady_state_initialization(self) -> None:
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        reset = Nav1p1_RI2021_SC(size=2)
        steady = Nav1p1_RI2021_SC(size=2)

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
        bc = Nav1p1_MA2025_BC(size=2)
        sc = Nav1p1_RI2021_SC(size=2)
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
        self.assertTrue(
            u.math.allclose(
                i_bc.to_decimal(_DENSITY_UNIT),
                i_sc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class NavMA20GrCTest(unittest.TestCase):
    def test_root_type_and_defaults(self) -> None:
        ch = Nav_MA2020_GrC(size=1)
        self.assertIs(Nav_MA2020_GrC.root_type, Sodium)
        expected = 13.0 * (u.mS / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                expected.to_decimal(u.mS / u.cm ** 2),
                atol=1e-12,
            )
        )

    def test_init_state_and_hidden_state_layout(self) -> None:
        ch = Nav_MA2020_GrC(size=2)
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
        ch = Nav_MA2020_GrC(size=1)
        V = _V([-60.0])
        factor = 3 ** (((ch.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)
        alfa = factor * ch.Aalfa * u.math.exp((V / u.mV) / ch.Valfa)
        beta = factor * ch.Abeta * u.math.exp(-(V / u.mV) / ch.Vbeta)
        teta = factor * ch.Ateta * u.math.exp(-(V / u.mV) / ch.Vteta)
        self.assertTrue(u.math.allclose(ch.f01(V), ch.n1 * alfa, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.b01(V), ch.n4 * beta, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.bip(V), teta, atol=1e-6))

    def test_current_uses_open_state_only(self) -> None:
        ch = Nav_MA2020_GrC(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.O.value = jnp.array([0.2])
        ch.OB.value = jnp.array([0.7])
        current = ch.current(V, na)
        expected = ch.g_max * ch.O.value * (na.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_selected_derivatives_match_manual_markov_balance(self) -> None:
        ch = Nav_MA2020_GrC(size=1)
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
        ch = Nav_MA2020_GrC(size=2)
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
        ch = NaFHF_MA2020_GrC(size=1)
        self.assertIs(NaFHF_MA2020_GrC.root_type, Sodium)
        expected = 13.0 * (u.mS / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                expected.to_decimal(u.mS / u.cm ** 2),
                atol=1e-12,
            )
        )

    def test_init_state_and_hidden_state_layout(self) -> None:
        ch = NaFHF_MA2020_GrC(size=2)
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
        ch = NaFHF_MA2020_GrC(size=1)
        V = _V([-60.0])
        factor = 3 ** (((ch.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)
        alfa = factor * ch.Aalfa * u.math.exp((V / u.mV) / ch.Valfa)
        self.assertTrue(u.math.allclose(ch.f33(V), ch.n3 * alfa * ch.c, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.b33(V), ch.n2 * alfa * ch.d, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.fl6(V), factor * ch.ALon * ch.c ** 2, atol=1e-6))
        self.assertTrue(u.math.allclose(ch.bl6(V), factor * ch.ALoff * ch.d ** 2, atol=1e-6))

    def test_current_uses_open_state_only(self) -> None:
        ch = NaFHF_MA2020_GrC(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.O.value = jnp.array([0.2])
        ch.OB.value = jnp.array([0.5])
        ch.L6.value = jnp.array([0.2])
        current = ch.current(V, na)
        expected = ch.g_max * ch.O.value * (na.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_selected_long_inactivation_derivatives_match_manual_balance(self) -> None:
        ch = NaFHF_MA2020_GrC(size=1)
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
        ch = NaFHF_MA2020_GrC(size=2)
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
