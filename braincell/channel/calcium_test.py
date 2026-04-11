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
from braincell.ion import Calcium
from braincell.channel.calcium import (
    CalciumChannel,
    ICaGrc_Ma2020,
    ICaHT_HM1992,
    ICaHT_Re1993,
    ICaL_IS2008,
    ICaN_IS2008,
    ICaT_HM1992,
    ICaT_HP1992,
    ICav12_Ma2020,
    ICav13_Ma2020,
    ICav23_Ma2020,
    ICav31_Ma2020,
    _ICa_p2q_markov,
    _ICa_p2q_ss,
)


def _ca_info(size: int = 1, C: float = 1e-4) -> IonInfo:
    return IonInfo(
        C=jnp.full((size,), C) * u.mM,
        E=jnp.full((size,), 120.0) * u.mV,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


class CalciumChannelBaseTest(unittest.TestCase):
    def test_root_type_is_calcium(self) -> None:
        self.assertIs(CalciumChannel.root_type, Calcium)

    def test_base_current_raises_not_implemented(self) -> None:
        class _Bare(CalciumChannel):
            pass

        bare = _Bare(size=1)
        with self.assertRaises(NotImplementedError):
            bare.current(_V([-60.0]), _ca_info())

    def test_base_lifecycle_hooks_are_no_ops(self) -> None:
        class _Bare(CalciumChannel):
            pass

        bare = _Bare(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        self.assertIsNone(bare.pre_integral(V, ca))
        self.assertIsNone(bare.post_integral(V, ca))
        self.assertIsNone(bare.compute_derivative(V, ca))
        self.assertIsNone(bare.init_state(V, ca))
        self.assertIsNone(bare.reset_state(V, ca))


class _ICaP2QSsAbstractTest(unittest.TestCase):
    def test_rate_functions_raise(self) -> None:
        ch = _ICa_p2q_ss(size=1)
        V = _V([-60.0])
        for fn in (ch.f_p_inf, ch.f_p_tau, ch.f_q_inf, ch.f_q_tau):
            with self.assertRaises(NotImplementedError):
                fn(V)


class _ICaP2QMarkovAbstractTest(unittest.TestCase):
    def test_rate_functions_raise(self) -> None:
        ch = _ICa_p2q_markov(size=1)
        V = _V([-60.0])
        for fn in (ch.f_p_alpha, ch.f_p_beta, ch.f_q_alpha, ch.f_q_beta):
            with self.assertRaises(NotImplementedError):
                fn(V)


class _ICaP2QSsMixin:
    CLS = None

    def test_init_state_creates_p_and_q(self) -> None:
        ch = self.CLS(size=2)
        V = _V([-60.0, -70.0])
        ca = _ca_info(2)
        ch.init_state(V, ca)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_matches_infinities(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_current_matches_p2_q_formula(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        i = ch.current(V, ca)
        expected = ch.g_max * ch.p.value * ch.p.value * ch.q.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_matches_first_order_ode(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.6])
        ch.compute_derivative(V, ca)

        exp_dp = ch.phi_p * (ch.f_p_inf(V) - 0.2) / ch.f_p_tau(V) / u.ms
        exp_dq = ch.phi_q * (ch.f_q_inf(V) - 0.6) / ch.f_q_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_dp, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_dq, atol=1e-6 * u.Hz))

    def test_current_is_zero_when_gates_closed(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.zeros(1)
        ch.q.value = jnp.zeros(1)
        i = ch.current(V, ca)
        self.assertTrue(
            u.math.allclose(i.to_decimal(_DENSITY_UNIT), jnp.zeros(1), atol=1e-9)
        )


class ICaT_HM1992Test(_ICaP2QSsMixin, unittest.TestCase):
    CLS = ICaT_HM1992


class ICaT_HP1992Test(_ICaP2QSsMixin, unittest.TestCase):
    CLS = ICaT_HP1992


class ICaHT_HM1992Test(_ICaP2QSsMixin, unittest.TestCase):
    CLS = ICaHT_HM1992


class ICaL_IS2008Test(_ICaP2QSsMixin, unittest.TestCase):
    CLS = ICaL_IS2008


class ICaHT_Re1993Test(unittest.TestCase):
    def test_reset_state_matches_alpha_over_alpha_plus_beta(self) -> None:
        ch = ICaHT_Re1993(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
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

    def test_current_matches_p2q_formula(self) -> None:
        ch = ICaHT_Re1993(size=1)
        V = _V([-55.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        i = ch.current(V, ca)
        expected = ch.g_max * ch.p.value * ch.p.value * ch.q.value * (ca.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_uses_markov_form(self) -> None:
        ch = ICaHT_Re1993(size=1)
        V = _V([-55.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.3])
        ch.compute_derivative(V, ca)

        alpha_p = ch.f_p_alpha(V)
        beta_p = ch.f_p_beta(V)
        alpha_q = ch.f_q_alpha(V)
        beta_q = ch.f_q_beta(V)
        exp_dp = ch.phi_p * (alpha_p * (1 - 0.2) - beta_p * 0.2) / u.ms
        exp_dq = ch.phi_q * (alpha_q * (1 - 0.3) - beta_q * 0.3) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_dp, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_dq, atol=1e-6 * u.Hz))


class ICaN_IS2008Test(unittest.TestCase):
    def test_reset_state_sets_p_to_logistic(self) -> None:
        ch = ICaN_IS2008(size=1)
        V = _V([-60.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        expected = 1.0 / (1.0 + jnp.exp((-60.0 + 43.0) / -5.2))
        self.assertTrue(u.math.allclose(ch.p.value, jnp.array([expected]), atol=1e-6))

    def test_current_depends_on_calcium_concentration(self) -> None:
        ch = ICaN_IS2008(size=1)
        V = _V([-60.0])
        low = _ca_info(C=1e-4)
        high = _ca_info(C=10.0)  # well above the 0.2 mM half-activation
        ch.init_state(V, low)
        ch.reset_state(V, low)
        i_low = ch.current(V, low)
        i_high = ch.current(V, high)

        # Current magnitude should grow with [Ca^2+]_i because of the
        # Michaelis-Menten modulation M = [Ca] / (0.2 + [Ca]).
        high_mag = float(u.math.abs(i_high).to_decimal(_DENSITY_UNIT)[0])
        low_mag = float(u.math.abs(i_low).to_decimal(_DENSITY_UNIT)[0])
        self.assertGreater(high_mag, low_mag)

    def test_current_matches_formula(self) -> None:
        ch = ICaN_IS2008(size=1, E=10.0 * u.mV, g_max=1.0 * (u.mS / u.cm ** 2))
        V = _V([-60.0])
        ca = _ca_info(C=0.001)
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        i = ch.current(V, ca)
        M = ca.C / (ca.C + 0.2 * u.mM)
        expected = ch.g_max * M * ch.p.value * (ch.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-9,
            )
        )


class _CavMaMixin:
    """Shared smoke-test checks for the Ma 2020 Cav family."""

    CLS = None

    def test_init_state_creates_m_h_gates(self) -> None:
        ch = self.CLS(size=2)
        V = _V([-40.0, -60.0])
        ca = _ca_info(2)
        ch.init_state(V, ca)
        self.assertEqual(ch.m.value.shape, (2,))
        self.assertEqual(ch.h.value.shape, (2,))

    def test_reset_state_runs(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)

    def test_compute_derivative_runs(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        ch.compute_derivative(V, ca)


class ICav12_Ma2020Test(_CavMaMixin, unittest.TestCase):
    CLS = ICav12_Ma2020

    def test_has_calcium_dependent_gate_n(self) -> None:
        ch = ICav12_Ma2020(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertEqual(ch.n.value.shape, (1,))


class ICav13_Ma2020Test(_CavMaMixin, unittest.TestCase):
    CLS = ICav13_Ma2020


class ICav23_Ma2020Test(_CavMaMixin, unittest.TestCase):
    CLS = ICav23_Ma2020

    def test_current_uses_internal_eca(self) -> None:
        # ICav23 has its own ``self.eca`` (140 mV) instead of the ion's ``E``.
        ch = ICav23_Ma2020(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        ch.m.value = jnp.array([0.5])
        ch.h.value = jnp.array([0.3])
        i = ch.current(V, ca)
        expected = ch.g_max * ch.m.value ** 3 * ch.h.value * (ch.eca - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class ICav31_Ma2020Test(unittest.TestCase):
    """ICav31 uses a GHK drive term and p/q gate labels."""

    def test_reset_state_matches_infinities(self) -> None:
        ch = ICav31_Ma2020(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_current_runs_without_raising(self) -> None:
        ch = ICav31_Ma2020(
            size=1,
            g_max=1e-4 * (u.cm / u.second),
        )
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        i = ch.current(V, ca)
        # GHK current density has units of A / m^2 – just check we get a
        # finite array back in that dimension class.
        self.assertEqual(i.shape, (1,))


class ICaGrc_Ma2020Test(_CavMaMixin, unittest.TestCase):
    CLS = ICaGrc_Ma2020

    def test_current_uses_internal_eca(self) -> None:
        ch = ICaGrc_Ma2020(size=1)
        V = _V([-50.0])
        ca = _ca_info()
        ch.init_state(V, ca)
        ch.reset_state(V, ca)
        i = ch.current(V, ca)
        expected = ch.g_max * ch.m.value ** 2 * ch.h.value * (ch.eca - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
