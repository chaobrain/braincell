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
from braincell.ion import Potassium
from braincell.channel.potassium import (
    IK_HH1952,
    IK_Kv_test,
    IK_Leak,
    IK_TM1991,
    IK_p4_markov,
    IKA1_HM1992,
    IKA2_HM1992,
    IKA_p4q_ss,
    IKDR_Ba2002,
    IKK2A_HM1992,
    IKK2B_HM1992,
    IKK2_pq_ss,
    IKM_Grc_Ma2020,
    IKNI_Ya1989,
    IKv11_Ak2007,
    IKv34_Ma2020,
    IKv43_Ma2020,
    PotassiumChannel,
)
from braincell.channel.potassium_v2 import IK_Leak_v2


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        C=jnp.full((size,), 0.04) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


class PotassiumChannelBaseTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(PotassiumChannel.root_type, Potassium)

    def test_base_current_raises_not_implemented(self) -> None:
        class _Bare(PotassiumChannel):
            pass

        bare = _Bare(size=1)
        with self.assertRaises(NotImplementedError):
            bare.current(_V([-60.0]), _k_info())

    def test_base_lifecycle_methods_are_no_ops(self) -> None:
        class _Bare(PotassiumChannel):
            pass

        bare = _Bare(size=1)
        V = _V([-60.0])
        info = _k_info()
        self.assertIsNone(bare.pre_integral(V, info))
        self.assertIsNone(bare.post_integral(V, info))
        self.assertIsNone(bare.compute_derivative(V, info))
        self.assertIsNone(bare.init_state(V, info))
        self.assertIsNone(bare.reset_state(V, info))


class IKp4MarkovAbstractTest(unittest.TestCase):
    def test_rate_functions_raise(self) -> None:
        ch = IK_p4_markov(size=1)
        V = _V([-60.0])
        with self.assertRaises(NotImplementedError):
            ch.f_p_alpha(V)
        with self.assertRaises(NotImplementedError):
            ch.f_p_beta(V)


class _IKMarkovMixin:
    """Validation shared across delayed-rectifier p^4 Markov subclasses."""

    CLS = None

    def test_init_state_creates_p_shaped_to_size(self) -> None:
        ch = self.CLS(size=3)
        V = _V([-60.0, -65.0, -70.0])
        k = _k_info(3)
        ch.init_state(V, k)
        self.assertEqual(ch.p.value.shape, (3,))
        self.assertTrue(u.math.allclose(ch.p.value, jnp.zeros(3)))

    def test_reset_state_matches_steady_state(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-65.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        alpha = ch.f_p_alpha(V)
        beta = ch.f_p_beta(V)
        self.assertTrue(
            u.math.allclose(ch.p.value, alpha / (alpha + beta), atol=1e-6)
        )

    def test_compute_derivative_matches_rate_form(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.compute_derivative(V, k)
        alpha = ch.f_p_alpha(V)
        beta = ch.f_p_beta(V)
        expected = ch.phi * (alpha * (1.0 - 0.2) - beta * 0.2) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_is_g_max_p4_times_driving_force(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.p.value ** 4 * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_current_is_zero_when_p_zero(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.zeros(1)
        i = ch.current(V, k)
        self.assertTrue(
            u.math.allclose(i.to_decimal(_DENSITY_UNIT), jnp.zeros(1), atol=1e-9)
        )


class IKDR_Ba2002Test(_IKMarkovMixin, unittest.TestCase):
    CLS = IKDR_Ba2002

    def test_temperature_factor_default(self) -> None:
        ch = IKDR_Ba2002(size=1, T=u.celsius2kelvin(36.0))
        self.assertTrue(u.math.allclose(ch.phi, jnp.ones(1), atol=1e-6))


class IK_TM1991Test(_IKMarkovMixin, unittest.TestCase):
    CLS = IK_TM1991


class IK_HH1952Test(_IKMarkovMixin, unittest.TestCase):
    CLS = IK_HH1952


class IKAp4qAbstractTest(unittest.TestCase):
    def test_rate_functions_raise(self) -> None:
        ch = IKA_p4q_ss(size=1)
        V = _V([-60.0])
        for fn in (ch.f_p_inf, ch.f_p_tau, ch.f_q_inf, ch.f_q_tau):
            with self.assertRaises(NotImplementedError):
                fn(V)


class _IKAp4qMixin:
    CLS = None

    def test_init_state_creates_p_and_q(self) -> None:
        ch = self.CLS(size=2)
        V = _V([-60.0, -70.0])
        k = _k_info(2)
        ch.init_state(V, k)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_sets_gates_to_inf(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_current_matches_p4_q_formula(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.p.value ** 4 * ch.q.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_respects_tau(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.p.value = jnp.array([0.2])
        ch.q.value = jnp.array([0.5])
        ch.compute_derivative(V, k)

        exp_dp = ch.phi_p * (ch.f_p_inf(V) - 0.2) / ch.f_p_tau(V) / u.ms
        exp_dq = ch.phi_q * (ch.f_q_inf(V) - 0.5) / ch.f_q_tau(V) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, exp_dp, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.q.derivative, exp_dq, atol=1e-6 * u.Hz))


class IKA1_HM1992Test(_IKAp4qMixin, unittest.TestCase):
    CLS = IKA1_HM1992


class IKA2_HM1992Test(_IKAp4qMixin, unittest.TestCase):
    CLS = IKA2_HM1992


class IKK2pqAbstractTest(unittest.TestCase):
    def test_rate_functions_raise(self) -> None:
        ch = IKK2_pq_ss(size=1)
        V = _V([-60.0])
        for fn in (ch.f_p_inf, ch.f_p_tau, ch.f_q_inf, ch.f_q_tau):
            with self.assertRaises(NotImplementedError):
                fn(V)


class _IKK2pqMixin:
    CLS = None

    def test_init_state_creates_p_and_q(self) -> None:
        ch = self.CLS(size=2)
        V = _V([-60.0, -70.0])
        k = _k_info(2)
        ch.init_state(V, k)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_reset_state_matches_infinities(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.q.value, ch.f_q_inf(V), atol=1e-6))

    def test_current_matches_pq_formula(self) -> None:
        ch = self.CLS(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.p.value * ch.q.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class IKK2A_HM1992Test(_IKK2pqMixin, unittest.TestCase):
    CLS = IKK2A_HM1992


class IKK2B_HM1992Test(_IKK2pqMixin, unittest.TestCase):
    CLS = IKK2B_HM1992


class IKNI_Ya1989Test(unittest.TestCase):
    def test_reset_state_matches_p_inf(self) -> None:
        ch = IKNI_Ya1989(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.p.value, ch.f_p_inf(V), atol=1e-6))

    def test_current_matches_formula(self) -> None:
        ch = IKNI_Ya1989(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.p.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class IKLeakTest(unittest.TestCase):
    def test_root_type_is_potassium(self) -> None:
        self.assertIs(IK_Leak.root_type, Potassium)

    def test_current_follows_ohms_law(self) -> None:
        ch = IK_Leak(size=1, g_max=0.005 * (u.mS / u.cm ** 2))
        V = _V([-60.0])
        k = _k_info()
        i = ch.current(V, k)
        expected = ch.g_max * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-9,
            )
        )

    def test_compute_derivative_is_noop(self) -> None:
        ch = IK_Leak(size=1)
        self.assertIsNone(ch.compute_derivative(_V([-60.0]), _k_info()))


class IKLeakV2Test(unittest.TestCase):
    def test_current_matches_legacy_implementation(self) -> None:
        legacy = IK_Leak(size=1, g_max=0.005 * (u.mS / u.cm ** 2))
        proto = IK_Leak_v2(size=1, g_max=0.005 * (u.mS / u.cm ** 2))
        V = _V([-60.0])
        k = _k_info()
        i_legacy = legacy.current(V, k)
        i_proto = proto.current(V, k)
        self.assertTrue(
            u.math.allclose(
                i_proto.to_decimal(_DENSITY_UNIT),
                i_legacy.to_decimal(_DENSITY_UNIT),
                atol=1e-9,
            )
        )

    def test_can_be_driven_by_potassium_ion_payload(self) -> None:
        ch = IK_Leak_v2(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        ch.compute_derivative(V, k)
        i = ch.current(V, k)
        self.assertEqual(i.shape, (1,))


class IKv11_Ak2007Test(unittest.TestCase):
    def test_reset_state_sets_p_to_steady_state(self) -> None:
        ch = IKv11_Ak2007(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        alpha = ch.f_p_alpha(V)
        beta = ch.f_p_beta(V)
        self.assertTrue(
            u.math.allclose(ch.p.value, alpha / (alpha + beta), atol=1e-6)
        )

    def test_current_without_gating_current(self) -> None:
        ch = IKv11_Ak2007(size=1, gateCurrent=0.0)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.p.value ** 4 * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class IKv34_Ma2020Test(unittest.TestCase):
    def test_init_state_creates_p_and_q(self) -> None:
        ch = IKv34_Ma2020(size=2)
        V = _V([-60.0, -40.0])
        k = _k_info(2)
        ch.init_state(V, k)
        self.assertEqual(ch.p.value.shape, (2,))
        self.assertEqual(ch.q.value.shape, (2,))

    def test_current_follows_p3_q_formula(self) -> None:
        ch = IKv34_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.p.value ** 3 * ch.q.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class IKv43_Ma2020Test(unittest.TestCase):
    def test_current_follows_p3_q_formula(self) -> None:
        ch = IKv43_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        expected = ch.g_max * ch.p.value ** 3 * ch.q.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_sigm_returns_logistic(self) -> None:
        ch = IKv43_Ma2020(size=1)
        x = jnp.array([0.0, 2.0, -2.0])
        result = ch.sigm(x, 1.0)
        # sigm(x, y) = 1 / (exp(x/y) + 1)
        expected = 1.0 / (jnp.exp(x) + 1.0)
        self.assertTrue(bool(jnp.allclose(result, expected, atol=1e-6)))


class IKM_Grc_Ma2020Test(unittest.TestCase):
    def test_current_uses_internal_ek(self) -> None:
        ch = IKM_Grc_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        i = ch.current(V, k)
        # Unlike other K channels, IKM_Grc_Ma2020 uses its own ``self.ek``
        # reversal potential, not the K ion info's E.
        expected = ch.g_max * ch.p.value * (ch.ek - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class IK_Kv_testTest(unittest.TestCase):
    def test_reset_state_matches_f_n_inf(self) -> None:
        ch = IK_Kv_test(size=1)
        V = _V([-20.0])
        k = _k_info()
        ch.init_state(V, k)
        ch.reset_state(V, k)
        self.assertTrue(u.math.allclose(ch.n.value, ch.f_n_inf(V), atol=1e-6))

    def test_current_matches_linear_gating(self) -> None:
        ch = IK_Kv_test(size=1, g_max=0.1 * (u.mS / u.cm ** 2))
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


if __name__ == "__main__":
    unittest.main()
