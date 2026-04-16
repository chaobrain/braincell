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
from braincell.ion import Sodium
from braincell.channel.sodium import (
    INa_Ba2002,
    INa_HH1952,
    INa_Rsg,
    INa_TM1991,
    INa_p3q_markov,
    SodiumChannel,
)
from braincell.channel.sodium_v2 import INa_HH1952_v2


def _na_info(size: int = 1) -> IonInfo:
    return IonInfo(
        C=jnp.full((size,), 0.04) * u.mM,
        E=jnp.full((size,), 50.0) * u.mV,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


class SodiumChannelBaseTest(unittest.TestCase):
    def test_root_type_is_sodium(self) -> None:
        self.assertIs(SodiumChannel.root_type, Sodium)

    def test_base_current_raises_not_implemented(self) -> None:
        class _Bare(SodiumChannel):
            pass

        bare = _Bare(size=1)
        with self.assertRaises(NotImplementedError):
            bare.current(_V([-60.0]), _na_info())

    def test_base_lifecycle_hooks_are_no_ops(self) -> None:
        class _Bare(SodiumChannel):
            pass

        bare = _Bare(size=2)
        V = _V([-60.0, -40.0])
        na = _na_info(2)
        self.assertIsNone(bare.pre_integral(V, na))
        self.assertIsNone(bare.post_integral(V, na))
        self.assertIsNone(bare.compute_derivative(V, na))
        self.assertIsNone(bare.init_state(V, na))
        self.assertIsNone(bare.reset_state(V, na))


class INaP3qMarkovAbstractTest(unittest.TestCase):
    """The ``INa_p3q_markov`` parent class must leave the rate functions abstract."""

    def test_rate_functions_raise(self) -> None:
        ch = INa_p3q_markov(size=1)
        V = _V([-60.0])
        for fn in (ch.f_p_alpha, ch.f_p_beta, ch.f_q_alpha, ch.f_q_beta):
            with self.assertRaises(NotImplementedError):
                fn(V)


class _Markov_p3q_Mixin:
    """Shared validation for every concrete ``INa_p3q_markov`` subclass."""

    CLS = None  # set on subclasses

    def _make(self, size=1, **kwargs):
        return self.CLS(size=size, **kwargs)

    def test_default_gmax_has_current_density_units(self) -> None:
        ch = self._make(size=1)
        # All defaults are in mS / cm^2 – retrieving the mantissa must succeed.
        _ = ch.g_max.to_decimal(u.mS / u.cm ** 2)

    def test_init_state_creates_gates_shaped_to_size(self) -> None:
        ch = self._make(size=4)
        V = _V([-60.0, -55.0, -50.0, -45.0])
        na = _na_info(4)

        ch.init_state(V, na)
        self.assertEqual(ch.p.value.shape, (4,))
        self.assertEqual(ch.q.value.shape, (4,))
        # Initial state: p and q are zeros until ``reset_state`` runs.
        self.assertTrue(u.math.allclose(ch.p.value, jnp.zeros(4)))
        self.assertTrue(u.math.allclose(ch.q.value, jnp.zeros(4)))

    def test_reset_state_sets_gates_to_steady_state(self) -> None:
        ch = self._make(size=1)
        V = _V([-65.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)

        # Steady state must match alpha / (alpha + beta).
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
        # Physical sanity: gating variables are in [0, 1].
        self.assertTrue(bool((ch.p.value >= 0).all() and (ch.p.value <= 1).all()))
        self.assertTrue(bool((ch.q.value >= 0).all() and (ch.q.value <= 1).all()))

    def test_compute_derivative_matches_hh_equation(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        # Drive the state off steady-state so the derivative is nonzero.
        ch.p.value = jnp.array([0.1])
        ch.q.value = jnp.array([0.9])
        ch.compute_derivative(V, na)

        alpha_p = ch.f_p_alpha(V)
        beta_p = ch.f_p_beta(V)
        alpha_q = ch.f_q_alpha(V)
        beta_q = ch.f_q_beta(V)

        expected_dp = ch.phi * (alpha_p * (1 - ch.p.value) - beta_p * ch.p.value) / u.ms
        expected_dq = ch.phi * (alpha_q * (1 - ch.q.value) - beta_q * ch.q.value) / u.ms

        self.assertTrue(
            u.math.allclose(ch.p.derivative, expected_dp, atol=1e-6 * u.Hz)
        )
        self.assertTrue(
            u.math.allclose(ch.q.derivative, expected_dq, atol=1e-6 * u.Hz)
        )

    def test_current_matches_p3q_g_formula(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)

        i = ch.current(V, na)
        expected = ch.g_max * ch.p.value ** 3 * ch.q.value * (na.E - V)
        # Compare in a common current-density unit.
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(unit), expected.to_decimal(unit), atol=1e-6
            )
        )

    def test_current_is_zero_when_gates_closed(self) -> None:
        ch = self._make(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        # p = q = 0 ⇒ current = 0.
        ch.p.value = jnp.zeros(1)
        ch.q.value = jnp.zeros(1)
        i = ch.current(V, na)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(u.mS / u.cm ** 2 * u.mV),
                jnp.zeros(1),
                atol=1e-9,
            )
        )


class INa_Ba2002Test(_Markov_p3q_Mixin, unittest.TestCase):
    CLS = INa_Ba2002

    def test_temperature_factor_is_derived_from_celsius(self) -> None:
        ch = INa_Ba2002(size=1, T=u.celsius2kelvin(36.0))
        # phi = 3 ** ((T - 36) / 10) = 1.0 at 36 C.
        self.assertTrue(u.math.allclose(ch.phi, jnp.ones(1), atol=1e-6))

        warm = INa_Ba2002(size=1, T=u.celsius2kelvin(46.0))
        self.assertTrue(u.math.allclose(warm.phi, 3.0 * jnp.ones(1), atol=1e-6))

    def test_V_sh_shifts_rate_functions(self) -> None:
        ch_a = INa_Ba2002(size=1, V_sh=-50.0 * u.mV)
        ch_b = INa_Ba2002(size=1, V_sh=-60.0 * u.mV)

        # Shifting V_sh by 10 mV is equivalent to evaluating the first channel
        # at V shifted by the same 10 mV.
        V = _V([-55.0])
        V_shifted = _V([-45.0])
        self.assertTrue(
            u.math.allclose(ch_a.f_p_alpha(V_shifted), ch_b.f_p_alpha(V), atol=1e-6)
        )


class INa_TM1991Test(_Markov_p3q_Mixin, unittest.TestCase):
    CLS = INa_TM1991


class INa_HH1952Test(_Markov_p3q_Mixin, unittest.TestCase):
    CLS = INa_HH1952


class INa_HH1952V2Test(unittest.TestCase):
    def test_reset_state_matches_legacy_implementation(self) -> None:
        legacy = INa_HH1952(size=1)
        proto = INa_HH1952_v2(size=1)
        V = _V([-65.0])
        na = _na_info()

        legacy.init_state(V, na)
        proto.init_state(V, na)
        legacy.reset_state(V, na)
        proto.reset_state(V, na)

        self.assertTrue(u.math.allclose(proto.p.value, legacy.p.value, atol=1e-6))
        self.assertTrue(u.math.allclose(proto.q.value, legacy.q.value, atol=1e-6))

    def test_compute_derivative_matches_legacy_implementation(self) -> None:
        legacy = INa_HH1952(size=1)
        proto = INa_HH1952_v2(size=1)
        V = _V([-60.0])
        na = _na_info()

        legacy.init_state(V, na)
        proto.init_state(V, na)
        legacy.reset_state(V, na)
        proto.reset_state(V, na)
        legacy.p.value = jnp.array([0.1])
        legacy.q.value = jnp.array([0.9])
        proto.p.value = jnp.array([0.1])
        proto.q.value = jnp.array([0.9])

        legacy.compute_derivative(V, na)
        proto.compute_derivative(V, na)

        self.assertTrue(u.math.allclose(proto.p.derivative, legacy.p.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(proto.q.derivative, legacy.q.derivative, atol=1e-6 * u.Hz))

    def test_current_matches_legacy_implementation(self) -> None:
        legacy = INa_HH1952(size=1)
        proto = INa_HH1952_v2(size=1)
        V = _V([-60.0])
        na = _na_info()

        legacy.init_state(V, na)
        proto.init_state(V, na)
        legacy.reset_state(V, na)
        proto.reset_state(V, na)

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


class INaRsgTest(unittest.TestCase):
    """``INa_Rsg`` uses an internal 12-state Markov scheme."""

    def test_init_state_creates_all_closed_inactivated_and_open_states(self) -> None:
        ch = INa_Rsg(size=3)
        V = _V([-60.0, -50.0, -40.0])
        na = _na_info(3)
        ch.init_state(V, na)

        # Every declared state must exist with the right shape.
        for name in ["C1", "C2", "C3", "C4", "C5", "I1", "I2", "I3", "I4", "I5", "O", "B"]:
            state = getattr(ch, name)
            self.assertEqual(state.value.shape, (3,))

        self.assertTrue(hasattr(ch, "state_names"))
        self.assertTrue(hasattr(ch, "state_pairs"))
        self.assertEqual(ch.redundant_state, "I6")

    def test_current_is_g_max_times_open_times_driving_force(self) -> None:
        ch = INa_Rsg(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)

        # Force the channel fully into the open state so the current formula
        # reduces to g_max * (E_Na - V).
        ch.O.value = jnp.ones(1)
        i = ch.current(V, na)
        expected = ch.g_max * jnp.ones(1) * (na.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(i.to_decimal(unit), expected.to_decimal(unit), atol=1e-6)
        )

    def test_compute_derivative_runs_and_sets_derivatives_on_every_state(self) -> None:
        ch = INa_Rsg(size=2)
        V = _V([-60.0, -50.0])
        na = _na_info(2)
        ch.init_state(V, na)
        # Set a non-degenerate open-state probability so derivatives are nonzero.
        ch.O.value = jnp.array([0.2, 0.4])
        ch.C1.value = jnp.array([0.4, 0.3])
        ch.compute_derivative(V, na)

        for name in ch.state_names:
            state = getattr(ch, name)
            self.assertEqual(state.derivative.shape, (2,))

    def test_current_returns_zero_when_O_is_zero(self) -> None:
        ch = INa_Rsg(size=1)
        V = _V([-60.0])
        na = _na_info()
        ch.init_state(V, na)
        ch.reset_state(V, na)
        ch.O.value = jnp.zeros(1)
        i = ch.current(V, na)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(u.mS / u.cm ** 2 * u.mV),
                jnp.zeros(1),
                atol=1e-9,
            )
        )


if __name__ == "__main__":
    unittest.main()
