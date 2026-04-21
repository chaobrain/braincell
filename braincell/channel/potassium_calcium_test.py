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
from braincell.ion import Calcium, Potassium
from braincell.channel.potassium_calcium import (
    IAHP_De1994,
    IKca1_1_Ma2020,
    IKca2_2_Ma2020,
    IKca3_1_Ma2020,
    KCaChannel,
)


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 140.0) * u.mM,
        Co=jnp.full((size,), 2.5) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
        valence=1,
    )


def _ca_info(size: int = 1, C: float = 1e-4) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), C) * u.mM,
        Co=jnp.full((size,), 2.0) * u.mM,
        E=jnp.full((size,), 120.0) * u.mV,
        valence=2,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


class KCaChannelBaseTest(unittest.TestCase):
    def test_root_type_is_joint_potassium_calcium(self) -> None:
        root = KCaChannel.root_type
        self.assertIsInstance(root, brainstate.mixin._JointGenericAlias)
        self.assertIn(Potassium, root.__args__)
        self.assertIn(Calcium, root.__args__)

    def test_current_owner_type_is_potassium(self) -> None:
        self.assertIs(KCaChannel.current_owner_type, Potassium)

    def test_base_current_raises_not_implemented(self) -> None:
        class _Bare(KCaChannel):
            pass

        bare = _Bare(size=1)
        with self.assertRaises(NotImplementedError):
            bare.current(_V([-60.0]), _k_info(), _ca_info())

    def test_base_lifecycle_hooks_are_no_ops(self) -> None:
        class _Bare(KCaChannel):
            pass

        bare = _Bare(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        self.assertIsNone(bare.pre_integral(V, k, ca))
        self.assertIsNone(bare.post_integral(V, k, ca))
        self.assertIsNone(bare.compute_derivative(V, k, ca))
        self.assertIsNone(bare.init_state(V, k, ca))
        self.assertIsNone(bare.reset_state(V, k, ca))


class IAHPDe1994Test(unittest.TestCase):
    def test_reset_state_matches_steady_state(self) -> None:
        ch = IAHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)

        C2 = ch.alpha * (ca.Ci / u.mM) ** ch.n
        expected = C2 / (C2 + ch.beta)
        self.assertTrue(u.math.allclose(ch.p.value, expected, atol=1e-6))

    def test_current_matches_p_squared_form(self) -> None:
        ch = IAHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        i = ch.current(V, k, ca)
        expected = ch.g_max * ch.p.value ** 2 * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_uses_first_order_kinetics(self) -> None:
        ch = IAHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.p.value = jnp.array([0.2])
        ch.compute_derivative(V, k, ca)

        C2 = ch.alpha * (ca.Ci / u.mM) ** ch.n
        C3 = C2 + ch.beta
        expected = ch.phi * (C2 / C3 - 0.2) * C3 / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))

    def test_steady_state_increases_with_calcium(self) -> None:
        # p_inf = alpha Ca^n / (alpha Ca^n + beta) is strictly increasing in Ca
        # for any positive n, so a higher calcium concentration must yield a
        # higher steady-state activation.
        ch = IAHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()

        ch.init_state(V, k, _ca_info(C=1e-4))
        ch.reset_state(V, k, _ca_info(C=1e-4))
        p_low = float(ch.p.value[0])

        ch.reset_state(V, k, _ca_info(C=1e-2))
        p_high = float(ch.p.value[0])
        self.assertGreater(p_high, p_low)

    def test_current_is_zero_when_p_zero(self) -> None:
        ch = IAHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        ch.p.value = jnp.zeros(1)
        i = ch.current(V, k, ca)
        self.assertTrue(
            u.math.allclose(i.to_decimal(_DENSITY_UNIT), jnp.zeros(1), atol=1e-9)
        )


class IKca3_1_Ma2020Test(unittest.TestCase):
    def test_reset_state_matches_p_inf(self) -> None:
        ch = IKca3_1_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.p_inf(V, ca), atol=1e-6))

    def test_current_matches_g_times_p_times_drive(self) -> None:
        ch = IKca3_1_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        i = ch.current(V, k, ca)
        expected = ch.g_max * ch.p.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_runs(self) -> None:
        ch = IKca3_1_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        ch.p.value = jnp.array([0.3])
        ch.compute_derivative(V, k, ca)
        self.assertEqual(ch.p.derivative.shape, (1,))


class IKca2_2_Ma2020Test(unittest.TestCase):
    """Smoke tests for the multi-state SK2 model.

    ``compute_derivative`` in the shipped implementation has a latent bug
    (``self.dirc2_t`` is referenced as an attribute instead of being invoked
    as a method), so we only exercise instantiation, initialization, and
    ``current`` here.
    """

    def test_init_state_creates_all_six_microstates(self) -> None:
        ch = IKca2_2_Ma2020(size=2)
        V = _V([-60.0, -50.0])
        k = _k_info(2)
        ca = _ca_info(2)
        ch.init_state(V, k, ca)
        for name in ("C1", "C2", "C3", "C4", "O1", "O2"):
            state = getattr(ch, name)
            self.assertEqual(state.value.shape, (2,))

    def test_states_normalize_to_one(self) -> None:
        ch = IKca2_2_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        # After init_state the six microstates must form a probability
        # distribution (sum to unity).
        total = (
            ch.C1.value
            + ch.C2.value
            + ch.C3.value
            + ch.C4.value
            + ch.O1.value
            + ch.O2.value
        )
        self.assertTrue(u.math.allclose(total, jnp.ones(1), atol=1e-6))

    def test_current_matches_open_states_times_drive(self) -> None:
        ch = IKca2_2_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        # Force the channel fully open so the drive formula can be checked.
        ch.C1.value = jnp.zeros(1)
        ch.C2.value = jnp.zeros(1)
        ch.C3.value = jnp.zeros(1)
        ch.C4.value = jnp.zeros(1)
        ch.O1.value = jnp.ones(1)
        ch.O2.value = jnp.zeros(1)
        i = ch.current(V, k, ca)
        expected = ch.g_max * (ch.O1.value + ch.O2.value) * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class IKca1_1_Ma2020Test(unittest.TestCase):
    """Smoke tests for the BK-type mSlo channel.

    ``compute_derivative`` has a latent bug (uses ``self.C1`` directly where
    the ``DiffEqState`` wrapper's ``.value`` is expected), so we only test
    instantiation, initialisation, normalization, and the ``current``
    formula.
    """

    def test_init_state_creates_five_closed_and_five_open_states(self) -> None:
        ch = IKca1_1_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        for i in range(5):
            self.assertEqual(getattr(ch, f"C{i}").value.shape, (1,))
            self.assertEqual(getattr(ch, f"O{i}").value.shape, (1,))

    def test_states_normalize_to_one(self) -> None:
        ch = IKca1_1_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        total = sum(getattr(ch, f"C{i}").value for i in range(5)) + sum(
            getattr(ch, f"O{i}").value for i in range(5)
        )
        self.assertTrue(u.math.allclose(total, jnp.ones(1), atol=1e-6))

    def test_current_sums_all_open_states(self) -> None:
        ch = IKca1_1_Ma2020(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        for i in range(5):
            setattr(getattr(ch, f"C{i}"), "value", jnp.zeros(1))
            setattr(getattr(ch, f"O{i}"), "value", jnp.full((1,), 0.2))
        i_val = ch.current(V, k, ca)
        expected = ch.g_max * (
            ch.O0.value + ch.O1.value + ch.O2.value + ch.O3.value + ch.O4.value
        ) * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i_val.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
