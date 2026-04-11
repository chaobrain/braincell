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
from braincell.channel.leaky import IL, LeakageChannel


class LeakageChannelBaseTest(unittest.TestCase):
    """Tests for the abstract :class:`LeakageChannel` base class."""

    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(LeakageChannel.root_type, HHTypedNeuron)

    def test_base_current_raises_not_implemented(self) -> None:
        # LeakageChannel is abstract-by-convention – the default ``current``
        # method must raise NotImplementedError so subclasses are forced to
        # override it.
        class _Bare(LeakageChannel):
            pass

        bare = _Bare(size=1)
        with self.assertRaises(NotImplementedError):
            bare.current(-60.0 * u.mV)

    def test_base_lifecycle_methods_are_no_ops(self) -> None:
        # pre_integral / post_integral / compute_derivative / init_state /
        # reset_state must all accept ``V`` and do nothing at the base level.
        class _Bare(LeakageChannel):
            pass

        bare = _Bare(size=1)
        V = jnp.array([-60.0]) * u.mV
        self.assertIsNone(bare.pre_integral(V))
        self.assertIsNone(bare.post_integral(V))
        self.assertIsNone(bare.compute_derivative(V))
        self.assertIsNone(bare.init_state(V))
        self.assertIsNone(bare.reset_state(V))


class ILTest(unittest.TestCase):
    """Tests for the concrete :class:`IL` leakage channel."""

    def test_default_parameters_are_stored_with_units(self) -> None:
        ch = IL(size=1)

        # Defaults: g_max = 0.1 mS/cm^2, E = -70 mV.
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                jnp.array([0.1]),
            )
        )
        self.assertTrue(
            u.math.allclose(ch.E.to_decimal(u.mV), jnp.array([-70.0]))
        )

    def test_custom_parameters_are_honoured(self) -> None:
        ch = IL(
            size=3,
            g_max=0.25 * (u.mS / u.cm ** 2),
            E=-55.0 * u.mV,
        )
        self.assertEqual(ch.varshape, (3,))
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm ** 2),
                jnp.full((3,), 0.25),
            )
        )
        self.assertTrue(
            u.math.allclose(ch.E.to_decimal(u.mV), jnp.full((3,), -55.0))
        )

    def test_current_follows_ohms_law(self) -> None:
        g_max = 0.1 * (u.mS / u.cm ** 2)
        E = -70.0 * u.mV
        ch = IL(size=1, g_max=g_max, E=E)

        V = jnp.array([-60.0]) * u.mV
        i = ch.current(V)

        # I = g (E - V) = 0.1 mS/cm^2 * (-70 - (-60)) mV = -1 mS·mV/cm^2.
        expected = jnp.array([-1.0])
        got = i.to_decimal(u.mS * u.mV / u.cm ** 2)
        self.assertTrue(u.math.allclose(got, expected, atol=1e-6))

    def test_current_is_zero_at_reversal_potential(self) -> None:
        E = -70.0 * u.mV
        ch = IL(size=1, g_max=0.1 * (u.mS / u.cm ** 2), E=E)

        i = ch.current(jnp.array([-70.0]) * u.mV)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(u.mS * u.mV / u.cm ** 2),
                jnp.array([0.0]),
                atol=1e-9,
            )
        )

    def test_current_sign_flips_around_reversal_potential(self) -> None:
        ch = IL(size=2, g_max=0.1 * (u.mS / u.cm ** 2), E=-70.0 * u.mV)

        V = jnp.array([-80.0, -60.0]) * u.mV  # below / above E
        i = ch.current(V).to_decimal(u.mS * u.mV / u.cm ** 2)

        # V < E ⇒ current positive (inward of the sign convention used here),
        # V > E ⇒ current negative. Exact signs: g*(E-V).
        self.assertGreater(float(i[0]), 0.0)
        self.assertLess(float(i[1]), 0.0)

    def test_current_vectorises_across_compartments(self) -> None:
        ch = IL(size=4, g_max=0.1 * (u.mS / u.cm ** 2), E=-70.0 * u.mV)
        V = jnp.array([-80.0, -70.0, -60.0, -50.0]) * u.mV
        i = ch.current(V).to_decimal(u.mS * u.mV / u.cm ** 2)
        expected = 0.1 * (jnp.array([-70.0, -70.0, -70.0, -70.0]) - jnp.array([-80.0, -70.0, -60.0, -50.0]))
        self.assertTrue(u.math.allclose(i, expected, atol=1e-6))

    def test_lifecycle_hooks_are_no_ops(self) -> None:
        ch = IL(size=1)
        V = jnp.array([-60.0]) * u.mV
        # None of these methods should raise or mutate the channel.
        self.assertIsNone(ch.pre_integral(V))
        self.assertIsNone(ch.post_integral(V))
        self.assertIsNone(ch.compute_derivative(V))
        self.assertIsNone(ch.init_state(V))
        self.assertIsNone(ch.reset_state(V))


if __name__ == "__main__":
    unittest.main()
