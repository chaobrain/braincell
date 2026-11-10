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
import warnings

import brainunit as u
import jax
import jax.numpy as jnp

from braincell._base_neuron import HHTypedNeuron
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
                ch.g_max.to_decimal(u.mS / u.cm**2),
                jnp.array([0.1]),
            )
        )
        self.assertTrue(u.math.allclose(ch.E.to_decimal(u.mV), jnp.array([-70.0])))

    def test_custom_parameters_are_honoured(self) -> None:
        ch = IL(
            size=3,
            g_max=0.25 * (u.mS / u.cm**2),
            E=-55.0 * u.mV,
        )
        self.assertEqual(ch.varshape, (3,))
        self.assertTrue(
            u.math.allclose(
                ch.g_max.to_decimal(u.mS / u.cm**2),
                jnp.full((3,), 0.25),
            )
        )
        self.assertTrue(u.math.allclose(ch.E.to_decimal(u.mV), jnp.full((3,), -55.0)))

    def test_current_follows_ohms_law(self) -> None:
        g_max = 0.1 * (u.mS / u.cm**2)
        E = -70.0 * u.mV
        ch = IL(size=1, g_max=g_max, E=E)

        V = jnp.array([-60.0]) * u.mV
        i = ch.current(V)

        # I = g (E - V) = 0.1 mS/cm^2 * (-70 - (-60)) mV = -1 mS·mV/cm^2.
        expected = jnp.array([-1.0])
        got = i.to_decimal(u.mS * u.mV / u.cm**2)
        self.assertTrue(u.math.allclose(got, expected, atol=1e-6))

    def test_current_is_zero_at_reversal_potential(self) -> None:
        E = -70.0 * u.mV
        ch = IL(size=1, g_max=0.1 * (u.mS / u.cm**2), E=E)

        i = ch.current(jnp.array([-70.0]) * u.mV)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(u.mS * u.mV / u.cm**2),
                jnp.array([0.0]),
                atol=1e-9,
            )
        )

    def test_current_sign_flips_around_reversal_potential(self) -> None:
        ch = IL(size=2, g_max=0.1 * (u.mS / u.cm**2), E=-70.0 * u.mV)

        V = jnp.array([-80.0, -60.0]) * u.mV  # below / above E
        i = ch.current(V).to_decimal(u.mS * u.mV / u.cm**2)

        # V < E ⇒ current positive (inward of the sign convention used here),
        # V > E ⇒ current negative. Exact signs: g*(E-V).
        self.assertGreater(float(i[0]), 0.0)
        self.assertLess(float(i[1]), 0.0)

    def test_current_vectorises_across_compartments(self) -> None:
        ch = IL(size=4, g_max=0.1 * (u.mS / u.cm**2), E=-70.0 * u.mV)
        V = jnp.array([-80.0, -70.0, -60.0, -50.0]) * u.mV
        i = ch.current(V).to_decimal(u.mS * u.mV / u.cm**2)
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


class ILDirectConstructionUnitTest(unittest.TestCase):
    """Parameter units are checked when ``IL`` is constructed directly (MECH-28)."""

    def test_wrong_units_warn_at_the_constructing_line(self) -> None:
        cases = [
            (0.1 * u.mV, ValueError, "has units incompatible with mS / cm\\^2; got 0.1 mV"),
            (0.1, TypeError, "requires a quantity compatible with mS / cm\\^2; got 0.1 \\(no unit\\)"),
        ]
        for g_max, error, problem in cases:
            with self.subTest(g_max=g_max):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    ch = IL(1, g_max=g_max)
                future = [w for w in caught if issubclass(w.category, FutureWarning)]
                self.assertEqual(len(future), 1)
                self.assertEqual(future[0].filename, __file__)
                self.assertRegex(
                    str(future[0].message),
                    f"^Channel 'IL' parameter 'g_max' {problem}\\. "
                    f"This will raise {error.__name__} at construction from braincell 0.4\\.$",
                )
                # The object is built as before: the value is stored unchanged.
                stored = ch.g_max.to_decimal(u.mV) if isinstance(g_max, u.Quantity) else ch.g_max
                self.assertTrue(u.math.allclose(stored, jnp.array([0.1])))

    def test_instance_name_appears_in_the_message(self) -> None:
        with self.assertWarnsRegex(FutureWarning, "^Channel 'leak' \\(IL\\) parameter 'E' "):
            IL(1, E=-70.0 * u.mS, name="leak")

    def test_subclass_warns_once_after_its_outermost_init(self) -> None:
        class Rescaled(IL):
            def __init__(self, size, g_max=0.1 * (u.mS / u.cm**2), E=-70.0 * u.mV, name=None):
                super().__init__(size, g_max=g_max, E=E, name=name)
                if not isinstance(g_max, u.Quantity):
                    self.g_max = g_max * (u.mS / u.cm**2)  # bare numbers are taken as mS/cm^2

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fixed = Rescaled(1, g_max=0.2)
        self.assertTrue(u.math.allclose(fixed.g_max.to_decimal(u.mS / u.cm**2), 0.2))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Rescaled(1, g_max=0.2 * u.mV)
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        self.assertEqual([w.filename for w in future], [__file__])
        self.assertIn("Channel 'Rescaled' parameter 'g_max'", str(future[0].message))

    def test_traced_values_with_units_do_not_warn_inside_jit(self) -> None:
        def build(g, E):
            ch = IL(1, g_max=g * (u.mS / u.cm**2), E=E * u.mV)
            return ch.current(jnp.array([-60.0]) * u.mV).to_decimal(u.mS * u.mV / u.cm**2)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            got = jax.jit(build)(0.1, -70.0)
            jax.grad(lambda g: build(g, -70.0).sum())(0.1)
        self.assertTrue(u.math.allclose(got, jnp.array([-1.0]), atol=1e-6))

    def test_traced_bare_values_warn_inside_jit(self) -> None:
        with self.assertWarnsRegex(FutureWarning, "'g_max' requires a quantity compatible with mS / cm\\^2") as caught:
            jax.jit(lambda g: IL(1, g_max=g).g_max)(0.1)
        self.assertEqual(caught.filename, __file__)


if __name__ == "__main__":
    unittest.main()
