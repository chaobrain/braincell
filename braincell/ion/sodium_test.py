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

from braincell._base_ion import Ion
from braincell._base_neuron import HHTypedNeuron
from braincell.channel.sodium import Na_TM1991
from braincell.ion._base import InitNernstIon
from braincell.ion._testing import V as _V, FixedIonContractTests
from braincell.ion.sodium import Sodium, SodiumFixed, SodiumInitNernst


class SodiumBaseTest(unittest.TestCase):
    """Tests for the abstract :class:`Sodium` base class."""

    def test_sodium_is_subclass_of_ion(self) -> None:
        self.assertTrue(issubclass(Sodium, Ion))

    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Sodium.root_type, HHTypedNeuron)


class SodiumFixedDefaultsTest(unittest.TestCase):
    """Defaults and parameter storage for :class:`SodiumFixed`."""

    def test_custom_scalar_parameters_are_honoured(self) -> None:
        na = SodiumFixed(size=3, E=40.0 * u.mV, Ci=0.5 * u.mM, Co=100.0 * u.mM, valence=1)
        self.assertTrue(u.math.allclose(na.E, 40.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(na.Ci, 0.5 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(na.Co, 100.0 * u.mM, atol=1e-9 * u.mM))


class SodiumFixedContainerTest(unittest.TestCase):
    """Ion-as-container behaviour: child channels are managed correctly."""

    def test_channels_kwarg_is_attached(self) -> None:
        na = SodiumFixed(size=1, INa=Na_TM1991(size=1))
        self.assertIn("INa", na.channels)
        self.assertIsInstance(na.channels["INa"], Na_TM1991)

    def test_current_with_channel_delegates_to_channel(self) -> None:
        na = SodiumFixed(size=1, INa=Na_TM1991(size=1))
        V = _V([-60.0])
        na.init_state(V)
        na.reset_state(V)
        i = na.current(V)
        # Channel.current returns g_max * m^3 * h * (E - V). We just require
        # the call to succeed and return a quantity of the expected shape.
        self.assertEqual(i.shape, (1,))

    def test_current_with_include_external_adds_registered_fn(self) -> None:
        na = SodiumFixed(size=1, INa=Na_TM1991(size=1))
        V = _V([-60.0])
        na.init_state(V)
        na.reset_state(V)
        base = na.current(V)

        delta = jnp.array([1.5]) * u.uA / u.cm**2

        def external(V_local, info):
            return delta

        na.register_external_current("ext", external)
        total = na.current(V, include_external=True)
        got = (total - base).to_decimal(u.uA / u.cm**2)
        self.assertTrue(u.math.allclose(got, jnp.array([1.5]), atol=1e-9))


class SodiumFixedLifecycleTest(unittest.TestCase):
    """init_state / reset_state / compute_derivative propagate to children."""

    def test_init_state_without_children_is_a_no_op(self) -> None:
        na = SodiumFixed(size=1)
        V = _V([-60.0])
        # Should simply succeed without raising even with no channels.
        na.init_state(V)

    def test_init_state_initialises_child_channel_gates(self) -> None:
        na = SodiumFixed(size=1, INa=Na_TM1991(size=1))
        V = _V([-60.0])
        na.init_state(V)
        ch = na.channels["INa"]
        # After init_state the m/h gate states must exist and have the right
        # shape (``Na_TM1991`` uses p (= m) and q (= h) attributes).
        self.assertEqual(ch.p.value.shape, (1,))
        self.assertEqual(ch.q.value.shape, (1,))

    def test_reset_state_is_idempotent(self) -> None:
        na = SodiumFixed(size=1, INa=Na_TM1991(size=1))
        V = _V([-60.0])
        na.init_state(V)
        na.reset_state(V)
        p_first = na.channels["INa"].p.value
        q_first = na.channels["INa"].q.value

        # Mutate then reset again – should recover the same steady state.
        na.channels["INa"].p.value = jnp.array([0.999])
        na.channels["INa"].q.value = jnp.array([0.001])
        na.reset_state(V)
        self.assertTrue(u.math.allclose(na.channels["INa"].p.value, p_first, atol=1e-9))
        self.assertTrue(u.math.allclose(na.channels["INa"].q.value, q_first, atol=1e-9))

    def test_compute_derivative_populates_child_derivatives(self) -> None:
        na = SodiumFixed(size=1, INa=Na_TM1991(size=1))
        V = _V([-60.0])
        na.init_state(V)
        na.reset_state(V)
        na.compute_derivative(V)
        # ``p`` and ``q`` are DiffEqStates; their ``derivative`` must now have
        # been assigned by the child channel.
        ch = na.channels["INa"]
        self.assertEqual(ch.p.derivative.shape, (1,))
        self.assertEqual(ch.q.derivative.shape, (1,))


class SodiumInitNernstTest(unittest.TestCase):
    def test_is_init_nernst_ion(self) -> None:
        self.assertTrue(issubclass(SodiumInitNernst, Sodium))
        self.assertTrue(issubclass(SodiumInitNernst, InitNernstIon))

    def test_E_is_initialized_from_nernst_on_init_and_reset(self) -> None:
        na = SodiumInitNernst(size=1)
        V = _V([-65.0])

        self.assertIsNone(na.E)
        na.init_state(V)
        expected = u.gas_constant * na.temp / (na.valence * u.faraday_constant) * u.math.log(na.Co / na.Ci)
        self.assertTrue(u.math.allclose(na.E.to_decimal(u.mV), expected.to_decimal(u.mV), atol=1e-6))

        first_E = na.E
        na.Ci = jnp.array([20.0]) * u.mM
        self.assertTrue(u.math.allclose(na.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

        na.reset_state(V)
        self.assertFalse(u.math.allclose(na.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

    def test_custom_Ci_and_Co_are_respected(self) -> None:
        na = SodiumInitNernst(size=1, Ci=15.0 * u.mM, Co=130.0 * u.mM)
        self.assertTrue(u.math.allclose(na.Ci, 15.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(na.Co, 130.0 * u.mM, atol=1e-9 * u.mM))


class SodiumFixedContractTest(FixedIonContractTests, unittest.TestCase):
    """The shared ``FixedIon`` contract, for sodium."""

    ION_CLASS = SodiumFixed
    FAMILY_CLASS = Sodium
    DEFAULT_E = 50.0 * u.mV
    DEFAULT_CI = 10.0 * u.mM
    DEFAULT_CO = 140.0 * u.mM
    DEFAULT_VALENCE = 1


if __name__ == "__main__":
    unittest.main()
