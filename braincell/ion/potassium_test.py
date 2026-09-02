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
from braincell.channel.potassium import K_TM1991
from braincell.ion._base import InitNernstIon
from braincell.ion._testing import V as _V, FixedIonContractTests
from braincell.ion.potassium import Potassium, PotassiumFixed, PotassiumInitNernst


class PotassiumBaseTest(unittest.TestCase):
    """Tests for the abstract :class:`Potassium` base class."""

    def test_potassium_is_subclass_of_ion(self) -> None:
        self.assertTrue(issubclass(Potassium, Ion))

    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Potassium.root_type, HHTypedNeuron)


class PotassiumFixedDefaultsTest(unittest.TestCase):
    """Defaults and parameter storage for :class:`PotassiumFixed`."""

    def test_custom_scalar_parameters_are_honoured(self) -> None:
        k = PotassiumFixed(size=2, E=-80.0 * u.mV, Ci=0.1 * u.mM, Co=3.0 * u.mM, valence=1)
        self.assertTrue(u.math.allclose(k.E, -80.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(k.Ci, 0.1 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(k.Co, 3.0 * u.mM, atol=1e-9 * u.mM))


class PotassiumFixedContainerTest(unittest.TestCase):
    """Ion-as-container behaviour."""

    def test_channels_kwarg_is_attached(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        self.assertIn("IK", k.channels)
        self.assertIsInstance(k.channels["IK"], K_TM1991)

    def test_current_with_channel_delegates_to_channel(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        i = k.current(V)
        self.assertEqual(i.shape, (1,))

    def test_current_with_include_external_adds_registered_fn(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        base = k.current(V)
        delta = jnp.array([2.5]) * u.uA / u.cm**2

        def external(V_local, info):
            return delta

        k.register_external_current("ext", external)
        total = k.current(V, include_external=True)
        got = (total - base).to_decimal(u.uA / u.cm**2)
        self.assertTrue(u.math.allclose(got, jnp.array([2.5]), atol=1e-9))


class PotassiumFixedLifecycleTest(unittest.TestCase):
    """init_state / reset_state / compute_derivative propagate to children.

    ``PotassiumFixed`` overrides ``reset_state`` with its own implementation
    that performs a hierarchy check plus per-channel ``reset_state`` calls –
    this suite exercises that override specifically.
    """

    def test_init_state_without_children_is_a_no_op(self) -> None:
        k = PotassiumFixed(size=1)
        k.init_state(_V([-60.0]))

    def test_reset_state_without_children_is_a_no_op(self) -> None:
        k = PotassiumFixed(size=1)
        # The override calls check_hierarchies which should succeed even with
        # zero channels.
        k.reset_state(_V([-60.0]))

    def test_init_state_initialises_child_channel_gate(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        ch = k.channels["IK"]
        # ``K_TM1991`` uses a single ``p`` gate for the n^4 formula.
        self.assertEqual(ch.p.value.shape, (1,))

    def test_reset_state_forwards_to_child(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        first = k.channels["IK"].p.value

        # Mutate and reset again – should recover exactly the same state.
        k.channels["IK"].p.value = jnp.array([0.999])
        k.reset_state(V)
        self.assertTrue(u.math.allclose(k.channels["IK"].p.value, first, atol=1e-9))

    def test_compute_derivative_populates_child_derivative(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        k.compute_derivative(V)
        self.assertEqual(k.channels["IK"].p.derivative.shape, (1,))


class PotassiumInitNernstTest(unittest.TestCase):
    def test_is_init_nernst_ion(self) -> None:
        self.assertTrue(issubclass(PotassiumInitNernst, Potassium))
        self.assertTrue(issubclass(PotassiumInitNernst, InitNernstIon))

    def test_E_is_initialized_from_nernst_on_init_and_reset(self) -> None:
        k = PotassiumInitNernst(size=1)
        V = _V([-65.0])

        self.assertIsNone(k.E)
        k.init_state(V)
        expected = u.gas_constant * k.temp / (k.valence * u.faraday_constant) * u.math.log(k.Co / k.Ci)
        self.assertTrue(u.math.allclose(k.E.to_decimal(u.mV), expected.to_decimal(u.mV), atol=1e-6))

        first_E = k.E
        k.Ci = jnp.array([10.0]) * u.mM
        self.assertTrue(u.math.allclose(k.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

        k.reset_state(V)
        self.assertFalse(u.math.allclose(k.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

    def test_custom_Ci_and_Co_are_respected(self) -> None:
        k = PotassiumInitNernst(size=1, Ci=60.0 * u.mM, Co=3.0 * u.mM)
        self.assertTrue(u.math.allclose(k.Ci, 60.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(k.Co, 3.0 * u.mM, atol=1e-9 * u.mM))


class PotassiumFixedContractTest(FixedIonContractTests, unittest.TestCase):
    """The shared ``FixedIon`` contract, for potassium."""

    ION_CLASS = PotassiumFixed
    FAMILY_CLASS = Potassium
    DEFAULT_E = -95.0 * u.mV
    DEFAULT_CI = 54.4 * u.mM
    DEFAULT_CO = 2.5 * u.mM
    DEFAULT_VALENCE = 1


if __name__ == "__main__":
    unittest.main()
