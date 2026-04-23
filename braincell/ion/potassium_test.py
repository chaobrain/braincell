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

from braincell._base import HHTypedNeuron, Ion, IonInfo
from braincell.channel.potassium import K_TM1991
from braincell.ion._base import InitNernstIon
from braincell.ion.potassium import Potassium, PotassiumFixed, PotassiumInitNernst


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


class PotassiumBaseTest(unittest.TestCase):
    """Tests for the abstract :class:`Potassium` base class."""

    def test_potassium_is_subclass_of_ion(self) -> None:
        self.assertTrue(issubclass(Potassium, Ion))

    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Potassium.root_type, HHTypedNeuron)

    def test_module_attribute_is_public_namespace(self) -> None:
        self.assertEqual(Potassium.__module__, "braincell.ion")


class PotassiumFixedDefaultsTest(unittest.TestCase):
    """Defaults and parameter storage for :class:`PotassiumFixed`."""

    def test_default_reversal_potential_is_minus_95_mV(self) -> None:
        k = PotassiumFixed(size=1)
        self.assertTrue(u.math.allclose(k.E, -95.0 * u.mV, atol=1e-9 * u.mV))

    def test_default_intracellular_and_extracellular_concentrations(self) -> None:
        k = PotassiumFixed(size=1)
        self.assertTrue(
            u.math.allclose(k.Ci, 54.4 * u.mM, atol=1e-9 * u.mM)
        )
        self.assertTrue(u.math.allclose(k.Co, 2.5 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(k.valence, jnp.ones((1,)), atol=1e-9))

    def test_varshape_matches_size(self) -> None:
        self.assertEqual(PotassiumFixed(size=1).varshape, (1,))
        self.assertEqual(PotassiumFixed(size=5).varshape, (5,))
        self.assertEqual(PotassiumFixed(size=(2, 3)).varshape, (2, 3))

    def test_custom_scalar_parameters_are_honoured(self) -> None:
        k = PotassiumFixed(size=2, E=-80.0 * u.mV, Ci=0.1 * u.mM, Co=3.0 * u.mM, valence=1)
        self.assertTrue(u.math.allclose(k.E, -80.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(k.Ci, 0.1 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(k.Co, 3.0 * u.mM, atol=1e-9 * u.mM))

    def test_callable_parameters_broadcast_across_size(self) -> None:
        k = PotassiumFixed(
            size=3,
            E=lambda shape: jnp.array([-90.0, -95.0, -100.0]) * u.mV,
            Ci=lambda shape: jnp.array([0.04, 0.05, 0.06]) * u.mM,
            Co=lambda shape: jnp.array([2.5, 2.6, 2.7]) * u.mM,
        )
        self.assertEqual(k.E.shape, (3,))
        self.assertEqual(k.Ci.shape, (3,))
        self.assertEqual(k.Co.shape, (3,))
        self.assertTrue(
            u.math.allclose(
                k.E,
                jnp.array([-90.0, -95.0, -100.0]) * u.mV,
                atol=1e-9 * u.mV,
            )
        )
        self.assertTrue(
            u.math.allclose(
                k.Ci,
                jnp.array([0.04, 0.05, 0.06]) * u.mM,
                atol=1e-9 * u.mM,
            )
        )


class PotassiumFixedPackInfoTest(unittest.TestCase):
    def test_pack_info_returns_ion_info(self) -> None:
        k = PotassiumFixed(size=1)
        info = k.pack_info()
        self.assertIsInstance(info, IonInfo)

    def test_pack_info_fields_match_stored_values(self) -> None:
        k = PotassiumFixed(size=1, E=-85.0 * u.mV, Ci=0.02 * u.mM, Co=2.8 * u.mM, valence=1)
        info = k.pack_info()
        self.assertTrue(u.math.allclose(info.Ci, 0.02 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(info.Co, 2.8 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(info.E, -85.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(info.valence, jnp.ones((1,)), atol=1e-9))


class PotassiumFixedContainerTest(unittest.TestCase):
    """Ion-as-container behaviour."""

    def test_no_channels_by_default(self) -> None:
        k = PotassiumFixed(size=1)
        self.assertEqual(k.channels, {})
        self.assertEqual(k.external_currents, {})

    def test_channels_kwarg_is_attached(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        self.assertIn("IK", k.channels)
        self.assertIsInstance(k.channels["IK"], K_TM1991)

    def test_current_without_channels_returns_none(self) -> None:
        k = PotassiumFixed(size=1)
        self.assertIsNone(k.current(_V([-60.0])))

    def test_current_with_channel_delegates_to_channel(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        i = k.current(V)
        self.assertEqual(i.shape, (1,))

    def test_register_external_current_rejects_duplicate_keys(self) -> None:
        k = PotassiumFixed(size=1)

        def fake(V, info):
            return jnp.array([1.0]) * u.uA / u.cm ** 2

        k.register_external_current("ext", fake)
        self.assertIn("ext", k.external_currents)
        with self.assertRaises(ValueError):
            k.register_external_current("ext", fake)

    def test_current_with_include_external_adds_registered_fn(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        base = k.current(V)
        delta = jnp.array([2.5]) * u.uA / u.cm ** 2

        def external(V_local, info):
            return delta

        k.register_external_current("ext", external)
        total = k.current(V, include_external=True)
        got = (total - base).to_decimal(u.uA / u.cm ** 2)
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
        self.assertTrue(
            u.math.allclose(k.channels["IK"].p.value, first, atol=1e-9)
        )

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
        expected = (
            u.gas_constant * k.temp / (k.valence * u.faraday_constant)
            * u.math.log(k.Co / k.Ci)
        )
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


if __name__ == "__main__":
    unittest.main()
