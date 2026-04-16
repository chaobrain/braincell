# -*- coding: utf-8 -*-

import unittest

import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import Channel
from braincell.ion import Calcium
from braincell.ion._template import Constant
from braincell.ion._template import DynamicNernst
from braincell.ion._template import InitNernst
from braincell.ion._template import IonData


class _RecorderChannel(Channel):
    root_type = Calcium

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.last_ion = None

    def init_state(self, V, ion, batch_size=None):
        _ = (V, batch_size)
        self.last_ion = ion

    def reset_state(self, V, ion, batch_size=None):
        _ = (V, batch_size)
        self.last_ion = ion

    def compute_derivative(self, V, ion):
        _ = V
        self.last_ion = ion

    def current(self, V, ion):
        _ = V
        self.last_ion = ion
        return 0.25 * u.mM / u.ms


class _ConstantIon(Constant, Calcium):
    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self.Ci = braintools.init.param(2.0e-4 * u.mM, self.varshape, allow_none=False)
        self.Co = braintools.init.param(2.0 * u.mM, self.varshape, allow_none=False)
        self.E = braintools.init.param(120.0 * u.mV, self.varshape, allow_none=False)
        self.valence = 2


class _InitNernstIon(InitNernst, Calcium):
    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self.Ci = braintools.init.param(2.0e-4 * u.mM, self.varshape, allow_none=False)
        self.Co = braintools.init.param(2.0 * u.mM, self.varshape, allow_none=False)
        self.valence = 2
        self.T = u.celsius2kelvin(36.0)
        self.E = None


class _DynamicNernstIon(DynamicNernst, Calcium):
    ci_initializer = 2.0e-4 * u.mM

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self.Co = braintools.init.param(2.0 * u.mM, self.varshape, allow_none=False)
        self.valence = 2
        self.T = u.celsius2kelvin(36.0)

    def ci_derivative(self, Ci, V, total_current):
        _ = V
        return 0.1 * total_current - 0.05 * Ci / u.ms


class IonTemplateTest(unittest.TestCase):
    def test_ion_data_aliases(self) -> None:
        info = IonData(
            Ci=jnp.array([0.1]) * u.mM,
            Co=jnp.array([2.0]) * u.mM,
            E=jnp.array([120.0]) * u.mV,
            valence=2,
        )
        self.assertTrue(u.math.allclose(info.C, info.Ci, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.C0, info.Co, atol=1e-12 * u.mM))
        self.assertEqual(info.z, info.valence)

    def test_constant_pack_info_and_child_derivative(self) -> None:
        ion = _ConstantIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        info = ion.pack_info()
        self.assertIsInstance(info, IonData)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.Co, ion.Co, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.E, ion.E, atol=1e-9 * u.mV))
        self.assertEqual(info.valence, 2)

        ion.init_state(V)
        ion.reset_state(V)
        ion.compute_derivative(V)
        self.assertIsInstance(ion.channels["probe"].last_ion, IonData)

    def test_init_nernst_only_updates_on_reset(self) -> None:
        ion = _InitNernstIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        first_E = ion.E
        expected = (
            u.gas_constant * ion.T / (ion.valence * u.faraday_constant)
            * u.math.log(ion.Co / ion.Ci)
        )
        self.assertTrue(u.math.allclose(first_E.to_decimal(u.mV), expected.to_decimal(u.mV), atol=1e-6))

        ion.Ci = braintools.init.param(1.0e-3 * u.mM, ion.varshape, allow_none=False)
        self.assertTrue(u.math.allclose(ion.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

        ion.reset_state(V)
        self.assertFalse(u.math.allclose(ion.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

    def test_dynamic_nernst_updates_E_and_Ci_derivative(self) -> None:
        ion = _DynamicNernstIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.reset_state(V)
        first_E = ion.E
        ion.Ci.value = jnp.array([1.0e-3]) * u.mM
        second_E = ion.E
        self.assertFalse(u.math.allclose(first_E.to_decimal(u.mV), second_E.to_decimal(u.mV), atol=1e-6))

        ion.compute_derivative(V)
        expected = 0.1 * (0.25 * u.mM / u.ms) - 0.05 * ion.Ci.value / u.ms
        self.assertTrue(
            u.math.allclose(
                ion.Ci.derivative.to_decimal(u.mM / u.ms),
                expected.to_decimal(u.mM / u.ms),
                atol=1e-6,
            )
        )
        info = ion.channels["probe"].last_ion
        self.assertIsInstance(info, IonData)
        self.assertTrue(u.math.allclose(info.C, ion.Ci.value, atol=1e-12 * u.mM))


if __name__ == "__main__":
    unittest.main()
