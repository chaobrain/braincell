# -*- coding: utf-8 -*-

import unittest

import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import Channel
from braincell._base import IonInfo
from braincell.ion import Calcium
from braincell.ion._base import Conserve
from braincell.ion._base import DynamicNernstIon
from braincell.ion._base import Factor
from braincell.ion._base import FixedIon
from braincell.ion._base import InitNernstIon
from braincell.ion._base import KineticIon
from braincell.ion._base import Reaction
from braincell.ion._base import Source
from braincell.ion._base import Species
from braincell.quad.protocol import DiffEqState


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


class _ConstantIon(Calcium, FixedIon):
    default_Ci = 2.0e-4 * u.mM
    default_Co = 2.0 * u.mM
    default_valence = 2

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_fixed_ion(E=120.0 * u.mV)


class _InitNernstIon(Calcium, InitNernstIon):
    default_Ci = 2.0e-4 * u.mM
    default_Co = 2.0 * u.mM
    default_valence = 2

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_nernst_ion(temp=u.celsius2kelvin(36.0))


class _DynamicNernstIon(Calcium, DynamicNernstIon):
    default_Co = 2.0 * u.mM
    default_valence = 2
    uses_total_current = True

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_dynamic_nernst_ion(
            temp=u.celsius2kelvin(36.0),
            Co=None,
            valence=None,
            Ci_initializer=2.0e-4 * u.mM,
        )

    def derivative(self, Ci, V, total_current=None):
        _ = V
        return 0.1 * total_current - 0.05 * Ci / u.ms


class _DynamicNernstIonNoCurrent(Calcium, DynamicNernstIon):
    default_Co = 2.0 * u.mM
    default_valence = 2
    uses_total_current = False

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.last_total_current = "unset"
        self._init_dynamic_nernst_ion(
            temp=u.celsius2kelvin(36.0),
            Co=None,
            valence=None,
            Ci_initializer=2.0e-4 * u.mM,
        )

    def current(self, V, include_external=False):
        _ = (V, include_external)
        raise AssertionError("current() should not be called when uses_total_current is False.")

    def derivative(self, Ci, V, total_current=None):
        _ = V
        self.last_total_current = total_current
        return -0.05 * Ci / u.ms


class _SimpleKineticIon(Calcium, KineticIon):
    default_Co = 2.0 * u.mM
    default_valence = 2

    factors = (
        Factor("cyto", lambda self: self.cyt_volume),
    )
    species = (
        Species("Ci", init=0.1 * u.mM, factor="cyto"),
        Species("B", init=1.0 * u.mM, factor="cyto"),
        Species("BC", init=0.0 * u.mM, factor="cyto"),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "B": 1},
            rhs={"BC": 1},
            forward=lambda self, V, x: self.kf * self.cyt_volume,
            backward=lambda self, V, x: self.kb * self.cyt_volume,
        ),
    )
    sources = ()
    conserves = (
        Conserve(
            species=("B", "BC"),
            algebraic="B",
            total=lambda self, V, x: self.Btot * self.cyt_volume,
        ),
    )

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            solver="euler",
            substeps=2,
        )
        self.cyt_volume = braintools.init.param(3.0 * u.um ** 3, self.varshape, allow_none=False)
        self.kf = braintools.init.param(2.0 / (u.mM * u.ms), self.varshape, allow_none=False)
        self.kb = braintools.init.param(0.5 / u.ms, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(1.0 * u.mM, self.varshape, allow_none=False)


class IonTemplateTest(unittest.TestCase):
    def test_constant_pack_info_and_child_derivative(self) -> None:
        ion = _ConstantIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.Co, ion.Co, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.E, ion.E, atol=1e-9 * u.mV))
        self.assertEqual(info.valence, 2)

        ion.init_state(V)
        ion.reset_state(V)
        ion.compute_derivative(V)
        self.assertIsInstance(ion.channels["probe"].last_ion, IonInfo)

    def test_species_first_inheritance_keeps_template_hooks_active(self) -> None:
        self.assertLess(_ConstantIon.__mro__.index(Calcium), _ConstantIon.__mro__.index(FixedIon))
        self.assertTrue(hasattr(_InitNernstIon, "_ion_init_state_hook"))
        self.assertTrue(hasattr(_DynamicNernstIon, "_ion_compute_derivative_hook"))

    def test_init_nernst_only_updates_on_reset(self) -> None:
        ion = _InitNernstIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        first_E = ion.E
        expected = (
            u.gas_constant * ion.temp / (ion.valence * u.faraday_constant)
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
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))

    def test_dynamic_nernst_skips_current_when_total_current_not_needed(self) -> None:
        ion = _DynamicNernstIonNoCurrent(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.reset_state(V)
        ion.compute_derivative(V)

        self.assertIsNone(ion.last_total_current)

    def test_kinetic_ion_init_and_reset_write_back_algebraic_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        self.assertIsInstance(ion.Ci, DiffEqState)
        self.assertIsInstance(ion.BC, DiffEqState)
        self.assertFalse(isinstance(ion.B, DiffEqState))
        self.assertTrue(u.math.allclose(ion.B, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.reset_state(V)
        self.assertTrue(u.math.allclose(ion.B, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_species_values_return_resolved_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.B = jnp.array([-999.0]) * u.mM
        values = ion.species_values()

        self.assertTrue(u.math.allclose(values["Ci"], ion.Ci.value, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(values["BC"], ion.BC.value, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(values["B"], jnp.array([0.75]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_compute_derivative_resolves_algebraic_species_first(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.B = jnp.array([-999.0]) * u.mM
        ion.compute_derivative(V)

        expected_flux = ion.cyt_volume * (
            ion.kf * (0.2 * u.mM) * (0.75 * u.mM) - ion.kb * (0.25 * u.mM)
        )
        expected_dCi = -expected_flux / ion.cyt_volume
        expected_dBC = expected_flux / ion.cyt_volume
        self.assertTrue(
            u.math.allclose(
                ion.Ci.derivative.to_decimal(u.mM / u.ms),
                expected_dCi.to_decimal(u.mM / u.ms),
                atol=1e-6,
            )
        )
        self.assertTrue(
            u.math.allclose(
                ion.BC.derivative.to_decimal(u.mM / u.ms),
                expected_dBC.to_decimal(u.mM / u.ms),
                atol=1e-6,
            )
        )

    def test_kinetic_ion_post_integral_refreshes_algebraic_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.BC.value = jnp.array([0.4]) * u.mM
        ion.B = jnp.array([9.9]) * u.mM
        ion.post_integral(V)
        self.assertTrue(u.math.allclose(ion.B, jnp.array([0.6]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_pack_info_uses_Ci_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()

        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))
        expected_E = (
            u.gas_constant * ion.temp / (ion.valence * u.faraday_constant)
        ) * u.math.log(ion.Co / ion.Ci.value)
        self.assertTrue(u.math.allclose(info.E.to_decimal(u.mV), expected_E.to_decimal(u.mV), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
