# -*- coding: utf-8 -*-

import unittest

import brainstate
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

    def __init__(self, size=1, species_initializers=None):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            species_initializers=species_initializers,
            solver="euler",
            substeps=2,
        )
        self.cyt_volume = braintools.init.param(3.0 * u.um ** 3, self.varshape, allow_none=False)
        self.kf = braintools.init.param(2.0 / (u.mM * u.ms), self.varshape, allow_none=False)
        self.kb = braintools.init.param(0.5 / u.ms, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(1.0 * u.mM, self.varshape, allow_none=False)


class _StoichKineticIon(Calcium, KineticIon):
    default_Co = 2.0 * u.mM
    default_valence = 2

    species = (
        Species("Ci", init=0.1 * u.mM),
        Species("A", init=1.0 * u.mM),
        Species("B", init=1.0 * u.mM),
        Species("C", init=0.0 * u.mM),
    )
    reactions = (
        Reaction(
            lhs={"A": 3, "B": 1},
            rhs={"C": 1},
            forward=lambda self, V, x: self.kf,
            backward=lambda self, V, x: self.kb,
        ),
    )
    sources = ()
    conserves = ()

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            solver="euler",
            substeps=1,
        )
        self.kf = braintools.init.param(2.0 / (u.mM ** 3 * u.ms), self.varshape, allow_none=False)
        self.kb = braintools.init.param(0.5 / u.ms, self.varshape, allow_none=False)


class _UnitMismatchKineticIon(Calcium, KineticIon):
    default_Co = 2.0 * u.mM
    default_valence = 2

    factors = (
        Factor("cyto", lambda self: self.cyt_volume),
    )
    species = (
        Species("Ci", init=0.1 * u.mM, factor="cyto"),
        Species("B", init=1.0 * u.mM),
        Species("BC", init=0.0 * u.mM),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "B": 1},
            rhs={"BC": 1},
            forward=lambda self, V, x: self.kf,
            backward=lambda self, V, x: self.kb,
        ),
    )
    sources = ()
    conserves = ()

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            solver="euler",
            substeps=1,
        )
        self.cyt_volume = braintools.init.param(3.0 * u.um ** 3, self.varshape, allow_none=False)
        self.kf = braintools.init.param(2.0 / (u.mM * u.ms), self.varshape, allow_none=False)
        self.kb = braintools.init.param(0.5 / u.ms, self.varshape, allow_none=False)


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
        self.assertIsInstance(ion.B, brainstate.HiddenState)
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.reset_state(V)
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_species_values_return_resolved_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.B.value = jnp.array([-999.0]) * u.mM
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
        ion.B.value = jnp.array([-999.0]) * u.mM
        ion.compute_derivative(V)

        expected_visible_flux = (2.0 * 0.2 * 0.75 - 0.5 * 0.25)
        self.assertTrue(
            u.math.allclose(
                ion.Ci.derivative.to_decimal(u.mM / u.ms),
                jnp.array([-expected_visible_flux]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            u.math.allclose(
                ion.BC.derivative.to_decimal(u.mM / u.ms),
                jnp.array([expected_visible_flux]),
                atol=1e-6,
            )
        )

    def test_kinetic_ion_post_integral_refreshes_algebraic_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.BC.value = jnp.array([0.4]) * u.mM
        ion.B.value = jnp.array([9.9]) * u.mM
        ion.post_integral(V)
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.6]) * u.mM, atol=1e-12 * u.mM))

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

    def test_kinetic_ion_species_initializers_override_declared_inits(self) -> None:
        ion = _SimpleKineticIon(
            size=1,
            species_initializers={
                "Ci": 0.2 * u.mM,
                "BC": 0.3 * u.mM,
            },
        )
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.2]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_reset_restores_species_initializer_overrides(self) -> None:
        ion = _SimpleKineticIon(
            size=1,
            species_initializers={
                "Ci": 0.2 * u.mM,
                "BC": 0.3 * u.mM,
            },
        )
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.Ci.value = jnp.array([0.9]) * u.mM
        ion.BC.value = jnp.array([0.8]) * u.mM
        ion.B.value = jnp.array([-99.0]) * u.mM

        ion.reset_state(V)

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.2]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_factor_crossing_amount_units_can_match_between_volume_and_area_species(self) -> None:
        cyto_amount = (1.0 * u.mM) * (3.0 * u.um ** 3)
        pump_amount = (1.0 * u.mM * u.um) * (3.0 * u.um ** 2)
        converted = pump_amount.in_unit(cyto_amount.unit)

        self.assertTrue(u.math.allclose(converted, 3.0 * cyto_amount.unit, atol=1e-12 * cyto_amount.unit))

    def test_stoichiometric_reaction_uses_power_and_signed_coefficients(self) -> None:
        ion = _StoichKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.A.value = jnp.array([2.0]) * u.mM
        ion.B.value = jnp.array([5.0]) * u.mM
        ion.C.value = jnp.array([7.0]) * u.mM

        ion.compute_derivative(V)

        expected_flux = 2.0 * (2.0 ** 3) * 5.0 - 0.5 * 7.0
        self.assertTrue(
            u.math.allclose(
                ion.A.derivative.to_decimal(u.mM / u.ms),
                jnp.array([-3.0 * expected_flux]),
                atol=1e-6,
            )
        )

    def test_mixed_factor_reaction_with_incompatible_scaled_units_raises(self) -> None:
        ion = _UnitMismatchKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        with self.assertRaises(Exception):
            ion.compute_derivative(V)


if __name__ == "__main__":
    unittest.main()
