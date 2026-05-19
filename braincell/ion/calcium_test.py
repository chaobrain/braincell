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


import types
import unittest

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._base import HHTypedNeuron, Ion, IonInfo
from braincell.channel.calcium import CaT_HM1992
from braincell.ion.calcium import (
    CdpStC_CAMOnly_MA2020_GoC,
    CdpStC_NoCAM_MA2020_GoC,
    CdpStC_MA2020_GoC,
    CdpHVA_SU2015_DCN,
    CdpLVA_SU2015_DCN,
    Calcium,
    CalciumDetailed,
    CalciumFirstOrder,
    CalciumFixed,
    CalciumInitNernst,
    ToyCaBindingKinetic_SU2015_DCN,
    ToyDiamFactorKinetic_SU2015_DCN,
    ToyCaBindingIcaSourceKinetic_SU2015_DCN,
    ToyCaPumpFactorKinetic_SU2015_DCN,
    ToyCaBindingSourceKinetic_SU2015_DCN,
)
from braincell.ion._base import DynamicNernstIon, InitNernstIon, KineticIon
from braincell.quad.protocol import DiffEqState


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


class CalciumBaseTest(unittest.TestCase):
    """Tests for the abstract :class:`Calcium` base class."""

    def test_calcium_is_subclass_of_ion(self) -> None:
        self.assertTrue(issubclass(Calcium, Ion))

    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Calcium.root_type, HHTypedNeuron)

    def test_module_attribute_is_public_namespace(self) -> None:
        self.assertEqual(Calcium.__module__, "braincell.ion")


class CalciumFixedDefaultsTest(unittest.TestCase):
    """Defaults and parameter storage for :class:`CalciumFixed`."""

    def test_default_reversal_potential_is_120_mV(self) -> None:
        ca = CalciumFixed(size=1)
        self.assertTrue(u.math.allclose(ca.E, 120.0 * u.mV, atol=1e-9 * u.mV))

    def test_default_intracellular_and_extracellular_concentrations(self) -> None:
        ca = CalciumFixed(size=1)
        self.assertTrue(
            u.math.allclose(ca.Ci, 5e-05 * u.mM, atol=1e-12 * u.mM)
        )
        self.assertTrue(u.math.allclose(ca.Co, 2.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(ca.valence, 2.0 * jnp.ones((1,)), atol=1e-9))

    def test_varshape_matches_size(self) -> None:
        self.assertEqual(CalciumFixed(size=1).varshape, (1,))
        self.assertEqual(CalciumFixed(size=4).varshape, (4,))

    def test_custom_parameters_are_honoured(self) -> None:
        ca = CalciumFixed(size=2, E=140.0 * u.mV, Ci=0.5e-3 * u.mM, Co=1.8 * u.mM, valence=2)
        self.assertTrue(u.math.allclose(ca.E, 140.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(
            u.math.allclose(ca.Ci, 0.5e-3 * u.mM, atol=1e-12 * u.mM)
        )
        self.assertTrue(u.math.allclose(ca.Co, 1.8 * u.mM, atol=1e-9 * u.mM))


class CalciumFixedPackInfoTest(unittest.TestCase):
    def test_pack_info_returns_ion_info(self) -> None:
        ca = CalciumFixed(size=1)
        info = ca.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, 5e-05 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.Co, 2.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(info.E, 120.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(info.valence, 2.0 * jnp.ones((1,)), atol=1e-9))


class CalciumFixedContainerTest(unittest.TestCase):
    def test_no_channels_by_default(self) -> None:
        ca = CalciumFixed(size=1)
        self.assertEqual(ca.channels, {})
        self.assertEqual(ca.external_currents, {})

    def test_channels_kwarg_is_attached(self) -> None:
        ca = CalciumFixed(size=1, ICa=CaT_HM1992(size=1))
        self.assertIn("ICa", ca.channels)
        self.assertIsInstance(ca.channels["ICa"], CaT_HM1992)

    def test_current_without_channels_returns_none(self) -> None:
        ca = CalciumFixed(size=1)
        self.assertIsNone(ca.current(_V([-60.0])))

    def test_current_with_channel_delegates_to_channel(self) -> None:
        ca = CalciumFixed(size=1, ICa=CaT_HM1992(size=1))
        V = _V([-60.0])
        ca.init_state(V)
        ca.reset_state(V)
        i = ca.current(V)
        self.assertEqual(i.shape, (1,))


class CalciumFixedLifecycleTest(unittest.TestCase):
    """``CalciumFixed`` overrides ``reset_state`` – exercise that path."""

    def test_init_state_and_reset_without_children_are_no_ops(self) -> None:
        ca = CalciumFixed(size=1)
        V = _V([-60.0])
        ca.init_state(V)
        ca.reset_state(V)

    def test_init_state_initialises_child_channel_gate(self) -> None:
        ca = CalciumFixed(size=1, ICa=CaT_HM1992(size=1))
        V = _V([-60.0])
        ca.init_state(V)
        ch = ca.channels["ICa"]
        self.assertEqual(ch.p.value.shape, (1,))
        self.assertEqual(ch.q.value.shape, (1,))

    def test_reset_state_forwards_to_child(self) -> None:
        ca = CalciumFixed(size=1, ICa=CaT_HM1992(size=1))
        V = _V([-60.0])
        ca.init_state(V)
        ca.reset_state(V)
        first_p = ca.channels["ICa"].p.value

        ca.channels["ICa"].p.value = jnp.array([0.999])
        ca.reset_state(V)
        self.assertTrue(
            u.math.allclose(ca.channels["ICa"].p.value, first_p, atol=1e-9)
        )


class CalciumInitNernstTest(unittest.TestCase):
    def test_is_init_nernst_ion(self) -> None:
        self.assertTrue(issubclass(CalciumInitNernst, Calcium))
        self.assertTrue(issubclass(CalciumInitNernst, InitNernstIon))

    def test_E_is_initialized_from_nernst_on_init_and_reset(self) -> None:
        ca = CalciumInitNernst(size=1)
        V = _V([-65.0])

        self.assertIsNone(ca.E)
        ca.init_state(V)
        expected = (
            u.gas_constant * ca.temp / (ca.valence * u.faraday_constant)
            * u.math.log(ca.Co / ca.Ci)
        )
        self.assertTrue(u.math.allclose(ca.E.to_decimal(u.mV), expected.to_decimal(u.mV), atol=1e-6))

        first_E = ca.E
        ca.Ci = jnp.array([1.0e-3]) * u.mM
        self.assertTrue(u.math.allclose(ca.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

        ca.reset_state(V)
        self.assertFalse(u.math.allclose(ca.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

    def test_custom_Ci_and_Co_are_respected(self) -> None:
        ca = CalciumInitNernst(size=1, Ci=1e-4 * u.mM, Co=1.8 * u.mM)
        self.assertTrue(u.math.allclose(ca.Ci, 1e-4 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ca.Co, 1.8 * u.mM, atol=1e-9 * u.mM))

class CalciumDetailedTest(unittest.TestCase):
    """Intracellular calcium pool with the Destexhe et al. (1993) model."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(CalciumDetailed, Calcium))
        self.assertTrue(issubclass(CalciumDetailed, DynamicNernstIon))
        self.assertTrue(CalciumDetailed.uses_total_current)

    def test_default_parameters(self) -> None:
        cd = CalciumDetailed(size=1)
        self.assertTrue(
            u.math.allclose(cd.d, 1.0 * u.um, atol=1e-9 * u.um)
        )
        self.assertTrue(
            u.math.allclose(cd.tau, 5.0 * u.ms, atol=1e-9 * u.ms)
        )
        self.assertTrue(
            u.math.allclose(cd.C_rest, 2.4e-4 * u.mM, atol=1e-12 * u.mM)
        )
        self.assertTrue(u.math.allclose(cd.Co, 2.0 * u.mM, atol=1e-9 * u.mM))
        # temp defaults to celsius-to-kelvin(36) = 309.15 K.
        self.assertTrue(
            u.math.allclose(cd.temp, u.celsius2kelvin(36.0), atol=1e-6 * u.kelvin)
        )

    def test_species_first_inheritance_without_intermediate_dynamic_base(self) -> None:
        self.assertLess(CalciumDetailed.__mro__.index(Calcium), CalciumDetailed.__mro__.index(DynamicNernstIon))

    def test_init_state_creates_diffeq_state_Ci(self) -> None:
        cd = CalciumDetailed(size=2)
        V = _V([-60.0, -55.0])
        cd.init_state(V)
        self.assertIsInstance(cd.Ci, DiffEqState)
        self.assertEqual(cd.Ci.value.shape, (2,))
        # Default initializer is Constant(2.4e-4 mM).
        self.assertTrue(
            u.math.allclose(
                cd.Ci.value,
                jnp.full((2,), 2.4e-4) * u.mM,
                atol=1e-12 * u.mM,
            )
        )

    def test_custom_Ci_initializer_is_respected(self) -> None:
        cd = CalciumDetailed(
            size=1,
            Ci_initializer=braintools.init.Constant(1e-3 * u.mM),
        )
        cd.init_state(_V([-60.0]))
        self.assertTrue(
            u.math.allclose(cd.Ci.value, jnp.array([1e-3]) * u.mM, atol=1e-12 * u.mM)
        )

    def test_reset_state_restores_Ci_to_initializer(self) -> None:
        cd = CalciumDetailed(size=1)
        V = _V([-60.0])
        cd.init_state(V)
        cd.Ci.value = jnp.array([0.01]) * u.mM
        cd.reset_state(V)
        self.assertTrue(
            u.math.allclose(
                cd.Ci.value, jnp.array([2.4e-4]) * u.mM, atol=1e-12 * u.mM
            )
        )

    def test_E_follows_nernst_formula(self) -> None:
        # E = (R temp) / (2 F) * ln(Co / C_i)
        cd = CalciumDetailed(size=1)
        cd.init_state(_V([-60.0]))
        expected = (
            u.gas_constant * cd.temp / (2 * u.faraday_constant)
            * u.math.log(cd.Co / cd.Ci.value)
        )
        self.assertTrue(u.math.allclose(cd.E, expected, atol=1e-6 * u.mV))

    def test_E_updates_with_C_dynamically(self) -> None:
        cd = CalciumDetailed(size=1)
        cd.init_state(_V([-60.0]))
        e_low = cd.E
        cd.Ci.value = jnp.array([1e-2]) * u.mM
        e_high = cd.E
        # Nernst: raising [Ca_i] shrinks log(Co/C_i), so E should shrink.
        self.assertLess(float(e_high[0] / u.mV), float(e_low[0] / u.mV))

    def test_E_zero_when_internal_equals_external(self) -> None:
        cd = CalciumDetailed(size=1)
        cd.init_state(_V([-60.0]))
        cd.Ci.value = jnp.array([2.0]) * u.mM  # == Co
        # log(1) = 0 ⇒ E = 0 mV.
        self.assertTrue(u.math.allclose(cd.E, 0.0 * u.mV, atol=1e-6 * u.mV))

    def test_derivative_without_current_matches_leak_formula(self) -> None:
        # With ICa pinned to zero, ``drive`` is clamped at 0 and the
        # derivative collapses to ``(C_rest - Ci) / tau``.
        cd = CalciumDetailed(size=1, tau=5.0 * u.ms, C_rest=1e-4 * u.mM)
        V = _V([-60.0])
        cd.init_state(V)
        cd.Ci.value = jnp.array([6e-4]) * u.mM

        def zero_current(self, V, include_external=True):
            return jnp.zeros(self.varshape) * u.uA / u.cm ** 2

        cd.current = types.MethodType(zero_current, cd)
        cd.compute_derivative(V)

        expected_rate = (1e-4 - 6e-4) / 5.0  # mM/ms
        got = cd.Ci.derivative.to_decimal(u.mM / u.ms)
        self.assertTrue(
            u.math.allclose(got, jnp.array([expected_rate]), atol=1e-9)
        )

    def test_compute_derivative_forwards_to_child_channel(self) -> None:
        cd = CalciumDetailed(size=1, ICa=CaT_HM1992(size=1))
        V = _V([-60.0])
        cd.init_state(V)
        cd.reset_state(V)
        cd.compute_derivative(V)
        # Child channel must have gate derivatives populated and the ion
        # itself must have a Ci derivative after the call.
        ch = cd.channels["ICa"]
        self.assertEqual(ch.p.derivative.shape, (1,))
        self.assertEqual(ch.q.derivative.shape, (1,))
        self.assertEqual(cd.Ci.derivative.shape, (1,))


class CdpHVA_SU2015_DCNTest(unittest.TestCase):
    """Imported DCN calcium-pool template from ``CdpHVA_SU15_DCN.mod``."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(CdpHVA_SU2015_DCN, Calcium))
        self.assertTrue(issubclass(CdpHVA_SU2015_DCN, DynamicNernstIon))
        self.assertTrue(CdpHVA_SU2015_DCN.uses_total_current)

    def test_default_parameters(self) -> None:
        ion = CdpHVA_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ion.kCa, jnp.asarray([3.45e-7]) / u.coulomb, atol=1e-12 / u.coulomb))
        self.assertTrue(u.math.allclose(ion.tauCa, 70.0 * u.ms, atol=1e-9 * u.ms))
        self.assertTrue(u.math.allclose(ion.caiBase, 50e-6 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.depth, 0.2 * u.um, atol=1e-9 * u.um))

    def test_init_state_uses_cai_base_when_initializer_omitted(self) -> None:
        ion = CdpHVA_SU2015_DCN(size=2)
        ion.init_state(_V([-60.0, -55.0]))
        self.assertIsInstance(ion.Ci, DiffEqState)
        self.assertTrue(
            u.math.allclose(
                ion.Ci.value,
                jnp.full((2,), 50e-6) * u.mM,
                atol=1e-12 * u.mM,
            )
        )

    def test_reset_state_restores_Ci_initializer(self) -> None:
        ion = CdpHVA_SU2015_DCN(size=1, Ci_initializer=braintools.init.Constant(8e-5 * u.mM))
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([0.01]) * u.mM
        ion.reset_state(V)
        self.assertTrue(
            u.math.allclose(ion.Ci.value, jnp.array([8e-5]) * u.mM, atol=1e-12 * u.mM)
        )

    def test_zero_ica_derivative_relaxes_to_cai_base(self) -> None:
        ion = CdpHVA_SU2015_DCN(size=1, tauCa=70.0 * u.ms, caiBase=50e-6 * u.mM)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([80e-6]) * u.mM

        def zero_current(self, V, include_external=True):
            return jnp.zeros(self.varshape) * u.mA / u.cm ** 2

        ion.current = types.MethodType(zero_current, ion)
        ion.compute_derivative(V)

        expected = -(80e-6 - 50e-6) / 70.0
        got = ion.Ci.derivative.to_decimal(u.mM / u.ms)
        self.assertTrue(u.math.allclose(got, jnp.array([expected]), atol=1e-12))

    def test_none_total_current_is_treated_as_zero_ica(self) -> None:
        ion = CdpHVA_SU2015_DCN(size=1, tauCa=70.0 * u.ms, caiBase=50e-6 * u.mM)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([80e-6]) * u.mM
        got = ion.derivative(ion.Ci.value, V, total_current=None).to_decimal(u.mM / u.ms)
        expected = jnp.array([-(80e-6 - 50e-6) / 70.0])
        self.assertTrue(u.math.allclose(got, expected, atol=1e-12))

    def test_pack_info_uses_runtime_Ci(self) -> None:
        ion = CdpHVA_SU2015_DCN(size=1)
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([8e-5]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))


class ToyCaBindingKinetic_SU2015_DCNTest(unittest.TestCase):
    """Minimal reversible binding toy for ``KineticIon`` validation."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(ToyCaBindingKinetic_SU2015_DCN, Calcium))
        self.assertTrue(issubclass(ToyCaBindingKinetic_SU2015_DCN, KineticIon))
        self.assertFalse(ToyCaBindingKinetic_SU2015_DCN.uses_total_current)

    def test_default_parameters(self) -> None:
        ion = ToyCaBindingKinetic_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ion.kf, jnp.asarray([2.0]) / (u.mM * u.ms), atol=1e-12 / (u.mM * u.ms)))
        self.assertTrue(u.math.allclose(ion.kb, jnp.asarray([0.5]) / u.ms, atol=1e-12 / u.ms))
        self.assertTrue(u.math.allclose(ion.Btot, 1.0 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.Co, 2.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(
            u.math.allclose(ion.temp, u.celsius2kelvin(36.0), atol=1e-6 * u.kelvin)
        )

    def test_init_state_honours_custom_species_initializers(self) -> None:
        ion = ToyCaBindingKinetic_SU2015_DCN(
            size=1,
            Ci_initializer=0.25 * u.mM,
            BC_initializer=0.3 * u.mM,
            Btot=1.0 * u.mM,
        )
        ion.init_state(_V([-60.0]))

        self.assertIsInstance(ion.Ci, DiffEqState)
        self.assertIsInstance(ion.BC, DiffEqState)
        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.25]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_reset_state_restores_custom_species_initializers(self) -> None:
        ion = ToyCaBindingKinetic_SU2015_DCN(
            size=1,
            Ci_initializer=0.25 * u.mM,
            BC_initializer=0.3 * u.mM,
            Btot=1.0 * u.mM,
        )
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([0.9]) * u.mM
        ion.BC.value = jnp.array([0.8]) * u.mM
        ion.reset_state(V)

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.25]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_species_values_resolve_algebraic_buffer(self) -> None:
        ion = ToyCaBindingKinetic_SU2015_DCN(size=1, Btot=1.2 * u.mM, BC_initializer=0.35 * u.mM)
        ion.init_state(_V([-60.0]))
        values = ion.species_values()
        self.assertTrue(u.math.allclose(values["Ci"], ion.Ci.value, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(values["BC"], ion.BC.value, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(values["B"], jnp.array([0.85]) * u.mM, atol=1e-12 * u.mM))

    def test_high_ci_low_bc_drives_binding_forward(self) -> None:
        ion = ToyCaBindingKinetic_SU2015_DCN(size=1, kf=2.0 / (u.mM * u.ms), kb=0.5 / u.ms, Btot=1.0 * u.mM)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.BC.value = jnp.array([0.1]) * u.mM
        ion.compute_derivative(V)

        self.assertLess(float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertGreater(float(ion.BC.derivative[0].to_decimal(u.mM / u.ms)), 0.0)

    def test_low_ci_high_bc_drives_reverse_release(self) -> None:
        ion = ToyCaBindingKinetic_SU2015_DCN(size=1, kf=0.05 / (u.mM * u.ms), kb=1.5 / u.ms, Btot=1.0 * u.mM)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([0.01]) * u.mM
        ion.BC.value = jnp.array([0.8]) * u.mM
        ion.compute_derivative(V)

        self.assertGreater(float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertLess(float(ion.BC.derivative[0].to_decimal(u.mM / u.ms)), 0.0)

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = ToyCaBindingKinetic_SU2015_DCN(size=1)
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))


class ToyCaBindingSourceKinetic_SU2015_DCNTest(unittest.TestCase):
    """Minimal reversible binding toy with constant ``Ci`` source."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(ToyCaBindingSourceKinetic_SU2015_DCN, Calcium))
        self.assertTrue(issubclass(ToyCaBindingSourceKinetic_SU2015_DCN, KineticIon))
        self.assertFalse(ToyCaBindingSourceKinetic_SU2015_DCN.uses_total_current)

    def test_default_parameters(self) -> None:
        ion = ToyCaBindingSourceKinetic_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ion.kf, jnp.asarray([2.0]) / (u.mM * u.ms), atol=1e-12 / (u.mM * u.ms)))
        self.assertTrue(u.math.allclose(ion.kb, jnp.asarray([0.5]) / u.ms, atol=1e-12 / u.ms))
        self.assertTrue(u.math.allclose(ion.Btot, 1.0 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.ci_source, 0.002 * u.mM / u.ms, atol=1e-12 * u.mM / u.ms))

    def test_init_state_honours_custom_species_initializers(self) -> None:
        ion = ToyCaBindingSourceKinetic_SU2015_DCN(
            size=1,
            Ci_initializer=0.25 * u.mM,
            BC_initializer=0.3 * u.mM,
            Btot=1.0 * u.mM,
        )
        ion.init_state(_V([-60.0]))

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.25]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_constant_source_increases_ci_derivative_relative_to_source_free_version(self) -> None:
        V = _V([-60.0])
        no_source = ToyCaBindingKinetic_SU2015_DCN(size=1, Btot=1.0 * u.mM)
        with_source = ToyCaBindingSourceKinetic_SU2015_DCN(size=1, Btot=1.0 * u.mM, ci_source=0.01 * u.mM / u.ms)
        no_source.init_state(V)
        with_source.init_state(V)
        no_source.Ci.value = jnp.array([0.2]) * u.mM
        no_source.BC.value = jnp.array([0.3]) * u.mM
        no_source.post_integral(V)
        with_source.Ci.value = jnp.array([0.2]) * u.mM
        with_source.BC.value = jnp.array([0.3]) * u.mM
        with_source.post_integral(V)

        no_source.compute_derivative(V)
        with_source.compute_derivative(V)

        self.assertGreater(
            float(with_source.Ci.derivative[0].to_decimal(u.mM / u.ms)),
            float(no_source.Ci.derivative[0].to_decimal(u.mM / u.ms)),
        )

    def test_zero_reaction_source_only_increases_ci(self) -> None:
        ion = ToyCaBindingSourceKinetic_SU2015_DCN(
            size=1,
            kf=0.0 / (u.mM * u.ms),
            kb=0.0 / u.ms,
            ci_source=0.01 * u.mM / u.ms,
            Btot=1.0 * u.mM,
            Ci_initializer=0.2 * u.mM,
            BC_initializer=0.3 * u.mM,
        )
        V = _V([-60.0])
        ion.init_state(V)
        ion.compute_derivative(V)

        self.assertTrue(u.math.allclose(ion.Ci.derivative, jnp.array([0.01]) * u.mM / u.ms, atol=1e-12 * u.mM / u.ms))
        self.assertTrue(u.math.allclose(ion.BC.derivative, jnp.array([0.0]) * u.mM / u.ms, atol=1e-12 * u.mM / u.ms))
        self.assertTrue(u.math.allclose(ion.B.value + ion.BC.value, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = ToyCaBindingSourceKinetic_SU2015_DCN(size=1)
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))


class ToyCaBindingIcaSourceKinetic_SU2015_DCNTest(unittest.TestCase):
    """Minimal reversible binding toy with current-driven ``Ci`` source."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(ToyCaBindingIcaSourceKinetic_SU2015_DCN, Calcium))
        self.assertTrue(issubclass(ToyCaBindingIcaSourceKinetic_SU2015_DCN, KineticIon))
        self.assertTrue(ToyCaBindingIcaSourceKinetic_SU2015_DCN.uses_total_current)

    def test_default_parameters(self) -> None:
        ion = ToyCaBindingIcaSourceKinetic_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ion.kf, jnp.asarray([2.0]) / (u.mM * u.ms), atol=1e-12 / (u.mM * u.ms)))
        self.assertTrue(u.math.allclose(ion.kb, jnp.asarray([0.5]) / u.ms, atol=1e-12 / u.ms))
        self.assertTrue(u.math.allclose(ion.Btot, 1.0 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.kCa, jnp.asarray([3.45e-7]) / u.coulomb, atol=1e-12 / u.coulomb))
        self.assertTrue(u.math.allclose(ion.depth, 0.2 * u.um, atol=1e-12 * u.um))

    def test_init_state_honours_custom_species_initializers(self) -> None:
        ion = ToyCaBindingIcaSourceKinetic_SU2015_DCN(
            size=1,
            Ci_initializer=0.25 * u.mM,
            BC_initializer=0.3 * u.mM,
            Btot=1.0 * u.mM,
        )
        ion.init_state(_V([-60.0]))
        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.25]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_none_total_current_matches_source_free_version(self) -> None:
        V = _V([-60.0])
        no_source = ToyCaBindingKinetic_SU2015_DCN(size=1, Btot=1.0 * u.mM)
        with_source = ToyCaBindingIcaSourceKinetic_SU2015_DCN(size=1, Btot=1.0 * u.mM)
        no_source.init_state(V)
        with_source.init_state(V)
        no_source.Ci.value = jnp.array([0.2]) * u.mM
        no_source.BC.value = jnp.array([0.3]) * u.mM
        no_source.post_integral(V)
        with_source.Ci.value = jnp.array([0.2]) * u.mM
        with_source.BC.value = jnp.array([0.3]) * u.mM
        with_source.post_integral(V)

        no_source.compute_derivative(V)
        got = with_source.sources[0].flux(with_source, V, with_source.species_values(), total_current=None)

        self.assertTrue(u.math.allclose(got, jnp.array([0.0]) * u.mM / u.ms, atol=1e-12 * u.mM / u.ms))
        with_source.compute_derivative(V)
        self.assertTrue(
            u.math.allclose(
                with_source.Ci.derivative.to_decimal(u.mM / u.ms),
                no_source.Ci.derivative.to_decimal(u.mM / u.ms),
                atol=1e-12,
            )
        )

    def test_positive_inward_current_increases_ci_derivative_relative_to_source_free_version(self) -> None:
        V = _V([-60.0])
        no_source = ToyCaBindingKinetic_SU2015_DCN(size=1, Btot=1.0 * u.mM)
        with_source = ToyCaBindingIcaSourceKinetic_SU2015_DCN(size=1, Btot=1.0 * u.mM)
        no_source.init_state(V)
        with_source.init_state(V)
        no_source.Ci.value = jnp.array([0.2]) * u.mM
        no_source.BC.value = jnp.array([0.3]) * u.mM
        no_source.post_integral(V)
        with_source.Ci.value = jnp.array([0.2]) * u.mM
        with_source.BC.value = jnp.array([0.3]) * u.mM
        with_source.post_integral(V)

        no_source.compute_derivative(V)
        with_source._cached_total_current = jnp.array([0.01]) * u.mA / (u.cm ** 2)
        with_source.compute_derivative(V)

        self.assertGreater(
            float(with_source.Ci.derivative[0].to_decimal(u.mM / u.ms)),
            float(no_source.Ci.derivative[0].to_decimal(u.mM / u.ms)),
        )


    def test_zero_reaction_current_only_changes_ci(self) -> None:
        ion = ToyCaBindingIcaSourceKinetic_SU2015_DCN(
            size=1,
            kf=0.0 / (u.mM * u.ms),
            kb=0.0 / u.ms,
            Btot=1.0 * u.mM,
            kCa=3.45e-7 / u.coulomb,
            depth=0.2 * u.um,
            Ci_initializer=0.2 * u.mM,
            BC_initializer=0.3 * u.mM,
        )
        V = _V([-60.0])
        ion.init_state(V)
        ion._cached_total_current = jnp.array([0.01]) * u.mA / (u.cm ** 2)
        ion.compute_derivative(V)

        expected = (3.45e-7 / 0.2) * 0.01 * 1e4
        self.assertTrue(u.math.allclose(ion.Ci.derivative, jnp.array([expected]) * u.mM / u.ms, atol=1e-12 * u.mM / u.ms))
        self.assertTrue(u.math.allclose(ion.BC.derivative, jnp.array([0.0]) * u.mM / u.ms, atol=1e-12 * u.mM / u.ms))
        self.assertTrue(u.math.allclose(ion.B.value + ion.BC.value, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = ToyCaBindingIcaSourceKinetic_SU2015_DCN(size=1)
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))


class ToyCaPumpFactorKinetic_SU2015_DCNTest(unittest.TestCase):
    """Minimal mixed-factor toy with cytosolic and pump-area species."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(ToyCaPumpFactorKinetic_SU2015_DCN, Calcium))
        self.assertTrue(issubclass(ToyCaPumpFactorKinetic_SU2015_DCN, KineticIon))
        self.assertTrue(ToyCaPumpFactorKinetic_SU2015_DCN.uses_total_current)

    def test_default_parameters(self) -> None:
        ion = ToyCaPumpFactorKinetic_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ion.kf, jnp.asarray([2.0]) / (u.mM * u.ms), atol=1e-12 / (u.mM * u.ms)))
        self.assertTrue(u.math.allclose(ion.kb, jnp.asarray([0.5]) / u.ms, atol=1e-12 / u.ms))
        self.assertTrue(u.math.allclose(ion.k_rel, jnp.asarray([0.05]) / u.ms, atol=1e-12 / u.ms))
        self.assertTrue(u.math.allclose(ion.PumpTot, 1.0 * u.mM * u.um, atol=1e-12 * u.mM * u.um))
        self.assertTrue(u.math.allclose(ion.cyt_volume, 3.0 * u.um ** 3, atol=1e-12 * u.um ** 3))
        self.assertTrue(u.math.allclose(ion.pump_area, 3.0 * u.um ** 2, atol=1e-12 * u.um ** 2))

    def test_init_state_honours_custom_species_initializers(self) -> None:
        ion = ToyCaPumpFactorKinetic_SU2015_DCN(
            size=1,
            Ci_initializer=0.25 * u.mM,
            PumpBound_initializer=0.3 * u.mM * u.um,
            PumpTot=1.0 * u.mM * u.um,
        )
        ion.init_state(_V([-60.0]))

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.25]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.PumpBound.value, jnp.array([0.3]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))
        self.assertTrue(u.math.allclose(ion.PumpFree.value, jnp.array([0.7]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))

    def test_species_values_resolve_area_factor_species(self) -> None:
        ion = ToyCaPumpFactorKinetic_SU2015_DCN(
            size=1,
            PumpTot=1.2 * u.mM * u.um,
            PumpBound_initializer=0.35 * u.mM * u.um,
        )
        ion.init_state(_V([-60.0]))
        values = ion.species_values()

        self.assertTrue(u.math.allclose(values["PumpBound"], ion.PumpBound.value, atol=1e-12 * u.mM * u.um))
        self.assertTrue(u.math.allclose(values["PumpFree"], jnp.array([0.85]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))

    def test_positive_inward_current_increases_ci_derivative(self) -> None:
        ion = ToyCaPumpFactorKinetic_SU2015_DCN(size=1)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.PumpBound.value = jnp.array([0.3]) * u.mM * u.um
        ion.post_integral(V)
        ion.compute_derivative(V)
        no_current = ion.Ci.derivative

        ion._cached_total_current = jnp.array([0.01]) * u.mA / (u.cm ** 2)
        ion.compute_derivative(V)
        self.assertGreater(
            float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)),
            float(no_current[0].to_decimal(u.mM / u.ms)),
        )

    def test_mixed_factor_reaction_matches_neuron_style_point_derivative(self) -> None:
        ion = ToyCaPumpFactorKinetic_SU2015_DCN(
            size=1,
            kf=2.0 / (u.mM * u.ms),
            kb=0.5 / u.ms,
            k_rel=0.05 / u.ms,
            PumpTot=1.0 * u.mM * u.um,
            Ci_initializer=0.2 * u.mM,
            PumpBound_initializer=0.3 * u.mM * u.um,
            cyt_volume=3.0 * u.um ** 3,
            pump_area=3.0 * u.um ** 2,
        )
        V = _V([-60.0])
        ion.init_state(V)
        ion.compute_derivative(V)

        expected_dci = jnp.array([-0.13]) * u.mM / u.ms
        expected_dpumpbound = jnp.array([0.115]) * u.mM * u.um / u.ms

        self.assertTrue(u.math.allclose(ion.Ci.derivative, expected_dci, atol=1e-12 * u.mM / u.ms))
        self.assertTrue(
            u.math.allclose(
                ion.PumpBound.derivative,
                expected_dpumpbound,
                atol=1e-12 * u.mM * u.um / u.ms,
            )
        )

    def test_zero_reaction_current_only_changes_ci(self) -> None:
        ion = ToyCaPumpFactorKinetic_SU2015_DCN(
            size=1,
            kf=0.0 / (u.mM * u.ms),
            kb=0.0 / u.ms,
            k_rel=0.0 / u.ms,
            PumpTot=1.0 * u.mM * u.um,
            Ci_initializer=0.2 * u.mM,
            PumpBound_initializer=0.3 * u.mM * u.um,
        )
        V = _V([-60.0])
        ion.init_state(V)
        ion._cached_total_current = jnp.array([0.01]) * u.mA / (u.cm ** 2)
        ion.compute_derivative(V)

        self.assertTrue(u.math.allclose(ion.PumpBound.derivative, jnp.array([0.0]) * u.mM * u.um / u.ms, atol=1e-12 * u.mM * u.um / u.ms))
        self.assertTrue(u.math.allclose(ion.PumpFree.value + ion.PumpBound.value, jnp.array([1.0]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))
        self.assertGreater(float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)), 0.0)

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = ToyCaPumpFactorKinetic_SU2015_DCN(size=1)
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))


class ToyDiamFactorKinetic_SU2015_DCNTest(unittest.TestCase):
    """Minimal geometry-derived factor toy with reversible pump binding."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(ToyDiamFactorKinetic_SU2015_DCN, Calcium))
        self.assertTrue(issubclass(ToyDiamFactorKinetic_SU2015_DCN, KineticIon))
        self.assertFalse(ToyDiamFactorKinetic_SU2015_DCN.uses_total_current)

    def test_default_parameters(self) -> None:
        ion = ToyDiamFactorKinetic_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ion.kf, jnp.asarray([2.0]) / (u.mM * u.ms), atol=1e-12 / (u.mM * u.ms)))
        self.assertTrue(u.math.allclose(ion.kb, jnp.asarray([0.5]) / u.ms, atol=1e-12 / u.ms))
        self.assertTrue(u.math.allclose(ion.PumpTot, 1.0 * u.mM * u.um, atol=1e-12 * u.mM * u.um))
        self.assertTrue(u.math.allclose(ion.depth, 1.0 * u.um, atol=1e-12 * u.um))
        self.assertTrue(u.math.allclose(ion.Co, 2.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(ion.temp, u.celsius2kelvin(36.0), atol=1e-6 * u.kelvin))

    def test_init_state_honours_custom_species_initializers(self) -> None:
        ion = ToyDiamFactorKinetic_SU2015_DCN(
            size=1,
            Ci_initializer=0.25 * u.mM,
            PumpBound_initializer=0.3 * u.mM * u.um,
            PumpTot=1.0 * u.mM * u.um,
        )
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        ion.init_state(_V([-60.0]))

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.25]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.PumpBound.value, jnp.array([0.3]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))
        self.assertTrue(u.math.allclose(ion.PumpFree.value, jnp.array([0.7]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))

    def test_species_values_resolve_geometry_factor_species(self) -> None:
        ion = ToyDiamFactorKinetic_SU2015_DCN(
            size=1,
            PumpTot=1.2 * u.mM * u.um,
            PumpBound_initializer=0.35 * u.mM * u.um,
        )
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        ion.init_state(_V([-60.0]))
        values = ion.species_values()

        self.assertTrue(u.math.allclose(values["PumpBound"], ion.PumpBound.value, atol=1e-12 * u.mM * u.um))
        self.assertTrue(u.math.allclose(values["PumpFree"], jnp.array([0.85]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))

    def test_geometry_factor_reaction_matches_neuron_style_point_derivative(self) -> None:
        ion = ToyDiamFactorKinetic_SU2015_DCN(
            size=1,
            kf=2.0 / (u.mM * u.ms),
            kb=0.5 / u.ms,
            PumpTot=1.0 * u.mM * u.um,
            depth=1.0 * u.um,
            Ci_initializer=0.2 * u.mM,
            PumpBound_initializer=0.3 * u.mM * u.um,
        )
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        V = _V([-60.0])
        ion.init_state(V)
        ion.compute_derivative(V)

        expected_dci = jnp.array([-0.13]) * u.mM / u.ms
        expected_dpumpbound = jnp.array([0.13]) * u.mM * u.um / u.ms

        self.assertTrue(u.math.allclose(ion.Ci.derivative, expected_dci, atol=1e-12 * u.mM / u.ms))
        self.assertTrue(u.math.allclose(ion.PumpBound.derivative, expected_dpumpbound, atol=1e-12 * u.mM * u.um / u.ms))

    def test_reset_state_restores_custom_species_initializers(self) -> None:
        ion = ToyDiamFactorKinetic_SU2015_DCN(
            size=1,
            Ci_initializer=0.25 * u.mM,
            PumpBound_initializer=0.3 * u.mM * u.um,
            PumpTot=1.0 * u.mM * u.um,
        )
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([0.9]) * u.mM
        ion.PumpBound.value = jnp.array([0.8]) * u.mM * u.um
        ion.reset_state(V)

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.25]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.PumpBound.value, jnp.array([0.3]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))
        self.assertTrue(u.math.allclose(ion.PumpFree.value, jnp.array([0.7]) * u.mM * u.um, atol=1e-12 * u.mM * u.um))

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = ToyDiamFactorKinetic_SU2015_DCN(size=1)
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))


class CdpStC_MA2020_GoCTest(unittest.TestCase):
    """Imported GoC calcium pool with preserved KINETIC reactions."""

    def _make_ion(self, **kwargs):
        ion = CdpStC_MA2020_GoC(size=1, **kwargs)
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        return ion

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(CdpStC_MA2020_GoC, Calcium))
        self.assertTrue(issubclass(CdpStC_MA2020_GoC, KineticIon))
        self.assertTrue(CdpStC_MA2020_GoC.uses_total_current)
        self.assertEqual(len(CdpStC_MA2020_GoC.reactions), 20)
        self.assertEqual(len(CdpStC_MA2020_GoC.sources), 1)
        self.assertEqual(len(CdpStC_MA2020_GoC.conserves), 1)

    def test_default_parameters(self) -> None:
        ion = self._make_ion()
        self.assertTrue(u.math.allclose(ion.temp, u.celsius2kelvin(25.0), atol=1e-6 * u.kelvin))
        self.assertTrue(u.math.allclose(ion.cainull, 45e-6 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.mginull, 0.59 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.TotalPump, 1e-9 * (u.mol / u.cm ** 2), atol=1e-18 * (u.mol / u.cm ** 2)))
        self.assertFalse(hasattr(ion, "celsius"))
        self.assertFalse(hasattr(ion, "ica_pmp"))
        self.assertFalse(hasattr(ion, "icazz"))
        self.assertFalse(hasattr(ion, "parea2"))

    def test_init_state_matches_mod_semantics_and_populates_geometry_fields(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))

        self.assertIsInstance(ion.Ci, DiffEqState)
        self.assertIsInstance(ion.pump, DiffEqState)
        self.assertFalse(isinstance(ion.pumpca, DiffEqState))
        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([45e-6]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.mg.value, jnp.array([0.59]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.CAM0.value, jnp.array([0.03]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.CAM4.value, jnp.array([0.0]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.pump.value, jnp.array([1e-9]) * (u.mol / u.cm ** 2), atol=1e-18 * (u.mol / u.cm ** 2)))
        self.assertTrue(u.math.allclose(ion.pumpca.value, jnp.array([0.0]) * (u.mol / u.cm ** 2), atol=1e-18 * (u.mol / u.cm ** 2)))

        expected_vrat = np.pi * (0.5 - (0.25 / (10.9495 - 1.0) / 2.0)) * 2.0 * (0.25 / (10.9495 - 1.0))
        self.assertAlmostEqual(float(ion.vrat), expected_vrat, places=12)
        self.assertAlmostEqual(float(ion.parea[0].to_decimal(u.um)), float(np.pi * 20.0), places=5)
        self.assertAlmostEqual(float(ion.dsq[0].to_decimal(u.um ** 2)), 400.0, places=6)
        self.assertAlmostEqual(
            float(ion.dsqvol[0].to_decimal(u.um ** 2)),
            400.0 * expected_vrat,
            places=5,
        )

    def test_species_initializer_rejects_algebraic_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "differential-species overrides"):
            self._make_ion(species_initializers={"pumpca": 0.0 * (u.mol / u.cm ** 2)})

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))

    def test_zero_total_current_produces_zero_source_flux(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        flux = ion.sources[0].flux(ion, _V([-60.0]), ion.species_values(), total_current=None)
        self.assertTrue(u.math.allclose(flux, jnp.array([0.0]) * u.mM * u.um ** 2 / u.ms, atol=1e-12 * u.mM * u.um ** 2 / u.ms))

    def test_positive_inward_current_produces_negative_ci_source_flux(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        flux = ion.sources[0].flux(
            ion,
            _V([-60.0]),
            ion.species_values(),
            total_current=jnp.array([0.01]) * u.mA / (u.cm ** 2),
        )
        self.assertLess(float(flux[0].to_decimal(u.mM * u.um ** 2 / u.ms)), 0.0)

    def test_conserve_keeps_pump_plus_pumpca_equal_total_scaled_pool(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        values = ion.species_values()
        total = ion.TotalPump * ion.parea
        combined = ion.pump.value * ion.parea + values["pumpca"] * ion.parea
        self.assertTrue(
            u.math.allclose(
                combined.to_decimal(total.unit),
                total.to_decimal(total.unit),
                atol=1e-12,
            )
        )

    def test_reaction_r2_pump_release_is_irreversible(self) -> None:
        self.assertIsNone(CdpStC_MA2020_GoC.reactions[1].backward)

    def test_reaction_r1_pump_binding_consumes_ci_and_pump(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.pump.value = jnp.array([1e-9]) * (u.mol / u.cm ** 2)
        ion.pumpca.value = jnp.array([0.0]) * (u.mol / u.cm ** 2)
        ion.compute_derivative(_V([-60.0]))
        self.assertLess(float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertLess(float(ion.pump.derivative[0].to_decimal(u.mol / (u.cm ** 2 * u.ms))), 0.0)

    def test_reaction_r2_pump_release_consumes_pumpca_when_ci_zero(self) -> None:
        ion = self._make_ion(species_initializers={"pump": 0.0 * (u.mol / u.cm ** 2)})
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.0]) * u.mM
        ion.pump.value = jnp.array([0.0]) * (u.mol / u.cm ** 2)
        ion.pumpca.value = jnp.array([1e-9]) * (u.mol / u.cm ** 2)
        ion.compute_derivative(_V([-60.0]))
        self.assertGreater(float(ion.pump.derivative[0].to_decimal(u.mol / (u.cm ** 2 * u.ms))), 0.0)

    def test_zero_ica_integration_stays_finite_and_conserves_pump_pool(self) -> None:
        ion = self._make_ion()
        V = _V([-60.0])
        ion.init_state(V)

        totals = []
        with brainstate.environ.context(dt=0.05 * u.ms):
            for _ in range(20):
                ion.make_integration(V)
                ion.post_integral(V)
                values = ion.species_values()
                totals.append(float((ion.pump.value[0] + values["pumpca"][0]).to_decimal(u.mol / u.cm ** 2)))

        values = ion.species_values()

        tracked = {
            "Ci": np.asarray(ion.Ci.value.to_decimal(u.mM)),
            "pump": np.asarray(ion.pump.value.to_decimal(u.mol / u.cm ** 2)),
            "pumpca": np.asarray(values["pumpca"].to_decimal(u.mol / u.cm ** 2)),
            "CAM0": np.asarray(ion.CAM0.value.to_decimal(u.mM)),
            "CAM1C": np.asarray(ion.CAM1C.value.to_decimal(u.mM)),
            "CAM1N": np.asarray(ion.CAM1N.value.to_decimal(u.mM)),
            "CAM2N": np.asarray(ion.CAM2N.value.to_decimal(u.mM)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        self.assertTrue(np.allclose(totals, totals[0], atol=1e-18))
        self.assertGreater(float(ion.CAM1N.value[0].to_decimal(u.mM)), 0.0)
        self.assertGreater(float(ion.CAM1C.value[0].to_decimal(u.mM)), 0.0)
        self.assertLess(float(ion.Ci.value[0].to_decimal(u.mM)), 1e-3)

    def test_reaction_r3_buffer_binding_consumes_ci_and_buff1(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.Buff1.value = jnp.array([1.0]) * u.mM
        ion.Buff1_ca.value = jnp.array([0.0]) * u.mM
        ion.compute_derivative(_V([-60.0]))
        self.assertLess(float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertLess(float(ion.Buff1.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertGreater(float(ion.Buff1_ca.derivative[0].to_decimal(u.mM / u.ms)), 0.0)

    def test_reaction_r8_mg_pv_binding_consumes_mg_and_pv(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.mg.value = jnp.array([1.0]) * u.mM
        ion.PV.value = jnp.array([0.5]) * u.mM
        ion.PV_mg.value = jnp.array([0.0]) * u.mM
        ion.compute_derivative(_V([-60.0]))
        self.assertLess(float(ion.mg.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertLess(float(ion.PV.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertGreater(float(ion.PV_mg.derivative[0].to_decimal(u.mM / u.ms)), 0.0)

    def test_reaction_r18_cam_mixed_branch_builds_cam1c1n(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.CAM1N.value = jnp.array([0.1]) * u.mM
        ion.CAM1C1N.value = jnp.array([0.0]) * u.mM
        ion.compute_derivative(_V([-60.0]))
        self.assertLess(float(ion.CAM1N.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertGreater(float(ion.CAM1C1N.derivative[0].to_decimal(u.mM / u.ms)), 0.0)


class CdpStC_CAMOnly_MA2020_GoCTest(unittest.TestCase):
    """Imported GoC calcium pool with only the calmodulin subnetwork."""

    def _make_ion(self, **kwargs):
        ion = CdpStC_CAMOnly_MA2020_GoC(size=1, **kwargs)
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        return ion

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(CdpStC_CAMOnly_MA2020_GoC, Calcium))
        self.assertTrue(issubclass(CdpStC_CAMOnly_MA2020_GoC, KineticIon))
        self.assertFalse(CdpStC_CAMOnly_MA2020_GoC.uses_total_current)
        self.assertEqual(len(CdpStC_CAMOnly_MA2020_GoC.reactions), 12)
        self.assertEqual(len(CdpStC_CAMOnly_MA2020_GoC.sources), 0)
        self.assertEqual(len(CdpStC_CAMOnly_MA2020_GoC.conserves), 0)

    def test_default_parameters(self) -> None:
        ion = self._make_ion()
        self.assertTrue(u.math.allclose(ion.temp, u.celsius2kelvin(25.0), atol=1e-6 * u.kelvin))
        self.assertTrue(u.math.allclose(ion.cainull, 45e-6 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.CAM_start, 0.03 * u.mM, atol=1e-12 * u.mM))
        self.assertFalse(hasattr(ion, "celsius"))

    def test_init_state_matches_mod_semantics(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([45e-6]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.CAM0.value, jnp.array([0.03]) * u.mM, atol=1e-12 * u.mM))
        for name in ("CAM1C", "CAM2C", "CAM1N2C", "CAM1N", "CAM2N", "CAM2N1C", "CAM1C1N", "CAM4"):
            self.assertTrue(u.math.allclose(getattr(ion, name).value, jnp.array([0.0]) * u.mM, atol=1e-15 * u.mM))

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))

    def test_first_step_cam_binding_moves_mass_out_of_cam0(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.compute_derivative(_V([-60.0]))
        self.assertLess(float(ion.CAM0.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertGreater(float(ion.CAM1C.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertGreater(float(ion.CAM1N.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertLess(float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)), 0.0)

    def test_short_integration_stays_finite(self) -> None:
        ion = self._make_ion()
        V = _V([-60.0])
        ion.init_state(V)
        with brainstate.environ.context(dt=0.05 * u.ms):
            for _ in range(4):
                ion.make_integration(V)
                ion.post_integral(V)
        tracked = {
            "Ci": np.asarray(ion.Ci.value.to_decimal(u.mM)),
            "CAM0": np.asarray(ion.CAM0.value.to_decimal(u.mM)),
            "CAM1C": np.asarray(ion.CAM1C.value.to_decimal(u.mM)),
            "CAM1N": np.asarray(ion.CAM1N.value.to_decimal(u.mM)),
            "CAM2N": np.asarray(ion.CAM2N.value.to_decimal(u.mM)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())


class CdpStC_NoCAM_MA2020_GoCTest(unittest.TestCase):
    """Imported GoC calcium pool without the calmodulin subnetwork."""

    def _make_ion(self, **kwargs):
        ion = CdpStC_NoCAM_MA2020_GoC(size=1, **kwargs)
        ion.diam_mid = jnp.asarray([20.0]) * u.um
        return ion

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(CdpStC_NoCAM_MA2020_GoC, Calcium))
        self.assertTrue(issubclass(CdpStC_NoCAM_MA2020_GoC, KineticIon))
        self.assertTrue(CdpStC_NoCAM_MA2020_GoC.uses_total_current)
        self.assertEqual(len(CdpStC_NoCAM_MA2020_GoC.reactions), 8)
        self.assertEqual(len(CdpStC_NoCAM_MA2020_GoC.sources), 1)
        self.assertEqual(len(CdpStC_NoCAM_MA2020_GoC.conserves), 1)

    def test_default_parameters(self) -> None:
        ion = self._make_ion()
        self.assertTrue(u.math.allclose(ion.temp, u.celsius2kelvin(25.0), atol=1e-6 * u.kelvin))
        self.assertTrue(u.math.allclose(ion.cainull, 45e-6 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.mginull, 0.59 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.TotalPump, 1e-9 * (u.mol / u.cm ** 2), atol=1e-18 * (u.mol / u.cm ** 2)))
        self.assertFalse(hasattr(ion, "celsius"))
        self.assertFalse(hasattr(ion, "ica_pmp"))
        self.assertFalse(hasattr(ion, "icazz"))
        self.assertFalse(hasattr(ion, "parea2"))

    def test_init_state_matches_mod_semantics_and_populates_geometry_fields(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))

        self.assertIsInstance(ion.Ci, DiffEqState)
        self.assertIsInstance(ion.pump, DiffEqState)
        self.assertFalse(isinstance(ion.pumpca, DiffEqState))
        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([45e-6]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.mg.value, jnp.array([0.59]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.PV.value, ion._ss_pv_free(), atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.PV_ca.value, ion._ss_pv_ca(), atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.PV_mg.value, ion._ss_pv_mg(), atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.pump.value, jnp.array([1e-9]) * (u.mol / u.cm ** 2), atol=1e-18 * (u.mol / u.cm ** 2)))
        self.assertTrue(u.math.allclose(ion.pumpca.value, jnp.array([0.0]) * (u.mol / u.cm ** 2), atol=1e-18 * (u.mol / u.cm ** 2)))

        expected_vrat = np.pi * (0.5 - (0.25 / (10.9495 - 1.0) / 2.0)) * 2.0 * (0.25 / (10.9495 - 1.0))
        self.assertAlmostEqual(float(ion.vrat), expected_vrat, places=12)
        self.assertAlmostEqual(float(ion.parea[0].to_decimal(u.um)), float(np.pi * 20.0), places=5)
        self.assertAlmostEqual(float(ion.dsq[0].to_decimal(u.um ** 2)), 400.0, places=6)
        self.assertAlmostEqual(float(ion.dsqvol[0].to_decimal(u.um ** 2)), 400.0 * expected_vrat, places=5)

    def test_species_initializer_rejects_algebraic_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "differential-species overrides"):
            self._make_ion(species_initializers={"pumpca": 0.0 * (u.mol / u.cm ** 2)})

    def test_pack_info_uses_runtime_ci(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))

    def test_zero_total_current_produces_zero_source_flux(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        flux = ion.sources[0].flux(ion, _V([-60.0]), ion.species_values(), total_current=None)
        self.assertTrue(u.math.allclose(flux, jnp.array([0.0]) * u.mM * u.um ** 2 / u.ms, atol=1e-12 * u.mM * u.um ** 2 / u.ms))

    def test_positive_inward_current_produces_negative_ci_source_flux(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        flux = ion.sources[0].flux(
            ion,
            _V([-60.0]),
            ion.species_values(),
            total_current=jnp.array([0.01]) * u.mA / (u.cm ** 2),
        )
        self.assertLess(float(flux[0].to_decimal(u.mM * u.um ** 2 / u.ms)), 0.0)

    def test_conserve_keeps_pump_plus_pumpca_equal_total_scaled_pool(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        values = ion.species_values()
        total = ion.TotalPump * ion.parea
        combined = ion.pump.value * ion.parea + values["pumpca"] * ion.parea
        self.assertTrue(
            u.math.allclose(
                combined.to_decimal(total.unit),
                total.to_decimal(total.unit),
                atol=1e-12,
            )
        )

    def test_reaction_r2_pump_release_is_irreversible(self) -> None:
        self.assertIsNone(CdpStC_NoCAM_MA2020_GoC.reactions[1].backward)

    def test_reaction_r1_pump_binding_consumes_ci_and_pump(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.pump.value = jnp.array([1e-9]) * (u.mol / u.cm ** 2)
        ion.pumpca.value = jnp.array([0.0]) * (u.mol / u.cm ** 2)
        ion.compute_derivative(_V([-60.0]))
        self.assertLess(float(ion.Ci.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertLess(float(ion.pump.derivative[0].to_decimal(u.mol / (u.cm ** 2 * u.ms))), 0.0)

    def test_reaction_r8_mg_pv_binding_consumes_mg_and_pv(self) -> None:
        ion = self._make_ion()
        ion.init_state(_V([-60.0]))
        ion.mg.value = jnp.array([1.0]) * u.mM
        ion.PV.value = jnp.array([0.5]) * u.mM
        ion.PV_mg.value = jnp.array([0.0]) * u.mM
        ion.compute_derivative(_V([-60.0]))
        self.assertLess(float(ion.mg.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertLess(float(ion.PV.derivative[0].to_decimal(u.mM / u.ms)), 0.0)
        self.assertGreater(float(ion.PV_mg.derivative[0].to_decimal(u.mM / u.ms)), 0.0)

    def test_zero_ica_integration_stays_finite_and_conserves_pump_pool(self) -> None:
        ion = self._make_ion()
        V = _V([-60.0])
        ion.init_state(V)

        totals = []
        with brainstate.environ.context(dt=0.05 * u.ms):
            for _ in range(20):
                ion.make_integration(V)
                ion.post_integral(V)
                values = ion.species_values()
                totals.append(float((ion.pump.value[0] + values["pumpca"][0]).to_decimal(u.mol / u.cm ** 2)))

        values = ion.species_values()
        tracked = {
            "Ci": np.asarray(ion.Ci.value.to_decimal(u.mM)),
            "pump": np.asarray(ion.pump.value.to_decimal(u.mol / u.cm ** 2)),
            "pumpca": np.asarray(values["pumpca"].to_decimal(u.mol / u.cm ** 2)),
            "mg": np.asarray(ion.mg.value.to_decimal(u.mM)),
            "Buff1": np.asarray(ion.Buff1.value.to_decimal(u.mM)),
            "Buff1_ca": np.asarray(ion.Buff1_ca.value.to_decimal(u.mM)),
            "PV": np.asarray(ion.PV.value.to_decimal(u.mM)),
            "PV_ca": np.asarray(ion.PV_ca.value.to_decimal(u.mM)),
            "PV_mg": np.asarray(ion.PV_mg.value.to_decimal(u.mM)),
        }
        for arr in tracked.values():
            self.assertTrue(np.isfinite(arr).all())

        self.assertTrue(np.allclose(totals, totals[0], atol=1e-18))
        self.assertLess(float(ion.Ci.value[0].to_decimal(u.mM)), 1e-3)
        self.assertLess(float(ion.pump.value[0].to_decimal(u.mol / u.cm ** 2)), 1e-8)


class CdpLVA_SU2015_DCNTest(unittest.TestCase):
    """Imported DCN low-voltage calcium-pool template from ``CdpLVA_SU15_DCN.mod``."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(CdpLVA_SU2015_DCN, Calcium))
        self.assertTrue(issubclass(CdpLVA_SU2015_DCN, DynamicNernstIon))
        self.assertTrue(CdpLVA_SU2015_DCN.uses_total_current)

    def test_default_parameters(self) -> None:
        ion = CdpLVA_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ion.kCal, jnp.asarray([3.45e-7]) / u.coulomb, atol=1e-12 / u.coulomb))
        self.assertTrue(u.math.allclose(ion.tauCal, 70.0 * u.ms, atol=1e-9 * u.ms))
        self.assertTrue(u.math.allclose(ion.caliBase, 50e-6 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.depth, 0.2 * u.um, atol=1e-9 * u.um))

    def test_init_state_uses_cali_base_when_initializer_omitted(self) -> None:
        ion = CdpLVA_SU2015_DCN(size=2)
        ion.init_state(_V([-60.0, -55.0]))
        self.assertIsInstance(ion.Ci, DiffEqState)
        self.assertTrue(
            u.math.allclose(
                ion.Ci.value,
                jnp.full((2,), 50e-6) * u.mM,
                atol=1e-12 * u.mM,
            )
        )

    def test_reset_state_restores_Ci_initializer(self) -> None:
        ion = CdpLVA_SU2015_DCN(size=1, Ci_initializer=braintools.init.Constant(9e-5 * u.mM))
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([0.01]) * u.mM
        ion.reset_state(V)
        self.assertTrue(
            u.math.allclose(ion.Ci.value, jnp.array([9e-5]) * u.mM, atol=1e-12 * u.mM)
        )

    def test_zero_ical_derivative_relaxes_to_cali_base(self) -> None:
        ion = CdpLVA_SU2015_DCN(size=1, tauCal=70.0 * u.ms, caliBase=50e-6 * u.mM)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([80e-6]) * u.mM

        def zero_current(self, V, include_external=True):
            return jnp.zeros(self.varshape) * u.mA / u.cm ** 2

        ion.current = types.MethodType(zero_current, ion)
        ion.compute_derivative(V)

        expected = -(80e-6 - 50e-6) / 70.0
        got = ion.Ci.derivative.to_decimal(u.mM / u.ms)
        self.assertTrue(u.math.allclose(got, jnp.array([expected]), atol=1e-12))

    def test_positive_inward_current_increases_cali(self) -> None:
        ion = CdpLVA_SU2015_DCN(size=1, tauCal=70.0 * u.ms, caliBase=50e-6 * u.mM)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([50e-6]) * u.mM
        deriv = ion.derivative(
            ion.Ci.value,
            V,
            total_current=jnp.array([0.01]) * u.mA / u.cm ** 2,
        ).to_decimal(u.mM / u.ms)
        self.assertGreater(float(deriv[0]), 0.0)

    def test_none_total_current_is_treated_as_zero_ical(self) -> None:
        ion = CdpLVA_SU2015_DCN(size=1, tauCal=70.0 * u.ms, caliBase=50e-6 * u.mM)
        V = _V([-60.0])
        ion.init_state(V)
        ion.Ci.value = jnp.array([80e-6]) * u.mM
        got = ion.derivative(ion.Ci.value, V, total_current=None).to_decimal(u.mM / u.ms)
        expected = jnp.array([-(80e-6 - 50e-6) / 70.0])
        self.assertTrue(u.math.allclose(got, expected, atol=1e-12))

    def test_pack_info_uses_runtime_Ci(self) -> None:
        ion = CdpLVA_SU2015_DCN(size=1)
        ion.init_state(_V([-60.0]))
        ion.Ci.value = jnp.array([9e-5]) * u.mM
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))


class CalciumFirstOrderTest(unittest.TestCase):
    """The simpler first-order calcium buffering model."""

    def test_is_subclass_of_calcium(self) -> None:
        self.assertTrue(issubclass(CalciumFirstOrder, Calcium))
        self.assertTrue(issubclass(CalciumFirstOrder, DynamicNernstIon))
        self.assertTrue(CalciumFirstOrder.uses_total_current)

    def test_default_parameters(self) -> None:
        cfo = CalciumFirstOrder(size=1)
        self.assertTrue(u.math.allclose(cfo.alpha, jnp.asarray(0.13), atol=1e-9))
        self.assertTrue(u.math.allclose(cfo.beta, jnp.asarray(0.075), atol=1e-9))
        self.assertTrue(u.math.allclose(cfo.Co, 2.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(
            u.math.allclose(cfo.temp, u.celsius2kelvin(36.0), atol=1e-6 * u.kelvin)
        )

    def test_custom_parameters_are_honoured(self) -> None:
        cfo = CalciumFirstOrder(
            size=1,
            alpha=0.2,
            beta=0.1,
            Co=1.5 * u.mM,
        )
        self.assertTrue(u.math.allclose(cfo.alpha, jnp.asarray(0.2), atol=1e-9))
        self.assertTrue(u.math.allclose(cfo.beta, jnp.asarray(0.1), atol=1e-9))
        self.assertTrue(u.math.allclose(cfo.Co, 1.5 * u.mM, atol=1e-9 * u.mM))

    def test_init_state_creates_diffeq_state_Ci(self) -> None:
        cfo = CalciumFirstOrder(size=3)
        cfo.init_state(_V([-60.0, -55.0, -50.0]))
        self.assertIsInstance(cfo.Ci, DiffEqState)
        self.assertEqual(cfo.Ci.value.shape, (3,))

    def test_custom_Ci_initializer_is_respected(self) -> None:
        cfo = CalciumFirstOrder(
            size=1,
            Ci_initializer=braintools.init.Constant(5e-4 * u.mM),
        )
        cfo.init_state(_V([-60.0]))
        self.assertTrue(
            u.math.allclose(cfo.Ci.value, jnp.array([5e-4]) * u.mM, atol=1e-12 * u.mM)
        )

    def test_reset_state_restores_Ci_to_initializer(self) -> None:
        cfo = CalciumFirstOrder(size=1)
        V = _V([-60.0])
        cfo.init_state(V)
        cfo.Ci.value = jnp.array([0.01]) * u.mM
        cfo.reset_state(V)
        self.assertTrue(
            u.math.allclose(
                cfo.Ci.value, jnp.array([2.4e-4]) * u.mM, atol=1e-12 * u.mM
            )
        )

    def test_E_follows_nernst_formula(self) -> None:
        cfo = CalciumFirstOrder(size=1)
        cfo.init_state(_V([-60.0]))
        expected = (
            u.gas_constant * cfo.temp / (2 * u.faraday_constant)
            * u.math.log(cfo.Co / cfo.Ci.value)
        )
        self.assertTrue(u.math.allclose(cfo.E, expected, atol=1e-6 * u.mV))


if __name__ == "__main__":
    unittest.main()
