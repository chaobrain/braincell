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

import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import HHTypedNeuron, Ion, IonInfo
from braincell.channel.calcium import CaT_HM1992
from braincell.ion.calcium import (
    Calcium,
    CalciumDetailed,
    CalciumFirstOrder,
    CalciumFixed,
    CalciumInitNernst,
)
from braincell.ion._base import DynamicNernstIon, InitNernstIon
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
