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

"""Tests for the nonspecific current-owner placeholder ion."""

import unittest

import brainunit as u
import jax.numpy as jnp

from braincell._base_ion import Ion
from braincell._base_neuron import HHTypedNeuron
from braincell.ion._base import FixedIon
from braincell.ion._testing import V, FixedIonContractTests
from braincell.ion.nonspecific import NonSpecific, NonSpecificFixed
from braincell.mech import get_registry


class NonSpecificBaseTest(unittest.TestCase):
    """The abstract placeholder family."""

    def test_is_an_ion_but_declares_no_real_species(self) -> None:
        self.assertTrue(issubclass(NonSpecific, Ion))
        self.assertEqual(NonSpecific.ion_symbol, "no")

    def test_root_type_is_inherited_from_ion(self) -> None:
        self.assertIs(NonSpecific.root_type, HHTypedNeuron)
        self.assertIs(NonSpecific.root_type, Ion.root_type)
        self.assertNotIn("root_type", NonSpecific.__dict__)

    def test_placeholder_defaults_are_the_documented_ones(self) -> None:
        # The docstring is explicit that these are arbitrary values chosen
        # so the ordinary Ion interfaces resolve, not physiological data.
        # Pinning them keeps that claim and the code in step.
        self.assertTrue(u.math.allclose(NonSpecific.default_Ci, 1.0 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(NonSpecific.default_Co, 1.0 * u.mM, atol=1e-12 * u.mM))
        self.assertEqual(NonSpecific.default_valence, 1)


class NonSpecificFixedContractTest(FixedIonContractTests, unittest.TestCase):
    """The shared ``FixedIon`` contract, for the nonspecific placeholder."""

    ION_CLASS = NonSpecificFixed
    FAMILY_CLASS = NonSpecific
    DEFAULT_E = 0.0 * u.mV
    DEFAULT_CI = 1.0 * u.mM
    DEFAULT_CO = 1.0 * u.mM
    DEFAULT_VALENCE = 1


class NonSpecificFixedTest(unittest.TestCase):
    """Behaviour specific to :class:`NonSpecificFixed`."""

    def test_is_registered_under_its_own_name(self) -> None:
        self.assertIs(get_registry().get("ion", "NonSpecificFixed"), NonSpecificFixed)

    def test_builds_on_the_fixed_ion_template(self) -> None:
        self.assertTrue(issubclass(NonSpecificFixed, FixedIon))

    def test_reversal_potential_defaults_to_zero_not_to_a_family_default(self) -> None:
        # Unlike the real species, a nonspecific owner has no Nernst
        # potential to fall back on, so its constructor supplies 0 mV
        # rather than leaving E unset.
        self.assertTrue(u.math.allclose(NonSpecificFixed(size=1).E, 0.0 * u.mV, atol=1e-12 * u.mV))

    def test_explicit_reversal_potential_is_honoured(self) -> None:
        no = NonSpecificFixed(size=2, E=-40.0 * u.mV)
        self.assertTrue(u.math.allclose(no.E, -40.0 * u.mV, atol=1e-12 * u.mV))

    def test_placeholder_concentrations_can_be_overridden(self) -> None:
        no = NonSpecificFixed(size=1, Ci=0.3 * u.mM, Co=4.0 * u.mM, valence=2)
        self.assertTrue(u.math.allclose(no.Ci, 0.3 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(no.Co, 4.0 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(no.valence, jnp.full((1,), 2), atol=1e-12))

    def test_carries_a_channel_and_reports_its_current(self) -> None:
        from braincell.channel.potassium_sodium import Kv1p5_MA2020_GrC

        # Kv1p5 is the shipped channel that declares a nonspecific current
        # owner, so it is the one that exercises this container for real.
        no = NonSpecificFixed(size=1, Ikv=Kv1p5_MA2020_GrC(size=1))
        self.assertIn("Ikv", no.channels)

    def test_external_current_is_added_to_the_total(self) -> None:
        no = NonSpecificFixed(size=1)
        delta = jnp.array([2.5]) * u.uA / u.cm**2

        def external(V_local, info):
            return delta

        no.register_external_current("ext", external)
        total = no.current(V([-60.0]), include_external=True)
        self.assertTrue(u.math.allclose(total.to_decimal(u.uA / u.cm**2), jnp.array([2.5]), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
