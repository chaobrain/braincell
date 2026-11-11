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

"""Catalogue comparison with NEURON: the potassium and sodium-activated potassium channels.

Every configuration and variant of a class in ``braincell.channel.potassium``
or ``braincell.channel.potassium_sodium`` runs the five protocols against
``tolerances.py`` (see ``_catalogue.py``). ``K_HH1952`` is compared with
NEURON's built-in ``hh`` (sodium and leak conductances zero, rate table off).
The root-cause tests re-run the patch of each T1 entry of this family, which
must bring it back to T0.
"""

import os
import unittest
from unittest import mock

import brainunit as u
import jax
import pytest

import braincell.channel.potassium as potassium
import braincell.channel.potassium_sodium as potassium_sodium
from validation.neuron._tolerance import CURRENT_CLAMP_PROTOCOLS, T0_LIMITS
from validation.neuron.channel_no_conc.tolerances import TABLE, UNCOVERED, VARIANTS

from ._catalogue import (
    CatalogueAssertions,
    class_name,
    config_path,
    current_clamp_error,
    family_configs,
    family_variants,
    skip_unless_compiled,
    template_path,
    voltage_clamp,
)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

pytestmark = pytest.mark.slow

FAMILY = frozenset(potassium.__all__) | frozenset(potassium_sodium.__all__)
"""Registry names of the classes in the two modules."""

CONFIGS = family_configs(FAMILY)
FAMILY_VARIANTS = family_variants(CONFIGS)
HH = "builtin/k_hh1952_hh"


class PotassiumCatalogueTest(CatalogueAssertions, unittest.TestCase):
    def test_family_is_configured_or_listed(self):
        configured = {class_name(config) for config in CONFIGS}
        uncovered = FAMILY & set(UNCOVERED)
        self.assertEqual(len(FAMILY), 40)
        self.assertEqual(len(CONFIGS), 31)
        self.assertEqual(configured | uncovered, FAMILY)
        self.assertEqual(
            uncovered,
            {
                "KA1_HM1992", "KA2_HM1992", "KDR_Ba2002", "KK2A_HM1992", "KK2B_HM1992", "KNI_Ya1989", "K_Leak",
                "K_TM1991", "NaKPump_Ca1999",
            },
        )

    def test_every_configuration_stays_within_its_limits(self):
        skip_unless_compiled(self, CONFIGS)
        for config in CONFIGS:
            self.assert_protocols(config, config)

    def test_default_off_options_stay_within_their_limits(self):
        skip_unless_compiled(self, CONFIGS)
        self.assertEqual(
            {name.split(":")[1] for name in FAMILY_VARIANTS},
            {"gateCurrent=1", "gnonspec"},
        )
        self.assertEqual(len(FAMILY_VARIANTS), 7)
        self.assert_variants(FAMILY_VARIANTS)


class PotassiumRootCauseTest(unittest.TestCase):
    """Patching only the named cause brings a T1 entry back to T0, or breaks a T0 one."""

    def test_kdr_zh19_mod_limit_branch_explains_the_ramp(self):
        config = "zh19_io/kdr_zh19_io"
        skip_unless_compiled(self, (config,))

        def mod_alpha(self, V):
            # The .mod evaluates a_n at v + 41.00001 when |v + 41| < 1e-6; BrainCell takes the exact limit.
            v = V.to_decimal(u.mV)
            y = u.math.where(u.math.abs(v + 41.0) < 1e-6, v + 41.00001, v + 41.0)
            return y / (1.0 - u.math.exp(-y / 10.0))

        self.assertEqual(TABLE.tier(config, "vc_ramp"), "T1")
        with mock.patch.object(potassium.Kdr_ZH2019_IO, "_n_alpha", mod_alpha):
            result = voltage_clamp.compare_protocol(config_path(config), template_path("vc_ramp"))
        self.assertLessEqual(result["e_I"], T0_LIMITS["e_I"])
        self.assertLessEqual(result["e_s"], T0_LIMITS["e_s"])

    def test_constant_gating_current_conductance_explains_the_current_clamp(self):
        name = "ma25_bc/kv1p1_ma25_bc:gateCurrent=1"
        variant = VARIANTS[name]
        skip_unless_compiled(self, (variant.config,))
        original = potassium._ExpRateGateCurrent.current

        def current(self, V, K):
            # NEURON's rates() runs in the DERIVATIVE block, so its conductance
            # estimate holds igate constant in v; hide igate's v-dependence alike.
            frozen = u.Quantity(jax.lax.stop_gradient(V.mantissa), unit=V.unit)
            conductive = self.g_max * self.conductance_factor(V, K) * (K.E - V)
            conductive_frozen = self.g_max * self.conductance_factor(frozen, K) * (K.E - frozen)
            return conductive + (original(self, frozen, K) - conductive_frozen)

        with mock.patch.object(potassium._ExpRateGateCurrent, "current", current):
            for protocol in CURRENT_CLAMP_PROTOCOLS:
                with self.subTest(protocol=protocol):
                    self.assertEqual(TABLE.tier(name, protocol), "T1")
                    error = current_clamp_error(variant.config, protocol, dict(variant.channel_params))
                    self.assertLessEqual(error, T0_LIMITS["e_V"])

    def test_hh_rate_table_breaks_the_ramp(self):
        skip_unless_compiled(self, (HH,))
        self.assertEqual(TABLE.tier(HH, "vc_ramp"), "T0")

        def table_on(h, mechanism_name):
            setattr(h, f"usetable_{mechanism_name}", 1)
            return True

        neuron_runner = voltage_clamp.neuron_runner
        with mock.patch.object(neuron_runner, "disable_rate_table", table_on):
            result = voltage_clamp.compare_protocol(config_path(HH), template_path("vc_ramp"))
        self.assertGreater(result["e_I"], 1e-6)

    def test_hh_q10_reference_is_6p3_degC(self):
        # With K_HH1952's default temp_ref = temp, the gate runs 3**2.97 times slower than hh at 36 degC.
        skip_unless_compiled(self, (HH,))
        params = {"temp_ref_celsius": 36.0}
        result = voltage_clamp.compare_protocol(config_path(HH), template_path("vc_steps"), channel_params=params)
        self.assertGreater(result["e_s"], 0.1)


if __name__ == "__main__":
    unittest.main()
