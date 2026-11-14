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

"""Catalogue comparison with NEURON: the calcium and calcium-activated potassium channels.

Every configuration and variant of a class in ``braincell.channel.calcium`` or
``braincell.channel.potassium_calcium`` runs the five protocols against
``tolerances.py`` (see ``_catalogue.py``). Two tests re-run the root-cause
patches of this family's voltage-clamp T1 entries, which must bring the
configuration back to T0.
"""

import json
import os
import unittest
from unittest import mock

import brainunit as u
import numpy as np
import pytest

import braincell.channel.calcium as calcium
import braincell.channel.potassium_calcium as potassium_calcium
from validation.neuron import _clamp
from validation.neuron._tolerance import T0_LIMITS, VOLTAGE_CLAMP_PROTOCOLS
from validation.neuron.channel_no_conc.tolerances import NON_FINITE_REFERENCE, TABLE, UNCOVERED

from ._catalogue import (
    CatalogueAssertions,
    class_name,
    config_path as _config_path,
    family_configs,
    family_variants,
    skip_unless_compiled,
    template_path as _template_path,
    voltage_clamp,
)
from ._helpers import TEMPLATES_ROOT, load_module

os.environ.setdefault("JAX_PLATFORMS", "cpu")

pytestmark = pytest.mark.slow

braincell_runner = load_module(TEMPLATES_ROOT / "braincell_runner.py", "channel_no_conc_runner_for_calcium")

FAMILY = frozenset(calcium.__all__) | frozenset(potassium_calcium.__all__)
"""Registry names of the classes in the two modules."""

CONFIGS = family_configs(FAMILY)
FAMILY_VARIANTS = family_variants(CONFIGS)


def _skip_unless_compiled(test: unittest.TestCase) -> None:
    skip_unless_compiled(test, CONFIGS)


class CalciumCatalogueTest(CatalogueAssertions, unittest.TestCase):
    def test_family_is_configured_or_listed(self):
        configured = {class_name(config) for config in CONFIGS}
        uncovered = FAMILY & set(UNCOVERED)
        self.assertEqual(len(FAMILY), 51)
        self.assertEqual(len(CONFIGS), 49)
        self.assertEqual(configured | uncovered, FAMILY)
        self.assertEqual(
            uncovered,
            {
                "AHP_De1994",
                "CaHT_Re1993",
                "CaL_Li2017",
                "CaN_IS2008",
                "CaT_HM1992",
                "CaT_HM1992_Shifted25",
                "CaT_HP1992",
            },
        )

    def test_every_configuration_stays_within_its_limits(self):
        _skip_unless_compiled(self)
        for config in CONFIGS:
            self.assert_protocols(config, config)

    def test_default_off_options_stay_within_their_limits(self):
        _skip_unless_compiled(self)
        self.assertEqual(set(FAMILY_VARIANTS), {"zh19_io/ca_zh19_io:freeze_m_inf=False"})
        self.assert_variants(FAMILY_VARIANTS)


class CalciumRootCauseTest(unittest.TestCase):
    """Patching only the named cause brings a T1 configuration back to T0."""

    def test_cav3p3_mod_ghk_formula_explains_the_ramp(self):
        config = "ma24_pc/cav3p3_ma24_pc"
        _skip_unless_compiled(self)

        def mod_ghk(V, ci, co, z, temp):
            # The .mod's w/(exp(w) - 1) without BrainCell's small-w branch.
            faraday, gas = calcium._CAV3P3_NMODL_FARADAY, calcium._CAV3P3_NMODL_GAS_CONSTANT
            w = (z * faraday * V) / (gas * (temp + calcium._CAV3P3_NMODL_TEMP_OFFSET))
            return -z * faraday * (co - ci * u.math.exp(w)) * w / (u.math.exp(w) - 1)

        self.assertEqual(TABLE.tier(config, "vc_ramp"), "T1")
        with mock.patch.object(calcium.Cav3p3_RI2021_SC, "ghk", staticmethod(mod_ghk)):
            result = voltage_clamp.compare_protocol(_config_path(config), _template_path("vc_ramp"))
        self.assertLessEqual(result["e_I"], 1e-10)

    def test_cav3p3_finite_mod_reference_compares_every_command(self):
        # MECH-43: against a copy of the .mod with a series branch at w = 0, the
        # 0 mV command left out against the original is finite, and the ramp's
        # T1 (the same 0/0 term near 0 mV) is T0.
        _skip_unless_compiled(self)
        self.assertEqual(len(NON_FINITE_REFERENCE), 3)
        for (config, protocol), (test_mV, _) in NON_FINITE_REFERENCE.items():
            finite = f"testing/{config.split('/')[1]}_finite"
            with self.subTest(config=config):
                self.assertEqual(protocol, "vc_steps")
                self.assertEqual(TABLE.tier(config, "vc_ramp"), "T1")
                case, commands = voltage_clamp.load_protocol(_config_path(finite), _template_path("vc_steps"))
                self.assertEqual(len(commands), 8)
                index = json.loads(_template_path("vc_steps").read_text())["voltage_clamp"]["test_mV"].index(test_mV)
                neuron = voltage_clamp.run_neuron_command(case, commands[index])
                self.assertTrue(np.all(np.isfinite(neuron["ix"])))
                for vc in VOLTAGE_CLAMP_PROTOCOLS:
                    self.assertEqual(TABLE.tier(finite, vc), "T0")
                    result = voltage_clamp.compare_protocol(_config_path(finite), _template_path(vc))
                    self.assertLessEqual(result["e_I"], T0_LIMITS["e_I"], vc)
                    self.assertLessEqual(result["e_s"], T0_LIMITS["e_s"], vc)

    def test_kca2p2_exact_initial_state_explains_the_steps(self):
        config = "ma20_goc/kca2p2_ma20_goc"
        _skip_unless_compiled(self)
        case, commands = voltage_clamp.load_protocol(_config_path(config), _template_path("vc_steps"))
        # BrainCell starts at the exact steady state; a constant clamp keeps it there.
        held = braincell_runner.run_case(case, solver=_clamp.voltage_replay_solver(np.full(commands[0].size + 1, commands[0][0])))
        steady = {gate.neuron: float(held["gates"][gate.braincell][0]) for gate in case.mapping_spec.gate_map}

        def initialize(h, segment, **kwargs):
            _clamp.initialize_neuron(h, segment, **kwargs)
            mechanism = getattr(segment, case.mapping_spec.neuron.mechanism_name)
            for state, value in {**steady, "c1": 1.0 - sum(steady.values())}.items():
                setattr(mechanism, state, value)

        self.assertEqual(TABLE.tier(config, "vc_steps"), "T1")
        result = voltage_clamp.compare_protocol(_config_path(config), _template_path("vc_steps"), initialize=initialize)
        self.assertLessEqual(result["e_I"], 1e-10)
        self.assertLessEqual(result["e_s"], 1e-10)


if __name__ == "__main__":
    unittest.main()
