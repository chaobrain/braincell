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

"""Catalogue comparison with NEURON: the sodium, HCN and leak channels.

Every configuration and variant of a class in ``braincell.channel.sodium``,
``braincell.channel.hyperpolarization_activated`` or
``braincell.channel.leaky`` runs the five protocols against ``tolerances.py``
(see ``_catalogue.py``). ``Na_HH1952`` is compared with NEURON's built-in
``hh`` (potassium and leak conductances zero, rate table off) and ``IL`` with
NEURON's built-in ``pas``. The root-cause tests re-run the patch of each T1
entry of this family, which must bring it back to T0.
"""

import os
import unittest
from unittest import mock

import brainunit as u
import jax
import numpy as np
import pytest

import braincell.channel.hyperpolarization_activated as hcn
import braincell.channel.leaky as leaky
import braincell.channel.sodium as sodium
from braincell.mech import get_registry
from validation.neuron import _clamp
from validation.neuron._tolerance import CURRENT_CLAMP_PROTOCOLS, T0_LIMITS, VOLTAGE_CLAMP_PROTOCOLS
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
from ._helpers import TEMPLATES_ROOT, load_module

os.environ.setdefault("JAX_PLATFORMS", "cpu")

pytestmark = pytest.mark.slow

braincell_runner = load_module(TEMPLATES_ROOT / "braincell_runner.py", "channel_no_conc_runner_for_sodium_hcn_leak")

FAMILY = frozenset(
    name
    for name in (*sodium.__all__, *hcn.__all__, *leaky.__all__)
    if name in set(get_registry().names("channel"))
)
"""Registry names of the classes in the three modules (``LeakageChannel`` is an unregistered base)."""

CONFIGS = family_configs(FAMILY)
FAMILY_VARIANTS = family_variants(CONFIGS)
HH = "builtin/na_hh1952_hh"
PAS = "builtin/il_pas"
NA_ZH19 = "zh19_io/na_zh19_io"
NAV1P1 = "ma25_bc/nav1p1_ma25_bc"


def _mod_limit_branch(offset: float, shifted: float, scale: float):
    """``Na_ZH19_IO.mod``'s rate ``scale*y/(1 - exp(-y/10))``, ``y = v + offset``, taken at ``v + shifted`` near ``y = 0``."""

    def rate(self, V):
        v = V.to_decimal(u.mV)
        y = u.math.where(u.math.abs(v + offset) < 1e-6, v + shifted, v + offset)
        return scale * y / (1.0 - u.math.exp(-y / 10.0))

    return rate


def _exact_initial_state(config: str, protocol: str):
    """NEURON initializer that overwrites ``STEADYSTATE sparse``'s states with BrainCell's exact steady state."""
    case, commands = voltage_clamp.load_protocol(config_path(config), template_path(protocol))
    # BrainCell starts at the exact steady state; a constant clamp keeps it there.
    held = braincell_runner.run_case(case, solver=_clamp.voltage_replay_solver(np.full(commands[0].size + 1, commands[0][0])))
    steady = {gate.neuron: float(held["gates"][gate.braincell][0]) for gate in case.mapping_spec.gate_map}

    def initialize(h, segment, **kwargs):
        _clamp.initialize_neuron(h, segment, **kwargs)
        mechanism = getattr(segment, case.mapping_spec.neuron.mechanism_name)
        # I6 is the state the .mod's CONSERVE statement leaves implicit.
        for state, value in {**steady, "I6": 1.0 - sum(steady.values())}.items():
            setattr(mechanism, state, value)

    return initialize


class SodiumHcnLeakCatalogueTest(CatalogueAssertions, unittest.TestCase):
    def test_family_is_configured_or_listed(self):
        configured = {class_name(config) for config in CONFIGS}
        uncovered = FAMILY & set(UNCOVERED)
        self.assertEqual(len(FAMILY), 24)
        self.assertEqual(len(CONFIGS), 20)
        self.assertEqual(configured | uncovered, FAMILY)
        self.assertEqual(uncovered, {"Na_Ba2002", "Na_Leak", "Na_TM1991", "HCN_HM1992"})
        self.assertEqual({config for config in CONFIGS if config.startswith("builtin/")}, {HH, PAS})

    def test_every_configuration_stays_within_its_limits(self):
        skip_unless_compiled(self, CONFIGS)
        for config in CONFIGS:
            self.assert_protocols(config, config)

    def test_default_off_options_stay_within_their_limits(self):
        skip_unless_compiled(self, CONFIGS)
        self.assertEqual({name.split(":")[1] for name in FAMILY_VARIANTS}, {"gateCurrent=1"})
        self.assertEqual(len(FAMILY_VARIANTS), 2)
        self.assert_variants(FAMILY_VARIANTS)


class SodiumHcnLeakRootCauseTest(unittest.TestCase):
    """Patching only the named cause brings a T1 entry back to T0, or breaks a T0 one."""

    def test_na_zh19_mod_limit_branches_explain_vinit_and_the_ramp(self):
        skip_unless_compiled(self, (NA_ZH19,))
        # The ramp sample at t = 64.025 ms lies 5.7e-10 mV from -41 mV (a_m's branch);
        # the vinit case starts at -50 mV (b_h's branch).
        patches = (
            mock.patch.object(sodium.Na_ZH2019_IO, "_m_alpha", _mod_limit_branch(41.0, 41.000001, 0.1)),
            mock.patch.object(sodium.Na_ZH2019_IO, "_h_beta", _mod_limit_branch(50.0, 50.000001, 1.0)),
        )
        self.assertEqual(TABLE.tier(NA_ZH19, "vinit_celsius"), "T1")
        self.assertEqual(TABLE.tier(NA_ZH19, "vc_ramp"), "T1")
        with patches[0], patches[1]:
            self.assertLessEqual(current_clamp_error(NA_ZH19, "vinit_celsius"), T0_LIMITS["e_V"])
            result = voltage_clamp.compare_protocol(config_path(NA_ZH19), template_path("vc_ramp"))
        self.assertLessEqual(result["e_I"], T0_LIMITS["e_I"])
        self.assertLessEqual(result["e_s"], T0_LIMITS["e_s"])

    def test_nav1p1_exact_initial_state_explains_the_states(self):
        skip_unless_compiled(self, (NAV1P1,))
        for name in (NAV1P1, f"{NAV1P1}:gateCurrent=1"):
            variant = VARIANTS.get(name)
            params = None if variant is None else dict(variant.channel_params)
            currents = () if variant is None else variant.extra_neuron_currents
            for protocol in VOLTAGE_CLAMP_PROTOCOLS:
                with self.subTest(config=name, protocol=protocol):
                    self.assertEqual(TABLE.tier(name, protocol), "T1")
                    result = voltage_clamp.compare_protocol(
                        config_path(NAV1P1),
                        template_path(protocol),
                        initialize=_exact_initial_state(NAV1P1, protocol),
                        channel_params=params,
                        extra_neuron_currents=currents,
                    )
                    self.assertLessEqual(result["e_I"], T0_LIMITS["e_I"])
                    self.assertLessEqual(result["e_s"], T0_LIMITS["e_s"])

    def test_nav1p1_constant_gating_current_conductance_explains_the_current_clamp(self):
        name = f"{NAV1P1}:gateCurrent=1"
        variant = VARIANTS[name]
        skip_unless_compiled(self, (variant.config,))
        original = sodium.Nav1p1_MA2025_BC.current
        conductive = sodium.Nav1p6_MA2020_GoC.current

        def current(self, V, Na):
            # NEURON's conductance estimate holds igate constant in v (as for Kv1.1);
            # hide igate's v-dependence from BrainCell's linearization alike.
            frozen = u.Quantity(jax.lax.stop_gradient(V.mantissa), unit=V.unit)
            return conductive(self, V, Na) + (original(self, frozen, Na) - conductive(self, frozen, Na))

        with mock.patch.object(sodium.Nav1p1_MA2025_BC, "current", current):
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

        with mock.patch.object(voltage_clamp.neuron_runner, "disable_rate_table", table_on):
            result = voltage_clamp.compare_protocol(config_path(HH), template_path("vc_ramp"))
        self.assertGreater(result["e_I"], 1e-6)

    def test_hh_q10_reference_is_6p3_degC(self):
        # With Na_HH1952's default temp_ref = temp, the gates run 3**2.97 times slower than hh at 36 degC.
        skip_unless_compiled(self, (HH,))
        params = {"temp_ref_celsius": 36.0}
        result = voltage_clamp.compare_protocol(config_path(HH), template_path("vc_steps"), channel_params=params)
        self.assertGreater(result["e_s"], 0.1)

    def test_pas_comparison_sees_a_change_of_the_leak_current(self):
        # The leak is stateless and its voltage-clamp e_I is exactly 0; a 0.1 % change must show.
        skip_unless_compiled(self, (PAS,))
        original = leaky.IL.current

        def current(self, V, *args):
            return 1.001 * original(self, V, *args)

        with mock.patch.object(leaky.IL, "current", current):
            result = voltage_clamp.compare_protocol(config_path(PAS), template_path("vc_steps"))
            self.assertGreater(result["e_I"], 5e-4)
            for protocol in CURRENT_CLAMP_PROTOCOLS:
                with self.subTest(protocol=protocol):
                    self.assertGreater(current_clamp_error(PAS, protocol), T0_LIMITS["e_V"])


if __name__ == "__main__":
    unittest.main()
