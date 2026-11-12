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

"""Shared body of the catalogue comparison tests (``test_catalogue_<family>.py``).

A family is a set of registry names. Every configuration of one of its classes
runs the three current-clamp templates it lists (``test_coverage`` checks
the list) and both voltage-clamp templates; the limits come from ``tolerances.TABLE`` (T0 unless an entry
raises them), leaving out the commands listed in ``NON_FINITE_REFERENCE``.
The family's ``VARIANTS`` run the same five protocols with their channel
parameters.
"""

import json
from pathlib import Path
import unittest

import numpy as np

from validation.neuron._tolerance import CURRENT_CLAMP_PROTOCOLS, T0_LIMITS, VOLTAGE_CLAMP_PROTOCOLS
from validation.neuron.channel_no_conc.tolerances import NON_FINITE_REFERENCE, TABLE, VARIANTS, Variant

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, compile_hint, load_module, uncompiled_models

experiment_schema = load_module(TEMPLATES_ROOT / "experiment_schema.py", "channel_no_conc_schema_for_catalogue")
compare_module = load_module(TEMPLATES_ROOT / "compare.py", "channel_no_conc_compare_for_catalogue")
voltage_clamp = load_module(TEMPLATES_ROOT / "voltage_clamp.py", "channel_no_conc_voltage_clamp_for_catalogue")


def config_path(config: str) -> Path:
    return CHANNEL_NO_CONC_ROOT / "configs" / f"{config}.json"


def template_path(protocol: str) -> Path:
    return CHANNEL_NO_CONC_ROOT / "templates" / f"{protocol}.json"


def class_name(config: str) -> str:
    return experiment_schema.load_model_config(config_path(config)).mapping_spec.braincell.class_name


def family_configs(family: frozenset[str]) -> tuple[str, ...]:
    """Configurations (paths under ``configs/`` without ``.json``) of the family's classes."""
    root = CHANNEL_NO_CONC_ROOT / "configs"
    names = (str(path.relative_to(root).with_suffix("")) for path in root.rglob("*.json"))
    return tuple(sorted(name for name in names if class_name(name) in family))


def family_variants(configs: tuple[str, ...]) -> dict[str, Variant]:
    return {name: variant for name, variant in VARIANTS.items() if variant.config in configs}


def current_clamp_error(config: str, protocol: str, channel_params: dict | None = None) -> float:
    sweep = experiment_schema.load_sweep_config(config_path(config), template_path(protocol))
    errors = []
    for payload in experiment_schema.expand_cases(sweep):
        payload = {**payload, "channel_params": {**payload["channel_params"], **(channel_params or {})}}
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        errors.append(compare_module.compare_case(case)["metrics"]["voltage"]["max_abs"])
    return float(np.max(errors))  # NaN propagates and fails the assertion


def skip_unless_compiled(test: unittest.TestCase, configs: tuple[str, ...]) -> None:
    missing = uncompiled_models(experiment_schema.load_model_config(config_path(config)).mod_dir for config in configs)
    if missing:
        test.skipTest(compile_hint(missing))


class CatalogueAssertions:
    """``unittest.TestCase`` mixin that asserts configurations and variants against the table."""

    def skipped_commands(self, config: str, protocol: str) -> tuple[int, ...]:
        """Indices of the commands with a non-finite NEURON run; checks they are still not finite."""
        if (config, protocol) not in NON_FINITE_REFERENCE:
            return ()
        test_mV, _ = NON_FINITE_REFERENCE[(config, protocol)]
        case, commands = voltage_clamp.load_protocol(config_path(config), template_path(protocol))
        index = json.loads(template_path(protocol).read_text())["voltage_clamp"]["test_mV"].index(test_mV)
        neuron = voltage_clamp.run_neuron_command(case, commands[index])
        self.assertFalse(np.all(np.isfinite(neuron["ix"])), (config, protocol, test_mV))
        return (index,)

    def assert_protocols(self, name: str, config: str, variant: Variant | None = None) -> dict[str, float]:
        """Assert the five protocols of ``config`` (or of ``variant``, table key ``name``)."""
        params = None if variant is None else dict(variant.channel_params)
        currents = () if variant is None else variant.extra_neuron_currents
        measured = {}
        for protocol in CURRENT_CLAMP_PROTOCOLS:
            with self.subTest(config=name, protocol=protocol):
                measured[protocol] = current_clamp_error(config, protocol, params)
                self.assertLessEqual(measured[protocol], TABLE.limit(name, protocol, "e_V"), f"{measured[protocol]:.3e} mV")
        for protocol in VOLTAGE_CLAMP_PROTOCOLS:
            with self.subTest(config=name, protocol=protocol):
                result = voltage_clamp.compare_protocol(
                    config_path(config),
                    template_path(protocol),
                    skip_commands=self.skipped_commands(config, protocol),
                    channel_params=params,
                    extra_neuron_currents=currents,
                )
                # The clamp follows its command; calva_su15_dcn's current holds it 3.3e-6 mV off.
                # BrainCell replays NEURON's voltage, so this does not enter e_I or e_s.
                self.assertLessEqual(result["clamp_error_mV"], 1e-5)
                for metric in ("e_I", "e_s"):
                    if result[metric] is None:  # e_s of a channel without gates (IL)
                        self.assertEqual(len(result["case"].mapping_spec.gate_map), 0, config)
                        continue
                    self.assertLessEqual(result[metric], TABLE.limit(name, protocol, metric), f"{metric}: {result[metric]:.3e}")
        return measured

    def assert_variants(self, variants: dict[str, Variant]) -> None:
        """Assert each variant, and that its parameters reach both sides."""
        for name, variant in variants.items():
            measured = self.assert_protocols(name, variant.config, variant)
            # BrainCell: each T1 current-clamp protocol leaves the configuration's T0.
            for protocol in CURRENT_CLAMP_PROTOCOLS:
                if TABLE.tier(name, protocol) == "T1":
                    self.assertGreater(measured[protocol], TABLE.limit(variant.config, protocol, "e_V"), (name, protocol))
            # Both sides: without NEURON's extra current, BrainCell's added current is unmatched.
            if variant.extra_neuron_currents:
                result = voltage_clamp.compare_protocol(
                    config_path(variant.config), template_path("vc_steps"), channel_params=dict(variant.channel_params)
                )
                self.assertGreater(result["e_I"], T0_LIMITS["e_I"], name)
