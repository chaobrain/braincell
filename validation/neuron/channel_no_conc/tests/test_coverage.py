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

"""Every registered channel has a NEURON comparison configuration or a reason.

The configurations are the JSON files under ``configs/``; the reasons are
``tolerances.UNCOVERED``. A new channel class registered without either fails
here, as does a stale name in the list. Every case of every configuration
compares a channel that conducts: its conductance (or permeability) is above
zero.
"""

from collections.abc import Iterable, Mapping
import dataclasses
import unittest

import braincell.channel  # noqa: F401  (registers the channel classes)
from braincell.mech import get_registry, register_channel
from validation.neuron.channel_no_conc.tolerances import UNCOVERED

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, load_module

experiment_schema = load_module(TEMPLATES_ROOT / "experiment_schema.py", "channel_no_conc_experiment_schema_for_coverage")

CONFIGS = sorted((CHANNEL_NO_CONC_ROOT / "configs").rglob("*.json"))
VOLTAGE_CLAMP_TEMPLATES = tuple(CHANNEL_NO_CONC_ROOT / "templates" / f"{name}.json" for name in ("vc_steps", "vc_ramp"))
CONDUCTANCES = frozenset({"g_max", "perm"})
"""BrainCell parameter names of a channel's conductance or permeability."""


def _configured() -> set[str]:
    return {experiment_schema.load_model_config(path).mapping_spec.braincell.class_name for path in CONFIGS}


def _registered() -> set[str]:
    return set(get_registry().names("channel"))


def coverage_problems(registered: Iterable[str], configured: Iterable[str], uncovered: Mapping[str, str]) -> list[str]:
    """Return one message per channel that breaks the coverage rule."""
    registered, configured = set(registered), set(configured)
    problems = [f"{name}: registered, but no configuration and not in UNCOVERED" for name in sorted(registered - configured - set(uncovered))]
    problems += [f"{name}: in UNCOVERED but not a registered channel" for name in sorted(set(uncovered) - registered)]
    problems += [f"{name}: in UNCOVERED but has a configuration" for name in sorted(set(uncovered) & configured)]
    problems += [f"{name}: configuration names an unregistered class" for name in sorted(configured - registered)]
    problems += [f"{name}: empty reason" for name, reason in sorted(uncovered.items()) if not reason.strip()]
    return problems


def conductance_problems(model_config) -> list[str]:
    """Return one message per case of ``model_config`` whose channel does not conduct.

    Every template the configuration lists and both voltage-clamp templates
    are expanded; a case fails when a parameter mapped to a BrainCell
    conductance is missing or not above zero, and a configuration fails when
    it maps none.
    """
    name = model_config.config_path.stem
    keys = sorted(key for key, side in model_config.mapping_spec.parameter_map.items() if side.braincell in CONDUCTANCES)
    if not keys:
        return [f"{name}: maps no BrainCell parameter in {sorted(CONDUCTANCES)}"]
    problems = []
    for template_path in (*model_config.template_paths, *VOLTAGE_CLAMP_TEMPLATES):
        sweep = experiment_schema.build_sweep_config(model_config, experiment_schema.load_scan_template(template_path))
        for case in experiment_schema.expand_cases(sweep):
            for key in keys:
                value = case["channel_params"].get(key)
                if value is None or not float(value) > 0.0:
                    problems.append(f"{name} {template_path.stem} {case['case_id']}: {key} = {value!r}")
    return problems


class ChannelCoverageTest(unittest.TestCase):
    def test_every_registered_channel_is_configured_or_listed(self):
        registered, configured = _registered(), _configured()
        self.assertGreater(len(registered), 100)
        self.assertGreater(len(CONFIGS), 90)
        self.assertEqual(coverage_problems(registered, configured, UNCOVERED), [])
        self.assertEqual(len(registered), len(configured) + len(UNCOVERED))

    def test_every_configuration_compares_a_conducting_channel(self):
        checked = [experiment_schema.load_model_config(path) for path in CONFIGS]
        self.assertEqual(len(checked), len(CONFIGS))
        self.assertGreater(len(checked), 90)
        self.assertEqual([problem for config in checked for problem in conductance_problems(config)], [])

    def test_a_zero_or_missing_conductance_fails(self):
        config = experiment_schema.load_model_config(CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "cav2p3_ma20_goc.json")
        zero = dataclasses.replace(config, defaults={"channel_params": {"g_max_S_cm2": 0.0}})
        problems = conductance_problems(zero)
        self.assertEqual(len(problems), 3 + 4 + 3 + 1 + 1)  # vinit_celsius, dc, ac cases; vc_steps, vc_ramp
        self.assertTrue(all(problem.endswith("g_max_S_cm2 = 0.0") for problem in problems), problems)
        missing = dataclasses.replace(config, defaults={})
        self.assertEqual(len(conductance_problems(missing)), 12)

    def test_a_configuration_without_conductance_fails(self):
        config = experiment_schema.load_model_config(CHANNEL_NO_CONC_ROOT / "configs" / "builtin" / "k_hh1952_hh.json")
        parameter_map = {key: side for key, side in config.mapping_spec.parameter_map.items() if side.braincell != "g_max"}
        no_conductance = dataclasses.replace(
            config, mapping_spec=dataclasses.replace(config.mapping_spec, parameter_map=parameter_map)
        )
        self.assertEqual(conductance_problems(no_conductance), ["k_hh1952_hh: maps no BrainCell parameter in ['g_max', 'perm']"])

    def test_a_new_channel_without_configuration_fails(self):
        name = "CoverageProbe_Test"
        base = get_registry().get("channel", "K_Kv_test")
        register_channel(name)(type(name, (base,), {}))
        try:
            problems = coverage_problems(_registered(), _configured(), UNCOVERED)
        finally:
            get_registry().unregister("channel", name)
        self.assertEqual(problems, [f"{name}: registered, but no configuration and not in UNCOVERED"])

    def test_a_listed_name_that_is_not_registered_fails(self):
        problems = coverage_problems(_registered(), _configured(), {**UNCOVERED, "NoSuchChannel": "reason"})
        self.assertEqual(problems, ["NoSuchChannel: in UNCOVERED but not a registered channel"])

    def test_a_listed_configured_channel_and_an_empty_reason_fail(self):
        problems = coverage_problems(_registered(), _configured(), {**UNCOVERED, "K_Kv_test": " "})
        self.assertEqual(problems, ["K_Kv_test: in UNCOVERED but has a configuration", "K_Kv_test: empty reason"])


if __name__ == "__main__":
    unittest.main()
