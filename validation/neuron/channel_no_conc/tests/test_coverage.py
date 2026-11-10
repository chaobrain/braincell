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
here, as does a stale name in the list.
"""

from collections.abc import Iterable, Mapping
import unittest

import braincell.channel  # noqa: F401  (registers the channel classes)
from braincell.mech import get_registry, register_channel
from validation.neuron.channel_no_conc.tolerances import UNCOVERED

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, load_module

experiment_schema = load_module(TEMPLATES_ROOT / "experiment_schema.py", "channel_no_conc_experiment_schema_for_coverage")

CONFIGS = sorted((CHANNEL_NO_CONC_ROOT / "configs").rglob("*.json"))


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


class ChannelCoverageTest(unittest.TestCase):
    def test_every_registered_channel_is_configured_or_listed(self):
        registered, configured = _registered(), _configured()
        self.assertGreater(len(registered), 100)
        self.assertGreater(len(CONFIGS), 90)
        self.assertEqual(coverage_problems(registered, configured, UNCOVERED), [])
        self.assertEqual(len(registered), len(configured) + len(UNCOVERED))

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
