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

"""Consistency of the tolerance table with the configurations the agreement test asserts.

These checks read JSON configurations and the tolerance table only, so they run
in the fast tier on every pull request; the agreement test itself is slow-tier.
"""

from pathlib import Path
import unittest

from validation.neuron._tolerance import CURRENT_CLAMP_PROTOCOLS
from validation.neuron.channel_no_conc.tolerances import KNOWN_DIFFERENCES, TABLE, VARIANTS

from ._helpers import CHANNEL_NO_CONC_ROOT
from .test_neuron_agreement import CONFIGS, experiment_schema


class ToleranceTableTest(unittest.TestCase):
    def test_current_clamp_entries_name_asserted_templates(self) -> None:
        for entry in TABLE.entries:
            if entry.protocol not in CURRENT_CLAMP_PROTOCOLS:
                continue
            if entry.config in VARIANTS:  # asserted by the catalogue test of its family
                config = VARIANTS[entry.config][0]
            else:
                config = entry.config
                self.assertIn(config, CONFIGS)
            model_config = experiment_schema.load_model_config(CHANNEL_NO_CONC_ROOT / "configs" / f"{config}.json")
            self.assertIn(entry.protocol, {Path(path).stem for path in model_config.template_paths}, entry.config)

    def test_every_asserted_configuration_declares_the_current_clamp_templates(self) -> None:
        for name in CONFIGS:
            model_config = experiment_schema.load_model_config(CHANNEL_NO_CONC_ROOT / "configs" / f"{name}.json")
            self.assertEqual({Path(path).stem for path in model_config.template_paths}, set(CURRENT_CLAMP_PROTOCOLS), name)

    def test_every_t1_entry_names_a_known_differences_heading(self) -> None:
        self.assertEqual(TABLE.missing_sections(KNOWN_DIFFERENCES.read_text()), [])


if __name__ == "__main__":
    unittest.main()
