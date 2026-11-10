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

"""Voltage agreement with NEURON for the configurations the full sweep singled out
and the testing Kv configuration.

Each configuration's limit under one template bounds the largest absolute
voltage difference (``e_V``) over every case of that template. The limits
come from the suite's tolerance table (``tolerances.py``): T0 unless a T1
entry, whose root cause is in ``validation/neuron/known-differences.md``,
raises it.
"""

import os
from pathlib import Path
import unittest

from validation.neuron._tolerance import CURRENT_CLAMP_PROTOCOLS
from validation.neuron.channel_no_conc.tolerances import KNOWN_DIFFERENCES, TABLE

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, load_module


os.environ.setdefault("JAX_PLATFORMS", "cpu")


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "channel_no_conc_experiment_schema_for_agreement_test",
)
compare_module = load_module(
    TEMPLATES_ROOT / "compare.py",
    "channel_no_conc_compare_for_agreement_test",
)

CONFIGS = (
    "ma24_pc/cav2p1_ma24_pc_frozen",
    "ma25_bc/cav2p1_ma25_bc_frozen",
    "ri21_sc/cav2p1_ri21_sc_frozen",
    "ma20_goc/cav3p1_ma20_goc_frozen",
    "ma24_pc/cav3p1_ma24_pc_frozen",
    "ma24_pc/cav3p3_ma24_pc_frozen",
    "ma24_pc/cav2p1_ma24_pc",
    "ma25_bc/cav2p1_ma25_bc",
    "ri21_sc/cav2p1_ri21_sc",
    "ma20_goc/cav3p1_ma20_goc",
    "ma24_pc/cav3p1_ma24_pc",
    "ma24_pc/cav3p1_test2_pc24",
    "ma24_pc/cav3p3_ma24_pc",
    "ri21_sc/cav3p3_ri21_sc",
    "su15_dcn/calva_su15_dcn",
    "zh19_io/na_zh19_io",
    "ma20_goc/kca2p2_ma20_goc",
    "ma20_grc/kca2p2_ma20_grc",
    "ma25_bc/kca2p2_ma25_bc",
    "ri21_sc/kca2p2_ri21_sc",
    "testing/kv_testing",
)
"""Configurations asserted under every current-clamp template they declare."""


def _max_voltage_difference(config_path: Path, template_path: Path) -> float:
    config = experiment_schema.load_sweep_config(config_path, template_path)
    worst = 0.0
    for payload in experiment_schema.expand_cases(config):
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        metrics = compare_module.compare_case(case)["metrics"]
        worst = max(worst, float(metrics["voltage"]["max_abs"]))
    return worst


class ToleranceTableTest(unittest.TestCase):
    def test_current_clamp_entries_name_asserted_templates(self) -> None:
        for entry in TABLE.entries:
            if entry.protocol not in CURRENT_CLAMP_PROTOCOLS:
                continue
            self.assertIn(entry.config, CONFIGS)
            model_config = experiment_schema.load_model_config(CHANNEL_NO_CONC_ROOT / "configs" / f"{entry.config}.json")
            self.assertIn(entry.protocol, {Path(path).stem for path in model_config.template_paths}, entry.config)

    def test_every_asserted_configuration_declares_the_current_clamp_templates(self) -> None:
        for name in CONFIGS:
            model_config = experiment_schema.load_model_config(CHANNEL_NO_CONC_ROOT / "configs" / f"{name}.json")
            self.assertEqual({Path(path).stem for path in model_config.template_paths}, set(CURRENT_CLAMP_PROTOCOLS), name)

    def test_every_t1_entry_names_a_known_differences_heading(self) -> None:
        self.assertEqual(TABLE.missing_sections(KNOWN_DIFFERENCES.read_text()), [])


class NeuronVoltageAgreementTest(unittest.TestCase):
    def test_voltage_difference_stays_within_limit(self) -> None:
        for name in CONFIGS:
            config_path = CHANNEL_NO_CONC_ROOT / "configs" / f"{name}.json"
            model_config = experiment_schema.load_model_config(config_path)
            if not (Path(model_config.mod_dir) / "x86_64" / "libnrnmech.so").resolve().exists():
                self.skipTest(f"NEURON mechanisms for {name} are not compiled into libnrnmech.so.")
            for template_path in model_config.template_paths:
                template = Path(template_path).stem
                limit = TABLE.limit(name, template, "e_V")
                with self.subTest(config=name, template=template):
                    observed = _max_voltage_difference(config_path, Path(template_path))
                    self.assertLessEqual(observed, limit, f"{name} {template}: {observed:.3e} mV")


if __name__ == "__main__":
    unittest.main()
