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
and the testing Kv and chloride-leak configurations.

Each configuration's limit under one template bounds the largest absolute
voltage difference (``e_V``) over every case of that template. The limits
come from the suite's tolerance table (``tolerances.py``): T0 unless a T1
entry, whose root cause is in ``validation/neuron/known-differences.md``,
raises it.
"""

import os
from pathlib import Path
import unittest

import numpy as np
import pytest

from validation.neuron.channel_no_conc.tolerances import TABLE

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, compile_hint, load_module, uncompiled_models


# Slow tier: runs nightly, not on every pull request. Every configuration here
# except testing/cl_leak_testing is also compared, under every template it
# lists, by the nightly catalogue modules (test_catalogue_*.py).
pytestmark = pytest.mark.slow


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
    "zh19_io/hcn_hm1992",
    "ma20_goc/kca2p2_ma20_goc",
    "ma20_grc/kca2p2_ma20_grc",
    "ma25_bc/kca2p2_ma25_bc",
    "ri21_sc/kca2p2_ri21_sc",
    "testing/kv_testing",
    "testing/cl_leak_testing",
)
"""Configurations asserted under every current-clamp template they declare."""


def _max_voltage_difference(config_path: Path, template_path: Path) -> float:
    config = experiment_schema.load_sweep_config(config_path, template_path)
    errors = [
        compare_module.compare_case(experiment_schema.ChannelNoConcCase.from_dict(payload))["metrics"]["voltage"]["max_abs"]
        for payload in experiment_schema.expand_cases(config)
    ]
    return float(np.max(errors))  # NaN propagates; max() would drop one that is not first


class NeuronVoltageAgreementTest(unittest.TestCase):
    def test_voltage_difference_stays_within_limit(self) -> None:
        for name in CONFIGS:
            config_path = CHANNEL_NO_CONC_ROOT / "configs" / f"{name}.json"
            model_config = experiment_schema.load_model_config(config_path)
            if missing := uncompiled_models((model_config.mod_dir,)):
                self.skipTest(compile_hint(missing))
            for template_path in model_config.template_paths:
                template = Path(template_path).stem
                limit = TABLE.limit(name, template, "e_V")
                with self.subTest(config=name, template=template):
                    observed = _max_voltage_difference(config_path, Path(template_path))
                    self.assertLessEqual(observed, limit, f"{name} {template}: {observed:.3e} mV")


if __name__ == "__main__":
    unittest.main()
