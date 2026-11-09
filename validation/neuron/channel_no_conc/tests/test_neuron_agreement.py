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

"""Voltage agreement with NEURON for the configurations the full sweep singled out.

Each entry bounds the largest absolute voltage difference over every case of
one template. Root causes and the reasoning behind each limit are in
``validation/neuron/known-differences.md`` ("Channel sweep differences").
"""

import os
from pathlib import Path
import unittest

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

_PARITY = 1e-6
"""Bound for configurations whose BrainCell class integrates like the ``.mod``."""


def _limits(vinit: float, dc: float, ac: float) -> dict[str, float]:
    return {"vinit_celsius": vinit, "dc": dc, "ac": ac}


_FROZEN_PARITY = _limits(_PARITY, _PARITY, _PARITY)

# Unfrozen GHK classes differentiate the drive in the implicit voltage step;
# the ``.mod`` computes it in the state block, so NEURON treats it as constant.
_CAV2P1_UNFROZEN = _limits(5e-4, 1.0, 2e-4)
_CAV3P1_UNFROZEN = _limits(0.06, 0.25, 0.25)
# Same cause, but the Cav3.3 current is small enough to stay within parity.
_CAV3P3_UNFROZEN = _limits(_PARITY, _PARITY, _PARITY)
# NEURON's forward-difference conductance on a stiff GHK current.
_CALVA = _limits(5e-4, 2e-4, 2e-4)
# The ``.mod``'s ``v + 50.000001`` branch at the vinit case ``v_init = -50 mV``.
_NA_ZH19 = _limits(5e-5, _PARITY, _PARITY)
# ``SOLVE kin STEADYSTATE sparse`` leaves NEURON's initial state 6e-9 off steady state.
_KCA2P2_STEADYSTATE_SPARSE = _limits(1e-5, 1e-5, 1e-5)

VOLTAGE_LIMITS_MV: dict[str, dict[str, float]] = {
    "ma24_pc/cav2p1_ma24_pc_frozen": _FROZEN_PARITY,
    "ma25_bc/cav2p1_ma25_bc_frozen": _FROZEN_PARITY,
    "ri21_sc/cav2p1_ri21_sc_frozen": _FROZEN_PARITY,
    "ma20_goc/cav3p1_ma20_goc_frozen": _FROZEN_PARITY,
    "ma24_pc/cav3p1_ma24_pc_frozen": _FROZEN_PARITY,
    "ma24_pc/cav3p3_ma24_pc_frozen": _FROZEN_PARITY,
    "ma24_pc/cav2p1_ma24_pc": _CAV2P1_UNFROZEN,
    "ma25_bc/cav2p1_ma25_bc": _CAV2P1_UNFROZEN,
    "ri21_sc/cav2p1_ri21_sc": _CAV2P1_UNFROZEN,
    "ma20_goc/cav3p1_ma20_goc": _CAV3P1_UNFROZEN,
    "ma24_pc/cav3p1_ma24_pc": _CAV3P1_UNFROZEN,
    "ma24_pc/cav3p1_test2_pc24": _CAV3P1_UNFROZEN,
    "ma24_pc/cav3p3_ma24_pc": _CAV3P3_UNFROZEN,
    "ri21_sc/cav3p3_ri21_sc": _CAV3P3_UNFROZEN,
    "su15_dcn/calva_su15_dcn": _CALVA,
    "zh19_io/na_zh19_io": _NA_ZH19,
    "ma20_goc/kca2p2_ma20_goc": _KCA2P2_STEADYSTATE_SPARSE,
    "ma20_grc/kca2p2_ma20_grc": _KCA2P2_STEADYSTATE_SPARSE,
    "ma25_bc/kca2p2_ma25_bc": _KCA2P2_STEADYSTATE_SPARSE,
    "ri21_sc/kca2p2_ri21_sc": _KCA2P2_STEADYSTATE_SPARSE,
}


def _max_voltage_difference(config_path: Path, template_path: Path) -> float:
    config = experiment_schema.load_sweep_config(config_path, template_path)
    worst = 0.0
    for payload in experiment_schema.expand_cases(config):
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        metrics = compare_module.compare_case(case)["metrics"]
        worst = max(worst, float(metrics["voltage"]["max_abs"]))
    return worst


class NeuronVoltageAgreementTest(unittest.TestCase):
    def test_limits_cover_every_declared_template(self) -> None:
        for name, limits in VOLTAGE_LIMITS_MV.items():
            model_config = experiment_schema.load_model_config(CHANNEL_NO_CONC_ROOT / "configs" / f"{name}.json")
            declared = {Path(path).stem for path in model_config.template_paths}
            self.assertEqual(declared, set(limits), name)

    def test_voltage_difference_stays_within_limit(self) -> None:
        for name, limits in VOLTAGE_LIMITS_MV.items():
            config_path = CHANNEL_NO_CONC_ROOT / "configs" / f"{name}.json"
            model_config = experiment_schema.load_model_config(config_path)
            if not (Path(model_config.mod_dir) / "x86_64" / "libnrnmech.so").resolve().exists():
                self.skipTest(f"NEURON mechanisms for {name} are not compiled into libnrnmech.so.")
            for template_path in model_config.template_paths:
                template = Path(template_path).stem
                with self.subTest(config=name, template=template):
                    observed = _max_voltage_difference(config_path, Path(template_path))
                    self.assertLessEqual(observed, limits[template], f"{name} {template}: {observed:.3e} mV")


if __name__ == "__main__":
    unittest.main()
