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

"""Voltage-clamp agreement with NEURON: one configuration per channel template.

NEURON clamps the soma with ``SEClamp`` under ``templates/vc_steps.json`` and
``templates/vc_ramp.json``; BrainCell replays NEURON's voltage. Limits come
from ``tolerances.py`` (T0 unless an entry raises them).
"""

import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from validation.neuron import _clamp
from validation.neuron._tolerance import VOLTAGE_CLAMP_PROTOCOLS
from validation.neuron.channel_no_conc.tolerances import TABLE

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, load_module

os.environ.setdefault("JAX_PLATFORMS", "cpu")

voltage_clamp = load_module(TEMPLATES_ROOT / "voltage_clamp.py", "channel_no_conc_voltage_clamp_for_test")

T0_CONFIGS = (
    "ma20_goc/kv3p4_ma20_goc",
    "ma20_goc/nav1p6_ma20_goc",
    "ma20_goc/cav3p1_ma20_goc",
    "ma20_goc/cav3p1_ma20_goc_frozen",
    "ma20_goc/cav1p2_ma20_goc",
    "ma24_pc/cav2p1_ma24_pc",
)


def _paths(config: str, protocol: str) -> tuple[Path, Path]:
    return CHANNEL_NO_CONC_ROOT / "configs" / f"{config}.json", CHANNEL_NO_CONC_ROOT / "templates" / f"{protocol}.json"


def _skip_unless_compiled(test: unittest.TestCase, config: str) -> None:
    case, _ = voltage_clamp.load_protocol(*_paths(config, "vc_steps"))
    if not (Path(case.mod_dir) / "x86_64" / "libnrnmech.so").exists():
        test.skipTest(f"NEURON mechanisms for {config} are not compiled into libnrnmech.so.")


def _initialize_after_finitialize(h, segment, *, v_init_mV, dt_ms, celsius, concentrations=None):
    """The wrong order: concentrations reach the segment only after finitialize."""
    _clamp.initialize_neuron(h, segment, v_init_mV=v_init_mV, dt_ms=dt_ms, celsius=celsius)
    for name, value in (concentrations or {}).items():
        setattr(segment, name, float(value))


class VoltageClampAgreementTest(unittest.TestCase):
    def test_tiers(self):
        for config in T0_CONFIGS:
            for protocol in VOLTAGE_CLAMP_PROTOCOLS:
                self.assertEqual(TABLE.tier(config, protocol), "T0", (config, protocol))

    def test_currents_and_states_stay_within_limits(self):
        for config in T0_CONFIGS:
            _skip_unless_compiled(self, config)
            for protocol in VOLTAGE_CLAMP_PROTOCOLS:
                with self.subTest(config=config, protocol=protocol):
                    result = voltage_clamp.compare_protocol(*_paths(config, protocol))
                    self.assertLessEqual(result["clamp_error_mV"], 1e-7)
                    for metric in ("e_I", "e_s"):
                        limit = TABLE.limit(config, protocol, metric)
                        self.assertLessEqual(result[metric], limit, f"{config} {protocol} {metric}: {result[metric]:.3e}")


class VoltageClampSensitivityTest(unittest.TestCase):
    """Each perturbation must push a T0 configuration out of T0."""

    def test_detects_a_gate_time_constant_off_by_0p1_percent(self):
        from braincell.channel import Kv3p4_MA2020_GoC

        config = "ma20_goc/kv3p4_ma20_goc"
        _skip_unless_compiled(self, config)
        original = Kv3p4_MA2020_GoC.f_m_tau
        with mock.patch.object(Kv3p4_MA2020_GoC, "f_m_tau", lambda self, V, K: 1.001 * original(self, V, K)):
            result = voltage_clamp.compare_protocol(*_paths(config, "vc_steps"))
        self.assertGreater(result["e_I"], TABLE.limit(config, "vc_steps", "e_I"))
        self.assertGreater(result["e_I"], 1e-4)

    def test_detects_replaying_the_step_start_voltage(self):
        config = "ma20_goc/kv3p4_ma20_goc"
        _skip_unless_compiled(self, config)
        result = voltage_clamp.compare_protocol(
            *_paths(config, "vc_steps"),
            replay_voltages=lambda v: np.concatenate([v[:1], v[:-1]]),
        )
        self.assertGreater(result["e_I"], 0.1)
        self.assertGreater(result["e_s"], TABLE.limit(config, "vc_steps", "e_s"))

    def test_concentrations_reach_neuron_before_finitialize(self):
        config = "ma20_goc/cav1p2_ma20_goc"
        _skip_unless_compiled(self, config)
        case, commands = voltage_clamp.load_protocol(*_paths(config, "vc_steps"))
        traces = voltage_clamp.run_neuron_command(case, commands[0])
        self.assertEqual(traces["concentrations"]["cai"][0], case.ion_state.Ci_mM)

        wrong = voltage_clamp.compare_protocol(*_paths(config, "vc_steps"), initialize=_initialize_after_finitialize)
        self.assertGreater(wrong["e_s"], 0.1)


if __name__ == "__main__":
    unittest.main()
