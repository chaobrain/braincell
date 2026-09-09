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

"""Protocol defaults and model lookup tests without simulator imports."""

from dataclasses import replace
import unittest

from .protocol import MODELS, Protocol, get_model


class ProtocolTest(unittest.TestCase):
    def test_original_notebook_protocols(self):
        expected = {
            "bc_ma2025": (0.05, 50, 10, 80, 0.05, 36, 64),
            "dcn_su2015": (0.1, 50, 10, 30, 0.1, 32, 64),
            "goc_ma2020": (0.1, 100, 10, 80, 0.2, 34, 32),
            "grc_ma2020": (0.1, 50, 10, 30, 0.01, 25, 64),
            "grc_ma2020_full": (0.1, 50, 10, 30, 0.01, 25, 64),
            "io_zh2019": (0.1, 100, 10, 80, 0.05, 36, 64),
            "pc_ma2024": (0.1, 20, 5, 10, 0.5, 36, 32),
            "sc_ma2021": (0.1, 100, 10, 80, 0.05, 32, 64),
        }
        self.assertEqual(set(MODELS), set(expected))
        for name, values in expected.items():
            with self.subTest(name=name):
                p = get_model(name).default_protocol
                self.assertEqual(
                    (p.dt_ms, p.duration_ms, p.delay_ms, p.stim_dur_ms, p.amp_nA, p.temperature_celsius, p.precision),
                    values,
                )
                self.assertEqual(p.v_init_mV, -65)

    def test_invalid_protocols(self):
        for options in (
            {"dt_ms": 0},
            {"duration_ms": -1},
            {"duration_ms": 0.15},
            {"precision": 16},
            {"amp_nA": float("nan")},
            {"dt_ms": True},
            {"delay_ms": -1},
            {"temperature_celsius": -273.15},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Protocol(**options)

    def test_signed_current_and_post_run_stimulus_are_valid(self):
        p = replace(Protocol(), amp_nA=-0.1, delay_ms=100, stim_dur_ms=0)
        self.assertEqual(p.amp_nA, -0.1)

    def test_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown cell"):
            get_model("missing")
