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

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common
import plot_diagnostics
import plot_results
import run_benchmark


class DiagnosticsTest(unittest.TestCase):
    def test_diagnostic_rows_derive_cold_start_iqr_and_bandwidth(self) -> None:
        payload = {
            "runs": [
                {
                    "backend": "braincell",
                    "batch_size": 100,
                    "build_seconds": 2.0,
                    "compilation_seconds": 3.0,
                    "timing": {
                        "measured": True,
                        "median_seconds": 4.0,
                        "q1_seconds": 3.8,
                        "q3_seconds": 4.2,
                        "cell_steps_per_second": 20_000.0,
                    },
                    "host_transfer": {
                        "median_seconds": 0.002,
                        "q1_seconds": 0.0015,
                        "q3_seconds": 0.0025,
                    },
                    "device_memory": {"peak_mib_in_use": 64.0},
                    "output_validation": {"trace_output_bytes": 2_000_000},
                }
            ]
        }
        row = plot_diagnostics.diagnostic_rows(payload)[0]
        self.assertEqual(row["cold_start_seconds"], 5.0)
        self.assertAlmostEqual(row["steady_relative_iqr_percent"], 10.0)
        self.assertAlmostEqual(row["effective_transfer_gb_per_second"], 1.0)
