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


class PlotResultsTest(unittest.TestCase):
    def test_speedup_uses_neuron_time_as_numerator(self) -> None:
        neuron = {
            "backend": "neuron",
            "batch_size": 100,
            "timing": {"median_seconds": 20.0},
        }
        backend = {
            "backend": "braincell",
            "batch_size": 100,
            "timing": {
                "median_seconds": 4.0,
                "q1_seconds": 3.0,
                "q3_seconds": 5.0,
            },
        }
        values = plot_results.speedup_values(backend, neuron)
        self.assertEqual(values["median"], 5.0)
        self.assertEqual(values["q1"], 4.0)
        self.assertAlmostEqual(values["q3"], 20.0 / 3.0)

    def test_neuron_speedup_is_exactly_one(self) -> None:
        neuron = {
            "backend": "neuron",
            "batch_size": 10,
            "timing": {
                "median_seconds": 2.0,
                "q1_seconds": 1.5,
                "q3_seconds": 2.5,
            },
        }
        self.assertEqual(
            plot_results.speedup_values(neuron, neuron),
            {"median": 1.0, "q1": 1.0, "q3": 1.0},
        )
