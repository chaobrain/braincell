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


class CommonTest(unittest.TestCase):
    def test_morphology_asset_hash(self) -> None:
        self.assertEqual(common.morphology_sha256(), common.MORPHOLOGY_SHA256)

    def test_amplitudes_are_deterministic_and_nonidentical(self) -> None:
        self.assertEqual(common.current_amplitudes(1), [0.7])
        values = common.current_amplitudes(10)
        self.assertEqual(len(values), 10)
        self.assertAlmostEqual(values[0], 0.69)
        self.assertAlmostEqual(values[-1], 0.71)
        self.assertEqual(len(set(values)), 10)

    def test_neuron_extrapolation_scales_samples_and_iqr(self) -> None:
        measured = common.add_throughput(
            common.timing_summary([1.0, 2.0, 3.0]), batch_size=10
        )
        projected = common.extrapolate_timing(measured, 100, source_size=10)
        self.assertEqual(projected["samples_seconds"], [100.0, 200.0, 300.0])
        self.assertEqual(projected["median_seconds"], 200.0)
        self.assertFalse(projected["measured"])
        self.assertEqual(projected["extrapolated_from"], 10)
        self.assertEqual(
            projected["cell_steps_per_second"],
            measured["cell_steps_per_second"],
        )

    @mock.patch("common.time.sleep")
    @mock.patch("common.subprocess.check_output")
    def test_gpu_selection_is_restricted_and_uses_median_utilization(self, check_output, _sleep) -> None:
        check_output.side_effect = [
            "2, GPU-two, 8, 1000, 81920\n3, GPU-three, 3, 2000, 81920\n",
            "2, GPU-two, 2, 1000, 81920\n3, GPU-three, 4, 2000, 81920\n",
            "2, GPU-two, 7, 1000, 81920\n3, GPU-three, 5, 2000, 81920\n",
        ]
        selected = common.query_gpus((2, 3), interval_seconds=0.0)
        self.assertEqual(selected["selected"]["physical_id"], 3)

    def test_accuracy_gate_accepts_identical_traces(self) -> None:
        trace = [[-62.0] * common.N_STEPS for _ in common.PROBE_BRANCHES]
        result = common.compare_traces({"neuron": trace, "braincell": trace, "jaxley": trace})
        self.assertTrue(result["passed"])

    def test_accuracy_gate_rejects_wrong_spike_count(self) -> None:
        reference = [[-62.0] * common.N_STEPS for _ in common.PROBE_BRANCHES]
        candidate = [row[:] for row in reference]
        candidate[0][100:102] = [10.0, -62.0]
        result = common.compare_traces({"neuron": reference, "braincell": candidate})
        self.assertFalse(result["passed"])
