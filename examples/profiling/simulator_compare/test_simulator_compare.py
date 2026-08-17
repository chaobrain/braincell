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


class RunnerTest(unittest.TestCase):
    def test_batch_size_parser(self) -> None:
        self.assertEqual(
            run_benchmark.parse_csv_ints("10,100,1000,10000"),
            (10, 100, 1000, 10000),
        )

    def test_neuron_backend_rejects_more_than_ten_direct_cells(self) -> None:
        source = (HERE / "backend_neuron.py").read_text()
        self.assertIn("batch_size not in (1, 10)", source)

    def test_runner_projects_neuron_from_one_n10_measurement(self) -> None:
        measured = {
            "backend": "neuron",
            "batch_size": 10,
            "timing": common.timing_summary([2.0, 3.0, 4.0]),
        }
        runs = run_benchmark.project_neuron_run(measured, (10, 100, 1000))
        self.assertEqual([run["batch_size"] for run in runs], [10, 100, 1000])
        self.assertTrue(runs[0]["timing"]["measured"])
        self.assertEqual(runs[1]["timing"]["median_seconds"], 30.0)
        self.assertEqual(runs[2]["timing"]["median_seconds"], 300.0)
        self.assertTrue(all(not run["timing"]["measured"] for run in runs[1:]))

    def test_gpu_environment_requires_same_jax_stack(self) -> None:
        base = {
            "batch_size": 10,
            "timing": {"measured": True},
            "device": "cuda:0",
            "software": {"python": "3.11", "jax": "0.10.1", "jaxlib": "0.10.1"},
        }
        result = run_benchmark.validate_gpu_environment(
            [{**base, "backend": "braincell"}, {**base, "backend": "jaxley"}]
        )
        self.assertEqual(result["jax"], "0.10.1")

        mismatched = {**base, "backend": "jaxley", "software": {**base["software"], "jax": "0.6.2"}}
        with self.assertRaises(RuntimeError):
            run_benchmark.validate_gpu_environment([{**base, "backend": "braincell"}, mismatched])

    def test_scaling_analysis_reports_crossover(self) -> None:
        runs = []
        for backend, intercept, slope in (
            ("braincell", 0.6, 0.0003),
            ("jaxley", 0.4, 0.0005),
        ):
            for batch_size in (100, 1000, 10000):
                runs.append(
                    {
                        "backend": backend,
                        "batch_size": batch_size,
                        "timing": {
                            "measured": True,
                            "median_seconds": intercept + slope * batch_size,
                        },
                    }
                )
        result = run_benchmark.scaling_analysis(runs)
        self.assertAlmostEqual(result["fits"]["braincell"]["seconds_per_cell"], 0.0003)
        self.assertAlmostEqual(result["fits"]["jaxley"]["seconds_per_cell"], 0.0005)
        self.assertAlmostEqual(result["crossover_batch_size"], 1000.0)

    def test_json_writer_rejects_infinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ValueError):
                common.write_json(path, {"value": float("inf")})


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


if __name__ == "__main__":
    unittest.main()
