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

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import run_benchmark  # noqa: E402


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


class PortableRunnerTest(unittest.TestCase):
    def test_interpreters_default_to_running_python(self):
        self.assertEqual(run_benchmark.DEFAULT_JAX_PYTHON, Path(sys.executable))
        self.assertEqual(run_benchmark.DEFAULT_NEURON_PYTHON, Path(sys.executable))

    def test_explicit_gpu_outside_historical_pair_is_accepted(self):
        class StopBeforeMeasurement(Exception):
            pass
        with mock.patch.object(sys, "argv", ["run_benchmark.py", "--gpu-candidates", "0"]), \
             mock.patch.object(run_benchmark, "assert_morphology_asset"), \
             mock.patch.object(run_benchmark, "query_gpus", side_effect=StopBeforeMeasurement) as query:
            with self.assertRaises(StopBeforeMeasurement):
                run_benchmark.main()
            query.assert_called_once_with((0,))

    def test_gpu_arguments_are_validated_before_measurement(self):
        for args in ([], ["--gpu-candidates", "-1"], ["--gpu-candidates", ""]):
            with self.subTest(args=args), mock.patch.object(sys, "argv", ["run_benchmark.py", *args]), \
                 mock.patch.object(run_benchmark, "query_gpus") as query, \
                 mock.patch.object(run_benchmark, "run_child") as worker:
                with self.assertRaises(SystemExit):
                    run_benchmark.main()
                query.assert_not_called()
                worker.assert_not_called()

    def test_neuron_only_does_not_require_gpu_selection(self):
        class StopBeforeMeasurement(Exception):
            pass
        with mock.patch.object(sys, "argv", ["run_benchmark.py", "--backend", "neuron"]), \
             mock.patch.object(run_benchmark, "assert_morphology_asset", side_effect=StopBeforeMeasurement), \
             mock.patch.object(run_benchmark, "query_gpus") as query:
            with self.assertRaises(StopBeforeMeasurement):
                run_benchmark.main()
            query.assert_not_called()
