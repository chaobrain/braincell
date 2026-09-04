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

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import brainunit as u
import jax
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import braincell_ablation
from backend_braincell import build_cell


class BrainCellAblationTest(unittest.TestCase):
    def test_prepare_case_rejects_invalid_configuration(self) -> None:
        invalid = (
            {"batch_size": 0, "batch_mode": "population", "spike_mode": "tracked", "duration_ms": 1.0},
            {"batch_size": 1, "batch_mode": "loop", "spike_mode": "tracked", "duration_ms": 1.0},
            {"batch_size": 1, "batch_mode": "population", "spike_mode": "maybe", "duration_ms": 1.0},
            {"batch_size": 1, "batch_mode": "population", "spike_mode": "tracked", "duration_ms": 0.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                braincell_ablation.prepare_case(**kwargs)

    @mock.patch("braincell_ablation.brainstate.environ.get", return_value=None)
    def test_spike_off_update_requires_dt(self, _get) -> None:
        with self.assertRaisesRegex(ValueError, "requires brainstate.environ"):
            braincell_ablation._update_dynamics_without_spikes(mock.Mock())

    def test_uniform_amplitude_shape_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "amplitudes_na must have shape"):
            build_cell(2, amplitudes_na=np.asarray([0.70], dtype=np.float32))

    def test_uniform_parameter_is_stored_as_scalar(self) -> None:
        cell, _ = build_cell(2, amplitudes_na=np.full((2,), 0.70, dtype=np.float32))
        _, parameters = braincell_ablation._inventory(cell)
        g_max = [row for row in parameters if row["path"].endswith("g_max")]
        self.assertTrue(g_max)
        self.assertTrue(all(row["stored_shape"] == [] for row in g_max))
        self.assertTrue(all(row["logical_shape"] == [2, cell.n_cv] for row in g_max))

    def test_spike_off_removes_bookkeeping_and_preserves_voltage(self) -> None:
        tracked = braincell_ablation.prepare_case(
            2,
            batch_mode="population",
            spike_mode="tracked",
            duration_ms=0.05,
        )
        off = braincell_ablation.prepare_case(
            2,
            batch_mode="population",
            spike_mode="off",
            duration_ms=0.05,
        )
        tracked.restore()
        off.restore()
        tracked_traces = tracked.normalize_host_traces(jax.device_get(tracked.simulate()))
        off_traces = off.normalize_host_traces(jax.device_get(off.simulate()))
        paths = {row["path"] for row in off.state_inventory}
        self.assertNotIn("spike", paths)
        self.assertNotIn("_event_previous_V", paths)
        np.testing.assert_allclose(off_traces, tracked_traces, rtol=1e-5, atol=1e-5)

    def test_vmap_matches_population(self) -> None:
        population = braincell_ablation.prepare_case(
            2,
            batch_mode="population",
            spike_mode="off",
            duration_ms=0.05,
        )
        vmapped = braincell_ablation.prepare_case(
            2,
            batch_mode="vmap",
            spike_mode="off",
            duration_ms=0.05,
        )
        population.restore()
        vmapped.restore()
        population_traces = population.normalize_host_traces(jax.device_get(population.simulate()))
        vmapped_traces = vmapped.normalize_host_traces(jax.device_get(vmapped.simulate()))
        self.assertEqual(population_traces.shape, (2, 3, 2))
        self.assertEqual(vmapped_traces.shape, population_traces.shape)
        np.testing.assert_allclose(vmapped_traces, population_traces, rtol=1e-5, atol=1e-5)

    def test_summary_reports_spike_and_vmap_effects(self) -> None:
        runs = []
        for batch_mode, tracked_time, off_time in (
            ("population", 4.0, 2.0),
            ("vmap", 3.0, 1.5),
        ):
            for spike_mode, runtime in (("tracked", tracked_time), ("off", off_time)):
                runs.append(
                    {
                        "batch_size": 10,
                        "case": {"batch_mode": batch_mode, "spike_mode": spike_mode},
                        "timing": {"median_seconds": runtime},
                        "device_memory": {"peak_mib_in_use": runtime * 10},
                        "output_validation": {"first_lane_trace_mv": [[[0.0, 1.0]]]},
                    }
                )
        summary = braincell_ablation.summarize_runs(runs)
        self.assertTrue(summary["passed"])
        spike_effects = [row for row in summary["effects"] if row["effect"] == "spike_off"]
        self.assertEqual([row["speedup"] for row in spike_effects], [2.0, 2.0])
        self.assertEqual(
            braincell_ablation._run_key(runs[0]),
            (10, "population", "tracked"),
        )

    def test_trace_validation_allows_fp32_noise_but_requires_spike_equivalence(self) -> None:
        def run(mode: str, trace) -> dict:
            return {
                "batch_size": 10,
                "case": {"batch_mode": mode, "spike_mode": "tracked"},
                "output_validation": {"first_lane_trace_mv": trace},
            }

        reference = [[-1.0, -0.5, 0.5, 1.0]]
        within_tolerance = [[-1.0, -0.5, 0.5001, 1.0]]
        rows = braincell_ablation._validate_traces([run("population", reference), run("vmap", within_tolerance)])
        vmapped = next(row for row in rows if row["batch_mode"] == "vmap")
        self.assertTrue(vmapped["passed"])
        self.assertTrue(vmapped["spike_crossings_match"])

        changed_crossing = [[-1.0, -0.5, -0.0001, -0.0001]]
        rows = braincell_ablation._validate_traces([run("population", reference), run("vmap", changed_crossing)])
        vmapped = next(row for row in rows if row["batch_mode"] == "vmap")
        self.assertFalse(vmapped["passed"])
        self.assertFalse(vmapped["spike_crossings_match"])

    def test_run_case_collects_timing_and_trace_metadata(self) -> None:
        batch_size = 2
        traces = np.zeros((batch_size, 3, braincell_ablation.N_STEPS), dtype=np.float32)
        prepared = braincell_ablation.PreparedCase(
            cell=object(),
            simulate=lambda: (np.zeros((1,), dtype=np.float32) * u.mV,),
            restore=mock.Mock(),
            normalize_host_traces=lambda _result: traces,
            state_inventory=(),
            parameter_inventory=(),
            morphology={"n_cv": 868},
        )
        with (
            mock.patch("braincell_ablation.assert_morphology_asset"),
            mock.patch("braincell_ablation.prepare_case", return_value=prepared),
            mock.patch("braincell_ablation._block"),
            mock.patch("braincell_ablation.jax.device_get", side_effect=lambda value: value),
            mock.patch("braincell_ablation.jax.devices", return_value=["cpu"]),
            mock.patch(
                "braincell_ablation._memory_metadata",
                return_value={"peak_bytes_in_use": 1024, "peak_mib_in_use": 1 / 1024, "bytes_limit": 2048},
            ),
            mock.patch("braincell_ablation.git_commit", return_value="commit"),
        ):
            result = braincell_ablation.run_case(
                batch_size,
                batch_mode="population",
                spike_mode="off",
                warmup=1,
                repeat=2,
                transfer_repeat=1,
            )
        self.assertEqual(result["output_validation"]["shape"], [2, 3, braincell_ablation.N_STEPS])
        self.assertEqual(len(result["timing"]["samples_seconds"]), 2)
        self.assertEqual(prepared.restore.call_count, 5)
        self.assertTrue(result["output_validation"]["all_finite"])

    def test_suite_writes_progress_and_resumes_completed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            args = SimpleNamespace(
                output=output,
                batch_sizes=(10,),
                gpu=7,
                warmup=2,
                repeat=7,
                transfer_repeat=3,
                idle_timeout=30.0,
                python=Path(sys.executable),
            )

            def fake_read(path: Path):
                if path.name == "progress.json":
                    return json.loads(path.read_text())
                name = path.stem
                batch_mode = "population" if name.startswith("population") else "vmap"
                spike_mode = "tracked" if "_tracked_" in name else "off"
                runtime = {
                    ("population", "tracked"): 4.0,
                    ("population", "off"): 3.0,
                    ("vmap", "tracked"): 3.5,
                    ("vmap", "off"): 2.5,
                }[(batch_mode, spike_mode)]
                return {
                    "batch_size": 10,
                    "case": {"batch_mode": batch_mode, "spike_mode": spike_mode},
                    "timing": {
                        "median_seconds": runtime,
                        "q1_seconds": runtime * 0.99,
                        "q3_seconds": runtime * 1.01,
                        "relative_iqr": 0.02,
                        "cell_steps_per_second": 1.0,
                    },
                    "device_memory": {"peak_mib_in_use": runtime * 10},
                    "output_validation": {"first_lane_trace_mv": [[[0.0, 1.0]]]},
                }

            process_monitor = {
                "benchmark_pid": 123,
                "observed_compute_pids": [123],
                "unexpected_compute_pids": [],
                "passed": True,
            }
            with (
                mock.patch("braincell_ablation.git_commit", return_value="commit"),
                mock.patch("braincell_ablation._wait_for_gpu_release"),
                mock.patch("braincell_ablation.query_gpus", return_value={"selected": {"physical_id": 7}}),
                mock.patch(
                    "braincell_ablation._run_monitored_child",
                    return_value=(0, process_monitor),
                ) as run_child,
                mock.patch("braincell_ablation.read_json", side_effect=fake_read),
            ):
                first = braincell_ablation.run_suite(args)
                second = braincell_ablation.run_suite(args)
            self.assertTrue(first["passed"])
            self.assertTrue(second["passed"])
            self.assertEqual(len(first["runs"]), 4)
            self.assertEqual(run_child.call_count, 4)
            self.assertTrue((Path(directory) / "progress.json").exists())
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".csv").exists())

    def test_case_cli_dispatches_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case.json"
            argv = [
                "braincell_ablation.py",
                "case",
                "--batch-size",
                "10",
                "--batch-mode",
                "population",
                "--spike-mode",
                "off",
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("braincell_ablation.run_case", return_value={"ok": True}) as run_case,
                mock.patch("braincell_ablation.write_json") as write_json,
            ):
                braincell_ablation.main()
            run_case.assert_called_once_with(
                10,
                batch_mode="population",
                spike_mode="off",
                warmup=2,
                repeat=7,
                transfer_repeat=3,
            )
            write_json.assert_called_once_with(output, {"ok": True})

    def test_suite_cli_dispatches_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            argv = [
                "braincell_ablation.py",
                "run",
                "--batch-sizes",
                "10,100,1000",
                "--python",
                sys.executable,
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("braincell_ablation.run_suite") as run_suite,
            ):
                braincell_ablation.main()
            args = run_suite.call_args.args[0]
            self.assertEqual(args.batch_sizes, (10, 100, 1000))
            self.assertEqual(args.gpu, 7)
            self.assertEqual(args.output, output)

    def test_process_monitor_parser_ignores_graphics_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pmon.log"
            path.write_text(
                "# gpu pid type sm mem command\n"
                "7 101 G - - Xorg\n"
                "7 202 C 99 20 python\n"
                "7 303 C 10 5 python\n"
                "7 404 C - - python\n"
            )
            self.assertEqual(braincell_ablation._compute_pids_from_monitor(path), (202, 303))

    @mock.patch("braincell_ablation.subprocess.check_output", return_value="445, 7\n")
    def test_gpu_load_parses_nvidia_smi_csv(self, check_output) -> None:
        self.assertEqual(braincell_ablation._gpu_load(7), (445.0, 7.0))
        self.assertIn("--id", check_output.call_args.args[0])

    @mock.patch("braincell_ablation.jax.devices")
    def test_memory_metadata_uses_allocator_stats(self, devices) -> None:
        device = mock.Mock()
        device.memory_stats.return_value = {
            "peak_bytes_in_use": 2 * 1024**2,
            "bytes_limit": 60 * 1024**3,
        }
        devices.return_value = [device]
        metadata = braincell_ablation._memory_metadata()
        self.assertEqual(metadata["peak_mib_in_use"], 2.0)
        self.assertEqual(metadata["bytes_limit"], 60 * 1024**3)

    @mock.patch("braincell_ablation.time.sleep")
    @mock.patch("braincell_ablation._gpu_load")
    def test_gpu_release_waits_for_allocator_cleanup(self, gpu_load, _sleep) -> None:
        gpu_load.side_effect = [
            (61_000.0, 0.0),
            (445.0, 0.0),
            (445.0, 0.0),
            (445.0, 0.0),
        ]
        braincell_ablation._wait_for_gpu_release(7)
        self.assertEqual(gpu_load.call_count, 4)

    def test_gpu_release_validates_samples_and_times_out(self) -> None:
        with self.assertRaisesRegex(ValueError, "stable_samples"):
            braincell_ablation._wait_for_gpu_release(7, stable_samples=0)
        with (
            mock.patch("braincell_ablation._gpu_load", return_value=(61_000.0, 99.0)),
            self.assertRaisesRegex(RuntimeError, "did not release"),
        ):
            braincell_ablation._wait_for_gpu_release(7, timeout_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
