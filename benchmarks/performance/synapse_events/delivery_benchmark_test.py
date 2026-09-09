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

"""Check real measurements and the bounded subprocess protocol."""

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
import brainunit as u

from benchmarks.performance.synapse_events import delivery_benchmark as b


class DeliveryBenchmarkTest(unittest.TestCase):
    def test_reset_reproducibility_accepts_float32_rounding_but_rejects_stale_state(self):
        previous = (np.array([-3.9105084], dtype=np.float32), np.array([0.002], dtype=np.float32))
        repeated = (np.array([-3.9105072], dtype=np.float32), np.array([0.002], dtype=np.float32))
        errors = b._check_reset_result(repeated, previous)
        self.assertGreater(errors["reset_voltage_max_error_mV"], 0)
        self.assertEqual(errors["reset_g_max_error"], 0)
        with self.assertRaises(AssertionError):
            b._check_reset_result((previous[0] + 0.01, previous[1]), previous)
        with self.assertRaises(AssertionError):
            b._check_reset_result((previous[0], previous[1] + 0.001), previous)

    def test_full_runner_without_synapses_advances_and_resets(self):
        from benchmarks.performance.synapse_events.benchmark import Case, build_workload

        case = Case(n=2, m=3, control="cell_only", duration_ms=1, dt_ms=0.25)
        work = build_workload(case)
        self.assertIsNone(work["node"])
        self.assertEqual(work["net"].connections.n_rows, 0)
        reset, run = b.full_runner(work, record=True)
        jax.block_until_ready(reset())
        steps = np.arange(case.steps, dtype=np.int32)
        first = jax.device_get(run(steps))
        self.assertEqual(first[1].shape, (0,))
        self.assertEqual(first[2][1].shape, (case.steps, 0))
        self.assertEqual(first[2][2].shape, (case.steps, 0))
        self.assertTrue(np.isfinite(first[0]).all())
        self.assertAlmostEqual(float(work["cell"].current_time.to_decimal(u.ms)), 1)
        jax.block_until_ready(reset())
        again = jax.device_get(run(steps))
        np.testing.assert_array_equal(first[0], again[0])

    def test_actual_four_controls_and_identical_shifted_schedules(self):
        case = b.QueryCase(n=2, m=3, k=2, duration_ms=20, dt_ms=1)
        self.assertEqual(set(b.cases_for("controls")), set(b.CONTROL_JOBS))
        source_times = {}
        for control in ("cell_only", "unconnected", "silent", "active"):
            with self.subTest(control=control):
                methods = ("direct", "padded") if control in ("silent", "active") else ()
                data = b.measure_full(case, methods=methods, repeat=1, warmup=0, control=control, validate_control=True)
                reference = data["rows"][0]["validation"]
                for row in data["rows"]:
                    self.assertEqual(row["status"], "ok")
                    self.assertEqual(row["validation"], reference)
                    self.assertTrue(row["validation"]["reset_reproducible"])
                self.assertEqual(reference["synapse_count"], 0 if control == "cell_only" else 6)
                self.assertEqual(reference["connection_rows"], 6 if methods else 0)
                self.assertEqual(reference["arrivals"] > 0, control == "active")
                if methods:
                    work = b.make_full_workload(case, "production", control=control)
                    source_times[control] = np.asarray(work["source"].event_times.to_decimal(u.ms))
        np.testing.assert_allclose(source_times["silent"] - source_times["active"], case.duration_ms + 10)

    def test_controls_validate_long_float32_clock_by_step_boundary(self):
        case = b.QueryCase(n=1, m=1)
        data = b.measure_full(case, methods=(), repeat=1, warmup=0, control="cell_only", validate_control=True)
        check = data["rows"][0]["validation"]
        self.assertEqual(check["final_step"], case.steps)
        self.assertLess(abs(check["final_time_ms"] - case.duration_ms), case.dt_ms / 2)

    def test_controls_coordinator_splits_baselines_and_workers_dispatch(self):
        seen = []

        def fake(cmd, **kwargs):
            job = cmd[cmd.index("--control-job") + 1]
            path = Path(cmd[cmd.index("--out") + 1])
            seen.append(job)
            path.write_text(json.dumps({"status": "ok"}))
            return subprocess.CompletedProcess(cmd, 0)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(b.subprocess, "run", side_effect=fake):
                self.assertEqual(b.main(["--phase", "controls", "--devices", "cpu", "--out", tmp]), 0)
            self.assertEqual(seen, list(b.CONTROL_JOBS))
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            for row in manifest["rows"]:
                self.assertEqual(row["methods"], ["production", *b.CONTROL_JOBS[row["case"]][1]])
                self.assertLessEqual(len(row["methods"]), 2)
            for job, (control, methods) in b.CONTROL_JOBS.items():
                with patch.object(b, "measure_full", return_value={"status": "ok"}) as measure:
                    self.assertEqual(
                        b.main(
                            [
                                "--phase",
                                "controls",
                                "--worker",
                                json.dumps(vars(b.QueryCase())),
                                "--devices",
                                jax.devices()[0].platform,
                                "--control-job",
                                job,
                                "--out",
                                str(Path(tmp) / f"{job}.json"),
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(measure.call_args.kwargs["methods"], methods)
                    self.assertEqual(measure.call_args.kwargs["control"], control)
                    self.assertTrue(measure.call_args.kwargs["validate_control"])
        for args in (["--methods", "direct"], ["--worker", "{}"]):
            with self.assertRaises(SystemExit):
                b.main(["--phase", "controls", *args])

    def test_matrix_and_actual_micro_measurements(self):
        self.assertEqual(len(b.cases_for("micro")), 6)
        self.assertEqual(set(b.cases_for("full")), {"small", "large", "long", "shared"})
        self.assertEqual(set(b.cases_for("profile")), {"sparse", "burst", "shared"})
        with self.assertRaises(ValueError):
            b.cases_for("invalid")
        case = b.QueryCase(n=1, m=2, k=2, duration_ms=2, dt_ms=0.5, pattern="sync")
        data = b.measure_micro(case, repeat=1, warmup=1, order=2)
        self.assertEqual(data["arrivals"], 2)
        self.assertEqual(len(data["rows"]), 5)
        for row in data["rows"]:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["count_max_error"], 0)
            self.assertGreater(row["timing"]["synapse"]["median_s"], 0)
        limited = b.measure_micro(replace(case, dt_ms=0.025), methods=("padded",), repeat=1, warmup=0)
        self.assertEqual(limited["rows"][0]["status"], "not_applicable")
        with self.assertRaises(ValueError):
            b.make_runner(None, [], 1, mode="invalid")

    def test_actual_full_measurement_and_padding_rejection(self):
        case = b.QueryCase(n=1, m=2, k=2, duration_ms=10, dt_ms=0.5)
        with tempfile.TemporaryDirectory() as tmp, patch("jax.profiler.trace", return_value=nullcontext()):
            data = b.measure_full(case, methods=("direct", "padded"), repeat=1, warmup=1, profile_dir=Path(tmp))
        self.assertEqual(data["rows"][0]["profile_status"], "ok")
        self.assertEqual(data["rows"][0]["method"], "production")
        self.assertTrue(all(r.get("voltage_max_error_mV", 0) < 1e-4 for r in data["rows"]))
        with tempfile.TemporaryDirectory() as tmp, patch("jax.profiler.trace", side_effect=RuntimeError("no trace")):
            limited = b.measure_full(
                replace(case, dt_ms=0.025), methods=("padded",), repeat=1, warmup=0, profile_dir=Path(tmp)
            )
        self.assertIn("unavailable", limited["rows"][0]["profile_status"])
        self.assertEqual(limited["rows"][1]["status"], "not_applicable")

    def test_profile_success_and_failure_are_separate_from_measurement(self):
        case = b.QueryCase(n=1, m=1, k=1, duration_ms=1, dt_ms=0.5, pattern="sync")
        for raises in (False, True):
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch("jax.profiler.start_trace", side_effect=RuntimeError("no profiler") if raises else None),
                patch("jax.profiler.stop_trace"),
            ):
                data = b.measure_micro(case, methods=("direct",), repeat=1, warmup=0, profile_dir=Path(tmp))
                self.assertEqual(data["rows"][0]["status"], "ok")
                self.assertIn("unavailable" if raises else "ok", data["rows"][0]["profile_status"])

    def test_worker_errors_and_cli_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "worker.json"
            argv = [
                "--worker",
                json.dumps(vars(b.QueryCase())),
                "--devices",
                jax.devices()[0].platform,
                "--out",
                str(out),
            ]
            with patch.object(b, "measure_micro", return_value={"status": "ok"}):
                self.assertEqual(b.main(argv), 0)
            with patch.object(b, "measure_full", return_value={"status": "ok"}):
                self.assertEqual(b.main(argv + ["--phase", "full"]), 0)
            with patch.object(b, "measure_micro", side_effect=RuntimeError("fixture")):
                self.assertEqual(b.main(argv), 1)
            with patch("jax.devices", return_value=[type("D", (), {"platform": "wrong"})()]):
                self.assertEqual(b.main(argv), 1)
        for extra in (
            ["--methods", "bad"],
            ["--methods", "direct,direct"],
            ["--rounds", "0"],
            ["--warmup", "-1"],
            ["--trace-full"],
            ["--cases", "bad"],
        ):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                b.main(extra)

    def test_coordinator_isolation_statuses_and_budget(self):
        def fake(cmd, **kwargs):
            path = Path(cmd[cmd.index("--out") + 1])
            device = cmd[cmd.index("--devices") + 1]
            self.assertEqual(kwargs["env"]["JAX_PLATFORM_NAME"], device)
            self.assertEqual(kwargs["timeout"], 120)
            if device == "gpu":
                raise subprocess.TimeoutExpired(cmd, 120)
            path.write_text(json.dumps({"status": "ok"}))
            return subprocess.CompletedProcess(cmd, 0)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = ["--out", str(out), "--cases", "small", "--devices", "both", "--gpu", "7"]
            with patch.object(b.subprocess, "run", side_effect=fake):
                self.assertEqual(b.main(args), 1)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual([r["status"] for r in manifest["rows"]], ["ok", "timeout"])
            self.assertTrue((out / "source_snapshot/braincell/experimental/scheduled_delivery.py").exists())
            with self.assertRaises(SystemExit):
                b.main(args)
            with patch.object(b.subprocess, "run", side_effect=fake):
                self.assertEqual(
                    b.main(
                        [
                            "--out",
                            str(Path(tmp) / "profile"),
                            "--cases",
                            "small",
                            "--devices",
                            "cpu",
                            "--phase",
                            "full",
                            "--trace-full",
                        ]
                    ),
                    0,
                )
            with patch.object(b.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)):
                self.assertEqual(
                    b.main(["--out", str(Path(tmp) / "failed"), "--cases", "small", "--devices", "cpu"]), 1
                )
            with patch.object(b.time, "monotonic", side_effect=[0, 2]):
                self.assertEqual(
                    b.main(
                        [
                            "--out",
                            str(Path(tmp) / "expired"),
                            "--cases",
                            "small",
                            "--devices",
                            "cpu",
                            "--budget-seconds",
                            "1",
                        ]
                    ),
                    1,
                )

    def test_full_default_uses_one_candidate_per_bounded_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--worker",
                json.dumps(vars(b.QueryCase())),
                "--phase",
                "full",
                "--devices",
                jax.devices()[0].platform,
                "--out",
                str(Path(tmp) / "worker.json"),
            ]
            with patch.object(b, "measure_full", return_value={"status": "ok"}) as measure:
                self.assertEqual(b.main(argv), 0)
                self.assertEqual(measure.call_args.kwargs["methods"], ("direct",))


class ExplicitDeviceTest(unittest.TestCase):
    def test_gpu_selection_is_required_before_launch(self):
        with patch.object(b.subprocess, "run") as launch:
            for args in (["--devices", "gpu"], ["--devices", "gpu", "--gpu", "-1"]):
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    b.main(args)
            launch.assert_not_called()
