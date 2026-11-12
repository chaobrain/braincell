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

import pytest
import jax
import numpy as np
import brainunit as u

from benchmarks.performance.synapse_events import delivery_benchmark as b


# Slow tier: runs nightly, not on every pull request (see the CI workflow).
pytestmark = pytest.mark.slow


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
                    if control == "silent" and row["method"] == "padded":
                        # prepare_run plans every event: the shifted ones need 8.2 slots per record here.
                        self.assertEqual(row["status"], "not_applicable")
                        self.assertGreater(row["padding_ratio"], 8)
                        continue
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


class RegressionCorrectnessTest(unittest.TestCase):
    """Each ``scheduled_delivery`` method reproduces the default path on the six regression models."""

    EXPECTED = {
        # name: (n, m, k, layout, ring), from the full matrix and the active model.
        "small": (1, 10, 10, "independent", None),
        "large": (100, 100, 10, "independent", None),
        "long": (10, 10, 10000, "independent", None),
        "shared": (100, 100, 10, "shared", None),
        "active": (20, 50, 11, None, False),
        "mixed": (20, 50, 11, None, True),
    }
    #: Models whose ``padded`` plan exceeds the padding ratio, with the ratio
    #: of the update entry's plan (every event) and of the run entry's window.
    NOT_APPLICABLE = {"small": (39.9, 40.0), "long": (12.0, 12.0)}

    def test_matrix(self):
        cases = b.cases_for("regression")
        self.assertEqual(tuple(cases), b.REGRESSION_MODELS)
        self.assertEqual(tuple(cases), tuple(self.EXPECTED))
        for name, (n, m, k, layout, ring) in self.EXPECTED.items():
            case = cases[name]
            self.assertEqual((case.n, case.m, case.k), (n, m, k), name)
            self.assertEqual(case.steps, 4000)
            if ring is None:
                self.assertEqual(case, b.cases_for("full")[name])
                self.assertEqual(case.layout, layout)
            else:
                self.assertEqual(case.ring, ring)

    def _check(self, name, precision):
        import brainstate

        with brainstate.environ.context(precision=precision):
            report = b.check_regression(name)
        self.assertEqual(report["steps"], 4000)
        for entry, position in (("update", 0), ("run", 1)):
            for method in ("direct", "padded"):
                row = report[entry][method]
                if method == "padded" and name in self.NOT_APPLICABLE:
                    self.assertEqual(row["status"], "not_applicable")
                    self.assertAlmostEqual(row["padding_ratio"], self.NOT_APPLICABLE[name][position], places=1)
                    self.assertIn('scheduled_delivery="direct"', row["reason"])
                else:
                    self.assertEqual(row["status"], "ok", (name, entry, method))
        if name in ("active", "mixed"):
            # Spiking models: every cell fires, the same spikes under each method.
            self.assertGreaterEqual(report["spikes_per_cell"][0], 2)
            self.assertGreater(report["run"]["default"]["spikes"], 0)
        return report

    def test_small_and_long_float64_and_float32(self):
        for name in ("small", "long"):
            for precision in (64, 32):
                with self.subTest(name=name, precision=precision):
                    self._check(name, precision)

    def test_large_and_shared_float64_and_float32(self):
        for name in ("large", "shared"):
            for precision in (64, 32):
                with self.subTest(name=name, precision=precision):
                    self._check(name, precision)

    def test_active_and_mixed_float64_and_float32(self):
        for name in ("active", "mixed"):
            for precision in (64, 32):
                with self.subTest(name=name, precision=precision):
                    self._check(name, precision)

    def test_recorded_input_is_the_planned_one_and_a_broken_plan_fails(self):
        from braincell.network import scheduled

        case = replace(b.cases_for("regression")["large"], n=2, m=3, duration_ms=10.0, dt_ms=0.5)
        calls = []
        deliver = scheduled._LayoutInput.deliver

        def counted(self, *args, **kwargs):
            calls.append(self.method)
            return deliver(self, *args, **kwargs)

        with patch.object(scheduled._LayoutInput, "deliver", counted):
            work = b.build_regression_workload("large", "direct", case=case)
            reset, run = b.full_runner(work, record=True)
            jax.block_until_ready(reset())
            voltage, g, drive, spike = jax.device_get(run(np.arange(case.steps, dtype=np.int32))[2])
        self.assertIn("direct", calls)
        # Independent reference: every source fires at its phase and 10 ms later,
        # rounded half up at float64; one source per synapse, 0.001 uS each.
        times = np.asarray(work["source"].event_times.to_decimal(u.ms), dtype=np.float64)
        steps = np.floor(times / case.dt_ms + 0.5).astype(int)
        expected = np.zeros((case.steps, 6))
        for row in range(6):
            for step in steps[row]:
                if step < case.steps:
                    expected[step, row] += 0.001
        np.testing.assert_allclose(drive, expected, rtol=1e-6, atol=1e-12)
        self.assertGreater(expected.sum(), 0)

        def dropped(self, output, counts, weights, *, t, dt):
            return output, counts

        with patch.object(scheduled._LayoutInput, "deliver", dropped), self.assertRaises(AssertionError):
            b.check_regression("large", methods=("direct",), case=case, chunk=10)


class RegressionMeasurementTest(unittest.TestCase):
    def test_worker_times_both_entries_and_reports_padding_limits(self):
        case = replace(b.cases_for("regression")["active"], duration_ms=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            rows = {}
            for entry in b.ENTRIES:
                for method in b.REGRESSION_METHODS:
                    path = Path(tmp) / f"{entry}_{method}.npz"
                    rows[entry, method] = b.measure_regression(
                        "active", method, entry, steps=(80, 40), warmup=1, repeat=2, check_path=path, case=case
                    )
                    row = rows[entry, method]
                    self.assertEqual(row["status"], "ok", row)
                    self.assertEqual(set(row["timing"]), {"80", "40"})
                    for item in row["timing"].values():
                        self.assertEqual(len(item["samples_s"]), 2)
                        self.assertGreater(item["first_call_s"], 0)
                        self.assertGreaterEqual(item["iqr_over_median"], 0)
                    self.assertTrue(path.exists())
            # The run entry builds a window plan per step count; the update entry one plan in prepare_run.
            self.assertIn("plan", rows["run", "direct"]["timing"]["80"])
            self.assertNotIn("plan", rows["run", "default"]["timing"]["80"])
            self.assertGreater(rows["update", "padded"]["plan"]["array_bytes"], 0)
            self.assertEqual(rows["update", "default"]["plan"]["array_bytes"], 0)
            checks = {key: np.load(Path(tmp) / f"{key[0]}_{key[1]}.npz") for key in rows}
            for entry in b.ENTRIES:
                reference = checks[entry, "default"]
                for method in ("direct", "padded"):
                    for field in ("V", "g", "spike_step", "spike_source"):
                        np.testing.assert_array_equal(checks[entry, method][field], reference[field])
            limited = b.measure_regression(
                "small",
                "padded",
                "run",
                steps=(400,),
                warmup=0,
                repeat=1,
                case=replace(b.cases_for("regression")["small"], duration_ms=10.0),
            )
            self.assertEqual(limited["status"], "not_applicable")
            self.assertEqual(limited["padding_ratio"], 40.0)
            self.assertGreater(limited["attempt_s"], 0)
            limited = b.measure_regression(
                "small",
                "padded",
                "update",
                steps=(400,),
                warmup=0,
                repeat=1,
                case=replace(b.cases_for("regression")["small"], duration_ms=10.0),
            )
            self.assertEqual(limited["status"], "not_applicable")
        with self.assertRaises(ValueError):
            b.measure_regression("small", "bucket", "run")
        with self.assertRaises(ValueError):
            b.check_regression("small", case=replace(b.cases_for("regression")["small"], duration_ms=10.0), chunk=7)

    def test_padding_ratio_and_rerun_reasons(self):
        self.assertEqual(b.padding_ratio(ValueError("needs 3244 bytes and 40 slots per record for")), 40.0)
        self.assertIsNone(b.padding_ratio(ValueError("other")))
        timing = {"4000": {"iqr_over_median": 0.05}, "2000": {"iqr_over_median": 0.2}}
        gpu = dict(utilization=40.0, compute_apps=[[5, 2048.0], [6, 258.0]])
        data = dict(status="ok", timing=timing, load_start=1.0, load_end=17.0, gpu_start=gpu, pid=9)
        reason = b._regression_worker_failed(data, max_iqr=0.1, max_load=16, gpu=1)
        self.assertIn("IQR/median 0.200", reason)
        self.assertIn("load 17.0", reason)
        self.assertIn("GPU 40% busy at start", reason)
        self.assertIn("another process holds 2048 MiB", reason)
        timing["2000"]["iqr_over_median"] = 0.01
        # An idle context of a process computing on another GPU is recorded, not re-run.
        idle = dict(data, load_end=2.0, gpu_start=dict(utilization=0.0, compute_apps=[[6, 258.0], [9, 4000.0]]))
        self.assertIsNone(b._regression_worker_failed(idle, max_iqr=0.1, max_load=16, gpu=1))
        self.assertIsNone(b._regression_worker_failed(dict(data, load_end=2.0), max_iqr=0.1, max_load=16, gpu=None))
        self.assertIsNone(
            b._regression_worker_failed(dict(status="not_applicable"), max_iqr=0.1, max_load=16, gpu=None)
        )
        self.assertEqual(b._busy_share({0: (100, 50)}, {0: (200, 75), 1: (10, 5)}), {"0": 0.75})
        self.assertIsNone(b._gpu_status(None))
        with patch.object(b.subprocess, "run", side_effect=OSError("no nvidia-smi")):
            self.assertIn("error", b._gpu_status(0))
        out = subprocess.CompletedProcess([], 0, stdout="7, 300\n")
        apps = subprocess.CompletedProcess([], 0, stdout="12, 258\n13, 40\n\n")
        with patch.object(b.subprocess, "run", side_effect=[out, apps]):
            self.assertEqual(
                b._gpu_status(1),
                dict(utilization=7.0, memory_mib=300.0, compute_pids=[12, 13], compute_apps=[[12, 258.0], [13, 40.0]]),
            )
        with patch("builtins.open", side_effect=OSError("no /proc")):
            self.assertEqual(b._cpu_ticks(), {})

    def test_coordinator_order_retries_lock_and_budget(self):
        launched = []
        results = iter(
            # first worker re-run twice for IQR, then flagged; the rest pass at once.
            [dict(iqr=0.5), dict(iqr=0.4), dict(iqr=0.3)] + [dict(iqr=0.01)] * 100
        )

        def fake(cmd, **kwargs):
            launched.append((cmd, kwargs))
            path = Path(cmd[cmd.index("--out") + 1])
            item = next(results)
            timing = {"40": dict(iqr_over_median=item["iqr"])}
            path.write_text(json.dumps(dict(status="ok", timing=timing, load_start=1.0, load_end=1.0)))
            return subprocess.CompletedProcess(cmd, 0)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            argv = [
                "--phase",
                "regression",
                "--devices",
                "cpu",
                "--cases",
                "small,active",
                "--entries",
                "run",
                "--steps",
                "40",
                "--rounds",
                "2",
                "--repeat",
                "2",
                "--warmup",
                "0",
                "--out",
                str(out),
                "--lock",
                "/tmp/lock",
                "--cpus",
                "0,2",
                "--single-thread",
                "--pause",
                "0",
                "--timeout",
                "60",
            ]
            with patch.object(b.subprocess, "run", side_effect=fake):
                self.assertEqual(b.main(argv), 0)
            manifest = json.loads((out / "manifest.json").read_text())
            rows = manifest["rows"]
            self.assertEqual(len(rows), 2 * 2 * 3)
            self.assertEqual(len(rows[0]["attempts"]), 3)
            self.assertIn("IQR/median", rows[0]["flagged"])
            self.assertTrue(all(row["flagged"] is None for row in rows[1:]))
            self.assertEqual(len(launched), 12 + 2)
            cmd, kwargs = launched[0]
            self.assertEqual(cmd[:4], ["flock", "/tmp/lock", "timeout", "60"])
            self.assertEqual(cmd[4:7], ["taskset", "-c", "0,2"])
            self.assertIsNone(kwargs["timeout"])
            self.assertIn("intra_op_parallelism_threads=1", kwargs["env"]["XLA_FLAGS"])
            # Each (round, model, entry) runs all three methods, in a seeded order.
            for round_id in range(2):
                for model in ("small", "active"):
                    group = [r for r in rows if r["round"] == round_id and r["case"] == model]
                    self.assertEqual(sorted(r["method"] for r in group), sorted(b.REGRESSION_METHODS))
                    self.assertEqual([r["method"] for r in group], group[0]["order"])
            self.assertEqual(manifest["protocol"]["steps"], [40])
            with self.assertRaises(SystemExit):
                b.main(argv)
            # Without a lock the coordinator bounds the worker itself; a missing file is a failure.
            with patch.object(b.subprocess, "run", return_value=subprocess.CompletedProcess([], 124)) as run:
                argv2 = [a for a in argv if a not in ("--lock", "/tmp/lock")]
                argv2[argv2.index("--out") + 1] = str(Path(tmp) / "timeout")
                self.assertEqual(b.main(argv2 + ["--retries", "0"]), 1)
                self.assertEqual(run.call_args.kwargs["timeout"], 60)
            statuses = {json.loads(p.read_text())["status"] for p in (Path(tmp) / "timeout").glob("cpu_*.json")}
            self.assertEqual(statuses, {"timeout"})
            with patch.object(b.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 60)):
                argv3 = list(argv2)
                argv3[argv3.index("--out") + 1] = str(Path(tmp) / "expired")
                self.assertEqual(b.main(argv3 + ["--retries", "0", "--cases", "small"]), 1)
            with patch.object(b.time, "monotonic", side_effect=[0] + [10] * 50):
                argv4 = list(argv2)
                argv4[argv4.index("--out") + 1] = str(Path(tmp) / "budget")
                self.assertEqual(b.main(argv4 + ["--budget-seconds", "1", "--cases", "small"]), 1)
            statuses = {json.loads(p.read_text())["status"] for p in (Path(tmp) / "budget").glob("cpu_*.json")}
            self.assertEqual(statuses, {"not_run"})
            gpu_cmds = []

            def fake_gpu(cmd, **kwargs):
                gpu_cmds.append(cmd)
                Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(dict(status="not_applicable")))
                return subprocess.CompletedProcess(cmd, 0)

            with patch.object(b.subprocess, "run", side_effect=fake_gpu):
                argv5 = list(argv2)
                argv5[argv5.index("--out") + 1] = str(Path(tmp) / "gpu")
                argv5[argv5.index("--devices") + 1] = "gpu"
                self.assertEqual(b.main(argv5 + ["--gpu", "1", "--cases", "small", "--methods", "padded"]), 0)
            self.assertEqual(gpu_cmds[0][gpu_cmds[0].index("--gpu") + 1], "1")

    def test_regression_cli_validation_and_worker(self):
        for extra in (
            ["--steps", "a"],
            ["--steps", "40,40"],
            ["--steps", "0"],
            ["--entries", "bad"],
            ["--retries", "-1"],
            ["--methods", "bucket"],
            ["--cases", "bad"],
            ["--worker", "small"],
            ["--worker", "bad", "--method", "default", "--entry", "run"],
        ):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                b.main(["--phase", "regression", "--devices", "cpu", *extra])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "w.json"
            argv = [
                "--phase",
                "regression",
                "--worker",
                "small",
                "--method",
                "direct",
                "--entry",
                "run",
                "--devices",
                jax.devices()[0].platform,
                "--out",
                str(out),
                "--steps",
                "40",
            ]
            with patch.object(b, "measure_regression", return_value=dict(status="ok", timing={})) as measure:
                self.assertEqual(b.main(argv), 0)
            data = json.loads(out.read_text())
            self.assertEqual(measure.call_args.args, ("small", "direct", "run"))
            self.assertEqual(measure.call_args.kwargs["steps"], (40,))
            for key in ("load_start", "load_end", "cpu_busy_share", "peak_rss_mb", "braincell", "environment"):
                self.assertIn(key, data)
            with patch.object(b, "measure_regression", side_effect=RuntimeError("fixture")):
                self.assertEqual(b.main(argv), 1)
            self.assertEqual(json.loads(out.read_text())["status"], "failed")
            with patch("jax.devices", return_value=[type("D", (), {"platform": "wrong"})()]):
                self.assertEqual(b.main(argv), 1)
