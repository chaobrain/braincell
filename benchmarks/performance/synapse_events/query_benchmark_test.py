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

"""Verify experiment controls, actual rollouts and isolated worker handling."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import brainstate
import brainunit as u
import jax
import numpy as np

from benchmarks.performance.synapse_events import query_benchmark as b
from braincell.experimental.scheduled_events import prepare_events


class QueryBenchmarkTest(unittest.TestCase):
    def test_cases_and_validation(self):
        cases = b.cases_for("all")
        self.assertEqual(
            set(g for gs in cases.values() for g in gs),
            {"smoke", "scaling", "schedule", "activity", "fanout", "blocks"},
        )
        self.assertEqual({c.k for c in b.cases_for("schedule")}, {10, 100, 1000, 10000})
        self.assertEqual(len(b.cases_for("scaling")), 18)
        self.assertEqual(b.QueryCase(duration_ms=100).key, b.QueryCase(duration_ms=100.0).key)
        for kwargs in (
            {"n": 0},
            {"m": True},
            {"k": 1.5},
            {"layout": "bad"},
            {"pattern": "bad"},
            {"delay": "bad"},
            {"duration_ms": np.nan},
            {"dt_ms": -1},
            {"duration_ms": 0.01},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                b.QueryCase(**kwargs)
        with self.assertRaises(ValueError):
            b.cases_for("bad")

    def test_schedule_invariance_and_fanout(self):
        cases = [b.QueryCase(n=2, m=3, k=k) for k in (10, 100)]
        sources = [b.build_source(c)[0] for c in cases]
        np.testing.assert_array_equal(sources[0]._event_times_ms, sources[1]._event_times_ms[:, :10])
        for pattern in ("phase", "sparse", "random", "sync", "burst", "dense", "silent"):
            cfg = replace(
                cases[0],
                pattern=pattern,
                k=320 if pattern in ("burst", "dense") else 10,
                fanout=3,
                delay="heterogeneous",
                layout="shared",
            )
            source, indices, delay, targets, n = b.build_source(cfg)
            self.assertEqual(indices.size, 18)
            self.assertEqual(n, 2)
            np.testing.assert_array_equal(np.bincount(indices), np.full(6, 3))
            self.assertTrue(np.all(np.asarray(delay / u.ms) >= 0))
            self.assertTrue(np.all(np.asarray(delay / u.ms) < 5))
            self.assertTrue(np.all(np.diff(source._event_times_ms, axis=1) >= 0))
            if pattern == "silent":
                self.assertTrue(np.all(source._event_times_ms > cfg.duration_ms))
        snapshots = []
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision):
                snapshots.append(b.build_source(replace(cases[0], pattern="random"))[0]._event_times_ms)
        np.testing.assert_array_equal(*snapshots)

    def test_actual_measurement_and_changed_weights(self):
        case = b.QueryCase(n=1, m=2, duration_ms=2.0, dt_ms=0.5, pattern="sync")
        result = b.measure_case(case, repeat=1, warmup=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["arrivals"], 2)
        self.assertEqual(len(result["rows"]), 4)
        for row in result["rows"]:
            self.assertEqual(row["query_max_error"], 0)
            for metric in row["timing"].values():
                self.assertGreater(metric["median_s"], 0)
        source, indices, delay, targets, n = b.build_source(case)
        plan = prepare_events(source, indices, delay=delay, dt=0.5 * u.ms, n_steps=4, method="bucket", block_size=1)
        run = b.make_runner(plan, targets, n, mode="synapse", record=True)
        import jax.numpy as jnp

        args = (plan.arrays, plan.reset(), jnp.arange(4))
        a = np.asarray(run(*args, jnp.ones(2), jnp.array(2.0))[3])
        doubled = np.asarray(run(*args, 2 * jnp.ones(2), jnp.array(2.0))[3])
        np.testing.assert_allclose(doubled, 2 * a)
        with self.assertRaises(ValueError):
            b.make_runner(plan, targets, n, mode="bad")

    def test_dense_and_burst_have_equal_delivered_counts(self):
        counts = []
        for pattern in ("dense", "burst"):
            case = b.QueryCase(n=10, m=100, k=320, pattern=pattern)
            source = b.build_source(case)[0]
            arrival_steps = np.floor(source._event_times_ms / case.dt_ms + 0.5)
            counts.append(int(np.sum(arrival_steps < case.steps)))
        self.assertEqual(counts, [320000, 320000])

    def test_worker_success_failure_and_backend_check(self):
        case = json.dumps(vars(b.QueryCase()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.json"
            argv = ["--worker", case, "--devices", jax.devices()[0].platform, "--out", str(path)]
            with patch.object(b, "measure_case", return_value={"status": "ok"}):
                self.assertEqual(b.main(argv), 0)
            with patch.object(b, "measure_case", side_effect=RuntimeError("fixture")):
                self.assertEqual(b.main(argv), 1)
            self.assertIn("fixture", json.loads(path.read_text())["error"])
            with patch("jax.devices", return_value=[type("Device", (), {"platform": "other"})()]):
                self.assertEqual(b.main(argv), 1)
        with self.assertRaises(SystemExit):
            b.main(["--repeat", "0"])

    def test_coordinator_device_env_resume_failure_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:

            def fake(command, **kwargs):
                device = command[command.index("--devices") + 1]
                env = kwargs["env"]
                self.assertEqual(env["JAX_PLATFORM_NAME"], device)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7" if device == "gpu" else "")
                path = Path(command[command.index("--out") + 1])
                path.write_text(json.dumps({"status": "ok"}))
                return subprocess.CompletedProcess(command, 0)

            argv = ["--out", tmp, "--gpu", "7"]
            with patch.object(b.subprocess, "run", side_effect=fake) as launch:
                self.assertEqual(b.main(argv), 0)
                self.assertEqual(launch.call_count, 2)
                self.assertEqual(b.main(argv + ["--resume"]), 0)
                self.assertEqual(launch.call_count, 2)
            with self.assertRaises(ValueError):
                b.main(argv + ["--resume", "--repeat", "6"])
            with patch.object(b.subprocess, "run", return_value=subprocess.CompletedProcess([], 9)):
                self.assertEqual(b.main(argv), 1)
            with patch.object(b.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 1)):
                self.assertEqual(b.main(argv), 1)


class ExplicitDeviceTest(unittest.TestCase):
    def test_gpu_selection_is_required_before_launch(self):
        with patch.object(b.subprocess, "run") as launch:
            for args in (["--devices", "gpu"], ["--devices", "gpu", "--gpu", "-1"]):
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    b.main(args)
            launch.assert_not_called()
