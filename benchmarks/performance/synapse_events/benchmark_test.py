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

"""Verify topology, event counting, repeated trials and coordinator isolation."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import braincell as bc
import brainstate
import brainunit as u
import jax
import numpy as np

from benchmarks.performance.synapse_events import benchmark as b


class ConfigurationTest(unittest.TestCase):
    def test_equivalent_numeric_protocols_have_same_identifier(self):
        self.assertEqual(b.Case(rate_hz=100, duration_ms=100).key, b.Case(rate_hz=100.0, duration_ms=100.0).key)

    def test_invalid_dimensions_and_protocols(self):
        for kwargs in (
            {"n": 0},
            {"m": True},
            {"seed": -1},
            {"number": -1},
            {"number": 1.5},
            {"dt_ms": 0},
            {"duration_ms": float("nan")},
            {"rate_hz": float("inf")},
            {"duration_ms": 0.01},
            {"duration_ms": 0.031},
            {"layout": "bad"},
            {"pattern": "bad"},
            {"delay": "bad"},
            {"control": "bad"},
            {"declaration": "bad"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                b.Case(**kwargs)

    def test_suites_include_required_axes_and_deduplicate(self):
        cases = b.cases_for("all")
        self.assertEqual(
            set(c for v in cases.values() for c in v),
            {"smoke", "scaling", "activity", "schedule", "delay", "declarations", "controls"},
        )
        self.assertEqual(len({c.key for c in cases}), len(cases))
        self.assertEqual({c.number for c, groups in cases.items() if "schedule" in groups}, {10, 100, 1000, 10000})
        self.assertEqual(len(b.cases_for("scaling", (2,), (3,))), 2)
        self.assertTrue(all(len(groups) > 1 for groups in b.cases_for("all").values() if "smoke" in groups))

    def test_statistics_and_csv(self):
        self.assertEqual(b.summarize_times([1, 3, 2])["median_s"], 2)
        self.assertEqual(b.summarize_times([1])["stdev_s"], 0)
        self.assertEqual(b._positive_csv("1,10"), (1, 10))
        for value in ("", "0", "1,no", "-1"):
            with self.assertRaises(Exception):
                b._positive_csv(value)


class WorkloadTest(unittest.TestCase):
    def test_precision_diagnostic_preserves_input_schedule(self):
        case = b.Case(n=1, m=2, duration_ms=0.5, rate_hz=1000, pattern="poisson")
        schedules = []
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision):
                work = b.build_workload(case)
                schedules.append(np.asarray(work["source"].event_times.to_decimal(u.ms)))
                self.assertEqual(work["cell"].V.value.dtype, np.dtype(f"float{precision}"))
        np.testing.assert_array_equal(*schedules)

    def test_topology_layouts_and_declaration_equivalence(self):
        case = b.Case(n=2, m=3, duration_ms=1, rate_hz=10000, pattern="sync")
        finals = []
        for layout in ("independent", "shared"):
            for declaration in ("batched", "per_cell"):
                cfg = replace(case, layout=layout, declaration=declaration)
                work = b.build_workload(cfg)
                cell = work["cell"]
                self.assertEqual(work["source"].size, 6)
                self.assertEqual(len(cell.connections), 6)
                np.testing.assert_array_equal(np.bincount(cell.connections.synapse.population_index), [3, 3])
                np.testing.assert_array_equal(np.bincount(cell.connections.source_index, minlength=6), np.ones(6))
                self.assertEqual(work["node"].g.value.size, 6 if layout == "independent" else 2)
                self.assertEqual(len(cell.connections.connect_names), 1 if declaration == "batched" else 2)
                reset, funcs, times = b.prepare_functions(cfg, work)
                jax.block_until_ready(reset())
                first = jax.block_until_ready(funcs["full"](times))
                jax.block_until_ready(reset())
                second = jax.block_until_ready(funcs["full"](times))
                for x, y in zip(first, second):
                    np.testing.assert_array_equal(x, y)
                finals.append(first)
                work["net"].reset_state()
                work["net"].run(dt=cfg.dt_ms * u.ms, duration=cfg.duration_ms * u.ms, event_backend="scatter")
                np.testing.assert_allclose(first[0], cell.V.value.to_decimal(u.mV), rtol=1e-6)
        for final in finals[1:]:
            for x, y in zip(final, finals[0]):
                np.testing.assert_allclose(x, y, rtol=2e-5, atol=1e-7)

    def test_future_events_preserve_arrivals_and_final_state(self):
        cfg = b.Case(n=2, m=2, duration_ms=1, rate_hz=10000)
        counts, finals = [], []
        for number in (10, 100):
            case = replace(cfg, number=number)
            work = b.build_workload(case)
            counts.append(b.arrival_counts(case, work))
            reset, funcs, times = b.prepare_functions(case, work)
            jax.block_until_ready(reset())
            finals.append(jax.block_until_ready(funcs["full"](times)))
        np.testing.assert_array_equal(counts[0], counts[1])
        for x, y in zip(*finals):
            np.testing.assert_array_equal(x, y)

    def test_half_step_ties_delays_and_multiple_events(self):
        case = b.Case(n=1, m=2, duration_ms=1, dt_ms=0.1)
        # 0.05 -> step 1; 0.95 -> step 10 (outside); duplicates preserve multiplicity.
        source = bc.EventSequence.from_times(([0.049, 0.05, 0.05, 0.95] * u.ms, [0.0, 0.9] * u.ms))
        # arrival_counts specifically exercises NetStim's padded representation.
        stim = bc.NetStim(size=2, start=[0.05, 0.0] * u.ms, interval=0.02 * u.ms, number=3)
        counts = b.arrival_counts(case, dict(source=stim, delay_ms=np.array([0.0, 1.0])))
        np.testing.assert_array_equal(counts, [3, 0])
        actual = source.event_count(np.arange(2), t=0.1 * u.ms, delay=0.0 * u.ms, dt=0.1 * u.ms)
        np.testing.assert_array_equal(actual, [2, 0])
        from braincell.network.event import round_half_up_steps_host

        np.testing.assert_array_equal(round_half_up_steps_host(np.array([0.49, 0.5, 9.5])), [0, 1, 10])

    def test_measure_controls_delays_and_noisy_sources(self):
        base = b.Case(n=1, m=2, duration_ms=0.5, rate_hz=10000)
        variants = [replace(base, control=c) for c in ("cell_only", "unconnected", "silent")]
        variants += [replace(base, delay=d, pattern="sync") for d in ("fixed", "heterogeneous")]
        variants += [replace(base, pattern="poisson"), replace(base, number=0)]
        for case in variants:
            with self.subTest(case=case):
                row = b.measure_case(case, warmup=0, repeat=1)
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["environment"]["x64"], bool(jax.config.jax_enable_x64))
                if case.control != "active" or case.number == 0 or case.delay == "fixed":
                    self.assertEqual(row["arrival_count"], 0)
                    self.assertIsNone(row["timing"]["full"]["events_per_second"])
                    np.testing.assert_array_equal(row["final_conductance_us"], [0])
        for warmup, repeat in ((-1, 1), (0, 0)):
            with self.assertRaises(ValueError):
                b.measure_case(base, warmup=warmup, repeat=repeat)

    def test_microbench_checksums_warmup_and_trace(self):
        case = b.Case(n=1, m=3, duration_ms=0.5, rate_hz=10000, pattern="sync")
        with patch("jax.profiler.trace") as trace:
            row = b.measure_case(case, warmup=1, repeat=2, trace_dir="unused")
        trace.assert_called_once_with("unused")
        self.assertEqual(len(row["timing"]["full"]["samples_s"]), 2)
        self.assertEqual(row["timing"]["query"]["checksum"], row["arrival_count"])


class CoordinatorTest(unittest.TestCase):
    def test_device_isolation_failures_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            worker_envs = []

            def run(command, **kwargs):
                if "--case-json" not in command:
                    return subprocess.CompletedProcess(command, 0, "metadata", "")
                env = kwargs["env"]
                worker_envs.append(env)
                path = Path(command[command.index("--out") + 1])
                if len(worker_envs) == 1:
                    b._write_json(path, {"status": "ok"})
                    return subprocess.CompletedProcess(command, 0, "", "")
                if len(worker_envs) == 2:
                    return subprocess.CompletedProcess(command, 1, "", "failure")
                raise subprocess.TimeoutExpired(command, 1)

            args = [
                "--gpu", "7",
                "--out",
                tmp,
                "--suite",
                "smoke",
                "--number",
                "3",
                "--trace",
                "--rate-hz",
                "10",
                "--pattern",
                "sync",
                "--precision",
                "64",
            ]
            with patch.object(b.subprocess, "run", side_effect=run):
                self.assertEqual(b.main(args), 1)
            self.assertEqual(worker_envs[0]["JAX_PLATFORMS"], "cpu")
            self.assertEqual(worker_envs[1]["JAX_PLATFORMS"], "cuda,cpu")
            self.assertEqual(worker_envs[1]["JAX_ENABLE_X64"], "true")
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual([r["status"] for r in manifest["rows"]], ["ok", "failed", "timeout", "timeout"])
            for row in manifest["rows"]:
                b._write_json(out / row["file"], {"status": "ok"})
            with patch.object(b.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as mock:
                self.assertEqual(b.main(args + ["--resume"]), 0)
                self.assertFalse(any("--case-json" in call.args[0] for call in mock.call_args_list))

    def test_worker_errors_and_argument_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = str(Path(tmp) / "worker.json")
            with patch.object(b, "measure_case", return_value={"status": "ok"}):
                self.assertEqual(b.main(["--case-json", "{}", "--out", dest]), 0)
            with (
                patch.object(b, "measure_case", return_value={"status": "ok"}),
                patch("brainstate.environ.set_precision") as precision,
            ):
                self.assertEqual(b.main(["--case-json", "{}", "--out", dest, "--precision", "64"]), 0)
                precision.assert_called_once_with(64)
            with patch.object(b, "measure_case", side_effect=RuntimeError("worker failure")):
                self.assertEqual(b.main(["--case-json", "{}", "--out", dest]), 1)
            self.assertIn("worker failure", json.loads(Path(dest).read_text())["error"])
            with self.assertRaises(SystemExit):
                b.main(["--repeat", "0"])



class PortableCoordinatorTest(unittest.TestCase):
    def test_worker_interpreter_uses_running_python(self):
        import sys
        self.assertEqual(b.DEFAULT_PYTHON, sys.executable)

    def test_gpu_coordinator_requires_an_explicit_device(self):
        with patch.object(b.subprocess, "run") as worker:
            with self.assertRaises(SystemExit):
                b.main(["--devices", "gpu"])
            worker.assert_not_called()

if __name__ == "__main__":
    unittest.main()
