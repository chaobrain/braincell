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

"""Check measurement accounting without running or timing any model."""

import argparse
import contextlib
import io
from pathlib import Path
import unittest

from benchmarks.performance.geometry_initialization import benchmark


class InitializationProtocolTest(unittest.TestCase):
    def test_workload_uses_valid_branch_types_and_exact_cv_policy(self):
        cell = benchmark.build_cell(128)
        self.assertEqual(tuple(branch.type for branch in cell.morpho.branches),
                         ("soma", "dendrite", "dendrite"))
        self.assertEqual(sum(cell.cv_policy.cv_per_branch), 128)

    def test_counts_have_no_hidden_compilation_or_validation_runs(self):
        args = argparse.Namespace(sizes=[12, 128, 512, 1024], warmup=1, repeat=5, platform="gpu")
        calls, snapshots = [], []

        def measure(size):
            calls.append(size)
            return dict.fromkeys(benchmark.PHASES, float(len(calls)))

        samples = benchmark.collect_trials(args, measure, lambda rows: snapshots.append(list(rows)))
        plan = benchmark.execution_plan(args)
        self.assertEqual(calls, [12] * 6 + [128] * 6 + [512] * 6 + [1024] * 6)
        self.assertEqual(len(samples), plan["fresh_models"])
        self.assertEqual(plan["total_timesteps"], 24)
        self.assertEqual(plan["first_jit_executions"], 24)
        self.assertEqual(plan["platform"], "gpu")
        self.assertEqual(sum(not row["warmup"] for row in samples), 20)
        self.assertEqual([len(rows) for rows in snapshots], list(range(1, 25)))
        summary = benchmark.summarize(samples)
        self.assertEqual(summary["12"]["init_state_s"], {"median": 4.0, "min": 2.0, "max": 6.0})
        self.assertEqual(summary["128"]["first_jit_step_s"]["median"], 10.0)

    def test_failure_keeps_completed_samples_without_retry(self):
        args = argparse.Namespace(sizes=[12], warmup=1, repeat=5)
        snapshots = []
        calls = 0

        def measure(size):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("deliberate failure")
            return dict.fromkeys(benchmark.PHASES, 0.01)

        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            benchmark.collect_trials(args, measure, lambda rows: snapshots.append(list(rows)))
        self.assertEqual(calls, 3)
        self.assertEqual(len(snapshots[-1]), 2)

    def test_exact_cv_counts(self):
        for total in (3, 12, 128, 512, 1024):
            counts = benchmark.cv_counts(total)
            self.assertEqual(sum(counts), total)
            self.assertGreaterEqual(min(counts), 1)
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_cli_rejects_missing_counts_or_invalid_sizes_before_import(self):
        root = Path(__file__).resolve().parents[3]
        base = ["--source-root", str(root), "--label", "test", "--output", "unused.json", "--platform", "cpu"]
        cases = [[], ["--sizes", "12", "--repeat", "5"],
                 ["--sizes", "2", "--repeat", "5", "--warmup", "1"],
                 ["--sizes", "12", "12", "--repeat", "5", "--warmup", "1"],
                 ["--sizes", "12", "--repeat", "0", "--warmup", "1"]]
        for case in cases:
            with self.subTest(case=case), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    benchmark.parse_args(base + case)
        args = benchmark.parse_args(base + ["--sizes", "12", "--repeat", "5", "--warmup", "1"])
        self.assertEqual(args.source_root, root)


if __name__ == "__main__":
    unittest.main()
