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

"""Check protocol accounting and a tiny untimed forward/reset workflow."""

import argparse
import contextlib
import io
from pathlib import Path
import unittest

from benchmarks.performance.geometry_initialization import population


class PopulationProtocolTest(unittest.TestCase):
    def test_reduced_matrix_counts_and_order(self):
        args = argparse.Namespace(platform="cpu", sizes=[128],
                                  pop_sizes=[1, 10, 100, 1000], steps=100, warmup=1, repeat=5)
        plan = population.execution_plan(args)
        self.assertEqual(plan["initializations"], 4)
        self.assertEqual(plan["first_jit_executions"], 4)
        self.assertEqual(plan["rollouts"] * 4, 112)
        self.assertEqual(plan["resets"] * 4, 112)
        self.assertEqual(plan["total_steps"] * 4, 11200)
        self.assertEqual(population.trial_kinds(1, 5),
                         ("compile", "warmup", "timed", "timed", "timed", "timed", "timed"))
        report = {"cases": []}
        seen = []

        def measure(args, case, save):
            seen.append((case["n_cv"], case["pop_size"]))
            case["status"] = "complete"

        population.collect_cases(args, report, measure, lambda: None)
        self.assertEqual(seen, [(n_cv, pop) for n_cv in args.sizes for pop in args.pop_sizes])
        self.assertEqual(len(report["cases"]), 4)

    def test_failed_case_remains_recorded_without_retry(self):
        args = argparse.Namespace(sizes=[12], pop_sizes=[1, 10, 100])
        report = {"cases": []}

        def fail(args, case, save):
            if case["pop_size"] == 10:
                raise RuntimeError("deliberate failure")
            case["status"] = "complete"

        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            population.collect_cases(args, report, fail, lambda: None)
        self.assertEqual([case["status"] for case in report["cases"]], ["complete", "building"])

    def test_cli_requires_explicit_counts_and_valid_population_sizes(self):
        root = Path(__file__).resolve().parents[3]
        base = ["--source-root", str(root), "--platform", "cpu", "--label", "test",
                "--output", "unused.json", "--sizes", "12", "--steps", "100", "--warmup", "1", "--repeat", "5"]
        for extra in ([], ["--pop-sizes", "0"], ["--pop-sizes", "10", "10"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    population.parse_args(base + extra)
        args = population.parse_args(base + ["--pop-sizes", "1", "10", "100", "1000"])
        self.assertEqual(args.pop_sizes, [1, 10, 100, 1000])


class PopulationWorkflowTest(unittest.TestCase):
    def test_two_population_reset_and_compiled_loop_without_timing(self):
        import brainstate
        import brainunit as u
        import numpy as np

        with brainstate.environ.context(precision=64, dt=0.025 * u.ms, t=0.0 * u.ms):
            cell = population.build_population(n_cv=3, pop_size=2)
            cell.init_state()
            self.assertEqual(cell.V.value.shape, (2, 3))
            # Historical array preparation shared one CV vector; current
            # runtime geometry may materialize an explicit population axis.
            self.assertIn(cell.runtime.cable.area.shape, ((3,), (2, 3)))
            area = np.broadcast_to(cell.runtime.cable.area.to_decimal(u.cm**2), (2, 3))
            np.testing.assert_array_equal(area[0], area[1])
            run = population.make_rollout(cell, steps=3)
            first = population.check_voltage(run(), n_cv=3, pop_size=2, steps=3)
            cell.reset_state()
            second = population.check_voltage(run(), n_cv=3, pop_size=2, steps=3)
            np.testing.assert_allclose(second, first, rtol=0, atol=1e-12)
            self.assertGreater(population.logical_state_bytes(cell), 0)


if __name__ == "__main__":
    unittest.main()
