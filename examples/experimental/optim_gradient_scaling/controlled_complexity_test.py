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

"""Tests for the controlled BPTT/RTRL complexity benchmark."""

import json
from pathlib import Path
import tempfile
import unittest

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.experimental.optim_gradient_scaling.controlled_complexity import (
    HH_GROUP_PARAMETER_COUNTS,
    ControlledCase,
    aggregate_results,
    balanced_cases,
    prepare_synthetic,
    run_suite,
    suite_cases,
)
from examples.experimental.optim_gradient_scaling.benchmark import (
    BenchmarkConfig,
    build_cell,
)


class ControlledComplexityBenchmarkTest(unittest.TestCase):
    def test_suite_has_22_unique_controlled_cases(self) -> None:
        cases = suite_cases("all")
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({case.id for case in cases}), 22)
        self.assertEqual(len(suite_cases("synthetic")), 12)
        self.assertEqual(len(suite_cases("hh_state")), 6)
        self.assertEqual(len(suite_cases("hh_parameter")), 4)
        self.assertTrue(all(case.n_theta == 3 for case in suite_cases("hh_state")))
        self.assertTrue(all(case.n_x == 132 for case in suite_cases("hh_parameter")))
        self.assertEqual(
            {case.parameter_group: case.n_theta for case in suite_cases("hh_parameter")},
            HH_GROUP_PARAMETER_COUNTS,
        )

    def test_case_validation_rejects_mixed_or_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            ControlledCase("synthetic", n_x=4, n_theta=5)
        with self.assertRaises(ValueError):
            ControlledCase("hh_state", n_x=12, n_theta=4, n_cv=3, parameter_group="all")
        with self.assertRaises(ValueError):
            ControlledCase("hh_parameter", n_x=12, n_theta=1, n_cv=3, parameter_group="branch")

    def test_balanced_schedule_changes_temporal_position_by_replicate(self) -> None:
        cases = suite_cases("all")
        orders = [balanced_cases(cases, replicate) for replicate in (1, 2, 3)]
        self.assertEqual(orders[0], cases)
        self.assertEqual(orders[1], tuple(reversed(cases)))
        self.assertTrue(all(set(order) == set(cases) for order in orders))
        positions = [{case.id: index for index, case in enumerate(order)} for order in orders]
        self.assertTrue(all(len({position[case.id] for position in positions}) == 3 for case in cases))

    def test_synthetic_bptt_and_rtrl_match_and_carry_is_exact(self) -> None:
        case = ControlledCase("synthetic", n_x=8, n_theta=3, num_steps=5, batch_size=2, n_seed=2)
        outputs = {}
        prepared_by_method = {}
        previous_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            for method in ("bptt", "rtrl"):
                prepared = prepare_synthetic(case, method)
                prepared_by_method[method] = prepared
                outputs[method] = jax.jit(prepared.function)(prepared.roots)
                self.assertEqual(prepared.parameter_count_per_seed, 3)
                self.assertEqual(prepared.state_scalar_count_per_seed, 16)
        finally:
            jax.config.update("jax_enable_x64", previous_x64)
        expected_carry = 8 * case.n_seed * case.batch_size * case.n_x * case.n_theta
        self.assertIsNone(prepared_by_method["bptt"].rtrl_carry_bytes)
        self.assertEqual(prepared_by_method["rtrl"].rtrl_carry_bytes, expected_carry)
        for bptt, rtrl in zip(outputs["bptt"], outputs["rtrl"]):
            np.testing.assert_allclose(rtrl, bptt, rtol=1e-10, atol=1e-11)
        self.assertEqual(np.asarray(outputs["bptt"][2]).shape, (2, 3))

    def test_hh_grouping_changes_only_parameter_root_count(self) -> None:
        config = BenchmarkConfig(n_cv=3, duration_ms=0.1, batch_size=2, n_seed=2)
        expected = {"all": 1, "population": 2, "cv": 3, "row": 6}
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            for group, count in expected.items():
                with self.subTest(group=group):
                    cell = build_cell(
                        config,
                        trainable=True,
                        trainable_channels=("leak",),
                        trainable_group_by=group,
                    )
                    roots = cell.trainables.parameters().states()
                    self.assertEqual(tuple(roots), ("leak.scale",))
                    self.assertEqual(sum(int(state.value.size) for state in roots.values()), count)
                    self.assertEqual(set(cell.channels.names), {"leak", "na", "k"})

    def test_dry_run_manifest_expands_three_replicates_to_132_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_suite(
                "all",
                output_dir=output,
                gpu=7,
                repeats=10,
                replicates=3,
                resume=False,
                dry_run=True,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["configs"]), 22)
            self.assertEqual(manifest["replicates"], 3)
            self.assertEqual(manifest["schedule"], "replicate_major_balanced")
            self.assertEqual((output / "results.csv").read_text(), "")

    def test_aggregate_pairs_methods_within_each_replicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trials = output / "trials"
            trials.mkdir(parents=True)
            case = ControlledCase("synthetic", n_x=8, n_theta=2, num_steps=3, batch_size=1, n_seed=1)
            for replicate in (1, 2):
                for method, delta in (("bptt", 0.0), ("rtrl", 1e-12 * replicate)):
                    stem = f"{case.id}__rep{replicate}__{method}"
                    np.savez(
                        trials / f"{stem}.npz",
                        gradient=np.asarray([[1.0, 2.0 + delta]]),
                        loss=np.asarray([3.0]),
                        losses=np.asarray([[1.0, 1.0, 1.0]]),
                    )
                    (trials / f"{stem}.json").write_text(
                        json.dumps(
                            {
                                **case.__dict__,
                                "config_id": case.id,
                                "pair_id": f"{case.id}__rep{replicate}",
                                "replicate": replicate,
                                "method": method,
                                "status": "ok",
                                "steady_median_seconds": 2.0 if method == "bptt" else 1.0,
                                "gradient_file": f"{stem}.npz",
                            }
                        )
                    )
            rows = aggregate_results(output)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["gradient_max_abs_error"] > 0.0 for row in rows))
            self.assertTrue(all(row["gradient_relative_l2_error"] > 0.0 for row in rows))
            self.assertTrue(all(row["bptt_over_rtrl_time"] == 2.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
