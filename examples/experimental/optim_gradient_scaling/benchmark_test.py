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

"""Tests for the block-exact BPTT/RTRL scaling benchmark."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.experimental.optim_gradient_scaling.benchmark import (
    MECHANISM_FACTORIAL_SPECS,
    BenchmarkConfig,
    MechanismSpec,
    aggregate_results,
    build_cell,
    prepare_benchmark,
    run_suite,
    suite_cases,
    suite_configs,
    _phase_metric_fields,
    _summarize_gpu_samples,
)


class ScalingBenchmarkTest(unittest.TestCase):
    def test_gpu_sample_summary_and_empty_fallback(self) -> None:
        empty = _summarize_gpu_samples(())
        self.assertEqual(empty["sample_count"], 0)
        self.assertIsNone(empty["process_peak_bytes"])
        self.assertIsNone(empty["gpu_util_median_percent"])

        summary = _summarize_gpu_samples(
            (
                {
                    "process_bytes": 100,
                    "gpu_util_percent": 20.0,
                    "memory_util_percent": 10.0,
                    "power_watts": 100.0,
                    "sm_clock_mhz": 1200.0,
                },
                {
                    "process_bytes": 300,
                    "gpu_util_percent": 80.0,
                    "memory_util_percent": 50.0,
                    "power_watts": 200.0,
                    "sm_clock_mhz": 1400.0,
                },
            )
        )
        self.assertEqual(summary["process_peak_bytes"], 300)
        self.assertEqual(summary["gpu_util_median_percent"], 50.0)
        self.assertEqual(summary["power_watts_max"], 200.0)
        fields = _phase_metric_fields("steady", summary)
        self.assertEqual(fields["gpu_samples_steady"], 2)
        self.assertEqual(fields["gpu_peak_steady_bytes"], 300)
        self.assertEqual(fields["gpu_util_steady_median_percent"], 50.0)

    def test_suite_sizes_and_axis_values(self) -> None:
        pilot = suite_configs("pilot")
        full = suite_configs("full")
        large_cv = suite_configs("large_cv")
        backsub_ab = suite_configs("backsub_ab")
        self.assertEqual(len(pilot), 9)
        self.assertEqual(len(full), 18)
        self.assertEqual([config.n_cv for config in large_cv], [13, 17, 25, 33])
        self.assertEqual([config.n_cv for config in backsub_ab], [9, 17, 25, 33])
        self.assertEqual({config.n_cv for config in full if config.duration_ms == 40.0}, {1, 3, 5, 7, 9})
        self.assertEqual({config.duration_ms for config in full if config.n_cv == 5}, {10.0, 20.0, 40.0, 80.0})
        self.assertTrue(set(pilot).issubset(full))
        factorial = suite_cases("mechanism_factorial")
        self.assertEqual(len(factorial), 30)
        self.assertEqual({case.config.n_cv for case in factorial}, {3, 5, 9, 17, 33})
        self.assertEqual({case.mechanism.name for case in factorial}, {spec.name for spec in MECHANISM_FACTORIAL_SPECS})
        self.assertEqual(len({case.id for case in factorial}), 30)

    def test_mechanism_spec_rejects_unknown_or_unpainted_trainables(self) -> None:
        with self.assertRaises(ValueError):
            MechanismSpec("bad", ("leak", "unknown"), ("leak",))
        with self.assertRaises(ValueError):
            MechanismSpec("bad", ("leak",), ("leak", "k"))

    def test_mechanism_spec_round_trips_through_json_lists(self) -> None:
        raw = json.loads(json.dumps(MECHANISM_FACTORIAL_SPECS[2].__dict__))
        restored = MechanismSpec(**raw)
        self.assertEqual(restored, MECHANISM_FACTORIAL_SPECS[2])
        self.assertIsInstance(restored.painted_channels, tuple)
        self.assertIsInstance(restored.trainable_channels, tuple)

    def test_factorial_cells_have_expected_states_and_parameter_directions(self) -> None:
        config = BenchmarkConfig(n_cv=3, duration_ms=0.1, batch_size=2, n_seed=2)
        expected = {
            "l_fit_l": (1, 1),
            "lk_fit_l": (2, 1),
            "lk_fit_lk": (2, 2),
            "lkn_fit_l": (4, 1),
            "lkn_fit_lk": (4, 2),
            "lkn_fit_lkn": (4, 3),
        }
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            for mechanism in MECHANISM_FACTORIAL_SPECS:
                with self.subTest(mechanism=mechanism.name):
                    cell = build_cell(config, trainable=True, mechanism=mechanism)
                    roots = cell.trainables.parameters().states()
                    states_per_cv, parameters_per_cv = expected[mechanism.name]
                    self.assertEqual(mechanism.state_variables_per_cv, states_per_cv)
                    self.assertEqual(sum(int(state.value.size) for state in roots.values()), parameters_per_cv * 3)
                    self.assertEqual(set(cell.channels.names), set(mechanism.painted_channels))
                    self.assertEqual({name.removesuffix(".scale") for name in roots}, set(mechanism.trainable_channels))

    def test_cv_and_batch_parameter_contract(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            for n_cv in (1, 3, 5):
                config = BenchmarkConfig(n_cv=n_cv, duration_ms=0.1, batch_size=2, n_seed=2)
                cell = build_cell(config, trainable=True)
                states = cell.trainables.parameters().states()
                self.assertEqual(cell.n_cv, n_cv)
                self.assertEqual(tuple(states), ("leak.scale", "na.scale", "k.scale"))
                expected_shape = () if n_cv == 1 else (n_cv,)
                self.assertTrue(all(state.value.shape == expected_shape for state in states.values()))
                self.assertEqual(cell.V.value.shape, (2, n_cv))

    def test_small_block_bptt_and_rtrl_match(self) -> None:
        config = BenchmarkConfig(n_cv=1, duration_ms=0.1, batch_size=2, n_seed=2)
        outputs = {}
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            for method in ("bptt", "rtrl"):
                prepared = prepare_benchmark(config, method)
                outputs[method] = jax.jit(prepared.function)(prepared.seed_roots)
        bptt_loss, bptt_losses, bptt_gradient = outputs["bptt"]
        rtrl_loss, rtrl_losses, rtrl_gradient = outputs["rtrl"]
        self.assertEqual(bptt_gradient.shape, (2, 3))
        self.assertEqual(rtrl_gradient.shape, (2, 3))
        np.testing.assert_allclose(rtrl_loss, bptt_loss, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(rtrl_losses, bptt_losses, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(rtrl_gradient, bptt_gradient, rtol=1e-8, atol=1e-9)

    def test_all_factorial_cases_match_between_bptt_and_rtrl(self) -> None:
        config = BenchmarkConfig(n_cv=3, duration_ms=0.1, batch_size=2, n_seed=2)
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            for mechanism in MECHANISM_FACTORIAL_SPECS:
                outputs = {}
                for method in ("bptt", "rtrl"):
                    prepared = prepare_benchmark(config, method, mechanism=mechanism)
                    outputs[method] = jax.jit(prepared.function)(prepared.seed_roots)
                    self.assertEqual(
                        prepared.active_state_count_per_trajectory,
                        mechanism.state_variables_per_cv * config.n_cv,
                    )
                    self.assertEqual(
                        prepared.parameter_count_per_seed,
                        mechanism.trainable_channels_per_cv * config.n_cv,
                    )
                for bptt, rtrl in zip(outputs["bptt"], outputs["rtrl"]):
                    np.testing.assert_allclose(rtrl, bptt, rtol=1e-8, atol=1e-9)

    def test_seed_vmap_has_no_cross_seed_gradient_block(self) -> None:
        config = BenchmarkConfig(n_cv=1, duration_ms=0.1, batch_size=1, n_seed=2)
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            prepared = prepare_benchmark(config, "rtrl")
            compiled = jax.jit(prepared.function)
            baseline = compiled(prepared.seed_roots)
            shifted = tuple(root.at[0].add(0.1) for root in prepared.seed_roots)
            changed = compiled(shifted)
        self.assertFalse(bool(np.allclose(np.asarray(baseline[2][0]), np.asarray(changed[2][0]))))
        np.testing.assert_allclose(changed[0][1], baseline[0][1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(changed[2][1], baseline[2][1], rtol=0.0, atol=0.0)

    def test_ordinary_and_recursive_gradients_match(self) -> None:
        config = BenchmarkConfig(n_cv=3, duration_ms=0.1, batch_size=2, n_seed=2)
        outputs = {}
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            for backsub in ("recursive", "ordinary"):
                with patch.dict("os.environ", {"BRAINCELL_DHS_BACKSUB": backsub}):
                    for method in ("bptt", "rtrl"):
                        prepared = prepare_benchmark(config, method)
                        outputs[(backsub, method)] = jax.jit(prepared.function)(prepared.seed_roots)
        for ordinary, recursive in zip(outputs[("ordinary", "rtrl")], outputs[("recursive", "rtrl")]):
            np.testing.assert_allclose(ordinary, recursive, rtol=1e-9, atol=1e-9)
        for backsub in ("recursive", "ordinary"):
            for bptt, rtrl in zip(outputs[(backsub, "bptt")], outputs[(backsub, "rtrl")]):
                np.testing.assert_allclose(rtrl, bptt, rtol=1e-8, atol=1e-9)

    def test_dry_run_writes_manifest_without_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_suite(
                "pilot",
                output_dir=output,
                gpu=7,
                repeats=2,
                resume=False,
                dry_run=True,
                python_executable=None,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["suite"], "pilot")
            self.assertEqual(len(manifest["configs"]), 9)
            self.assertEqual((output / "results.csv").read_text(), "")

    def test_aggregate_results_compares_gradient_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trials = output / "trials"
            trials.mkdir(parents=True)
            config = BenchmarkConfig(1, 0.1, 1, 1)
            for method, gradient in (("bptt", np.asarray([[1.0, 2.0]])), ("rtrl", np.asarray([[1.0, 2.0 + 1e-12]]))):
                np.savez(
                    trials / f"{method}.npz", gradient=gradient, loss=np.asarray([3.0]), losses=np.asarray([[3.0]])
                )
                (trials / f"{config.id}__{method}.json").write_text(
                    json.dumps(
                        {
                            **config.__dict__,
                            "config_id": config.id,
                            "method": method,
                            "status": "ok",
                            "steady_median_seconds": 2.0 if method == "bptt" else 1.0,
                            "gradient_file": f"{method}.npz",
                        }
                    )
                )
            rows = aggregate_results(output)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["gradient_max_abs_error"] > 0.0 for row in rows))
            self.assertTrue(all(row["bptt_over_rtrl_time"] == 2.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
