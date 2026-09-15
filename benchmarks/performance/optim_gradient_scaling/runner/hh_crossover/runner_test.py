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

from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover.runner import (
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
    _stablehlo_operation_counts,
    _summarize_gpu_samples,
)


class ScalingBenchmarkTest(unittest.TestCase):
    def test_stablehlo_operation_counts_are_diagnostic_only(self) -> None:
        ir = '\n'.join([
            '%0 = "stablehlo.gather"(%arg0) : () -> ()',
            '%1 = "stablehlo.scatter"(%arg0) : () -> ()',
            '%2 = "stablehlo.gather"(%arg0) : () -> ()',
        ])
        counts = _stablehlo_operation_counts(ir)
        self.assertEqual(counts["gather"], 2)
        self.assertEqual(counts["scatter"], 1)
        self.assertEqual(counts["while"], 0)

    def test_rtrl_profile_suite_has_only_three_full_hh_cases(self) -> None:
        cases = suite_cases("rtrl_profile")
        self.assertEqual(tuple(case.config.n_cv for case in cases), (1, 5, 21))
        self.assertTrue(all(case.mechanism == MECHANISM_FACTORIAL_SPECS[-1] for case in cases))

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


class ExplicitDeviceTest(unittest.TestCase):
    def test_device_selection_precedes_worker_launch(self):
        from unittest.mock import patch
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        with patch.object(driver, "run_suite") as launch:
            for args in (["run"], ["run", "--gpu", "-1"]):
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    driver.main(args)
            launch.assert_not_called()

    def test_selected_device_and_interpreter_are_forwarded(self):
        from unittest.mock import patch
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        with patch.object(driver, "run_suite") as launch:
            driver.main(["run", "--gpu", "0", "--python", "custom-python"])
            self.assertEqual(launch.call_args.kwargs["gpu"], 0)
            self.assertEqual(launch.call_args.kwargs["python_executable"], Path("custom-python"))


class CrossoverProtocolTest(unittest.TestCase):
    def test_primitive_counts_accept_self_referencing_jaxpr_property(self):
        from types import SimpleNamespace
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        graph = SimpleNamespace(eqns=[SimpleNamespace(
            primitive=SimpleNamespace(name="add"), params={})])
        graph.jaxpr = graph  # JAX 0.11 Jaxpr compatibility property.
        self.assertEqual(driver._jaxpr_primitive_counts(graph), {"add": 1})

    """Validate orchestration without compiling or executing a neuron model."""

    def test_matrix_and_dry_run_counts(self):
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        cases = driver.suite_cases("hh_crossover")
        self.assertEqual(len(cases), 12)
        self.assertEqual([case.config.n_cv for case in cases], [1]*3 + [21]*3 + [41]*3 + [81]*3)
        self.assertEqual([case.mechanism.trainable_channels_per_cv for case in cases], [1, 2, 3]*4)
        self.assertTrue(all(case.mechanism.state_variables_per_cv == 4 for case in cases))
        with tempfile.TemporaryDirectory() as directory, patch.object(driver, "_provenance", return_value={}), \
                patch.object(driver.subprocess, "run") as launch:
            output = Path(directory)
            driver.run_suite("hh_crossover", output_dir=output, gpu=0, repeats=5, warmups=2,
                             gpu_monitor=False, resume=False, dry_run=True)
            launch.assert_not_called()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["planned_counts"], {
                "workers": 24, "target_rollouts": 24, "first_executions": 24,
                "extra_warmups": 48, "timed_executions": 120,
                "gradient_calls": 192, "total_workload_calls": 216,
            })
            order = manifest["execution_order"]
            self.assertEqual([row["method"] for row in order[:6]],
                             ["bptt", "rtrl", "rtrl", "bptt", "bptt", "rtrl"])
            for row in order:
                command = row["command"]
                self.assertEqual(command[command.index("--warmups") + 1], "2")
                self.assertEqual(command[command.index("--repeats") + 1], "5")
                self.assertIn("--no-gpu-monitor", command)
            self.assertEqual(len(list(__import__('csv').DictReader((output / 'paired_results.csv').open()))), 12)
            with self.assertRaises(FileExistsError):
                driver.run_suite("hh_crossover", output_dir=output, gpu=0, repeats=5,
                                 resume=False, dry_run=True)

    def test_filtered_matrix_and_unlimited_workers(self):
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory, patch.object(driver, "_provenance", return_value={}), \
                patch.object(driver.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as launch:
            output = Path(directory)
            driver.run_suite("hh_crossover", output_dir=output, gpu=0, repeats=5, warmups=2,
                             gpu_monitor=False, resume=False, dry_run=False, cv_values=(1, 21, 41),
                             worker_timeout_seconds=0, budget_seconds=0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["configs"]), 9)
            self.assertEqual(manifest["planned_counts"]["workers"], 18)
            self.assertEqual(manifest["planned_counts"]["gradient_calls"], 144)
            self.assertEqual(manifest["planned_counts"]["total_workload_calls"], 162)
            self.assertEqual(launch.call_count, 18)
            self.assertTrue(all(call.kwargs["timeout"] is None for call in launch.call_args_list))
            self.assertTrue(all(call.kwargs["env"]["JAX_ENABLE_COMPILATION_CACHE"] == "false"
                                for call in launch.call_args_list))
            self.assertEqual(len(list((output / "trials").glob("*.json"))), 18)

    def test_hh_intermediate_cv_point_is_explicitly_supported(self):
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        with tempfile.TemporaryDirectory() as directory, patch.object(driver, "_provenance", return_value={}):
            output = Path(directory)
            driver.run_suite("hh_crossover", output_dir=output, gpu=0, repeats=5, warmups=2,
                             gpu_monitor=False, resume=False, dry_run=True, cv_values=(61,),
                             worker_timeout_seconds=0, budget_seconds=0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["configs"]), 3)
            self.assertEqual({config["n_cv"] for config in manifest["configs"]}, {61})
            self.assertEqual(manifest["planned_counts"]["workers"], 6)

    def test_worker_executes_exactly_first_plus_warmups_plus_repeats(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        prepared = SimpleNamespace(seed_roots=(np.ones((16, 1)),), function=object(),
                                   rtrl_carry_bytes=None, state_scalar_count_per_seed=64,
                                   parameter_count_per_seed=1, active_state_count_per_trajectory=4)
        compiled = MagicMock(return_value=(np.ones(16), np.ones((16, 1600)), np.ones((16, 1))))
        compiled.memory_analysis.return_value = SimpleNamespace(argument_size_in_bytes=128,
            output_size_in_bytes=256, temp_size_in_bytes=512, alias_size_in_bytes=0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(driver, "prepare_benchmark", return_value=prepared) as prepare, \
                patch.object(driver.jax, "jit") as jit, \
                patch.object(driver.jax, "devices", return_value=[SimpleNamespace(device_kind="fake")]), \
                patch.object(driver.jax, "default_backend", return_value="fake"), \
                patch.object(driver, "_GpuPhaseMonitor") as monitor:
            jit.return_value.lower.return_value.compile.return_value = compiled
            monitor.return_value.stop.return_value = driver._summarize_gpu_samples(())
            result = driver.run_trial(driver.BenchmarkConfig(1, 40.0, 16, 16), "bptt", repeats=5,
                warmups=2, output_path=Path(directory)/"trial.json", physical_gpu=0, gpu_monitor=False,
                mechanism=driver.suite_cases("hh_crossover")[0].mechanism)
            prepare.assert_called_once()
            self.assertEqual(compiled.call_count, 8)
            self.assertEqual(result["first_executions_completed"], 1)
            self.assertEqual(len(result["warmup_seconds"]), 2)
            self.assertEqual(len(result["steady_seconds"]), 5)
            self.assertEqual(result["target_rollouts_completed"], 1)
            self.assertTrue(all(call.args == (None,) for call in monitor.call_args_list))
            jit.return_value.lower.return_value.compile.assert_called_once()

    def test_compile_diagnostics_reuses_one_trace_and_compilation(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        prepared = SimpleNamespace(seed_roots=(np.ones((16, 1)),), function=object(),
                                   rtrl_carry_bytes=None, state_scalar_count_per_seed=64,
                                   parameter_count_per_seed=1, active_state_count_per_trajectory=4)
        compiled = MagicMock(return_value=(np.ones(16), np.ones((16, 1600)), np.ones((16, 1))))
        compiled.memory_analysis.return_value = SimpleNamespace(argument_size_in_bytes=128,
            output_size_in_bytes=256, temp_size_in_bytes=512, alias_size_in_bytes=0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(driver, "prepare_benchmark", return_value=prepared) as prepare, \
                patch.object(driver.jax, "jit") as jit, \
                patch.object(driver.jax, "devices", return_value=[SimpleNamespace(device_kind="fake")]), \
                patch.object(driver.jax, "default_backend", return_value="fake"), \
                patch.object(driver, "_GpuPhaseMonitor") as monitor:
            traced = SimpleNamespace(jaxpr=SimpleNamespace(eqns=[]))
            lowered = MagicMock()
            lowered.compile.return_value = compiled
            lowered.compiler_ir.return_value = "module {}"
            traced.lower = MagicMock(return_value=lowered)
            jit.return_value.trace.return_value = traced
            monitor.return_value.stop.return_value = driver._summarize_gpu_samples(())
            result = driver.run_trial(driver.BenchmarkConfig(1, 40.0, 16, 16), "bptt", repeats=5,
                warmups=2, compile_diagnostics=True, output_path=Path(directory)/"trial.json", physical_gpu=0, gpu_monitor=False,
                mechanism=driver.suite_cases("hh_crossover")[0].mechanism)
            prepare.assert_called_once()
            self.assertEqual(compiled.call_count, 8)
            self.assertEqual(result["first_executions_completed"], 1)
            self.assertEqual(len(result["warmup_seconds"]), 2)
            self.assertEqual(len(result["steady_seconds"]), 5)
            self.assertEqual(result["target_rollouts_completed"], 1)
            self.assertTrue(all(call.args == (None,) for call in monitor.call_args_list))
            jit.return_value.trace.assert_called_once()
            traced.lower.assert_called_once()
            lowered.compile.assert_called_once()
            self.assertAlmostEqual(result["compile_seconds"], sum(result[k] for k in
                ("trace_seconds", "lower_seconds", "xla_compile_seconds")))
            self.assertEqual(result["jaxpr_primitive_counts"], {})
            self.assertEqual((Path(directory)/result["stablehlo_file"]).read_text(), "module {}")

    def test_diagnostic_export_error_preserves_completed_measurement(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        prepared = SimpleNamespace(seed_roots=(np.ones((16, 1)),), function=object(),
                                   rtrl_carry_bytes=None, state_scalar_count_per_seed=64,
                                   parameter_count_per_seed=1, active_state_count_per_trajectory=4)
        compiled = MagicMock(return_value=(np.ones(16), np.ones((16, 1600)), np.ones((16, 1))))
        compiled.memory_analysis.return_value = SimpleNamespace(argument_size_in_bytes=128,
            output_size_in_bytes=256, temp_size_in_bytes=512, alias_size_in_bytes=0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(driver, "prepare_benchmark", return_value=prepared) as prepare, \
                patch.object(driver.jax, "jit") as jit, \
                patch.object(driver.jax, "devices", return_value=[SimpleNamespace(device_kind="fake")]), \
                patch.object(driver.jax, "default_backend", return_value="fake"), \
                patch.object(driver, "_GpuPhaseMonitor") as monitor, \
                patch.object(driver, "_jaxpr_primitive_counts", side_effect=RuntimeError("export failed")):
            traced = SimpleNamespace(jaxpr=SimpleNamespace(eqns=[]))
            lowered = MagicMock()
            lowered.compile.return_value = compiled
            lowered.compiler_ir.return_value = "module {}"
            traced.lower = MagicMock(return_value=lowered)
            jit.return_value.trace.return_value = traced
            monitor.return_value.stop.return_value = driver._summarize_gpu_samples(())
            result = driver.run_trial(driver.BenchmarkConfig(1, 40.0, 16, 16), "bptt", repeats=5,
                warmups=2, compile_diagnostics=True, output_path=Path(directory)/"trial.json", physical_gpu=0, gpu_monitor=False,
                mechanism=driver.suite_cases("hh_crossover")[0].mechanism)
            prepare.assert_called_once()
            self.assertEqual(compiled.call_count, 8)
            self.assertEqual(result["first_executions_completed"], 1)
            self.assertEqual(len(result["warmup_seconds"]), 2)
            self.assertEqual(len(result["steady_seconds"]), 5)
            self.assertEqual(result["target_rollouts_completed"], 1)
            self.assertTrue(all(call.args == (None,) for call in monitor.call_args_list))
            jit.return_value.trace.assert_called_once()
            traced.lower.assert_called_once()
            lowered.compile.assert_called_once()
            self.assertAlmostEqual(result["compile_seconds"], sum(result[k] for k in
                ("trace_seconds", "lower_seconds", "xla_compile_seconds")))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["diagnostics_status"], "error")
            self.assertEqual(result["diagnostics_error"], "export failed")
            self.assertEqual(result["temporary_bytes"], 512)
            self.assertTrue((Path(directory)/result["gradient_file"]).exists())
            saved = json.loads((Path(directory)/"trial.json").read_text())
            self.assertEqual(saved["status"], "ok")
            self.assertEqual(len(saved["steady_seconds"]), 5)

    def test_timeout_preserves_completed_counts_and_never_retries(self):
        import subprocess
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        cases = driver.suite_cases("hh_crossover")[:1]
        def timeout(command, **kwargs):
            trial = Path(command[command.index("--output") + 1])
            trial.write_text(json.dumps({"status": "running", "phase": "steady",
                                        "first_executions_completed": 1,
                                        "warmup_seconds": [1, 1], "steady_seconds": [1]}))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with tempfile.TemporaryDirectory() as directory, patch.object(driver, "_provenance", return_value={}), \
                patch.object(driver, "suite_cases", return_value=cases), \
                patch.object(driver.subprocess, "run", side_effect=timeout) as launch:
            output = Path(directory)
            driver.run_suite("hh_crossover", output_dir=output, gpu=0, repeats=5, warmups=2,
                             resume=False, dry_run=False, worker_timeout_seconds=20, budget_seconds=60)
            self.assertEqual(launch.call_count, 2)
            trials = list((output / "trials").glob("*.json"))
            self.assertEqual(len(trials), 2)
            for path in trials:
                row = json.loads(path.read_text())
                self.assertEqual(row["status"], "timeout")
                self.assertEqual(row["steady_seconds"], [1])
            counts = json.loads((output / "actual_counts.json").read_text())
            self.assertEqual(counts["gradient_calls_completed"], 8)
            self.assertTrue(counts["incomplete_counts_are_lower_bounds"])

    def test_budget_marks_remaining_trials_without_launch(self):
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        with tempfile.TemporaryDirectory() as directory, patch.object(driver, "_provenance", return_value={}), \
                patch.object(driver.time, "monotonic", side_effect=[0] + [100]*26), \
                patch.object(driver.subprocess, "run") as launch:
            output = Path(directory)
            driver.run_suite("hh_crossover", output_dir=output, gpu=0, repeats=5,
                             resume=False, dry_run=False, budget_seconds=10)
            launch.assert_not_called()
            rows = [json.loads(path.read_text()) for path in (output / "trials").glob("*.json")]
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(row["status"] == "not_run_budget" for row in rows))

    def test_mismatch_and_nonfinite_do_not_establish_crossovers(self):
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        for rtrl_gradient in (np.array([3.0]), np.array([np.nan])):
            with self.subTest(rtrl_gradient=rtrl_gradient), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                trials = output / "trials"
                trials.mkdir()
                for method, gradient in (("bptt", np.array([1.])), ("rtrl", rtrl_gradient)):
                    np.savez(trials/f"{method}.npz", gradient=gradient, loss=np.array([1.]), losses=np.array([1.]))
                    (trials/f"{method}.json").write_text(json.dumps({
                        "config_id": "test", "method": method, "status": "ok",
                        "steady_median_seconds": 1., "gradient_file": f"{method}.npz"}))
                rows = driver.aggregate_results(output)
                self.assertTrue(all(not row["usable_for_speed"] for row in rows))

    def test_failure_classification(self):
        from benchmarks.performance.optim_gradient_scaling.runner.hh_crossover import runner as driver
        self.assertEqual(driver._failure_status("RESOURCE_EXHAUSTED: allocation failed"), "oom")
        self.assertEqual(driver._failure_status("LLVM ERROR: failed"), "error")
