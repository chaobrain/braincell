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

"""GPU scaling benchmark for reverse BPTT and block-exact full RTRL.

The public ``run`` command launches every method/configuration in a fresh
subprocess. The private ``worker`` command owns one JAX process and writes one
structured trial. Generated results live under the ignored ``artifacts/``
directory beside this module.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import brainstate
import brainunit as u
import jax
import numpy as np

from validation.optim._workload import (
    BACKSUBS,
    BenchmarkConfig,
    DT_MS,
    FULL_HH_SPEC,
    MECHANISM_FACTORIAL_SPECS,
    METHODS,
    MechanismSpec,
    PreparedBenchmark as PreparedBenchmark,
    RNG_SEED,
    _BASE_G_MAX as _BASE_G_MAX,
    _CHANNEL_ORDER as _CHANNEL_ORDER,
    _STATE_VARIABLES_PER_CHANNEL as _STATE_VARIABLES_PER_CHANNEL,
    _tree_nbytes as _tree_nbytes,
    build_cell as build_cell,
    build_morphology as build_morphology,
    current_amplitudes as current_amplitudes,
    prepare_benchmark,
    seed_parameter_roots as seed_parameter_roots,
    simulate_voltage as simulate_voltage,
    target_row_scales as target_row_scales,
)

BASELINE = {"n_cv": 5, "duration_ms": 40.0, "batch_size": 16, "n_seed": 16}
ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "rtrl_bptt_scaling"


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark configuration plus an optional mechanism case suffix."""

    config: BenchmarkConfig
    mechanism: MechanismSpec = FULL_HH_SPEC
    suffix: str | None = None

    @property
    def id(self) -> str:
        """Return the persistent case identifier."""
        return self.config.id if self.suffix is None else f"{self.config.id}__{self.suffix}"


def suite_configs(name: str) -> tuple[BenchmarkConfig, ...]:
    """Return deterministic pilot, full, or large-CV configurations."""
    if name == "large_cv":
        return tuple(BenchmarkConfig(c, 40.0, 16, 16) for c in (13, 17, 25, 33))
    if name == "backsub_ab":
        return tuple(BenchmarkConfig(c, 40.0, 16, 16) for c in (9, 17, 25, 33))
    baseline = BenchmarkConfig(**BASELINE)
    configs = {baseline}
    if name == "pilot":
        configs.update(BenchmarkConfig(c, 40.0, 16, 16) for c in (1, 9))
        configs.update(BenchmarkConfig(5, t, 16, 16) for t in (10.0, 80.0))
        configs.update(BenchmarkConfig(5, 40.0, b, 16) for b in (1, 32))
        configs.update(BenchmarkConfig(5, 40.0, 16, s) for s in (1, 32))
    elif name == "full":
        configs.update(BenchmarkConfig(c, 40.0, 16, 16) for c in (1, 3, 5, 7, 9))
        configs.update(BenchmarkConfig(5, t, 16, 16) for t in (10.0, 20.0, 40.0, 80.0))
        configs.update(BenchmarkConfig(5, 40.0, b, 16) for b in (1, 4, 16, 32))
        configs.update(BenchmarkConfig(5, 40.0, 16, s) for s in (1, 4, 16, 32))
        configs.update(
            {
                BenchmarkConfig(9, 80.0, 32, 32),
                BenchmarkConfig(9, 40.0, 16, 32),
                BenchmarkConfig(5, 80.0, 32, 16),
                BenchmarkConfig(5, 40.0, 32, 32),
            }
        )
    else:
        raise ValueError(f"Unknown suite {name!r}.")
    return tuple(sorted(configs))


def suite_cases(name: str) -> tuple[BenchmarkCase, ...]:
    """Return persistent cases for a standard or mechanism-factorial suite."""
    if name == "rtrl_profile":
        return tuple(
            BenchmarkCase(BenchmarkConfig(c, 40.0, 16, 16), FULL_HH_SPEC, "full_hh")
            for c in (1, 5, 21)
        )
    if name == "hh_crossover":
        return tuple(
            BenchmarkCase(BenchmarkConfig(c, 40.0, 16, 16), mechanism, mechanism.name)
            for c in (1, 21, 41, 81)
            for mechanism in MECHANISM_FACTORIAL_SPECS
            if mechanism.painted_channels == FULL_HH_SPEC.painted_channels
        )
    if name == "mechanism_factorial":
        return tuple(
            BenchmarkCase(
                BenchmarkConfig(n_cv, 40.0, 16, 16),
                mechanism,
                suffix=mechanism.name,
            )
            for n_cv in (3, 5, 9, 17, 33)
            for mechanism in MECHANISM_FACTORIAL_SPECS
        )
    return tuple(BenchmarkCase(config) for config in suite_configs(name))


def run_trial(
    config: BenchmarkConfig,
    method: str,
    *,
    repeats: int,
    output_path: Path,
    physical_gpu: int | None,
    backsub: str = "recursive",
    mechanism: MechanismSpec = FULL_HH_SPEC,
    config_id: str | None = None,
    warmups: int = 0,
    gpu_monitor: bool = True,
    compile_diagnostics: bool = False,
    dhs_jvp_mode: str = "generic",
    rtrl_jvp_mode: str = "linearize",
) -> dict[str, object]:
    """Compile, execute, and persist one isolated benchmark trial."""
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative.")
    if backsub not in BACKSUBS:
        raise ValueError(f"backsub must be one of {BACKSUBS!r}.")
    if dhs_jvp_mode not in {"generic", "explicit"}:
        raise ValueError("dhs_jvp_mode must be 'generic' or 'explicit'.")
    if rtrl_jvp_mode not in {"linearize", "direct"}:
        raise ValueError("rtrl_jvp_mode must be 'linearize' or 'direct'.")
    os.environ["BRAINCELL_DHS_BACKSUB"] = backsub
    os.environ["BRAINCELL_DHS_CUSTOM_JVP"] = "1" if dhs_jvp_mode == "explicit" else "0"
    os.environ["BRAINCELL_RTRL_JVP_MODE"] = rtrl_jvp_mode
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        **asdict(config),
        "config_id": config.id if config_id is None else config_id,
        "method": method,
        "backsub": backsub,
        "mechanism_case": mechanism.name,
        "painted_channels": mechanism.painted_label,
        "trainable_channels": mechanism.trainable_label,
        "state_variables_per_cv": mechanism.state_variables_per_cv,
        "trainable_channels_per_cv": mechanism.trainable_channels_per_cv,
        "num_steps": config.num_steps,
        "repeats": repeats,
        "status": "running",
        "warmups": warmups,
        "gpu_monitor": gpu_monitor,
        "compile_diagnostics": compile_diagnostics,
        "dhs_jvp_mode": dhs_jvp_mode,
        "rtrl_jvp_mode": rtrl_jvp_mode,
        "target_rollouts_completed": 0,
        "first_executions_completed": 0,
        "warmup_seconds": [],
        "steady_seconds": [],
        "phase": "prepare",
        "dt_ms": DT_MS,
        "rng_seed": RNG_SEED,
        "precision": "float64",
        "solver": "staggered",
        "n_x": mechanism.state_variables_per_cv * config.n_cv,
        "n_theta": mechanism.trainable_channels_per_cv * config.n_cv,
        "software": _software_versions(),
    }
    _write_json(output_path, result)
    monitor_gpu = physical_gpu if gpu_monitor else None
    try:
        with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
            preparation_started = time.perf_counter()
            prepared = prepare_benchmark(config, method, mechanism=mechanism)
            result["preparation_seconds"] = time.perf_counter() - preparation_started
            if (prepared.active_state_count_per_trajectory != result["n_x"] or
                    prepared.parameter_count_per_seed != result["n_theta"]):
                raise RuntimeError("Materialized state or parameter dimensions disagree with the case.")
            result.update(target_rollouts_completed=1, phase="compile",
                          materialization_mode=getattr(prepared, "materialization_mode", None),
                          state_scalar_count_per_seed=prepared.state_scalar_count_per_seed,
                          parameter_count_per_seed=prepared.parameter_count_per_seed,
                          rtrl_carry_bytes=prepared.rtrl_carry_bytes)
            _write_json(output_path, result)
            arguments = (prepared.seed_roots,)
            compile_monitor = _GpuPhaseMonitor(monitor_gpu)
            compile_monitor.start()
            if compile_diagnostics:
                result["phase"] = "trace"
                _write_json(output_path, result)
                started = time.perf_counter()
                traced = jax.jit(prepared.function).trace(*arguments)
                result["trace_seconds"] = time.perf_counter() - started
                result["phase"] = "lower"
                _write_json(output_path, result)
                started = time.perf_counter()
                lowered = traced.lower()
                result["lower_seconds"] = time.perf_counter() - started
                result["phase"] = "xla_compile"
                _write_json(output_path, result)
                started = time.perf_counter()
                compiled = lowered.compile()
                result["xla_compile_seconds"] = time.perf_counter() - started
                compile_seconds = sum(result[key] for key in ("trace_seconds", "lower_seconds", "xla_compile_seconds"))
            else:
                started = time.perf_counter()
                compiled = jax.jit(prepared.function).lower(*arguments).compile()
                compile_seconds = time.perf_counter() - started
            compile_metrics = compile_monitor.stop()
            result.update(compile_seconds=compile_seconds, phase="first")
            _write_json(output_path, result)

            first_monitor = _GpuPhaseMonitor(monitor_gpu)
            first_monitor.start()
            started = time.perf_counter()
            first_output = compiled(*arguments)
            _block_until_ready(first_output)
            first_seconds = time.perf_counter() - started
            first_metrics = first_monitor.stop()
            result.update(first_seconds=first_seconds, first_executions_completed=1, phase="warmup")
            _write_json(output_path, result)
            warmup_seconds = result["warmup_seconds"]
            for _ in range(warmups):
                started = time.perf_counter()
                output = compiled(*arguments)
                _block_until_ready(output)
                warmup_seconds.append(time.perf_counter() - started)
                _write_json(output_path, result)
            result["phase"] = "steady"
            _write_json(output_path, result)

            steady_monitor = _GpuPhaseMonitor(monitor_gpu)
            steady_monitor.start()
            steady = result["steady_seconds"]
            for _ in range(repeats):
                started = time.perf_counter()
                output = compiled(*arguments)
                _block_until_ready(output)
                steady.append(time.perf_counter() - started)
                _write_json(output_path, result)
            steady_metrics = steady_monitor.stop()

            loss, losses, gradient = first_output
            gradient_np = np.asarray(gradient)
            loss_np = np.asarray(loss)
            losses_np = np.asarray(losses)
            gradient_path = output_path.with_suffix(".npz")
            np.savez_compressed(gradient_path, loss=loss_np, losses=losses_np, gradient=gradient_np)
            memory = compiled.memory_analysis()
            result.update(
                {
                    "status": "ok" if all(np.isfinite(x).all() for x in (loss_np, losses_np, gradient_np)) else "numerical_failure",
                    "phase": "complete",
                    "device_kind": jax.devices()[0].device_kind,
                    "backend": jax.default_backend(),
                    "device": str(jax.devices()[0]),
                    "jax_version": jax.__version__,
                    "compile_seconds": compile_seconds,
                    "first_seconds": first_seconds,
                    "steady_seconds": steady,
                    "steady_median_seconds": float(np.median(steady)),
                    "steady_iqr_seconds": float(np.quantile(steady, 0.75) - np.quantile(steady, 0.25)),
                    "steady_min_seconds": float(np.min(steady)),
                    "steady_p90_seconds": float(np.quantile(steady, 0.9)),
                    "argument_bytes": int(memory.argument_size_in_bytes),
                    "output_bytes": int(memory.output_size_in_bytes),
                    "temporary_bytes": int(memory.temp_size_in_bytes),
                    "alias_bytes": int(memory.alias_size_in_bytes),
                    "rtrl_carry_bytes": prepared.rtrl_carry_bytes,
                    "state_scalar_count_per_seed": prepared.state_scalar_count_per_seed,
                    "parameter_count_per_seed": prepared.parameter_count_per_seed,
                    "parameter_count_total": prepared.parameter_count_per_seed * config.n_seed,
                    "n_x": prepared.active_state_count_per_trajectory,
                    "n_theta": prepared.parameter_count_per_seed,
                    "active_state_estimate_per_seed": (config.batch_size * prepared.active_state_count_per_trajectory),
                    "gradient_shape": list(gradient_np.shape),
                    "loss_shape": list(loss_np.shape),
                    "losses_shape": list(losses_np.shape),
                    "gradient_l2": float(np.linalg.norm(gradient_np)),
                    "loss_mean": float(np.mean(loss_np)),
                    "gradient_file": gradient_path.name,
                    "host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                    "throughput_seed_batch_steps_per_second": (
                        config.n_seed * config.batch_size * config.num_steps / float(np.median(steady))
                    ),
                }
            )
            result.update(_phase_metric_fields("compile", compile_metrics))
            result.update(_phase_metric_fields("first", first_metrics))
            result.update(_phase_metric_fields("steady", steady_metrics))
            # Persist the measured result before optional graph diagnostics.
            _write_json(output_path, result)
            if compile_diagnostics:
                try:
                    # Reuse traced/lowered objects; no extra model calls.
                    ir = str(lowered.compiler_ir())
                    ir_path = output_path.with_suffix(".stablehlo.mlir")
                    ir_path.write_text(ir, encoding="utf-8")
                    result.update(
                        stablehlo_file=ir_path.name,
                        stablehlo_bytes=len(ir.encode("utf-8")),
                        stablehlo_operation_counts=_stablehlo_operation_counts(ir),
                        jaxpr_primitive_counts=_jaxpr_primitive_counts(traced.jaxpr),
                        diagnostics_status="ok",
                    )
                    try:
                        cost = compiled.cost_analysis()
                        if isinstance(cost, list):
                            cost = cost[0] if len(cost) == 1 else cost
                        if isinstance(cost, dict):
                            result["xla_cost_analysis"] = {
                                str(key): float(value) if isinstance(value, (int, float, np.number)) else value
                                for key, value in cost.items()
                            }
                        else:
                            result["xla_cost_analysis_status"] = "unsupported"
                            result["xla_cost_analysis_type"] = type(cost).__name__
                    except Exception as exc:
                        result["xla_cost_analysis_status"] = "error"
                        result["xla_cost_analysis_error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    result.update(diagnostics_status="error", diagnostics_error_type=type(exc).__name__,
                                  diagnostics_error=str(exc))
                    print(f"Compile diagnostic export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    except Exception as exc:
        result.update({"status": _failure_status(str(exc)), "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(output_path, result)
        raise
    _write_json(output_path, result)
    return result


def run_suite(
    suite: str,
    *,
    output_dir: Path,
    gpu: int,
    repeats: int,
    resume: bool,
    dry_run: bool,
    python_executable: Path | None = None,
    backsub: str = "recursive",
    warmups: int = 0,
    gpu_monitor: bool = True,
    compile_diagnostics: bool = False,
    dhs_jvp_mode: str = "generic",
    rtrl_jvp_mode: str = "linearize",
    worker_timeout_seconds: float = 1200.0,
    budget_seconds: float = 7200.0,
    cv_values: tuple[int, ...] | None = None,
) -> Path:
    """Launch isolated workers and aggregate their results."""
    run_started = time.monotonic()
    cases = suite_cases(suite)
    if cv_values is not None:
        available = {case.config.n_cv for case in cases}
        requested = tuple(dict.fromkeys(int(value) for value in cv_values))
        if not requested or any(value < 1 or value % 2 == 0 for value in requested):
            raise ValueError("cv_values must contain positive odd CV counts.")
        if suite == "hh_crossover":
            # Keep the standard suite points and permit explicitly requested
            # odd intermediate points for boundary refinement (for example C=61).
            mechanisms = tuple(case.mechanism for case in cases[:3])
            cases = [
                BenchmarkCase(BenchmarkConfig(value, 40.0, 16, 16), mechanism, mechanism.name)
                for value in requested for mechanism in mechanisms
            ]
        elif not set(requested).issubset(available):
            raise ValueError("cv_values must select existing suite CV counts.")
        else:
            cases = [case for case in cases if case.config.n_cv in requested]
    if repeats < 1 or warmups < 0 or worker_timeout_seconds < 0 or budget_seconds < 0:
        raise ValueError("Invalid repetitions, warmups, timeout or budget.")
    if suite == "hh_crossover" and resume:
        raise ValueError("hh_crossover requires a fresh run; failed trials are not retried.")
    if suite == "hh_crossover" and (output_dir / "raw" / "manifest.json").exists():
        raise FileExistsError("hh_crossover requires a new output directory.")
    if backsub not in BACKSUBS:
        raise ValueError(f"backsub must be one of {BACKSUBS!r}.")
    if dhs_jvp_mode not in {"generic", "explicit"}:
        raise ValueError("dhs_jvp_mode must be 'generic' or 'explicit'.")
    if rtrl_jvp_mode not in {"linearize", "direct"}:
        raise ValueError("rtrl_jvp_mode must be 'linearize' or 'direct'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = output_dir / "raw" / "trials"
    log_dir = output_dir / "raw" / "logs"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": suite,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "repeats": repeats,
        "warmups": warmups,
        "gpu_monitor": gpu_monitor,
        "compile_diagnostics": compile_diagnostics,
        "dhs_jvp_mode": dhs_jvp_mode,
        "rtrl_jvp_mode": rtrl_jvp_mode,
        "worker_timeout_seconds": worker_timeout_seconds,
        "budget_seconds": budget_seconds,
        "cv_values": cv_values,
        "compilation_cache_enabled": False,
        "independent_rounds": 1,
        "planned_counts": {
            "workers": len(cases) * 2,
            "target_rollouts": len(cases) * 2,
            "first_executions": len(cases) * 2,
            "extra_warmups": len(cases) * 2 * warmups,
            "timed_executions": len(cases) * 2 * repeats,
            "gradient_calls": len(cases) * 2 * (1 + warmups + repeats),
            "total_workload_calls": len(cases) * 2 * (2 + warmups + repeats),
        },
        "provenance": _provenance(output_dir, gpu),
        "dt_ms": DT_MS,
        "rng_seed": RNG_SEED,
        "methods": METHODS,
        "backsub": backsub,
        "python_executable": str(python_executable or sys.executable),
        "configs": [
            asdict(case.config)
            | {
                "config_id": case.id,
                "mechanism": asdict(case.mechanism),
            }
            for case in cases
        ],
    }
    _write_json(output_dir / "raw" / "manifest.json", manifest)
    commands = []
    worker_python = str(python_executable or sys.executable)
    for case_index, case in enumerate(cases):
        config = case.config
        methods = tuple(reversed(METHODS)) if suite == "hh_crossover" and case_index % 2 else METHODS
        for method in methods:
            backsub_suffix = "" if backsub == "recursive" else f"__{backsub}"
            trial_path = trial_dir / f"{case.id}__{method}{backsub_suffix}.json"
            if resume and _trial_succeeded(trial_path):
                continue
            command = [
                worker_python,
                str(Path(__file__).resolve()),
                "worker",
                "--config",
                json.dumps(asdict(config)),
                "--config-id",
                case.id,
                "--mechanism",
                json.dumps(asdict(case.mechanism)),
                "--method",
                method,
                "--repeats",
                str(repeats),
                "--warmups",
                str(warmups),
                "--gpu-monitor" if gpu_monitor else "--no-gpu-monitor",
                *( ["--compile-diagnostics"] if compile_diagnostics else [] ),
                "--dhs-jvp-mode",
                dhs_jvp_mode,
                "--rtrl-jvp-mode",
                rtrl_jvp_mode,
                "--output",
                str(trial_path),
                "--physical-gpu",
                str(gpu),
                "--backsub",
                backsub,
            ]
            commands.append((case, method, trial_path, command))
    manifest["execution_order"] = [
        {"config_id": case.id, "method": method, "command": command}
        for case, method, _, command in commands
    ]
    _write_json(output_dir / "raw" / "manifest.json", manifest)
    if dry_run:
        for _, _, _, command in commands:
            print(" ".join(command))
        aggregate_results(output_dir)
        return output_dir

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAX_PLATFORMS": "cuda",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
    )
    for index, (case, method, trial_path, command) in enumerate(commands, start=1):
        remaining = budget_seconds - (time.monotonic() - run_started) if budget_seconds else float("inf")
        base = {
            **asdict(case.config), "config_id": case.id, "mechanism_case": case.mechanism.name,
            "method": method, "execution_index": index,
            "n_x": case.mechanism.state_variables_per_cv * case.config.n_cv,
            "n_theta": case.mechanism.trainable_channels_per_cv * case.config.n_cv,
        }
        if remaining <= 0:
            _write_json(trial_path, base | {"status": "not_run_budget"})
            aggregate_results(output_dir)
            continue
        print(f"[{index}/{len(commands)}] {case.id} {method}", flush=True)
        timeout = min(worker_timeout_seconds or float("inf"), remaining)
        timeout = None if timeout == float("inf") else timeout
        log_path = log_dir / f"{case.id}__{method}.log"
        failure = None
        worker_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           text=True, check=False, timeout=timeout)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = None
                failure = "budget_timeout" if remaining <= (worker_timeout_seconds or float("inf")) else "timeout"
            except OSError as exc:
                returncode = None
                failure = "launch_error"
                log.write(str(exc))
        row = base | (_read_json(trial_path) if trial_path.exists() else {})
        row.update(execution_index=index, returncode=returncode,
                   worker_wall_seconds=time.monotonic() - worker_started)
        if failure:
            row["status"] = failure
        elif returncode != 0 and row.get("status") not in {"oom", "numerical_failure", "error"}:
            row["status"] = _failure_status(log_path.read_text())
        elif returncode == 0 and row.get("status") not in {"ok", "numerical_failure"}:
            row["status"] = "missing_result"
        _write_json(trial_path, row)
        aggregate_results(output_dir)
        print(f"  {row['status']} ({row['worker_wall_seconds']:.1f}s)", flush=True)
    manifest["wall_seconds"] = time.monotonic() - run_started
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "raw" / "manifest.json", manifest)

    return output_dir


def aggregate_results(output_dir: Path) -> list[dict[str, object]]:
    """Combine trial JSON files and attach pairwise correctness metrics."""
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    trial_dir = output_dir / "raw" / "trials"
    rows = [_read_json(path) for path in sorted(trial_dir.glob("*.json"))]
    by_config: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_config.setdefault(str(row["config_id"]), {})[str(row["method"])] = row
    for methods in by_config.values():
        bptt = methods.get("bptt")
        rtrl = methods.get("rtrl")
        if not bptt or not rtrl or bptt.get("status") != "ok" or rtrl.get("status") != "ok":
            continue
        bptt_data = np.load(trial_dir / str(bptt["gradient_file"]))
        rtrl_data = np.load(trial_dir / str(rtrl["gradient_file"]))
        gradient_abs = np.abs(bptt_data["gradient"] - rtrl_data["gradient"])
        loss_abs = np.abs(bptt_data["loss"] - rtrl_data["loss"])
        scale = np.maximum(np.abs(bptt_data["gradient"]), 1e-30)
        relative_l2 = float(np.linalg.norm(gradient_abs) / max(
            float(np.linalg.norm(bptt_data["gradient"])), float(np.linalg.norm(rtrl_data["gradient"])), 1e-30))
        finite = all(np.isfinite(data[key]).all() for data in (bptt_data, rtrl_data)
                     for key in ("gradient", "loss", "losses"))
        agreement = bool(finite and
                         np.allclose(bptt_data["loss"], rtrl_data["loss"], rtol=1e-7, atol=1e-8) and
                         np.allclose(bptt_data["losses"], rtrl_data["losses"], rtol=1e-7, atol=1e-8) and
                         (relative_l2 <= 1e-6 or float(np.max(gradient_abs)) <= 1e-8))
        comparison = {
            "gradient_relative_l2_error": relative_l2,
            "correctness_pass": agreement,
            "usable_for_speed": agreement,
            "rtrl_over_bptt_time": float(rtrl["steady_median_seconds"]) / float(bptt["steady_median_seconds"]),
            "gradient_max_abs_error": float(np.max(gradient_abs)),
            "gradient_max_rel_error": float(np.max(gradient_abs / scale)),
            "loss_max_abs_error": float(np.max(loss_abs)),
            "bptt_over_rtrl_time": (float(bptt["steady_median_seconds"]) / float(rtrl["steady_median_seconds"])),
        }
        bptt.update(comparison)
        rtrl.update(comparison)
    _write_csv(output_dir / "raw" / "results.csv", rows)
    if (output_dir / "raw" / "manifest.json").exists():
        manifest = _read_json(output_dir / "raw" / "manifest.json")
        if manifest.get("suite") == "hh_crossover":
            _write_crossover_summary(output_dir, rows, manifest)
    return rows


def _software_versions() -> dict[str, str]:
    versions = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
    for package in ("jax", "jaxlib", "brainstate", "brainunit", "braincell"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed as distribution"
    return versions


def _provenance(output_dir: Path, gpu: int) -> dict[str, object]:
    def git(*arguments):
        return subprocess.run(["git", "-C", str(_REPO_ROOT), *arguments],
                              capture_output=True, check=True).stdout

    diff = git("diff", "HEAD", "--binary")
    (output_dir / "raw" / "source.patch").write_bytes(diff)
    digest = hashlib.sha256()
    source_files = {}
    for name in git("ls-files").decode().splitlines():
        path = _REPO_ROOT / name
        if path.is_file():
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            source_files[name] = content_hash
            digest.update(f"{name}\0{content_hash}\n".encode())
    _write_json(output_dir / "raw" / "source_files.json", source_files)
    try:
        hardware = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=name,uuid,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        hardware = "unavailable"
    return {
        "git_head": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git("status", "--short").decode(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_tree_sha256": digest.hexdigest(),
        "software": _software_versions(), "gpu": hardware,
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("JAX_", "XLA_", "BRAINCELL_")) or key in
                        ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
    }


def _failure_status(message: str) -> str:
    lower = message.lower()
    if any(token in lower for token in ("out of memory", "resource_exhausted", "cuda_error_out_of_memory")):
        return "oom"
    return "error"


def _write_crossover_summary(output_dir: Path, rows, manifest) -> None:
    pairs = {(row["config_id"], row["method"]): row for row in rows}
    paired = []
    for config in manifest["configs"]:
        c = config["n_cv"]
        multiplier = len(config["mechanism"]["trainable_channels"])
        pair = {"config_id": config["config_id"], "n_cv": c, "n_x": 4 * c,
                "n_theta": multiplier * c, "parameter_multiplier": multiplier}
        methods = [pairs.get((config["config_id"], method), {}) for method in METHODS]
        for method, row in zip(METHODS, methods):
            pair[f"{method}_status"] = row.get("status", "pending")
            for key in ("steady_median_seconds", "steady_iqr_seconds", "temporary_bytes", "argument_bytes",
                        "output_bytes", "alias_bytes", "rtrl_carry_bytes", "compile_seconds"):
                pair[f"{method}_{key}"] = row.get(key)
        bptt, rtrl = methods
        valid = bptt.get("status") == rtrl.get("status") == "ok" and bptt.get("correctness_pass", False)
        pair["usable_for_speed"] = valid
        pair["gradient_relative_l2_error"] = bptt.get("gradient_relative_l2_error")
        ratio = bptt.get("rtrl_over_bptt_time") if valid else None
        pair["rtrl_over_bptt_time"] = ratio
        pair["classification"] = (
            "unavailable" if ratio is None else "near" if 0.9 <= ratio <= 1.1 else
            "rtrl_faster" if ratio < 0.9 else "bptt_faster"
        )
        paired.append(pair)
    _write_csv(output_dir / "raw" / "paired_results.csv", paired)
    counts = {
        "workers_launched": sum(row.get("status") != "not_run_budget" for row in rows),
        "methods_ok": sum(row.get("status") == "ok" for row in rows),
        "target_rollouts_completed": sum(row.get("target_rollouts_completed", 0) for row in rows),
        "first_executions_completed": sum(row.get("first_executions_completed", 0) for row in rows),
        "extra_warmups_completed": sum(len(row.get("warmup_seconds", [])) for row in rows),
        "timed_executions_completed": sum(len(row.get("steady_seconds", [])) for row in rows),
        "incomplete_counts_are_lower_bounds": any(row.get("status") not in ("ok", "numerical_failure", "not_run_budget") for row in rows),
    }
    counts["gradient_calls_completed"] = sum(counts[key] for key in
        ("first_executions_completed", "extra_warmups_completed", "timed_executions_completed"))
    counts["total_workload_calls_completed"] = counts["gradient_calls_completed"] + counts["target_rollouts_completed"]
    _write_json(output_dir / "raw" / "actual_counts.json", counts)

    def number(value, scale=1.0):
        return "—" if value is None else f"{value / scale:.4g}"

    lines = ["# HH crossover coarse scan", "",
             "Times are seconds (median ± IQR; IQR is spread, not a confidence interval). Memory is XLA temporary MiB, not measured device peak.", "",
             "| C | Nx | Ntheta | BPTT s ± IQR | RTRL s ± IQR | RTRL/BPTT | BPTT MiB | RTRL MiB | Classification / status |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for pair in paired:
        timing = [f"{number(pair[f'{method}_steady_median_seconds'])} ± {number(pair[f'{method}_steady_iqr_seconds'])}" for method in METHODS]
        status = pair["classification"]
        if status == "unavailable":
            status += f" ({pair['bptt_status']}/{pair['rtrl_status']})"
        lines.append(f"| {pair['n_cv']} | {pair['n_x']} | {pair['n_theta']} | {timing[0]} | {timing[1]} | "
                     f"{number(pair['rtrl_over_bptt_time'])} | {number(pair['bptt_temporary_bytes'], 2**20)} | "
                     f"{number(pair['rtrl_temporary_bytes'], 2**20)} | {status} |")
    lines += ["", "## Candidate brackets", ""]
    candidates = []
    for multiplier in (1, 2, 3):
        line = [pair for pair in paired if pair["parameter_multiplier"] == multiplier]
        for left, right in zip(line, line[1:]):
            a, b = left["rtrl_over_bptt_time"], right["rtrl_over_bptt_time"]
            if a is not None and b is not None and (a - 1) * (b - 1) < 0:
                candidates.append(f"- Ntheta={multiplier}C: observed median ratio changes sides between C={left['n_cv']} and C={right['n_cv']}; requires confirmation.")
    lines += candidates or ["No adjacent measured points with a valid median-ratio sign change yet. This does not rule out crossings between sampled points."]
    lines += ["", "One independent round only; near means ratio within [0.9, 1.1]. Failed pairs cannot establish a crossover.", "",
              "Actual completed calls: `" + json.dumps(counts, sort_keys=True) + "`", ""]
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


class _GpuPhaseMonitor:
    """Poll process memory and device activity during one benchmark phase."""

    def __init__(self, physical_gpu: int | None, interval: float = 0.01) -> None:
        self.physical_gpu = physical_gpu
        self.interval = interval
        self.samples: list[dict[str, float | int | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.physical_gpu is None:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        if self._thread is None:
            return _summarize_gpu_samples(())
        self._stop.set()
        self._thread.join(timeout=2.0)
        return _summarize_gpu_samples(self.samples)

    def _poll(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.physical_gpu))
            pid = os.getpid()
            while not self._stop.is_set():
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                used = sum(int(item.usedGpuMemory) for item in processes if int(item.pid) == pid)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.samples.append(
                    {
                        "process_bytes": used,
                        "gpu_util_percent": float(utilization.gpu),
                        "memory_util_percent": float(utilization.memory),
                        "power_watts": float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
                        "sm_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                    }
                )
                self._stop.wait(self.interval)
        except ImportError:
            self._poll_nvidia_smi()
        except Exception:
            return

    def _poll_nvidia_smi(self) -> None:
        pid = os.getpid()
        while not self._stop.is_set():
            try:
                device = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.physical_gpu),
                        "--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2.0,
                )
                processes = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.physical_gpu),
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2.0,
                )
                used = 0
                for line in processes.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and int(fields[0]) == pid:
                        used += int(float(fields[1])) * 1024 * 1024
                device_fields = [field.strip() for field in device.stdout.strip().split(",")]
                if len(device_fields) == 4:
                    self.samples.append(
                        {
                            "process_bytes": used,
                            "gpu_util_percent": _optional_float(device_fields[0]),
                            "memory_util_percent": _optional_float(device_fields[1]),
                            "power_watts": _optional_float(device_fields[2]),
                            "sm_clock_mhz": _optional_float(device_fields[3]),
                        }
                    )
            except Exception:
                return
            self._stop.wait(max(self.interval, 0.05))


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_gpu_samples(samples) -> dict[str, float | int | None]:
    samples = tuple(samples)
    summary: dict[str, float | int | None] = {"sample_count": len(samples)}
    specifications = {
        "process_bytes": ("process_peak_bytes", "max"),
        "gpu_util_percent": ("gpu_util", "distribution"),
        "memory_util_percent": ("memory_util", "distribution"),
        "power_watts": ("power_watts", "median_max"),
        "sm_clock_mhz": ("sm_clock_mhz", "median"),
    }
    for source, (target, reduction) in specifications.items():
        values = np.asarray(
            [sample[source] for sample in samples if sample.get(source) is not None],
            dtype=np.float64,
        )
        if reduction == "max":
            summary[target] = None if values.size == 0 else int(np.max(values))
        elif reduction == "distribution":
            summary[f"{target}_median_percent"] = None if values.size == 0 else float(np.median(values))
            summary[f"{target}_p90_percent"] = None if values.size == 0 else float(np.quantile(values, 0.9))
            summary[f"{target}_max_percent"] = None if values.size == 0 else float(np.max(values))
        elif reduction == "median_max":
            summary[f"{target}_median"] = None if values.size == 0 else float(np.median(values))
            summary[f"{target}_max"] = None if values.size == 0 else float(np.max(values))
        else:
            summary[f"{target}_median"] = None if values.size == 0 else float(np.median(values))
    return summary


def _phase_metric_fields(phase: str, summary) -> dict[str, float | int | None]:
    return {
        f"gpu_samples_{phase}": summary["sample_count"],
        f"gpu_peak_{phase}_bytes": summary["process_peak_bytes"],
        f"gpu_util_{phase}_median_percent": summary["gpu_util_median_percent"],
        f"gpu_util_{phase}_p90_percent": summary["gpu_util_p90_percent"],
        f"gpu_util_{phase}_max_percent": summary["gpu_util_max_percent"],
        f"gpu_memory_util_{phase}_median_percent": summary["memory_util_median_percent"],
        f"gpu_memory_util_{phase}_p90_percent": summary["memory_util_p90_percent"],
        f"gpu_memory_util_{phase}_max_percent": summary["memory_util_max_percent"],
        f"gpu_power_{phase}_median_watts": summary["power_watts_median"],
        f"gpu_power_{phase}_max_watts": summary["power_watts_max"],
        f"gpu_sm_clock_{phase}_median_mhz": summary["sm_clock_mhz_median"],
    }


def _stablehlo_operation_counts(ir: str) -> dict[str, int]:
    """Count operation declarations in a StableHLO module for diagnostics."""
    operations = (
        "gather",
        "scatter",
        "broadcast_in_dim",
        "concatenate",
        "reshape",
        "while",
        "slice",
        "dynamic_update_slice",
    )
    return {operation: ir.count(f'"stablehlo.{operation}"') for operation in operations}


def _jaxpr_primitive_counts(jaxpr) -> dict[str, int]:
    """Count nested graph operations without expanding scan iteration counts."""
    counts = {}

    def visit(value):
        # Newer JAX exposes Jaxpr.jaxpr as a self-reference.
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                name = equation.primitive.name
                counts[name] = counts.get(name, 0) + 1
                visit(equation.params)
        elif hasattr(value, "jaxpr") and value.jaxpr is not value:
            visit(value.jaxpr)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(jaxpr)
    return counts


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _trial_succeeded(path: Path) -> bool:
    return path.exists() and _read_json(path).get("status") == "ok"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row if key != "steady_seconds"})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _default_output_dir(suite: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ARTIFACT_ROOT / f"{suite}_{stamp}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run an isolated benchmark suite.")
    run.add_argument(
        "--suite",
        choices=("pilot", "full", "large_cv", "backsub_ab", "mechanism_factorial", "hh_crossover", "rtrl_profile"),
        default="pilot",
    )
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--warmups", type=int, default=0, help="Extra warmups after the separately recorded first execution")
    run.add_argument("--gpu-monitor", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--compile-diagnostics", action="store_true")
    run.add_argument("--dhs-jvp-mode", choices=("generic", "explicit"), default="generic")
    run.add_argument("--rtrl-jvp-mode", choices=("linearize", "direct"), default="linearize")
    run.add_argument("--worker-timeout-seconds", type=float, default=1200.0, help="0 disables the worker timeout")
    run.add_argument("--budget-seconds", type=float, default=7200.0, help="0 disables the suite time limit")
    run.add_argument("--cv-values", type=int, nargs="+", help="Select existing suite CV counts")
    run.add_argument("--python", type=Path, help="Python executable used by isolated GPU workers.")
    run.add_argument("--backsub", choices=BACKSUBS, default="recursive")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True)
    worker.add_argument("--config-id")
    worker.add_argument("--mechanism")
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--repeats", type=int, required=True)
    worker.add_argument("--warmups", type=int, default=0)
    worker.add_argument("--gpu-monitor", action=argparse.BooleanOptionalAction, default=True)
    worker.add_argument("--compile-diagnostics", action="store_true")
    worker.add_argument("--dhs-jvp-mode", choices=("generic", "explicit"), default="generic")
    worker.add_argument("--rtrl-jvp-mode", choices=("linearize", "direct"), default="linearize")
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--physical-gpu", type=int)
    worker.add_argument("--backsub", choices=BACKSUBS, default="recursive")
    return parser


def main(argv=None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if args.command == "worker":
        config = BenchmarkConfig(**json.loads(args.config))
        mechanism = FULL_HH_SPEC if args.mechanism is None else MechanismSpec(**json.loads(args.mechanism))
        run_trial(
            config,
            args.method,
            repeats=args.repeats,
            output_path=args.output,
            physical_gpu=args.physical_gpu,
            backsub=args.backsub,
            mechanism=mechanism,
            config_id=args.config_id,
            warmups=args.warmups,
            gpu_monitor=args.gpu_monitor,
            compile_diagnostics=args.compile_diagnostics,
            dhs_jvp_mode=args.dhs_jvp_mode,
            rtrl_jvp_mode=args.rtrl_jvp_mode,
        )
        return
    output_dir = args.output_dir or _default_output_dir(args.suite)
    completed = run_suite(
        args.suite,
        output_dir=output_dir,
        gpu=args.gpu,
        repeats=args.repeats,
        resume=args.resume,
        dry_run=args.dry_run,
        python_executable=args.python,
        backsub=args.backsub,
        warmups=args.warmups,
        gpu_monitor=args.gpu_monitor,
        compile_diagnostics=args.compile_diagnostics,
        dhs_jvp_mode=args.dhs_jvp_mode,
        rtrl_jvp_mode=args.rtrl_jvp_mode,
        worker_timeout_seconds=args.worker_timeout_seconds,
        budget_seconds=args.budget_seconds,
        cv_values=None if args.cv_values is None else tuple(args.cv_values),
    )
    print(completed)


if __name__ == "__main__":
    main()
