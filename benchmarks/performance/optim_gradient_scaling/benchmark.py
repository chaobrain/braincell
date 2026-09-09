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

"""A100 scaling benchmark for reverse BPTT and block-exact full RTRL.

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
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
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
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "rtrl_bptt_scaling"


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
) -> dict[str, object]:
    """Compile, execute, and persist one isolated benchmark trial."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if backsub not in BACKSUBS:
        raise ValueError(f"backsub must be one of {BACKSUBS!r}.")
    os.environ["BRAINCELL_DHS_BACKSUB"] = backsub
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
    }
    try:
        with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
            prepared = prepare_benchmark(config, method, mechanism=mechanism)
            arguments = (prepared.seed_roots,)
            compile_monitor = _GpuPhaseMonitor(physical_gpu)
            compile_monitor.start()
            started = time.perf_counter()
            compiled = jax.jit(prepared.function).lower(*arguments).compile()
            compile_seconds = time.perf_counter() - started
            compile_metrics = compile_monitor.stop()

            first_monitor = _GpuPhaseMonitor(physical_gpu)
            first_monitor.start()
            started = time.perf_counter()
            first_output = compiled(*arguments)
            _block_until_ready(first_output)
            first_seconds = time.perf_counter() - started
            first_metrics = first_monitor.stop()

            steady_monitor = _GpuPhaseMonitor(physical_gpu)
            steady_monitor.start()
            steady = []
            for _ in range(repeats):
                started = time.perf_counter()
                output = compiled(*arguments)
                _block_until_ready(output)
                steady.append(time.perf_counter() - started)
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
                    "status": "ok",
                    "backend": jax.default_backend(),
                    "device": str(jax.devices()[0]),
                    "jax_version": jax.__version__,
                    "compile_seconds": compile_seconds,
                    "first_seconds": first_seconds,
                    "steady_seconds": steady,
                    "steady_median_seconds": float(np.median(steady)),
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
    except Exception as exc:
        result.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
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
) -> Path:
    """Launch isolated workers and aggregate their results."""
    cases = suite_cases(suite)
    if backsub not in BACKSUBS:
        raise ValueError(f"backsub must be one of {BACKSUBS!r}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = output_dir / "trials"
    log_dir = output_dir / "logs"
    trial_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    manifest = {
        "suite": suite,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "repeats": repeats,
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
    _write_json(output_dir / "manifest.json", manifest)
    commands = []
    worker_python = str(python_executable or sys.executable)
    for case in cases:
        config = case.config
        for method in METHODS:
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
                "--output",
                str(trial_path),
                "--physical-gpu",
                str(gpu),
                "--backsub",
                backsub,
            ]
            commands.append((case, method, trial_path, command))
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
        }
    )
    for index, (case, method, trial_path, command) in enumerate(commands, start=1):
        config = case.config
        print(f"[{index}/{len(commands)}] {case.id} {method}", flush=True)
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
        (log_dir / f"{case.id}__{method}.log").write_text(
            completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0 and not trial_path.exists():
            _write_json(
                trial_path,
                {
                    **asdict(config),
                    "config_id": case.id,
                    "mechanism_case": case.mechanism.name,
                    "method": method,
                    "status": "subprocess_error",
                    "returncode": completed.returncode,
                },
            )
        aggregate_results(output_dir)
    return output_dir


def aggregate_results(output_dir: Path) -> list[dict[str, object]]:
    """Combine trial JSON files and attach pairwise correctness metrics."""
    trial_dir = output_dir / "trials"
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
        comparison = {
            "gradient_max_abs_error": float(np.max(gradient_abs)),
            "gradient_max_rel_error": float(np.max(gradient_abs / scale)),
            "loss_max_abs_error": float(np.max(loss_abs)),
            "bptt_over_rtrl_time": (float(bptt["steady_median_seconds"]) / float(rtrl["steady_median_seconds"])),
        }
        bptt.update(comparison)
        rtrl.update(comparison)
    _write_csv(output_dir / "results.csv", rows)
    return rows


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


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _trial_succeeded(path: Path) -> bool:
    return path.exists() and _read_json(path).get("status") == "ok"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        choices=("pilot", "full", "large_cv", "backsub_ab", "mechanism_factorial"),
        default="pilot",
    )
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    run.add_argument("--repeats", type=int, default=10)
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
    )
    print(completed)


if __name__ == "__main__":
    main()
