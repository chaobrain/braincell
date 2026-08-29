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
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.filter import AllRegion, at
from examples.experimental.online_learning.rollout_gradients import build_rollout_value_and_grad

DT_MS = 0.025
RNG_SEED = 20260828
METHODS = ("bptt", "rtrl")
BACKSUBS = ("recursive", "ordinary")
BASELINE = {"n_cv": 5, "duration_ms": 40.0, "batch_size": 16, "n_seed": 16}
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "rtrl_bptt_scaling"

_BASE_G_MAX = {
    "leak": 0.1 * u.mS / u.cm**2,
    "na": 120.0 * u.mS / u.cm**2,
    "k": 10.0 * u.mS / u.cm**2,
}


@dataclass(frozen=True, order=True)
class BenchmarkConfig:
    """One static benchmark configuration."""

    n_cv: int
    duration_ms: float
    batch_size: int
    n_seed: int

    def __post_init__(self) -> None:
        if self.n_cv < 1 or self.n_cv % 2 == 0:
            raise ValueError("n_cv must be a positive odd integer.")
        if self.duration_ms <= 0.0:
            raise ValueError("duration_ms must be positive.")
        if self.batch_size < 1 or self.n_seed < 1:
            raise ValueError("batch_size and n_seed must be positive.")
        steps = self.duration_ms / DT_MS
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1e-10):
            raise ValueError(f"duration_ms must be an integer multiple of {DT_MS} ms.")

    @property
    def num_steps(self) -> int:
        return int(round(self.duration_ms / DT_MS))

    @property
    def id(self) -> str:
        duration = f"{self.duration_ms:g}".replace(".", "p")
        return f"c{self.n_cv}_t{duration}_b{self.batch_size}_s{self.n_seed}"


class PreparedBenchmark(NamedTuple):
    """Compiled-input metadata for one method/configuration."""

    function: object
    seed_roots: object
    state_scalar_count_per_seed: int
    parameter_count_per_seed: int
    rtrl_carry_bytes: int | None


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


def build_morphology(n_cv: int) -> braincell.Morphology:
    """Build a soma with two equally segmented dendritic arms."""
    if n_cv < 1 or n_cv % 2 == 0:
        raise ValueError("n_cv must be a positive odd integer.")
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    segments = (n_cv - 1) // 2
    arm_specs = (
        ("dend_a", "basal_dendrite", 80.0, 2.0, 1.2),
        ("dend_b", "apical_dendrite", 120.0, 2.5, 1.0),
    )
    for prefix, branch_type, total_length, proximal_radius, terminal_radius in arm_specs:
        parent = "soma"
        for segment in range(segments):
            lo = segment / segments
            hi = (segment + 1) / segments
            branch = braincell.Branch.from_lengths(
                lengths=[total_length / segments] * u.um,
                radii=[
                    proximal_radius + lo * (terminal_radius - proximal_radius),
                    proximal_radius + hi * (terminal_radius - proximal_radius),
                ]
                * u.um,
                type=branch_type,
            )
            name = f"{prefix}_{segment}"
            morphology.attach(parent=parent, child_branch=branch, child_name=name, parent_x=1.0)
            parent = name
    return morphology


def target_row_scales(n_cv: int) -> dict[str, np.ndarray]:
    """Return smooth deterministic per-CV target conductance scales."""
    position = np.linspace(0.0, 1.0, n_cv, dtype=np.float64)
    return {
        "leak": 1.05 + 0.15 * np.cos(np.pi * position),
        "na": 0.95 + 0.20 * np.sin(np.pi * (position + 0.15)),
        "k": 1.10 - 0.15 * np.cos(np.pi * position),
    }


def current_amplitudes(batch_size: int) -> object:
    """Return one DC-current protocol per batch member."""
    return u.Quantity(np.linspace(0.03, 0.08, batch_size, dtype=np.float64), u.nA)


def build_cell(config: BenchmarkConfig, *, trainable: bool) -> braincell.Cell:
    """Build one batch-population Cell whose parameters are shared over batch."""
    scales = target_row_scales(config.n_cv)
    cell = braincell.Cell(
        build_morphology(config.n_cv),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(config.batch_size,),
        V_init=-65.0 * u.mV,
        solver="staggered",
    )
    channel_scales = {
        name: 1.0 if trainable else float(scales[name][0]) if config.n_cv == 1 else scales[name] for name in _BASE_G_MAX
    }
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("IL", name="leak", g_max=channel_scales["leak"] * _BASE_G_MAX["leak"]),
        braincell.mech.Channel("Na_HH1952", name="na", g_max=channel_scales["na"] * _BASE_G_MAX["na"]),
        braincell.mech.Channel("K_HH1952", name="k", g_max=channel_scales["k"] * _BASE_G_MAX["k"]),
    )
    cell.place(
        at("soma", 0.5),
        braincell.mech.CurrentClamp(
            delay=0.0 * u.ms,
            durations=(config.duration_ms + DT_MS) * u.ms,
            amplitudes=current_amplitudes(config.batch_size),
        ),
    )
    if trainable:
        for name in ("leak", "na", "k"):
            cell.channels[name].trainable(g_max=braincell.trainable.scale(group_by="cv", name=f"{name}.scale"))
    cell.init_state()
    if cell.n_cv != config.n_cv:
        raise RuntimeError(f"Requested {config.n_cv} CVs but built {cell.n_cv}.")
    return cell


def simulate_voltage(cell: braincell.Cell, times_ms) -> object:
    """Return pre-step voltage samples with shape ``(time, batch, cv)``."""
    cell.reset_state()

    def step(time_ms):
        voltage = cell.V.value.to_decimal(u.mV)
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return voltage

    return brainstate.transform.for_loop(step, times_ms)


def seed_parameter_roots(parameter_states, *, n_seed: int) -> tuple[object, ...]:
    """Return deterministic seed-leading optimizer roots."""
    random = brainstate.random.RandomState(RNG_SEED)
    roots = []
    for state in parameter_states.values():
        shape = tuple(state.value.shape)
        roots.append(random.uniform(0.75, 1.25, size=(n_seed,) + shape))
    return tuple(roots)


def prepare_benchmark(config: BenchmarkConfig, method: str) -> PreparedBenchmark:
    """Build one seed-blocked gradient kernel and its static inputs."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}.")
    times_ms = jnp.arange(config.num_steps, dtype=jnp.float64) * DT_MS
    measured_backsub = os.environ.get("BRAINCELL_DHS_BACKSUB", "recursive")
    os.environ["BRAINCELL_DHS_BACKSUB"] = "recursive"
    try:
        target_cell = build_cell(config, trainable=False)
        target_voltage = simulate_voltage(target_cell, times_ms)
    finally:
        os.environ["BRAINCELL_DHS_BACKSUB"] = measured_backsub
    candidate = build_cell(config, trainable=True)

    def rollout_step(data):
        time_ms, target_mv = data
        voltage = candidate.V.value.to_decimal(u.mV)
        local_loss = jnp.mean((voltage - target_mv) ** 2) / config.num_steps
        with brainstate.environ.context(t=time_ms * u.ms):
            candidate.update()
        return local_loss

    engine = build_rollout_value_and_grad(candidate, step=rollout_step, method=method)
    engine.prepare((times_ms[0], target_voltage[0]))
    parameter_states = engine.parameter_states
    seed_roots = seed_parameter_roots(parameter_states, n_seed=config.n_seed)
    names = engine.parameter_names

    if method == "bptt":
        one_seed = engine._bptt
        carry_bytes = None
    else:
        one_seed = engine._rtrl
        one_roots = tuple(root[0] for root in seed_roots)
        _values, one_tangents = engine._initial_full_carry(one_roots)
        carry_bytes = config.n_seed * _tree_nbytes(one_tangents)

    def seed_step(roots):
        result = one_seed(roots, (times_ms, target_voltage))
        gradient = jnp.concatenate([jnp.ravel(result.gradients[name]) for name in names])
        return result.loss, result.losses, gradient

    function = jax.vmap(seed_step)
    initial_values = engine._initial_primal_values(tuple(root[0] for root in seed_roots))
    state_count = sum(
        int(np.prod(np.shape(leaf), dtype=np.int64)) if np.shape(leaf) else 1
        for leaf in jax.tree.leaves(initial_values)
    )
    parameter_count = sum(int(np.prod(root.shape[1:], dtype=np.int64)) for root in seed_roots)
    return PreparedBenchmark(
        function=function,
        seed_roots=seed_roots,
        state_scalar_count_per_seed=state_count,
        parameter_count_per_seed=parameter_count,
        rtrl_carry_bytes=carry_bytes,
    )


def run_trial(
    config: BenchmarkConfig,
    method: str,
    *,
    repeats: int,
    output_path: Path,
    physical_gpu: int | None,
    backsub: str = "recursive",
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
        "config_id": config.id,
        "method": method,
        "backsub": backsub,
        "num_steps": config.num_steps,
        "repeats": repeats,
        "status": "running",
    }
    try:
        with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
            prepared = prepare_benchmark(config, method)
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
                    "active_state_estimate_per_seed": 4 * config.batch_size * config.n_cv,
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
    configs = suite_configs(suite)
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
        "configs": [asdict(config) | {"config_id": config.id} for config in configs],
    }
    _write_json(output_dir / "manifest.json", manifest)
    commands = []
    worker_python = str(python_executable or sys.executable)
    for config in configs:
        for method in METHODS:
            backsub_suffix = "" if backsub == "recursive" else f"__{backsub}"
            trial_path = trial_dir / f"{config.id}__{method}{backsub_suffix}.json"
            if resume and _trial_succeeded(trial_path):
                continue
            command = [
                worker_python,
                str(Path(__file__).resolve()),
                "worker",
                "--config",
                json.dumps(asdict(config)),
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
            commands.append((config, method, trial_path, command))
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
    for index, (config, method, trial_path, command) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {config.id} {method}", flush=True)
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
        (log_dir / f"{config.id}__{method}.log").write_text(
            completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0 and not trial_path.exists():
            _write_json(
                trial_path,
                {
                    **asdict(config),
                    "config_id": config.id,
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


def _tree_nbytes(tree) -> int:
    return sum(
        int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(tree)
    )


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
    run.add_argument("--suite", choices=("pilot", "full", "large_cv", "backsub_ab"), default="pilot")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--gpu", type=int, default=7)
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--python", type=Path, help="Python executable used by isolated GPU workers.")
    run.add_argument("--backsub", choices=BACKSUBS, default="recursive")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--repeats", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--physical-gpu", type=int)
    worker.add_argument("--backsub", choices=BACKSUBS, default="recursive")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        config = BenchmarkConfig(**json.loads(args.config))
        run_trial(
            config,
            args.method,
            repeats=args.repeats,
            output_path=args.output,
            physical_gpu=args.physical_gpu,
            backsub=args.backsub,
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
