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

"""Benchmark BrainCell population, state-vmap, and spike bookkeeping costs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import platform
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import brainstate
import brainunit as u
import jax
import numpy as np

import braincell
from braincell._parameter_schema import RuntimeParameterState

from backend_braincell import build_cell
from common import (
    DT_MS,
    N_STEPS,
    PROBE_BRANCHES,
    TSTOP_MS,
    add_throughput,
    assert_morphology_asset,
    git_commit,
    query_gpus,
    read_json,
    timing_summary,
    write_json,
)
from run_benchmark import parse_csv_ints

HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/swl/anaconda3/envs/braincell_311/bin/python")
BATCH_MODES = ("population", "vmap")
SPIKE_MODES = ("tracked", "off")
AMPLITUDE_NA = 0.70
TRACE_RTOL = 1e-5
TRACE_ATOL_MV = 2e-4


@dataclass(frozen=True)
class PreparedCase:
    """Hold one configured ablation case and its callable boundaries."""

    cell: object
    simulate: Callable[[], tuple]
    restore: Callable[[], None]
    normalize_host_traces: Callable[[tuple], np.ndarray]
    state_inventory: tuple[dict[str, object], ...]
    parameter_inventory: tuple[dict[str, object], ...]
    morphology: dict[str, object]


def _state_path(path: object) -> str:
    if isinstance(path, tuple):
        return "/".join(str(part) for part in path)
    return str(path)


def _stored_value(value: object) -> object:
    return u.get_mantissa(value) if isinstance(value, u.Quantity) else value


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(size) for size in getattr(_stored_value(value), "shape", ()))


def _inventory(cell: object) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    dynamic_rows = []
    parameter_rows = []
    for path, state in brainstate.graph.states(cell).items():
        row = {
            "path": _state_path(path),
            "state_type": type(state).__name__,
            "stored_shape": list(_shape(state.value)),
            "dtype": str(getattr(_stored_value(state.value), "dtype", type(_stored_value(state.value)).__name__)),
        }
        if isinstance(state, RuntimeParameterState):
            row.update(
                {
                    "axis": str(state.axis),
                    "logical_shape": list(state.shape),
                }
            )
            parameter_rows.append(row)
        else:
            dynamic_rows.append(row)
    return tuple(dynamic_rows), tuple(parameter_rows)


def _update_dynamics_without_spikes(cell) -> None:
    if brainstate.environ.get("dt", None) is None:
        raise ValueError("Cell update requires brainstate.environ['dt'] to be set.")
    with jax.named_scope("braincell:cell_update:solver"):
        cell.solver(cell)
    with jax.named_scope("braincell:cell_update:clear_ion_total_current_cache"):
        cell.clear_ion_total_current_cache()


def _disable_spike_bookkeeping(cell: object) -> None:
    if cell.network_owner is not None or len(cell.connections) != 0:
        raise ValueError("spike-off ablation only supports an isolated Cell without connections")
    if any(layout.kind.startswith("synapse:") for layout in cell.layouts):
        raise ValueError("spike-off ablation does not support runtime synapses")
    del cell.spike
    del cell._event_previous_V
    cell._update_dynamics = types.MethodType(_update_dynamics_without_spikes, cell)


def _mapped_states(cell: object) -> dict[object, object]:
    return {
        path: state
        for path, state in brainstate.graph.states(cell).items()
        if not isinstance(state, RuntimeParameterState)
    }


def _expand_mapped_states(states: dict[object, object], batch_size: int) -> None:
    for state in states.values():
        value = state.value
        state.value = u.math.broadcast_to(value, (batch_size,) + _shape(value))


def _snapshot(states: dict[object, object]) -> tuple[tuple[object, object], ...]:
    return tuple((state, state.value) for state in states.values())


def _restore(snapshot: tuple[tuple[object, object], ...]) -> None:
    for state, value in snapshot:
        state.value = value


def prepare_case(
    batch_size: int,
    *,
    batch_mode: str,
    spike_mode: str,
    duration_ms: float = TSTOP_MS,
) -> PreparedCase:
    """Build one population/vmap and tracked/off ablation case."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_mode not in BATCH_MODES:
        raise ValueError(f"unknown batch mode: {batch_mode!r}")
    if spike_mode not in SPIKE_MODES:
        raise ValueError(f"unknown spike mode: {spike_mode!r}")
    if duration_ms <= 0.0:
        raise ValueError("duration_ms must be positive")

    cell_batch_size = batch_size if batch_mode == "population" else 1
    amplitudes = np.full((cell_batch_size,), AMPLITUDE_NA, dtype=np.float32)
    cell, morphology = build_cell(cell_batch_size, amplitudes_na=amplitudes)
    if spike_mode == "off":
        _disable_spike_bookkeeping(cell)

    states = _mapped_states(cell)
    if batch_mode == "vmap":
        _expand_mapped_states(states, batch_size)
    initial = _snapshot(states)
    probe_names = tuple(f"v_b{branch}" for branch in PROBE_BRANCHES)

    if batch_mode == "population":

        def simulate() -> tuple:
            result = cell.run(dt=DT_MS * u.ms, duration=duration_ms * u.ms)
            return tuple(result.traces[name] for name in probe_names)

        def normalize_host_traces(host_traces: tuple) -> np.ndarray:
            return np.stack(
                [np.asarray(trace.to_decimal(u.mV)).T for trace in host_traces],
                axis=1,
            )

    else:

        @brainstate.transform.vmap(
            in_axes=(),
            out_axes=0,
            axis_size=batch_size,
            in_states={0: states},
            out_states={0: states},
        )
        def simulate() -> tuple:
            result = cell.run(dt=DT_MS * u.ms, duration=duration_ms * u.ms)
            return tuple(result.traces[name] for name in probe_names)

        def normalize_host_traces(host_traces: tuple) -> np.ndarray:
            return np.stack(
                [np.asarray(trace.to_decimal(u.mV))[..., 0] for trace in host_traces],
                axis=1,
            )

    state_inventory, parameter_inventory = _inventory(cell)
    return PreparedCase(
        cell=cell,
        simulate=simulate,
        restore=lambda: _restore(initial),
        normalize_host_traces=normalize_host_traces,
        state_inventory=state_inventory,
        parameter_inventory=parameter_inventory,
        morphology=morphology,
    )


def _block(value: object) -> None:
    jax.block_until_ready(value)


def _memory_metadata() -> dict[str, object]:
    stats = jax.devices()[0].memory_stats() or {}
    peak_bytes = int(stats.get("peak_bytes_in_use", 0))
    return {
        "peak_bytes_in_use": peak_bytes,
        "peak_mib_in_use": peak_bytes / (1024**2),
        "bytes_limit": int(stats.get("bytes_limit", 0)),
    }


def run_case(
    batch_size: int,
    *,
    batch_mode: str,
    spike_mode: str,
    warmup: int,
    repeat: int,
    transfer_repeat: int,
) -> dict[str, object]:
    """Compile and time one isolated BrainCell ablation case."""
    assert_morphology_asset()
    build_start = time.perf_counter()
    prepared = prepare_case(batch_size, batch_mode=batch_mode, spike_mode=spike_mode)
    build_seconds = time.perf_counter() - build_start

    def simulate():
        prepared.restore()
        return prepared.simulate()

    compile_start = time.perf_counter()
    result = simulate()
    _block(result)
    compilation_seconds = time.perf_counter() - compile_start
    for _ in range(warmup):
        result = simulate()
        _block(result)

    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = simulate()
        _block(result)
        samples.append(time.perf_counter() - started)

    transfer_samples = []
    host_result = None
    for _ in range(transfer_repeat):
        result = simulate()
        _block(result)
        started = time.perf_counter()
        host_result = jax.device_get(result)
        transfer_samples.append(time.perf_counter() - started)
    if host_result is None:
        host_result = jax.device_get(result)

    traces = prepared.normalize_host_traces(host_result)
    expected_shape = (batch_size, len(PROBE_BRANCHES), N_STEPS)
    if traces.shape != expected_shape:
        raise RuntimeError(f"unexpected trace shape: expected {expected_shape}, got {traces.shape}")
    if not np.isfinite(traces).all():
        raise RuntimeError("BrainCell ablation produced non-finite values")

    timing = add_throughput(timing_summary(samples), batch_size=batch_size)
    timing["relative_iqr"] = (float(timing["q3_seconds"]) - float(timing["q1_seconds"])) / float(
        timing["median_seconds"]
    )
    return {
        "schema_version": 1,
        "backend": "braincell",
        "batch_size": batch_size,
        "case": {"batch_mode": batch_mode, "spike_mode": spike_mode},
        "config": {
            "amplitude_na": AMPLITUDE_NA,
            "dt_ms": DT_MS,
            "tstop_ms": TSTOP_MS,
            "n_steps": N_STEPS,
            "warmup": warmup,
            "repeat": repeat,
            "transfer_repeat": transfer_repeat,
        },
        "timing": timing,
        "host_transfer": timing_summary(transfer_samples) if transfer_samples else None,
        "build_seconds": build_seconds,
        "compilation_seconds": compilation_seconds,
        "device_memory": _memory_metadata(),
        "device": str(jax.devices()[0]),
        "morphology": prepared.morphology,
        "state_inventory": prepared.state_inventory,
        "parameter_inventory": prepared.parameter_inventory,
        "software": {
            "python": platform.python_version(),
            "braincell": braincell.__version__,
            "braincell_git_commit": git_commit(Path(braincell.__file__).resolve().parents[1]),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
        },
        "process": {"pid": os.getpid()},
        "output_validation": {
            "all_finite": True,
            "shape": list(traces.shape),
            "trace_output_bytes": int(traces.nbytes),
            "first_lane_trace_mv": traces[0].tolist(),
        },
    }


def _compute_pids_from_monitor(path: Path) -> tuple[int, ...]:
    pids = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0].startswith("#") or fields[2] != "C":
            continue
        try:
            pid = int(fields[1])
            sm_utilization = int(fields[3])
        except ValueError:
            continue
        if sm_utilization > 0:
            pids.add(pid)
    return tuple(sorted(pids))


def _run_monitored_child(
    command: list[str],
    *,
    gpu: int,
    monitor_path: Path,
) -> tuple[int, dict[str, object]]:
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    with monitor_path.open("w") as monitor_handle:
        monitor = subprocess.Popen(
            ["nvidia-smi", "pmon", "-i", str(gpu), "-s", "um", "-d", "1", "-c", "600"],
            stdout=monitor_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            child = subprocess.Popen(command, cwd=HERE, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            child_pid = child.pid
            return_code = child.wait()
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    compute_pids = _compute_pids_from_monitor(monitor_path)
    unexpected = tuple(pid for pid in compute_pids if pid != child_pid)
    return return_code, {
        "benchmark_pid": child_pid,
        "observed_compute_pids": list(compute_pids),
        "unexpected_compute_pids": list(unexpected),
        "passed": len(unexpected) == 0,
    }


def _gpu_load(gpu: int) -> tuple[float, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    memory_used, utilization = (float(field.strip()) for field in output.strip().split(","))
    return memory_used, utilization


def _wait_for_gpu_release(
    gpu: int,
    *,
    timeout_seconds: float = 30.0,
    max_memory_used_mib: float = 10240.0,
    max_utilization_percent: float = 20.0,
    stable_samples: int = 3,
) -> None:
    if stable_samples <= 0:
        raise ValueError("stable_samples must be positive")
    deadline = time.monotonic() + timeout_seconds
    consecutive_idle = 0
    while True:
        memory_used, utilization = _gpu_load(gpu)
        if memory_used <= max_memory_used_mib and utilization <= max_utilization_percent:
            consecutive_idle += 1
            if consecutive_idle >= stable_samples:
                return
        else:
            consecutive_idle = 0
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU {gpu} did not release after a benchmark child: "
                f"{memory_used:.0f} MiB used, {utilization:.0f}% utilization"
            )
        time.sleep(1.0)


def _case_name(batch_mode: str, spike_mode: str, batch_size: int) -> str:
    return f"{batch_mode}_{spike_mode}_n{batch_size}"


def _run_key(run: dict[str, object]) -> tuple[int, str, str]:
    return (
        int(run["batch_size"]),
        str(run["case"]["batch_mode"]),
        str(run["case"]["spike_mode"]),
    )


def _validate_traces(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = []
    by_size = {}
    for run in runs:
        by_size.setdefault(int(run["batch_size"]), {})[(run["case"]["batch_mode"], run["case"]["spike_mode"])] = (
            np.asarray(run["output_validation"]["first_lane_trace_mv"], dtype=np.float64)
        )
    for batch_size, traces in sorted(by_size.items()):
        reference = traces[("population", "tracked")]
        reference_crossings = ((reference[:, :-1] < 0.0) & (reference[:, 1:] >= 0.0)).sum(axis=1)
        for case, trace in sorted(traces.items()):
            delta = np.abs(trace - reference)
            crossings = ((trace[:, :-1] < 0.0) & (trace[:, 1:] >= 0.0)).sum(axis=1)
            crossings_match = bool(np.array_equal(crossings, reference_crossings))
            comparisons.append(
                {
                    "batch_size": batch_size,
                    "batch_mode": case[0],
                    "spike_mode": case[1],
                    "max_abs_delta_mv": float(delta.max(initial=0.0)),
                    "rmse_mv": float(np.sqrt(np.mean(np.square(trace - reference)))),
                    "spike_crossings": crossings.tolist(),
                    "reference_spike_crossings": reference_crossings.tolist(),
                    "spike_crossings_match": crossings_match,
                    "passed": bool(
                        crossings_match
                        and np.allclose(
                            trace,
                            reference,
                            rtol=TRACE_RTOL,
                            atol=TRACE_ATOL_MV,
                        )
                    ),
                }
            )
    return comparisons


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    """Build pairwise speedup rows and voltage validation for a suite."""
    indexed = {_run_key(run): run for run in runs}
    effects = []
    for batch_size in sorted({key[0] for key in indexed}):
        for batch_mode in BATCH_MODES:
            tracked = indexed[(batch_size, batch_mode, "tracked")]
            off = indexed[(batch_size, batch_mode, "off")]
            effects.append(
                {
                    "batch_size": batch_size,
                    "effect": "spike_off",
                    "batch_mode": batch_mode,
                    "speedup": tracked["timing"]["median_seconds"] / off["timing"]["median_seconds"],
                    "peak_memory_ratio": tracked["device_memory"]["peak_mib_in_use"]
                    / off["device_memory"]["peak_mib_in_use"],
                }
            )
        for spike_mode in SPIKE_MODES:
            population = indexed[(batch_size, "population", spike_mode)]
            vmapped = indexed[(batch_size, "vmap", spike_mode)]
            effects.append(
                {
                    "batch_size": batch_size,
                    "effect": "vmap_over_population",
                    "spike_mode": spike_mode,
                    "speedup": population["timing"]["median_seconds"] / vmapped["timing"]["median_seconds"],
                    "peak_memory_ratio": population["device_memory"]["peak_mib_in_use"]
                    / vmapped["device_memory"]["peak_mib_in_use"],
                }
            )
    trace_validation = _validate_traces(runs)
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": sorted(
            runs,
            key=lambda run: (
                int(run["batch_size"]),
                str(run["case"]["batch_mode"]),
                str(run["case"]["spike_mode"]),
            ),
        ),
        "effects": effects,
        "trace_validation": trace_validation,
        "trace_tolerance": {
            "rtol": TRACE_RTOL,
            "atol_mv": TRACE_ATOL_MV,
            "spike_threshold_mv": 0.0,
        },
        "passed": all(row["passed"] for row in trace_validation),
    }


def _write_csv(path: Path, payload: dict[str, object]) -> None:
    rows = []
    for run in payload["runs"]:
        rows.append(
            {
                "batch_size": run["batch_size"],
                "batch_mode": run["case"]["batch_mode"],
                "spike_mode": run["case"]["spike_mode"],
                "median_seconds": run["timing"]["median_seconds"],
                "q1_seconds": run["timing"]["q1_seconds"],
                "q3_seconds": run["timing"]["q3_seconds"],
                "relative_iqr": run["timing"]["relative_iqr"],
                "cell_steps_per_second": run["timing"]["cell_steps_per_second"],
                "peak_mib_in_use": run["device_memory"]["peak_mib_in_use"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(args) -> dict[str, object]:
    args.output = args.output.resolve()
    raw_dir = args.output.parent / "raw"
    monitor_dir = args.output.parent / "monitor"
    progress_path = args.output.parent / "progress.json"
    progress_config = {
        "batch_sizes": list(args.batch_sizes),
        "amplitude_na": AMPLITUDE_NA,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "transfer_repeat": args.transfer_repeat,
        "gpu": args.gpu,
        "braincell_git_commit": git_commit(Path(braincell.__file__).resolve().parents[1]),
    }
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress.get("config") != progress_config:
            raise RuntimeError(f"existing progress configuration does not match this run: {progress_path}")
        runs = list(progress.get("runs", []))
        print(f"Resuming {len(runs)} completed cases from {progress_path}", flush=True)
    else:
        runs = []
    completed = {_run_key(run) for run in runs}
    for batch_size in args.batch_sizes:
        for batch_mode in BATCH_MODES:
            for spike_mode in SPIKE_MODES:
                name = _case_name(batch_mode, spike_mode, batch_size)
                key = (batch_size, batch_mode, spike_mode)
                if key in completed:
                    print(f"[{name}] already complete", flush=True)
                    continue
                attempts = []
                for attempt in range(2):
                    _wait_for_gpu_release(args.gpu, timeout_seconds=args.idle_timeout)
                    selection = query_gpus((args.gpu,))
                    raw_path = raw_dir / f"{name}_attempt{attempt + 1}.json"
                    command = [
                        str(args.python),
                        str(Path(__file__).resolve()),
                        "case",
                        "--batch-size",
                        str(batch_size),
                        "--batch-mode",
                        batch_mode,
                        "--spike-mode",
                        spike_mode,
                        "--warmup",
                        str(args.warmup),
                        "--repeat",
                        str(args.repeat),
                        "--transfer-repeat",
                        str(args.transfer_repeat),
                        "--output",
                        str(raw_path),
                    ]
                    print(f"[{name}] attempt={attempt + 1} GPU={args.gpu}", flush=True)
                    return_code, process_monitor = _run_monitored_child(
                        command,
                        gpu=args.gpu,
                        monitor_path=monitor_dir / f"{name}_attempt{attempt + 1}.log",
                    )
                    _wait_for_gpu_release(args.gpu, timeout_seconds=args.idle_timeout)
                    if return_code != 0:
                        raise RuntimeError(f"{name} failed with exit code {return_code}")
                    run = read_json(raw_path)
                    run["gpu_selection"] = selection
                    run["process_monitor"] = process_monitor
                    attempts.append(run)
                    stable = float(run["timing"]["relative_iqr"]) <= 0.10
                    if process_monitor["passed"] and stable:
                        break
                    print(
                        f"[{name}] rejecting attempt {attempt + 1}: "
                        f"unexpected_pids={process_monitor['unexpected_compute_pids']} "
                        f"relative_iqr={run['timing']['relative_iqr']:.4f}",
                        flush=True,
                    )
                selected = attempts[-1]
                selected["attempt_count"] = len(attempts)
                selected["stable"] = float(selected["timing"]["relative_iqr"]) <= 0.10
                selected["valid"] = bool(selected["stable"] and selected["process_monitor"]["passed"])
                if not selected["valid"]:
                    raise RuntimeError(f"{name} did not produce a valid result after two attempts")
                runs.append(selected)
                completed.add(key)
                write_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "config": progress_config,
                        "runs": runs,
                    },
                )
    payload = summarize_runs(runs)
    payload["gpu"] = {"physical_id": args.gpu}
    payload["config"] = {
        "batch_sizes": list(args.batch_sizes),
        "amplitude_na": AMPLITUDE_NA,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "transfer_repeat": args.transfer_repeat,
    }
    write_json(args.output, payload)
    _write_csv(args.output.with_suffix(".csv"), payload)
    if not payload["passed"]:
        raise RuntimeError("voltage trace validation failed")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    case = subparsers.add_parser("case")
    case.add_argument("--batch-size", type=int, required=True)
    case.add_argument("--batch-mode", choices=BATCH_MODES, required=True)
    case.add_argument("--spike-mode", choices=SPIKE_MODES, required=True)
    case.add_argument("--warmup", type=int, default=2)
    case.add_argument("--repeat", type=int, default=7)
    case.add_argument("--transfer-repeat", type=int, default=3)
    case.add_argument("--output", type=Path, required=True)

    suite = subparsers.add_parser("run")
    suite.add_argument("--batch-sizes", type=parse_csv_ints, default=(10, 100, 1000))
    suite.add_argument("--gpu", type=int, default=7)
    suite.add_argument("--warmup", type=int, default=2)
    suite.add_argument("--repeat", type=int, default=7)
    suite.add_argument("--transfer-repeat", type=int, default=3)
    suite.add_argument("--idle-timeout", type=float, default=900.0)
    suite.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    suite.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.transfer_repeat < 0:
        parser.error("repeat must be positive and warmup/transfer-repeat non-negative")
    if args.command == "case":
        if args.batch_size <= 0:
            parser.error("batch-size must be positive")
        write_json(
            args.output,
            run_case(
                args.batch_size,
                batch_mode=args.batch_mode,
                spike_mode=args.spike_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                transfer_repeat=args.transfer_repeat,
            ),
        )
    else:
        if not args.python.exists():
            parser.error(f"Python interpreter does not exist: {args.python}")
        if args.idle_timeout <= 0.0:
            parser.error("idle-timeout must be positive")
        run_suite(args)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
