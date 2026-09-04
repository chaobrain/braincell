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

"""Controlled synthetic and full-HH BPTT/RTRL complexity benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
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

from examples.experimental.optim.gradients import build_rollout_value_and_grad
from examples.experimental.optim_gradient_scaling.benchmark import (
    BACKSUBS,
    DT_MS,
    FULL_HH_SPEC,
    METHODS,
    RNG_SEED,
    _GpuPhaseMonitor,
    _block_until_ready,
    _phase_metric_fields,
    _read_json,
    _trial_succeeded,
    _tree_nbytes,
    _write_csv,
    _write_json,
    BenchmarkConfig,
    build_cell,
    seed_parameter_roots,
    simulate_voltage,
)

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "rtrl_bptt_scaling"
DEFAULT_STEPS = 1600
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEEDS = 16
SYNTHETIC_STATE_SIZES = (32, 64, 128, 256, 512, 1024)
SYNTHETIC_PARAMETER_SIZES = (1, 2, 4, 8, 16, 32, 64)
HH_CV_SIZES = (3, 5, 9, 17, 33, 65)
HH_GROUPS = ("all", "population", "cv", "row")
HH_GROUP_PARAMETER_COUNTS = {
    "all": 1,
    "population": DEFAULT_BATCH_SIZE,
    "cv": 33,
    "row": DEFAULT_BATCH_SIZE * 33,
}


@dataclass(frozen=True, order=True)
class ControlledCase:
    """One logical controlled-complexity configuration."""

    workload: str
    n_x: int
    n_theta: int
    n_cv: int | None = None
    parameter_group: str | None = None
    num_steps: int = DEFAULT_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    n_seed: int = DEFAULT_SEEDS

    def __post_init__(self) -> None:
        if self.workload not in {"synthetic", "hh_state", "hh_parameter"}:
            raise ValueError("Unknown controlled workload.")
        if min(self.n_x, self.n_theta, self.num_steps, self.batch_size, self.n_seed) < 1:
            raise ValueError("Controlled dimensions must be positive.")
        if self.workload == "synthetic":
            if self.n_theta > self.n_x or self.n_cv is not None or self.parameter_group is not None:
                raise ValueError("Synthetic cases require N_theta <= N_x and no HH metadata.")
        elif self.n_cv is None or self.n_cv < 1 or self.n_cv % 2 == 0:
            raise ValueError("HH cases require a positive odd CV count.")
        elif self.n_x != 4 * self.n_cv:
            raise ValueError("Full-HH cases require N_x = 4C.")
        elif self.workload == "hh_state" and (self.n_theta != 3 or self.parameter_group != "all"):
            raise ValueError("HH state cases require three globally shared parameters.")
        elif self.workload == "hh_parameter" and self.parameter_group not in HH_GROUPS:
            raise ValueError("HH parameter cases require a supported grouping.")

    @property
    def id(self) -> str:
        if self.workload == "synthetic":
            return f"synthetic_nx{self.n_x}_ntheta{self.n_theta}"
        if self.workload == "hh_state":
            return f"hh_state_c{self.n_cv}_global3"
        return f"hh_parameter_c{self.n_cv}_leak_{self.parameter_group}"


class PreparedControlled(NamedTuple):
    """Compiled inputs and dimension metadata for one controlled case."""

    function: object
    roots: object
    state_scalar_count_per_seed: int
    parameter_count_per_seed: int
    rtrl_carry_bytes: int | None


def suite_cases(name: str) -> tuple[ControlledCase, ...]:
    """Return deterministic logical cases for a controlled suite."""
    synthetic = {ControlledCase("synthetic", n_x, 8) for n_x in SYNTHETIC_STATE_SIZES} | {
        ControlledCase("synthetic", 512, n_theta) for n_theta in SYNTHETIC_PARAMETER_SIZES
    }
    hh_state = {ControlledCase("hh_state", 4 * n_cv, 3, n_cv=n_cv, parameter_group="all") for n_cv in HH_CV_SIZES}
    hh_parameter = {
        ControlledCase(
            "hh_parameter",
            4 * 33,
            HH_GROUP_PARAMETER_COUNTS[group],
            n_cv=33,
            parameter_group=group,
        )
        for group in HH_GROUPS
    }
    by_suite = {
        "synthetic": synthetic,
        "hh_state": hh_state,
        "hh_parameter": hh_parameter,
        "all": synthetic | hh_state | hh_parameter,
    }
    if name not in by_suite:
        raise ValueError(f"Unknown suite {name!r}.")
    return tuple(sorted(by_suite[name]))


def balanced_cases(cases: tuple[ControlledCase, ...], replicate: int) -> tuple[ControlledCase, ...]:
    """Return a deterministic replicate-major order that balances time drift."""
    if replicate < 1:
        raise ValueError("replicate must be positive.")
    if replicate % 3 == 1:
        return cases
    if replicate % 3 == 2:
        return tuple(reversed(cases))
    interleaved = cases[1::2] + cases[::2]
    if not interleaved:
        return cases
    rotation = (replicate // 3 - 1) % len(interleaved)
    return interleaved[rotation:] + interleaved[:rotation]


def _synthetic_transition(state, theta, drive, *, num_steps: int):
    parameter_field = jnp.pad(theta, (0, state.shape[-1] - theta.shape[0]))
    mixed = 0.6 * state + 0.2 * jnp.roll(state, -1, axis=-1) + 0.2 * jnp.roll(state, 1, axis=-1)
    next_state = jnp.tanh(mixed + 0.02 * parameter_field[None, :] + drive)
    local_loss = jnp.mean(next_state * next_state) / num_steps
    return next_state, local_loss


def prepare_synthetic(case: ControlledCase, method: str) -> PreparedControlled:
    """Build one independently controlled synthetic gradient kernel."""
    if case.workload != "synthetic":
        raise ValueError("prepare_synthetic requires a synthetic case.")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}.")
    random = brainstate.random.RandomState(RNG_SEED)
    roots = random.uniform(-0.2, 0.2, size=(case.n_seed, case.n_theta), dtype=jnp.float64)
    phase = jnp.linspace(0.0, 4.0 * jnp.pi, case.num_steps, dtype=jnp.float64)
    batch_offsets = jnp.linspace(-0.02, 0.02, case.batch_size, dtype=jnp.float64)
    drives = 0.01 * jnp.sin(phase)[:, None, None] + batch_offsets[None, :, None]
    initial_state = jnp.zeros((case.batch_size, case.n_x), dtype=jnp.float64)

    if method == "bptt":

        def one_seed(theta):
            def objective(parameter):
                def scan_step(state, drive):
                    next_state, local_loss = _synthetic_transition(state, parameter, drive, num_steps=case.num_steps)
                    return next_state, local_loss

                _, losses = jax.lax.scan(scan_step, initial_state, drives)
                return jnp.sum(losses), losses

            (loss, losses), gradient = jax.value_and_grad(objective, has_aux=True)(theta)
            return loss, losses, gradient

        carry_bytes = None
    else:
        parameter_basis = jnp.eye(case.n_theta, dtype=jnp.float64)

        def one_seed(theta):
            sensitivities = jnp.zeros((case.n_theta, case.batch_size, case.n_x), dtype=jnp.float64)
            gradient = jnp.zeros((case.n_theta,), dtype=jnp.float64)

            def scan_step(carry, drive):
                state, state_sensitivities, accumulated_gradient = carry

                def transition(state_value, parameter_value):
                    return _synthetic_transition(state_value, parameter_value, drive, num_steps=case.num_steps)

                (next_state, local_loss), linear_map = jax.linearize(transition, state, theta)
                next_sensitivities, local_gradient = jax.vmap(linear_map)(state_sensitivities, parameter_basis)
                return (
                    next_state,
                    next_sensitivities,
                    accumulated_gradient + local_gradient,
                ), local_loss

            (_, _, gradient), losses = jax.lax.scan(
                scan_step,
                (initial_state, sensitivities, gradient),
                drives,
            )
            return jnp.sum(losses), losses, gradient

        carry_bytes = 8 * case.n_seed * case.n_theta * case.batch_size * case.n_x

    return PreparedControlled(
        function=jax.vmap(one_seed),
        roots=roots,
        state_scalar_count_per_seed=case.batch_size * case.n_x,
        parameter_count_per_seed=case.n_theta,
        rtrl_carry_bytes=carry_bytes,
    )


def prepare_hh(case: ControlledCase, method: str) -> PreparedControlled:
    """Build one fixed-mechanism HH controlled gradient kernel."""
    if case.workload not in {"hh_state", "hh_parameter"}:
        raise ValueError("prepare_hh requires an HH case.")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}.")
    config = BenchmarkConfig(
        n_cv=int(case.n_cv),
        duration_ms=case.num_steps * DT_MS,
        batch_size=case.batch_size,
        n_seed=case.n_seed,
    )
    times_ms = jnp.arange(case.num_steps, dtype=jnp.float64) * DT_MS
    target = build_cell(config, trainable=False, mechanism=FULL_HH_SPEC)
    target_voltage = simulate_voltage(target, times_ms)
    trainable_channels = ("leak", "na", "k") if case.workload == "hh_state" else ("leak",)
    candidate = build_cell(
        config,
        trainable=True,
        mechanism=FULL_HH_SPEC,
        trainable_channels=trainable_channels,
        trainable_group_by=str(case.parameter_group),
    )

    def rollout_step(data):
        time_ms, target_mv = data
        voltage = candidate.V.value.to_decimal(u.mV)
        local_loss = jnp.mean((voltage - target_mv) ** 2) / case.num_steps
        with brainstate.environ.context(t=time_ms * u.ms):
            candidate.update()
        return local_loss

    engine = build_rollout_value_and_grad(candidate, step=rollout_step, method=method)
    engine.prepare((times_ms[0], target_voltage[0]))
    seed_roots = seed_parameter_roots(engine.parameter_states, n_seed=case.n_seed)
    if method == "bptt":
        one_seed = engine._bptt
        carry_bytes = None
    else:
        one_seed = engine._rtrl
        _, one_tangents = engine._initial_full_carry(tuple(root[0] for root in seed_roots))
        carry_bytes = case.n_seed * _tree_nbytes(one_tangents)
    names = engine.parameter_names

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
    parameter_count = sum(int(np.prod(root.shape[1:])) for root in seed_roots)
    if parameter_count != case.n_theta:
        raise RuntimeError(f"Expected N_theta={case.n_theta}, materialized {parameter_count}.")
    return PreparedControlled(function, seed_roots, state_count, parameter_count, carry_bytes)


def prepare_case(case: ControlledCase, method: str) -> PreparedControlled:
    """Prepare a synthetic or HH controlled case."""
    return prepare_synthetic(case, method) if case.workload == "synthetic" else prepare_hh(case, method)


def _compiler_cost_fields(compiled) -> dict[str, object]:
    analysis = compiled.cost_analysis()
    if isinstance(analysis, (tuple, list)):
        analysis = analysis[0] if analysis else {}
    serializable = {str(key): float(value) for key, value in dict(analysis).items() if np.isscalar(value)}
    normalized = {key.lower().replace(" ", "_"): value for key, value in serializable.items()}
    return {
        "compiler_cost_analysis": json.dumps(serializable, sort_keys=True),
        "compiler_flops": normalized.get("flops"),
        "compiler_transcendentals": normalized.get("transcendentals"),
        "compiler_bytes_accessed": normalized.get("bytes_accessed"),
    }


def run_trial(
    case: ControlledCase,
    method: str,
    *,
    replicate: int,
    repeats: int,
    output_path: Path,
    physical_gpu: int | None,
    backsub: str = "recursive",
) -> dict[str, object]:
    """Compile, execute, and persist one controlled worker trial."""
    if method not in METHODS or backsub not in BACKSUBS:
        raise ValueError("Unsupported method or backsub mode.")
    if replicate < 1 or repeats < 1:
        raise ValueError("replicate and repeats must be positive.")
    os.environ["BRAINCELL_DHS_BACKSUB"] = backsub
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        **asdict(case),
        "config_id": case.id,
        "pair_id": f"{case.id}__rep{replicate}",
        "replicate": replicate,
        "method": method,
        "backsub": backsub,
        "repeats": repeats,
        "status": "running",
    }
    try:
        with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
            prepared = prepare_case(case, method)
            arguments = (prepared.roots,)
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
            loss_np, losses_np, gradient_np = map(np.asarray, (loss, losses, gradient))
            gradient_path = output_path.with_suffix(".npz")
            np.savez_compressed(gradient_path, loss=loss_np, losses=losses_np, gradient=gradient_np)
            memory = compiled.memory_analysis()
            logical_base = case.num_steps * case.n_seed * case.batch_size * case.n_x
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
                    "parameter_count_total": prepared.parameter_count_per_seed * case.n_seed,
                    "logical_bptt_work_units": logical_base if case.workload == "synthetic" else None,
                    "logical_rtrl_work_units": (logical_base * case.n_theta if case.workload == "synthetic" else None),
                    "gradient_shape": list(gradient_np.shape),
                    "loss_shape": list(loss_np.shape),
                    "losses_shape": list(losses_np.shape),
                    "gradient_l2": float(np.linalg.norm(gradient_np)),
                    "loss_mean": float(np.mean(loss_np)),
                    "gradient_file": gradient_path.name,
                    "host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                }
            )
            result.update(_compiler_cost_fields(compiled))
            result.update(_phase_metric_fields("compile", compile_metrics))
            result.update(_phase_metric_fields("first", first_metrics))
            result.update(_phase_metric_fields("steady", steady_metrics))
    except Exception as exc:
        result.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(output_path, result)
        raise
    _write_json(output_path, result)
    return result


def aggregate_results(output_dir: Path) -> list[dict[str, object]]:
    """Aggregate controlled trials and attach replicate-paired correctness."""
    trial_dir = output_dir / "trials"
    rows = [_read_json(path) for path in sorted(trial_dir.glob("*.json"))]
    pairs: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), {})[str(row["method"])] = row
    for methods in pairs.values():
        bptt, rtrl = methods.get("bptt"), methods.get("rtrl")
        if not bptt or not rtrl or bptt.get("status") != "ok" or rtrl.get("status") != "ok":
            continue
        bptt_data = np.load(trial_dir / str(bptt["gradient_file"]))
        rtrl_data = np.load(trial_dir / str(rtrl["gradient_file"]))
        gradient_abs = np.abs(bptt_data["gradient"] - rtrl_data["gradient"])
        loss_abs = np.abs(bptt_data["loss"] - rtrl_data["loss"])
        scale = np.maximum.reduce(
            [np.abs(bptt_data["gradient"]), np.abs(rtrl_data["gradient"]), np.full_like(gradient_abs, 1e-30)]
        )
        comparison = {
            "gradient_max_abs_error": float(np.max(gradient_abs)),
            "gradient_max_rel_error": float(np.max(gradient_abs / scale)),
            "gradient_relative_l2_error": float(
                np.linalg.norm(gradient_abs)
                / max(
                    float(np.linalg.norm(bptt_data["gradient"])),
                    float(np.linalg.norm(rtrl_data["gradient"])),
                    1e-30,
                )
            ),
            "loss_max_abs_error": float(np.max(loss_abs)),
            "bptt_over_rtrl_time": float(bptt["steady_median_seconds"]) / float(rtrl["steady_median_seconds"]),
        }
        bptt.update(comparison)
        rtrl.update(comparison)
    _write_csv(output_dir / "results.csv", rows)
    return rows


def run_suite(
    suite: str,
    *,
    output_dir: Path,
    gpu: int,
    repeats: int,
    replicates: int,
    resume: bool,
    dry_run: bool,
    python_executable: Path | None = None,
    backsub: str = "recursive",
) -> Path:
    """Launch isolated controlled workers and aggregate their results."""
    cases = suite_cases(suite)
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_dir, log_dir = output_dir / "trials", output_dir / "logs"
    trial_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    worker_python = str(python_executable or sys.executable)
    _write_json(
        output_dir / "manifest.json",
        {
            "suite": suite,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu,
            "repeats": repeats,
            "replicates": replicates,
            "methods": METHODS,
            "backsub": backsub,
            "schedule": "replicate_major_balanced",
            "python_executable": worker_python,
            "configs": [asdict(case) | {"config_id": case.id} for case in cases],
        },
    )
    commands = []
    for replicate in range(1, replicates + 1):
        methods = METHODS if replicate % 2 == 1 else tuple(reversed(METHODS))
        for case in balanced_cases(cases, replicate):
            for method in methods:
                stem = f"{case.id}__rep{replicate}__{method}"
                trial_path = trial_dir / f"{stem}.json"
                if resume and _trial_succeeded(trial_path):
                    continue
                command = [
                    worker_python,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--case",
                    json.dumps(asdict(case)),
                    "--method",
                    method,
                    "--replicate",
                    str(replicate),
                    "--repeats",
                    str(repeats),
                    "--output",
                    str(trial_path),
                    "--physical-gpu",
                    str(gpu),
                    "--backsub",
                    backsub,
                ]
                commands.append((case, replicate, method, trial_path, command))
    if dry_run:
        for *_, command in commands:
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
    for index, (case, replicate, method, trial_path, command) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {case.id} rep={replicate} {method}", flush=True)
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
        (log_dir / f"{case.id}__rep{replicate}__{method}.log").write_text(
            completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0 and not trial_path.exists():
            _write_json(
                trial_path,
                {
                    **asdict(case),
                    "config_id": case.id,
                    "pair_id": f"{case.id}__rep{replicate}",
                    "replicate": replicate,
                    "method": method,
                    "status": "subprocess_error",
                    "returncode": completed.returncode,
                },
            )
        aggregate_results(output_dir)
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run isolated controlled benchmark workers.")
    run.add_argument("--suite", choices=("synthetic", "hh_state", "hh_parameter", "all"), default="all")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--gpu", type=int, default=7)
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--replicates", type=int, default=3)
    run.add_argument("--python", type=Path)
    run.add_argument("--backsub", choices=BACKSUBS, default="recursive")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--case", required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--replicate", type=int, required=True)
    worker.add_argument("--repeats", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--physical-gpu", type=int)
    worker.add_argument("--backsub", choices=BACKSUBS, default="recursive")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        run_trial(
            ControlledCase(**json.loads(args.case)),
            args.method,
            replicate=args.replicate,
            repeats=args.repeats,
            output_path=args.output,
            physical_gpu=args.physical_gpu,
            backsub=args.backsub,
        )
        return
    output_dir = args.output_dir or ARTIFACT_ROOT / "controlled_complexity_a100"
    print(
        run_suite(
            args.suite,
            output_dir=output_dir,
            gpu=args.gpu,
            repeats=args.repeats,
            replicates=args.replicates,
            resume=args.resume,
            dry_run=args.dry_run,
            python_executable=args.python,
            backsub=args.backsub,
        )
    )


if __name__ == "__main__":
    main()
