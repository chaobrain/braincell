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

"""Compare population initialization and compiled forward-simulation throughput.

Each case constructs one Cell. The same compiled rollout is reused after reset;
initialization, first compilation/execution, resets and warm execution are timed
separately. No training, backwards pass, or geometry mutation is performed.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

if __package__:
    from . import benchmark as initialization
else:
    import benchmark as initialization


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--pop-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, values, minimum in (("sizes", args.sizes, 3), ("pop-sizes", args.pop_sizes, 1)):
        if len(set(values)) != len(values) or any(value < minimum for value in values):
            parser.error(f"--{name} requires distinct integers >= {minimum}")
    if args.steps < 1 or args.warmup < 0 or args.repeat < 1 or args.gpu < 0:
        parser.error("steps/repeat must be positive; warmup/gpu must be nonnegative")
    if Path(args.label).name != args.label or args.label in (".", ".."):
        parser.error("--label must be a single filename component")
    args.source_root = args.source_root.resolve()
    if not (args.source_root / "braincell" / "__init__.py").is_file():
        parser.error("--source-root must be a BrainCell checkout")
    return args


def trial_kinds(warmup, repeat):
    return ("compile",) + ("warmup",) * warmup + ("timed",) * repeat


def execution_plan(args):
    cases = len(args.sizes) * len(args.pop_sizes)
    runs = cases * len(trial_kinds(args.warmup, args.repeat))
    return {
        "platform": args.platform, "precision_bits": 64,
        "sizes": args.sizes, "pop_sizes": args.pop_sizes,
        "cases": cases, "fresh_models": cases, "initializations": cases,
        "first_jit_executions": cases,
        "extra_warmup_rollouts_per_case": args.warmup,
        "timed_rollouts_per_case": args.repeat,
        "rollouts": runs, "resets": runs, "steps_per_rollout": args.steps,
        "total_steps": runs * args.steps, "dt_ms": 0.025,
        "independent_processes": 1, "automatic_retries": 0,
        "extra_validation_rollouts": 0,
        "note": "One first-use initialization per case; not a repeated-init benchmark. "
        "Compile once, then reuse. Reset and host checks excluded from rollout timing. "
        "Clear compilation caches between cases, never between a case's runs.",
    }


def build_population(n_cv, pop_size):
    """Nonuniform initial voltage prevents identical population trajectories."""
    import brainunit as u
    import numpy as np

    cell = initialization.build_cell(n_cv, pop_size=pop_size)
    offset = np.linspace(0.0, 10.0, pop_size)[:, None]
    spatial = 2.0 * np.sin(np.linspace(0.0, np.pi, n_cv))[None, :]
    cell.V_init = u.Quantity(-65.0 + offset + spatial, u.mV)
    return cell


def make_rollout(cell, steps):
    """Return a reusable stateful compiled loop with a dynamic voltage input."""
    import brainstate
    import brainunit as u
    import jax.numpy as jnp

    dt = 0.025 * u.ms

    def rollout():
        def step(index):
            with brainstate.environ.context(t=index * dt):
                cell.update()
        brainstate.transform.for_loop(step, jnp.arange(steps))
        return cell.V.value

    return brainstate.transform.jit(rollout)


def check_voltage(voltage, *, n_cv, pop_size, steps):
    """Validate already-executed samples without an extra simulation."""
    import brainunit as u
    import numpy as np

    values = np.asarray(voltage.to_decimal(u.mV))
    if values.shape != (pop_size, n_cv) or values.dtype != np.float64:
        raise AssertionError(f"Unexpected voltage shape/dtype: {values.shape}, {values.dtype}")
    if not np.all(np.isfinite(values)) or values.min() < -65.000001 or values.max() > -52.999999:
        raise AssertionError("Passive rollout is nonfinite or outside its initial/resting voltage range")
    # Equal geometry and uniform leak imply that adding a constant voltage
    # offset to one population member decays identically at every CV.
    if pop_size > 1:
        offset = 10.0 * (1.0 + 0.025 * 0.1) ** (-steps)
        np.testing.assert_allclose(values[-1] - values[0], offset, rtol=1e-7, atol=1e-7)
    return values


def logical_state_bytes(cell):
    """Count logical array bytes, not unique physical storage or peak memory."""
    import brainstate
    import jax

    arrays = {}
    for state in brainstate.graph.states(cell).values():
        for leaf in jax.tree.leaves(state.value):
            if hasattr(leaf, "nbytes"):
                arrays[id(leaf)] = int(leaf.nbytes)
    return sum(arrays.values())


def measure_case(args, case, save):
    import brainstate
    import brainunit as u
    import jax
    import numpy as np

    n_cv, pop_size = case["n_cv"], case["pop_size"]

    def sync():
        jax.effects_barrier()
        jax.block_until_ready(jax.live_arrays())

    gc.collect()
    jax.clear_caches()
    with brainstate.environ.context(precision=64, dt=0.025 * u.ms, t=0.0 * u.ms):
        start = time.perf_counter()
        cell = build_population(n_cv, pop_size)
        sync()
        case["declaration_s"] = time.perf_counter() - start
        start = time.perf_counter()
        cell.init_state()
        sync()
        case["init_state_s"] = time.perf_counter() - start
        case["voltage_shape"] = list(cell.V.value.shape)
        case["n_point"] = cell.runtime.n_point
        case["logical_state_bytes"] = logical_state_bytes(cell)
        cable = getattr(cell.runtime, "cable", None)
        case["cable_array_shapes"] = None if cable is None else [list(value.shape) for value in cable]
        case["cable_array_bytes"] = None if cable is None else sum(value.mantissa.nbytes for value in cable)
        case["status"] = "initialized"
        save()

        runner = make_rollout(cell, args.steps)
        reference = None
        case["runs"] = []
        for kind in trial_kinds(args.warmup, args.repeat):
            start = time.perf_counter()
            cell.reset_state()
            sync()
            reset_s = time.perf_counter() - start
            start = time.perf_counter()
            voltage = runner()
            jax.block_until_ready(voltage)
            sync()
            elapsed = time.perf_counter() - start
            values = check_voltage(voltage, n_cv=n_cv, pop_size=pop_size, steps=args.steps)
            if reference is None:
                reference = values.copy()
            else:
                np.testing.assert_allclose(values, reference, rtol=1e-12, atol=1e-10)
            case["runs"].append({"kind": kind, "reset_s": reset_s, "rollout_s": elapsed})
            save()

        measured = [run["rollout_s"] for run in case["runs"] if run["kind"] == "timed"]
        median = statistics.median(measured)
        case["steady"] = {"median_s": median, "min_s": min(measured), "max_s": max(measured),
                          "population_step_us": median * 1e6 / args.steps,
                          "cell_step_us": median * 1e6 / (args.steps * pop_size),
                          "cell_steps_per_second": args.steps * pop_size / median}
        voltage_dir = args.output.parent / (args.label + "_voltages")
        voltage_dir.mkdir(parents=True, exist_ok=True)
        voltage_path = voltage_dir / f"cv_{n_cv}_pop_{pop_size}.npz"
        np.savez_compressed(voltage_path, final_voltage_mv=reference)
        case["voltage_file"] = str(voltage_path.resolve())
        case["status"] = "complete"
        save()


def collect_cases(args, report, measure, save):
    for n_cv in args.sizes:
        for pop_size in args.pop_sizes:
            case = {"n_cv": n_cv, "pop_size": pop_size, "status": "building"}
            report["cases"].append(case)
            save()
            measure(args, case, save)


def main(argv=None):
    args = parse_args(argv)
    if any(name in sys.modules for name in ("jax", "braincell", "brainstate")):
        raise RuntimeError("Start a fresh process before importing JAX/BrainCell")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.platform == "cpu" else "cuda"
    os.environ["JAX_PLATFORM_NAME"] = "cpu" if args.platform == "cpu" else "gpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.platform == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(args.source_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write("{}\n")
    report = {"label": args.label, "plan": execution_plan(args), "cases": [], "status": "initializing"}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        report["environment"] = initialization.environment(args.source_root)
        env = report["environment"]
        env["shared_driver_sha256"] = env["driver_sha256"]
        env["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not Path(env["braincell_import"]).is_relative_to(args.source_root):
            raise RuntimeError("Imported the wrong BrainCell source tree")
        report["status"] = "running"
        save()
        collect_cases(args, report, measure_case, save)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    print(json.dumps({"output": str(args.output), "status": report["status"], "cases": len(report["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
