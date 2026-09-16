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

"""Measure fresh-cell initialization separately from first compiled execution.

No JAX or BrainCell import occurs until the source tree and backend/x64 environment
have been selected. Each trial constructs a new model and executes exactly one
compiled timestep. This measures startup, not steady-state simulation speed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


PHASES = ("declaration_s", "discretization_s", "init_state_s", "first_jit_step_s")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(size < 3 for size in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("--sizes must contain distinct integers >= 3")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")
    if args.gpu < 0:
        parser.error("--gpu must be >= 0")
    args.source_root = args.source_root.resolve()
    if not (args.source_root / "braincell" / "__init__.py").is_file():
        parser.error("--source-root must contain braincell/__init__.py")
    return args


def execution_plan(args):
    """Return the exact number of fresh models and timesteps, without running."""
    trials = len(args.sizes) * (args.warmup + args.repeat)
    return {
        "platform": args.platform,
        "precision_bits": 64,
        "sizes": args.sizes,
        "independent_processes": 1,
        "warmup_trials_per_size": args.warmup,
        "measured_trials_per_size": args.repeat,
        "fresh_models": trials,
        "first_jit_executions": trials,
        "timesteps_per_trial": 1,
        "dt_ms": 0.025,
        "total_timesteps": trials,
        "extra_validation_rollouts": 0,
        "extra_reset_calls": 0,
        "automatic_retries": 0,
        "phases": PHASES,
        "note": "Each fresh model compiles once; the first warmup trial includes first use. "
        "No extra first-use or steady-state runs. init_state includes its own rediscretization.",
    }


def cv_counts(n_cv):
    """Distribute exactly n_cv CVs over soma and two daughter branches."""
    return (n_cv // 3 + (n_cv % 3 > 0), n_cv // 3 + (n_cv % 3 > 1), n_cv // 3)


def build_cell(n_cv, *, pop_size=1):
    import brainunit as u
    from braincell import Branch, Cell, CVPerBranchList, Morphology, mech
    from braincell.filter import AllRegion

    soma = Branch.from_lengths(lengths=[40.0] * u.um, radii=[8.0, 8.0] * u.um, type="soma")
    morpho = Morphology.from_root(soma, name="soma")
    for name, radii in (("a", [2.0, 1.5, 1.0]), ("b", [1.8, 1.2, 0.8])):
        dend = Branch.from_lengths(lengths=[80.0, 120.0] * u.um, radii=radii * u.um, type="dendrite")
        morpho.attach(parent="soma", child_branch=dend, child_name=name)
    cell = Cell(morpho, pop_size=pop_size, cv_policy=CVPerBranchList(cv_counts(n_cv)),
                V_init=-60.0 * u.mV, solver="staggered")
    cell.paint(AllRegion(), mech.CableProperty(
        resting_potential=-60.0 * u.mV,
        membrane_capacitance=1.0 * u.uF / u.cm**2,
        axial_resistivity=100.0 * u.ohm * u.cm,
    ))
    cell.paint(AllRegion(), mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    return cell


def measure_trial(n_cv):
    """Construct, lower, initialize and execute one fresh passive fork."""
    import brainstate
    import brainunit as u
    import jax
    import numpy as np

    # Synchronize all dispatched work, including quantities outside State and
    # lazy coefficient buffers; device_get/voltage-only fences can miss these.
    def sync():
        jax.effects_barrier()
        jax.block_until_ready(jax.live_arrays())

    gc.collect()
    with brainstate.environ.context(precision=64, dt=0.025 * u.ms, t=0.0 * u.ms):
        start = time.perf_counter()
        cell = build_cell(n_cv)
        sync()
        declaration = time.perf_counter() - start

        start = time.perf_counter()
        cvs = cell.cvs
        sync()
        discretization = time.perf_counter() - start
        if len(cvs) != n_cv:
            raise AssertionError(f"Expected {n_cv} CVs, got {len(cvs)}")

        start = time.perf_counter()
        cell.init_state()
        sync()
        initialization = time.perf_counter() - start

        def step():
            cell.update()
            return cell.V.value

        start = time.perf_counter()
        voltage = brainstate.transform.jit(step)()
        jax.block_until_ready(voltage)
        sync()
        first_step = time.perf_counter() - start
        values = np.asarray(voltage.to_decimal(u.mV))
        if values.dtype != np.float64 or not np.all(np.isfinite(values)):
            raise AssertionError("Expected finite float64 voltage")
        # Leak drives a uniform resting state toward -65 mV. Axial flow is zero
        # in this case, so this checks the executed sample without another run.
        expected = (-60.0 + 0.025 * 0.1 * -65.0) / (1.0 + 0.025 * 0.1)
        np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-10)
        result = dict(zip(PHASES, (declaration, discretization, initialization, first_step)))
        result.update(n_cv=n_cv, n_point=cell.runtime.n_point,
                      voltage_min_mv=float(values.min()), voltage_max_mv=float(values.max()))
        return result


def collect_trials(args, measure, save):
    """Keep raw warmup samples and flush after every completed fresh model."""
    samples = []
    for size in args.sizes:
        for index in range(args.warmup + args.repeat):
            sample = measure(size)
            samples.append({"size": size, "index": index,
                            "warmup": index < args.warmup, **sample})
            save(samples)
    return samples


def summarize(samples):
    result = {}
    for size in dict.fromkeys(sample["size"] for sample in samples):
        measured = [s for s in samples if s["size"] == size and not s["warmup"]]
        if measured:
            result[str(size)] = {phase: {
                "median": statistics.median(s[phase] for s in measured),
                "min": min(s[phase] for s in measured),
                "max": max(s[phase] for s in measured),
            } for phase in PHASES}
    return result


def environment(source_root):
    import braincell
    import jax

    source_files = sorted((source_root / "braincell").rglob("*.py"))
    hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_files}
    versions = {}
    for package in ("jax", "jaxlib", "brainstate", "brainunit", "numpy"):
        versions[package] = metadata.version(package)
    cpuinfo = Path("/proc/cpuinfo")
    cpu_model = platform.processor()
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "os": platform.platform(),
        "cpu": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "versions": versions,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "braincell_import": str(Path(braincell.__file__).resolve()),
        "source_root": str(source_root),
        "git_head": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "-C", str(source_root), "status", "--short"], text=True),
        "source_sha256": hashes,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv=None):
    args = parse_args(argv)
    if any(name in sys.modules for name in ("jax", "braincell", "brainstate")):
        raise RuntimeError("Run as a fresh process before importing JAX/BrainCell")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.platform == "cpu" else "cuda"
    os.environ["JAX_PLATFORM_NAME"] = "cpu" if args.platform == "cpu" else "gpu"
    if args.platform == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "1"
    # Persistent compilation cache would make the second source tree unfairly
    # inherit the first one's compiled executables.
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    sys.path.insert(0, str(args.source_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write("{}\n")
    report = {"label": args.label, "plan": execution_plan(args), "samples": [], "status": "initializing"}

    def save(samples):
        report["samples"] = list(samples)
        report["summary"] = summarize(samples)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        report["environment"] = environment(args.source_root)
        imported = Path(report["environment"]["braincell_import"])
        if not imported.is_relative_to(args.source_root):
            raise RuntimeError(f"Wrong source tree imported: {imported}")
        report["status"] = "running"
        save([])
        collect_trials(args, measure_trial, save)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save(report["samples"])
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
