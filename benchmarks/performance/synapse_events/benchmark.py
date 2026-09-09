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

"""Measure scheduled NetStim events with isolated CPU and GPU workers.

Run from the checkout with ``python -m benchmarks.performance.synapse_events.benchmark``.
The coordinator imports no accelerator libraries; each worker selects its device
before importing JAX. All repeated model steps execute in compiled transforms.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_PYTHON = sys.executable


@dataclass(frozen=True)
class Case:
    """Describe one fixed-size, fixed-duration workload.

    Parameters
    ----------
    n, m : int
        Cell count and independent inputs per cell.
    layout : str
        ``independent`` assigns one synapse per source; ``shared`` one per cell.
    rate_hz : float
        Reciprocal mean source interval, in Hz.
    pattern : str
        ``phase``, ``sync`` or ``poisson`` source schedules.
    number : int or None
        Finite event count per source, or automatically choose a covering count.
    delay : str
        ``zero``, ``fixed`` (1 ms), or ``heterogeneous`` (uniform 0--5 ms).
    declaration : str
        One ``batched`` connection call or one call ``per_cell``.
    control : str
        ``active``, ``cell_only``, ``unconnected`` or ``silent``.
    duration_ms, dt_ms : float
        Simulation duration and integration step, in ms.
    seed : int
        Reproducible source and phase seed.

    Notes
    -----
    CLI physical values use the units in their field names and are converted
    to brainunit quantities at the model boundary. ``number=None`` chooses a
    sufficiently long finite schedule; noisy schedules are checked for exhaustion.
    """

    n: int = 1
    m: int = 10
    layout: str = "independent"
    rate_hz: float = 100.0
    pattern: str = "phase"
    number: int | None = None
    delay: str = "zero"
    declaration: str = "batched"
    control: str = "active"
    duration_ms: float = 100.0
    dt_ms: float = 0.025
    seed: int = 7

    def __post_init__(self):
        for name in ("n", "m", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer in its valid range")
        if self.number is not None and (
            isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 0
        ):
            raise ValueError("number must be a nonnegative integer or None")
        for name in ("rate_hz", "duration_ms", "dt_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.steps < 1 or not math.isclose(self.steps * self.dt_ms, self.duration_ms, abs_tol=1e-9):
            raise ValueError("duration must be a positive integer multiple of dt")
        for name, options in {
            "layout": ("independent", "shared"),
            "pattern": ("phase", "sync", "poisson"),
            "delay": ("zero", "fixed", "heterogeneous"),
            "declaration": ("batched", "per_cell"),
            "control": ("active", "cell_only", "unconnected", "silent"),
        }.items():
            if getattr(self, name) not in options:
                raise ValueError(f"invalid {name}")

    @property
    def steps(self):
        """Return the number of integration steps."""
        return round(self.duration_ms / self.dt_ms)

    @property
    def key(self):
        """Return a stable configuration identifier."""
        # CLI floats and Python integer literals must identify the same protocol.
        config = asdict(self)
        for field in ("rate_hz", "duration_ms", "dt_ms"):
            config[field] = float(config[field])
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        return f"n{self.n}_m{self.m}_{self.layout}_{self.control}_{digest}"


def cases_for(suite, ns=(1, 10, 100), ms=(1, 10, 100)):
    """Return deduplicated cases and the experiments each belongs to.

    Parameters
    ----------
    suite : str
        ``all``, ``smoke``, or one experiment name below.
    ns, ms : tuple of int
        Cell and input counts for the scaling experiment.

    Returns
    -------
    dict
        Case objects mapped to experiment-name lists.
    """
    groups = {name: [] for name in ("smoke", "scaling", "activity", "schedule", "delay", "declarations", "controls")}
    for layout in ("independent", "shared"):
        base = Case(layout=layout)
        representative = replace(base, n=10)
        groups["smoke"].append(base)
        groups["scaling"].extend(replace(base, n=n, m=m) for n in ns for m in ms)
        groups["activity"].extend(
            replace(representative, m=m, rate_hz=rate, pattern=pattern)
            for m in (10, 100)
            for rate, pattern in ((10, "phase"), (100, "phase"), (1000, "phase"), (100, "sync"), (100, "poisson"))
        )
        groups["schedule"].extend(replace(representative, number=k) for k in (10, 100, 1000, 10000))
        groups["delay"].extend(replace(representative, delay=d) for d in ("zero", "fixed", "heterogeneous"))
        groups["declarations"].extend(
            replace(base, n=n, declaration=d) for n in (10, 100) for d in ("batched", "per_cell")
        )
        groups["controls"].extend(
            replace(base, n=n, control=c) for n in (1, 10, 100) for c in ("active", "unconnected", "silent")
        )
    groups["controls"].extend(Case(n=n, control="cell_only") for n in (1, 10, 100))
    selected = groups if suite == "all" else {suite: groups[suite]}
    cases = {}
    for group, members in selected.items():
        for case in members:
            cases.setdefault(case, []).append(group)
    return cases


def build_workload(case, *, cell_type=None):
    """Construct and prepare the actual BrainCell workload.

    Parameters
    ----------
    case : Case
        Validated experiment configuration.
    cell_type : type or None, optional
        Experimental Cell subclass; defaults to the production Cell.

    Returns
    -------
    dict
        Network, Cell, source, connection metadata and prepared runtime layout.
    """
    import braincell as bc
    import brainstate
    import brainunit as u
    import jax
    import numpy as np
    from braincell.filter import AllRegion, LocsetMask

    start = time.perf_counter()
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = (bc.Cell if cell_type is None else cell_type)(
        bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=case.n, V_init=-65.0 * u.mV
    )
    cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    if case.control != "cell_only":
        count = case.m if case.layout == "independent" else 1
        cell.place(
            LocsetMask.from_columns(np.zeros(count, dtype=int), np.full(count, 0.5)),
            bc.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV),
        )
    net = bc.Network("event_benchmark", seed=case.seed)
    post = net.add_population("post", cell)
    source = None
    size = case.n * case.m
    delay_ms = np.zeros(size)
    source_seconds = 0.0
    if case.control in ("active", "silent"):
        source_start = time.perf_counter()
        interval = 1000.0 / case.rate_hz
        expected = case.duration_ms / interval
        number = (
            case.number
            if case.number is not None
            else math.ceil(expected + 12 * math.sqrt(expected + 1) + 20 if case.pattern == "poisson" else expected + 1)
        )
        # Generate identical host schedules on CPU for both execution devices.
        with jax.default_device(jax.devices("cpu")[0]), brainstate.environ.context(precision=32):
            rng = brainstate.random.RandomState(case.seed)
            phases = np.asarray(rng.uniform(0.0, interval, size=size), dtype=np.float64)
            starts = phases if case.pattern == "phase" else np.zeros(size)
            if case.control == "silent":
                starts = starts + case.duration_ms + interval
            source = bc.NetStim(
                size=size,
                start=starts * u.ms,
                interval=interval * u.ms,
                number=number,
                noise=float(case.pattern == "poisson"),
                seed=case.seed,
            )
            if case.delay == "heterogeneous":
                delay_ms = np.asarray(rng.uniform(0.0, 5.0, size=size), dtype=np.float64)
            elif case.delay == "fixed":
                delay_ms[:] = 1.0
        stim = net.add_population("stim", source)
        source_seconds = time.perf_counter() - source_start
        targets = post.synapses["exp"]
        target_indices = np.arange(size) if case.layout == "independent" else np.repeat(np.arange(case.n), case.m)
        ranges = (
            [(0, size)] if case.declaration == "batched" else [(i * case.m, (i + 1) * case.m) for i in range(case.n)]
        )
        for i, (lo, hi) in enumerate(ranges):
            net.connect(
                f"input_{i}",
                source=stim.event_outputs["spike"][lo:hi],
                synapse=targets[target_indices[lo:hi]],
                weight=0.001 * u.uS,
                delay=delay_ms[lo:hi] * u.ms,
            )
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    net.prepare_run(dt=case.dt_ms * u.ms, event_backend="scatter")
    states = tuple(brainstate.graph.states(cell).values())
    jax.block_until_ready(tuple(s.value for s in states))
    prepare_seconds = time.perf_counter() - start
    layouts = tuple(cell._event_layouts())
    layout = layouts[0] if layouts else None
    node = cell.runtime.get_runtime_node(layout.id) if layout is not None else None
    return dict(
        net=net,
        cell=cell,
        source=source,
        delay_ms=delay_ms,
        layout=layout,
        node=node,
        states=states,
        build_seconds=build_seconds,
        source_seconds=source_seconds,
        prepare_seconds=prepare_seconds,
    )


def arrival_counts(case, workload):
    """Count delivered source events independently on the host.

    Returns
    -------
    ndarray
        Count per source in the simulated step window, after per-contact delay.
    """
    import numpy as np
    import brainunit as u
    from braincell.network.event import round_half_up_steps_host

    source = workload["source"]
    if source is None:
        return np.zeros(0, dtype=np.int64)
    times = np.asarray(source.event_times.to_decimal(u.ms))
    steps = round_half_up_steps_host((times + workload["delay_ms"][:, None]) / case.dt_ms)
    return ((steps >= 0) & (steps < case.steps) & source._event_mask).sum(axis=1)


def prepare_functions(case, workload):
    """Compile-ready wrappers around production paths, without time-step Python loops.

    Returns
    -------
    tuple
        Reset callable, per-mode callables, and dynamic device time arguments.
    """
    import brainstate
    import brainunit as u
    import jax.numpy as jnp
    import numpy as np

    net, cell, node = (workload[k] for k in ("net", "cell", "node"))

    @brainstate.transform.jit
    def reset():
        net.reset_state()
        return tuple(s.value for s in workload["states"])

    @brainstate.transform.jit
    def full(times):
        def step(_):
            net.update()

        brainstate.transform.for_loop(step, times)
        conductance = (
            jnp.zeros((case.n,)) if node is None else (node.g.value.to_decimal(u.uS).reshape(case.n, -1).sum(axis=1))
        )
        return cell.V.value.to_decimal(u.mV), conductance

    modes = {"full": full}
    source, layout = workload["source"], workload["layout"]
    if source is not None:
        indices = np.arange(source.size, dtype=np.int32)
        template = cell.runtime.get_event_buffer(layout.id)

        def query(t):
            return source.event_count(
                indices, t=t * u.ms, delay=workload["delay_ms"] * u.ms, dt=case.dt_ms * u.ms
            ).sum()

        def aggregate(t):
            with brainstate.environ.context(dt=case.dt_ms * u.ms):
                values = cell._evaluate_contact_inputs(layout, t=t * u.ms, template=template)
            return values.to_decimal(u.uS).sum()

        def wrap(function):
            @brainstate.transform.jit
            def run(times):
                def step(total, t):
                    return total + function(t), None

                total, _ = brainstate.transform.scan(step, jnp.asarray(0.0), times)
                return total

            return run

        modes.update(query=wrap(query), aggregate=wrap(aggregate))
    return reset, modes, jnp.arange(case.steps, dtype=jnp.float32) * case.dt_ms


def summarize_times(samples):
    """Return seconds-based statistics retaining every measurement.

    Parameters
    ----------
    samples : list of float
        Nonempty wall-time measurements in seconds.

    Returns
    -------
    dict
        Raw samples, median, extrema and sample standard deviation.
    """
    return dict(
        samples_s=samples,
        median_s=statistics.median(samples),
        min_s=min(samples),
        max_s=max(samples),
        stdev_s=statistics.stdev(samples) if len(samples) > 1 else 0.0,
    )


def measure_case(case, *, warmup=2, repeat=5, trace_dir=None):
    """Measure one case and validate microbenchmark event totals.

    Parameters
    ----------
    case : Case
        Configuration to execute in the already-selected device process.
    warmup, repeat : int
        Additional warmups after the cold invocation and timed repetitions.
    trace_dir : str or None
        Optional trace destination, measured separately from steady samples.

    Returns
    -------
    dict
        Timings, final states, counts, configuration and machine metadata.
    """
    import brainunit as u
    import jax
    import numpy as np

    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be nonnegative and repeat positive")
    workload = build_workload(case)
    reset, modes, times = prepare_functions(case, workload)
    jax.block_until_ready(times)
    counts = arrival_counts(case, workload)
    total = int(counts.sum())
    source = workload["source"]
    if source is not None and case.number is None and case.control == "active":
        last = np.asarray(source.event_times.to_decimal(u.ms))[:, -1]
        if np.any(last < case.duration_ms):
            raise ValueError("finite NetStim schedule exhausted; increase --number")
    timing = {}
    final = None
    # These are independent benchmark trials; each model trajectory is compiled.
    for name, function in modes.items():
        reset_start = time.perf_counter()
        jax.block_until_ready(reset())
        reset_first = time.perf_counter() - reset_start
        start = time.perf_counter()
        result = jax.block_until_ready(function(times))
        first = time.perf_counter() - start
        for _ in range(warmup):
            jax.block_until_ready(reset())
            jax.block_until_ready(function(times))
        samples, resets = [], []
        for _ in range(repeat):
            start = time.perf_counter()
            jax.block_until_ready(reset())
            resets.append(time.perf_counter() - start)
            start = time.perf_counter()
            result = jax.block_until_ready(function(times))
            samples.append(time.perf_counter() - start)
        timing[name] = dict(
            **summarize_times(samples),
            first_call_s=first,
            reset_first_call_s=reset_first,
            reset=summarize_times(resets),
            us_per_step=statistics.median(samples) * 1e6 / case.steps,
            events_per_second=total / statistics.median(samples) if total else None,
        )
        if name == "full":
            final = [np.asarray(value).tolist() for value in result]
            if not all(np.all(np.isfinite(value)) for value in result):
                raise ValueError("nonfinite final state")
        else:
            expected = total if name == "query" else total * 0.001
            np.testing.assert_allclose(float(result), expected, rtol=3e-4, atol=1e-5)
            timing[name]["checksum"] = float(result)
    if trace_dir:
        jax.block_until_ready(reset())
        with jax.profiler.trace(str(trace_dir)):
            jax.block_until_ready(modes["full"](times))
    device = jax.devices()[0]
    memory = device.memory_stats()
    return dict(
        status="ok",
        case=asdict(case),
        key=case.key,
        timing=timing,
        final_voltage_mv=final[0],
        final_conductance_us=final[1],
        arrival_count=total,
        arrivals_per_source=counts.tolist(),
        source_count=0 if source is None else source.size,
        synapse_count=0 if workload["node"] is None else int(workload["node"].g.value.size),
        connection_count=len(workload["cell"].connections),
        schedule_shape=None if source is None else list(source._event_times_ms.shape),
        schedule_host_bytes=0 if source is None else source._event_times_ms.nbytes + source._event_mask.nbytes,
        build_s=workload["build_seconds"],
        source_build_s=workload["source_seconds"],
        prepare_s=workload["prepare_seconds"],
        device_memory_stats=memory,
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            device=str(device),
            device_kind=device.device_kind,
            x64=bool(jax.config.jax_enable_x64),
            versions={p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "brainstate", "brainunit")},
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            xla_flags=os.environ.get("XLA_FLAGS", ""),
        ),
    )


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _positive_csv(value):
    try:
        result = tuple(int(item) for item in value.split(","))
        if not result or min(result) < 1:
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc


def main(argv=None):
    """Run the coordinator or a single isolated worker.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments; defaults to the process arguments.

    Returns
    -------
    int
        Zero for success, one if any case failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "smoke", "scaling", "activity", "schedule", "delay", "declarations", "controls"),
        default="smoke",
    )
    parser.add_argument("--cells", type=_positive_csv, default=(1, 10, 100))
    parser.add_argument("--inputs", type=_positive_csv, default=(1, 10, 100))
    parser.add_argument("--devices", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; required for GPU coordinator runs")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--number", type=int)
    parser.add_argument("--rate-hz", type=float, help="Override the selected suite's source rates")
    parser.add_argument(
        "--pattern", choices=("phase", "sync", "poisson"), help="Override the selected suite's patterns"
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=(32, 64),
        default=32,
        help="Numerical precision; use a separate output directory for each precision",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts" / "baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trace", action="store_true", help="Capture a separate full-rollout trace for every selected case"
    )
    parser.add_argument("--case-json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if not args.case_json and args.devices != "cpu" and args.gpu is None:
        parser.error("--gpu is required when --devices includes gpu")
    if args.warmup < 0 or args.repeat < 1 or args.timeout <= 0:
        parser.error("invalid warmup, repeat, timeout or GPU index")
    if args.case_json:
        try:
            import brainstate

            brainstate.environ.set_precision(args.precision)
            case = Case(**json.loads(args.case_json))
            result = measure_case(
                case,
                warmup=args.warmup,
                repeat=args.repeat,
                trace_dir=args.out.parent / (args.out.stem + "_trace") if args.trace else None,
            )
            _write_json(args.out, result)
            return 0
        except Exception:
            _write_json(args.out, dict(status="failed", error=traceback.format_exc()))
            traceback.print_exc()
            return 1
    selected = {}
    for case, groups in cases_for(args.suite, args.cells, args.inputs).items():
        updates = dict(duration_ms=args.duration_ms, dt_ms=args.dt_ms)
        if args.number is not None:
            updates["number"] = args.number
        if args.rate_hz is not None:
            updates["rate_hz"] = args.rate_hz
        if args.pattern is not None:
            updates["pattern"] = args.pattern
        selected.setdefault(replace(case, **updates), []).extend(groups)
    args.out.mkdir(parents=True, exist_ok=True)
    devices = ("cpu", "gpu") if args.devices == "both" else (args.devices,)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    gpu_status = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used", "--format=csv"],
            capture_output=True,
            text=True,
        ).stdout
        if __import__("shutil").which("nvidia-smi")
        else "unavailable"
    )
    rows = []
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=git_head,
        git_status=git_status,
        gpu_status=gpu_status,
        command=sys.argv,
        precision=args.precision,
        warmup=args.warmup,
        repeat=args.repeat,
        cpu_info=Path("/proc/cpuinfo").read_text().split("\n\n")[0]
        if Path("/proc/cpuinfo").exists()
        else platform.processor(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        rows=rows,
    )
    for case, groups in selected.items():
        for device in devices:
            output = args.out / f"{device}_{case.key}.json"
            entry = dict(key=case.key, case=asdict(case), device=device, groups=groups, file=output.name)
            if args.resume and output.exists() and json.loads(output.read_text()).get("status") == "ok":
                entry["status"] = "ok"
            else:
                env = os.environ.copy()
                env.update(
                    JAX_PLATFORMS="cpu" if device == "cpu" else "cuda,cpu",
                    JAX_PLATFORM_NAME="cpu" if device == "cpu" else "gpu",
                    JAX_ENABLE_X64=str(args.precision == 64).lower(),
                    CUDA_VISIBLE_DEVICES="" if device == "cpu" else str(args.gpu),
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    JAX_ENABLE_COMPILATION_CACHE="false",
                )
                command = [
                    args.python,
                    "-m",
                    "benchmarks.performance.synapse_events.benchmark",
                    "--case-json",
                    json.dumps(asdict(case)),
                    "--out",
                    str(output.resolve()),
                    "--warmup",
                    str(args.warmup),
                    "--repeat",
                    str(args.repeat),
                    "--precision",
                    str(args.precision),
                ]
                if args.trace:
                    command.append("--trace")
                print(f"[{len(rows) + 1}/{len(selected) * len(devices)}] {device} {case.key} {groups}", flush=True)
                try:
                    run = subprocess.run(
                        command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout
                    )
                    entry["status"] = "ok" if run.returncode == 0 else "failed"
                    output.with_suffix(".log").write_text(run.stdout + run.stderr)
                    if run.returncode:
                        print(run.stderr[-2500:], flush=True)
                except subprocess.TimeoutExpired as exc:
                    entry["status"] = "timeout"
                    _write_json(output, dict(status="timeout", error=str(exc)))
            rows.append(entry)
            _write_json(args.out / "manifest.json", manifest)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
