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

"""Compare scheduled queries in isolated CPU/GPU processes.

Run with python -m benchmarks.performance.synapse_events.query_benchmark --suite all.
Physical CLI fields are in ms and converted to brainunit quantities at entry.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from benchmarks.performance.synapse_events.benchmark import DEFAULT_PYTHON, ROOT, summarize_times, _write_json


@dataclass(frozen=True)
class QueryCase:
    """Describe source count, schedule, fanout and target layout for one trial.

    Parameters
    ----------
    n, m, k, fanout, block : int
        Cell count, sources per cell, events per source, fanout and bucket block.
    layout, pattern, delay : str
        Independent/shared targets; phase/sparse/random/sync/burst/silent
        schedules; zero/heterogeneous connection delays.
    duration_ms, dt_ms : float
        Positive simulation window and fixed step, in ms.
    """

    n: int = 1
    m: int = 10
    k: int = 10
    fanout: int = 1
    block: int = 128
    layout: str = "independent"
    pattern: str = "phase"
    delay: str = "zero"
    duration_ms: float = 100.0
    dt_ms: float = 0.025

    def __post_init__(self):
        import math

        for field in ("n", "m", "k", "fanout", "block"):
            x = getattr(self, field)
            if isinstance(x, bool) or not isinstance(x, int) or x < 1:
                raise ValueError(f"invalid {field}")
        if self.layout not in ("independent", "shared") or self.pattern not in (
            "phase",
            "sparse",
            "random",
            "sync",
            "burst",
            "dense",
            "silent",
        ):
            raise ValueError("invalid layout or pattern")
        if self.delay not in ("zero", "heterogeneous"):
            raise ValueError("invalid delay")
        if not all(math.isfinite(x) and x > 0 for x in (self.duration_ms, self.dt_ms)):
            raise ValueError("invalid duration or dt")
        if self.steps < 1 or not math.isclose(self.steps * self.dt_ms, self.duration_ms, abs_tol=1e-9):
            raise ValueError("duration must be an integer multiple of dt")

    @property
    def steps(self):
        """Return the fixed rollout length."""
        return round(self.duration_ms / self.dt_ms)

    @property
    def key(self):
        """Return a stable case identifier."""
        data = asdict(self) | {"duration_ms": float(self.duration_ms), "dt_ms": float(self.dt_ms)}
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
        return f"n{self.n}_m{self.m}_k{self.k}_{self.pattern}_{self.layout}_f{self.fanout}_b{self.block}_{digest}"


def cases_for(suite):
    """Return deduplicated cases for smoke, scaling, schedule, activity, fanout or blocks.

    Parameters
    ----------
    suite : str
        Named experiment or ``all``.

    Returns
    -------
    dict
        Cases mapped to their experiment groups.
    """
    groups = {
        "smoke": [QueryCase()],
        "scaling": [
            QueryCase(n=n, m=m, layout=l) for n in (1, 10, 100) for m in (1, 10, 100) for l in ("independent", "shared")
        ],
        "schedule": [QueryCase(n=10, m=10, k=k) for k in (10, 100, 1000, 10000)],
        "activity": [
            QueryCase(n=10, m=100, pattern=p, k=320 if p in ("burst", "dense") else 10)
            for p in ("phase", "sparse", "random", "sync", "burst", "dense", "silent")
        ],
        "fanout": [QueryCase(n=10, m=10, fanout=f, delay=d) for f in (1, 10, 100) for d in ("zero", "heterogeneous")],
        "blocks": [
            QueryCase(n=10, m=100, pattern=p, k=320 if p == "burst" else 10, block=b)
            for p in ("sparse", "burst")
            for b in (32, 128, 512)
        ],
    }
    if suite not in (*groups, "all"):
        raise ValueError("unknown suite")
    result = {}
    for name, cases in groups.items():
        if suite in (name, "all"):
            for case in cases:
                result.setdefault(case, []).append(name)
    return result


def build_source(case):
    """Build identical host schedules and connection arrays on either backend.

    Parameters
    ----------
    case : QueryCase
        Fixed experimental configuration.

    Returns
    -------
    tuple
        NetStim, source indices, delay quantity, target indices and target count.
    """
    import brainstate
    import brainunit as u
    import jax
    import numpy as np
    from braincell.network.event import NetStim

    size = case.n * case.m
    with jax.default_device(jax.devices("cpu")[0]), brainstate.environ.context(precision=32):
        rng = brainstate.random.RandomState(7)
        # Reserve a final interval so every dense event stays inside the
        # default window after nearest-step delivery quantization.
        interval = 100.0 if case.pattern == "sparse" else (100.0 / (case.k + 1) if case.pattern == "dense" else 10.0)
        phase = np.asarray(rng.uniform(0.0, interval, size=size))
        starts = np.zeros(size) if case.pattern in ("sync", "burst") else phase
        if case.pattern == "silent":
            starts = starts + case.duration_ms + 100
        source = NetStim(
            size=size,
            number=case.k,
            start=starts * u.ms,
            interval=interval * u.ms,
            noise=1.0 if case.pattern == "random" else 0.0,
            seed=7,
        )
        if case.pattern == "burst":
            # A controlled schedule fixture, consumed by the unmodified actual
            # NetStim.event_count method. 32 coincident spikes per 10 ms burst.
            times = np.broadcast_to((np.arange(case.k) // 32) * 10.0, (size, case.k)).copy()
            object.__setattr__(source, "_event_times_ms", times)
        indices = np.repeat(np.arange(size, dtype=np.int32), case.fanout)
        delay = (
            np.asarray(rng.uniform(0.0, 5.0, size=indices.size))
            if case.delay == "heterogeneous"
            else np.zeros(indices.size)
        )
    targets = np.arange(indices.size, dtype=np.int32) if case.layout == "independent" else indices // case.m
    return source, indices, delay * u.ms, targets, int(targets.max()) + 1


def make_runner(plan, targets, n_targets, *, mode, record=False):
    """Compile query-only or query/weight/ExpSyn execution.

    Parameters
    ----------
    plan : EventPlan
        Prepared query.
    targets : array-like
        Target row per connection.
    n_targets : int
        Number of ExpSyn state rows.
    mode : str
        ``query`` or ``synapse``.
    record : bool
        Return full count/conductance trajectories for untimed validation.

    Returns
    -------
    callable
        Compiled function of device arrays, cursor, steps, weights and tau in ms.
    """
    import brainstate
    import brainunit as u
    import jax.numpy as jnp
    from braincell.synapse import ExpSyn

    if mode not in ("query", "synapse"):
        raise ValueError("unknown measurement mode")
    target = jnp.asarray(targets)
    node = ExpSyn(size=n_targets, tau=2 * u.ms)
    node.init_state()

    @brainstate.transform.jit
    def run(arrays, cursor, ks, weights, tau):
        node.g.value = jnp.zeros(n_targets, dtype=weights.dtype) * u.uS
        decay = jnp.exp(-plan.dt.to_decimal(u.ms) / tau)

        def step(carry, k):
            p, total = carry
            p, count = plan.query(arrays, p, k)
            if mode == "synapse":
                drive = jnp.zeros(n_targets, dtype=weights.dtype).at[target].add(count * weights)
                node.apply_events(drive * u.uS)
                node.g.value = node.g.value * decay
                value = node.g.value.to_decimal(u.uS)
            else:
                value = count
            return (p, total + value.sum()), value if record else None

        (p, total), trace = brainstate.transform.scan(step, (cursor, jnp.zeros((), dtype=weights.dtype)), ks)
        return p, total, node.g.value.to_decimal(u.uS), trace

    return run


def measure_case(case, *, repeat=5, warmup=2):
    """Measure and validate all four candidates on the selected device.

    Parameters
    ----------
    case : QueryCase
        Workload configuration.
    repeat, warmup : int
        Number of synchronized measured and warmup executions.

    Returns
    -------
    dict
        Raw timings, preparation costs, memory, validation and environment.
    """
    import brainunit as u
    import jax
    import jax.numpy as jnp
    import numpy as np
    from braincell.experimental.scheduled_events import METHODS, prepare_events

    t0 = time.perf_counter()
    source, indices, delay, targets, ntarget = build_source(case)
    build_s = time.perf_counter() - t0
    ks = jnp.arange(case.steps, dtype=jnp.int32)
    weights = jnp.linspace(0.0005, 0.0015, indices.size)
    tau = jnp.array(2.0)
    jax.block_until_ready((ks, weights, tau))
    rows, references = [], {}
    for method in METHODS:
        # Equal cold-cache conditions for each candidate's preparation/first call.
        jax.clear_caches()
        start = time.perf_counter()
        plan = prepare_events(
            source, indices, delay=delay, dt=case.dt_ms * u.ms, n_steps=case.steps, method=method, block_size=case.block
        )
        prep = time.perf_counter() - start
        start = time.perf_counter()
        state = jax.block_until_ready(plan.reset())
        reset_s = time.perf_counter() - start
        row = dict(
            method=method,
            prepare_cold_s=prep,
            reset_cold_s=reset_s,
            preparation=plan.preparation,
            query_array_bytes=sum(x.size * x.dtype.itemsize for x in plan.arrays),
            cursor_bytes=state.size * state.dtype.itemsize,
            timing={},
        )
        args = (plan.arrays, state, ks, weights, tau)
        for mode in ("query", "synapse"):
            run = make_runner(plan, targets, ntarget, mode=mode)
            start = time.perf_counter()
            result = jax.block_until_ready(run(*args))
            first = time.perf_counter() - start
            for _ in range(warmup):
                jax.block_until_ready(run(*args))
            samples = []
            for _ in range(repeat):
                start = time.perf_counter()
                result = jax.block_until_ready(run(*args))
                samples.append(time.perf_counter() - start)
            row["timing"][mode] = summarize_times(samples) | {"first_call_s": first, "checksum": float(result[1])}
            # Full trajectories are validated separately, not materialized in
            # the timed rollout. Neither timed benchmark is a component percent.
            check = make_runner(plan, targets, ntarget, mode=mode, record=True)
            trace = np.asarray(jax.block_until_ready(check(*args))[3])
            if method == "current":
                references[mode] = trace
            elif mode == "query":
                np.testing.assert_array_equal(trace, references[mode])
            else:
                np.testing.assert_allclose(trace, references[mode], rtol=3e-5, atol=1e-7)
            row[f"{mode}_max_error"] = float(np.max(np.abs(trace - references[mode])))
        rows.append(row)
        del plan, args, run, check, trace
    device = jax.devices()[0]
    return dict(
        status="ok",
        case=asdict(case),
        key=case.key,
        build_s=build_s,
        rows=rows,
        arrivals=int(references["query"].sum()),
        count_sha256=hashlib.sha256(references["query"].tobytes()).hexdigest(),
        schedule_sha256=hashlib.sha256(source._event_times_ms.tobytes()).hexdigest(),
        source_host_bytes=source._event_times_ms.nbytes + source._event_mask.nbytes,
        connections=indices.size,
        targets=ntarget,
        environment=dict(
            python=sys.version,
            device_kind=device.device_kind,
            platform=device.platform,
            x64=bool(jax.config.jax_enable_x64),
            versions={p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "brainstate", "brainunit")},
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ),
        device_allocator_stats=device.memory_stats(),
    )


def main(argv=None):
    """Run isolated cases, retaining logs, manifest and failures.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments.

    Returns
    -------
    int
        Zero if every requested case succeeded, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("smoke", "all", "scaling", "schedule", "activity", "fanout", "blocks"), default="smoke"
    )
    parser.add_argument("--devices", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; required for GPU coordinator runs")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts" / "queries")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if not args.worker and args.devices != "cpu" and args.gpu is None:
        parser.error("--gpu is required when --devices includes gpu")
    if args.repeat < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("repeat/timeout must be positive and warmup nonnegative")
    if args.worker:
        try:
            import brainstate
            import jax

            if jax.devices()[0].platform != args.devices:
                raise RuntimeError("requested backend was not selected")
            with brainstate.environ.context(precision=args.precision):
                result = measure_case(QueryCase(**json.loads(args.worker)), repeat=args.repeat, warmup=args.warmup)
            _write_json(args.out, result)
            return 0
        except Exception:
            _write_json(args.out, dict(status="failed", error=traceback.format_exc()))
            traceback.print_exc()
            return 1
    selected = {replace(c, duration_ms=args.duration_ms, dt_ms=args.dt_ms): g for c, g in cases_for(args.suite).items()}
    args.out.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        ROOT / "braincell/experimental/scheduled_events.py",
        ROOT / "braincell/network/event.py",
        ROOT / "braincell/synapse/exponential.py",
    ]
    protocol = dict(
        suite=args.suite,
        devices=args.devices,
        gpu=args.gpu,
        precision=args.precision,
        repeat=args.repeat,
        warmup=args.warmup,
        python=args.python,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    old_path = args.out / "manifest.json"
    if args.resume and old_path.exists() and json.loads(old_path.read_text())["protocol"] != protocol:
        raise ValueError("resume requires identical code and protocol; use a new output directory")
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=protocol,
        cpu_info=Path("/proc/cpuinfo").read_text().split("\n\n")[0],
        rows=[],
    )
    devices = ("cpu", "gpu") if args.devices == "both" else (args.devices,)
    for case, groups in selected.items():
        for device in devices:
            path = args.out / f"{device}_{case.key}.json"
            record = dict(file=path.name, device=device, groups=groups, key=case.key)
            if not (args.resume and path.exists() and json.loads(path.read_text()).get("status") == "ok"):
                env = os.environ | {
                    "JAX_PLATFORMS": "cuda,cpu" if device == "gpu" else "cpu",
                    "JAX_PLATFORM_NAME": device,
                    "CUDA_VISIBLE_DEVICES": str(args.gpu) if device == "gpu" else "",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "JAX_ENABLE_COMPILATION_CACHE": "false",
                }
                command = [
                    args.python,
                    "-m",
                    __spec__.name,
                    "--worker",
                    json.dumps(asdict(case)),
                    "--devices",
                    device,
                    "--out",
                    str(path),
                    "--repeat",
                    str(args.repeat),
                    "--warmup",
                    str(args.warmup),
                    "--precision",
                    str(args.precision),
                ]
                with path.with_suffix(".log").open("w") as log:
                    try:
                        proc = subprocess.run(
                            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                        )
                        if proc.returncode and (
                            not path.exists() or json.loads(path.read_text()).get("status") == "ok"
                        ):
                            _write_json(path, dict(status="failed", error=f"exit {proc.returncode}"))
                    except subprocess.TimeoutExpired:
                        _write_json(path, dict(status="failed", error="worker timeout"))
            record["status"] = json.loads(path.read_text())["status"]
            manifest["rows"].append(record)
            _write_json(old_path, manifest)
            print(
                f"{len(manifest['rows'])}/{len(selected) * len(devices)} {device} {case.key}: {record['status']}",
                flush=True,
            )
    return int(any(r["status"] != "ok" for r in manifest["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
