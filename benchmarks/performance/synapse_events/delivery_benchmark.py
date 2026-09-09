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

"""Run bounded direct-delivery experiments in isolated CPU/GPU workers."""

import argparse
from dataclasses import asdict, replace
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

from benchmarks.performance.synapse_events.benchmark import DEFAULT_PYTHON, ROOT, _write_json, summarize_times
from benchmarks.performance.synapse_events.query_benchmark import QueryCase, build_source

METHODS = ("current", "scan", "bucket", "direct", "padded")
CONTROL_JOBS = {
    "cell_only": ("cell_only", ()),
    "unconnected": ("unconnected", ()),
    "silent_direct": ("silent", ("direct",)),
    "silent_padded": ("silent", ("padded",)),
    "active_direct": ("active", ("direct",)),
    "active_padded": ("active", ("padded",)),
}


def cases_for(phase):
    """Return the bounded six-case micro or four-case full-model matrix.

    Parameters
    ----------
    phase : str
        ``micro``, ``full``, ``profile`` or the paired ``controls`` jobs.

    Returns
    -------
    dict
        Stable descriptive names mapped to QueryCase objects.
    """
    cases = dict(
        small=QueryCase(),
        large=QueryCase(n=100, m=100),
        long=QueryCase(n=10, m=10, k=10000),
        sparse=QueryCase(n=10, m=100, pattern="sparse"),
        burst=QueryCase(n=10, m=100, k=320, pattern="burst"),
        dense=QueryCase(n=10, m=100, k=320, pattern="dense"),
    )
    if phase == "controls":
        return {name: cases["large"] for name in CONTROL_JOBS}
    if phase == "full":
        return {k: cases[k] for k in ("small", "large", "long")} | {"shared": replace(cases["large"], layout="shared")}
    if phase == "profile":
        return {k: cases[k] for k in ("sparse", "burst")} | {"shared": replace(cases["large"], layout="shared")}
    if phase != "micro":
        raise ValueError("unknown phase")
    return cases


def prepare_plan(case, method, source, indices, delay, targets, n_targets):
    """Prepare one candidate with the common source/grid protocol.

    Parameters
    ----------
    case : QueryCase
        Window and block configuration.
    method : str
        Candidate identifier.
    source, indices, delay, targets, n_targets : object
        Values returned by ``build_source``.

    Returns
    -------
    object
        Query or direct-delivery plan.
    """
    import brainunit as u
    from braincell.experimental.scheduled_delivery import prepare_delivery
    from braincell.experimental.scheduled_events import prepare_events

    kwargs = dict(delay=delay, dt=case.dt_ms * u.ms, n_steps=case.steps, method=method, block_size=case.block)
    if method in ("direct", "padded"):
        return prepare_delivery(source, indices, targets, n_targets=n_targets, **kwargs)
    return prepare_events(source, indices, **kwargs)


def make_runner(plan, targets, n_targets, *, mode="synapse", record=False):
    """Compile weighted delivery alone or delivery plus the real ExpSyn.

    Parameters
    ----------
    plan : object
        Prepared query or delivery plan.
    targets : array-like
        Target indices for query baselines.
    n_targets : int
        Target layout size.
    mode : str, optional
        ``delivery`` or ``synapse`` (default).
    record : bool, optional
        Return full trajectories for untimed validation.

    Returns
    -------
    callable
        Compiled function of arrays, integer steps, weights and tau.
    """
    import brainstate
    import brainunit as u
    import jax
    import jax.numpy as jnp
    from braincell.synapse import ExpSyn

    if mode not in ("delivery", "synapse"):
        raise ValueError("invalid mode")
    target = jnp.asarray(targets)
    node = ExpSyn(size=n_targets, tau=2 * u.ms)
    node.init_state()

    # Query plans carry dt; direct plans keep their timestep in the driver.
    @brainstate.transform.jit
    def run(arrays, ks, weights, tau, dt_ms):
        node.g.value = jnp.zeros(n_targets, dtype=weights.dtype) * u.uS
        decay = jnp.exp(-dt_ms / tau)

        def step(total, k):
            with jax.named_scope("braincell:scheduled_delivery:" + plan.method):
                if plan.method in ("direct", "padded"):
                    drive = plan.deliver(arrays, k, weights * u.uS).to_decimal(u.uS)
                else:
                    _, count = plan.query(arrays, jnp.empty(0, dtype=jnp.int32), k)
                    drive = jnp.zeros(n_targets, dtype=weights.dtype).at[target].add(count * weights)
            if mode == "synapse":
                node.apply_events(drive * u.uS)
                node.g.value = node.g.value * decay
                value = node.g.value.to_decimal(u.uS)
            else:
                value = drive
            return total + value.sum(), value if record else None

        total, trace = brainstate.transform.scan(step, jnp.zeros((), dtype=weights.dtype), ks)
        return total, node.g.value.to_decimal(u.uS), trace

    return run


def _environment():
    import jax

    device = jax.devices()[0]
    return dict(
        python=sys.version,
        device_kind=device.device_kind,
        platform=device.platform,
        x64=bool(jax.config.jax_enable_x64),
        versions={p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "brainstate", "brainunit")},
        affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def _measure(run, args, repeat, warmup):
    import jax

    start = time.perf_counter()
    jax.block_until_ready(run(*args))
    first = time.perf_counter() - start
    for _ in range(warmup):
        jax.block_until_ready(run(*args))
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(run(*args))
        samples.append(time.perf_counter() - start)
    return summarize_times(samples) | dict(first_call_s=first)


def measure_micro(case, *, methods=METHODS, repeat=5, warmup=2, order=0, profile_dir=None):
    """Measure candidates and validate complete target-count/drive/g traces.

    Parameters
    ----------
    case : QueryCase
        Experimental workload.
    methods : tuple of str
        Selected candidates.
    repeat, warmup, order : int
        Repeat counts and cyclic candidate ordering.
    profile_dir : Path or None
        Optional profiler output root; profile runs are not baseline timings.

    Returns
    -------
    dict
        Timings, failures, validation errors and measurement provenance.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from braincell.experimental.scheduled_delivery import PaddingLimitError

    began = time.perf_counter()
    source, indices, delay, targets, ntarget = build_source(case)
    build_s = time.perf_counter() - began
    ks = jnp.arange(case.steps, dtype=jnp.int32)
    weights = jnp.linspace(0.0005, 0.0015, indices.size)
    base = prepare_plan(case, "current", source, indices, delay, targets, ntarget)
    refs = {}
    for kind, mode, w in (
        ("count", "delivery", jnp.ones_like(weights)),
        ("drive", "delivery", weights),
        ("g", "synapse", weights),
    ):
        check = make_runner(base, targets, ntarget, mode=mode, record=True)
        refs[kind] = np.asarray(jax.device_get(check(base.arrays, ks, w, jnp.array(2.0), jnp.array(case.dt_ms))[2]))
    rows = []
    methods = tuple(methods)
    methods = methods[order % len(methods) :] + methods[: order % len(methods)]
    for method in methods:
        jax.clear_caches()
        start = time.perf_counter()
        try:
            plan = prepare_plan(case, method, source, indices, delay, targets, ntarget)
        except PaddingLimitError as exc:
            rows.append(
                dict(
                    method=method, status="not_applicable", reason=str(exc), prepare_cold_s=time.perf_counter() - start
                )
            )
            continue
        row = dict(
            method=method,
            status="ok",
            prepare_cold_s=time.perf_counter() - start,
            preparation=plan.preparation,
            array_bytes=sum(x.size * x.dtype.itemsize for x in plan.arrays),
            timing={},
        )
        args = (plan.arrays, ks, weights, jnp.array(2.0), jnp.array(case.dt_ms))
        for mode in ("delivery", "synapse"):
            run = make_runner(plan, targets, ntarget, mode=mode)
            row["timing"][mode] = _measure(run, args, repeat, warmup)
            if profile_dir is not None and mode == "synapse":
                path = profile_dir / method
                path.mkdir(parents=True, exist_ok=True)
                try:
                    # Lowering text is diagnostic only, never included in timing.
                    if hasattr(run, "lower"):
                        (path / "lowered.txt").write_text(str(run.lower(*args).compiler_ir()))
                    jax.profiler.start_trace(str(path))
                    with jax.profiler.TraceAnnotation("braincell:scheduled_steady"):
                        jax.block_until_ready(run(*args))
                    jax.profiler.stop_trace()
                    row["profile_status"] = "ok"
                except Exception as exc:
                    row["profile_status"] = f"unavailable: {exc}"
        for kind, mode, w in (
            ("count", "delivery", jnp.ones_like(weights)),
            ("drive", "delivery", weights),
            ("g", "synapse", weights),
        ):
            check = make_runner(plan, targets, ntarget, mode=mode, record=True)
            trace = np.asarray(jax.device_get(check(plan.arrays, ks, w, jnp.array(2.0), jnp.array(case.dt_ms))[2]))
            error = float(np.max(np.abs(trace - refs[kind]), initial=0))
            row[kind + "_max_error"] = error
            if kind == "count":
                np.testing.assert_array_equal(trace, refs[kind])
            else:
                np.testing.assert_allclose(trace, refs[kind], rtol=3e-5, atol=1e-7)
        rows.append(row)
    return dict(
        status="ok",
        case=asdict(case),
        rows=rows,
        build_s=build_s,
        arrivals=int(refs["count"].sum()),
        schedule_sha256=hashlib.sha256(source._event_times_ms.tobytes()).hexdigest(),
        count_sha256=hashlib.sha256(refs["count"].astype(np.int32).tobytes()).hexdigest(),
        environment=_environment(),
        device_allocator_stats=jax.devices()[0].memory_stats(),
        worker_wall_s=time.perf_counter() - began,
    )


def make_full_workload(case, method, *, control="active"):
    """Build the original passive Cell/Network with an optional input adapter.

    Parameters
    ----------
    case : QueryCase
        N/M/K, layout and time grid.
    method : str
        ``production`` or a candidate name.
    control : str, optional
        ``active``, ``silent``, ``unconnected`` or ``cell_only``.

    Returns
    -------
    dict
        Original benchmark workload with preparation statistics.
    """
    import brainunit as u
    from benchmarks.performance.synapse_events.benchmark import Case, build_workload
    from braincell.experimental.scheduled_cell import ScheduledDeliveryCell

    cfg = Case(
        n=case.n,
        m=case.m,
        number=case.k,
        layout=case.layout,
        dt_ms=case.dt_ms,
        duration_ms=case.duration_ms,
        delay=case.delay,
        control=control,
    )
    work = build_workload(cfg, cell_type=None if method == "production" else ScheduledDeliveryCell)
    start = time.perf_counter()
    plans = (
        ()
        if method == "production"
        else work["cell"].prepare_scheduled_delivery(
            method=method, dt=case.dt_ms * u.ms, n_steps=case.steps, block_size=case.block
        )
    )
    work["plan_prepare_s"] = time.perf_counter() - start
    work["array_bytes"] = sum(x.size * x.dtype.itemsize for p in plans for x in p.arrays)
    work["cfg"] = cfg
    return work


def full_runner(work, *, record=False):
    """Compile real Network steps, optionally returning all voltage/g samples.

    Parameters
    ----------
    work : dict
        Prepared workload.
    record : bool
        Return trajectories during untimed validation.

    Returns
    -------
    tuple
        Reset and rollout callables.
    """
    import brainstate
    import brainunit as u
    import jax.numpy as jnp

    net, cell, node = work["net"], work["cell"], work["node"]

    def conductance():
        return jnp.empty((0,), dtype=cell.V.value.dtype) if node is None else node.g.value.to_decimal(u.uS)

    @brainstate.transform.jit
    def reset():
        net.reset_state()
        return cell.V.value

    @brainstate.transform.jit
    def run(steps):
        def step(_):
            if record:
                if node is None:
                    drive = conductance()
                else:
                    with brainstate.environ.context(dt=net._prepared_run[2]):
                        drive = cell._evaluate_contact_inputs(
                            work["layout"],
                            t=cell.current_time,
                            template=cell.runtime.get_event_buffer(work["layout"].id),
                        ).to_decimal(u.uS)
            net.update()
            if record:
                return cell.V.value.to_decimal(u.mV), conductance(), drive
            return None

        traces = brainstate.transform.for_loop(step, steps)
        return cell.V.value.to_decimal(u.mV), conductance(), traces

    return reset, run


def measure_full(
    case, *, methods=METHODS, repeat=5, warmup=2, order=0, profile_dir=None, control="active", validate_control=False
):
    """Compare actual solver trajectories and timed complete rollouts.

    Parameters
    ----------
    case : QueryCase
        Complete-model workload.
    methods : tuple of str
        Candidate adapters; production is always measured first.
    repeat, warmup, order : int
        Repeat counts and candidate ordering.
    profile_dir : Path or None
        Optional real-network trace root; these workers are diagnostic only.
    control : str, optional
        Model control passed to the original workload builder.
    validate_control : bool, optional
        Also verify topology, arrivals, finite traces and reset reproducibility.

    Returns
    -------
    dict
        Full-model timing and same-device trajectory errors.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from braincell.experimental.scheduled_delivery import PaddingLimitError

    began = time.perf_counter()
    rows = []
    refs = None
    steps = jnp.arange(case.steps, dtype=jnp.int32)
    methods = tuple(methods)
    if methods:
        methods = methods[order % len(methods) :] + methods[: order % len(methods)]
    for method in ("production", *methods):
        jax.clear_caches()
        print(f"full {method}: building ({time.perf_counter() - began:.1f}s)", flush=True)
        try:
            work = make_full_workload(case, method, control=control)
        except PaddingLimitError as exc:
            rows.append(dict(method=method, status="not_applicable", reason=str(exc)))
            continue
        reset, run = full_runner(work)
        print(f"full {method}: compiling ({time.perf_counter() - began:.1f}s)", flush=True)
        # Reset is compiled/synchronized separately and excluded from steady timing.
        start = time.perf_counter()
        jax.block_until_ready(reset())
        reset_cold_s = time.perf_counter() - start

        def trial():
            reset_start = time.perf_counter()
            jax.block_until_ready(reset())
            reset_times.append(time.perf_counter() - reset_start)
            start = time.perf_counter()
            result = jax.block_until_ready(run(steps))
            return time.perf_counter() - start, result

        reset_times = []
        first, result = trial()
        for _ in range(warmup):
            trial()
        samples = [trial()[0] for _ in range(repeat)]
        profile_status = None
        if profile_dir is not None:
            path = profile_dir / method
            path.mkdir(parents=True, exist_ok=True)
            jax.block_until_ready(reset())
            try:
                if hasattr(run, "lower"):
                    (path / "lowered.txt").write_text(str(run.lower(steps).compiler_ir()))
                with jax.profiler.trace(str(path)):
                    with jax.profiler.TraceAnnotation("braincell:full_scheduled_steady"):
                        jax.block_until_ready(run(steps))
                profile_status = "ok"
            except Exception as exc:
                profile_status = f"unavailable: {exc}"
        print(f"full {method}: validating ({time.perf_counter() - began:.1f}s)", flush=True)
        check_reset, check = full_runner(work, record=True)
        jax.block_until_ready(check_reset())
        voltage, g, drive = jax.device_get(check(steps)[2])
        checks = {}
        if validate_control:
            checks = _check_control(work, case, control, voltage, g, drive)
            jax.block_until_ready(reset())
            repeated = jax.device_get(run(steps))
            checks.update(_check_reset_result(repeated[:2], result[:2]))
            checks["reset_reproducible"] = True
        if refs is None:
            refs = voltage, g, drive
        errors = []
        for actual, expected, atol in ((voltage, refs[0], 1e-4), (g, refs[1], 1e-7), (drive, refs[2], 1e-7)):
            errors.append(float(np.max(np.abs(actual - expected), initial=0)))
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=atol)
        rows.append(
            dict(
                method=method,
                status="ok",
                build_s=work["build_seconds"],
                prepare_cold_s=work["prepare_seconds"] + work["plan_prepare_s"],
                reset_cold_s=reset_cold_s,
                array_bytes=work["array_bytes"],
                voltage_max_error_mV=errors[0],
                g_max_error=errors[1],
                drive_max_error=errors[2],
                profile_status=profile_status,
                control=control,
                validation=checks,
                reset_timing=summarize_times(reset_times[-repeat:]),
                timing={"full": summarize_times(samples) | dict(first_call_s=first)},
            )
        )
        print(f"full {method}: checked ({time.perf_counter() - began:.1f}s)", flush=True)
    return dict(
        status="ok",
        case=asdict(case),
        control=control,
        rows=rows,
        environment=_environment(),
        worker_wall_s=time.perf_counter() - began,
    )


def _check_reset_result(repeated, previous):
    """Allow float32 reduction rounding while rejecting unreinitialized state."""
    import numpy as np

    errors = {}
    for actual, expected, name, atol in zip(
        repeated,
        previous,
        ("reset_voltage_max_error_mV", "reset_g_max_error"),
        (1e-4, 1e-7),
    ):
        expected = np.asarray(expected)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=atol)
        errors[name] = float(np.max(np.abs(actual - expected), initial=0))
    return errors


def _check_control(work, case, control, voltage, g, drive):
    """Verify structure and actual arrivals outside all performance timers."""
    import brainunit as u
    import jax.numpy as jnp
    import numpy as np
    from braincell.network.event import _round_half_up_steps

    expected_synapses = 0 if control == "cell_only" else case.n * case.m
    expected_connections = case.n * case.m if control in ("active", "silent") else 0
    synapses = 0 if work["node"] is None else work["node"].g.value.size
    connections = work["net"].connections.n_rows
    assert synapses == expected_synapses, (synapses, expected_synapses)
    assert connections == expected_connections, (connections, expected_connections)
    assert all(np.isfinite(value).all() for value in (voltage, g, drive)), "nonfinite state/input"
    source = work["source"]
    arrivals, schedule_hash, source_shape = 0, None, None
    if source is not None:
        times = np.asarray(source.event_times.to_decimal(u.ms))
        mask = np.asarray(source._event_mask)
        source_shape = list(times.shape)
        assert source_shape == [case.n * case.m, case.k]
        schedule_hash = hashlib.sha256(times.tobytes() + mask.tobytes()).hexdigest()
        steps = np.asarray(
            _round_half_up_steps((jnp.asarray(times) + jnp.asarray(work["delay_ms"])[:, None]) / case.dt_ms)
        )
        selected = mask & (steps >= 0) & (steps < case.steps)
        arrivals = int(selected.sum())
        # One source/connection per independent target, all weights .001 uS.
        expected = np.zeros((case.steps, case.n * case.m), dtype=drive.dtype)
        src, ordinal = np.nonzero(selected)
        np.add.at(expected, (steps[src, ordinal].astype(int), src), 0.001)
        np.testing.assert_allclose(drive.reshape(expected.shape), expected, rtol=1e-6, atol=1e-9)
    if control != "active":
        assert arrivals == 0
        np.testing.assert_array_equal(g, np.zeros_like(g))
        np.testing.assert_array_equal(drive, np.zeros_like(drive))
    else:
        assert arrivals > 0 and np.max(g) > 0
    # The production clock repeatedly adds float32 dt. Validate the delivered
    # step, not exact equality to a host multiplication of duration.
    final_time = work["cell"].current_time.to_decimal(u.ms)
    final_step = int(_round_half_up_steps(final_time / case.dt_ms))
    assert final_step == case.steps
    assert abs(float(final_time) - case.duration_ms) < case.dt_ms / 2
    return dict(
        synapse_count=synapses,
        connection_rows=connections,
        arrivals=arrivals,
        source_shape=source_shape,
        schedule_sha256=schedule_hash,
        drive_sha256=hashlib.sha256(drive.tobytes()).hexdigest(),
        finite=True,
        final_time_ms=float(final_time),
        final_step=final_step,
    )


def main(argv=None):
    """Run bounded isolated workers and retain every status in a manifest.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments.

    Returns
    -------
    int
        Nonzero if any requested worker fails or the coordinator budget expires.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("micro", "full", "profile", "controls"), default="micro")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--devices", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; required for GPU coordinator runs")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts" / "delivery")
    parser.add_argument("--methods", help="defaults to direct for full, all candidates otherwise")
    parser.add_argument(
        "--trace-full", action="store_true", help="capture full-network traces; exclude from timing reports"
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--control-job", choices=tuple(CONTROL_JOBS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if not args.worker and args.devices != "cpu" and args.gpu is None:
        parser.error("--gpu is required when --devices includes gpu")
    if args.trace_full and args.phase != "full":
        parser.error("--trace-full requires --phase full")
    if args.methods is None:
        args.methods = (
            "direct" if args.phase == "full" else ("direct,padded" if args.phase == "controls" else ",".join(METHODS))
        )
    methods = tuple(args.methods.split(","))
    if not methods or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        parser.error("methods must be unique known names")
    if min(args.rounds, args.repeat, args.timeout, args.budget_seconds) <= 0 or args.warmup < 0:
        parser.error("rounds/repeat/timeout/budget must be positive")
    if args.phase == "controls" and methods != ("direct", "padded"):
        parser.error("controls fixes direct/padded pairs; select jobs with --cases")
    if args.worker and args.phase == "controls" and args.control_job is None:
        parser.error("controls worker requires --control-job")
    if args.worker:
        try:
            import brainstate
            import jax

            if jax.devices()[0].platform != args.devices:
                raise RuntimeError("requested backend was not selected")
            with brainstate.environ.context(precision=args.precision):
                case = QueryCase(**json.loads(args.worker))
                kwargs = dict(methods=methods, repeat=args.repeat, warmup=args.warmup, order=args.order)
                if args.phase == "controls":
                    control, candidates = CONTROL_JOBS[args.control_job]
                    kwargs["methods"] = candidates
                    result = measure_full(case, control=control, validate_control=True, **kwargs)
                elif args.phase == "full":
                    result = measure_full(
                        case, profile_dir=args.out.with_suffix("") if args.trace_full else None, **kwargs
                    )
                else:
                    result = measure_micro(
                        case, profile_dir=args.out.with_suffix("") if args.phase == "profile" else None, **kwargs
                    )
            _write_json(args.out, result)
            return 0
        except Exception:
            _write_json(args.out, dict(status="failed", error=traceback.format_exc()))
            traceback.print_exc()
            return 1
    cases = cases_for(args.phase)
    if args.cases != "all":
        names = args.cases.split(",")
        if any(n not in cases for n in names):
            parser.error("unknown case")
        cases = {n: cases[n] for n in names}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        parser.error("output already has a manifest; choose a fresh directory")
    files = [
        Path(__file__),
        ROOT / "braincell/experimental/scheduled_events.py",
        ROOT / "braincell/experimental/scheduled_delivery.py",
        ROOT / "braincell/experimental/scheduled_cell.py",
        ROOT / "benchmarks/performance/synapse_events/query_benchmark.py",
        ROOT / "benchmarks/performance/synapse_events/benchmark.py",
        ROOT / "braincell/_multi_compartment/cell.py",
        ROOT / "braincell/network/event.py",
        ROOT / "braincell/network/engine.py",
    ]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        target = args.out / "source_snapshot" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    devices = ("cpu", "gpu") if args.devices == "both" else (args.devices,)
    protocol = dict(
        phase=args.phase,
        trace_full=args.trace_full,
        order_start=args.order,
        cases=list(cases),
        devices=devices,
        methods=methods,
        rounds=args.rounds,
        repeat=args.repeat,
        warmup=args.warmup,
        precision=args.precision,
        timeout=args.timeout,
        gpu=args.gpu,
        budget_seconds=args.budget_seconds,
        sources=sources,
    )
    if args.phase == "controls":
        protocol["control_jobs"] = {
            name: dict(control=CONTROL_JOBS[name][0], methods=["production", *CONTROL_JOBS[name][1]]) for name in cases
        }
        protocol["case_configs"] = {name: asdict(case) for name, case in cases.items()}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol, rows=[])
    deadline = time.monotonic() + args.budget_seconds
    for round_id in range(args.rounds):
        for name, case in cases.items():
            for device in devices:
                path = args.out / f"{device}_{name}_r{round_id}.json"
                record = dict(file=path.name, device=device, case=name, round=round_id)
                if args.phase == "controls":
                    record.update(protocol["control_jobs"][name])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _write_json(path, dict(status="not_run", reason="coordinator budget exhausted"))
                else:
                    env = os.environ | dict(
                        JAX_PLATFORMS="cuda,cpu" if device == "gpu" else "cpu",
                        JAX_PLATFORM_NAME=device,
                        CUDA_VISIBLE_DEVICES=str(args.gpu) if device == "gpu" else "",
                        XLA_PYTHON_CLIENT_PREALLOCATE="false",
                        JAX_ENABLE_COMPILATION_CACHE="false",
                    )
                    cmd = [
                        args.python,
                        "-m",
                        __spec__.name,
                        "--worker",
                        json.dumps(asdict(case)),
                        "--phase",
                        args.phase,
                        "--devices",
                        device,
                        "--methods",
                        args.methods,
                        "--out",
                        str(path),
                        "--repeat",
                        str(args.repeat),
                        "--warmup",
                        str(args.warmup),
                        "--order",
                        str(round_id + args.order),
                        "--precision",
                        str(args.precision),
                    ]
                    if args.trace_full:
                        cmd.append("--trace-full")
                    if args.phase == "controls":
                        cmd.extend(["--control-job", name])
                    with path.with_suffix(".log").open("w") as log:
                        try:
                            result = subprocess.run(
                                cmd,
                                cwd=ROOT,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=min(args.timeout, remaining),
                            )
                            if result.returncode and not path.exists():
                                _write_json(path, dict(status="failed", error=f"exit {result.returncode}"))
                        except subprocess.TimeoutExpired:
                            _write_json(path, dict(status="timeout", reason="worker deadline exceeded"))
                record["status"] = json.loads(path.read_text())["status"]
                manifest["rows"].append(record)
                _write_json(manifest_path, manifest)
                print(f"{args.phase} round {round_id} {device} {name}: {record['status']}", flush=True)
    return int(any(r["status"] != "ok" for r in manifest["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
