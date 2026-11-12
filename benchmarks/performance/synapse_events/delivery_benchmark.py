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
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from benchmarks.performance.synapse_events.benchmark import DEFAULT_PYTHON, ROOT, _write_json, summarize_times
from benchmarks.performance.synapse_events.query_benchmark import QueryCase, build_source

METHODS = ("current", "scan", "bucket", "direct", "padded")
#: Full-model methods: the default path and the two ``scheduled_delivery`` values.
FULL_METHODS = ("production", "direct", "padded")
CONTROL_JOBS = {
    "cell_only": ("cell_only", ()),
    "unconnected": ("unconnected", ()),
    "silent_direct": ("silent", ("direct",)),
    "silent_padded": ("silent", ("padded",)),
    "active_direct": ("active", ("direct",)),
    "active_padded": ("active", ("padded",)),
}
#: Models of the regression phase: the ``full`` matrix and the active models.
REGRESSION_MODELS = ("small", "large", "long", "shared", "active", "mixed")
#: Methods of the regression phase: ``default`` is ``scheduled_delivery=None``.
REGRESSION_METHODS = ("default", "direct", "padded")
#: Entries of the regression phase: ``Network.run`` and ``prepare_run`` + ``update``.
ENTRIES = ("run", "update")


@dataclass(frozen=True)
class ActiveCase:
    """The multi-CV active model of the regression phase.

    ``n`` HH cables (one 200 um branch, radius 1 um, ``CVPerBranch(10)``,
    ``V_th`` 0 mV; Na_HH1952 120 and K_HH1952 36 mS/cm^2 with E_Na 50 and
    E_K -77 mV, IL 0.3 mS/cm^2 at -54.3 mV), each with ``m`` independent
    ExpSyn (tau 2 ms, e 0 mV) at x = 0.05 .. 0.95. One NetStim source per
    synapse: interval 10 ms, ``k`` events, start phases U[0, 10) ms
    (``brainstate.random.RandomState(7)``), seed 7; weight 0.001 uS, no
    delay. ``ring`` adds a live connection: the spike of cell ``i`` reaches
    the first 5 ExpSyn of cell ``i + 1`` (mod ``n``), delay 1 ms, weight
    0.003 uS.
    """

    ring: bool = False
    n: int = 20
    m: int = 50
    k: int = 11
    duration_ms: float = 100.0
    dt_ms: float = 0.025

    @property
    def steps(self):
        """Return the fixed rollout length."""
        return round(self.duration_ms / self.dt_ms)


def cases_for(phase):
    """Return the bounded six-case micro, four-case full-model or six-model regression matrix.

    Parameters
    ----------
    phase : str
        ``micro``, ``full``, ``profile``, ``regression`` or the paired
        ``controls`` jobs.

    Returns
    -------
    dict
        Stable descriptive names mapped to QueryCase objects (ActiveCase
        for the regression's ``active`` and ``mixed``).
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
    if phase == "regression":
        return cases_for("full") | {"active": ActiveCase(), "mixed": ActiveCase(ring=True)}
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


def padding_ratio(error):
    """Slots per record a ``PaddingLimitError`` reports, or ``None`` if its message names none."""
    import re

    match = re.search(r"needs \d+ bytes and ([0-9.eE+-]+) slots per record", str(error))
    return None if match is None else float(match.group(1))


def _plan_statistics(net):
    """Preparation time (host build + upload), bytes and layouts of the plan a Network built last."""
    plans = net._scheduled_plans
    if plans is None or not plans.preparation:
        return dict(plan_prepare_s=0.0, array_bytes=0, plan_layouts=[])
    prep = plans.preparation
    return dict(
        plan_prepare_s=prep["build_s"] + prep["upload_s"],
        array_bytes=int(prep["array_bytes"]),
        plan_layouts=prep["layouts"],
    )


def make_full_workload(case, method, *, control="active", prepare=True, recordings=()):
    """Build the passive Cell/Network, prepared for the default path or a ``scheduled_delivery`` method.

    Parameters
    ----------
    case : QueryCase
        N/M/K, layout and time grid.
    method : str
        ``production`` or ``default`` (``scheduled_delivery=None``), ``direct`` or ``padded``.
    control : str, optional
        ``active``, ``silent``, ``unconnected`` or ``cell_only``.
    prepare : bool, optional
        Call ``Network.prepare_run(scheduled_delivery=...)``, whose plan
        covers every event (default); otherwise only ``init_state``, for a
        workload driven by ``Network.run``.
    recordings : tuple of (str, object), optional
        ``Cell.record`` calls made before initialization.

    Returns
    -------
    dict
        Benchmark workload with the plan's preparation statistics.

    Raises
    ------
    braincell.network.scheduled.PaddingLimitError
        If ``padded`` exceeds its limits.
    """
    from benchmarks.performance.synapse_events.benchmark import Case, build_workload

    if method not in FULL_METHODS and method != "default":
        raise ValueError(f"full-model method must be one of {FULL_METHODS} or 'default'")
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
    work = build_workload(cfg, scheduled_delivery=_delivery(method), prepare=prepare, recordings=recordings)
    work.update(_plan_statistics(work["net"]))
    work["cfg"] = cfg
    return work


def _delivery(method):
    """``scheduled_delivery`` of a method name: ``None`` for the default path."""
    return None if method in ("production", "default") else method


def _capture_drive(work):
    """State holding the scheduled input the last step computed for the workload's layout.

    The Cell's ``_evaluate_contact_inputs`` is wrapped on this instance, so
    the recorded input is the one the step used: under ``scheduled_delivery``
    it comes from the plan, at the loop's integer step. The Network's compiled
    step keeps the trace it was first called with, so wrap a workload before
    it first runs.
    """
    import brainstate
    import brainunit as u
    import jax.numpy as jnp

    if "drive_state" in work:
        return work["drive_state"]
    cell, layout, node = work["cell"], work["layout"], work["node"]
    state = brainstate.ShortTermState(jnp.zeros(node.g.value.shape, dtype=cell.V.value.dtype))
    original = cell._evaluate_contact_inputs

    def evaluate(target, *, t, template, scheduled_only=True):
        drive = original(target, t=t, template=template, scheduled_only=scheduled_only)
        if scheduled_only and int(target.id) == int(layout.id):
            state.value = jnp.asarray(drive.to_decimal(u.uS), dtype=state.value.dtype)
        return drive

    object.__setattr__(cell, "_evaluate_contact_inputs", evaluate)
    work["drive_state"] = state
    return state


def full_runner(work, *, record=False):
    """Compile real Network steps, optionally returning all voltage/g/input/spike samples.

    Parameters
    ----------
    work : dict
        Prepared workload.
    record : bool
        Return the per-step trajectories ``(V, g, input, spike)`` during
        untimed validation; ``input`` is the scheduled input the step used.

    Returns
    -------
    tuple
        Reset and rollout callables.
    """
    import brainstate
    import brainunit as u
    import jax.numpy as jnp

    net, cell, node = work["net"], work["cell"], work["node"]
    drive = _capture_drive(work) if record and node is not None else None

    def conductance():
        return jnp.empty((0,), dtype=cell.V.value.dtype) if node is None else node.g.value.to_decimal(u.uS)

    @brainstate.transform.jit
    def reset():
        net.reset_state()
        return cell.V.value

    @brainstate.transform.jit
    def run(steps):
        def step(_):
            spikes = net.update()
            if record:
                value = conductance() if drive is None else drive.value
                spike = jnp.concatenate([jnp.ravel(spikes[name]) for name in sorted(spikes)])
                return cell.V.value.to_decimal(u.mV), conductance(), value, spike
            return None

        traces = brainstate.transform.for_loop(step, steps)
        return cell.V.value.to_decimal(u.mV), conductance(), traces

    return reset, run


def measure_full(
    case,
    *,
    methods=FULL_METHODS[1:],
    repeat=5,
    warmup=2,
    order=0,
    profile_dir=None,
    control="active",
    validate_control=False,
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
    from braincell.network.scheduled import PaddingLimitError

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
            rows.append(dict(method=method, status="not_applicable", reason=str(exc), padding_ratio=padding_ratio(exc)))
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
        # A fresh workload: the input capture must be in place before the step is first traced.
        check_work = make_full_workload(case, method, control=control)
        check_reset, check = full_runner(check_work, record=True)
        jax.block_until_ready(check_reset())
        voltage, g, drive, spike = jax.device_get(check(steps)[2])
        checks = {}
        if validate_control:
            checks = _check_control(check_work, case, control, voltage, g, drive)
            jax.block_until_ready(reset())
            repeated = jax.device_get(run(steps))
            checks.update(_check_reset_result(repeated[:2], result[:2]))
            checks["reset_reproducible"] = True
        if refs is None:
            refs = voltage, g, drive, spike
        errors = []
        for actual, expected, atol in ((voltage, refs[0], 1e-4), (g, refs[1], 1e-7), (drive, refs[2], 1e-7)):
            errors.append(float(np.max(np.abs(actual - expected), initial=0)))
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=atol)
        np.testing.assert_array_equal(spike, refs[3])
        rows.append(
            dict(
                method=method,
                status="ok",
                build_s=work["build_seconds"],
                # prepare_run, the plan included (plan_prepare_s of it).
                prepare_cold_s=work["prepare_seconds"],
                plan_prepare_s=work["plan_prepare_s"],
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
    import numpy as np
    from braincell.network.event import round_half_up_steps_host

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
        # Arrival steps are rounded at float64 on the host, as production rounds them (CELL-46).
        steps = round_half_up_steps_host((times + np.asarray(work["delay_ms"], dtype=np.float64)[:, None]) / case.dt_ms)
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
    # current_time is step·dt rounded once in the run precision (CELL-45);
    # check the step it names, not equality with a float64 duration.
    final_time = work["cell"].current_time.to_decimal(u.ms)
    final_step = int(round_half_up_steps_host(np.asarray(final_time, dtype=np.float64) / case.dt_ms))
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


def build_active_workload(case, *, scheduled_delivery=None, prepare=True, recordings=()):
    """Construct the active (or, with ``case.ring``, mixed) model of :class:`ActiveCase`.

    Parameters
    ----------
    case : ActiveCase
        Model configuration.
    scheduled_delivery : {None, "direct", "padded"}, optional
        Passed to ``Network.prepare_run``.
    prepare : bool, optional
        Call ``Network.prepare_run`` (default); otherwise only ``init_state``.
    recordings : tuple of (str, object), optional
        ``Cell.record`` calls made before initialization.

    Returns
    -------
    dict
        The keys of ``benchmark.build_workload``.
    """
    import braincell as bc
    import brainstate
    import brainunit as u
    import jax
    import numpy as np
    from braincell import mech
    from braincell.filter import AllRegion, LocsetMask

    start = time.perf_counter()
    branch = bc.Branch.from_lengths(lengths=[200.0] * u.um, radii=[1.0, 1.0] * u.um)
    cell = bc.Cell(
        bc.Morphology.from_root(branch),
        cv_policy=bc.CVPerBranch(10),
        pop_size=case.n,
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
    )
    cell.paint(
        AllRegion(),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    cell.place(
        LocsetMask.from_columns(np.zeros(case.m, dtype=int), np.linspace(0.05, 0.95, case.m)),
        mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV),
    )
    net = bc.Network("active", seed=7)
    post = net.add_population("post", cell)
    size = case.n * case.m
    source_start = time.perf_counter()
    # Draw the schedule at float32 on the CPU, as build_workload does, so that
    # every precision and device sees the same events.
    with jax.default_device(jax.devices("cpu")[0]), brainstate.environ.context(precision=32):
        phases = np.asarray(brainstate.random.RandomState(7).uniform(0.0, 10.0, size=size), dtype=np.float64)
        source = bc.NetStim(size=size, start=phases * u.ms, interval=10.0 * u.ms, number=case.k, seed=7)
    stim = net.add_population("stim", source)
    source_seconds = time.perf_counter() - source_start
    net.connect("drive", source=stim, synapse=post.synapses["exp"][np.arange(size)], weight=0.001 * u.uS)
    if case.ring:
        src = np.repeat(np.arange(case.n), 5)
        tgt = ((src + 1) % case.n) * case.m + np.tile(np.arange(5), case.n)
        net.connect(
            "ring",
            source=post.event_outputs["spike"][src],
            synapse=post.synapses["exp"][tgt],
            weight=0.003 * u.uS,
            delay=1.0 * u.ms,
        )
    for name, observable in recordings:
        cell.record(name, observable)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if prepare:
        net.prepare_run(dt=case.dt_ms * u.ms, event_backend="scatter", scheduled_delivery=scheduled_delivery)
    else:
        net.init_state()
    states = tuple(brainstate.graph.states(cell).values())
    jax.block_until_ready(tuple(s.value for s in states))
    prepare_seconds = time.perf_counter() - start
    layout = tuple(cell._event_layouts())[0]
    return dict(
        net=net,
        cell=cell,
        source=source,
        delay_ms=np.zeros(size),
        layout=layout,
        node=cell.runtime.get_runtime_node(layout.id),
        states=states,
        build_seconds=build_seconds,
        source_seconds=source_seconds,
        prepare_seconds=prepare_seconds,
    )


def build_regression_workload(name, method, *, prepare=True, case=None, recordings=()):
    """Build one regression model, prepared for ``method`` when ``prepare``.

    Parameters
    ----------
    name : str
        A model of ``cases_for("regression")``.
    method : str
        ``default``, ``direct`` or ``padded``.
    prepare : bool, optional
        Call ``Network.prepare_run(scheduled_delivery=...)`` (the ``update``
        entry); otherwise only ``init_state`` (the ``run`` entry).
    case : QueryCase or ActiveCase, optional
        Replaces the named configuration (tests use shorter windows).
    recordings : tuple of (str, object), optional
        ``Cell.record`` calls made before initialization.

    Returns
    -------
    dict
        The workload, with the plan's preparation statistics.
    """
    case = cases_for("regression")[name] if case is None else case
    if isinstance(case, ActiveCase):
        work = build_active_workload(case, scheduled_delivery=_delivery(method), prepare=prepare, recordings=recordings)
        work.update(_plan_statistics(work["net"]))
    else:
        work = make_full_workload(case, method, prepare=prepare, recordings=recordings)
    work["case"] = case
    return work


def _cpu_ticks():
    """``{cpu: (total, idle)}`` jiffies of every CPU from ``/proc/stat``; empty where unavailable."""
    ticks = {}
    try:
        with open("/proc/stat") as handle:
            for line in handle:
                name, *values = line.split()
                if name.startswith("cpu") and name[3:].isdigit():
                    values = [int(v) for v in values[:8]]
                    ticks[int(name[3:])] = (sum(values), values[3] + values[4])
    except OSError:
        pass
    return ticks


def _busy_share(before, after):
    """Busy fraction of each CPU between two ``_cpu_ticks`` samples."""
    share = {}
    for cpu, (total, idle) in after.items():
        if cpu in before and total > before[cpu][0]:
            share[str(cpu)] = 1.0 - (idle - before[cpu][1]) / (total - before[cpu][0])
    return share


#: A worker is re-run if, when it starts, its GPU is at least this busy (percent) ...
GPU_BUSY_UTILIZATION = 5.0
#: ... or another process holds at least this much of its memory (MiB). A process
#: whose JAX enumerated every GPU but computes on another one holds an idle context
#: of about 260 MiB; that is recorded but does not trigger a re-run.
GPU_BUSY_MIB = 1024.0


def _gpu_status(gpu):
    """Utilization, memory and compute processes (pid, MiB) of a physical GPU, from ``nvidia-smi``."""
    if gpu is None:
        return None
    try:
        query = ["nvidia-smi", f"--id={gpu}", "--format=csv,noheader,nounits"]
        used = subprocess.run(
            query + ["--query-gpu=utilization.gpu,memory.used"], capture_output=True, text=True, timeout=30
        ).stdout.split(",")
        apps = subprocess.run(
            query + ["--query-compute-apps=pid,used_memory"], capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
        apps = [[int(pid), float(mib)] for pid, mib in (line.split(",") for line in apps if line.strip())]
        return dict(
            utilization=float(used[0]),
            memory_mib=float(used[1]),
            compute_pids=[pid for pid, _ in apps],
            compute_apps=apps,
        )
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return dict(error=str(exc))


def _iqr_over_median(samples):
    import numpy as np

    q1, median, q3 = np.percentile(samples, [25, 50, 75])
    return float((q3 - q1) / median)


def measure_regression(name, method, entry, *, steps=(4000, 2000), warmup=2, repeat=10, check_path=None, case=None):
    """Time one model, method and entry in this process (one worker of the regression phase).

    Parameters
    ----------
    name : str
        A model of ``cases_for("regression")``.
    method : str
        ``default``, ``direct`` or ``padded``.
    entry : str
        ``run``: ``Network.run(duration=n * dt, scheduled_delivery=...)``
        after ``reset_state()``, whose plan is the window ``[0, n)``, built
        by the first call of each ``n`` and a cache hit afterwards.
        ``update``: ``prepare_run(scheduled_delivery=...)``, whose plan
        covers every event, then a compiled ``for_loop`` of ``n``
        ``update()`` calls after a compiled reset.
    steps : tuple of int, optional
        Step counts, in order; each has a first (compiling) call, then
        ``warmup`` and ``repeat`` calls. The resets are not timed.
    warmup, repeat : int, optional
        Untimed and timed calls after the first, per step count.
    check_path : pathlib.Path or None, optional
        Where to save, as ``.npz``, the final V and g and the spikes
        (step and source) of one more untimed call of ``steps[0]`` steps.
    case : QueryCase or ActiveCase, optional
        Replaces the named configuration.

    Returns
    -------
    dict
        ``status`` ``ok`` or ``not_applicable`` (``PaddingLimitError``, with
        its message, padding ratio and the time of the attempt), and per step
        count the plan, first call and timed samples.
    """
    import brainunit as u
    import jax
    import jax.numpy as jnp
    import numpy as np
    from braincell.network.scheduled import PaddingLimitError

    if method not in REGRESSION_METHODS or entry not in ENTRIES:
        raise ValueError("unknown regression method or entry")
    case = cases_for("regression")[name] if case is None else case
    began = time.perf_counter()
    result = dict(model=name, method=method, entry=entry, steps=list(steps), warmup=warmup, repeat=repeat)
    delivery = _delivery(method)
    try:
        work = build_regression_workload(name, method, prepare=entry == "update", case=case)
    except PaddingLimitError as exc:
        return result | dict(
            status="not_applicable",
            reason=str(exc),
            padding_ratio=padding_ratio(exc),
            attempt_s=time.perf_counter() - began,
        )
    net, cell, node, states = work["net"], work["cell"], work["node"], work["states"]
    result.update(build_s=work["build_seconds"], prepare_s=work["prepare_seconds"])
    if entry == "update":
        result.update(plan=_plan_statistics(net))
    dt = case.dt_ms * u.ms
    kwargs = {} if delivery is None else dict(scheduled_delivery=delivery)

    if entry == "run":

        def call(n):
            net.reset_state()
            jax.block_until_ready(tuple(s.value for s in states))
            start = time.perf_counter()
            out = net.run(dt=dt, duration=n * case.dt_ms * u.ms, **kwargs)
            jax.block_until_ready(tuple(s.value for s in states))
            return time.perf_counter() - start, out

    else:
        reset, rollout = full_runner(work)

        def call(n):
            ks = jnp.arange(n, dtype=jnp.int32)
            jax.block_until_ready(reset())
            start = time.perf_counter()
            jax.block_until_ready(rollout(ks))
            return time.perf_counter() - start, None

    timings = {}
    for n in steps:
        before = None if delivery is None or net._scheduled_plans is None else net._scheduled_plans.preparation
        try:
            first, _ = call(n)
        except PaddingLimitError as exc:
            return result | dict(
                status="not_applicable",
                reason=str(exc),
                padding_ratio=padding_ratio(exc),
                attempt_s=time.perf_counter() - began,
            )
        item = dict(first_call_s=first)
        if entry == "run" and delivery is not None and net._scheduled_plans.preparation is not before:
            # This call built the window's plan: its time is part of first_call_s.
            item["plan"] = _plan_statistics(net)
        for _ in range(warmup):
            call(n)
        samples = [call(n)[0] for _ in range(repeat)]
        item.update(summarize_times(samples), iqr_over_median=_iqr_over_median(samples))
        timings[str(n)] = item
    result["timing"] = timings
    if check_path is not None:
        n = steps[0]
        if entry == "run":
            _, out = call(n)
            spike = out.events.get("post", {}).get("spike") if hasattr(out.events, "get") else None
        else:
            _, _ = call(n)
            spike = None
        voltage = np.asarray(jax.device_get(cell.V.value.to_decimal(u.mV)))
        g = np.asarray(jax.device_get(node.g.value.to_decimal(u.uS)))
        if entry == "update" and isinstance(case, ActiveCase):
            spike_steps, spike_sources = _update_spikes(work, n)
        elif spike is not None:
            spike_steps = np.asarray(
                np.round(np.asarray(spike.time.to_decimal(u.ms), dtype=np.float64) / case.dt_ms), dtype=np.int64
            )
            spike_sources = np.asarray(spike.source_id, dtype=np.int64)
        else:
            spike_steps = spike_sources = np.zeros(0, dtype=np.int64)
        check_path = Path(check_path)
        check_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(check_path, V=voltage, g=g, spike_step=spike_steps, spike_source=spike_sources)
        result.update(check=check_path.name, spikes=int(spike_steps.size))
    return result | dict(status="ok", worker_s=time.perf_counter() - began)


def _update_spikes(work, n):
    """(step, source) of every spike of a compiled ``update()`` rollout of ``n`` steps from reset."""
    import brainstate
    import jax
    import jax.numpy as jnp
    import numpy as np

    net = work["net"]

    @brainstate.transform.jit
    def rollout():
        net.reset_state()
        return brainstate.transform.for_loop(lambda _: net.update()["post"], jnp.arange(n, dtype=jnp.int32))

    spikes = np.asarray(jax.device_get(rollout())).reshape(n, -1)
    step, source = np.nonzero(spikes)
    return step.astype(np.int64), source.astype(np.int64)


def _tolerances(dtype):
    """``(rtol, {quantity: atol})`` of the regression checks for a run precision."""
    import numpy as np

    if np.dtype(dtype) == np.float64:
        return 1e-12, dict(V=1e-12, g=1e-12, input=1e-12)
    return 3e-5, dict(V=1e-4, g=1e-7, input=1e-7)


def check_regression(name, *, methods=("direct", "padded"), case=None, chunk=500):
    """Compare each method with the default path step by step, on both entries.

    ``update`` entry: the default and each method's Network advance in
    lockstep, ``chunk`` steps at a time, and every step's V (mV), g (uS),
    scheduled input (uS) and spikes are compared. ``run`` entry: one
    ``Network.run`` of the whole window per Network, recording V on every
    step, and the final g and the spike events (step and source). The
    tolerances follow the run precision: float64 ``rtol = atol = 1e-12``;
    float32 ``rtol = 3e-5`` with ``atol`` 1e-4 mV for V and 1e-7 uS for g and
    the input. Spikes must be identical.

    Parameters
    ----------
    name : str
        A model of ``cases_for("regression")``.
    methods : tuple of str, optional
        Methods compared with the default path.
    case : QueryCase or ActiveCase, optional
        Replaces the named configuration.
    chunk : int, optional
        Steps per compiled call of the ``update`` comparison.

    Returns
    -------
    dict
        Per entry and method: ``ok`` with the largest absolute errors and
        the spike count, or ``not_applicable`` with the ``PaddingLimitError``
        padding ratio and message. Raises ``AssertionError`` on a mismatch.
    """
    import brainunit as u
    import jax
    import numpy as np
    from braincell.network.scheduled import PaddingLimitError

    case = cases_for("regression")[name] if case is None else case
    n_steps = case.steps
    if n_steps % chunk:
        raise ValueError("chunk must divide the window's steps")
    report = dict(model=name, steps=n_steps, update={}, run={})

    # update entry: lockstep chunks.
    works, runners = {}, {}
    for method in ("default", *methods):
        try:
            works[method] = build_regression_workload(name, method, case=case)
        except PaddingLimitError as exc:
            report["update"][method] = dict(status="not_applicable", padding_ratio=padding_ratio(exc), reason=str(exc))
            continue
        runners[method] = full_runner(works[method], record=True)
        jax.block_until_ready(runners[method][0]())
    dtype = np.asarray(works["default"]["cell"].V.value.to_decimal(u.mV)).dtype
    rtol, atol = _tolerances(dtype)
    errors = {m: dict(V=0.0, g=0.0, input=0.0) for m in runners if m != "default"}
    spikes = 0
    ks = np.arange(chunk, dtype=np.int32)
    for _ in range(n_steps // chunk):
        traces = {m: jax.device_get(run(ks)[2]) for m, (_, run) in runners.items()}
        reference = traces["default"]
        spikes += int(np.count_nonzero(reference[3]))
        for method in errors:
            for label, actual, expected in zip(("V", "g", "input"), traces[method][:3], reference[:3]):
                np.testing.assert_allclose(
                    actual, expected, rtol=rtol, atol=atol[label], err_msg=f"{name} update {method} {label}"
                )
                errors[method][label] = max(errors[method][label], float(np.max(np.abs(actual - expected), initial=0)))
            np.testing.assert_array_equal(traces[method][3], reference[3], err_msg=f"{name} update {method} spikes")
    for method, error in errors.items():
        report["update"][method] = dict(status="ok", max_abs_error=error, spikes=spikes)
    report["update"]["default"] = dict(status="ok", spikes=spikes)
    report["update_planned_records"] = {
        m: sum(layout["records"] for layout in w["plan_layouts"]) for m, w in works.items() if m != "default"
    }
    del works, runners, traces

    # run entry: one window [0, n_steps).
    results = {}
    for method in ("default", *methods):
        work = build_regression_workload(name, method, prepare=False, case=case, recordings=(("V", "V"),))
        net, node = work["net"], work["node"]
        kwargs = {} if _delivery(method) is None else dict(scheduled_delivery=_delivery(method))
        try:
            out = net.run(dt=case.dt_ms * u.ms, duration=case.duration_ms * u.ms, **kwargs)
        except PaddingLimitError as exc:
            report["run"][method] = dict(status="not_applicable", padding_ratio=padding_ratio(exc), reason=str(exc))
            continue
        spike = out.events["post"]["spike"]  # one event per cell and threshold crossing
        results[method] = dict(
            V=np.asarray(out.traces["post"]["V"].to_decimal(u.mV)),
            g=np.asarray(node.g.value.to_decimal(u.uS)),
            spike_time=np.asarray(spike.time.to_decimal(u.ms)),
            spike_source=np.asarray(spike.source_id),
        )
    reference = results["default"]
    per_cell = np.bincount(reference["spike_source"].astype(int), minlength=case.n)
    report["spikes_per_cell"] = [int(per_cell.min()), int(per_cell.max())]
    for method, got in results.items():
        if method == "default":
            report["run"][method] = dict(status="ok", spikes=int(reference["spike_source"].size))
            continue
        error = {}
        for label in ("V", "g"):  # V on every step, g at the end
            np.testing.assert_allclose(
                got[label], reference[label], rtol=rtol, atol=atol[label], err_msg=f"{name} run {method} {label}"
            )
            error[label] = float(np.max(np.abs(got[label] - reference[label]), initial=0))
        np.testing.assert_array_equal(got["spike_time"], reference["spike_time"], err_msg=f"{name} run {method} spikes")
        np.testing.assert_array_equal(got["spike_source"], reference["spike_source"])
        report["run"][method] = dict(status="ok", max_abs_error=error, spikes=int(got["spike_source"].size))
    return report


def _regression_worker_failed(data, *, max_iqr, max_load, gpu, sibling=None):
    """Why an accepted-looking worker must be re-run, or ``None``.

    ``sibling`` is ``None`` or ``(cpus, max_busy)``: re-run if any of these CPUs
    (the hyperthread siblings of the pinned ones) was busier than ``max_busy``
    over the worker's run.
    """
    if data.get("status") != "ok":
        return None
    reasons = []
    iqr = max(item["iqr_over_median"] for item in data["timing"].values())
    if iqr > max_iqr:
        reasons.append(f"IQR/median {iqr:.3f} > {max_iqr}")
    load = max(data.get("load_start", 0.0), data.get("load_end", 0.0))
    if load >= max_load:
        reasons.append(f"load {load:.1f} >= {max_load}")
    if sibling is not None:
        cpus, max_busy = sibling
        share = data.get("cpu_busy_share") or {}
        for cpu in cpus:
            busy = share.get(str(cpu))
            if busy is not None and busy > max_busy:
                reasons.append(f"CPU {cpu} {busy:.0%} busy > {max_busy:.0%}")
    status = data.get("gpu_start") or {}
    if gpu is not None:
        if status.get("utilization", 0.0) >= GPU_BUSY_UTILIZATION:
            reasons.append(f"GPU {status['utilization']:.0f}% busy at start")
        held = [mib for pid, mib in status.get("compute_apps", []) if pid != data.get("pid") and mib >= GPU_BUSY_MIB]
        if held:
            reasons.append(f"another process holds {max(held):.0f} MiB of the GPU")
    return "; ".join(reasons) or None


def _git(*args):
    """Output of a git command in the checkout, or ``None`` without git."""
    try:
        with subprocess.Popen(
            ["git", "-C", str(ROOT), *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        ) as process:
            return process.communicate()[0].strip()
    except OSError:
        return None


def run_regression(args, models):
    """Coordinate the regression phase: one fresh worker per (round, model, entry, method, device).

    The methods of each (round, model, entry) run back to back in a random
    order seeded by ``args.seed``; a worker whose IQR/median, load or GPU
    occupancy breaks the limits is re-run up to ``args.retries`` times.
    Every attempt is kept; the manifest names the accepted one.
    """
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        raise SystemExit("output already has a manifest; choose a fresh directory")
    files = [
        Path(__file__),
        ROOT / "benchmarks/performance/synapse_events/benchmark.py",
        ROOT / "braincell/network/scheduled.py",
        ROOT / "braincell/network/engine.py",
        ROOT / "braincell/network/event.py",
        ROOT / "braincell/_multi_compartment/cell.py",
    ]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    dirty = None if status is None else bool(status)
    devices = ("cpu", "gpu") if args.devices == "both" else (args.devices,)
    methods = tuple(args.methods.split(","))
    protocol = dict(
        phase="regression",
        cases=list(models),
        entries=list(args.entries),
        methods=list(methods),
        devices=devices,
        rounds=args.rounds,
        warmup=args.warmup,
        repeat=args.repeat,
        steps=list(args.steps),
        precision=args.precision,
        cpus=args.cpus,
        lock=args.lock,
        pause_s=args.pause,
        max_iqr=args.max_iqr,
        max_load=args.max_load,
        sibling_cpus=args.sibling_cpus,
        max_sibling_busy=args.max_sibling_busy,
        retries=args.retries,
        seed=args.seed,
        gpu=args.gpu,
        timeout=args.timeout,
        git_sha=sha,
        dirty=dirty,
        sources=sources,
    )
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol, rows=[])
    deadline = time.monotonic() + args.budget_seconds
    for round_id in range(args.rounds):
        for model in models:
            for entry in args.entries:
                order = list(methods)
                random.Random(f"{args.seed}:{round_id}:{model}:{entry}").shuffle(order)
                for method in order:
                    for device in devices:
                        record = dict(
                            device=device,
                            case=model,
                            entry=entry,
                            method=method,
                            round=round_id,
                            order=order,
                            attempts=[],
                        )
                        for attempt in range(args.retries + 1):
                            stem = f"{device}_{model}_{entry}_{method}_r{round_id}_a{attempt}"
                            path = out / f"{stem}.json"
                            if deadline - time.monotonic() <= 0:
                                _write_json(path, dict(status="not_run", reason="coordinator budget exhausted"))
                            else:
                                _launch_regression_worker(args, device, model, entry, method, out, stem)
                            data = json.loads(path.read_text())
                            rerun = _regression_worker_failed(
                                data,
                                max_iqr=args.max_iqr,
                                max_load=args.max_load,
                                gpu=args.gpu if device == "gpu" else None,
                                sibling=None
                                if args.max_sibling_busy is None
                                else (args.sibling_cpus, args.max_sibling_busy),
                            )
                            record["attempts"].append(dict(file=path.name, status=data["status"], rerun=rerun))
                            record.update(file=path.name, status=data["status"], flagged=rerun)
                            print(
                                f"regression round {round_id} {device} {model} {entry} {method} attempt {attempt}: "
                                f"{data['status']}{'' if rerun is None else ' (' + rerun + ')'}",
                                flush=True,
                            )
                            if args.pause:
                                time.sleep(args.pause)
                            if rerun is None or data["status"] != "ok":
                                break
                        manifest["rows"].append(record)
                        _write_json(manifest_path, manifest)
    return int(any(r["status"] not in ("ok", "not_applicable") for r in manifest["rows"]))


def _launch_regression_worker(args, device, model, entry, method, out, stem):
    """Run one regression worker under the optional lock and CPU pinning."""
    path = out / f"{stem}.json"
    env = os.environ | dict(
        JAX_PLATFORMS="cuda,cpu" if device == "gpu" else "cpu",
        JAX_PLATFORM_NAME=device,
        CUDA_VISIBLE_DEVICES=str(args.gpu) if device == "gpu" else "",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_ENABLE_COMPILATION_CACHE="false",
    )
    if args.single_thread:
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        ).strip()
    cmd = [
        args.python,
        "-m",
        __spec__.name,
        "--phase",
        "regression",
        "--worker",
        model,
        "--method",
        method,
        "--entry",
        entry,
        "--devices",
        device,
        "--out",
        str(path),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--steps",
        ",".join(map(str, args.steps)),
        "--precision",
        str(args.precision),
    ]
    if device == "gpu":
        cmd += ["--gpu", str(args.gpu)]
    if args.cpus:
        cmd = ["taskset", "-c", args.cpus] + cmd
    timeout = None
    if args.lock:
        # The worker's timeout starts once it holds the lock.
        cmd = ["flock", args.lock, "timeout", str(int(args.timeout))] + cmd
    else:
        timeout = args.timeout
    with path.with_suffix(".log").open("w") as log:
        try:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if not path.exists():
                _write_json(
                    path,
                    dict(status="timeout" if result.returncode == 124 else "failed", error=f"exit {result.returncode}"),
                )
        except subprocess.TimeoutExpired:
            _write_json(path, dict(status="timeout", reason="worker deadline exceeded"))


def regression_worker(args):
    """One regression worker: measure, add the environment, write ``args.out``."""
    import resource

    ticks = _cpu_ticks()
    load_start = os.getloadavg()[0]
    gpu_start = _gpu_status(args.gpu) if args.devices == "gpu" else None
    import brainstate
    import jax

    import braincell

    print("braincell", braincell.__file__, flush=True)
    if jax.devices()[0].platform != args.devices:
        raise RuntimeError("requested backend was not selected")
    with brainstate.environ.context(precision=args.precision):
        data = measure_regression(
            args.worker,
            args.method,
            args.entry,
            steps=args.steps,
            warmup=args.warmup,
            repeat=args.repeat,
            check_path=args.out.with_suffix(".npz"),
        )
    data.update(
        braincell=braincell.__file__,
        pid=os.getpid(),
        precision=args.precision,
        load_start=load_start,
        load_end=os.getloadavg()[0],
        cpu_busy_share=_busy_share(ticks, _cpu_ticks()),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        gpu_start=gpu_start,
        gpu_end=_gpu_status(args.gpu) if args.devices == "gpu" else None,
        environment=_environment(),
    )
    _write_json(args.out, data)
    return 0


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
    parser.add_argument("--phase", choices=("micro", "full", "profile", "controls", "regression"), default="micro")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--devices", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; required for GPU coordinator runs")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts" / "delivery")
    parser.add_argument(
        "--methods", help="defaults to direct for full, default,direct,padded for regression, all candidates otherwise"
    )
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
    regression = parser.add_argument_group("regression phase")
    regression.add_argument("--entries", default="run,update", help="run and/or update")
    regression.add_argument("--steps", default="4000,2000", help="step counts per worker, in order")
    regression.add_argument("--lock", help="hold this flock file for each worker")
    regression.add_argument("--cpus", help="pin each worker with taskset -c")
    regression.add_argument("--single-thread", action="store_true", help="single-threaded XLA CPU")
    regression.add_argument("--pause", type=float, default=3.0, help="seconds to wait after each worker")
    regression.add_argument("--max-iqr", type=float, default=0.1, help="re-run a worker above this IQR/median")
    regression.add_argument("--max-load", type=float, default=16.0, help="re-run a worker at this 1-min load")
    regression.add_argument(
        "--sibling-cpus", help="CPUs (the hyperthread siblings of --cpus) whose busy share --max-sibling-busy bounds"
    )
    regression.add_argument(
        "--max-sibling-busy", type=float, help="re-run a worker if a --sibling-cpus CPU was busier (0-1) over its run"
    )
    regression.add_argument("--retries", type=int, default=2, help="re-runs per worker at most")
    regression.add_argument("--seed", type=int, default=0, help="seed of the method order")
    regression.add_argument("--method", choices=REGRESSION_METHODS, help=argparse.SUPPRESS)
    regression.add_argument("--entry", choices=ENTRIES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if not args.worker and args.devices != "cpu" and args.gpu is None:
        parser.error("--gpu is required when --devices includes gpu")
    if args.trace_full and args.phase != "full":
        parser.error("--trace-full requires --phase full")
    if args.methods is None:
        args.methods = {"full": "direct", "controls": "direct,padded", "regression": ",".join(REGRESSION_METHODS)}.get(
            args.phase, ",".join(METHODS)
        )
    methods = tuple(args.methods.split(","))
    known = {"full": FULL_METHODS[1:], "controls": FULL_METHODS[1:], "regression": REGRESSION_METHODS}
    if (
        not methods
        or any(m not in known.get(args.phase, METHODS) for m in methods)
        or len(set(methods)) != len(methods)
    ):
        parser.error("methods must be unique known names")
    if min(args.rounds, args.repeat, args.timeout, args.budget_seconds) <= 0 or args.warmup < 0:
        parser.error("rounds/repeat/timeout/budget must be positive")
    if args.phase == "regression":
        try:
            args.steps = tuple(int(n) for n in args.steps.split(","))
            if args.sibling_cpus is not None:
                args.sibling_cpus = tuple(int(c) for c in args.sibling_cpus.split(","))
        except ValueError:
            parser.error("--steps and --sibling-cpus must be comma-separated integers")
        if (args.sibling_cpus is None) != (args.max_sibling_busy is None) or not (
            args.max_sibling_busy is None or 0 <= args.max_sibling_busy < 1
        ):
            parser.error("--sibling-cpus and --max-sibling-busy go together, with a busy share in [0, 1)")
        args.entries = tuple(args.entries.split(","))
        if not args.steps or min(args.steps) < 1 or len(set(args.steps)) != len(args.steps):
            parser.error("--steps must be distinct positive step counts")
        if (
            not args.entries
            or any(e not in ENTRIES for e in args.entries)
            or len(set(args.entries)) != len(args.entries)
        ):
            parser.error("--entries must be unique names of run, update")
        if args.retries < 0 or args.pause < 0:
            parser.error("--retries and --pause must be nonnegative")
        if args.worker:
            if args.method is None or args.entry is None:
                parser.error("a regression worker needs --method and --entry")
            if args.worker not in cases_for("regression"):
                parser.error("unknown regression model")
            try:
                return regression_worker(args)
            except Exception:
                _write_json(args.out, dict(status="failed", error=traceback.format_exc()))
                traceback.print_exc()
                return 1
        cases = cases_for("regression")
        names = list(cases) if args.cases == "all" else args.cases.split(",")
        if any(n not in cases for n in names):
            parser.error("unknown case")
        return run_regression(args, names)
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
