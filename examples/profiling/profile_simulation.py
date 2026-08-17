#!/usr/bin/env python3
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

"""Profile BrainCell example simulations without changing public APIs."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Any, Callable


REPO_ROOT = next(
    (
        candidate
        for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents)
        if (candidate / "braincell").exists() and (candidate / "examples").exists()
    ),
    Path(__file__).resolve().parents[2],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CASES = {
    "neuron_compare_cell": "examples.profiling.cases.neuron_compare_cell",
    "cerebellar_probability_network": "examples.profiling.cases.cerebellar_probability_network",
}


def main(argv: list[str] | None = None) -> int:
    """Run the profiling command."""
    args = _parse_args(argv)
    if args.platform != "auto":
        os.environ.setdefault("JAX_PLATFORMS", args.platform)

    case_module = importlib.import_module(CASES[args.case])
    workload = case_module.create_workload(args)

    import jax

    tracemalloc.start()
    rows: list[dict[str, Any]] = []
    memory_profiles: list[str] = []
    materialized = None

    with _maybe_jax_trace(
        jax,
        args.trace_dir if args.trace_phase == "all" else None,
        device_only=args.trace_device_only,
    ):
        if hasattr(workload, "build_phases"):
            for phase_name, phase_fn in workload.build_phases():
                _measure_phase(rows, phase_name, phase_fn)
        else:
            _measure_phase(rows, "build", workload.build)
        _measure_phase(rows, "init_reset", workload.init_reset)

        for index in range(args.warmup):
            workload.reset_for_run()
            result = _measure_phase(
                rows,
                "warmup_run",
                workload.run,
                block=workload.block,
                iteration=index,
            )
            if args.device_memory_profile:
                memory_profiles.append(
                    _save_device_memory_profile(
                        jax,
                        args.device_memory_profile,
                        label=f"warmup_{index}",
                    )
                )

        steady_results = []
        with _maybe_jax_trace(
            jax,
            args.trace_dir if args.trace_phase == "steady" else None,
            device_only=args.trace_device_only,
        ):
            for index in range(args.repeat):
                workload.reset_for_run()
                result = _measure_phase(
                    rows,
                    "steady_run",
                    workload.run,
                    block=workload.block,
                    iteration=index,
                )
                steady_results.append(result)

        if steady_results:
            materialized = _measure_phase(
                rows,
                "materialize",
                lambda: workload.materialize(steady_results[-1]),
            )
            if args.device_memory_profile:
                memory_profiles.append(
                    _save_device_memory_profile(
                        jax,
                        args.device_memory_profile,
                        label="steady_final",
                    )
                )

    output = {
        "case": args.case,
        "metadata": workload.metadata(),
        "environment": _environment_metadata(jax),
        "phases": rows,
        "summary": _summarize(rows),
        "materialized": materialized,
        "memory_profiles": memory_profiles,
    }

    _print_summary(output)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {out_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--case", choices=sorted(CASES), default="neuron_compare_cell")
    known, _ = base.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case", choices=sorted(CASES), default=known.case)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dt-ms", type=float, default=None)
    parser.add_argument("--duration-ms", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--device-memory-profile", default=None)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu", "cuda", "tpu"), default="auto")
    parser.add_argument(
        "--trace-phase",
        choices=("all", "steady"),
        default="all",
        help="Profiler trace coverage. 'steady' traces only JIT-warmed steady runs.",
    )
    parser.add_argument(
        "--trace-device-only",
        action="store_true",
        help="Reduce Python trace noise and keep HLO/device metadata when tracing.",
    )

    case_module = importlib.import_module(CASES[known.case])
    case_module.add_case_args(parser)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1.")
    return args


def _measure_phase(
    rows: list[dict[str, Any]],
    phase: str,
    fn: Callable[[], Any],
    *,
    block: Callable[[Any], None] | None = None,
    iteration: int | None = None,
) -> Any:
    import jax

    rss_before = _rss_bytes()
    current_before, peak_before = tracemalloc.get_traced_memory()
    start = time.perf_counter()
    with jax.profiler.TraceAnnotation(_phase_label(phase, iteration)):
        value = fn()
        if block is not None:
            block(value)
    elapsed_s = time.perf_counter() - start
    current_after, peak_after = tracemalloc.get_traced_memory()
    rss_after = _rss_bytes()
    row = {
        "phase": phase,
        "iteration": iteration,
        "wall_time_s": elapsed_s,
        "tracemalloc_current_before_bytes": current_before,
        "tracemalloc_current_after_bytes": current_after,
        "tracemalloc_peak_before_bytes": peak_before,
        "tracemalloc_peak_after_bytes": peak_after,
    }
    if rss_before is not None and rss_after is not None:
        row.update({"rss_before_bytes": rss_before, "rss_after_bytes": rss_after})
    rows.append(row)
    return value


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in sorted({row["phase"] for row in rows}):
        values = [row["wall_time_s"] for row in rows if row["phase"] == phase]
        item = {
            "count": len(values),
            "min_s": min(values),
            "max_s": max(values),
            "mean_s": statistics.fmean(values),
        }
        if len(values) > 1:
            item["median_s"] = statistics.median(values)
            item["stdev_s"] = statistics.stdev(values)
        summary[phase] = item
    return summary


def _print_summary(output: dict[str, Any]) -> None:
    env = output["environment"]
    print(f"case: {output['case']}")
    print(f"backend: {env.get('default_backend')} devices: {env.get('devices')}")
    print("\nphase                         iter    wall_time_s")
    print("-" * 54)
    for row in output["phases"]:
        iteration = "" if row["iteration"] is None else str(row["iteration"])
        print(f"{row['phase']:<30} {iteration:>4} {row['wall_time_s']:>14.6f}")


def _environment_metadata(jax) -> dict[str, Any]:
    try:
        default_backend = jax.default_backend()
        devices = [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover - defensive path
        default_backend = None
        devices = [f"device query failed: {type(exc).__name__}: {exc}"]
    return {
        "default_backend": default_backend,
        "devices": devices,
        "jax_version": getattr(jax, "__version__", None),
        "python": sys.version,
    }


@contextlib.contextmanager
def _maybe_jax_trace(jax, trace_dir: str | None, *, device_only: bool = False):
    if not trace_dir:
        yield
        return
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(
        str(path),
        create_perfetto_trace=True,
        profiler_options=_profile_options(jax, device_only=device_only),
    ):
        yield


def _profile_options(jax, *, device_only: bool):
    """Return JAX profiler options for the requested trace mode."""
    if not device_only:
        return None
    options = jax.profiler.ProfileOptions()
    if hasattr(options, "python_tracer_level"):
        options.python_tracer_level = 0
    if hasattr(options, "host_tracer_level"):
        options.host_tracer_level = 2
    if hasattr(options, "enable_hlo_proto"):
        options.enable_hlo_proto = True
    return options


def _save_device_memory_profile(jax, base_path: str, *, label: str) -> str:
    path = Path(base_path)
    if path.suffix:
        out = path.with_name(f"{path.stem}_{label}{path.suffix}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{label}.prof"
    out.parent.mkdir(parents=True, exist_ok=True)
    jax.profiler.save_device_memory_profile(str(out))
    return str(out)


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return int(psutil.Process().memory_info().rss)


def _phase_label(phase: str, iteration: int | None) -> str:
    if iteration is None:
        return phase
    return f"{phase}_{iteration}"


if __name__ == "__main__":
    raise SystemExit(main())
