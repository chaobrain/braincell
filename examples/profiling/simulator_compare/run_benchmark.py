"""Orchestrate the three isolated benchmark backends."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import (
    HERE,
    MORPHOLOGY_SHA256,
    assert_morphology_asset,
    compare_traces,
    extrapolate_timing,
    query_gpus,
    read_json,
    write_json,
)

DEFAULT_JAX_PYTHON = Path("/home/swl/anaconda3/envs/braincell_311/bin/python")
DEFAULT_NEURON_PYTHON = Path("/home/swl/anaconda3/envs/neuron/bin/python")
GPU_BACKENDS = ("braincell", "jaxley")


def parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def run_child(
    backend: str,
    *,
    python: Path,
    batch_size: int,
    warmup: int,
    repeat: int,
    output: Path,
    physical_gpu: int | None,
    include_trace: bool,
    transfer_repeat: int,
    braincell_linearizer: str,
) -> dict:
    command = [
        str(python),
        str(HERE / f"backend_{backend}.py"),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--output",
        str(output),
    ]
    if include_trace:
        command.append("--include-trace")
    if backend in GPU_BACKENDS:
        command.extend(["--transfer-repeat", str(transfer_repeat)])
    if backend == "braincell":
        command.extend(["--linearizer", braincell_linearizer])
    env = os.environ.copy()
    if physical_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    print(f"[{backend}] N={batch_size}: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=HERE, env=env)
    return read_json(output)


def validate_morphologies(runs: list[dict]) -> dict:
    measured = [run for run in runs if run["timing"]["measured"]]
    expected = {(217, 868)}
    observed = {(run["morphology"]["n_branches"], run["morphology"]["n_cv"]) for run in measured}
    if observed != expected:
        raise RuntimeError(f"backend morphology/discretization mismatch: {observed}")
    representative = {}
    for run in measured:
        representative.setdefault(run["backend"], run["morphology"])
    areas = {backend: morph["total_area_um2"] for backend, morph in representative.items()}
    lengths = {backend: morph["total_length_um"] for backend, morph in representative.items()}
    if max(areas.values()) - min(areas.values()) > max(areas.values()) * 1e-6:
        raise RuntimeError(f"backend morphology area mismatch: {areas}")
    if max(lengths.values()) - min(lengths.values()) > 1e-5:
        raise RuntimeError(f"backend morphology length mismatch: {lengths}")
    return {"counts_match": True, "geometry_match": True, "areas_um2": areas, "lengths_um": lengths}


def project_neuron_run(measured_run: dict, batch_sizes: tuple[int, ...]) -> list[dict]:
    """Keep N=10 measured and linearly project larger NEURON sizes."""
    measured_size = 10
    if measured_run["backend"] != "neuron" or measured_run["batch_size"] != measured_size:
        raise ValueError("NEURON projection source must be the measured N=10 run")
    projected_runs = []
    for size in batch_sizes:
        if size < measured_size or size % measured_size:
            raise ValueError("NEURON timing sizes must be 10 or integer multiples of 10")
        if size == measured_size:
            projected_runs.append(measured_run)
            continue
        projected = {**measured_run, "batch_size": size}
        projected["timing"] = extrapolate_timing(
            measured_run["timing"], size // measured_size, source_size=measured_size
        )
        projected_runs.append(projected)
    return projected_runs


def validate_gpu_environment(runs: list[dict]) -> dict | None:
    gpu_runs = [run for run in runs if run["backend"] in GPU_BACKENDS and run["timing"]["measured"]]
    if not gpu_runs:
        return None
    if any(not str(run["device"]).startswith("cuda") for run in gpu_runs):
        raise RuntimeError("BrainCell and Jaxley timing runs must use CUDA")
    versions = {
        (run["software"]["python"], run["software"]["jax"], run["software"]["jaxlib"])
        for run in gpu_runs
    }
    if len(versions) != 1:
        raise RuntimeError(f"GPU backends did not use one Python/JAX stack: {versions}")
    python_version, jax_version, jaxlib_version = versions.pop()
    return {"python": python_version, "jax": jax_version, "jaxlib": jaxlib_version, "logical_device": "cuda:0"}


def scaling_analysis(runs: list[dict], *, min_batch_size: int = 100) -> dict:
    fits = {}
    for backend in GPU_BACKENDS:
        points = sorted(
            (
                (float(run["batch_size"]), float(run["timing"]["median_seconds"]))
                for run in runs
                if run["backend"] == backend
                and run["timing"]["measured"]
                and run["batch_size"] >= min_batch_size
            ),
        )
        if len(points) < 2:
            continue
        x_mean = sum(x for x, _ in points) / len(points)
        y_mean = sum(y for _, y in points) / len(points)
        denominator = sum((x - x_mean) ** 2 for x, _ in points)
        slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
        intercept = y_mean - slope * x_mean
        residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
        total = sum((y - y_mean) ** 2 for _, y in points)
        fits[backend] = {
            "batch_sizes": [int(x) for x, _ in points],
            "intercept_seconds": intercept,
            "seconds_per_cell": slope,
            "r_squared": 1.0 - residual / total if total else 1.0,
        }
    crossover = None
    if set(fits) == set(GPU_BACKENDS):
        braincell = fits["braincell"]
        jaxley = fits["jaxley"]
        slope_delta = jaxley["seconds_per_cell"] - braincell["seconds_per_cell"]
        if slope_delta:
            crossover = (
                braincell["intercept_seconds"] - jaxley["intercept_seconds"]
            ) / slope_delta
    return {
        "model": "median_seconds = intercept_seconds + seconds_per_cell * batch_size",
        "minimum_batch_size": min_batch_size,
        "fits": fits,
        "crossover_batch_size": crossover,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", choices=(*GPU_BACKENDS, "neuron"), dest="backends")
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=(10, 100, 1000, 10000))
    parser.add_argument("--gpu-candidates", type=parse_csv_ints, default=(2, 3))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--neuron-warmup", type=int, default=1)
    parser.add_argument("--neuron-repeat", type=int, default=10)
    parser.add_argument("--transfer-repeat", type=int, default=3)
    parser.add_argument("--braincell-linearizer", choices=("point", "generic"), default="point")
    parser.add_argument("--jax-python", type=Path, default=DEFAULT_JAX_PYTHON)
    parser.add_argument("--neuron-python", type=Path, default=DEFAULT_NEURON_PYTHON)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "benchmark.json")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.jax_python = args.jax_python.resolve()
    args.neuron_python = args.neuron_python.resolve()
    backends = tuple(dict.fromkeys(args.backends or (*GPU_BACKENDS, "neuron")))
    if (
        args.warmup < 0
        or args.repeat <= 0
        or args.neuron_warmup < 0
        or args.neuron_repeat <= 0
        or args.transfer_repeat < 0
    ):
        parser.error("repeat values must be positive and warmup/transfer-repeat values non-negative")
    if any(gpu not in (2, 3) for gpu in args.gpu_candidates):
        parser.error("GPU candidates are restricted to physical devices 2 and 3")
    for backend, python in (("JAX", args.jax_python), ("NEURON", args.neuron_python)):
        if not python.exists():
            parser.error(f"{backend} interpreter does not exist: {python}")

    assert_morphology_asset()
    gpu_selection = None
    physical_gpu = None
    if any(backend in GPU_BACKENDS for backend in backends):
        gpu_selection = query_gpus(args.gpu_candidates)
        physical_gpu = int(gpu_selection["selected"]["physical_id"])
        print(f"Selected physical GPU {physical_gpu} ({gpu_selection['selected']['uuid']})", flush=True)

    raw_dir = args.output.parent / "raw"
    runs: list[dict] = []
    accuracy_traces: dict[str, list[list[float]]] = {}
    if not args.skip_accuracy:
        for backend in backends:
            run = run_child(
                backend,
                python=args.jax_python if backend in GPU_BACKENDS else args.neuron_python,
                batch_size=1,
                warmup=0,
                repeat=1,
                output=raw_dir / f"accuracy_{backend}.json",
                physical_gpu=physical_gpu if backend in GPU_BACKENDS else None,
                include_trace=True,
                transfer_repeat=0,
                braincell_linearizer=args.braincell_linearizer,
            )
            accuracy_traces[backend] = run.pop("trace_mv")

    for backend in backends:
        if backend == "neuron":
            measured_size = 10
            run = run_child(
                backend,
                python=args.neuron_python,
                batch_size=measured_size,
                warmup=args.neuron_warmup,
                repeat=args.neuron_repeat,
                output=raw_dir / "timing_neuron_n10.json",
                physical_gpu=None,
                include_trace=False,
                transfer_repeat=0,
                braincell_linearizer=args.braincell_linearizer,
            )
            runs.extend(project_neuron_run(run, args.batch_sizes))
            continue
        for size in args.batch_sizes:
            runs.append(
                run_child(
                    backend,
                    python=args.jax_python,
                    batch_size=size,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    output=raw_dir / f"timing_{backend}_n{size}.json",
                    physical_gpu=physical_gpu,
                    include_trace=False,
                    transfer_repeat=args.transfer_repeat,
                    braincell_linearizer=args.braincell_linearizer,
                )
            )

    accuracy = None
    if not args.skip_accuracy:
        if set(accuracy_traces) == {"braincell", "jaxley", "neuron"}:
            accuracy = compare_traces(accuracy_traces)
        else:
            accuracy = {"passed": None, "reason": "all three backends are required", "available": sorted(accuracy_traces)}
    morphology = validate_morphologies(runs)
    gpu_environment = validate_gpu_environment(runs)
    scaling = scaling_analysis(runs)
    payload = {
        "schema_version": 2,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "platform": platform.platform()},
        "morphology_sha256": MORPHOLOGY_SHA256,
        "gpu_selection": gpu_selection,
        "requested_batch_sizes": list(args.batch_sizes),
        "braincell_linearizer": args.braincell_linearizer,
        "runs": sorted(runs, key=lambda run: (run["backend"], run["batch_size"])),
        "accuracy": accuracy,
        "morphology_validation": morphology,
        "gpu_environment_validation": gpu_environment,
        "scaling_analysis": scaling,
    }
    write_json(args.output, payload)
    print(f"Wrote {args.output}", flush=True)
    if accuracy is not None and accuracy.get("passed") is False:
        print("Accuracy gate failed; performance plot was not generated.", file=sys.stderr)
        raise SystemExit(2)
    if not args.no_plot:
        plot_output = args.output.with_suffix(".png")
        subprocess.run(
            [str(args.jax_python), str(HERE / "plot_results.py"), str(args.output), "--output", str(plot_output)],
            check=True,
            cwd=HERE,
        )
        if all(any(run["backend"] == backend for run in runs) for backend in GPU_BACKENDS):
            subprocess.run(
                [
                    str(args.jax_python),
                    str(HERE / "plot_diagnostics.py"),
                    str(args.output),
                    "--output-prefix",
                    str(args.output.with_suffix("")),
                ],
                check=True,
                cwd=HERE,
            )


if __name__ == "__main__":
    main()
