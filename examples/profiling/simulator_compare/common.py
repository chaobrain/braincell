"""Shared configuration and validation for the simulator comparison."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
MORPHOLOGY_PATH = HERE / "n144.swc"
MORPHOLOGY_SHA256 = "68af99cf829e22005b2177b3d2ff4316bdaec2fcf0f1e4b85f81cdd2a8c01069"

DT_MS = 0.025
TSTOP_MS = 20.0
N_STEPS = 800
STIM_START_STEP = 120
STIM_END_STEP = 200
STIM_DELAY_MS = 3.0
STIM_DURATION_MS = 2.0
V_INIT_MV = -62.0
TEMPERATURE_C = 6.3
N_CV_PER_BRANCH = 4
N_BRANCHES = 217
N_CV = 868
TOTAL_LENGTH_UM = 14438.39058205983
PROBE_BRANCHES = (0, 34, 54)
PROBE_X = 0.85
# Full branch lengths, used to map Jaxley branch ids onto NEURON sections.
PROBE_BRANCH_LENGTHS_UM = (22.18148613, 38.620829739, 5.313313249)

ACCURACY_LIMITS = {
    "max_spike_time_delta_ms": 0.5,
    "max_spike_amplitude_delta_mv": 3.0,
    "max_rmse_mv": 2.0,
}


def morphology_sha256(path: Path = MORPHOLOGY_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_amplitudes(batch_size: int) -> list[float]:
    """Return deterministic, non-identical amplitudes in nA."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size == 1:
        return [0.7]
    return [0.69 + 0.02 * index / (batch_size - 1) for index in range(batch_size)]


def timing_summary(samples_seconds: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in samples_seconds]
    if not samples:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "q1_seconds": _quantile(ordered, 0.25),
        "q3_seconds": _quantile(ordered, 0.75),
        "measured": True,
    }


def add_throughput(timing: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
    result = dict(timing)
    result["cell_steps_per_second"] = batch_size * N_STEPS / result["median_seconds"]
    return result


def extrapolate_timing(timing: dict[str, Any], factor: int, *, source_size: int) -> dict[str, Any]:
    if factor <= 0:
        raise ValueError("extrapolation factor must be positive")
    result = {
        key: ([float(v) * factor for v in value] if key == "samples_seconds" else float(value) * factor)
        for key, value in timing.items()
        if key in {"samples_seconds", "median_seconds", "q1_seconds", "q3_seconds"}
    }
    result.update({"measured": False, "extrapolated_from": source_size, "scale_factor": factor})
    if "cell_steps_per_second" in timing:
        result["cell_steps_per_second"] = float(timing["cell_steps_per_second"])
    return result


def compare_traces(traces: dict[str, list[list[float]]], *, dt_ms: float = DT_MS) -> dict[str, Any]:
    """Compare each backend against NEURON and apply the publication gate."""
    import numpy as np

    if "neuron" not in traces:
        raise ValueError("accuracy comparison requires a NEURON reference")
    reference = np.asarray(traces["neuron"], dtype=float)
    if reference.shape != (len(PROBE_BRANCHES), N_STEPS):
        raise ValueError(f"unexpected NEURON trace shape {reference.shape}")

    comparisons: dict[str, Any] = {}
    passed = True
    for backend, raw in traces.items():
        if backend == "neuron":
            continue
        trace = np.asarray(raw, dtype=float)
        if trace.shape != reference.shape:
            raise ValueError(f"unexpected {backend} trace shape {trace.shape}")
        probe_metrics = []
        for probe_index in range(reference.shape[0]):
            ref = reference[probe_index]
            candidate = trace[probe_index]
            ref_spikes = _spike_indices(ref)
            candidate_spikes = _spike_indices(candidate)
            counts_match = len(ref_spikes) == len(candidate_spikes)
            if counts_match and ref_spikes:
                spike_time_delta = max(
                    abs(left - right) * dt_ms
                    for left, right in zip(ref_spikes, candidate_spikes)
                )
            elif counts_match:
                spike_time_delta = 0.0
            else:
                spike_time_delta = None
            amplitude_delta = abs(float(candidate.max()) - float(ref.max()))
            rmse = float(np.sqrt(np.mean((candidate - ref) ** 2)))
            probe_passed = (
                counts_match
                and spike_time_delta is not None
                and spike_time_delta <= ACCURACY_LIMITS["max_spike_time_delta_ms"]
                and amplitude_delta <= ACCURACY_LIMITS["max_spike_amplitude_delta_mv"]
                and rmse <= ACCURACY_LIMITS["max_rmse_mv"]
            )
            passed = passed and probe_passed
            probe_metrics.append(
                {
                    "branch": PROBE_BRANCHES[probe_index],
                    "reference_spike_count": len(ref_spikes),
                    "candidate_spike_count": len(candidate_spikes),
                    "max_spike_time_delta_ms": spike_time_delta,
                    "spike_amplitude_delta_mv": amplitude_delta,
                    "rmse_mv": rmse,
                    "passed": probe_passed,
                }
            )
        comparisons[backend] = {"probes": probe_metrics, "passed": all(p["passed"] for p in probe_metrics)}
    return {"passed": passed, "limits": ACCURACY_LIMITS, "comparisons": comparisons}


def query_gpus(
    candidates: tuple[int, ...],
    *,
    samples: int = 3,
    interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Sample physical GPUs and select the least busy allowed device."""
    if not candidates:
        raise ValueError("at least one GPU candidate is required")
    observed: dict[int, list[dict[str, Any]]] = {index: [] for index in candidates}
    for sample_index in range(samples):
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            index = int(fields[0])
            if index in observed:
                observed[index].append(
                    {
                        "uuid": fields[1],
                        "utilization_percent": float(fields[2]),
                        "memory_used_mib": float(fields[3]),
                        "memory_total_mib": float(fields[4]),
                    }
                )
        if sample_index + 1 < samples:
            time.sleep(interval_seconds)

    summaries = []
    for index in candidates:
        rows = observed[index]
        if len(rows) != samples:
            raise RuntimeError(f"GPU {index} was missing from nvidia-smi output")
        summaries.append(
            {
                "physical_id": index,
                "uuid": rows[0]["uuid"],
                "median_utilization_percent": statistics.median(r["utilization_percent"] for r in rows),
                "median_memory_used_mib": statistics.median(r["memory_used_mib"] for r in rows),
                "memory_total_mib": rows[0]["memory_total_mib"],
                "samples": rows,
            }
        )
    eligible = [
        row
        for row in summaries
        if row["median_utilization_percent"] <= 20.0 and row["median_memory_used_mib"] <= 10240.0
    ]
    if not eligible:
        raise RuntimeError("neither GPU 2 nor GPU 3 is idle enough for a reliable benchmark")
    selected = min(eligible, key=lambda row: (row["median_utilization_percent"], row["median_memory_used_mib"]))
    return {"selected": selected, "candidates": summaries}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def base_metadata(backend: str, batch_size: int) -> dict[str, Any]:
    return {
        "backend": backend,
        "batch_size": batch_size,
        "config": {
            "dt_ms": DT_MS,
            "tstop_ms": TSTOP_MS,
            "n_steps": N_STEPS,
            "stimulus_steps": [STIM_START_STEP, STIM_END_STEP],
            "temperature_celsius": TEMPERATURE_C,
            "v_init_mv": V_INIT_MV,
            "cv_per_branch": N_CV_PER_BRANCH,
            "probe_branches": list(PROBE_BRANCHES),
            "probe_x": PROBE_X,
            "morphology_sha256": morphology_sha256(),
        },
    }


def assert_morphology_asset() -> None:
    digest = morphology_sha256()
    if digest != MORPHOLOGY_SHA256:
        raise RuntimeError(f"n144.swc SHA256 mismatch: expected {MORPHOLOGY_SHA256}, got {digest}")


def _spike_indices(trace) -> list[int]:
    return [index for index in range(1, len(trace)) if trace[index - 1] < 0.0 <= trace[index]]


def _quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
