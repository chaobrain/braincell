"""Serial NEURON backend, deliberately capped at ten cells."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
from neuron import h
import neuron

from common import (
    DT_MS,
    MORPHOLOGY_PATH,
    N_BRANCHES,
    N_CV,
    N_CV_PER_BRANCH,
    N_STEPS,
    PROBE_BRANCHES,
    PROBE_BRANCH_LENGTHS_UM,
    PROBE_X,
    STIM_DELAY_MS,
    STIM_DURATION_MS,
    TOTAL_LENGTH_UM,
    V_INIT_MV,
    assert_morphology_asset,
    add_throughput,
    base_metadata,
    timing_summary,
    write_json,
)


def build_cell():
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    for section in tuple(h.allsec()):
        h.delete_section(sec=section)
    reader = h.Import3d_SWC_read()
    reader.input(str(MORPHOLOGY_PATH))
    h.Import3d_GUI(reader, False).instantiate(None)
    sections = tuple(h.allsec())
    if len(sections) != N_BRANCHES:
        raise RuntimeError(f"expected {N_BRANCHES} NEURON sections, got {len(sections)}")

    for section in sections:
        section.nseg = N_CV_PER_BRANCH
        section.insert("hh")
        section.Ra = 1000.0
        section.cm = 1.0
        section.gnabar_hh = 0.120
        section.gkbar_hh = 0.036
        section.gl_hh = 0.0003
        section.ena = 50.0
        section.ek = -77.0
        section.el_hh = -54.3
    h.celsius = 6.3
    h.dt = DT_MS

    stim = h.IClamp(sections[0](PROBE_X))
    stim.delay = STIM_DELAY_MS
    stim.dur = STIM_DURATION_MS
    stim.amp = 0.7

    probe_sections = []
    for branch, target_length in zip(PROBE_BRANCHES, PROBE_BRANCH_LENGTHS_UM):
        _, section = min((abs(float(sec.L) - target_length), sec) for sec in sections)
        if abs(float(section.L) - target_length) > 2e-4:
            raise RuntimeError(f"could not map Jaxley branch {branch} into NEURON")
        probe_sections.append(section)

    vectors = []
    for section in probe_sections:
        vector = h.Vector()
        vector.record(section(PROBE_X)._ref_v)
        vectors.append(vector)

    total_length_um = sum(float(section.L) for section in sections)
    total_area_um2 = sum(float(h.area(segment.x, sec=section)) for section in sections for segment in section)
    morphology = {
        "n_branches": len(sections),
        "n_cv": sum(int(section.nseg) for section in sections),
        "total_length_um": total_length_um,
        "total_area_um2": total_area_um2,
        "probe_section_names": [section.name() for section in probe_sections],
    }
    if morphology["n_cv"] != N_CV:
        raise RuntimeError(f"unexpected NEURON discretization: {morphology}")
    if not np.isclose(total_length_um, TOTAL_LENGTH_UM, rtol=0.0, atol=1e-5):
        raise RuntimeError(f"unexpected NEURON morphology length: {total_length_um}")
    return stim, vectors, morphology


def simulate_serial(stim, vectors, batch_size: int):
    for amplitude in np.linspace(0.69, 0.71, batch_size) if batch_size > 1 else (0.7,):
        stim.amp = float(amplitude)
        h.finitialize(V_INIT_MV)
        h.fcurrent()
        for _ in range(N_STEPS):
            h.fadvance()
    traces = np.stack([np.asarray(vector, dtype=float)[1:] for vector in vectors])
    if traces.shape != (len(PROBE_BRANCHES), N_STEPS):
        raise RuntimeError(f"unexpected NEURON trace shape: {traces.shape}")
    return traces


def run_backend(batch_size: int, warmup: int, repeat: int, include_trace: bool) -> dict:
    if batch_size not in (1, 10):
        raise ValueError("NEURON is intentionally capped at batch sizes 1 and 10")
    assert_morphology_asset()
    build_start = time.perf_counter()
    stim, vectors, morphology = build_cell()
    build_seconds = time.perf_counter() - build_start
    for _ in range(warmup):
        traces = simulate_serial(stim, vectors, batch_size)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        traces = simulate_serial(stim, vectors, batch_size)
        samples.append(time.perf_counter() - start)

    payload = base_metadata("neuron", batch_size)
    payload.update(
        {
            "timing": add_throughput(timing_summary(samples), batch_size=batch_size),
            "host_transfer": None,
            "build_seconds": build_seconds,
            "compilation_seconds": 0.0,
            "morphology": morphology,
            "output_shape": [batch_size, len(PROBE_BRANCHES), N_STEPS],
            "software": {"python": platform.python_version(), "neuron": neuron.__version__},
            "device": "cpu-serial",
            "device_memory": None,
            "execution": {
                "jitted": False,
                "synchronization": "synchronous CPU execution",
                "host_transfer_included_in_primary_timing": True,
            },
            "output_validation": {
                "all_finite": bool(np.isfinite(traces).all()),
                "trace_output_bytes": int(traces.nbytes),
                "first_amplitude_na": 0.69 if batch_size > 1 else 0.7,
                "last_amplitude_na": 0.71 if batch_size > 1 else 0.7,
            },
        }
    )
    if include_trace:
        payload["trace_mv"] = traces.tolist()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(1, 10), required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, run_backend(args.batch_size, args.warmup, args.repeat, args.include_trace))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
