#!/usr/bin/env python3
"""Run one NEURON-side case for the multi-compartment cable template."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .case_schema import MultiCompartmentCableCase
    from .stimulus_utils import current_at_ms
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from case_schema import MultiCompartmentCableCase  # type: ignore
    from stimulus_utils import current_at_ms  # type: ignore


def run_case(case: MultiCompartmentCableCase) -> dict[str, Any]:
    from neuron import h

    load_swc_morphology = _load_swc_morphology_helper()

    secs = load_swc_morphology(case.swc.path)
    h.load_file("stdrun.hoc")

    for sec in secs:
        sec.nseg = int(case.cv_policy.cv_per_branch)
        sec.Ra = float(case.cable.ra_ohm_cm)
        sec.cm = float(case.cable.cm_uF_cm2)

    stim = h.IClamp(secs[0](0.5))
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.0

    times_ms = np.arange(
        0.0,
        float(case.simulation.duration_ms),
        float(case.simulation.dt_ms),
        dtype=float,
    )
    compartment_labels = _compartment_labels(secs)
    section_order = _section_order(secs)
    voltage_mV = np.empty((times_ms.shape[0] + 1, len(compartment_labels)), dtype=float)

    h.dt = float(case.simulation.dt_ms)
    h.finitialize(float(case.simulation.v_init_mV))
    voltage_mV[0, :] = _sample_segment_voltages(secs)

    try:
        for index, t_ms in enumerate(times_ms):
            stim.amp = float(current_at_ms(case.stimulus, float(t_ms)))
            h.fadvance()
            voltage_mV[index + 1, :] = _sample_segment_voltages(secs)
    finally:
        for sec in secs:
            try:
                h.delete_section(sec=sec)
            except Exception:
                pass

    return {
        "time_ms": times_ms,
        # Cell.run() returns post-update probe values, so drop the initial
        # pre-step NEURON sample and keep one sample per driven step.
        "voltage_mV": voltage_mV[1:, :],
        "compartment_labels": compartment_labels,
        "section_order": section_order,
    }


def _sample_segment_voltages(secs) -> np.ndarray:
    samples: list[float] = []
    for sec in secs:
        for seg in sec:
            samples.append(float(seg.v))
    return np.asarray(samples, dtype=float)


def _section_order(secs) -> list[dict[str, Any]]:
    return [
        {
            "section_index": index,
            "section_name": sec.name(),
        }
        for index, sec in enumerate(secs)
    ]


def _compartment_labels(secs) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for section_index, sec in enumerate(secs):
        for local_index, seg in enumerate(sec):
            labels.append(
                {
                    "compartment_index": len(labels),
                    "section_index": int(section_index),
                    "section_name": sec.name(),
                    "local_index": int(local_index),
                    "x": float(seg.x),
                }
            )
    return labels


def _load_swc_morphology_helper():
    module_path = Path(__file__).resolve().parents[3] / "multi_compartment" / "neuron_diff.py"
    spec = importlib.util.spec_from_file_location("multi_compartment_neuron_diff_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load neuron_diff helper from {module_path!s}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_swc_morphology


def main() -> int:
    raise NotImplementedError("Run this module through run_case(case) for now.")


if __name__ == "__main__":
    raise SystemExit(main())
