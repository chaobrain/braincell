#!/usr/bin/env python3
"""Run one NEURON-side case for the multi-compartment cable template."""



from pathlib import Path
from typing import Any

import numpy as np

try:
    from .case_schema import MultiCompartmentCableCase
    from .morphology_io import delete_neuron_sections, load_neuron_sections, locate_root_neuron_soma
    from .stimulus import current_at_ms
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from case_schema import MultiCompartmentCableCase  # type: ignore
    from morphology_io import delete_neuron_sections, load_neuron_sections, locate_root_neuron_soma  # type: ignore
    from stimulus import current_at_ms  # type: ignore


def run_case(case: MultiCompartmentCableCase) -> dict[str, Any]:
    from neuron import h

    secs = load_neuron_sections(case)
    h.load_file("stdrun.hoc")

    for sec in secs:
        sec.nseg = int(case.cv_policy.cv_per_branch)
        sec.Ra = float(case.cable.ra_ohm_cm)
        sec.cm = float(case.cable.cm_uF_cm2)

    root_soma = locate_root_neuron_soma(secs)
    stim = h.IClamp(root_soma(0.5))
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.0

    time_ms = np.arange(
        0.0,
        float(case.simulation.duration_ms),
        float(case.simulation.dt_ms),
        dtype=float,
    )
    time_ms = np.round(time_ms, decimals=12)
    if time_ms.size == 0:
        raise ValueError("simulation.duration_ms must produce at least one sample.")

    compartment_labels = _compartment_labels(secs)
    section_order = _section_order(secs)
    t_vec = h.Vector().record(h._ref_t)
    voltage_vecs = [h.Vector().record(seg._ref_v) for sec in secs for seg in sec]
    amp_values = np.asarray([current_at_ms(case.stimulus, float(t_ms)) for t_ms in time_ms], dtype=float)
    amp_time_vec = h.Vector(time_ms)
    amp_value_vec = h.Vector(amp_values)
    # Keep the clamp piecewise-constant across each dt step so h.run() matches
    # the legacy per-step fadvance() baseline at stimulus edges.
    amp_value_vec.play(stim._ref_amp, amp_time_vec)

    h.cvode_active(0)
    h.dt = float(case.simulation.dt_ms)
    h.steps_per_ms = 1.0 / h.dt
    h.dt = float(case.simulation.dt_ms)
    h.v_init = float(case.simulation.v_init_mV)
    h.finitialize(h.v_init)

    try:
        h.tstop = float(case.simulation.duration_ms)
        h.run()
    finally:
        amp_value_vec.play_remove()
        delete_neuron_sections(secs)

    _validate_recorded_length(t_vec, expected_samples=time_ms.size)
    voltage_mV = np.column_stack([
        _vector_to_1d(vec, name=f"voltage[{index}]")[1:]
        for index, vec in enumerate(voltage_vecs)
    ])

    return {
        "time_ms": time_ms,
        # Braincell returns one post-update sample per driven step. NEURON
        # record vectors include the initial state at t=0, so drop it.
        "voltage_mV": voltage_mV,
        "compartment_labels": compartment_labels,
        "section_order": section_order,
    }


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


def _vector_to_1d(vec: object, *, name: str) -> np.ndarray:
    value = np.asarray(vec, dtype=float)
    value = np.squeeze(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeeze, got shape={value.shape!r}.")
    return value.reshape(-1)


def _validate_recorded_length(t_vec: object, *, expected_samples: int) -> None:
    recorded_time_ms = _vector_to_1d(t_vec, name="neuron.time_ms")
    expected = expected_samples + 1
    if recorded_time_ms.size != expected:
        raise ValueError(
            "NEURON run()/record() returned an unexpected sample count: "
            f"expected {expected}, got {recorded_time_ms.size}."
        )


def main() -> int:
    raise NotImplementedError("Run this module through run_case(case) for now.")


if __name__ == "__main__":
    raise SystemExit(main())
