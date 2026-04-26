"""Compare one cable case between braincell and NEURON."""



import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .braincell_runner import run_case as run_braincell_case
    from .case_schema import MultiCompartmentCableCase
    from .mapping import build_mapping
    from .metrics import metric_record_for_pair
    from .neuron_runner import run_case as run_neuron_case
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from braincell_runner import run_case as run_braincell_case  # type: ignore
    from case_schema import MultiCompartmentCableCase  # type: ignore
    from mapping import build_mapping  # type: ignore
    from metrics import metric_record_for_pair  # type: ignore
    from neuron_runner import run_case as run_neuron_case  # type: ignore


def load_case(case_path: str | Path) -> MultiCompartmentCableCase:
    payload = json.loads(Path(case_path).read_text())
    return MultiCompartmentCableCase.from_dict(payload)


def compare_case(case: MultiCompartmentCableCase) -> dict[str, Any]:
    braincell_result = run_braincell_case(case)
    neuron_result = run_neuron_case(case)

    _ensure_compatible(
        braincell_result=braincell_result,
        neuron_result=neuron_result,
    )

    mapping = build_mapping(
        case,
        braincell_result=braincell_result,
        neuron_result=neuron_result,
    )
    braincell_voltage = np.asarray(braincell_result["voltage_mV"], dtype=float)
    neuron_voltage = np.asarray(neuron_result["voltage_mV"], dtype=float)
    braincell_indices = np.asarray(
        [pair["braincell_compartment_index"] for pair in mapping.compartment_pairs],
        dtype=int,
    )
    neuron_indices = np.asarray(
        [pair["neuron_compartment_index"] for pair in mapping.compartment_pairs],
        dtype=int,
    )
    aligned_braincell_voltage = braincell_voltage[:, braincell_indices]
    aligned_neuron_voltage = neuron_voltage[:, neuron_indices]
    metrics = _compute_metrics(
        braincell_voltage=aligned_braincell_voltage,
        neuron_voltage=aligned_neuron_voltage,
    )
    alignment = {
        "compartment_count": int(len(mapping.compartment_pairs)),
        "braincell_labels": [
            {
                **braincell_result["compartment_labels"][pair["braincell_compartment_index"]],
                "canonical_name": pair["braincell_canonical_name"],
                "canonical_label": pair["braincell_canonical_label"],
            }
            for pair in mapping.compartment_pairs
        ],
        "neuron_labels": [
            {
                **neuron_result["compartment_labels"][pair["neuron_compartment_index"]],
                "canonical_name": pair["neuron_canonical_name"],
                "canonical_label": pair["neuron_canonical_label"],
            }
            for pair in mapping.compartment_pairs
        ],
        "branch_pairs": list(mapping.branch_pairs),
        "compartment_pairs": list(mapping.compartment_pairs),
        "stimulus_target_pair": mapping.stimulus_target_pair,
    }
    return {
        "case_id": case.case_id,
        "template_family": case.template_family,
        "time_ms": np.asarray(braincell_result["time_ms"], dtype=float).tolist(),
        "braincell": {
            "voltage_mV": aligned_braincell_voltage.tolist(),
            "branch_order": braincell_result["branch_order"],
        },
        "neuron": {
            "voltage_mV": aligned_neuron_voltage.tolist(),
            "section_order": neuron_result["section_order"],
        },
        "alignment": alignment,
        "metrics": metrics,
    }


def _ensure_compatible(*, braincell_result: dict[str, Any], neuron_result: dict[str, Any]) -> None:
    braincell_time = np.asarray(braincell_result["time_ms"], dtype=float)
    neuron_time = np.asarray(neuron_result["time_ms"], dtype=float)
    if braincell_time.shape != neuron_time.shape or not np.allclose(braincell_time, neuron_time):
        raise ValueError("braincell and NEURON time axes do not match.")

    braincell_voltage = np.asarray(braincell_result["voltage_mV"], dtype=float)
    neuron_voltage = np.asarray(neuron_result["voltage_mV"], dtype=float)
    if braincell_voltage.shape != neuron_voltage.shape:
        raise ValueError(
            "braincell and NEURON voltage shapes do not match: "
            f"{braincell_voltage.shape!r} vs {neuron_voltage.shape!r}."
        )
    if len(braincell_result["compartment_labels"]) != len(neuron_result["compartment_labels"]):
        raise ValueError("braincell and NEURON compartment label counts do not match.")


def _compute_metrics(*, braincell_voltage: np.ndarray, neuron_voltage: np.ndarray) -> dict[str, Any]:
    per_cv = []
    for index in range(braincell_voltage.shape[1]):
        per_cv.append(
            {
                "compartment_index": index,
                **metric_record_for_pair(
                    braincell_voltage[:, index],
                    neuron_voltage[:, index],
                ),
            }
        )

    voltage_sum = metric_record_for_pair(
        np.sum(braincell_voltage, axis=1),
        np.sum(neuron_voltage, axis=1),
    )
    voltage_midpoint_mean = {
        field: float(np.mean([row[field] for row in per_cv]))
        for field in ("mae", "rmse", "max_abs", "rel_mae_pct")
    }
    return {
        "observables": {
            "voltage_midpoint_mean": voltage_midpoint_mean,
            "voltage_sum": voltage_sum,
        },
        "per_cv": per_cv,
    }
