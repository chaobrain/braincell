#!/usr/bin/env python3
"""Compare one HH + fixed-ion single-compartment case between braincell and NEURON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .braincell_single_case import run_case as run_braincell_case
    from .case_schema import SingleCompartmentChannelHHFixedIonCase
    from .neuron_single_case import run_case as run_neuron_case
    from .pair_manifest import get_pair_entry
except ImportError:  # pragma: no cover
    import importlib.util
    import sys

    def _load_local_module(module_name: str, path: Path):
        module_key = f"sc_channel_hh_fixed_ion__{module_name}"
        module = sys.modules.get(module_key)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(module_key, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_key] = module
        spec.loader.exec_module(module)
        return module

    _here = Path(__file__).resolve().parent
    _braincell_single_case = _load_local_module("braincell_single_case", _here / "braincell_single_case.py")
    run_braincell_case = _braincell_single_case.run_case
    _case_schema = _load_local_module("case_schema", _here / "case_schema.py")
    SingleCompartmentChannelHHFixedIonCase = _case_schema.SingleCompartmentChannelHHFixedIonCase
    _neuron_single_case = _load_local_module("neuron_single_case", _here / "neuron_single_case.py")
    run_neuron_case = _neuron_single_case.run_case
    _pair_manifest = _load_local_module("pair_manifest", _here / "pair_manifest.py")
    get_pair_entry = _pair_manifest.get_pair_entry


def load_case(case_path: str | Path) -> SingleCompartmentChannelHHFixedIonCase:
    path = Path(case_path)
    payload = json.loads(path.read_text())
    return SingleCompartmentChannelHHFixedIonCase.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single-compartment HH + fixed-ion comparison case.",
    )
    parser.add_argument("case_path", help="Path to a JSON case file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def compare_case(case: SingleCompartmentChannelHHFixedIonCase) -> dict[str, Any]:
    braincell_raw = run_braincell_case(case)
    neuron_raw = run_neuron_case(case)

    braincell_result = _normalize_runner_result(braincell_raw, side="braincell")
    neuron_result = _normalize_runner_result(neuron_raw, side="neuron")
    neuron_trimmed = _align_time_axes(braincell_result=braincell_result, neuron_result=neuron_result)

    gate_pairs = _resolve_gate_pairs(
        case,
        braincell_gates=braincell_result["gates"],
        neuron_gates=neuron_result["gates"],
    )

    metrics = {
        "voltage": _metric_record_for_pair(
            braincell_result["voltage_mV"],
            neuron_result["voltage_mV"],
        ),
        "current": {
            "ix": _metric_record_for_pair(
                braincell_result["current"]["ix"],
                neuron_result["current"]["ix"],
            ),
        },
        "gates": {
            pair["canonical_name"]: _metric_record_for_pair(
                braincell_result["gates"][pair["braincell_gate"]],
                neuron_result["gates"][pair["neuron_gate"]],
            )
            for pair in gate_pairs
        },
    }

    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "template_family": case.template_family,
        "template_variant": case.template_variant,
        "time_ms": braincell_result["time_ms"].tolist(),
        "braincell": _serialize_runner_result(braincell_result),
        "neuron": _serialize_runner_result(neuron_result),
        "alignment": {
            "time_axis_trimmed_neuron_initial_sample": neuron_trimmed,
            "gates": gate_pairs,
        },
        "metrics": metrics,
    }


def _normalize_runner_result(result: dict[str, Any], *, side: str) -> dict[str, Any]:
    time_ms = _ensure_1d(result.get("time_ms"), name=f"{side}.time_ms")
    voltage_mV = _ensure_1d(result.get("voltage_mV"), name=f"{side}.voltage_mV")
    current_data = result.get("current")
    if not isinstance(current_data, dict):
        raise ValueError(f"{side}.current must be a mapping.")
    current_ix = _ensure_1d(current_data.get("ix"), name=f"{side}.current.ix")
    gates_data = result.get("gates")
    if not isinstance(gates_data, dict):
        raise ValueError(f"{side}.gates must be a mapping.")

    if voltage_mV.shape != time_ms.shape:
        raise ValueError(
            f"{side}.voltage_mV shape must match {side}.time_ms: "
            f"{voltage_mV.shape!r} vs {time_ms.shape!r}."
        )
    if current_ix.shape != time_ms.shape:
        raise ValueError(
            f"{side}.current.ix shape must match {side}.time_ms: "
            f"{current_ix.shape!r} vs {time_ms.shape!r}."
        )

    gates = {
        str(gate_name): _ensure_1d(gate_trace, name=f"{side}.gates.{gate_name}")
        for gate_name, gate_trace in gates_data.items()
    }
    for gate_name, gate_trace in gates.items():
        if gate_trace.shape != time_ms.shape:
            raise ValueError(
                f"{side}.gates.{gate_name} shape must match {side}.time_ms: "
                f"{gate_trace.shape!r} vs {time_ms.shape!r}."
            )

    return {
        "time_ms": time_ms,
        "voltage_mV": voltage_mV,
        "current": {"ix": current_ix},
        "gates": gates,
    }


def _align_time_axes(*, braincell_result: dict[str, Any], neuron_result: dict[str, Any]) -> bool:
    braincell_time = braincell_result["time_ms"]
    neuron_time = neuron_result["time_ms"]
    if braincell_time.shape == neuron_time.shape and np.allclose(braincell_time, neuron_time):
        return False

    if braincell_time.shape == neuron_time.shape and _is_constant_step_shift(braincell_time, neuron_time):
        return False

    if neuron_time.shape[0] == braincell_time.shape[0] + 1 and np.allclose(neuron_time[1:], braincell_time):
        _trim_initial_sample(neuron_result)
        return True

    raise ValueError("braincell and NEURON time axes do not match.")


def _trim_initial_sample(result: dict[str, Any]) -> None:
    result["time_ms"] = result["time_ms"][1:]
    result["voltage_mV"] = result["voltage_mV"][1:]
    result["current"]["ix"] = result["current"]["ix"][1:]
    result["gates"] = {
        gate_name: gate_trace[1:]
        for gate_name, gate_trace in result["gates"].items()
    }


def _is_constant_step_shift(braincell_time: np.ndarray, neuron_time: np.ndarray) -> bool:
    if braincell_time.shape[0] < 2:
        return False
    braincell_dt = np.diff(braincell_time)
    neuron_dt = np.diff(neuron_time)
    if not np.allclose(braincell_dt, neuron_dt):
        return False
    offset = neuron_time - braincell_time
    return np.allclose(offset, offset[0]) and np.isclose(offset[0], braincell_dt[0])


def _resolve_gate_pairs(
    case: SingleCompartmentChannelHHFixedIonCase,
    *,
    braincell_gates: dict[str, np.ndarray],
    neuron_gates: dict[str, np.ndarray],
) -> list[dict[str, str]]:
    pair_entry = get_pair_entry(case.pair_id)
    gate_name_map = case.compare.gate_name_map or pair_entry.gate_name_map

    if gate_name_map is None:
        if set(braincell_gates) != set(neuron_gates):
            raise ValueError(
                "braincell and NEURON gate sets do not match and no gate_name_map was provided: "
                f"{sorted(braincell_gates)!r} vs {sorted(neuron_gates)!r}."
            )
        return [
            {
                "canonical_name": gate_name,
                "braincell_gate": gate_name,
                "neuron_gate": gate_name,
            }
            for gate_name in sorted(braincell_gates)
        ]

    missing_braincell = sorted(set(gate_name_map) - set(braincell_gates))
    if missing_braincell:
        raise ValueError(f"gate_name_map references missing braincell gates: {missing_braincell!r}.")

    missing_neuron = sorted(set(gate_name_map.values()) - set(neuron_gates))
    if missing_neuron:
        raise ValueError(f"gate_name_map references missing NEURON gates: {missing_neuron!r}.")

    unmapped_braincell = sorted(set(braincell_gates) - set(gate_name_map))
    if unmapped_braincell:
        raise ValueError(f"gate_name_map does not cover all braincell gates: {unmapped_braincell!r}.")

    unmapped_neuron = sorted(set(neuron_gates) - set(gate_name_map.values()))
    if unmapped_neuron:
        raise ValueError(f"gate_name_map does not cover all NEURON gates: {unmapped_neuron!r}.")

    return [
        {
            "canonical_name": braincell_gate,
            "braincell_gate": braincell_gate,
            "neuron_gate": neuron_gate,
        }
        for braincell_gate, neuron_gate in gate_name_map.items()
    ]


def _metric_record_for_pair(braincell_trace: np.ndarray, neuron_trace: np.ndarray) -> dict[str, float]:
    if braincell_trace.shape != neuron_trace.shape:
        raise ValueError(
            "braincell and NEURON trace shapes do not match: "
            f"{braincell_trace.shape!r} vs {neuron_trace.shape!r}."
        )
    abs_diff = np.abs(braincell_trace - neuron_trace)
    return _metric_record(abs_diff, reference=neuron_trace)


def _metric_record(abs_diff: np.ndarray, *, reference: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
    max_abs = float(np.max(abs_diff))
    ref_scale = float(np.mean(np.abs(reference)))
    if ref_scale <= 1e-12:
        rel_mae_pct = 0.0 if mae <= 1e-12 else float("inf")
    else:
        rel_mae_pct = float((mae / ref_scale) * 100.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "rel_mae_pct": rel_mae_pct,
    }


def _serialize_runner_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "voltage_mV": result["voltage_mV"].tolist(),
        "current": {"ix": result["current"]["ix"].tolist()},
        "gates": {
            gate_name: gate_trace.tolist()
            for gate_name, gate_trace in result["gates"].items()
        },
    }


def _ensure_1d(arr: object, *, name: str) -> np.ndarray:
    value = np.asarray(arr, dtype=float)
    value = np.squeeze(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D after squeeze, got shape={value.shape!r}.")
    return value.reshape(-1)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    case = load_case(args.case_path)
    result = compare_case(case)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
