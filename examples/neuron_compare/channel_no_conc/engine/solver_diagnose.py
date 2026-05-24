#!/usr/bin/env python3
"""Minimal solver diagnostics for channel_no_conc voltage drift cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .braincell_runner import run_case as run_braincell_case
    from .experiment_schema import ChannelNoConcCase
    from .metrics import metric_record_for_pair
    from .neuron_runner import run_case as run_neuron_case
    from .outputs import write_json
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from braincell_runner import run_case as run_braincell_case  # type: ignore
    from experiment_schema import ChannelNoConcCase  # type: ignore
    from metrics import metric_record_for_pair  # type: ignore
    from neuron_runner import run_case as run_neuron_case  # type: ignore
    from outputs import write_json  # type: ignore


DEFAULT_TARGET_CASES = ("dc__002", "dc__003")
DEFAULT_SOLVERS = ("staggered", "exp_euler")
DEFAULT_CHECKPOINTS_MS = (1.00, 1.01, 1.10, 2.0, 3.0, 5.0, 10.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose BrainCell solver drift for existing expanded channel_no_conc cases.")
    parser.add_argument("expanded_cases_path", help="Path to expanded_cases.json.")
    parser.add_argument("--out-dir", required=True, help="Directory for diagnostic outputs.")
    parser.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        help="Optional case_id to include. Repeatable. Defaults to dc__002 and dc__003.",
    )
    parser.add_argument(
        "--solver",
        dest="solvers",
        action="append",
        help="Optional BrainCell solver name to compare. Repeatable. Defaults to staggered and exp_euler.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    expanded_cases_path = Path(args.expanded_cases_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    case_ids = tuple(args.case_ids) if args.case_ids else DEFAULT_TARGET_CASES
    solvers = tuple(args.solvers) if args.solvers else DEFAULT_SOLVERS

    report = diagnose_expanded_cases(
        expanded_cases_path=expanded_cases_path,
        case_ids=case_ids,
        solvers=solvers,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "solver_diagnostic_report.json", report)
    _write_summary_csv(out_dir / "solver_diagnostic_summary.csv", report)
    return 0


def diagnose_expanded_cases(
    *,
    expanded_cases_path: str | Path,
    case_ids: tuple[str, ...] = DEFAULT_TARGET_CASES,
    solvers: tuple[str, ...] = DEFAULT_SOLVERS,
) -> dict[str, Any]:
    expanded_cases_path = Path(expanded_cases_path).expanduser().resolve()
    payloads = json.loads(expanded_cases_path.read_text())
    if not isinstance(payloads, list):
        raise ValueError("expanded_cases.json must contain a list payload.")

    selected = [payload for payload in payloads if str(payload.get("case_id", "")) in set(case_ids)]
    if len(selected) == 0:
        raise ValueError(
            f"No matching cases found in {expanded_cases_path!s} for case_ids={case_ids!r}."
        )

    cases = [ChannelNoConcCase.from_dict(payload) for payload in selected]
    case_reports = [_diagnose_case(case, solvers=solvers) for case in cases]
    order = {name: index for index, name in enumerate(solvers)}
    return {
        "expanded_cases_path": str(expanded_cases_path),
        "case_ids": list(case_ids),
        "solvers": list(solvers),
        "checkpoints_ms": list(DEFAULT_CHECKPOINTS_MS),
        "cases": case_reports,
        "summary": _build_summary(case_reports, solver_order=order),
    }


def _diagnose_case(case: ChannelNoConcCase, *, solvers: tuple[str, ...]) -> dict[str, Any]:
    neuron = run_neuron_case(case)
    solver_reports = []
    for solver in solvers:
        braincell = run_braincell_case(case, solver=solver)
        solver_reports.append(
            _build_solver_report(
                case=case,
                solver=solver,
                braincell=braincell,
                neuron=neuron,
            )
        )
    return {
        "case_id": case.case_id,
        "run_id": case.run_id,
        "stimulus": {
            "kind": case.stimulus.kind,
            "amp_nA": float(case.stimulus.amp_nA) if hasattr(case.stimulus, "amp_nA") else None,
            "delay_ms": float(case.stimulus.delay_ms) if hasattr(case.stimulus, "delay_ms") else None,
            "dur_ms": float(case.stimulus.dur_ms) if hasattr(case.stimulus, "dur_ms") else None,
        },
        "solvers": solver_reports,
        "improvement": _compare_solver_reports(solver_reports),
    }


def _build_solver_report(
    *,
    case: ChannelNoConcCase,
    solver: str,
    braincell: dict[str, Any],
    neuron: dict[str, Any],
) -> dict[str, Any]:
    voltage_braincell = np.asarray(braincell["voltage_mV"], dtype=float)
    voltage_neuron = np.asarray(neuron["voltage_mV"], dtype=float)
    gate_braincell = np.asarray(next(iter(braincell["gates"].values())), dtype=float)
    gate_neuron = np.asarray(next(iter(neuron["gates"].values())), dtype=float)
    current_braincell = np.asarray(braincell["current"]["ix"], dtype=float)
    current_neuron = np.asarray(neuron["current"]["ix"], dtype=float)
    current_time_ms = np.asarray(braincell["time_ms"], dtype=float)

    voltage_metrics = metric_record_for_pair(voltage_braincell, voltage_neuron)
    gate_metrics = metric_record_for_pair(gate_braincell, gate_neuron)
    current_metrics = metric_record_for_pair(current_braincell[:-1], current_neuron[1:])
    checkpoints = _sample_checkpoints(
        time_ms=np.asarray(neuron["time_ms"], dtype=float),
        voltage_braincell=voltage_braincell,
        voltage_neuron=voltage_neuron,
        gate_braincell=gate_braincell,
        gate_neuron=gate_neuron,
        current_braincell=current_braincell,
        current_neuron=current_neuron,
    )

    return {
        "solver": solver,
        "voltage": voltage_metrics,
        "gate_m": gate_metrics,
        "current_ix_shifted": current_metrics,
        "alignment": {
            "current_time_ms": current_time_ms[:-1].tolist(),
            "neuron_shift_steps": 1,
            "braincell_drop_tail_steps": 1,
        },
        "checkpoints": checkpoints,
    }


def _sample_checkpoints(
    *,
    time_ms: np.ndarray,
    voltage_braincell: np.ndarray,
    voltage_neuron: np.ndarray,
    gate_braincell: np.ndarray,
    gate_neuron: np.ndarray,
    current_braincell: np.ndarray,
    current_neuron: np.ndarray,
) -> list[dict[str, float]]:
    rows = []
    for target_ms in DEFAULT_CHECKPOINTS_MS:
        index = int(np.argmin(np.abs(time_ms - target_ms)))
        rows.append(
            {
                "target_ms": float(target_ms),
                "sample_ms": float(time_ms[index]),
                "voltage_diff_mV": float(voltage_braincell[index] - voltage_neuron[index]),
                "voltage_braincell_mV": float(voltage_braincell[index]),
                "voltage_neuron_mV": float(voltage_neuron[index]),
                "gate_m_diff": float(gate_braincell[index] - gate_neuron[index]),
                "current_ix_diff_mA_cm2": float(current_braincell[index] - current_neuron[index]),
            }
        )
    return rows


def _compare_solver_reports(solver_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_solver = {report["solver"]: report for report in solver_reports}
    if "staggered" not in by_solver or "exp_euler" not in by_solver:
        return {
            "baseline_solver": None,
            "candidate_solver": None,
            "voltage_mae_ratio": None,
            "voltage_max_abs_ratio": None,
            "one_order_of_magnitude_better": False,
        }
    baseline = by_solver["staggered"]
    candidate = by_solver["exp_euler"]
    voltage_mae_ratio = _safe_ratio(candidate["voltage"]["mae"], baseline["voltage"]["mae"])
    voltage_max_abs_ratio = _safe_ratio(candidate["voltage"]["max_abs"], baseline["voltage"]["max_abs"])
    return {
        "baseline_solver": "staggered",
        "candidate_solver": "exp_euler",
        "voltage_mae_ratio": voltage_mae_ratio,
        "voltage_max_abs_ratio": voltage_max_abs_ratio,
        "one_order_of_magnitude_better": (
            voltage_mae_ratio is not None
            and voltage_max_abs_ratio is not None
            and voltage_mae_ratio <= 0.1
            and voltage_max_abs_ratio <= 0.1
        ),
    }


def _build_summary(case_reports: list[dict[str, Any]], *, solver_order: dict[str, int]) -> dict[str, Any]:
    flattened = []
    for case_report in case_reports:
        for solver_report in case_report["solvers"]:
            flattened.append(
                {
                    "case_id": case_report["case_id"],
                    "solver": solver_report["solver"],
                    "voltage_mae": float(solver_report["voltage"]["mae"]),
                    "voltage_max_abs": float(solver_report["voltage"]["max_abs"]),
                    "gate_m_mae": float(solver_report["gate_m"]["mae"]),
                    "current_ix_shifted_mae": float(solver_report["current_ix_shifted"]["mae"]),
                }
            )
    flattened.sort(key=lambda row: (row["case_id"], solver_order.get(row["solver"], 1_000_000)))
    return {
        "rows": flattened,
        "n_cases_confirmed_one_order_magnitude": int(
            sum(1 for case_report in case_reports if case_report["improvement"]["one_order_of_magnitude_better"])
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _write_summary_csv(out_path: str | Path, report: dict[str, Any]) -> None:
    import csv

    rows = report["summary"]["rows"]
    fieldnames = (
        "case_id",
        "solver",
        "voltage_mae",
        "voltage_max_abs",
        "gate_m_mae",
        "current_ix_shifted_mae",
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
