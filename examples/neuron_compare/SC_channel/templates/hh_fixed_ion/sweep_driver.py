#!/usr/bin/env python3
"""Cartesian sweep driver for HH + fixed-ion single-compartment comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .case_schema import SingleCompartmentChannelHHFixedIonCase
    from .compare_single_case import compare_case
    from .sweep_config import SweepConfig, config_to_payload, expand_cases, load_config
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
    _case_schema = _load_local_module("case_schema", _here / "case_schema.py")
    SingleCompartmentChannelHHFixedIonCase = _case_schema.SingleCompartmentChannelHHFixedIonCase
    _compare_single_case = _load_local_module("compare_single_case", _here / "compare_single_case.py")
    compare_case = _compare_single_case.compare_case
    _sweep_config = _load_local_module("sweep_config", _here / "sweep_config.py")
    SweepConfig = _sweep_config.SweepConfig
    config_to_payload = _sweep_config.config_to_payload
    expand_cases = _sweep_config.expand_cases
    load_config = _sweep_config.load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or expand a HH + fixed-ion sweep configuration.",
    )
    parser.add_argument("--config", required=True, help="Path to a sweep config JSON file.")
    parser.add_argument("--out-dir", help="Optional output directory.")
    parser.add_argument("--expand-only", action="store_true", help="Only write expanded_cases.json without running compare.")
    parser.set_defaults(plot=None)
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate per-case comparison plots.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot generation.")
    return parser


def default_output_dir(config: SweepConfig) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "hh_fixed_ion"
        / config.config_id
    )


def run_config(
    config: SweepConfig,
    *,
    out_dir: str | Path | None = None,
    expand_only: bool = False,
    plot: bool | None = None,
) -> int:
    expanded_cases = expand_cases(config)
    resolved_out_dir = Path(out_dir) if out_dir is not None else default_output_dir(config)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    (resolved_out_dir / "normalized_config.json").write_text(
        json.dumps(config_to_payload(config), indent=2, sort_keys=True) + "\n"
    )
    (resolved_out_dir / "expanded_cases.json").write_text(
        json.dumps(expanded_cases, indent=2, sort_keys=True) + "\n"
    )

    if expand_only:
        return 0

    do_plot = config.outputs.plot if plot is None else bool(plot)
    case_results_dir = resolved_out_dir / "case_results"
    case_results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = resolved_out_dir / "plots"
    if do_plot:
        plots_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, Any]] = []
    success_results: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []

    for case_payload in expanded_cases:
        group_id = str(case_payload.get("group_id", ""))
        case_id = str(case_payload["case_id"])
        try:
            case = SingleCompartmentChannelHHFixedIonCase.from_dict(case_payload)
            result = compare_case(case)
            success_results.append(result)
            (case_results_dir / f"{case_id}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            for observable, metric in _iter_metrics(result["metrics"]):
                csv_rows.append(
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "status": "ok",
                        "stimulus_kind": str(case.stimulus.kind),
                        "temperature_celsius": case.simulation.temperature_celsius,
                        "v_init_mV": case.simulation.v_init_mV,
                        "n_samples": len(result["time_ms"]),
                        "observable": observable,
                        "mae": metric["mae"],
                        "rmse": metric["rmse"],
                        "max_abs": metric["max_abs"],
                        "rel_mae_pct": metric["rel_mae_pct"],
                        "error_message": "",
                    }
                )
            if do_plot:
                _save_case_plot(plots_dir / f"{case_id}.png", result)
        except Exception as exc:
            failed_cases.append(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "error_message": str(exc),
                    "case": case_payload,
                }
            )
            csv_rows.append(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "status": "failed",
                    "stimulus_kind": case_payload.get("stimulus", {}).get("kind", ""),
                    "temperature_celsius": case_payload.get("simulation", {}).get("temperature_celsius", ""),
                    "v_init_mV": case_payload.get("simulation", {}).get("v_init_mV", ""),
                    "n_samples": "",
                    "observable": "",
                    "mae": "",
                    "rmse": "",
                    "max_abs": "",
                    "rel_mae_pct": "",
                    "error_message": str(exc),
                }
            )

    _write_case_metrics_csv(csv_rows, resolved_out_dir / "case_metrics.csv")
    aggregate = _aggregate_case_metrics(config, success_results=success_results, failed_cases=failed_cases, total_cases=len(expanded_cases))
    (resolved_out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    return 0 if len(success_results) > 0 else 1


def _iter_metrics(metrics: dict[str, Any]) -> list[tuple[str, dict[str, float]]]:
    records: list[tuple[str, dict[str, float]]] = [("voltage", metrics["voltage"])]
    records.extend((f"current.{name}", record) for name, record in metrics.get("current", {}).items())
    records.extend((f"gates.{name}", record) for name, record in metrics.get("gates", {}).items())
    return records


def _write_case_metrics_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "case_id",
        "group_id",
        "status",
        "stimulus_kind",
        "temperature_celsius",
        "v_init_mV",
        "observable",
        "n_samples",
        "mae",
        "rmse",
        "max_abs",
        "rel_mae_pct",
        "error_message",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_case_metrics(
    config: SweepConfig,
    *,
    success_results: list[dict[str, Any]],
    failed_cases: list[dict[str, Any]],
    total_cases: int,
) -> dict[str, Any]:
    observable_rows: dict[str, list[dict[str, float]]] = {}
    for result in success_results:
        for observable, metric in _iter_metrics(result["metrics"]):
            observable_rows.setdefault(observable, []).append(metric)

    observables = {
        observable: {
            "n_cases": len(rows),
            "mae_mean": float(sum(row["mae"] for row in rows) / len(rows)),
            "rmse_mean": float(sum(row["rmse"] for row in rows) / len(rows)),
            "max_abs_max": float(max(row["max_abs"] for row in rows)),
            "rel_mae_pct_mean": float(sum(row["rel_mae_pct"] for row in rows) / len(rows)),
        }
        for observable, rows in observable_rows.items()
    }

    return {
        "config_id": config.config_id,
        "template_family": config.template_family,
        "template_variant": config.template_variant,
        "n_total_cases": total_cases,
        "n_success_cases": len(success_results),
        "n_failed_cases": len(failed_cases),
        "observables": observables,
        "failed_cases": failed_cases,
    }


def _save_case_plot(out_path: Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    observables = [
        ("voltage_mV", "Voltage (mV)"),
        ("current.ix", "Current ix"),
        *[(f"gates.{gate_name}", f"Gate {gate_name}") for gate_name in sorted(result["braincell"]["gates"])],
    ]
    time_ms = result["time_ms"]
    fig, axes = plt.subplots(len(observables), 1, figsize=(10, 3.0 * len(observables)), sharex=True)
    if len(observables) == 1:
        axes = [axes]
    for axis, (observable, label) in zip(axes, observables):
        if observable == "voltage_mV":
            braincell_trace = result["braincell"]["voltage_mV"]
            neuron_trace = result["neuron"]["voltage_mV"]
        elif observable == "current.ix":
            braincell_trace = result["braincell"]["current"]["ix"]
            neuron_trace = result["neuron"]["current"]["ix"]
        else:
            gate_name = observable.split(".", 1)[1]
            braincell_trace = result["braincell"]["gates"][gate_name]
            neuron_trace = result["neuron"]["gates"][gate_name]
        axis.plot(time_ms, neuron_trace, label="NEURON", linewidth=1.5)
        axis.plot(time_ms, braincell_trace, label="braincell", linewidth=1.2, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (ms)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    return run_config(
        config,
        out_dir=args.out_dir,
        expand_only=bool(args.expand_only),
        plot=args.plot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
