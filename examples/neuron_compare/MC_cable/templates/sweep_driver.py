#!/usr/bin/env python3
"""Cartesian sweep driver for multi-compartment cable voltage comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .case_schema import MultiCompartmentCableCase
    from .compare_MC_cable import compare_case
    from .sweep_config import SweepConfig, config_to_payload, expand_cases, load_config
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    _templates_root = _here.parent
    for candidate in (_here, _templates_root, _templates_root.parent):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from case_schema import MultiCompartmentCableCase  # type: ignore
    from compare_MC_cable import compare_case  # type: ignore
    from sweep_config import SweepConfig, config_to_payload, expand_cases, load_config  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or expand a multi-compartment cable sweep configuration.")
    parser.add_argument("--config", required=True, help="Path to a sweep config JSON file.")
    parser.add_argument("--out-dir", help="Optional output directory.")
    parser.add_argument("--expand-only", action="store_true", help="Only write expanded_cases.json without running compare.")
    parser.set_defaults(plot=None)
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate per-case comparison plots.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot generation.")
    return parser


def default_output_dir(config: SweepConfig) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
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
            case = MultiCompartmentCableCase.from_dict(case_payload)
            result = compare_case(case)
            success_results.append(result)
            (case_results_dir / f"{case_id}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            overall = result["metrics"]["overall"]
            csv_rows.append(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "status": "ok",
                    "stimulus_kind": str(case.stimulus.kind),
                    "swc_path": case.swc.path,
                    "dt_ms": case.simulation.dt_ms,
                    "cv_per_branch": case.cv_policy.cv_per_branch,
                    "n_samples": len(result["time_ms"]),
                    "mae": overall["mae"],
                    "rmse": overall["rmse"],
                    "max_abs": overall["max_abs"],
                    "rel_mae_pct": overall["rel_mae_pct"],
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
                    "swc_path": case_payload.get("swc", {}).get("path", ""),
                    "dt_ms": case_payload.get("simulation", {}).get("dt_ms", ""),
                    "cv_per_branch": case_payload.get("cv_policy", {}).get("cv_per_branch", ""),
                    "n_samples": "",
                    "mae": "",
                    "rmse": "",
                    "max_abs": "",
                    "rel_mae_pct": "",
                    "error_message": str(exc),
                }
            )

    _write_case_metrics_csv(csv_rows, resolved_out_dir / "case_metrics.csv")
    aggregate = _aggregate_case_metrics(
        config,
        success_results=success_results,
        failed_cases=failed_cases,
        total_cases=len(expanded_cases),
    )
    (resolved_out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    if do_plot:
        _save_summary_plots(plots_dir, csv_rows=csv_rows)
    return 0 if len(failed_cases) == 0 else 1


def _write_case_metrics_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "case_id",
        "group_id",
        "status",
        "stimulus_kind",
        "swc_path",
        "dt_ms",
        "cv_per_branch",
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
    if success_results:
        maes = [result["metrics"]["overall"]["mae"] for result in success_results]
        rmses = [result["metrics"]["overall"]["rmse"] for result in success_results]
        max_abs = [result["metrics"]["overall"]["max_abs"] for result in success_results]
        rels = [result["metrics"]["overall"]["rel_mae_pct"] for result in success_results]
        overall = {
            "mae_mean": float(sum(maes) / len(maes)),
            "rmse_mean": float(sum(rmses) / len(rmses)),
            "max_abs_max": float(max(max_abs)),
            "rel_mae_pct_mean": float(sum(rels) / len(rels)),
        }
    else:
        overall = {}

    return {
        "config_id": config.config_id,
        "template_family": config.template_family,
        "n_total_cases": total_cases,
        "n_success_cases": len(success_results),
        "n_failed_cases": len(failed_cases),
        "overall": overall,
        "failed_cases": failed_cases,
    }


def _save_case_plot(out_path: Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    time_ms = np.asarray(result["time_ms"], dtype=float)
    braincell_voltage = np.asarray(result["braincell"]["voltage_mV"], dtype=float)
    neuron_voltage = np.asarray(result["neuron"]["voltage_mV"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for index in range(braincell_voltage.shape[1]):
        axes[0].plot(time_ms, neuron_voltage[:, index], linewidth=1.0, alpha=0.7)
        axes[0].plot(time_ms, braincell_voltage[:, index], linewidth=1.0, linestyle="--", alpha=0.7)
        axes[1].plot(time_ms, np.abs(braincell_voltage[:, index] - neuron_voltage[:, index]), linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("NEURON (solid) vs braincell (dashed)")
    axes[1].set_ylabel("|delta V| (mV)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title("Per-compartment absolute error")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_summary_plots(out_dir: Path, *, csv_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ok_rows = [row for row in csv_rows if row["status"] == "ok"]
    if len(ok_rows) == 0:
        return

    grouped_by_swc: dict[str, list[dict[str, Any]]] = {}
    grouped_by_combo: dict[str, list[dict[str, Any]]] = {}
    for row in ok_rows:
        swc_label = Path(str(row["swc_path"])).name
        grouped_by_swc.setdefault(swc_label, []).append(row)
        combo_label = f"{swc_label}\ncv={row['cv_per_branch']}"
        grouped_by_combo.setdefault(combo_label, []).append(row)

    _save_metric_boxplot(
        out_dir / "summary_mae_boxplot_by_swc.png",
        grouped=grouped_by_swc,
        metric_key="mae",
        title="MAE by SWC",
        ylabel="MAE (mV)",
    )
    _save_metric_boxplot(
        out_dir / "summary_max_abs_boxplot_by_swc_cv.png",
        grouped=grouped_by_combo,
        metric_key="max_abs",
        title="Max |delta V| by SWC and CV",
        ylabel="Max |delta V| (mV)",
    )


def _save_metric_boxplot(
    out_path: Path,
    *,
    grouped: dict[str, list[dict[str, Any]]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(grouped)
    values = [
        np.asarray([float(row[metric_key]) for row in grouped[label]], dtype=float)
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(labels)), 5.5))
    box = ax.boxplot(values, patch_artist=True, tick_labels=labels, showmeans=True)
    palette = ["#dbe8f5", "#bfd7ea", "#f2d5a0", "#d8c7ff", "#c7e9c0", "#f7c6c7"]
    for patch, color in zip(box["boxes"], palette * (len(box["boxes"]) // len(palette) + 1)):
        patch.set_facecolor(color)
        patch.set_edgecolor("#2f3e46")
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("#d1495b")
        median.set_linewidth(1.6)
    for mean in box["means"]:
        mean.set_marker("o")
        mean.set_markerfacecolor("#1b4332")
        mean.set_markeredgecolor("#1b4332")
        mean.set_markersize(5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
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
